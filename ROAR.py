import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from ROPE import RotaryEmbedding
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoTokenizer
from rdkit.Chem import MolToSmiles, MolFromSmiles

debug = False
torch._dynamo.config.cache_size_limit = 64
#torch._dynamo.config.guard_nn_modules = True

#make alphas for B, L sequences
def make_non_uniform_alphas1D(x, smoothing_ratio=0, return_probability_dist=True):
    assert len(x.shape) == 2
    if debug:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L = x.shape
    pool_width = int(L*smoothing_ratio)*2+1
    step_multiplier = 1.1
    noise_magnitude = 1
    normal_dist = torch.distributions.Normal(0,1)
    base = torch.zeros((B, 1, L), device=device)
    dimensions = L
    while int(dimensions) > 0:
        rand = torch.empty((B, 1, int(dimensions)), device=device).normal_(0,1) * noise_magnitude
        base += F.interpolate(rand, L, mode="linear")
        dimensions = dimensions/step_multiplier
        noise_magnitude *= step_multiplier
    
    base = base.reshape(B, L)
    base = F.avg_pool1d(base, pool_width,1, padding=pool_width//2, count_include_pad=False)
    base = base - base.mean(axis=1).unsqueeze(1)
    base = base / base.std(axis=1).unsqueeze(1)
    
    if return_probability_dist:
        dist = normal_dist.cdf(base)
        dist = dist - dist.min(axis=1)[0].unsqueeze(1)
        dist = dist / dist.max(axis=1)[0].unsqueeze(1)
        return dist
    else:
        return base

def scramble_order(tab_data, seq_data, device, biased_scramble=True, smoothing_ratio=0, return_index=False, simple_scramble=False):
    x_type_tab, x_tab, pos_tab1 = tab_data
    x_tab, x_type_tab, pos_tab1, = x_tab.to(device), x_type_tab.to(device), pos_tab1.to(device),
    
    x_type, x, pos1, = seq_data
    x, x_type, pos1, = x.to(device), x_type.to(device), pos1.to(device),
    
    alphas = torch.empty((x.shape[0], x_tab.shape[1]+x.shape[1]), device=device).uniform_(0, 0.000001)
    
    B, L = x.shape
    if biased_scramble:
        if simple_scramble:
            alphas_tab = torch.randint(0,2, (x_tab.shape[0],1), device=device).float()
        else:
            alphas_tab = torch.empty((x_tab.shape[0],1), device=device).uniform_(0,1)
        alphas[:,0:x_tab.shape[1]] += alphas_tab
        if simple_scramble:
            alphas_seq = torch.arange(L, device=device)
            alphas_seq = alphas_seq/alphas_seq.max()
            alphas_seq = alphas_seq.repeat(B, 1).reshape(B,-1)
        else:
            alphas_seq = make_non_uniform_alphas1D(x, smoothing_ratio=smoothing_ratio)
        
        alphas[:,x_tab.shape[1]:x_tab.shape[1]+x.shape[1]] += alphas_seq
    
    x = torch.cat([x_tab, x.reshape(B,-1)], axis=1)
    x_type = torch.cat([x_type_tab, x_type.reshape(B,-1)], axis=1)
    pos1 = torch.cat([pos_tab1, pos1.reshape(B,-1), ], axis=1)
    
    _, index = alphas.sort(axis=1, descending=False)

    seq_order1 = torch.gather(pos1, 1, index)
    x_reordered = torch.gather(x, 1, index)
    x_type_reordered = torch.gather(x_type, 1, index)
    
    if return_index:
        return index, x_type_reordered, x_reordered,  (seq_order1,)
    else:
        return x_type_reordered, x_reordered,  (seq_order1,)

#tokens values reserved in advance
def embed_sample(batch, device):
    x, y, = batch
    x = x.to(device)
    y = y.to(device)

    B, L = y.shape
    seq_pos1 = torch.arange(L, device=device).repeat(B, 1)
    seq_value = y
    seq_type = torch.zeros(seq_value.shape, device=device) + 2

    tab_cols = x.shape[1]
    tab_value = x
    tab_type = torch.arange(tab_cols, device=device).repeat(x.shape[0],1) + 3
    tab_pos1 = torch.zeros(tab_type.shape[0], tab_type.shape[1])

    tab_data = (tab_type.long(), tab_value.long(), tab_pos1.float(),)
    seq_data = (seq_type.long(), seq_value.long(), seq_pos1.float(),)

    return tab_data, seq_data

def encode_with_tokenizer(tokenizer, line, max_len):
    line = tokenizer.encode(line)[:max_len]
    base = np.zeros(max_len, dtype=int)
    seq = np.array(line).astype(int)
    base[0:seq.shape[0]] = seq
    return base.reshape(-1)

def prepare_smiles_without_properties(smiles_strings, tokenizer, max_len=256, num_tab_columns=338):
    items = []
    for smiles in smiles_strings:
        items.append(encode_with_tokenizer(tokenizer, smiles, max_len))
    ss = np.concatenate(items, dtype="short").reshape(-1, max_len)
    properties = np.zeros((ss.shape[0],num_tab_columns))
    return properties, ss

def make_diag(length):
    array = torch.ones((length+1, length))
    for i in range(0,length+1):
        array[i,i:] = 0
    return array

class ToBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.round(x)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone() # pass-through

class BinaryQuantizer(nn.Module):
    def forward(self, x, add_noise=False, round_during_training=True):
        if add_noise and self.training:
            x = x + torch.empty_like(x).normal_()
        x = torch.sigmoid(x)
        if not self.training or round_during_training: 
            x = ToBinary.apply(x)
        return x

####################################################################################
# Transformer Code Autoencoder
####################################################################################

#Applies a binary mask to a vector (the embedding) then does things to allow the network to gracefully handle masked items and distinguish them from actual 0s
#For an N length embedding vector:
#1) Multiply by the binary mask, zeroing out masked item
#2) Concat the mask and embedding vector to make a 2 X N matrix
#3) Multiply each element in this matrix by a learned weight and add a learned bias term
#4) Pass each Mask/Embedding element pair through a neural network (same network for all items, applied separately)
class ApplyMask(nn.Module):
    def __init__(self, num_bits, is_binary):
        super(ApplyMask, self).__init__()
        self.is_binary = is_binary
        self.element_weight = nn.Parameter(torch.ones((num_bits, 2)) + torch.normal(0,0.01,(num_bits, 2)))
        self.element_bias = nn.Parameter(torch.zeros((num_bits, 2)) + torch.normal(0,0.01,(num_bits, 2)))
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(512, 1)
        self.swiglu = SwiGLU()

    def forward(self, x, mask):
        in_shape = x.shape
        if self.is_binary:
            x = x*2 - 1
        x = x * mask
        x = torch.stack([x,mask], 2)
        x = x * self.element_weight + self.element_bias
        return self.fc2(self.swiglu(self.fc1(x))).reshape(in_shape)

#Implements Swish gated linear units
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

#Implements the positionwise feedforward network for the transformer
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff*2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swiglu = SwiGLU()

    def forward(self, x):
        return self.fc2(self.swiglu(self.fc1(x)))

#Implements a basic feedforward network applied to translate transformer outputs into the enbedding (encoder) and from the embedding back to the input to a transformer (decoder)
class TranslationFF(nn.Module):
    def __init__(self, d_reserved, d_model, d_ff):
        super(TranslationFF, self).__init__()
        self.norm = nn.LayerNorm(d_reserved+d_model)
        self.fc1 = nn.Linear(d_reserved+d_model, d_ff*2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swiglu = SwiGLU()
        
    
    def forward(self, reserved, x):
        x_base = x.clone()
        x = torch.cat([reserved, x], axis=1)
        return self.fc2(self.swiglu(self.fc1(self.norm(x)))) + x_base

#Implements a layer of the transformer encoder with rotary positional encodings
#Can rotate keys/queries across multiple axis if input is more than 1d (like an image)
#A portion of each k/v is left unrotated, how much depends on how many axis need to be encoded
class EncoderLayer(nn.Module):
    def __init__(self, n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = num_heads
        self.d_ff = d_ff
        self.num_rotation_axis = num_rotation_axis
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.type_embedding = nn.Embedding(n_type_embedding, 2 * d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dim_per_rope = (d_model//num_heads)//(num_rotation_axis+1)
        self.rotary_emb = RotaryEmbedding(dim = self.dim_per_rope, cache_if_possible=False)
        

    #takes type, value, position, triplets
    #each is a vector of the same length where each element describes type, value, or position about the input data
    #seq_order is a tuple which can contain multiple axis
    def forward(self, x_type, x_value, seq_order):
        B, T, C = x_value.size()

        #eake queries, keys, values
        q, k, v  = self.c_attn(self.norm1(x_value)).split(self.d_model, dim=2)

        #encodes the TYPE of value into the key/queries by adding a linear projection to each k/q
        q_embed, k_embed = self.type_embedding(x_type).split(self.d_model, dim=2)
        q = q + q_embed
        k = k + k_embed

        #break k/q/v into distinct heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        #Add rotations to the correct portion of the keys/queries using the positions listed in the seq_order tuple
        for i, s in enumerate(seq_order):
            q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s.unsqueeze(1))
            k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s.unsqueeze(1))

        #Use flash attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(1, 2).contiguous().view(B, T, C)
        
        x_value = x_value + attn_output
        ff_output = self.feed_forward(self.norm2(x_value))
        x_value = x_value + ff_output
        return x_value

#Implements a tranformer encoder which produces a single output vector
#Options allow it to encode into binary vectors and to create embeddings based on partial inputs
class Encoder(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=False, learn_partial_encodings=False, num_rotation_axis=1):
        super(Encoder, self).__init__()
        self.binary_encodings = binary_encodings
        self.n_bits = n_bits
        self.learn_partial_encodings = learn_partial_encodings
        self.embedding = nn.Embedding(n_embedding, d_model)
        self.type_embedding = nn.Embedding(n_type_embedding, d_model)
        encoder = []
        for i in range(depth):
            encoder.append(EncoderLayer(n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=num_rotation_axis))
        self.encoder = nn.ModuleList(encoder)
        translation_layers = []
        for _ in range(n_translation_layers):
            translation_layers.append(TranslationFF(d_model, d_model, d_ff))
        self.translation = nn.ModuleList(translation_layers)
        self.norm_out = nn.LayerNorm(d_model)
        self.lin_out = nn.Linear(d_model, self.n_bits)
        self.binary_quantize = BinaryQuantizer()
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x_type, x_value, seq_order):
        device = self.dummy_param.device
        B, L = x_type.shape
        mask = torch.ones(L, device=device).repeat(B,1)
        if self.training and self.learn_partial_encodings:
            mask = torch.arange(L, device=device).repeat(B,1)/L < torch.empty((B,1), device=device).uniform_(0,1)
        
        #add pooling token pos
        new_seq_order = []
        for s in seq_order:
            new_seq_order.append(torch.cat([torch.zeros((B,1), device=device),s*mask], axis=1).float())
        seq_order = new_seq_order

        #add pooling token, pooling is done at this token. If strings include CLS the CLS will be scrambled like all other tokens and just marks the start
        x_value = torch.cat([torch.ones((B,1), device=device),x_value*mask], axis=1).long()
        x_type = torch.cat([torch.ones((B,1), device=device),x_type*mask], axis=1).long()

        #initial input to the transformer
        x_value = self.embedding(x_value) + self.type_embedding(x_type)
        
        for block in self.encoder:
            x_value = block(x_type, x_value, seq_order=seq_order)
        x_value = x_value[:,0,:]
        base_value = x_value.clone()
        for block in self.translation:
            x_value = block(base_value, x_value)
        x_value = self.lin_out(self.norm_out(x_value))

        if self.binary_encodings:
            x_value = self.binary_quantize(x_value)
        elif self.training: 
            x_value = x_value + torch.empty_like(x_value).normal_(std=0.01)
            
        return x_value

#Implements a layer of the transformer decoder with rotary positional encodings
#Can rotate keys/queries across multiple axis if input is more than 1d (like an image)
#A portion of each k/v is left unrotated, how much depends on how many axis need to be encoded
class DecoderLayer(nn.Module):
    def __init__(self, n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = num_heads
        self.d_ff = d_ff
        self.num_rotation_axis = num_rotation_axis
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.type_embedding = nn.Embedding(n_type_embedding, 2 * d_model)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dim_per_rope = (d_model//num_heads)//(num_rotation_axis+1)
        self.rotary_emb = RotaryEmbedding(dim = self.dim_per_rope, cache_if_possible=False)
    
    #CURRENT position/type is encoded into the queries while "NEXT" position is encoded into the keys
    #This allows information to be routed to allow for autoregressive prediction of "NEXT" tokens
    #Predicting "NEXT" tokens of unknown position and type is impossible so we have to give the self attenton mechanism this information
    def forward(self, x_type, x_value, seq_order):
        B, T, C = x_value.size()
        q, k, v  = self.c_attn(self.norm1(x_value)).split(self.d_model, dim=2)
        q_embed, k_embed = self.type_embedding(x_type).split(self.d_model, dim=2)

        #Encodings of type are offset for queries/keys in the decoder. See above note.
        q = q + q_embed[:,:-1]
        k = k + k_embed[:,1:]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        
        #Encodings of position are offset for queries/keys in the decoder. See above note.
        for i, s in enumerate(seq_order):
            q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(q[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s[:,:-1].unsqueeze(1))
            k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)] = self.rotary_emb.rotate_queries_or_keys(k[:,:,:,self.dim_per_rope*i:self.dim_per_rope*(i+1)], seq_order=s[:,1:].unsqueeze(1))

        #Flash attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, C)
        
        x_value = x_value + attn_output
        ff_output = self.feed_forward(self.norm2(x_value))
        x_value = x_value + ff_output
        return x_value

#Implements a tranformer decoder which takes a single vector embedding and autoregressively decodes it in a specified order
#Options allow it to use binary vectors and to create hierarchical embeddings by masking parts of the input embeddings
class Decoder(nn.Module):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=False, ordered_encodings=False, num_rotation_axis=1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.ordered_encodings = ordered_encodings
        self.binary_encodings = binary_encodings
        self.n_bits = n_bits
        self.apply_mask = ApplyMask(n_bits, is_binary=binary_encodings)
        self.embedding = nn.Embedding(n_embedding, d_model)
        self.type_embedding = nn.Embedding(n_type_embedding, d_model)
        decoder = []
        for i in range(depth):
            decoder.append(DecoderLayer(n_type_embedding, d_model, num_heads, d_ff, num_rotation_axis=num_rotation_axis))
        self.decoder = nn.ModuleList(decoder)
        self.lin_in = nn.Linear(self.n_bits, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.lin_out = nn.Linear(d_model, n_embedding)
        
        translation_layers = []
        for _ in range(n_translation_layers):
            translation_layers.append(TranslationFF(d_model,d_model, d_ff))
        self.translation = nn.ModuleList(translation_layers)
        self.register_buffer('mask', make_diag(self.n_bits))

    #makes a mask used to force the encoder to push more information into earlier dimensions of the embedding vector
    #masks the last X items in the vector where X is random
    #this forces more information into earlier dimensions
    def make_rand_mask(self, x_in, num_allowed_nodes=None):
        device = self.mask.device
        rand_mask = None
        B = x_in.shape[0]
        if self.ordered_encodings:
            #mask nodes to force more information into fewer nodes
            if num_allowed_nodes is not None:
                rand_mask = self.mask[num_allowed_nodes].repeat(B,1)
            else:
                r = torch.empty((B,), device=device).uniform_(0,1) ** 2
                r = r - r.min()
                r = r / r.max()
                r = torch.nan_to_num(r)
                rand_index = (r * (self.mask.shape[0]-1)).round().int()
                if self.training:
                    rand_mask = self.mask[rand_index]
                else:
                    rand_mask = self.mask[-1].repeat(B,1)
        else:
            rand_mask = self.mask[-1].repeat(B,1)
        return rand_mask

    def forward(self, x_type, x_value, seq_order, enc=None, num_allowed_nodes=None):
        device = self.mask.device

        #makes the mask which is randomly applied to the input embedding to force a hierarchy
        #masks the last X items in the vector where X is random
        #this forces more information into earlier dimensions
        if enc is None:
            enc = torch.zeros((x_value.shape[0],self.n_bits), device=device)
            enc_mask = torch.zeros(enc.shape, device=device)
        else:
            enc_mask = self.make_rand_mask(x_value, num_allowed_nodes=num_allowed_nodes)

        #add a leading zero to the vectors containing the positions of each element due to the right shift of decoder inputs needed for autoregressive causal masked training
        new_seq_order = []
        for s in seq_order:
            new_seq_order.append(torch.cat([torch.zeros((x_value.shape[0],1), device=device),s], axis=1).float())
        seq_order = new_seq_order

        #apply translate from embedding to a middle form before giving the vector to the transformer decoder
        enc = self.apply_mask(enc, enc_mask)
        enc = self.norm_in(self.lin_in(enc))
        base_value = enc.clone()
        
        for block in self.translation:
            enc = block(base_value, enc)
        
        enc = enc.unsqueeze(1)
        
        seq_len = x_value.shape[1]

        #add a leading one to the vectors containing the positions of each element due to the right shift of decoder inputs needed for autoregressive causal masked training
        #the token "1" identifies that this is the input token
        x_type = torch.cat([torch.ones((x_type.shape[0],1), device=device),x_type], axis=1).long()
        x_value = self.embedding(x_value) + self.type_embedding(x_type[:,:-1])

        #add the embedding to the input at the first position. Shift the sequence right one element for autoregressive training
        x_value = torch.cat([enc, x_value], axis=1)[:,0:seq_len,:]
        for block in self.decoder:
            x_value = block(x_type, x_value, seq_order=seq_order)

        x_value = self.lin_out(self.norm_out(x_value)).permute(0,2,1)
        return x_value

#Base Random Order Autoregressive Transformer Autoencoder class
#n_embedding = number of distinct VALUES for inputs
#n_type_embedding = number of distinct TYPES of inputs
#d_model base model dimension
#num_heads = number of attention heads, heads are of size d_modes//num_heads
#d_ff = dimensionality of the feedforward dimension of the feedforward layer, should be larger than d_model 2X or 4X as large are common
#depth = number of transformer layers in the encoder and decoder respectively. i.e. 8 = 8 encoder layers + 8 decoder layers
#n_translation_layers = number of feedforward layers after the encoder to translate output -> embedding and before the decoder to translate embedding -> decoder input
#n_bits = dimensionality of the embedding
#binary_encodings = round embedding values to binary. T/F
#ordered_encodings = learn hierarchical embedding vector
#make_encodings = learn to make embeddings T/F, this model works fine as a decoder only model, it doesn't have to make embeddings
#learn_partial_encodings = learn to make embeddings from incomplete input
#num_rotation_axis = number of positional embedding axis needed for the input. i.e tabular data = 0, sequence = 1, image = 2, etc.
class ChemROAR(nn.Module, PyTorchModelHubMixin):
    def __init__(self, n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits=256, binary_encodings=False, ordered_encodings=False, 
                 make_encodings=True, learn_partial_encodings=False, num_rotation_axis=1):
        super(ChemROAR, self).__init__()
        self.n_bits = n_bits
        self.make_encodings = make_encodings
        self.ordered_encodings = ordered_encodings
        self.binary_encodings = binary_encodings

        if make_encodings:
            self.encoder = Encoder(n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=binary_encodings, learn_partial_encodings=learn_partial_encodings, num_rotation_axis=num_rotation_axis)
        
        self.decoder = Decoder(n_embedding, n_type_embedding, d_model, num_heads, d_ff, depth, n_translation_layers, n_bits, binary_encodings=binary_encodings, ordered_encodings=ordered_encodings, num_rotation_axis=num_rotation_axis)
        
        self.register_buffer('mask', make_diag(self.n_bits))
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/SmilesTokenizer_PubChem_1M")
        self.best_complexity = -1

    #Generates molecules based on a provided node in the binary tree embedding
    #TODO: add KV cache
    def generate_molecules(self, target_node_vector, batch_size, evaluate_after_n_tokens=256, temperature=0.5, topk=5, topp=0.75):
        device = self.mask.device
        print("Generating samples with temp of", temperature, "topk of", topk, "and topp of", topp)
        assert isinstance(target_node_vector, list) or isinstance(target_node_vector, pd.Series) or isinstance(target_node_vector, np.ndarray) or isinstance(target_node_vector, torch.Tensor), "Expected input to be a a list/series/array/tensor"
        
        if not isinstance(target_node_vector, torch.Tensor):
            target_node_vector = torch.tensor(target_node_vector)
        target_node_vector = target_node_vector.float()
        
        self.eval()
        template_batch = (torch.zeros((batch_size,0)), torch.zeros((batch_size,256)))
        with torch.no_grad():
            _, x_test_seq = embed_sample(template_batch, device=device)
            x_test_type_orig, x_test_value_orig, seq_order_orig = x_test_seq
            x_test_value_orig = x_test_value_orig.clone()
            
            num_allowed_nodes = len(target_node_vector)
            
            enc = torch.zeros(self.n_bits, device=device)
            enc[:len(target_node_vector)] = target_node_vector.float()
            enc = enc.repeat(x_test_value_orig.shape[0],1)
            
            pbar = tqdm(range(x_test_value_orig.shape[1]))
            for i in pbar:
                x_test_type = x_test_type_orig[:,:i+1]
                x_test_value = x_test_value_orig[:,:i+1]
                seq_order = (seq_order_orig[:,:i+1],)

                output = self.decoder(x_test_type, x_test_value, seq_order, enc=enc, num_allowed_nodes=num_allowed_nodes)
                output = output.float().detach()
                
                gumbels = -torch.empty_like(output, device=device).exponential_().log()
                gumbel_probs = F.softmax(output + gumbels*temperature, dim=1)
                
                #top k
                p = (1/output.shape[1]) * topk
                quantiles = output.quantile(1-p, dim=1).reshape(output.shape[0],1,output.shape[2])
                gumbel_probs[output < quantiles] = 0
                
                #top p
                s = torch.softmax(output, dim=1)
                v, _ = s.sort(dim=1, descending=True)
                m = v.cumsum(dim=1) <= topp
                m[:,0,:] = True
                v = (m * v)
                v[v==0] = torch.inf
                mv, _ = v.min(dim=1)
                mv = mv.reshape(mv.shape[0], 1, mv.shape[1])
                gumbel_probs[s < mv] = 0
                
                x = gumbel_probs.argmax(dim=1)
                
                x_test_value_orig[:,i] = x[:,-1]

                #test to see how things are going
                if i % evaluate_after_n_tokens == 0 and i > 0:
                    x_test_value_final = x_test_value_orig.clone()
                    x_test_type_final = x_test_type_orig.clone()
                    x_test_seq_order_final = (seq_order_orig.clone(),)

                    sequence = x_test_value_final
                    
                    #make sure that the sequence has a stop token and zero everything after it
                    sequence = sequence * (((sequence == 13).cumsum(axis=1) == 0) | ((sequence == 13) & ((sequence == 13).cumsum(axis=1) == 1)))
                    x_test_value_final = sequence

                    #feed the generated sequence back into the encoder to make sure it matches the target encoding
                    re_enc = self.encoder(x_test_type_final, x_test_value_final, x_test_seq_order_final).float().detach()

                    #pull out ones which match and replicate them across the batch
                    success_key = torch.arange(x_test_value_orig.shape[0], device=device)
                    success_key = success_key[(re_enc[:,:target_node_vector.shape[0]].cpu() == target_node_vector).float().mean(axis=1) == 1]
                    
                    print(f"{len(success_key)} possibly valid sequences")
                    
                    if len(success_key) == 0:
                        return []

                    #fill the non-matching indices within the batch with matching sequences
                    if len(success_key) < x_test_value_orig.shape[0]:
                        temp_key = torch.randint(low=0, high=len(success_key), size=(x_test_value.shape[0] - len(success_key),), device=device)
                        temp_key = success_key[temp_key]
                        success_key = torch.cat([success_key, temp_key])

                    x_test_type_orig = x_test_type_orig[success_key]
                    x_test_value_orig = x_test_value_orig[success_key]
                    seq_order_orig = seq_order_orig[success_key]
                
            x_test_value_final = x_test_value_orig.clone()
            x_test_type_final = x_test_type_orig.clone()
            x_test_seq_order_final = (seq_order_orig.clone(),)

            sequence = x_test_value_final

            #make sure that the sequence has a stop token and zero everything after it
            sequence = sequence * (((sequence == 13).cumsum(axis=1) == 0) | ((sequence == 13) & ((sequence == 13).cumsum(axis=1) == 1)))
            x_test_value_final = sequence

            #run the generated sequence through the encoder
            re_enc = self.encoder(x_test_type_final, x_test_value_final, x_test_seq_order_final)
            re_enc = re_enc.float().detach()

            #check if it matches
            success_key = (re_enc[:,:target_node_vector.shape[0]].cpu() == target_node_vector).float().mean(axis=1) == 1
            
            #pull out ones which match and make sure they have a stop token indicating a completed sequence
            matching_sequences = x_test_value_final[success_key].cpu()
            matching_sequences = matching_sequences[(matching_sequences == 13).sum(axis=1) > 0]
            matching_sequences = matching_sequences * ((matching_sequences == 13).cumsum(axis=1) == 0)

            #gather generated sequences, convert back to strings, check validity using RDKit, de-duplicate
            valid_sequences = []
            for i in range(matching_sequences.shape[0]):
                smiles_str = self.tokenizer.decode(matching_sequences[i]).partition("[CLS]")[2].partition("[PAD]")[0].replace(" ", "")
                m = MolFromSmiles(smiles_str)
                if m is not None and len(smiles_str) > 0:
                    valid_sequences.append(MolToSmiles(m))
            valid_sequences = list(pd.Series(valid_sequences).drop_duplicates())
            
            return valid_sequences

    #embeds smiles strings into hierarchical binary embeddings
    def embed(self, x, batch=64):
        torch.manual_seed(0)
        self.eval()
        assert isinstance(x, list) or isinstance(x, str) or isinstance(x, pd.Series) or isinstance(x, np.array), "Expected input to be a string or list/array/series of strings"
        if isinstance(x, str):
            x = [x]
        tab_data, seq_data = prepare_smiles_without_properties(x, self.tokenizer)
        tab_data, seq_data = torch.tensor(tab_data), torch.tensor(seq_data)
        embeddings = []
        with torch.no_grad():
            i = 0
            while i < len(seq_data):
                x_tab_test, x_seq_test = tab_data[i:i+batch], seq_data[i:i+batch]
                x_test_tab, x_test_seq = embed_sample((x_tab_test, x_seq_test), device=self.mask.device)
                x_test_type_reordered, x_test_value_reordered, seq_order = scramble_order(x_test_tab, x_test_seq, device=self.mask.device, simple_scramble=True)
                embeddings.append(self.encoder(x_test_type_reordered, x_test_value_reordered, seq_order).cpu().numpy())
                i += batch
        
        return np.concatenate(embeddings)
        
    def forward(self, decoder_input, encoder_input=None, enc=None, num_allowed_nodes=None):
        if encoder_input is None:
            encoder_input = decoder_input
        encoder_type, encoder_value, encoder_seq_order = encoder_input
        if enc is None and self.make_encodings:
            enc = self.encoder(encoder_type, encoder_value, encoder_seq_order)

        decoder_type, decoder_value, decoder_seq_order = decoder_input
        out = self.decoder(decoder_type, decoder_value, decoder_seq_order, enc, num_allowed_nodes=num_allowed_nodes)
        
        return enc, out