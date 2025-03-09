# ChemROAR
ChemROAR is a novel generative embedding architecture for clustering and generating drug molecules. It finds heirarchical clusters of molecules with specific properties and then generates new molecules using those clusters. This allows for the generation of molecules with arbitrary properties without needing much data or any additional model training.

ChemROAR is a transformer based autoencoder that generates a heirarchical binary embedding which naturally clusters data into discrete groups. It uses a Random Order AutoRegressive (ROAR) decoder to learn complex dependencies between SMILES sequences and other molecular properties in the dataset. It was trained on PubChem10m.

ROAR models are capable of modeling mixed-type data, allowing seamless integration of SMILES sequences with tabular molecular properties during training. Each token input in a ROAR model consists of a Type, Position, Value triplit. This format allows multiple data modalities to be included in the same context without any architectural considerations and without requiring an impractically large model vocabulary. Random order modeling forces ChemROAR to learn to predict molecular properties given SMILES sequences and vice versa. As autoregressive models they also remain capable of generative modeling.

ChemROAR combines an encoder which produces heirarchical discrete embeddings and a random order decoder. This forces the model to learn to hierarchically cluster molecules based on high-level  similarities. Higher-level clusters are less specific and contain more molecules. Lower-level clusters are more specifc and contain fewer molecules. Every cluster is well defined and contains a specific number of molecules. This allows us to use conventional statistics to quantify uncertainty about the properties of each cluster and calculate a confidence interval. This means that we can generate new drug molecules and have a well defined idea of how likely they are to do what we want.

# Example: Clusters of Molecules Which Inhibit HIV Replication
### We observe clusters where >90% of molecules inhibit HIV compared to the baseline 3.5% seen in the dataset as a whole
![Clusters Found by ChemROAR](resources/clusters.png)

# Examples of Molecules Generated by ChemROAR Using the Cluster Most Likely to Inhibit HIV Replication
### Obvious similarities can be seen between different molecules in this cluster
![New Molecules Generated By Med-ROAR](resources/example_molecules.png)

# Usage
See [the demo notebook](demo.ipynb) for an illustrated version
```
import pandas as pd
import torch
import ROAR
from utils import make_binary_plot

data = <load your data here>

smiles_strings = data[<get smiles strings>]
property = data[<get a known property>]

#load model from huggingface
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ROAR.ChemROAR.from_pretrained("willbaskett/ChemROAR").to(device)

embed data
embeddings = model.embed(smiles_strings)
X = pd.DataFrame(embeddings).astype(bool)
y = property.copy()

encodings = X.copy()
encodings["label"] = y

#get clusters of molecules most associated with the property, sorted.
clusters = make_binary_plot(encodings).sort_values("ci_l", ascending=False)

#get the node associated with the best cluster, defined as a T/F traversal of the tree
target_node_vector = torch.tensor(clusters.iloc[0].key).float()

#generate ~100 molecules from the best cluster
possible_solutions = []
while len(possible_solutions) < 100:
    generated_molecules = model.generate_molecules(target_node_vector, batch_size=128, evaluate_after_n_tokens=128, temperature=1, topk=500, topp=0.95)
    possible_solutions += generated_molecules
    possible_solutions = list(set(possible_solutions))
    print(f"{len(possible_solutions)} total molecules found so far")

```
