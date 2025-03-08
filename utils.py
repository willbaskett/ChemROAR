import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import xgboost as xgb
import scipy.stats as st
from statsmodels.stats.proportion import proportion_confint
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import scipy
import sys
import time

def calculate_p(df1, df2, is_proportion=False):
    total_1 = df1.notna().sum(axis=0)
    total_2 = df2.notna().sum(axis=0)
    index = df1.columns
    p_1 = df1.mean(axis=0)
    p_2 = df2.mean(axis=0)
    diff = p_1 - p_2
    if is_proportion:
        p_hat = (df1.sum(axis=0) + df2.sum(axis=0))/(total_1 + total_2)
        z = (p_1 - p_2)/np.sqrt(p_hat*(1-p_hat)*(1/total_1 + 1/total_2))
        p = st.norm.sf(abs(z))*2
    else:
        s, p = st.ttest_ind(df1, df2, nan_policy="omit")
    
    return pd.DataFrame({"p1":p_1, "p2":p_2, "overall_diff": diff,"p":p}, index=index)

#tests how much accuracy suffers when each column is replaced by reversed values
#larger drops in accuracy signify more important features
def test_importance(clf, X, y, is_classification):
    importances = []
    if is_classification:
        score = roc_auc_score
        base_predictions = np.array(score(y, clf.predict_proba(X)[:, 1]))
    else:
        score = root_mean_squared_error
        base_predictions = np.array(score(y, clf.predict(X)))
        
    for i in range(X.shape[1]):
        X_copy = X.copy()
        X_copy[:,i] = np.random.permutation(X_copy[:,i])
        if is_classification:
            changed_prediction = np.array(score(y, clf.predict_proba(X_copy)[:, 1]))
        else:
            changed_prediction = np.array(score(y, clf.predict(X_copy)))
        
        importances.append(changed_prediction.reshape(-1,1))

    base_predictions = base_predictions.reshape(-1,1)
    changed_predictions = np.concatenate(importances, axis=1)
    if not is_classification:
        base_predictions *= -1
        changed_predictions *= -1
    return base_predictions, changed_predictions

#calculate how much randomizing a feature degrades model performance for all features
#loss is performance due to randomization is feature importance
#repeated for n_samples separate fits on different train/test splits
def get_feature_importance(clf, X, y, classes_weights, is_classification, n_samples=100, test_size=0.1, p_kept=0.9):
    colnames = X.columns
    
    X = np.array(X)
    y = np.array(y)
    bp, cp = [], []
    for i in tqdm(range(n_samples)):
        if is_classification:
            X_train, X_test, y_train, y_test, cw_train, cw_test = train_test_split(X, y, classes_weights, test_size=test_size, random_state=i)
            clf.fit(X_train, y_train, sample_weight=cw_train)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
            clf.fit(X_train, y_train)
        
        base_prediction, changed_prediction = test_importance(clf, X_test, y_test, is_classification=is_classification)
        bp.append(base_prediction)
        cp.append(changed_prediction)
    
    base_prediction = np.concatenate(bp)
    changed_prediction = np.concatenate(cp)
    
    df1 = pd.DataFrame(base_prediction.repeat((changed_prediction.shape[1]),axis=1), columns=colnames)
    df2 = pd.DataFrame(changed_prediction, columns=colnames)
    
    imp = calculate_p(df1, df2, is_proportion=False)
    imp.loc[imp.overall_diff <= 0, "p"] = 1
    imp = imp.sort_values("p")
    
    n_keep = int(len(imp)*p_kept)
    
    keep_features = imp[imp.overall_diff > 0]
    keep_features = keep_features[:n_keep]
    
    keep_features = keep_features.index
    return keep_features, imp, base_prediction.mean()

def fit(model, *args, **kwargs):
    class BarStdout:
        def write(self, text):
            if "totalling" in text and "fits" in text:
                self.bar_size = int(text.split("totalling")[1].split("fits")[0][1:-1])
                self.bar = tqdm(range(self.bar_size))
                self.count = 0
                return
            if "CV" in text and "END" in text and hasattr(self,"bar"):
                self.count += 1
                self.bar.update(n=self.count-self.bar.n)
                if self.count%(self.bar_size//10)==0:
                    time.sleep(0.1)
        def flush(self, text=None):
            pass
    sys.stdout.flush()
    default_stdout= sys.stdout
    sys.stdout = BarStdout()
    model.verbose = 10
    model.fit(*args, **kwargs)
    sys.stdout = default_stdout
    return model

def run_cv(X, y, max_classes=5, cv=10):
    best_score = -np.inf
    best_clf = None
    best_feature_set = None

    n_unique = y.iloc[:,0].unique().shape[0]
    y = np.array(y).reshape(-1)

    is_classification = True
    if n_unique > max_classes:
        is_classification = False

    classes_weights = None
    if is_classification:
        classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y)

    if is_classification: 
        estimator = xgb.XGBClassifier(
            nthread=8,
            seed=42
            )
    else:
        estimator = xgb.XGBRegressor(
            nthread=8,
            seed=42
            )
        
    parameters = {
        'max_depth': [1, 2, 4, 8],
        'n_estimators': [1, 2, 4, 8, 16, 32, 64, 128],
        }

    if is_classification:
        print("Fitting Classification")
        clf = GridSearchCV(estimator=estimator, scoring="roc_auc", param_grid=parameters, cv=cv)
    else:
        print("Fitting Regression")
        clf = GridSearchCV(estimator=estimator, scoring="neg_root_mean_squared_error", param_grid=parameters, cv=cv)
    
    print("-----------------------------------------")
    print("Fitting with ", X.shape[1], " features")
    clf = fit(clf, X, y, sample_weight=classes_weights)
    print("BEST CV SCORE: ", clf.best_score_)
    
    f, imp, base_score = get_feature_importance(clf.best_estimator_, X, y, classes_weights, is_classification=is_classification)
    imp.to_csv("feature_importances.csv")
    print("SCORE: ",base_score)
    if base_score >= best_score:
        best_score = base_score
        best_clf = clf.best_estimator_
        best_feature_set = X.columns
    
    print("Best Score: ", best_score)
    return best_score, best_clf, best_feature_set

def backward_selector(X, y, p_kept=0.9, max_classes=5, patience=3):

    X, X_eval, y, y_eval = train_test_split(X, y, test_size=0.5, random_state=42)
    best_score = -np.inf
    best_clf = None
    best_feature_set = None

    n_unique = y.iloc[:,0].unique().shape[0]
    y = np.array(y).reshape(-1)

    is_classification = True
    if n_unique > max_classes:
        is_classification = False

    classes_weights = None
    if is_classification:
        classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y)

    if is_classification: 
        estimator = xgb.XGBClassifier(
            nthread=16,
            seed=42
            )
    else:
        estimator = xgb.XGBRegressor(
            nthread=16,
            seed=42
            )
        
    parameters = {
        'max_depth': [1, 2, 4, 8],
        'n_estimators': [1, 2, 4, 8, 16,32, 64, 128,]
        }

    if is_classification:
        print("Fitting Classification")
        clf = GridSearchCV(estimator=estimator, scoring="roc_auc", param_grid=parameters, cv=10)
    else:
        print("Fitting Regression")
        clf = GridSearchCV(estimator=estimator, scoring="neg_root_mean_squared_error", param_grid=parameters, cv=10)

    iters_since_best = 0
    while X.shape[1] > 0 and iters_since_best < patience:
        print("-----------------------------------------")
        print("Fitting with ", X.shape[1], " features")
        clf = fit(clf, X, y, sample_weight=classes_weights)
        print("BEST CV SCORE: ", clf.best_score_)
        
        eval_clf = clf.best_estimator_
        #eval_clf.fit(X, y, sample_weight=classes_weights)
        
        score = roc_auc_score
        eval_score = score(y_eval, eval_clf.predict_proba(X_eval)[:, 1])
        
        f, imp, base_score = get_feature_importance(clf.best_estimator_, X, y, classes_weights, p_kept=p_kept, is_classification=is_classification)
        imp.to_csv("feature_importances.csv")
        print("SCORE: ",eval_score)
        print("RANDOM CV SCORE: ", base_score)
        if eval_score >= best_score:
            best_score = eval_score
            best_clf = clf.best_estimator_
            best_feature_set = X.columns
            iters_since_best = 0
            imp.to_csv("best_feature_importances.csv")
        else:
            iters_since_best += 1
        
        X_eval = X_eval.loc[:,f]
        X = X.loc[:,f]
    
    print("Best Score: ", best_score)
    return best_score, best_clf, best_feature_set

def calculate_tree_p(x1, x2, s1, s2, c1, c2, is_binary=False, alpha=0.01):
    if is_binary:
        k1 = x1*c1
        k2 = x2*c2
        ci_l, ci_h = proportion_confint(k1, c1, alpha, method="beta")
        p_hat = (k1 + k2)/(c1 + c2)
        z = (x1-x2)/np.sqrt((p_hat*(1-p_hat)) * (1/c1 + 1/c2))
        p = scipy.stats.norm.sf(np.abs(z))*2 
    else:
        n_sided = 2 # 2-sided test
        z_crit = scipy.stats.norm.ppf(1-alpha/n_sided)
        ci_z = (s1/np.sqrt(c1))
        ci_l, ci_h = x1 - ci_z*z_crit, x1 + ci_z*z_crit
        z = (x1-x2)/np.sqrt((s1**2)/c1 + (s2**2)/c2)
        p = scipy.stats.norm.sf(np.abs(z))*2
    return np.abs(z), ci_l, ci_h, p

def make_binary_plot(encodings, levels=8, n_tree_bits = 32):
    X = encodings[encodings.columns[encodings.columns != "label"]]
    y = encodings["label"]
    y_is_binary = y.nunique() == 2
    y_is_categorical = y.nunique() < 25 and not y_is_binary
    
    notnan_key = ~y.isna()
    X = X[notnan_key]
    y = y[notnan_key]

    
    if encodings.shape[1] < n_tree_bits:
        n_tree_bits = encodings.shape[1]
        
    data_dict = {}
    base_data = np.array(X.astype(bool))
    first_item = y.unique()[0]
    base_labels = np.array(y).reshape(-1)
    if y_is_binary:
        try:
            base_labels = base_labels.astype(float)
        except:
            base_labels = base_labels == first_item
    
    for i in range(n_tree_bits):
        for j in range(len(base_data)):
            key = tuple(np.array(base_data[j,:i]))
            if key not in data_dict:
                data_dict[key] = [base_labels[j]]
            else:
                data_dict[key].append(base_labels[j])
        for key in data_dict:
            if isinstance(data_dict[key], list):
                data_dict[key] = {"mean":np.array(data_dict[key]).mean(), "std":np.array(data_dict[key]).std(ddof=1), "count":len(data_dict[key])}
                b = data_dict[tuple()]
                d = data_dict[key]
                z, ci_l, ci_h, p = calculate_tree_p(d["mean"], b["mean"], d["std"], b["std"], d["count"], b["count"], is_binary=y_is_binary)
                data_dict[key]["ci_l"] = ci_l
                data_dict[key]["ci_h"] = ci_h
                data_dict[key]["z"] = z
                data_dict[key]["p"] = p
                if len(key) > 0:
                    if "children" not in data_dict[key[:-1]]:
                        data_dict[key[:-1]]["children"] = [key]
                    else:
                        data_dict[key[:-1]]["children"].append(key)
        k, l, s, c, ci_l, ci_h, z, p, = [], [], [], [], [], [], [], []
    
    if not y_is_categorical:
        mean_y = base_labels.mean()
        std_y = base_labels.std()
    
    lines_to_plot = []
    max_val = (2 ** levels)
    for key in data_dict:
        if "children" in data_dict[key]:
            level = len(key)
            if level >= levels: break
            max_val_for_parent_level = 2 ** level - 1
            max_val_for_child_level = 2 ** (level+1) - 1
            for child_key in data_dict[key]["children"]:
                if data_dict[child_key]["count"] >= 10:
                    if level == 0:
                        parent_point = max_val/2
                    else:
                        parent_mult_matrix = 2 ** np.flip(np.arange(level)).reshape(-1,  level).astype(int)
                        parent_point = ((np.array(key).astype(int)[:level] * parent_mult_matrix).sum())
                        offset_parent = max_val / ((2 ** level)*2)
                        parent_point = 2*offset_parent*parent_point + offset_parent
                    child_mult_matrix = 2 ** np.flip(np.arange(level+1)).reshape(-1,  level+1).astype(int)
                    child_point = (np.array(child_key).astype(int)[:level+1] * child_mult_matrix).sum()
                    offset_child = max_val / ((2 ** (level+1))*2)
                    child_point = (2*offset_child*child_point + offset_child)
                    child_value = data_dict[child_key]["mean"]
                    if y_is_binary:
                        child_value -= mean_y
                        child_value /= (mean_y*(1-mean_y))
                    else:
                        child_value -= mean_y
                        child_value /= std_y
                    child_value += 3
                    child_value /= 6
                    child_value = np.clip(child_value, 0.00001, 0.99999)
                    lines_to_plot.append(((parent_point, child_point), (level*-1, (level+1)*-1), child_value))
    
    plt.figure(figsize=(25,3))
    for x in lines_to_plot:
        plt.plot(x[0], x[1], color=plt.cm.coolwarm(x[2]), linewidth=2)
    
    # Add labels and title (optional)
    plt.xlabel('Cluster')
    plt.ylabel('Tree Level')
    plt.title('Binary Tree')
    
    # Show the plot
    plt.show()
    
    k, l, s, c, ci_l, ci_h, z, p, = [], [], [], [], [], [], [], []
        
    for key in data_dict:
        result = data_dict[key]
        k.append(np.array(key))
        l.append(len(key))
        s.append(result["mean"])
        ci_l.append(result["ci_l"])
        ci_h.append(result["ci_h"])
        c.append(result["count"])
        z.append(result["z"])
        p.append(result["p"])
    df = pd.DataFrame({"key":k,"depth": l, "mean":s,"ci_l":ci_l, "ci_h":ci_h,  "count":c, "z":z, "p":p})
    print(df.loc[0])
    df = df.sort_values("depth").drop_duplicates(subset=["mean", "count", "z"])
    df = df.sort_values(["z"], ascending=False)
    df = df[df["count"] >= 10]
    df["fdr_p"] = st.false_discovery_control(df["p"], method="by")
    return df

def make_binary_plot_v1(encodings, levels=7, n_tree_bits = 32):
    X = encodings[encodings.columns[encodings.columns != "label"]]
    y = encodings["label"]
    y_is_binary = y.nunique() == 2
    y_is_categorical = y.nunique() < 25 and not y_is_binary
    max_val = (2 ** levels)
    df = pd.DataFrame({"leaf":[], "level":[], "y":[]})
    for i in range(0,levels+1):
        max_val_for_level = 2 ** i - 1
        
        if i == 0:
            leaf = max_val/2
        else:
            center_val = (max_val - max_val_for_level)/2
            mult_matrix = 2 ** np.flip(np.arange(i)).reshape(-1,  i).astype(int)
            leaf = ((np.array(X).astype(int)[:,:i] * mult_matrix).sum(axis=1))
            offset = max_val / ((2 ** i)*2)
            leaf = 2*offset*leaf + offset
        df_part = pd.DataFrame({"leaf":leaf, "level":levels-i, "y":y})
        df = pd.concat([df, df_part])
    
    df["yq"] = y
    if not y_is_categorical and not y_is_binary:
        df["yq"] = pd.qcut(df["y"], 10, duplicates="drop")
    plt.figure(figsize=(20,3))
    palette = None
    if not y_is_categorical:
        palette = "coolwarm"
    if y_is_binary:
        palette = "prism"
    sns.scatterplot(data=df, x="leaf", y="level", hue="yq", alpha=0.1, palette=palette, legend=False).set_title("Plot")


    notnan_key = ~y.isna()
    X = X[notnan_key]
    y = y[notnan_key]

    
    if encodings.shape[1] < n_tree_bits:
        n_tree_bits = encodings.shape[1]
        
    data_dict = {}
    base_data = np.array(X.astype(bool))
    first_item = y.unique()[0]
    base_labels = np.array(y).reshape(-1)
    if y_is_binary:
        try:
            base_labels = base_labels.astype(float)
        except:
            base_labels = base_labels == first_item
    
    for i in range(n_tree_bits):
        for j in range(len(base_data)):
            key = tuple(np.array(base_data[j,:i]))
            if key not in data_dict:
                data_dict[key] = [base_labels[j]]
            else:
                data_dict[key].append(base_labels[j])
        for key in data_dict:
            if isinstance(data_dict[key], list):
                data_dict[key] = {"mean":np.array(data_dict[key]).mean(), "std":np.array(data_dict[key]).std(ddof=1), "count":len(data_dict[key])}
                b = data_dict[tuple()]
                d = data_dict[key]
                z, ci_l, ci_h, p = calculate_tree_p(d["mean"], b["mean"], d["std"], b["std"], d["count"], b["count"], is_binary=y_is_binary)
                data_dict[key]["ci_l"] = ci_l
                data_dict[key]["ci_h"] = ci_h
                data_dict[key]["z"] = z
                data_dict[key]["p"] = p
                if len(key) > 0:
                    if "children" not in data_dict[key[:-1]]:
                        data_dict[key[:-1]]["children"] = [key]
                    else:
                        data_dict[key[:-1]]["children"].append(key)
        k, l, s, c, ci_l, ci_h, z, p, = [], [], [], [], [], [], [], []
    
    for key in data_dict:
        result = data_dict[key]
        k.append(np.array(key))
        l.append(len(key))
        s.append(result["mean"])
        ci_l.append(result["ci_l"])
        ci_h.append(result["ci_h"])
        c.append(result["count"])
        z.append(result["z"])
        p.append(result["p"])
    df = pd.DataFrame({"key":k,"depth": l, "mean":s,"ci_l":ci_l, "ci_h":ci_h,  "count":c, "z":z, "p":p})
    print(df.loc[0])
    df = df.sort_values("depth").drop_duplicates(subset=["mean", "count", "z"])
    df = df.sort_values(["z"], ascending=False)
    df = df[df["count"] >= 5]
    df["fdr_p"] = st.false_discovery_control(df["p"], method="by")
    return df