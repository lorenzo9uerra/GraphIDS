# SAFE Model Modifications

Due to the absence of an explicit license for the original SAFE model, we do **not redistribute its code**. 
However, we provide the following instructions on how to apply our modifications externally to the original code and reproduce the results shown in our paper. We have received explicit permission from the SAFE authors to use their code for research purposes.

## Prerequisites

Before applying our changes, you need to manually download the original SAFE model code from the repository:

- SAFE repository: [https://github.com/ElvinLit/SAFE/](https://github.com/ElvinLit/SAFE/)

We provide an `environment.yaml` file in the same directory of this README to facilitate setting up the necessary dependencies for the SAFE model. You can create and activate the conda environment with the following command:

```bash
conda env create -f environment.yaml
conda activate SAFE
```

This ensures all the required dependencies are installed with the correct versions.

You can clone the original SAFE repository from GitHub:
```bash
git clone https://github.com/ElvinLit/SAFE/
```

## Instructions to Apply Modifications

We have preprocessed the NetFlow datasets using the following code:
```python
df = pd.read_csv(os.path.join(sys.argv[2], name, f"{name}.csv"))
if "CSE-CIC-IDS2018" in name:
    df = df.groupby(by="Attack").sample(frac=0.2, random_state=seed)
X = df.drop(columns=["Attack", "Label"])
y = df[["Attack", "Label"]]
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
# We exclude IP addresses from the features to prevent overfitting,
# as in some datasets attacks only originate from a small number of IPs
edge_features = [col for col in X.columns if col not in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]]

df = pd.concat([X, y], axis=1)
df_train, df_val_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=y["Attack"])
df_train = df_train[df_train["Label"] == 0]
df_val, df_test = train_test_split(df_val_test, test_size=0.5, random_state=seed, stratify=df_val_test["Attack"])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train[edge_features].values)
X_val = scaler.transform(df_val[edge_features].values)
X_test = scaler.transform(df_test[edge_features].values)

y_train = df_train["Label"].values
y_val = df_val["Label"].values
y_test = df_test["Label"].values
```

Refer to the file `SAFE/SAFE_MAE.ipynb` in the original repository to train the MAE model and produce a model checkpoint. Do not proceed to the "Reconstruction Classification" section of the notebook, as it would use the reconstruction error to perform the classification, which is not the process explained in the SAFE paper. However, examining the ROC curve for the reconstruction error classification, it has an AUC of 0.5, indicating no discriminative ability between normal and anomalous traffic.
To preprocess the data, be sure not modify the labels like it was shown in the original notebook for label 11 and make sure to exclude anomalous samples from the training set as shown above, to create a comparable setting to the other models. The model should not be trained to reconstruct anomalies.

Refer to the file `SAFE/SAFE_Detection.ipynb` to load the MAE checkpoint, create the embeddings and use the LOF anomaly detection algorithm to perform the final classification. Fit the LOF model exclusively on benign data, consistent with MAE and our models.

To perform the final classification, we introduced the following custom code:

```python
val_scores = best_lof.decision_function(val_latent_features)
val_scores_inverted = -val_scores

test_scores = best_lof.decision_function(test_latent_features)
test_scores_inverted = -test_scores

pr_auc = average_precision_score(y_test, test_scores_inverted, average="macro")

thresholds = np.linspace(min(val_scores_inverted), max(val_scores_inverted), 500)
best_f1 = 0
best_threshold = 0

for threshold in thresholds:
    y_pred = (val_scores_inverted > threshold).astype(int)
    f1 = f1_score(y_val, y_pred, average="macro")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold}")
print(f"Best validation F1-score: {best_f1}")

y_pred = (test_scores_inverted > best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("PR-AUC:", pr_auc)
print(f"Accuracy: {accuracy}")
print(f"Precision (macro): {precision_macro}")
print(f"Recall (macro): {recall_macro}")
print(f"F1 Score (macro): {f1_macro}")
print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))
```

Unifying the code under a single Python script could simplify the process.