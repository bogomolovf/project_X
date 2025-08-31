import os, yaml, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))
raw_dir = params["data"]["raw"]
proc_dir = params["data"]["processed"]
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(proc_dir, exist_ok=True)

raw_csv = os.path.join(raw_dir, "dataset.csv")
if not os.path.exists(raw_csv):
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame
    df.to_csv(raw_csv, index=False)

df = pd.read_csv(raw_csv)
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["train"]["test_size"], random_state=params["train"]["random_state"], stratify=y
)
X_train.to_csv(os.path.join(proc_dir, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(proc_dir, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(proc_dir, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(proc_dir, "y_test.csv"), index=False)
