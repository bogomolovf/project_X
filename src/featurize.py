import os, yaml, pandas as pd
from sklearn.preprocessing import StandardScaler

p = yaml.safe_load(open("params.yaml"))
proc, feat = p["data"]["processed"], p["data"]["features"]
os.makedirs(feat, exist_ok=True)

Xtr = pd.read_csv(os.path.join(proc, "X_train.csv"))
Xte = pd.read_csv(os.path.join(proc, "X_test.csv"))

if p["train"].get("scale", True):
    scaler = StandardScaler()
    Xtr[:] = scaler.fit_transform(Xtr)
    Xte[:] = scaler.transform(Xte)

Xtr.to_csv(os.path.join(feat, "X_train.csv"), index=False)
Xte.to_csv(os.path.join(feat, "X_test.csv"), index=False)

