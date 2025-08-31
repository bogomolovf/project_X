import os, yaml, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

p = yaml.safe_load(open("params.yaml"))
feat = p["data"]["features"]
proc = p["data"]["processed"]

Xte = pd.read_csv(os.path.join(feat, "X_test.csv"))
yte = pd.read_csv(os.path.join(proc, "y_test.csv"))["target"]
model = joblib.load("models/model.pkl")

proba = model.predict_proba(Xte)[:, 1]
fpr, tpr, _ = roc_curve(yte, proba)
roc_auc = auc(fpr, tpr)

out_dir = "reports/plots"
os.makedirs(out_dir, exist_ok=True)


pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(out_dir, "roc.csv"), index=False)


plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(out_dir, "roc.png"), bbox_inches="tight")
plt.close()
print(f"Saved: {out_dir}/roc.csv and {out_dir}/roc.png")
