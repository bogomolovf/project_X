import os, json, subprocess, yaml, joblib, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "proj1-local")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "proj1-model")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


p = yaml.safe_load(open("params.yaml"))
feat, proc = p["data"]["features"], p["data"]["processed"]

Xtr = pd.read_csv(os.path.join(feat, "X_train.csv"))
Xte = pd.read_csv(os.path.join(feat, "X_test.csv"))
ytr = pd.read_csv(os.path.join(proc, "y_train.csv"))["target"]
yte = pd.read_csv(os.path.join(proc, "y_test.csv"))["target"]

model_type = p["model"]["type"]
if model_type == "RandomForestClassifier":
    model = RandomForestClassifier(
        n_estimators=int(p["model"].get("n_estimators", 100)),
        max_depth=None if p["model"].get("max_depth") in [None, "None"] else int(p["model"]["max_depth"]),
        random_state=p["train"]["random_state"]
    )
else:
    model = LogisticRegression(max_iter=500, random_state=p["train"]["random_state"])

run_name = f"{model_type}-md={p['model'].get('max_depth')}-ne={p['model'].get('n_estimators',100)}"

def log_params_flat(prefix, d):
    for k, v in d.items():
        mlflow.log_param(f"{prefix}.{k}", json.dumps(v) if isinstance(v, (dict, list)) else v)

def get_git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return None

with mlflow.start_run(run_name=run_name) as run:
    log_params_flat("data", p["data"])
    log_params_flat("train", p["train"])
    log_params_flat("model", p["model"])
    rev = get_git_rev()
    if rev: mlflow.set_tag("git_rev", rev)

    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, proba)),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


    mlflow.log_metrics(metrics)


    signature = infer_signature(Xtr, model.predict(Xtr))
    input_example = Xtr.head(5)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",                        
        registered_model_name=REGISTERED_MODEL_NAME,
        signature=signature,
        input_example=input_example,
    )

print(json.dumps(metrics, indent=2))

