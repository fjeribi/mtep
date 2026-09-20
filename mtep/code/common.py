"""
common.py -- shared paths, data loading, windowing, models and metrics.

Every script imports from here so that all studies use the same folds,
the same models, the same hyperparameters and the same metrics.
Paths are resolved relative to the repository root, so the release runs
from any location:  python code/<script>.py
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset.npz")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
SEED = 1337

# ---------------- single hyperparameter set, used by every study -------------
RF_PARAMS = dict(n_estimators=400, class_weight="balanced_subsample",
                 min_samples_leaf=3, n_jobs=1)
LR_PARAMS = dict(max_iter=3000, class_weight="balanced", C=1.0)
SVM_PARAMS = dict(class_weight="balanced", C=1.0, kernel="rbf", gamma="scale")
TF_PARAMS = dict(d_model=48, nhead=4, layers=2, dropout=0.2, lr=1e-3,
                 weight_decay=1e-2, epochs=15, patience=4, batch=256)
N_BOOT = 2000          # participant-clustered bootstrap replicates


def rpath(*p):
    os.makedirs(RES, exist_ok=True)
    return os.path.join(RES, *p)


def save_json(obj, *p):
    path = rpath(*p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)
    return path


def load_json(*p):
    with open(rpath(*p)) as fh:
        return json.load(fh)


def load():
    """Feature matrix, press events, meta and feature groups.

    Groups are assigned by name prefix: eeg_*, face_*, and perf_* (task and
    timing telemetry, which includes the per-round frame count because it is a
    direct measure of round duration).
    meta columns: session index, participant id, day, round id (1-based),
    number of rounds in session.
    """
    d = np.load(DATA, allow_pickle=True)
    names = [str(n) for n in d["feature_names"]]
    groups = dict(
        eeg=np.array([i for i, n in enumerate(names) if n.startswith("eeg_")]),
        face=np.array([i for i, n in enumerate(names) if n.startswith("face_")]),
        perf=np.array([i for i, n in enumerate(names) if n.startswith("perf_")]))
    return (d["X"].astype(np.float32), d["y_event"].astype(int),
            d["meta"].astype(int), names, groups)


def session_order(meta):
    """Yield (session, row indices sorted by round) for every session."""
    sess, rnd = meta[:, 0], meta[:, 3]
    for s in np.unique(sess):
        m = np.where(sess == s)[0]
        yield s, m[np.argsort(rnd[m])]


def summarise(A):
    """Window summary for classical models: per-feature mean, SD, final value."""
    return np.concatenate([A.mean(1), A.std(1), A[:, -1, :]], axis=1)


def fit_scaler(Xtr):
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    med = np.nanmedian(flat, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    flat = np.where(np.isnan(flat), med, flat)
    return med, flat.mean(0), flat.std(0) + 1e-6


def apply_scaler(X, sc):
    med, mu, sd = sc
    X = np.where(np.isnan(X), med, X)
    return np.clip((X - mu) / sd, -8, 8).astype(np.float32)


# ------------------------------- models --------------------------------------
def make_classical(kind, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    if kind == "logreg":
        return LogisticRegression(**LR_PARAMS)
    if kind == "rf":
        return RandomForestClassifier(random_state=seed, **RF_PARAMS)
    if kind == "svm":
        return SVC(random_state=seed, **SVM_PARAMS)
    raise ValueError(kind)


def train_transformer(Xtr, ytr, Xva, yva, seed=SEED):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    P = TF_PARAMS
    W, d_in = Xtr.shape[1], Xtr.shape[2]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            dm = P["d_model"]
            self.proj = nn.Linear(d_in, dm)
            pe = torch.zeros(W, dm)
            pos = torch.arange(W).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, dm, 2).float() * (-np.log(10000.0) / dm))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(dm, P["nhead"], dm * 4, P["dropout"],
                                               batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, P["layers"])
            self.head = nn.Sequential(nn.LayerNorm(dm), nn.Dropout(P["dropout"]),
                                      nn.Linear(dm, 1))

        def forward(self, x):
            return self.head(self.enc(self.proj(x) + self.pe)[:, -1]).squeeze(-1)

    model = Net()
    pos_w = torch.tensor([max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum()))])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    Xv = torch.from_numpy(Xva)
    best, state, bad = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(P["epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), P["batch"]):
            idx = perm[i:i + P["batch"]]
            opt.zero_grad()
            crit(model(Xt[idx]), yt[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xv)).numpy()
        score = roc_auc_score(yva, p) if len(np.unique(yva)) > 1 else 0.5
        if score > best:
            best, bad = score, 0
            state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= P["patience"]:
                break
    if state:
        model.load_state_dict(state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(X))).numpy()
    return predict


# ------------------------------- metrics -------------------------------------
def metrics(y, p, thr=0.5):
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                                 average_precision_score, confusion_matrix)
    yhat = (p >= thr).astype(int)
    out = dict(balanced_accuracy=float(balanced_accuracy_score(y, yhat)),
               prevalence=float(np.mean(y)), n=int(len(y)), n_pos=int(np.sum(y)))
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out["confusion"] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


def cluster_bootstrap_auc(y, p, groups, n_boot=N_BOOT, seed=SEED):
    """95% CI for pooled ROC-AUC, resampling participants with replacement."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[idx])) > 1:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def loso(Xw, yw, uw, kind, seed=SEED):
    """Grouped leave-one-participant-out CV, identical for every model.

    In every outer fold the same validation participants (1/8 of the training
    participants, at least one) are removed from the training set for ALL
    models, so every model is fitted on exactly the same participants. The
    Transformer uses the validation participants for early stopping; the
    classical models do not use them at all. If the validation participants
    lack one of the classes, the fold is flagged (fallback) and the
    Transformer's early stopping falls back to the training set.
    """
    users = np.unique(uw)
    rng = np.random.RandomState(seed)
    oof = np.full(len(yw), np.nan)
    folds, n_fallback = [], 0
    for u in users:
        te = uw == u
        tr_users = np.setdiff1d(users, [u])
        va_users = rng.choice(tr_users, size=max(1, len(tr_users) // 8), replace=False)
        va = np.isin(uw, va_users)
        tr = (~te) & (~va)
        if len(np.unique(yw[tr])) < 2:
            continue
        sc = fit_scaler(Xw[tr])
        Xtr, Xte = apply_scaler(Xw[tr], sc), apply_scaler(Xw[te], sc)
        if kind == "transformer":
            Xva, yva = apply_scaler(Xw[va], sc), yw[va]
            if len(np.unique(yva)) < 2:
                n_fallback += 1
                Xva, yva = Xtr, yw[tr]
            p = train_transformer(Xtr, yw[tr], Xva, yva, seed)(Xte)
        else:
            clf = make_classical(kind, seed)
            clf.fit(summarise(Xtr), yw[tr])
            if kind == "svm":
                # logistic link on the decision function: ranking is unchanged and
                # the 0.5 threshold coincides with the SVM's own decision boundary
                p = 1.0 / (1.0 + np.exp(-clf.decision_function(summarise(Xte))))
            else:
                p = clf.predict_proba(summarise(Xte))[:, 1]
        oof[te] = p
        if len(np.unique(yw[te])) > 1:
            m = metrics(yw[te], p)
            m["participant"] = int(u)
            folds.append(m)
    ok = ~np.isnan(oof)
    pooled = metrics(yw[ok], oof[ok])
    pooled["auc_ci95"] = cluster_bootstrap_auc(yw[ok], oof[ok], uw[ok], seed=seed)
    agg = {}
    for k in ["balanced_accuracy", "roc_auc", "pr_auc"]:
        v = [f[k] for f in folds if f[k] is not None]
        agg[k] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)),
                      n_folds=len(v)) if v else None
    return dict(model=kind, n_windows=int(ok.sum()), prevalence=float(yw[ok].mean()),
                n_participants_scored=int(len(folds)),
                n_participants_total=int(len(users)),
                transformer_fallback_folds=n_fallback,
                pooled=pooled, fold_mean=agg, per_fold=folds,
                oof=[float(x) for x in oof[ok]], y=[int(x) for x in yw[ok]],
                user=[int(x) for x in uw[ok]])
