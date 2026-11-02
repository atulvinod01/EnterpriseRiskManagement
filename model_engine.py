"""Hybrid fraud detector: statistical checks + Isolation Forest + XGBoost, explained with SHAP.

Input: one row per transaction with customer_id, timestamp, amount, device_id, city and
account_open_date. is_fraud (0/1) is optional and is only needed to train or to measure.
Every feature uses only the customer's earlier transactions, so scoring a transaction
never peeks at the future.
"""
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

REQUIRED = ["customer_id", "timestamp", "amount", "device_id", "city", "account_open_date"]
FEATURES = ["amount_vs_history", "new_device", "new_location", "account_age_days",
            "night_time", "txns_last_24h", "no_history"]
MODEL_INPUTS = FEATURES + ["rule_hits", "anomaly_score"]
LABELS = {
    "amount_vs_history": "Amount vs. history", "new_device": "New device",
    "new_location": "Login location", "account_age_days": "Account age",
    "night_time": "Time of day", "txns_last_24h": "Activity in last 24h",
    "no_history": "No history", "rule_hits": "Rule checks", "anomaly_score": "Unusual pattern",
}


def build_features(raw):
    """Transaction-level features from each customer's own earlier activity."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    df = raw.copy()
    if "transaction_id" not in df.columns:
        df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["account_open_date"] = pd.to_datetime(df["account_open_date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="raise")
    df = df.sort_values(["customer_id", "timestamp"], kind="stable", ignore_index=True)

    g = df.groupby("customer_id", sort=False)
    first = g.cumcount() == 0
    prior_median = g["amount"].transform(lambda s: s.shift().expanding().median())
    df["amount_vs_history"] = (df["amount"] / prior_median).fillna(1.0)
    df["new_device"] = ((df.groupby(["customer_id", "device_id"]).cumcount() == 0) & ~first).astype(int)
    df["new_location"] = ((df.groupby(["customer_id", "city"]).cumcount() == 0) & ~first).astype(int)
    df["account_age_days"] = (df["timestamp"] - df["account_open_date"]).dt.days.clip(lower=0)
    df["night_time"] = (df["timestamp"].dt.hour < 5).astype(int)
    df["txns_last_24h"] = (
        df.set_index("timestamp").groupby("customer_id", sort=False)["amount"]
        .rolling("24h").count().to_numpy() - 1
    ).astype(int)
    df["no_history"] = first.astype(int)
    return df.sort_values("timestamp", kind="stable", ignore_index=True)


def rule_hits(f):
    """Layer 1, statistical checks: how many classic red flags a transaction trips (0 to 5)."""
    return ((f["amount_vs_history"] >= 5).astype(int) + f["new_device"] + f["new_location"]
            + (f["account_age_days"] < 90).astype(int) + f["night_time"])


class FraudEngine:
    def __init__(self, seed=42):
        self.iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
        self.clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="aucpr", random_state=seed)
        self.explainer = None

    def _inputs(self, f):
        X = f[FEATURES].astype(float).copy()
        X["rule_hits"] = rule_hits(f)
        X["anomaly_score"] = -self.iso.decision_function(X[FEATURES])  # higher = more unusual
        return X

    def fit(self, f):
        if "is_fraud" not in f.columns:
            raise ValueError("Training needs an is_fraud column.")
        y = f["is_fraud"].astype(int)
        if y.nunique() < 2:
            raise ValueError("Training needs both fraud and non-fraud examples.")
        self.iso.fit(f.loc[y == 0, FEATURES].astype(float))  # learn what normal looks like
        X = self._inputs(f)
        self.clf.set_params(scale_pos_weight=(y == 0).sum() / (y == 1).sum())
        self.clf.fit(X, y)
        self.explainer = shap.TreeExplainer(self.clf)
        return self

    def score(self, f):
        """Fraud probability for each transaction, plus the model inputs used."""
        X = self._inputs(f)
        return self.clf.predict_proba(X)[:, 1], X

    def explain(self, X):
        """SHAP values in log-odds, one row per transaction, one column per model input."""
        return pd.DataFrame(self.explainer.shap_values(X), columns=X.columns, index=X.index)


def reason(feature, value, shap_value, city=None):
    """One plain-English line for a risk driver."""
    up = shap_value > 0
    v = value
    text = {
        "amount_vs_history": f"Amount is {v:.1f}x this customer's usual spend",
        "new_device": "First time this device has been used on the account" if v else "Device used here before",
        "new_location": (f"First transaction from {city}" if city else "First transaction from this city") if v
        else "From a city this customer uses",
        "account_age_days": f"Account is {int(v):,} days old",
        "night_time": "Made between midnight and 5 am" if v else "Made during normal hours",
        "txns_last_24h": f"{int(v)} other transactions in the previous 24 hours",
        "no_history": "No earlier transactions to compare against" if v else "Has transaction history",
        "rule_hits": f"Trips {int(v)} of 5 rule checks",
        "anomaly_score": "Overall pattern looks unusual" if up else "Overall pattern looks normal",
    }[feature]
    return f"{text} ({'raises' if up else 'lowers'} risk)"


def drivers(X_row, shap_row, city=None, top=4):
    """Top business-level risk drivers for one transaction, largest effect first.

    The rule and anomaly layers are left out here; they summarise these same signals.
    """
    order = shap_row[FEATURES].abs().sort_values(ascending=False).index[:top]
    return [{"feature": c, "label": LABELS[c], "shap": float(shap_row[c]),
             "reason": reason(c, X_row[c], shap_row[c], city)} for c in order]


def exposure(f, flagged, amount_col="amount"):
    """Share of fraud money stopped if the flagged transactions are blocked."""
    fraud = f["is_fraud"].astype(bool)
    total = f.loc[fraud, amount_col].sum()
    return float(f.loc[fraud & flagged, amount_col].sum() / total) if total else float("nan")


def flag_top(scores, budget):
    """Flag the riskiest `budget` share of transactions (an analyst's daily review capacity)."""
    k = max(1, int(round(len(scores) * budget)))
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[np.argsort(-np.asarray(scores), kind="stable")[:k]] = True
    return flagged


def evaluate(f, scores, budget=0.02):
    """Hold-out metrics for the model and for the rules-only baseline at the same alert budget."""
    y = f["is_fraud"].astype(int).to_numpy()
    model = flag_top(scores, budget)
    # rules-only baseline: rank by rule hits, break ties by amount vs history
    rules = flag_top(rule_hits(f) + f["amount_vs_history"].rank(pct=True).to_numpy() / 2, budget)
    out = {}
    for name, flg in (("model", model), ("rules", rules)):
        out[name] = {
            "alerts": int(flg.sum()),
            "precision": float(y[flg].mean()),
            "recall": float(y[flg].sum() / y.sum()),
            "exposure_caught": exposure(f, flg),
        }
    out["model"]["roc_auc"] = float(roc_auc_score(y, scores))
    out["model"]["pr_auc"] = float(average_precision_score(y, scores))
    return out


def time_split(f, test_share=0.3):
    """Train on the earlier part, test on the later part, as it would run in production."""
    cut = f["timestamp"].quantile(1 - test_share)
    return f[f["timestamp"] < cut], f[f["timestamp"] >= cut]


if __name__ == "__main__":
    from data_gen import generate_data

    # tiny feature check
    t = pd.DataFrame({
        "customer_id": ["a", "a", "a", "b"], "amount": [100, 100, 900, 50],
        "timestamp": ["2026-01-01 10:00", "2026-01-02 10:00", "2026-01-02 23:00", "2026-01-01 03:00"],
        "device_id": ["d1", "d1", "d2", "e1"], "city": ["X", "X", "Y", "Z"], "account_open_date": "2025-12-01",
    })
    tf = build_features(t).set_index(["customer_id", "amount"])
    a3 = tf.loc[("a", 900)]
    assert a3.amount_vs_history == 9 and a3.new_device == 1 and a3.new_location == 1 and a3.txns_last_24h == 1
    assert tf.loc[("b", 50)].night_time == 1 and tf.loc[("b", 50)].no_history == 1
    assert tf.loc[("a", 100)].new_device.tolist() == [0, 0]

    f = build_features(generate_data())
    train, test = time_split(f)
    eng = FraudEngine().fit(train)
    s, X = eng.score(test)
    r = evaluate(test, s)
    print(r)
    assert r["model"]["exposure_caught"] > r["rules"]["exposure_caught"]
    sv = eng.explain(X.iloc[[int(np.argmax(s))]])
    for d in drivers(X.iloc[int(np.argmax(s))], sv.iloc[0]):
        print(" ", d["reason"])
    print("engine self-check ok")
