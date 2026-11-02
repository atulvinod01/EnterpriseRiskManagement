import altair as alt
import pandas as pd
import streamlit as st

import model_engine as me
from data_gen import generate_data

st.set_page_config(page_title="Risk Engine", layout="wide")
st.title("Risk Engine")
st.caption("Fraud detection that explains itself: statistical checks, an anomaly detector and a "
           "gradient-boosted classifier, with SHAP turning every score into its risk drivers.")


@st.cache_data(show_spinner="Generating sample transactions…")
def sample_raw():
    return generate_data()


@st.cache_resource(show_spinner="Training on sample data…")
def sample_engine():
    f = me.build_features(sample_raw())
    return me.FraudEngine().fit(f)


@st.cache_resource(show_spinner="Training on your data…")
def train_engine(f):
    return me.FraudEngine().fit(f)


with st.sidebar:
    source = st.radio("Data", ["Sample data (synthetic)", "Upload CSV"])
    budget = st.slider("Review budget (% of transactions an analyst checks)", 0.5, 10.0, 2.0, 0.5) / 100
    st.download_button("Download a sample CSV", sample_raw().head(5000).to_csv(index=False),
                       "sample_transactions.csv", "text/csv")
    st.caption("CSV columns: " + ", ".join(me.REQUIRED) +
               ". Optional: transaction_id, is_fraud (0/1). With is_fraud the engine trains on "
               "your data and measures itself; without it, the model trained on the sample data scores yours.")

# ---- load, train, score ----
if source == "Upload CSV":
    up = st.file_uploader("Transactions CSV", type="csv")
    if not up:
        st.info("Upload a CSV to score it, or switch to the sample data.")
        st.stop()
    try:
        f = me.build_features(pd.read_csv(up))
    except Exception as e:
        st.error(f"Couldn't read that file: {e}")
        st.stop()
else:
    f = me.build_features(sample_raw())

labelled = "is_fraud" in f.columns and f["is_fraud"].astype(int).sum() >= 20 and f["is_fraud"].nunique() == 2
if source == "Sample data (synthetic)" or labelled:
    train, scored = me.time_split(f)
    if scored["is_fraud"].astype(int).sum() == 0:
        st.error("No fraud in the latest 30% of your data, so there's nothing to measure against.")
        st.stop()
    engine = train_engine(train)
    note = (f"Trained on {len(train):,} earlier transactions, tested on the latest {len(scored):,} "
            f"({scored['timestamp'].min():%d %b} to {scored['timestamp'].max():%d %b %Y}), which the model never saw.")
else:
    engine, scored = sample_engine(), f
    note = (f"No usable is_fraud labels, so {len(f):,} transactions were scored with the model trained on sample "
            "data. Accuracy figures need labels.")
    labelled = False
measured = source == "Sample data (synthetic)" or labelled

scored = scored.reset_index(drop=True)
probs, X = engine.score(scored)
scored["risk"] = probs
flagged = me.flag_top(probs, budget)
alerts = scored[flagged].sort_values("risk", ascending=False)
sv = engine.explain(X.loc[alerts.index])
reasons = {i: me.drivers(X.loc[i], sv.loc[i], scored.at[i, "city"]) for i in alerts.index}
level = pd.cut(alerts["risk"], [-1, 0.5, 0.8, 1.01], labels=["Low", "Medium", "High"])

st.caption(note)

# ---- headline ----
c = st.columns(4)
c[0].metric("Transactions scored", f"{len(scored):,}")
c[1].metric("Alerts to review", f"{len(alerts):,}", f"{budget:.1%} of volume", delta_color="off")
if measured:
    ev = me.evaluate(scored, probs, budget)
    m, r = ev["model"], ev["rules"]
    left_rules, left_model = 1 - r["exposure_caught"], 1 - m["exposure_caught"]
    c[2].metric("Fraud money stopped", f"{m['exposure_caught']:.0%}",
                f"{m['exposure_caught'] - r['exposure_caught']:+.0%} vs rules only")
    c[3].metric("Alerts that are fraud", f"{m['precision']:.0%}", f"{m['precision'] - r['precision']:+.0%} vs rules only")
    total = scored.loc[scored["is_fraud"].astype(bool), "amount"].sum()
    st.markdown(
        f"**In plain words:** checking the riskiest {budget:.1%} of transactions catches "
        f"**{m['recall']:.0%} of fraud cases** and **{m['exposure_caught']:.0%} of the {total:,.0f} at stake**. "
        f"{m['precision']:.0%} of alerts are real fraud, so analysts spend their time on real cases. "
        f"A rules-only check with the same budget leaves {left_rules:.1%} of fraud money exposed; "
        f"the engine leaves {left_model:.1%}"
        + (f", **{1 - left_model / left_rules:.0%} less exposure**." if left_rules > 0 else ".")
    )
    with st.expander("Model scorecard"):
        st.dataframe(pd.DataFrame(ev).T.rename(index={"model": "Risk engine", "rules": "Rules only"}),
                     width="stretch")
else:
    c[2].metric("High-risk alerts", f"{(level == 'High').sum():,}")
    c[3].metric("Money in alerts", f"{alerts['amount'].sum():,.0f}")

# ---- alert queue ----
st.subheader("Alert queue")
queue = pd.DataFrame({
    "Transaction": alerts["transaction_id"], "Customer": alerts["customer_id"],
    "Time": alerts["timestamp"].dt.strftime("%d %b %H:%M"), "Amount": alerts["amount"].round(0),
    "City": alerts["city"], "Risk": alerts["risk"].round(3), "Level": level.astype(str),
    "Main reason": [reasons[i][0]["reason"] for i in alerts.index],
})
if measured:
    queue["Actually fraud"] = alerts["is_fraud"].astype(bool)
st.dataframe(queue, hide_index=True, width="stretch", height=320)

log = queue.copy()
for k in range(3):
    log[f"Reason {k + 1}"] = [reasons[i][k]["reason"] if len(reasons[i]) > k else "" for i in alerts.index]
log["Rule checks tripped"] = X.loc[alerts.index, "rule_hits"].astype(int).to_numpy()
st.download_button("Download alert log for audit (CSV)", log.drop(columns="Main reason").to_csv(index=False),
                   "alert_log.csv", "text/csv")

# ---- one case ----
st.subheader("Why it fired")
pick = st.selectbox("Alert", alerts.index, format_func=lambda i: (
    f"{scored.at[i, 'transaction_id']} · {scored.at[i, 'amount']:,.0f} · risk {scored.at[i, 'risk']:.2f}"))
row, ds = scored.loc[pick], reasons[pick]
left, right = st.columns([3, 2])
with left:
    lvl = level.loc[pick]
    st.markdown(f"#### Transaction flagged · {lvl} risk")
    st.markdown(f"{row['amount']:,.0f} by {row['customer_id']} in {row['city']} on {row['timestamp']:%d %b %Y %H:%M}, "
                f"device {row['device_id']}.")
    for d in ds:
        st.markdown(f"- {d['reason']}")
    ru, an = int(X.at[pick, "rule_hits"]), float(X.at[pick, "anomaly_score"])
    st.caption(f"Layer 1, rule checks: {ru} of 5 tripped. Layer 2, anomaly detector: "
               f"{'unusual' if an > 0 else 'normal-looking'} ({an:+.2f}). Layer 3, classifier: risk {row['risk']:.2f}.")
with right:
    chart = pd.DataFrame({"Driver": [d["label"] for d in ds], "Effect": [d["shap"] for d in ds],
                          "Direction": ["Raises risk" if d["shap"] > 0 else "Lowers risk" for d in ds]})
    st.altair_chart(alt.Chart(chart).mark_bar().encode(
        x=alt.X("Effect", title="Push on risk (SHAP, log-odds)"),
        y=alt.Y("Driver", sort=None, title=None),
        color=alt.Color("Direction", scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                                     range=["#d6452f", "#2f8f5b"]), legend=None),
        tooltip=["Driver", alt.Tooltip("Effect", format=".2f"), "Direction"],
    ), width="stretch")

with st.expander("What drives risk across all alerts"):
    imp = sv[me.FEATURES].abs().mean().rename(index=me.LABELS).sort_values(ascending=False)
    st.bar_chart(imp, horizontal=True)
