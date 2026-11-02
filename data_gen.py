"""Synthetic card transactions with labelled fraud, for demos and for training the default model.

Fraud here is account takeover: someone else uses the account, usually from a new device
and a new city, often for an amount far above the customer's normal spend, more often on
young accounts and at night. Legitimate customers also travel, change phones and make
the odd big purchase, so no single signal gives fraud away.
"""
import numpy as np
import pandas as pd

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata",
          "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Dubai", "Singapore", "London"]


def generate_data(n_customers=1500, days=120, fraud_rate=0.015, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-06-01")
    rows = []
    for c in range(n_customers):
        cid = f"C{c:05d}"
        opened = start - pd.Timedelta(days=int(rng.gamma(2.0, 400)) + 3)
        home = rng.choice(CITIES[:11])
        devices = [f"D{c:05d}-{k}" for k in range(rng.integers(1, 3))]
        typical = rng.lognormal(7.0, 0.8)  # rupees; median around 1,100
        n = rng.poisson(days * rng.uniform(0.15, 0.8)) + 2
        times = np.sort(start + pd.to_timedelta(rng.uniform(0, days * 86400, n), unit="s"))
        for t in map(pd.Timestamp, times):
            fraud = rng.random() < fraud_rate * (2.5 if (t - opened).days < 90 else 1.0)
            if fraud:
                amount = typical * rng.lognormal(1.6, 0.9)
                device = devices[0] if rng.random() < 0.2 else f"X{rng.integers(10**8)}"
                city = home if rng.random() < 0.3 else rng.choice(CITIES)
                if rng.random() < 0.5:  # many takeovers happen at night
                    t = t.normalize() + pd.Timedelta(hours=int(rng.integers(0, 5)), minutes=int(rng.integers(60)))
            else:
                amount = typical * rng.lognormal(0, 0.6) * (rng.lognormal(1.2, 0.5) if rng.random() < 0.03 else 1)
                if rng.random() < 0.02:  # new phone, kept from then on
                    devices.append(f"D{c:05d}-{len(devices)}")
                    device = devices[-1]
                else:
                    device = rng.choice(devices)
                city = rng.choice(CITIES) if rng.random() < 0.04 else home
            rows.append((cid, t, round(float(amount), 2), device, city, opened.date(), fraud))
    df = pd.DataFrame(rows, columns=["customer_id", "timestamp", "amount", "device_id", "city",
                                     "account_open_date", "is_fraud"])
    df = df.sort_values("timestamp", ignore_index=True)
    df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_data()
    df.to_csv("data/sample_transactions.csv", index=False)
    print(f"{len(df):,} transactions, {df.is_fraud.sum():,} fraud, written to data/sample_transactions.csv")
