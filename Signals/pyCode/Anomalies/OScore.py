import pandas as pd
import numpy as np
from pathlib import Path

# A similar construction of the OScore is provided by OAP called OScore.py under pyCode/Predictors. However, that code uses some different variables than what Stambaugh used. 
# The code is essentially the same as that provided by OAP, but modified to use the correct variables.

# safe math operations
def safe_div(a, b):
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)

def safe_log(x):
    return np.where((x <= 0) | pd.isna(x), np.nan, np.log(x))

# DATA LOAD
cols = ["permno","time_avail_m","datadate","at","dlc","dltt","act","lct","lt","ni","pi"]
df = pd.read_parquet("../pyData/Intermediate/m_aCompustat.parquet", columns=cols)

df["time_avail_m"] = pd.to_datetime(df["time_avail_m"])
df = df.drop_duplicates(subset=["permno","time_avail_m"]).sort_values(["permno","time_avail_m"])

# Calculate 12-month lag of NI using calendar-based approach
df["time_avail_m_lag12"] = df["time_avail_m"] - pd.DateOffset(months=12)

lag = df[["permno","time_avail_m","ni"]].copy()
lag = lag.rename(columns={"time_avail_m":"time_avail_m_lag12", "ni":"ni_lag12"})

df = df.merge(lag, on=["permno","time_avail_m_lag12"], how="left")
df = df.drop(columns=["time_avail_m_lag12"])


# SIZE = log(AT)
df["SIZE"] = safe_log(df["at"])

# TLTA = (DLC + DLTT) / AT   (book value of debt / assets)
df["TLTA"] = safe_div(df["dlc"].fillna(0) + df["dltt"].fillna(0), df["at"])

# WCTA = (ACT - LCT) / AT
df["WCTA"] = safe_div(df["act"] - df["lct"], df["at"])

# CLCA = LCT / ACT
df["CLCA"] = safe_div(df["lct"], df["act"])

# OENEG = 1 if LT > AT else 0
df["OENEG"] = (df["lt"] > df["at"]).astype(int)

# NITA = NI / AT
df["NITA"] = safe_div(df["ni"], df["at"])

# FUTL = PI / LT   (funds provided by operations / total liabilities)
df["FUTL"] = safe_div(df["pi"], df["lt"])

# INTWO = 1 if NI negative for last 2 years
df["INTWO"] = ((df["ni"] < 0) & (df["ni_lag12"] < 0)).astype(int)

# CHIN = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)
den = np.abs(df["ni"]) + np.abs(df["ni_lag12"])
df["CHIN"] = np.where((den == 0) | pd.isna(den), np.nan, (df["ni"] - df["ni_lag12"]) / den)

# ---------- Ohlson O-score ----------
df["OScore"] = (
    -1.32
    - 0.407 * df["SIZE"]
    + 6.03  * df["TLTA"]
    - 1.43  * df["WCTA"]
    + 0.076 * df["CLCA"]
    - 1.72  * df["OENEG"]
    - 2.37  * df["NITA"]
    - 1.83  * df["FUTL"]
    + 0.285 * df["INTWO"]
    - 0.521 * df["CHIN"]
)

# ---------- output ----------
out = df[["permno","time_avail_m","OScore"]].copy()
out = out.dropna(subset=["OScore"])

out["yyyymm"] = out["time_avail_m"].dt.year * 100 + out["time_avail_m"].dt.month
out = out[["permno","yyyymm","OScore"]].copy()
out["permno"] = out["permno"].astype(int)
out["yyyymm"] = out["yyyymm"].astype(int)

Path("../pyData/Predictors").mkdir(parents=True, exist_ok=True)
out.to_csv("../pyData/Predictors/OScore.csv", index=False)

print(f"OScore.csv saved with {len(out):,} rows")
