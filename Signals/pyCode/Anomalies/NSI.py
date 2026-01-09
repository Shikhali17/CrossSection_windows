# ABOUTME: Following Fama and French (2008), net issuance as the annual log change


import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

df = pd.read_parquet(
    r"..\pyData\Intermediate\m_aCompustat.parquet",
    columns=["permno", "time_avail_m", "csho", "ajex", "datadate"]
)

# Description/Construction for this anomaly is identical to the AssetGrowth anomaly provided by OAP.
# Construction closely follows the OAP code AssetGrowth.py
df = df.sort_values(["permno", "time_avail_m"])
df["shares_adj"] = df["csho"] * df["ajex"]
df["l12_adjsh"] = df.groupby("permno")["shares_adj"].shift(12)
df["NSI"] = np.where(
    df["l12_adjsh"] == 0,
    np.nan,  # Division by zero = missing
    np.log(df["shares_adj"] / df["l12_adjsh"])  # pandas: missing/missing = NaN naturally
)

df["time_avail_m"] = pd.to_datetime(df["time_avail_m"])
df = df[["permno", "time_avail_m", "NSI"]].copy()

# Drop missing signal values
df = df.dropna(subset=["NSI"]).copy()

# Convert time_avail_m to yyyymm format
df["yyyymm"] = (
    df["time_avail_m"].dt.year * 100
    + df["time_avail_m"].dt.month
)

# Keep required columns
df = df[["permno", "yyyymm", "NSI"]].copy()
df["NSI"].replace([np.inf, -np.inf], np.nan, inplace=True)
# Convert to integers where appropriate
df["permno"] = df["permno"].astype(int)
df["yyyymm"] = df["yyyymm"].astype(int)

df.to_csv("../pyData/Predictors/NSI.csv", index=False)
