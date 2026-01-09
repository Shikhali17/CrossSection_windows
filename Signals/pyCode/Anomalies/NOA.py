# ABOUTME: Net Operating Assets following Hirshleifer et al. 2004 Table 4
# ABOUTME: calculates NOA = (OA - OL) / lagged total assets where OA = at - che and OL = at - dltt - mib - dc - ceq.

"""
NOA.py

Usage:
    Run from [Repo-Root]/Signals/pyCode/
    python3 Predictors/NOA.py

Inputs:
    - m_aCompustat.parquet: Monthly Compustat data with columns [gvkey, permno, time_avail_m, at, che, dltt, mib, dc, ceq]

Outputs:
    - NOA.csv: CSV file with columns [permno, yyyymm, NOA]
"""

import pandas as pd
import numpy as np

# This code is nearly identical to the construction of NOA.py provided by OAP under pyCode/Predictors. However, the formula for the operating liabilities is slightly different:

# OAP operating liability: df["OL"] = df["at"] - df["dltt"] - df["mib"] - df["dc"] - df["ceq"]
# Stambaugh operating liability: df["OL"] = df["at"] - df["dlc"] - df["dltt"] - df["ceq"] - df["mib"] - df["pstk"]
# So it's missing dlc, debt in current liabilities, and pstk, preferred stock.

print("Starting NOA calculation...")

# DATA LOAD
m_aCompustat = pd.read_parquet(
    "../pyData/Intermediate/m_aCompustat.parquet",
    columns=[
        "gvkey",
        "permno",
        "time_avail_m",
        "at",
        "che",
        'dlc',
        "dltt",
        "mib",
        "ceq",
        'pstk',
    ],
)
df = m_aCompustat.copy()
print(f"Loaded m_aCompustat data: {len(df)} observations")

# SIGNAL CONSTRUCTION

# Remove duplicate permno-month observations
df = df.drop_duplicates(subset=["permno", "time_avail_m"], keep="first")
print(f"After deduplicating by permno time_avail_m: {len(df)} observations")

# Sort
df = df.sort_values(["permno", "time_avail_m"])

df["mib"] = df["mib"].fillna(0)
df["pstk"] = df["pstk"].fillna(0)

# Operating Assets: OA = AT - CHE
df["OA"] = df["at"] - df["che"]

# Operating Liabilities: OL = AT - DLC - DLTT - CEQ - MIB - PSTK
df["OL"] = df["at"] - df["dlc"] - df["dltt"] - df["ceq"] - df["mib"] - df["pstk"]

# Net Operating Assets (unscaled): NOA_num = OA - OL
df["NOA_num"] = df["OA"] - df["OL"]

# Lagged total assets: use 12-month lag in availability-month panel
df["at_lag12"] = df.groupby("permno")["at"].shift(12)

# Scale by lagged AT
df["NOA"] = np.where(
    (df["at_lag12"].isna()) | (df["at_lag12"] == 0),
    np.nan,
    df["NOA_num"] / df["at_lag12"]
)

df["time_avail_m"] = pd.to_datetime(df["time_avail_m"])

print(f"NOA calculated for {df['NOA'].notna().sum()} observations")

# SAVE
result = df[["permno", "time_avail_m", "NOA"]].copy()
result = result.dropna(subset=["NOA"])

# Convert time_avail_m to yyyymm format
result["yyyymm"] = (
    result["time_avail_m"].dt.year * 100 + result["time_avail_m"].dt.month
)
result = result[["permno", "yyyymm", "NOA"]].copy()

# Convert to integers
result["permno"] = result["permno"].astype(int)
result["yyyymm"] = result["yyyymm"].astype(int)

print(f"Final output: {len(result)} observations")
result.to_csv("../pyData/Predictors/NOA.csv", index=False)
print("NOA.csv saved successfully")

