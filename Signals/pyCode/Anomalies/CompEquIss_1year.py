# ABOUTME: Composite equity issuance similar to Daniel and Titman 2006, Table 3
# ABOUTME: calculates 1 year growth rate of market value minus 1 year stock return.

"""
CompEquIss.py

Usage:
    Run from [Repo-Root]/Signals/pyCode/
    python3 Predictors/CompEquIss.py

Inputs:
    - SignalMasterTable.parquet: Monthly signal master table with columns [permno, time_avail_m, ret, mve_c]

Outputs:
    - CompEquIss.csv: CSV file with columns [permno, yyyymm, CompEquIss]
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Note, this code is the exact same as CompEquIss.py provided by OAP under pyCode/Predictors, but instead of computing 5 years of growth, it does it for 1 year. 
# Furthermore, it is lagged by 4 months, unlike CompEquIss.py. Explanation provided below.

# Add utils directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.stata_replication import stata_multi_lag
from utils.save_standardized import save_predictor


print("Starting CompEquIss predictor...")

# DATA LOAD
print("Loading SignalMasterTable...")
df = pd.read_parquet(
    "../pyData/Intermediate/SignalMasterTable.parquet",
    columns=["permno", "time_avail_m", "ret", "mve_c"],
)


print(f"Loaded {len(df):,} SignalMasterTable observations")

# SIGNAL CONSTRUCTION
print("Constructing CompEquIss signal...")


# Sort data by permno and time for lag operations
df = df.sort_values(["permno", "time_avail_m"])

# Create cumulative return index starting at 1 for each permno
df["tempIdx"] = df.groupby("permno")["ret"].transform(lambda x: (1 + x).cumprod())

# Create 12-month lags with calendar validation
print("Creating 60-month lags with calendar validation...")
df = stata_multi_lag(df, "permno", "time_avail_m", "tempIdx", [12])
df = stata_multi_lag(df, "permno", "time_avail_m", "mve_c", [12])

# Calculate buy-and-hold returns over 12 months
df["tempBH"] = (df["tempIdx"] - df["tempIdx_lag12"]) / df["tempIdx_lag12"]

# Calculate composite equity issuance as log growth in market value minus buy-and-hold returns
df["CompEquIss"] = np.log(df["mve_c"] / df["mve_c_lag12"]) - df["tempBH"]


print(
    f"Generated CompEquIss values for {df['CompEquIss'].notna().sum():,} observations"
)


df["time_avail_m"] = pd.to_datetime(df["time_avail_m"])

df_final = df[["permno", "time_avail_m", "CompEquIss"]].copy()
df_final = df.copy()
df_final["time_avail_m"] = pd.to_datetime(df_final["time_avail_m"])

# LAG BY 4 MONTHS (make CompEquIss available 4 months later). This is because CompEquIss is computed using market data. If not lagged, the variable will be available immediately alongside market data. 
# However, in their documentation, Stambaugh lagged CompEquIss by 4 months so that it is available 4 months later (4 months after the market data used to calcualte it) so that its release is coincidence with that of other accounting factors like NSI.
df_final["time_avail_m"] = (df_final["time_avail_m"].dt.to_period('M') + 4).dt.to_timestamp()

# SAVE
print("Saving predictor...")

save_predictor(df_final, "CompEquIss")

print("CompEquIss predictor completed successfully!")

