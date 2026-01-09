import pandas as pd
import numpy as np

# 1) Month spine (month t = beginning of month)
smt = pd.read_parquet(
    "../pyData/Intermediate/SignalMasterTable.parquet",
    columns=["permno", "gvkey", "time_avail_m"]
).dropna(subset=["gvkey"]).copy()

smt["time_avail_m"] = pd.to_datetime(smt["time_avail_m"])
smt = smt.sort_values(["gvkey", "time_avail_m"])

# 2) Monthly expanded Quarterly data with RDQ
q = pd.read_parquet(
    "../pyData/Intermediate/CompustatQuarterly.parquet",
    columns=["gvkey", "datadateq", "rdq", "ibq", "atq"]
).copy()

q["datadateq"] = pd.to_datetime(q["datadateq"], errors="coerce")
q["rdq"]       = pd.to_datetime(q["rdq"], errors="coerce")

q["rdq"] = q["rdq"].fillna(q["datadateq"])

q = q.dropna(subset=["gvkey", "datadateq"])

q = (q.sort_values(["gvkey", "datadateq", "rdq"])
       .drop_duplicates(subset=["gvkey", "datadateq"], keep="last"))

print()

# 3) Prior-quarter assets (ATQ from immediately prior quarter)
q = q.sort_values(["gvkey", "datadateq"])
q["atq_prev"] = q.groupby("gvkey")["atq"].shift(1)
q.loc[q["atq_prev"] <= 0, "atq_prev"] = np.nan

# 4) For each month t, pick latest quarter with RDQ < month_start(t)
#    -> enforce strictness by allow_exact_matches=False

smt["gvkey"] = pd.to_numeric(smt["gvkey"], errors="coerce").astype("Int64")
q["gvkey"]   = pd.to_numeric(q["gvkey"], errors="coerce").astype("Int64")

smt["month_end"] = smt["time_avail_m"] #+ pd.offsets.MonthEnd(0)
smt = smt.sort_values(["month_end", "gvkey"]).reset_index(drop=True)
q2  = q.sort_values(["rdq", "gvkey"]).reset_index(drop=True)


roa = pd.merge_asof(
    smt,
    q2[["gvkey","rdq","ibq",'atq',"atq_prev","datadateq"]],
    left_on="month_end",
    right_on="rdq",
    by="gvkey",
    direction="backward",
    allow_exact_matches=False
)
# 5) ROA_t = IBQ_selected / ATQ_previous-quarter
roa["ROA"] = roa["ibq"] / roa["atq_prev"]


# Convert time_avail_m to yyyymm format
roa["yyyymm"] = (
    roa["time_avail_m"].dt.year * 100
    + roa["time_avail_m"].dt.month
)

# Convert to integers where appropriate
roa["permno"] = roa["permno"].astype(int)
roa["yyyymm"] = roa["yyyymm"].astype(int)

roa = roa[["permno","yyyymm","ROA"]].copy()

roa.to_csv("../pyData/Predictors/ROA.csv", index=False)
