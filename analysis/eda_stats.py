"""Per-session statistics for the starting-kit data."""
import glob, re, sys
import numpy as np, pandas as pd

DATA = sys.argv[1]
def phase_to_label(p):
    if p in ("Stroop","TMCT","Real opinion","Opposite opinion","Subtraction"): return "Stress"
    if p in ("Warm-up","Sprint","Cooldown") or "rpm" in p.lower() or re.match(r"^\d+\s*rpm",p): return "Activity"
    if p in ("Baseline","Rest"): return "Rest"
    return None

rows=[]; allphases={}
for f in sorted(glob.glob(f"{DATA}/*/*.parquet")):
    prot=f.split("/")[-2]; df=pd.read_parquet(f)
    idx=df.index; dt=np.diff(idx.values).astype("timedelta64[ns]").astype(float)/1e9
    lab=df.Phase.map(phase_to_label)
    for p,n in df.Phase.value_counts().items(): allphases.setdefault(prot,{}).setdefault(p,0); allphases[prot][p]+=n
    r=dict(prot=prot,sid=df.user_id.iloc[0],dur_min=(idx[-1]-idx[0]).total_seconds()/60,
           n=len(df),dt_med=np.median(dt),dt_max=dt.max(),n_gap_gt1s=int((dt>1).sum()),
           mono=bool(idx.is_monotonic_increasing),dup=int(idx.duplicated().sum()),
           start=str(idx[0]))
    for L in ["Rest","Stress","Activity"]: r[L+"_min"]=(lab==L).sum()/64/60
    r["unscored_min"]=lab.isna().sum()/64/60
    for c in ["BVP","EDA","HR","IBI_duration","Temperature"]: r[c+"_n"]=int(df[c].notna().sum())
    r["EDA_med"]=df.EDA.median(); r["TEMP_med"]=df.Temperature.median(); r["HR_med"]=df.HR.median()
    r["phase_seq"]=" > ".join(k for k,_ in __import__("itertools").groupby(df.Phase))
    rows.append(r)
S=pd.DataFrame(rows)
pd.set_option("display.width",250,"display.max_columns",50,"display.max_colwidth",400,"display.max_rows",200)
S.to_csv("session_stats.csv",index=False)
print(S.drop(columns=["phase_seq","start"]).round(2).to_string())
print("\nPHASES per protocol:"); [print(k,v) for k,v in allphases.items()]
print("\nPhase sequences:"); 
for _,r in S.iterrows(): print(r.prot,r.sid,"|",r.phase_seq)
print("\nTotal minutes per label:"); print(S[["Rest_min","Stress_min","Activity_min","unscored_min"]].sum().round(1))
print(S.groupby("prot")[["Rest_min","Stress_min","Activity_min","unscored_min","dur_min"]].sum().round(1))
