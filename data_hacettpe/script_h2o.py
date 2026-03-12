"""
BOYA FABRIKASI - HYBRID: H2O (Cmd 19/20/21) + LightGBM (Cmd 22)
================================================================
H2O, Cmd 22'nin 1.6M satirini 5-fold CV ile isleyemiyor.
Cozum: Cmd 19/20/21 icin H2O StackedEnsemble, Cmd 22 icin LightGBM.
"""

import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import lightgbm as lgb
import os, gc, warnings
warnings.filterwarnings("ignore")

DATA_DIR = "aiclubdatathon-26"

# ============================================================
# 1. H2O BASLAT + VERI YUKLE
# ============================================================
print("=" * 60)
print("HYBRID: H2O + LightGBM")
print("=" * 60)
print("\n[1] H2O baslatiliyor...")
h2o.init(nthreads=-1, max_mem_size="12G")

print("\n[2] Veri yukleniyor...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

for df in [train, test]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["starttime"] = pd.to_datetime(df["starttime"], format="mixed", utc=True)
    df["endtime"]   = pd.to_datetime(df["endtime"],   format="mixed", utc=True)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)
    df["process_id"] = (df["machineid"].astype(str) + "_" +
                        df["batchkey"].astype(str) + "_" +
                        df["commandno"].astype(str))

train = train.sort_values(["process_id","timestamp"]).reset_index(drop=True)
test  = test.sort_values(["process_id","timestamp"]).reset_index(drop=True)

# Batch istatistikleri
tnz = train[train["bk_level"] > 0].copy()
mc_stats = (tnz.groupby(["machineid","commandno"])["bk_level"]
              .agg(mc_mean="mean", mc_median="median").reset_index())
batch_stats = (tnz.groupby(["machineid","batchkey","commandno"])["bk_level"]
                  .agg(batch_mean="mean", batch_first="first", batch_max="max").reset_index())
del tnz; gc.collect()

train = train[train["bk_level"] != 0].reset_index(drop=True)
print(f"  Train (bk>0): {len(train):,} | Test: {len(test):,}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[3] Feature engineering...")
VALVE_COLS = ["fast_dosage_valve","slow_dosage_valve","kk_dosage_valve",
              "bk_dosage_valve","kk_bk_common_discharge","bk_irtibat_valve","kk_irtibat_valve"]

def add_features(df):
    df = df.copy()
    dur = (df["endtime"] - df["starttime"]).dt.total_seconds().clip(lower=1)
    ela = (df["timestamp"] - df["starttime"]).dt.total_seconds()
    df["progress"]  = (ela / dur).clip(0, 1).astype("float32")
    df["elapsed"]   = ela.astype("float32")
    df["remaining"] = (dur - ela).astype("float32")
    df["proc_dur"]  = dur.astype("float32")
    for v in VALVE_COLS:
        df[f"cum_{v}"] = df.groupby("process_id")[v].cumsum().astype("float32")
    df["n_valves"]   = df[VALVE_COLS].sum(axis=1).astype("int8")
    df["cum_valves"] = df.groupby("process_id")["n_valves"].cumsum().astype("float32")
    for v in VALVE_COLS:
        df[f"chg_{v}"] = df.groupby("process_id")[v].diff().fillna(0).astype("int8")
    for col in ["kk_level","ak_level"]:
        init = df.groupby("process_id")[col].transform("first")
        df[f"init_{col}"] = init.astype("float32")
        df[f"dlt_{col}"]  = (df[col] - init).astype("float32")
        df[f"d1_{col}"]   = df.groupby("process_id")[col].diff(1).fillna(0).astype("float32")
        df[f"d5_{col}"]   = df.groupby("process_id")[col].diff(5).fillna(0).astype("float32")
        df[f"r10_{col}"]  = (df.groupby("process_id")[col]
                              .transform(lambda x: x.rolling(10, min_periods=1).mean())
                              .astype("float32"))
    df["tgt_x_prog"]  = (df["bk_target_level"] * df["progress"]).astype("float32")
    df["tgt_x_rem"]   = (df["bk_target_level"] * (1 - df["progress"])).astype("float32")
    df["tgt_x_cum"]   = (df["bk_target_level"] * df["cum_valves"]).astype("float32")
    df["kk_gap"]      = (df["kk_target_level"] - df["kk_level"]).astype("float32")
    df["kk_ratio"]    = (df["kk_level"] / (df["ak_level"] + 1)).astype("float32")
    df["is_bk_drain"] = df["commandno"].isin([20, 22]).astype("int8")
    df["is_transfer"] = df["commandno"].isin([19, 20]).astype("int8")
    df["is_dosage"]   = df["commandno"].isin([21, 22]).astype("int8")
    df["proc_len"]    = df.groupby("process_id")["timestamp"].transform("count").astype("int32")
    df["step"]        = df.groupby("process_id").cumcount().astype("int32")
    df["step_ratio"]  = (df["step"] / df["proc_len"].clip(lower=1)).astype("float32")
    df["machine_cmd"] = (df["machineid"] * 100 + df["commandno"]).astype("int32")
    if "dosage_curve_type" in df.columns:
        df["curve_code"] = df["dosage_curve_type"].fillna("NONE").astype("category").cat.codes.astype("int8")
    else:
        df["curve_code"] = np.int8(-1)
    df["mixer_on"] = ((df["kk_mikser_robotu"]==1) | (df["bk_mikser_robotu"]==1)).astype("int8")
    return df

train = add_features(train)
test  = add_features(test)

train = train.merge(mc_stats, on=["machineid","commandno"], how="left")
train = train.merge(batch_stats, on=["machineid","batchkey","commandno"], how="left")
test  = test.merge(mc_stats, on=["machineid","commandno"], how="left")
test  = test.merge(batch_stats, on=["machineid","batchkey","commandno"], how="left")

stat_cols = ["mc_mean","mc_median","batch_mean","batch_first","batch_max"]
for col in stat_cols:
    if col in train.columns:
        fv = train[col].median()
        train[col] = train[col].fillna(fv).astype("float32")
        test[col]  = test[col].fillna(fv).astype("float32")

EXCLUDE = {"timestamp","starttime","endtime","process_id","batchkey",
           "row_id","Id","bk_level","dosage_curve_type"}
FEATURES = [c for c in train.columns if c not in EXCLUDE]
TARGET = "bk_level"
print(f"  Toplam feature: {len(FEATURES)}")

# ============================================================
# 3. H2O AutoML: Cmd 19, 20, 21 (kucuk veri)
# ============================================================
print("\n[4] H2O AutoML: Cmd 19, 20, 21...")
preds = np.zeros(len(test), dtype="float32")

for cmd in [19, 20, 21]:
    mask_tr = train["commandno"] == cmd
    mask_te = test["commandno"] == cmd
    n_tr, n_te = mask_tr.sum(), mask_te.sum()
    if n_te == 0:
        continue
    
    print(f"\n  [Cmd {cmd}] Train={n_tr:,} | Test={n_te:,}")
    
    tr_h2o = h2o.H2OFrame(train.loc[mask_tr, FEATURES + [TARGET]])
    te_h2o = h2o.H2OFrame(test.loc[mask_te, FEATURES])
    
    for col in ["machineid","commandno","prgno","stepno","command_repetition","curve_code"]:
        if col in tr_h2o.columns:
            try:
                tr_h2o[col] = tr_h2o[col].asfactor()
                te_h2o[col] = te_h2o[col].asfactor()
            except: pass
    
    aml = H2OAutoML(
        max_runtime_secs=300, max_models=20, seed=42,
        sort_metric="MAE", nfolds=5,
        keep_cross_validation_predictions=True,
        include_algos=["GBM","XGBoost","DRF","StackedEnsemble"],
    )
    aml.train(x=FEATURES, y=TARGET, training_frame=tr_h2o)
    
    best = aml.leader
    print(f"    Best: {best.model_id}")
    print(aml.leaderboard.head(rows=3))
    
    p = best.predict(te_h2o).as_data_frame()["predict"].values
    preds[mask_te] = p.astype("float32")
    
    h2o.remove_all()
    gc.collect()

h2o.shutdown(prompt=False)
print("\n  H2O kapatildi.")

# ============================================================
# 4. LightGBM: Cmd 22 (1.6M satir — H2O isleyemiyor)
# ============================================================
print("\n[5] LightGBM: Cmd 22...")
mask_tr22 = train["commandno"] == 22
mask_te22 = test["commandno"] == 22
print(f"  Train={mask_tr22.sum():,} | Test={mask_te22.sum():,}")

model_22 = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.03, max_depth=12, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.05, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1,
)
model_22.fit(train.loc[mask_tr22, FEATURES], train.loc[mask_tr22, "bk_level"])

mae22 = np.mean(np.abs(
    train.loc[mask_tr22, "bk_level"] - model_22.predict(train.loc[mask_tr22, FEATURES])
))
print(f"  Cmd 22 LGB Train MAE: {mae22:.4f}")

preds[mask_te22] = model_22.predict(test.loc[mask_te22, FEATURES])
preds = np.clip(preds, 0.0, 100.0)

# ============================================================
# 5. SUBMISSION
# ============================================================
print("\n[6] Submission...")
submission = pd.DataFrame({"Id": test["row_id"], "Predicted": preds})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv("submission_hybrid_h2o.csv", index=False)

print(f"\n{'='*60}")
print("TAMAMLANDI: submission_hybrid_h2o.csv")
print(f"  Shape:    {submission.shape}")
print(f"  Aralik:   [{preds.min():.2f}, {preds.max():.2f}]")
print(f"  Ortalama: {preds.mean():.2f}")
print(f"{'='*60}")
