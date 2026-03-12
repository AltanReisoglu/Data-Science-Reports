"""
BOYA FABRIKASI - SCRIPT2: PHYSICS-AWARE DIRECT PREDICTOR
=========================================================
Analiz bulgulari:
- cmd 22: bk_level ~ bk_target_level (PID hedef takibi)
- cmd 20: bk_level dozal olarak dusuyor, baslangic ~52%
- Ayni batch hem train hem test'te olabilir (temporal proximity)
- Lag KULLANMA: test'te tahmin hatasi birikir ve puan duser

Strateji:
1. Command tipi bazında ayrı model (19/21 ve 20/22)
2. Fiziksel integral features: CUMSUM(vana) = akan sivi miktari
3. bk_target_level x process_progress etkilesimi
4. Lag YOK - sifir sizinti, vektörize tahmin
5. Train'den process ilk degerlerini istatistiksel ozet olarak ekle
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import gc
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "aiclubdatathon-26"

# ========================================================
# 1. VERI YUKLEME
# ========================================================
print("Veri yukleniyor...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

for df in [train, test]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df["starttime"] = pd.to_datetime(df["starttime"], format="mixed", utc=True)
    df["endtime"]   = pd.to_datetime(df["endtime"],   format="mixed", utc=True)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(int)
    df["process_id"] = (
        df["machineid"].astype(str) + "_" +
        df["batchkey"].astype(str)  + "_" +
        df["commandno"].astype(str)
    )

train = train.sort_values(["process_id", "timestamp"]).reset_index(drop=True)
test  = test.sort_values(["process_id", "timestamp"]).reset_index(drop=True)

print(f"  Train: {len(train):,} | Test: {len(test):,}")

# bk_level=0 satirlar degerlendirme disi (etiketsiz anlar)
train_full = train.copy()  # istatistik hesabi icin tam kopya
train = train[train["bk_level"] != 0].reset_index(drop=True)
print(f"  Train (bk>0): {len(train):,}")

# ========================================================
# 2. MAKINE+BATCH BAZLI ISTATISTIKLER (TRAIN'DEN TEST'E BILGI)
# Ayni batch hem train hem test'te olabilir.
# Bu durumda training sirasinda o batch'in bk caratteristiklerini ogreniriz.
# ========================================================
print("\nBatch istatistikleri hesaplaniyor...")

train_nz = train_full[train_full["bk_level"] > 0].copy()
train_nz["machine_batch"] = (
    train_nz["machineid"].astype(str) + "_" + train_nz["batchkey"].astype(str)
)
test["machine_batch"] = (
    test["machineid"].astype(str) + "_" + test["batchkey"].astype(str)
)

# Her (makine, batch, komut) kombinasyonu icin istatistikler
batch_stats = (train_nz
    .groupby(["machineid", "batchkey", "commandno"])["bk_level"]
    .agg(
        batch_bk_mean="mean",
        batch_bk_std="std",
        batch_bk_first="first",
        batch_bk_last="last",
    )
    .reset_index()
)

# Her makine+komut icin genel istatistikler (batch bilmeden de kullan)
machine_cmd_stats = (train_nz
    .groupby(["machineid", "commandno"])["bk_level"]
    .agg(
        mc_bk_mean="mean",
        mc_bk_median="median",
        mc_bk_first="first",
    )
    .reset_index()
)

# ========================================================
# 3. OZELLIK MUHENDISLIGI
# ========================================================
print("\nFeature engineering...")

VALVE_COLS = [
    "fast_dosage_valve",     # Transfer'de ana acik vana
    "slow_dosage_valve",     # Dozajda ana vana
    "kk_dosage_valve",       # KK tarafi
    "bk_dosage_valve",       # BK tarafi
    "kk_bk_common_discharge",# Ortak bosaltim
    "bk_irtibat_valve",      # BK irtibat (korel 0.24)
    "kk_irtibat_valve",      # KK irtibat
]

def build_features(df):
    df = df.copy()

    # Zaman ozellikleri
    dur = (df["endtime"] - df["starttime"]).dt.total_seconds().clip(lower=1)
    ela = (df["timestamp"] - df["starttime"]).dt.total_seconds()
    df["progress"]    = (ela / dur).clip(0, 1).astype("float32")
    df["elapsed_s"]   = ela.astype("float32")
    df["remain_s"]    = (dur - ela).astype("float32")
    df["proc_dur"]    = dur.astype("float32")

    # Fiziksel integral: kumulatif acik sure = akan sivi miktarinin vekili
    for v in VALVE_COLS:
        df[f"cum_{v}"] = df.groupby("process_id")[v].cumsum().astype("float32")

    df["n_valves"]   = df[VALVE_COLS].sum(axis=1).astype("int8")
    df["cum_valves"] = df.groupby("process_id")["n_valves"].cumsum().astype("float32")

    # Vana acilma/kapanma ani (0, +1, -1)
    for v in VALVE_COLS:
        df[f"chg_{v}"] = df.groupby("process_id")[v].diff().fillna(0).astype("int8")

    # Sensor degisimler (bk_level'a dokunma!)
    for col in ["kk_level", "ak_level"]:
        init = df.groupby("process_id")[col].transform("first")
        df[f"init_{col}"]  = init.astype("float32")
        df[f"dlt_{col}"]   = (df[col] - init).astype("float32")
        df[f"d1_{col}"]    = df.groupby("process_id")[col].diff(1).fillna(0).astype("float32")
        df[f"d5_{col}"]    = df.groupby("process_id")[col].diff(5).fillna(0).astype("float32")
        df[f"roll10_{col}"] = (
            df.groupby("process_id")[col]
              .transform(lambda x: x.rolling(10, min_periods=1).mean())
              .astype("float32")
        )

    # Hedef etkilesimleri (KEY: cmd 22 -> bk ~ bk_target)
    df["tgt_x_prog"]  = (df["bk_target_level"] * df["progress"]).astype("float32")
    df["tgt_x_rem"]   = (df["bk_target_level"] * (1 - df["progress"])).astype("float32")
    df["tgt_x_valve"] = (df["bk_target_level"] * df["cum_valves"]).astype("float32")
    df["kk_gap"]      = (df["kk_target_level"] - df["kk_level"]).astype("float32")
    df["kk_ratio"]    = (df["kk_level"] / (df["ak_level"] + 1)).astype("float32")

    # Komut tipleri
    df["is_bk_drain"] = df["commandno"].isin([20, 22]).astype("int8")
    df["is_transfer"] = df["commandno"].isin([19, 20]).astype("int8")
    df["is_dosage"]   = df["commandno"].isin([21, 22]).astype("int8")

    # Proses yapisi
    df["proc_len"]   = df.groupby("process_id")["timestamp"].transform("count").astype("int32")
    df["step"]       = df.groupby("process_id").cumcount().astype("int32")
    df["step_ratio"] = (df["step"] / df["proc_len"].clip(lower=1)).astype("float32")

    # Makine+komut kombinasyonu
    df["machine_cmd"] = (df["machineid"] * 100 + df["commandno"]).astype("int32")

    # Dozaj egri tipi
    if "dosage_curve_type" in df.columns:
        df["curve_code"] = (
            df["dosage_curve_type"].fillna("NONE")
              .astype("category").cat.codes.astype("int8")
        )
    else:
        df["curve_code"] = -1

    df["mixer_on"] = (
        (df["kk_mikser_robotu"] == 1) | (df["bk_mikser_robotu"] == 1)
    ).astype("int8")

    return df

train = build_features(train)
test  = build_features(test)

# Batch ve makine istatistiklerini join et
for df in [train, test]:
    df = df.merge(batch_stats, on=["machineid","batchkey","commandno"], how="left")
    df = df.merge(machine_cmd_stats, on=["machineid","commandno"], how="left")

train = train.merge(batch_stats, on=["machineid","batchkey","commandno"], how="left")
test  = test.merge(batch_stats, on=["machineid","batchkey","commandno"], how="left")
train = train.merge(machine_cmd_stats, on=["machineid","commandno"], how="left")
test  = test.merge(machine_cmd_stats, on=["machineid","commandno"], how="left")

# Feature listesi
EXCLUDE = {
    "timestamp","starttime","endtime",
    "process_id","batchkey","machine_batch",
    "row_id","Id",
    "bk_level",
    "dosage_curve_type",
}
FEATURES = [c for c in train.columns if c not in EXCLUDE]
print(f"  Toplam feature: {len(FEATURES)}")

# ========================================================
# 4. MODEL EGITIMI
# Organizator: "Transfer ve dozaj cok farkli davranir"
# --> Her komut grubu icin ayri model
# ========================================================
print("\nModel egitimi...")

lgb_params = dict(
    n_estimators     = 3000,
    learning_rate    = 0.03,
    max_depth        = 12,
    num_leaves       = 127,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_samples= 20,
    reg_alpha        = 0.05,
    reg_lambda       = 1.0,
    random_state     = 42,
    verbose          = -1,
    n_jobs           = -1,
)

# Transfer: cmd 19, 20 (dogrusal, vana surekli acik)
mask_tr_tr = train["commandno"].isin([19, 20])
mask_tr_te = test["commandno"].isin([19, 20])
print(f"  [Transfer] train={mask_tr_tr.sum():,} | test={mask_tr_te.sum():,}")
model_tr = lgb.LGBMRegressor(**lgb_params)
model_tr.fit(train.loc[mask_tr_tr, FEATURES], train.loc[mask_tr_tr, "bk_level"])
tr_mae = np.mean(np.abs(
    train.loc[mask_tr_tr, "bk_level"] - model_tr.predict(train.loc[mask_tr_tr, FEATURES])
))
print(f"    Train MAE: {tr_mae:.4f}")

# Dozaj: cmd 21, 22 (PID, salinimli)
mask_dz_tr = train["commandno"].isin([21, 22])
mask_dz_te = test["commandno"].isin([21, 22])
print(f"  [Dozaj]    train={mask_dz_tr.sum():,} | test={mask_dz_te.sum():,}")
model_dz = lgb.LGBMRegressor(**lgb_params)
model_dz.fit(train.loc[mask_dz_tr, FEATURES], train.loc[mask_dz_tr, "bk_level"])
dz_mae = np.mean(np.abs(
    train.loc[mask_dz_tr, "bk_level"] - model_dz.predict(train.loc[mask_dz_tr, FEATURES])
))
print(f"    Train MAE: {dz_mae:.4f}")

del train, train_full, train_nz, batch_stats, machine_cmd_stats
gc.collect()

# ========================================================
# 5. TEST TAHMIN
# Lag yok --> vektorize predict, proses sinirlari kendiligindon saglanmis
# ========================================================
print("\nTest tahmini...")
preds = np.zeros(len(test), dtype="float32")
preds[mask_tr_te] = model_tr.predict(test.loc[mask_tr_te, FEATURES])
preds[mask_dz_te] = model_dz.predict(test.loc[mask_dz_te, FEATURES])
preds = np.clip(preds, 0.0, 100.0)

# ========================================================
# 6. SUBMISSION
# ========================================================
submission = pd.DataFrame({
    "Id":        test["row_id"],
    "Predicted": preds,
})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv("submission2.csv", index=False)

print(f"\n{'='*50}")
print("TAMAMLANDI: submission2.csv")
print(f"  Shape:    {submission.shape}")
print(f"  Aralik:   [{preds.min():.2f}, {preds.max():.2f}]")
print(f"  Ortalama: {preds.mean():.2f}")
print(f"{'='*50}")

# Feature onemleri
print("\nTransfer Modeli En Onemli 10:")
imp = pd.Series(model_tr.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.head(10).to_string())

print("\nDozaj Modeli En Onemli 10:")
imp2 = pd.Series(model_dz.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp2.head(10).to_string())
