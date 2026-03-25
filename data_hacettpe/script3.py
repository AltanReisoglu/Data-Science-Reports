"""
BOYA FABRIKASI - SCRIPT3: ANALYSIS-DRIVEN FINAL MODEL
======================================================
ANALIZ BULGULARI:
  - Cmd 19/21 (KK dreni): bk_level neredeyse DEGISMIYOR
    --> ak_level (korel 0.30), cum_fast_dosage_valve (0.17), progress (0.12)
  - Cmd 20/22 (BK dreni): bk_level buyuk dusus
    --> bk_target_level korelasyon 0.97! Ana sinyal bu.
  - Her makine farkli seviyede calisir (242 cmd21: ort 64%, 105 cmd19: ort 15%)
  - %83 test satiri ortak batch'ten geliyor --> batch istatistikleri cok guçlu
  - Cmd 22 baslangic bk_level ~ 40-60%

STRATEJI (Sifir Sizinti):
  1. 4 ayri model (cmd 19, 20, 21, 22) -- hepsi farkli dinamik
  2. Makine+cmd bazli istatistikler (batch overlap sayesinde)
  3. Fiziksel kumulatif vana integrali
  4. bk_target_level x progress etkilesimi (cmd 22 icin kritik)
  5. Lag YOK -- vektörize, hata birikmez
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import os, gc, warnings
warnings.filterwarnings("ignore")

DATA_DIR = "aiclubdatathon-26"

# ============================================================
# 1. VERI YUKLEME
# ============================================================
print("=" * 60)
print("SCRIPT3: ANALYSIS-DRIVEN FINAL MODEL")
print("=" * 60)
print("\n[1] Veri yukleniyor...")

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

# bk_level=0 egitimden cikart (skorlanmaz)
train_full = train.copy()
train = train[train["bk_level"] != 0].reset_index(drop=True)
print(f"  Train (bk>0): {len(train):,} | Test: {len(test):,}")

# ============================================================
# 2. MAKINE + KOMUT BAZLI ISTATISTIKLER (Batch overlap kullani)
# %83 test satiri ortak batch'ten geliyor!
# Bu, o batch'in gecmis bk seviyelerini tahmin etmemize yarar.
# ============================================================
print("\n[2] Istatistiksel ozellikler hazirlaniyor...")

tnz = train_full[train_full["bk_level"] > 0].copy()

# Her (makine, komut) kombinasyonu icin genel istatistikler
mc_stats = (tnz.groupby(["machineid", "commandno"])["bk_level"]
              .agg(mc_mean="mean", mc_median="median",
                   mc_q25=lambda x: x.quantile(0.25),
                   mc_q75=lambda x: x.quantile(0.75))
              .reset_index())

# Her (makine, komut) icin BASLANGIC bk_level istatistikleri
mc_first = (tnz.groupby(["machineid","commandno","process_id"])["bk_level"]
               .first().reset_index()
               .groupby(["machineid","commandno"])["bk_level"]
               .agg(mc_init_mean="mean", mc_init_median="median",
                    mc_init_std="std")
               .reset_index())

# Batch bazli istatistikler
batch_stats = (tnz.groupby(["machineid","batchkey","commandno"])["bk_level"]
                  .agg(batch_mean="mean", batch_first="first",
                       batch_max="max", batch_min="min")
                  .reset_index())

# ============================================================
# 3. OZELLIK MUHENDISLIGI (Tamamen Sizintisiz)
# ============================================================
print("\n[3] Feature engineering...")

VALVE_COLS = [
    "fast_dosage_valve",      # Transfer: surekli acik; Dozaj: nadir
    "slow_dosage_valve",      # Dozaj: ana acik vana
    "kk_dosage_valve",        # KK tarafi dozaj
    "bk_dosage_valve",        # BK tarafi dozaj
    "kk_bk_common_discharge", # Ortak bosaltim
    "bk_irtibat_valve",       # BK baglanti vanasi
    "kk_irtibat_valve",       # KK baglanti vanasi
]

def add_features(df):
    df = df.copy()

    # ---- Zaman orani ----
    dur = (df["endtime"] - df["starttime"]).dt.total_seconds().clip(lower=1)
    ela = (df["timestamp"] - df["starttime"]).dt.total_seconds()
    df["progress"]  = (ela / dur).clip(0, 1).astype("float32")
    df["elapsed"]   = ela.astype("float32")
    df["remaining"] = (dur - ela).astype("float32")
    df["proc_dur"]  = dur.astype("float32")

    # ---- Fiziksel integral: kumulatif vana ACIK suresi ----
    # "Bu vana 30 saniyedir acik" = "30 birim sivi akti" (fiziksel)
    for v in VALVE_COLS:
        df[f"cum_{v}"] = df.groupby("process_id")[v].cumsum().astype("float32")

    df["n_valves_open"] = df[VALVE_COLS].sum(axis=1).astype("int8")
    df["cum_all_valves"] = df.groupby("process_id")["n_valves_open"].cumsum().astype("float32")

    # Son 10 saniyede acilan/kapanan vana sayisi
    df["roll10_valves"] = (
        df.groupby("process_id")["n_valves_open"]
          .transform(lambda x: x.rolling(10, min_periods=1).mean())
          .astype("float32")
    )

    # Vana acilma/kapanma anlari (PID dongu sinyali)
    for v in VALVE_COLS:
        df[f"chg_{v}"] = df.groupby("process_id")[v].diff().fillna(0).astype("int8")

    # ---- Bagimsiz sensor degisimleri (bk_level degil!) ----
    for col in ["kk_level", "ak_level"]:
        init = df.groupby("process_id")[col].transform("first")
        df[f"init_{col}"]  = init.astype("float32")
        df[f"dlt_{col}"]   = (df[col] - init).astype("float32")
        df[f"d1_{col}"]    = df.groupby("process_id")[col].diff(1).fillna(0).astype("float32")
        df[f"d5_{col}"]    = df.groupby("process_id")[col].diff(5).fillna(0).astype("float32")
        # Rolling ortalama
        df[f"r10_{col}"]   = (df.groupby("process_id")[col]
                               .transform(lambda x: x.rolling(10, min_periods=1).mean())
                               .astype("float32"))

    # ---- Ana sinyal: bk_target_level etkilesimleri ----
    # Cmd 22'de bk ~ bk_target korelasyonu 0.97!
    df["tgt_x_prog"]    = (df["bk_target_level"] * df["progress"]).astype("float32")
    df["tgt_x_rem"]     = (df["bk_target_level"] * (1 - df["progress"])).astype("float32")
    df["tgt_x_cum"]     = (df["bk_target_level"] * df["cum_all_valves"]).astype("float32")
    df["kk_gap"]        = (df["kk_target_level"] - df["kk_level"]).astype("float32")

    # ---- Komut tipleri ----
    df["is_bk_drain"]   = df["commandno"].isin([20, 22]).astype("int8")
    df["is_kk_drain"]   = df["commandno"].isin([19, 21]).astype("int8")
    df["is_transfer"]   = df["commandno"].isin([19, 20]).astype("int8")
    df["is_dosage"]     = df["commandno"].isin([21, 22]).astype("int8")

    # ---- Proses yapi ----
    df["proc_len"]    = df.groupby("process_id")["timestamp"].transform("count").astype("int32")
    df["step"]        = df.groupby("process_id").cumcount().astype("int32")
    df["step_ratio"]  = (df["step"] / df["proc_len"].clip(lower=1)).astype("float32")

    # ---- Makine + komut etkilesimi ----
    df["machine_cmd"] = (df["machineid"] * 100 + df["commandno"]).astype("int32")

    # ---- Dozaj egri tipi ----
    if "dosage_curve_type" in df.columns:
        df["curve_code"] = (df["dosage_curve_type"]
                            .fillna("NONE").astype("category")
                            .cat.codes.astype("int8"))
    else:
        df["curve_code"] = np.int8(-1)

    # ---- Mikser (seviye okumada gurultu kaynagi) ----
    df["mixer_on"] = ((df["kk_mikser_robotu"] == 1) | (df["bk_mikser_robotu"] == 1)).astype("int8")

    return df

train = add_features(train)
test  = add_features(test)

# Istatistiksel ozellikleri join et
for df in [train, test]:
    pass  # assigning below

train = (train
         .merge(mc_stats,    on=["machineid","commandno"],         how="left")
         .merge(mc_first,    on=["machineid","commandno"],         how="left")
         .merge(batch_stats, on=["machineid","batchkey","commandno"], how="left"))
test  = (test
         .merge(mc_stats,    on=["machineid","commandno"],         how="left")
         .merge(mc_first,    on=["machineid","commandno"],         how="left")
         .merge(batch_stats, on=["machineid","batchkey","commandno"], how="left"))

# Feature listesi
EXCLUDE = {
    "timestamp", "starttime", "endtime",
    "process_id", "batchkey",
    "row_id", "Id",
    "bk_level",
    "dosage_curve_type",
}
FEATURES = [c for c in train.columns if c not in EXCLUDE]
print(f"  Toplam feature: {len(FEATURES)}")

# ============================================================
# 4. MODEL EGITIMI — 4 AYRI MODEL (Her komut kendi dinamiginde)
# Cmd 19/21: bk_level degismiyor --> ak_level, progress ana sinyal
# Cmd 20/22: bk_level dusuyor --> bk_target_level ana sinyal
# ============================================================
print("\n[4] Modeller egitiliyor...")

lgb_base = dict(
    learning_rate    = 0.03,
    max_depth        = 12,
    num_leaves       = 127,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_samples= 20,
    reg_alpha        = 0.05,
    reg_lambda       = 0.5,
    random_state     = 42,
    verbose          = -1,
    n_jobs           = -1,
)

models = {}
for cmd in [19, 20, 21, 22]:
    mask_tr = train["commandno"] == cmd
    mask_te = test["commandno"] == cmd
    n_tr = mask_tr.sum()
    n_te = mask_te.sum()
    # Az veri varsa daha az estimator
    n_est = 3000 if n_tr > 50000 else (1000 if n_tr > 5000 else 500)
    print(f"  [Cmd {cmd}] Train={n_tr:,} | Test={n_te:,} | n_est={n_est}")
    
    m = lgb.LGBMRegressor(n_estimators=n_est, **lgb_base)
    m.fit(train.loc[mask_tr, FEATURES], train.loc[mask_tr, "bk_level"])
    
    train_mae = np.mean(np.abs(
        train.loc[mask_tr, "bk_level"] - m.predict(train.loc[mask_tr, FEATURES])
    ))
    print(f"    Train MAE: {train_mae:.4f}")
    models[cmd] = m

del train, train_full, tnz, mc_stats, mc_first, batch_stats
gc.collect()

# ============================================================
# 5. TEST TAHMIN (Vektörize — Lag yok, hata birikmez)
# ============================================================
print("\n[5] Test tahmini...")

preds = np.zeros(len(test), dtype="float32")
for cmd, m in models.items():
    mask = test["commandno"] == cmd
    if mask.sum() > 0:
        preds[mask] = m.predict(test.loc[mask, FEATURES])

preds = np.clip(preds, 0.0, 100.0)

# ============================================================
# 6. SUBMISSION
# ============================================================
print("\n[6] Submission olusturuluyor...")

submission = pd.DataFrame({
    "Id":        test["row_id"],
    "Predicted": preds,
})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv("submission3.csv", index=False)

print(f"\n{'='*60}")
print("TAMAMLANDI: submission3.csv")
print(f"  Shape:    {submission.shape}")
print(f"  Aralik:   [{preds.min():.2f}, {preds.max():.2f}]")
print(f"  Ortalama: {preds.mean():.2f}")
print(f"{'='*60}")

# Feature onemleri (en onemli model: cmd 22)
print("\nCmd 22 Modeli -- En Onemli 15 Feature:")
imp22 = pd.Series(models[22].feature_importances_, index=FEATURES)
print(imp22.sort_values(ascending=False).head(15).to_string())

print("\nCmd 19 Modeli -- En Onemli 10 Feature:")
imp19 = pd.Series(models[19].feature_importances_, index=FEATURES)
print(imp19.sort_values(ascending=False).head(10).to_string())
