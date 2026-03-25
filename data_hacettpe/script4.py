"""
SCRIPT4: CATBOOST GPU + LIGHTGBM ENSEMBLE  (Hedef: MAE < 20)
=============================================================
script3 (score 25) ustune eklenen iyilestirmeler:
  1. Makineye ozel bk_level - bk_target_level offset (bias duzeltme)
  2. Akis integralleri: cum(diff(kk_level)), cum(diff(ak_level))
  3. Vana gecis sayisi (PID aktivite gostergesi)
  4. Proses fazı: baslangic/orta/son
  5. CatBoost GPU + LightGBM ensemble (0.5/0.5 blend)
  6. Daha fazla rolling istatistik
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
import os, gc, warnings, time
warnings.filterwarnings("ignore")

DATA_DIR = "aiclubdatathon-26"
t0 = time.time()

# ============================================================
# 1. VERI YUKLEME
# ============================================================
print("=" * 60)
print("SCRIPT4: CATBOOST GPU + LGB ENSEMBLE")
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

# bk_level=0 cikar
train_full = train.copy()
train = train[train["bk_level"] != 0].reset_index(drop=True)
print(f"  Train (bk>0): {len(train):,} | Test: {len(test):,}")

# ============================================================
# 2. ISTATISTIKSEL OZELLIKLER (Guclendirilmis)
# ============================================================
print("\n[2] Istatistiksel ozellikler hazirlaniyor...")
tnz = train_full[train_full["bk_level"] > 0].copy()

# --- Makine + Komut bazli genel istatistikler ---
mc_stats = (tnz.groupby(["machineid","commandno"])["bk_level"]
              .agg(mc_mean="mean", mc_median="median",
                   mc_q25=lambda x: x.quantile(0.25),
                   mc_q75=lambda x: x.quantile(0.75),
                   mc_std="std")
              .reset_index())

# --- Makine + Komut bazli BASLANGIC bk_level ---
proc_first = (tnz.sort_values(["process_id","timestamp"])
                  .groupby("process_id").agg(
                      machineid=("machineid","first"),
                      commandno=("commandno","first"),
                      bk_first=("bk_level","first"),
                      bk_last=("bk_level","last"))
                  .reset_index())
mc_first = (proc_first.groupby(["machineid","commandno"])
                      .agg(mc_init_mean=("bk_first","mean"),
                           mc_init_median=("bk_first","median"),
                           mc_init_std=("bk_first","std"),
                           mc_final_mean=("bk_last","mean"))
                      .reset_index())

# --- YENI: Makine + Komut bazli TARGET OFFSET ---
# Her makine bk_target'ten ne kadar sapıyor?
tnz_22 = tnz[tnz["commandno"].isin([20, 22])].copy()
if len(tnz_22) > 0:
    tnz_22["offset"] = tnz_22["bk_level"] - tnz_22["bk_target_level"]
    mc_offset = (tnz_22.groupby(["machineid","commandno"])["offset"]
                       .agg(mc_offset_mean="mean", mc_offset_std="std",
                            mc_offset_median="median")
                       .reset_index())
else:
    mc_offset = pd.DataFrame(columns=["machineid","commandno",
                                       "mc_offset_mean","mc_offset_std","mc_offset_median"])

# --- Batch bazli istatistikler ---
batch_stats = (tnz.groupby(["machineid","batchkey","commandno"])["bk_level"]
                  .agg(batch_mean="mean", batch_first="first",
                       batch_last="last",
                       batch_max="max", batch_min="min",
                       batch_std="std", batch_range=lambda x: x.max()-x.min())
                  .reset_index())

# --- YENI: Batch bazli proses suresi ---
proc_dur = (tnz.groupby("process_id").agg(
                machineid=("machineid","first"),
                batchkey=("batchkey","first"),
                commandno=("commandno","first"),
                dur=("timestamp", lambda x: (x.max()-x.min()).total_seconds()))
            .reset_index())
batch_dur = (proc_dur.groupby(["machineid","batchkey","commandno"])["dur"]
                     .agg(batch_avg_dur="mean")
                     .reset_index())

del proc_first, proc_dur, tnz_22
gc.collect()

# ============================================================
# 3. FEATURE ENGINEERING (Guclendirilmis)
# ============================================================
print("\n[3] Feature engineering...")

VALVE_COLS = [
    "fast_dosage_valve","slow_dosage_valve","kk_dosage_valve",
    "bk_dosage_valve","kk_bk_common_discharge","bk_irtibat_valve","kk_irtibat_valve",
]

def add_features(df):
    df = df.copy()

    # ---- Zaman ----
    dur = (df["endtime"] - df["starttime"]).dt.total_seconds().clip(lower=1)
    ela = (df["timestamp"] - df["starttime"]).dt.total_seconds()
    df["progress"]  = (ela / dur).clip(0, 1).astype("float32")
    df["elapsed"]   = ela.astype("float32")
    df["remaining"] = (dur - ela).astype("float32")
    df["proc_dur"]  = dur.astype("float32")
    df["progress2"] = (df["progress"] ** 2).astype("float32")
    df["progress3"] = (df["progress"] ** 3).astype("float32")
    # Faz gostergesi (baslangic/orta/bitis)
    df["phase_start"] = (df["progress"] < 0.1).astype("int8")
    df["phase_end"]   = (df["progress"] > 0.9).astype("int8")

    # ---- Fiziksel integral: kumulatif vana suresi ----
    for v in VALVE_COLS:
        df[f"cum_{v}"] = df.groupby("process_id")[v].cumsum().astype("float32")

    df["n_valves_open"]  = df[VALVE_COLS].sum(axis=1).astype("int8")
    df["cum_all_valves"] = df.groupby("process_id")["n_valves_open"].cumsum().astype("float32")

    # YENI: Vana gecis sayisi (toggle count — PID aktivite)
    for v in VALVE_COLS:
        df[f"chg_{v}"] = df.groupby("process_id")[v].diff().fillna(0).astype("int8")
    df["valve_toggles"] = df[[f"chg_{v}" for v in VALVE_COLS]].abs().sum(axis=1).astype("int8")
    df["cum_toggles"]   = df.groupby("process_id")["valve_toggles"].cumsum().astype("float32")

    # Rolling vana aktivitesi
    df["roll10_valves"] = (
        df.groupby("process_id")["n_valves_open"]
          .transform(lambda x: x.rolling(10, min_periods=1).mean())
          .astype("float32"))
    df["roll30_valves"] = (
        df.groupby("process_id")["n_valves_open"]
          .transform(lambda x: x.rolling(30, min_periods=1).mean())
          .astype("float32"))

    # ---- Sensor degisimleri ----
    for col in ["kk_level", "ak_level"]:
        init = df.groupby("process_id")[col].transform("first")
        df[f"init_{col}"]  = init.astype("float32")
        df[f"dlt_{col}"]   = (df[col] - init).astype("float32")
        df[f"d1_{col}"]    = df.groupby("process_id")[col].diff(1).fillna(0).astype("float32")
        df[f"d5_{col}"]    = df.groupby("process_id")[col].diff(5).fillna(0).astype("float32")
        df[f"d10_{col}"]   = df.groupby("process_id")[col].diff(10).fillna(0).astype("float32")
        df[f"r10_{col}"]   = (df.groupby("process_id")[col]
                               .transform(lambda x: x.rolling(10, min_periods=1).mean())
                               .astype("float32"))
        df[f"r30_{col}"]   = (df.groupby("process_id")[col]
                               .transform(lambda x: x.rolling(30, min_periods=1).mean())
                               .astype("float32"))
        # YENI: Kumulatif akis integrali
        df[f"flow_{col}"]  = df.groupby("process_id")[f"d1_{col}"].cumsum().astype("float32")

    # ---- bk_target_level etkilesimleri ----
    df["tgt_x_prog"]    = (df["bk_target_level"] * df["progress"]).astype("float32")
    df["tgt_x_rem"]     = (df["bk_target_level"] * (1 - df["progress"])).astype("float32")
    df["tgt_x_cum"]     = (df["bk_target_level"] * df["cum_all_valves"]).astype("float32")
    df["tgt_x_prog2"]   = (df["bk_target_level"] * df["progress2"]).astype("float32")
    # YENI: target velocity (hedef ne hızda değişiyor)
    df["tgt_velocity"]  = df.groupby("process_id")["bk_target_level"].diff(1).fillna(0).astype("float32")
    df["tgt_accel"]     = df.groupby("process_id")["tgt_velocity"].diff(1).fillna(0).astype("float32")
    df["cum_tgt_vel"]   = df.groupby("process_id")["tgt_velocity"].cumsum().astype("float32")

    # ---- Hedef seviyeleri ----
    df["kk_gap"]        = (df["kk_target_level"] - df["kk_level"]).astype("float32")
    df["kk_ratio"]      = (df["kk_level"] / (df["ak_level"] + 1)).astype("float32")

    # ---- Komut tipleri ----
    df["is_bk_drain"]   = df["commandno"].isin([20, 22]).astype("int8")
    df["is_kk_drain"]   = df["commandno"].isin([19, 21]).astype("int8")
    df["is_transfer"]   = df["commandno"].isin([19, 20]).astype("int8")
    df["is_dosage"]     = df["commandno"].isin([21, 22]).astype("int8")

    # ---- Proses yapi ----
    df["proc_len"]    = df.groupby("process_id")["timestamp"].transform("count").astype("int32")
    df["step"]        = df.groupby("process_id").cumcount().astype("int32")
    df["step_ratio"]  = (df["step"] / df["proc_len"].clip(lower=1)).astype("float32")
    df["machine_cmd"] = (df["machineid"] * 100 + df["commandno"]).astype("int32")

    # Dozaj egri tipi
    if "dosage_curve_type" in df.columns:
        df["curve_code"] = (df["dosage_curve_type"]
                            .fillna("NONE").astype("category")
                            .cat.codes.astype("int8"))
    else:
        df["curve_code"] = np.int8(-1)

    # Mikser
    df["mixer_on"] = ((df["kk_mikser_robotu"]==1) | (df["bk_mikser_robotu"]==1)).astype("int8")

    # YENI: fabric_weight x progress
    df["fw_x_prog"] = (df["fabric_weight"] * df["progress"]).astype("float32")

    return df

train = add_features(train)
test  = add_features(test)

# Istatistikleri join et
for stats, keys in [
    (mc_stats,    ["machineid","commandno"]),
    (mc_first,    ["machineid","commandno"]),
    (mc_offset,   ["machineid","commandno"]),
    (batch_stats, ["machineid","batchkey","commandno"]),
    (batch_dur,   ["machineid","batchkey","commandno"]),
]:
    train = train.merge(stats, on=keys, how="left")
    test  = test.merge(stats,  on=keys, how="left")

# NaN doldur (median)
for col in train.columns:
    if train[col].dtype in ["float32","float64"] and train[col].isna().any():
        fv = train[col].median()
        train[col] = train[col].fillna(fv).astype("float32")
        test[col]  = test[col].fillna(fv).astype("float32")

# Feature listesi
EXCLUDE = {
    "timestamp", "starttime", "endtime",
    "process_id", "batchkey",
    "row_id", "Id",
    "bk_level",
    "dosage_curve_type",
}
FEATURES = [c for c in train.columns if c not in EXCLUDE]
TARGET = "bk_level"
print(f"  Toplam feature: {len(FEATURES)}")

del train_full, tnz, mc_stats, mc_first, mc_offset, batch_stats, batch_dur
gc.collect()

# ============================================================
# 4. MODEL EGITIMI: CATBOOST GPU + LIGHTGBM per COMMAND
# ============================================================
print(f"\n[4] Modeller egitiliyor... ({time.time()-t0:.0f}s)")

lgb_params = dict(
    learning_rate=0.03, max_depth=12, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.05, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1,
)

cat_params = dict(
    task_type="GPU",
    learning_rate=0.05,
    depth=10,
    l2_leaf_reg=5,
    random_seed=42,
    verbose=200,
    od_type="Iter",
    od_wait=50,
)

preds_lgb = np.zeros(len(test), dtype="float64")
preds_cat = np.zeros(len(test), dtype="float64")

for cmd in [19, 20, 21, 22]:
    mask_tr = train["commandno"] == cmd
    mask_te = test["commandno"] == cmd
    n_tr = mask_tr.sum()
    n_te = mask_te.sum()

    if n_te == 0:
        continue

    # Dynamik n_estimators
    n_est_lgb = 4000 if n_tr > 100000 else (2000 if n_tr > 10000 else 800)
    n_est_cat = 3000 if n_tr > 100000 else (1500 if n_tr > 10000 else 600)

    print(f"\n  [Cmd {cmd}] Train={n_tr:,} | Test={n_te:,}")

    X_tr = train.loc[mask_tr, FEATURES]
    y_tr = train.loc[mask_tr, TARGET]
    X_te = test.loc[mask_te, FEATURES]

    # --- LightGBM ---
    print(f"    LGB (n_est={n_est_lgb})...", end=" ", flush=True)
    m_lgb = lgb.LGBMRegressor(n_estimators=n_est_lgb, **lgb_params)
    m_lgb.fit(X_tr, y_tr)
    p_lgb = m_lgb.predict(X_te)
    mae_lgb = np.mean(np.abs(y_tr - m_lgb.predict(X_tr)))
    print(f"Train MAE: {mae_lgb:.4f}")

    # --- CatBoost GPU ---
    print(f"    CatBoost GPU (n_est={n_est_cat})...", flush=True)
    m_cat = CatBoostRegressor(iterations=n_est_cat, **cat_params)
    m_cat.fit(X_tr, y_tr, verbose=200)
    p_cat = m_cat.predict(X_te)
    mae_cat = np.mean(np.abs(y_tr - m_cat.predict(X_tr)))
    print(f"    CatBoost Train MAE: {mae_cat:.4f}")

    # Ensemble (0.5 / 0.5)
    preds_lgb[mask_te] = p_lgb
    preds_cat[mask_te] = p_cat

    del X_tr, y_tr, X_te, m_lgb, m_cat
    gc.collect()

# Feature onemleri: en son cmd 22 modeli icin (debug amaçli, tekrar egit)
# (zaten yukarida silindi, sadece loglarina bakiyoruz)

# ============================================================
# 5. ENSEMBLE + SUBMISSION
# ============================================================
print(f"\n[5] Ensemble + Submission... ({time.time()-t0:.0f}s)")

# Agirlikli ortalama
W_LGB = 0.50
W_CAT = 0.50
preds = W_LGB * preds_lgb + W_CAT * preds_cat
preds = np.clip(preds, 0.0, 100.0).astype("float32")

submission = pd.DataFrame({
    "Id":        test["row_id"],
    "Predicted": preds,
})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv("submission4.csv", index=False)

# Ayri modellerin submissionlarini da kaydet (karsilastirma icin)
sub_lgb = pd.DataFrame({"Id": test["row_id"], "Predicted": np.clip(preds_lgb, 0, 100)})
sub_lgb = sub_lgb.sort_values("Id").reset_index(drop=True)
sub_lgb.to_csv("submission4_lgb_only.csv", index=False)

sub_cat = pd.DataFrame({"Id": test["row_id"], "Predicted": np.clip(preds_cat, 0, 100)})
sub_cat = sub_cat.sort_values("Id").reset_index(drop=True)
sub_cat.to_csv("submission4_cat_only.csv", index=False)

print(f"\n{'='*60}")
print("TAMAMLANDI!")
print(f"  submission4.csv          (Ensemble {W_LGB:.0%} LGB + {W_CAT:.0%} Cat)")
print(f"  submission4_lgb_only.csv (Sadece LightGBM)")
print(f"  submission4_cat_only.csv (Sadece CatBoost)")
print(f"  Aralik:   [{preds.min():.2f}, {preds.max():.2f}]")
print(f"  Ortalama: {preds.mean():.2f}")
print(f"  Toplam sure: {time.time()-t0:.0f} saniye")
print(f"{'='*60}")
