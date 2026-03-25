"""
KAGGLE ICIN PURE H2O AUTOML PIPELINE
=====================================
Bu script Kaggle ortaminda uzun sureli calismak uzere tasarlanmistir.
- Tum komutlar (19, 20, 21, 22) icin sadece H2O AutoML kullanilir.
- Skorun dusmemesi icin karmasik overfit yaratan feature'lar yerine 
  Skoru 25.0 getiren guvenilir 'script3' feature seti kullanilmistir.
- Cmd 22 (1.6M satir) icin sure siniri (max_runtime_secs) KADIRILMIS,
  bunun yerine max_models=50 ayarlanmistir (Saatlerce surebilir).
- Kaggle'in 30GB RAM sinirina takilmamak icin agresif bellek temizligi yapilir.

ONEMLI: Kaggle uzerinde calistirirken Notebook sure siniri (>9 saat) ve 
High-RAM ayarlarinin (30GB) acik oldugundan emin olun.
"""

import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import os, gc, warnings

warnings.filterwarnings("ignore")

# Kaggle dataset yolu - Kaggle'da calistirirken bunu guncelleyebilirsiniz
# Ornegin: DATA_DIR = "/kaggle/input/boya-fabrikasi-datathon" if os.path.exists("/kaggle") else "aiclubdatathon-26"
DATA_DIR = "/kaggle/input/boya-fabrikasi-datathon" if os.path.exists("/kaggle") else "aiclubdatathon-26"

# ============================================================
# 1. H2O BASLAT
# ============================================================
print("=" * 60)
print("PURE H2O AUTOML PIPELINE (KAGGLE EDITION)")
print("=" * 60)

# Kaggle kernel'larinda Java/H2O sorunsuz calisir. 
# Mevcut tum RAM'i kullanmasi icin 28G ayarliyoruz. 
# Eger yerel makinedeyseniz, kendi makine RAM'inize gore ayarlayin.
print("\n[1] H2O baslatiliyor (Max RAM: 28G) ...")
try:
    h2o.init(nthreads=-1, max_mem_size="28G")
except Exception as e:
    print(f"Hata olustu: {e}")
    print("Kaggle'da Java kurulumuyla ilgili sorun varsa !apt-get update && apt-get install default-jre -y komutunu notebook bash'e ekleyin.")
    raise

# ============================================================
# 2. VERI YUKLEME + TEMEL ISLEMLER
# ============================================================
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

# Zaman serisi oldugu icin siralama cok onemli
train = train.sort_values(["process_id", "timestamp"]).reset_index(drop=True)
test  = test.sort_values(["process_id", "timestamp"]).reset_index(drop=True)

# bk_level=0 egitimden cikar (skorlanmaz)
train_full = train.copy()
train = train[train["bk_level"] != 0].reset_index(drop=True)
tnz = train_full[train_full["bk_level"] > 0].copy()

# ============================================================
# 3. ISTATISTIKSEL OZELLIKLER (25 score getiren temiz featurelar)
# ============================================================
print("\n[3] Istatistiksel ozellikler hazirlaniyor...")

mc_stats = (tnz.groupby(["machineid", "commandno"])["bk_level"]
              .agg(mc_mean="mean", mc_median="median")
              .reset_index())

batch_stats = (tnz.groupby(["machineid", "batchkey", "commandno"])["bk_level"]
                  .agg(batch_mean="mean", batch_first="first", batch_max="max")
                  .reset_index())

del tnz, train_full
gc.collect()

print(f"  Train (bk>0): {len(train):,} | Test: {len(test):,}")

# ============================================================
# 4. GUVENILIR FEATURE ENGINEERING (No Leakage)
# ============================================================
print("\n[4] Feature engineering...")

VALVE_COLS = ["fast_dosage_valve", "slow_dosage_valve", "kk_dosage_valve",
              "bk_dosage_valve", "kk_bk_common_discharge", "bk_irtibat_valve", "kk_irtibat_valve"]

def add_features(df):
    df = df.copy()
    
    # --- Zaman Orani ---
    dur = (df["endtime"] - df["starttime"]).dt.total_seconds().clip(lower=1)
    ela = (df["timestamp"] - df["starttime"]).dt.total_seconds()
    df["progress"]  = (ela / dur).clip(0, 1).astype("float32")
    df["elapsed"]   = ela.astype("float32")
    df["remaining"] = (dur - ela).astype("float32")
    df["proc_dur"]  = dur.astype("float32")

    # --- Kümülatif Vana Aktiviteleri (Fiziksel Akis Integralleri) ---
    for v in VALVE_COLS:
        df[f"cum_{v}"] = df.groupby("process_id")[v].cumsum().astype("float32")
    
    df["n_valves"]   = df[VALVE_COLS].sum(axis=1).astype("int8")
    df["cum_valves"] = df.groupby("process_id")["n_valves"].cumsum().astype("float32")

    # --- Sensor Degisimleri ---
    for col in ["kk_level", "ak_level"]:
        init = df.groupby("process_id")[col].transform("first")
        df[f"init_{col}"] = init.astype("float32")
        df[f"dlt_{col}"]  = (df[col] - init).astype("float32")
        df[f"d1_{col}"]   = df.groupby("process_id")[col].diff(1).fillna(0).astype("float32")
        df[f"d5_{col}"]   = df.groupby("process_id")[col].diff(5).fillna(0).astype("float32")

    # --- bk_target_level Etkilesimleri (Cmd 20/22 icin ana sinyal) ---
    df["tgt_x_prog"]  = (df["bk_target_level"] * df["progress"]).astype("float32")
    df["tgt_x_rem"]   = (df["bk_target_level"] * (1 - df["progress"])).astype("float32")
    df["tgt_x_cum"]   = (df["bk_target_level"] * df["cum_valves"]).astype("float32")
    df["kk_gap"]      = (df["kk_target_level"] - df["kk_level"]).astype("float32")

    # --- Kategorik ve Yapi Isaretleri ---
    df["is_bk_drain"] = df["commandno"].isin([20, 22]).astype("int8")
    df["is_transfer"] = df["commandno"].isin([19, 20]).astype("int8")
    df["is_dosage"]   = df["commandno"].isin([21, 22]).astype("int8")
    
    df["proc_len"]    = df.groupby("process_id")["timestamp"].transform("count").astype("int32")
    df["step"]        = df.groupby("process_id").cumcount().astype("int32")
    df["step_ratio"]  = (df["step"] / df["proc_len"].clip(lower=1)).astype("float32")

    if "dosage_curve_type" in df.columns:
        df["curve_code"] = df["dosage_curve_type"].fillna("NONE").astype("category").cat.codes.astype("int8")
    else:
        df["curve_code"] = np.int8(-1)

    df["mixer_on"] = ((df["kk_mikser_robotu"]==1) | (df["bk_mikser_robotu"]==1)).astype("int8")
    return df

train = add_features(train)
test  = add_features(test)

# --- Istatistikleri Join Et ---
train = train.merge(mc_stats, on=["machineid", "commandno"], how="left")
train = train.merge(batch_stats, on=["machineid", "batchkey", "commandno"], how="left")
test  = test.merge(mc_stats,  on=["machineid", "commandno"], how="left")
test  = test.merge(batch_stats, on=["machineid", "batchkey", "commandno"], how="left")

# --- H2O Hata Yapmasin Diye NaN Degerleri Doldur ---
# (NaN birakilirsa H2O numerik kolonu kategorik sanip cokebilir)
stat_cols = ["mc_mean", "mc_median", "batch_mean", "batch_first", "batch_max"]
for col in stat_cols:
    if col in train.columns:
        fill_val = train[col].median()
        train[col] = train[col].fillna(fill_val).astype("float32")
        test[col]  = test[col].fillna(fill_val).astype("float32")

# --- Modellenmeyecek Kolonlari Cikar ---
EXCLUDE = {"timestamp", "starttime", "endtime", "process_id", "batchkey",
           "row_id", "Id", "bk_level", "dosage_curve_type"}
FEATURES = [c for c in train.columns if c not in EXCLUDE]
TARGET = "bk_level"

print(f"  Toplam Kullanilacak Feature Sayisi: {len(FEATURES)}")


# ============================================================
# 5. KAGGLE ICIN PURE H2O AUTOML 
# ============================================================
print("\n[5] H2O AutoML Egitimi Basliyor (Uzun surecek!!!)...")

preds = np.zeros(len(test), dtype="float32")

for cmd in [19, 20, 21, 22]:
    mask_tr = train["commandno"] == cmd
    mask_te = test["commandno"] == cmd
    n_tr = mask_tr.sum()
    n_te = mask_te.sum()
    
    if n_te == 0:
        continue
    
    print(f"\n" + "="*50)
    print(f"  [Cmd {cmd}] Train={n_tr:,} | Test={n_te:,}")
    print("="*50)
    
    # RAM tasarrufu ve Hiz icin H2O'ya sadece gerekli satirlari/kolonlari gonderiyoruz
    tr_cols = [c for c in FEATURES if c in train.columns] + [TARGET]
    te_cols = [c for c in FEATURES if c in test.columns]
    
    tr_h2o = h2o.H2OFrame(train.loc[mask_tr, tr_cols])
    te_h2o = h2o.H2OFrame(test.loc[mask_te,  te_cols])
    
    # H2O kategorik donusum - ID/Enum tarzi kolonlar agac algoritmalarinda daha iyi yarilir
    cat_cols = ["machineid", "commandno", "prgno", "stepno", "command_repetition", "curve_code"]
    for col in cat_cols:
        if col in tr_h2o.columns:
            try:
                tr_h2o[col] = tr_h2o[col].asfactor()
                te_h2o[col] = te_h2o[col].asfactor()
            except:
                pass
    
    # Kaggle Ayarlari: Cmd 22 (1.6M satir) max performansi hedefler
    if cmd == 22:
        print("  Cmd 22 isleniyor. Bu komut 1.6 Milyon satira sahip.")
        print("  Zaman siniri (max_runtime_secs) KADIRILDI. Cok sayida model degeledirilecek.")
        aml = H2OAutoML(
            max_models=25,               # Kaggle'da 30G RAM ile guvende kalmak icin 25-30 model ideal. 
                                         # Eger daha fazla isterseniz 50 yapabilirsiniz, ama ensemble cok buyurse RAM bitebilir.
            seed=42,
            sort_metric="MAE",
            nfolds=5,                    # Gucu saglayan yapi tasi: 5 Katli Cross Validation!
            keep_cross_validation_predictions=True,
            include_algos=["GBM", "XGBoost", "DRF", "StackedEnsemble"],
        )
    else:
        # Kucuk komutlar max 15 dakika icinde tamamlanabilir
        aml = H2OAutoML(
            max_runtime_secs=900,        # 15 dakika limit
            max_models=20,
            seed=42,
            sort_metric="MAE",
            nfolds=5,
            keep_cross_validation_predictions=True,
            include_algos=["GBM", "XGBoost", "DRF", "StackedEnsemble"],
        )
    
    # Train
    print("  Egitim basladi, lutfen bekleyin...")
    aml.train(
        x=[c for c in FEATURES if c in tr_h2o.columns], 
        y=TARGET, 
        training_frame=tr_h2o
    )
    
    best = aml.leader
    print(f"\n  En iyi model secildi!: {best.model_id}")
    
    print("  --- Leaderboard (Ilk 5 Model) ---")
    lb = aml.leaderboard
    print(lb.head(rows=5))
    
    # Tahmin Al
    print("\n  Test uzerinde tahmin yapiliyor...")
    p = best.predict(te_h2o).as_data_frame()["predict"].values
    preds[mask_te] = p.astype("float32")
    
    # [KAGGLE RAM YONETIMI ICIN KRITIK] Her iterasyonda H2O cluster'in hafizasini sifirla
    h2o.remove_all()
    del tr_h2o, te_h2o, aml, best, lb
    gc.collect()

# Negatif tahminleri kirp
preds = np.clip(preds, 0.0, 100.0)

# ============================================================
# 6. SUBMISSION OLUSTUR
# ============================================================
print("\n[6] Submission dosyasi olusturuluyor...")
submission = pd.DataFrame({
    "Id":        test["row_id"],
    "Predicted": preds,
})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv("submission_h2o_kaggle.csv", index=False)

print(f"\n{'='*60}")
print("TAMAMLANDI: submission_h2o_kaggle.csv")
print(f"  Test Shape:    {submission.shape}")
print(f"  Tahmin Aralik: [{preds.min():.2f}, {preds.max():.2f}]")
print(f"  Ortalama:      {preds.mean():.2f}")
print("Artik bu csv dosyasini indirebilir veya Output sekmesinden submit edebilirsiniz!")
print(f"{'='*60}")

try:
    h2o.shutdown(prompt=False)
except:
    pass
