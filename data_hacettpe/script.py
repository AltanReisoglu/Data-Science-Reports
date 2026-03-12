"""
═══════════════════════════════════════════════════════════════════════════
 BOYA FABRİKASI DATATHONu — SON VE ROBUST ÇÖZÜM
═══════════════════════════════════════════════════════════════════════════

FİZİKSEL ANLAMA:
  Command 20/22 (BK dreni): bk_level DÜŞER, başlangıç ~50-60%
  Command 19/21 (KK dreni): bk_level pek değişmez
  En güçlü sinyal: bk_target_level (korel. 0.79)
  İkinci sinyal: bk_irtibat_valve (BK vanası açık olunca seviye değişir)

STRATEJİ:
  1. LAG KULLANMA → her prosesin başlangıcı bilinmez, lag=0 hatası yapar
  2. YERİNE: Kümülatif vana süresi (fiziksel integral) kullan
     → "Bu vana X saniyedir açık" = "Bu kadar sıvı aktı" demek
  3. Komut tipine göre ayrı özellikler + ortak model
  4. Sıfır sızıntı: bk_level değerlerinden HİÇBİR feature türetilmez
  
TEST INFERENCE: Doğrudan tahmin (iteratif loop yok!)
  → Lag kullanmadığımız için döngü gereksiz, vektörize predict yeterli

METRIK: MAE (sadece bk_level > 0 satırlar)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import os, gc, warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'aiclubdatathon-26'

# ─────────────────────────────────────────────────────
# 1. VERİ YÜKLEME
# ─────────────────────────────────────────────────────
print("◆ Veri yükleniyor...")
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

for df in [train, test]:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df['starttime'] = pd.to_datetime(df['starttime'], format='mixed', utc=True)
    df['endtime']   = pd.to_datetime(df['endtime'],   format='mixed', utc=True)
    for c in df.select_dtypes(include='bool').columns:
        df[c] = df[c].astype(int)
    # Her proses: makine + batch + komut birleşimi
    df['process_id'] = (df['machineid'].astype(str) + '_' +
                        df['batchkey'].astype(str) + '_' +
                        df['commandno'].astype(str))

# Kritik: Sıralama ÖNCE yapılmalı (cumsum doğru çalışsın)
train = train.sort_values(['process_id', 'timestamp']).reset_index(drop=True)
test  = test.sort_values(['process_id', 'timestamp']).reset_index(drop=True)

# bk_level=0 satırlar değerlendirme dışı → eğitimden çıkar  
train = train[train['bk_level'] != 0].reset_index(drop=True)

print(f"  Train (bk>0): {len(train):,} satır")
print(f"  Test:         {len(test):,} satır")

# ─────────────────────────────────────────────────────
# 2. ÖZELLİK MÜHENDİSLİĞİ (TAMAMEN SIZINTISIZ)
# Kural: bk_level değerinden HİÇBİR şey türetme!
# ─────────────────────────────────────────────────────
print("\n◆ Feature mühendisliği...")

VALVE_COLS = [
    'fast_dosage_valve',   # Transfer'de ana vana, dozajda nadir
    'slow_dosage_valve',   # Dozajda ana vana
    'kk_dosage_valve',     # KK tarafı dozaj
    'bk_dosage_valve',     # BK tarafı dozaj
    'kk_bk_common_discharge',  # Ortak boşaltım
    'bk_irtibat_valve',    # BK irtibat (korel. 0.24!)
    'kk_irtibat_valve',    # KK irtibat
]

def build_features(df):
    df = df.copy()

    # ── Zaman Özellikleri ──
    dur = (df['endtime'] - df['starttime']).dt.total_seconds().clip(lower=1)
    ela = (df['timestamp'] - df['starttime']).dt.total_seconds()
    df['progress']      = (ela / dur).clip(0, 1).astype('float32')
    df['elapsed_s']     = ela.astype('float32')
    df['remaining_s']   = (dur - ela).astype('float32')
    df['proc_duration'] = dur.astype('float32')

    # ── FİZİKSEL İNTEGRAL: Kümülatif Vana Açık Süresi ──
    # 1 saniyede 1 ölçüm → cumsum = kaç saniyedir açık = ne kadar aktı
    for v in VALVE_COLS:
        df[f'cum_{v}'] = df.groupby('process_id')[v].cumsum().astype('float32')

    # Toplam açık vana sayısı ve kümülatifi
    df['n_valves_open'] = df[VALVE_COLS].sum(axis=1).astype('int8')
    df['cum_total_valve'] = df.groupby('process_id')['n_valves_open'].cumsum().astype('float32')

    # Anlık vana değişimleri (açılma/kapanma sinyali)
    for v in VALVE_COLS:
        df[f'chg_{v}'] = df.groupby('process_id')[v].diff().fillna(0).astype('int8')

    # ── Sensör Sütunları Zaman İçindeki Değişim ──
    # Bu sütunlar bk_level DEĞİL, bağımsız sensörler
    for col in ['kk_level', 'ak_level']:
        init = df.groupby('process_id')[col].transform('first')
        df[f'init_{col}']    = init.astype('float32')
        df[f'delta_{col}']   = (df[col] - init).astype('float32')  # başlangıçtan değişim
        df[f'diff1_{col}']   = df.groupby('process_id')[col].diff(1).fillna(0).astype('float32')
        df[f'diff5_{col}']   = df.groupby('process_id')[col].diff(5).fillna(0).astype('float32')
        # Rolling ortalama (son 10 saniyenin ortalaması)
        df[f'roll10_{col}']  = (df.groupby('process_id')[col]
                                  .transform(lambda x: x.rolling(10, min_periods=1).mean())
                                  .astype('float32'))

    # ── HEDEF ETKİLEŞİMLERİ (En Önemli!) ──
    # bk_target_level korelasyon 0.79 → prosesi bu seviyeye götürmeye çalışıyoruz
    df['target_x_prog']     = (df['bk_target_level'] * df['progress']).astype('float32')
    df['target_x_remain']   = (df['bk_target_level'] * (1 - df['progress'])).astype('float32')
    df['target_x_valvesum'] = (df['bk_target_level'] * df['cum_total_valve']).astype('float32')

    # ── Komut Tipi Bayrakları ──
    df['is_bk_drain']   = df['commandno'].isin([20, 22]).astype('int8')  # BK tarafını drene eder
    df['is_kk_drain']   = df['commandno'].isin([19, 21]).astype('int8')  # KK tarafını drene eder
    df['is_transfer']   = df['commandno'].isin([19, 20]).astype('int8')  # Hızlı, doğrusal
    df['is_dosage']     = df['commandno'].isin([21, 22]).astype('int8')  # Yavaş, PID salınım

    # ── Proses Yapısı ──
    df['proc_length']   = df.groupby('process_id')['timestamp'].transform('count').astype('int32')
    df['step']          = df.groupby('process_id').cumcount().astype('int32')
    df['step_ratio']    = (df['step'] / df['proc_length'].clip(lower=1)).astype('float32')

    # ── KK/BK Kazanı Oranı ──
    df['kk_ak_ratio']   = (df['kk_level'] / (df['ak_level'] + 1)).astype('float32')
    df['kk_target_gap'] = (df['kk_target_level'] - df['kk_level']).astype('float32')

    # ── Dozaj Eğri Tipi ──
    if 'dosage_curve_type' in df.columns:
        df['curve_code'] = (df['dosage_curve_type']
                            .fillna('NONE').astype('category').cat.codes.astype('int8'))

    # ── Mikser (Kazanda dalgalanma yaratır → seviye ölçümünü etkiler) ──
    df['mixer_on'] = ((df['kk_mikser_robotu'] == 1) | (df['bk_mikser_robotu'] == 1)).astype('int8')

    return df

train = build_features(train)
test  = build_features(test)

# ─────────────────────────────────────────────────────
# 3. FEATURE LİSTESİNİ BELİRLE
# ─────────────────────────────────────────────────────
EXCLUDE = {
    'timestamp', 'starttime', 'endtime',  # datetime objeler
    'process_id', 'batchkey',             # string ID'ler
    'row_id', 'Id',                       # submission key
    'bk_level',                           # HEDEF
    'dosage_curve_type',                  # kategorik ham sütun (curve_code kullanıyoruz)
}
FEATURES = [c for c in train.columns if c not in EXCLUDE]
print(f"  Toplam feature: {len(FEATURES)}")

# ─────────────────────────────────────────────────────
# 4. MODEL EĞİTİMİ (2 AYRI MODEL: Transfer + Dozaj)
# ─────────────────────────────────────────────────────
print("\n◆ Model eğitimi başlıyor...")

lgb_params = dict(
    n_estimators    = 3000,
    learning_rate   = 0.03,
    max_depth       = 12,
    num_leaves      = 127,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    min_child_samples = 20,
    reg_alpha       = 0.05,
    reg_lambda      = 1.0,
    random_state    = 42,
    verbose         = -1,
    n_jobs          = -1,
)

# Transfer modeli (19, 20) — doğrusal, hızlı boşaltma
mask_tr_train = train['commandno'].isin([19, 20])
mask_tr_test  = test['commandno'].isin([19, 20])
print(f"  [Transfer] Train: {mask_tr_train.sum():,} | Test: {mask_tr_test.sum():,}")

model_tr = lgb.LGBMRegressor(**lgb_params)
model_tr.fit(train.loc[mask_tr_train, FEATURES], train.loc[mask_tr_train, 'bk_level'])
tr_mae = np.mean(np.abs(
    train.loc[mask_tr_train, 'bk_level'] -
    model_tr.predict(train.loc[mask_tr_train, FEATURES])
))
print(f"    Train MAE: {tr_mae:.4f}")

# Dozaj modeli (21, 22) — PID kontrolü, salınım
mask_dz_train = train['commandno'].isin([21, 22])
mask_dz_test  = test['commandno'].isin([21, 22])
print(f"  [Dozaj]    Train: {mask_dz_train.sum():,} | Test: {mask_dz_test.sum():,}")

model_dz = lgb.LGBMRegressor(**lgb_params)
model_dz.fit(train.loc[mask_dz_train, FEATURES], train.loc[mask_dz_train, 'bk_level'])
dz_mae = np.mean(np.abs(
    train.loc[mask_dz_train, 'bk_level'] -
    model_dz.predict(train.loc[mask_dz_train, FEATURES])
))
print(f"    Train MAE: {dz_mae:.4f}")

del train
gc.collect()

# ─────────────────────────────────────────────────────
# 5. TEST TAHMİNİ (Vektörize — Döngü Yok!)
# Lag kullanmadığımız için iteratif inference GEREKMİYOR
# ─────────────────────────────────────────────────────
print("\n◆ Test tahmini yapılıyor...")

preds = np.zeros(len(test), dtype=np.float32)
preds[mask_tr_test] = model_tr.predict(test.loc[mask_tr_test, FEATURES])
preds[mask_dz_test] = model_dz.predict(test.loc[mask_dz_test, FEATURES])

# Fiziksel sınır: bk_level ∈ [0, 100]
preds = np.clip(preds, 0.0, 100.0)

# ─────────────────────────────────────────────────────
# 6. SUBMISSION DOSYASI
# ─────────────────────────────────────────────────────
print("\n◆ Submission oluşturuluyor...")

submission = pd.DataFrame({
    'Id':        test['row_id'],
    'Predicted': preds
})
submission = submission.sort_values('Id').reset_index(drop=True)
submission.to_csv('submission_final.csv', index=False)

print(f"\n{'═'*55}")
print(" TAMAMLANDI!")
print(f"{'═'*55}")
print(f"  Dosya:    submission_final.csv")
print(f"  Shape:    {submission.shape}")
print(f"  Aralık:   [{submission['Predicted'].min():.2f}, {submission['Predicted'].max():.2f}]")
print(f"  Ortalama: {submission['Predicted'].mean():.2f}")
print(f"{'═'*55}")

# ── Feature Önemleri (Debug) ──
print("\nTransfer Modeli — En Önemli 10 Feature:")
imp_tr = pd.Series(model_tr.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp_tr.head(10).to_string())

print("\nDozaj Modeli — En Önemli 10 Feature:")
imp_dz = pd.Series(model_dz.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp_dz.head(10).to_string())
