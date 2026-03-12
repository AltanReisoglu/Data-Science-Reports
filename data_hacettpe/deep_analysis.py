"""
DERİN VERİ ANALİZİ — Model yazmadan önce veriyi tam anla
"""
import pandas as pd
import numpy as np

print("Veriler yükleniyor...")
train = pd.read_csv('aiclubdatathon-26/train.csv')
test  = pd.read_csv('aiclubdatathon-26/test.csv')

for df in [train, test]:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
    df['starttime'] = pd.to_datetime(df['starttime'], format='mixed', utc=True)
    df['endtime']   = pd.to_datetime(df['endtime'],   format='mixed', utc=True)
    for c in df.select_dtypes(include='bool').columns:
        df[c] = df[c].astype(int)
    df['process_id'] = (df['machineid'].astype(str) + '_' +
                        df['batchkey'].astype(str) + '_' + df['commandno'].astype(str))

train = train.sort_values(['process_id','timestamp']).reset_index(drop=True)
test  = test.sort_values(['process_id','timestamp']).reset_index(drop=True)

print(f"\nTrain: {len(train):,} | Test: {len(test):,}")

# ── 1. BK_LEVEL = 0 analizi ──
print("\n═══ 1. BK_LEVEL = 0 Analizi ═══")
print(f"Train bk_level=0 satır: {(train['bk_level']==0).sum():,} ({(train['bk_level']==0).mean()*100:.1f}%)")
# Test'te bk_level yok ama bk_target_level var
print(f"Test bk_target_level sıfır oranı: {(test['bk_target_level']==0).mean()*100:.1f}%")

# ── 2. Her komut için bk_level dinamiği ──
print("\n═══ 2. Proses İçi bk_level Dinamiği ═══")
train_nz = train[train['bk_level'] > 0].copy()
train_nz['proc_dur'] = (train_nz['endtime'] - train_nz['starttime']).dt.total_seconds()
train_nz['progress'] = ((train_nz['timestamp'] - train_nz['starttime']).dt.total_seconds() 
                        / train_nz['proc_dur'].clip(lower=1)).clip(0,1)

for cmd in [19, 20, 21, 22]:
    sub = train_nz[train_nz['commandno'] == cmd].copy()
    # İlk ve son bk_level
    first_bk = sub.groupby('process_id')['bk_level'].first()
    last_bk  = sub.groupby('process_id')['bk_level'].last()
    print(f"\nCommand {cmd} — {sub['process_id'].nunique()} proses, {len(sub):,} satır")
    print(f"  İlk bk_level:  mean={first_bk.mean():.1f}, min={first_bk.min():.1f}, max={first_bk.max():.1f}")
    print(f"  Son bk_level:  mean={last_bk.mean():.1f}")
    print(f"  Delta (son-ilk): mean={( last_bk - first_bk).mean():.2f}")
    print(f"  Proses süresi:  mean={sub.groupby('process_id')['proc_dur'].first().mean():.0f}s")
    # bk_target_level ile bk_level ilişkisi
    corr = sub[['bk_level','bk_target_level','progress']].corr()['bk_level']
    print(f"  Korelasyon bk~bk_target: {corr['bk_target_level']:.4f}")
    print(f"  Korelasyon bk~progress:  {corr['progress']:.4f}")
    
    # Transfer ise: hangi vana açık?
    valve_cols = ['fast_dosage_valve','slow_dosage_valve','kk_dosage_valve',
                  'bk_dosage_valve','kk_bk_common_discharge','bk_irtibat_valve','kk_irtibat_valve']
    top_valve = sub[valve_cols].mean().sort_values(ascending=False).head(3)
    print(f"  En aktif vanalar: {top_valve.to_dict()}")

# ── 3. Test veri yapısı — bk_level yok ama ne var? ──
print("\n═══ 3. Test Veri Yapısı ═══")
print("Test kolonları:", list(test.columns))
print(f"Test komut dağılımı:\n{test['commandno'].value_counts()}")
print(f"\nTest bk_target_level istatistikleri:")
print(test.groupby('commandno')['bk_target_level'].describe())

# ── 4. Veri bölümü analizi (train/test proses örtüşmesi) ──
print("\n═══ 4. Train/Test Proses Örtüşmesi ═══")
# Aynı makine+batch'in hem train hem test'te olup olmadığı
train['machine_batch'] = train['machineid'].astype(str) + '_' + train['batchkey'].astype(str)
test['machine_batch']  = test['machineid'].astype(str)  + '_' + test['batchkey'].astype(str)
overlap = len(set(train['machine_batch']) & set(test['machine_batch']))
print(f"Aynı batch'in hem train hem test'te olma sayısı: {overlap}")
print("(Bu varsa, test proseslerinin önceki/sonraki bk_level değerleri train'den çıkarılabilir!)")

# ── 5. Bir proseste bk_level yüzeyinin nasıl değiştiğini gör ──
print("\n═══ 5. Örnek Proses Analizi ═══")
# Komut 22'de (en zor) bir örnek proses
sample_pid_22 = train_nz[train_nz['commandno']==22].groupby('process_id').size()
sample_pid_22 = sample_pid_22[sample_pid_22 > 50].index[5]
ex22 = train_nz[train_nz['process_id'] == sample_pid_22][['progress','bk_level','bk_target_level','slow_dosage_valve','fast_dosage_valve']].head(30)
print(f"Proses {sample_pid_22} (cmd 22):")
print(ex22.to_string(index=False))

sample_pid_20 = train_nz[train_nz['commandno']==20].groupby('process_id').size()
sample_pid_20 = sample_pid_20[sample_pid_20 > 50].index[3]
ex20 = train_nz[train_nz['process_id'] == sample_pid_20][['progress','bk_level','bk_target_level','fast_dosage_valve','bk_irtibat_valve']].head(20)
print(f"\nProses {sample_pid_20} (cmd 20):")
print(ex20.to_string(index=False))
