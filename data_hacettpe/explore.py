"""
Veri keşfi: Her prosesin fiziksel mantığını anla
- Command 20/22: BK kazanından boşaltma → bk_level DÜŞER
- Command 19/21: KK kazanından boşaltma → bk_level pek değişmez mi?
- Vanalar ne zaman açık?
"""
import pandas as pd
import numpy as np

print("Loading sample...")
train = pd.read_csv('aiclubdatathon-26/train.csv', nrows=500000)

for col in ['timestamp','starttime','endtime']:
    train[col] = pd.to_datetime(train[col], format='mixed', utc=True)

for c in train.select_dtypes(include='bool').columns:
    train[c] = train[c].astype(int)

train['process_id'] = (train['machineid'].astype(str) + '_' + 
                      train['batchkey'].astype(str) + '_' + 
                      train['commandno'].astype(str))
train = train.sort_values(['process_id','timestamp'])

print("\n═══ BK_LEVEL DEĞİŞİMİ KOMUTLARA GÖRE ═══")
for cmd in [19,20,21,22]:
    sub = train[train['commandno']==cmd]
    first = sub.groupby('process_id')['bk_level'].first()
    last  = sub.groupby('process_id')['bk_level'].last()
    delta = last - first
    nonzero_first = first[first > 0]
    print(f"\nCommand {cmd} ({len(sub.groupby('process_id'))} proses):")
    print(f"  Başlangıç bk_level: mean={nonzero_first.mean():.1f}, median={nonzero_first.median():.1f}")
    print(f"  Bitiş bk_level:     mean={last[first>0].mean():.1f}")
    print(f"  Delta (son-ilk):    mean={delta[first>0].mean():.2f}")
    
    valve_cols = ['fast_dosage_valve','slow_dosage_valve','kk_dosage_valve',
                  'bk_dosage_valve','kk_bk_common_discharge','bk_irtibat_valve','kk_irtibat_valve']
    print(f"  Ortalama açık vana: {sub[valve_cols].mean().sort_values(ascending=False).head(3).to_dict()}")

print("\n═══ VANA VE BK_LEVEL KORELASYONU ═══")
train_nz = train[train['bk_level'] > 0].copy()
train_nz['cum_fast'] = train_nz.groupby('process_id')['fast_dosage_valve'].cumsum()
train_nz['cum_bk_dos'] = train_nz.groupby('process_id')['bk_dosage_valve'].cumsum()
train_nz['progress'] = (
    (train_nz['timestamp'] - train_nz['starttime']).dt.total_seconds() / 
    ((train_nz['endtime'] - train_nz['starttime']).dt.total_seconds().clip(lower=1))
).clip(0, 1)

sample = train_nz.sample(50000, random_state=42)
corr_cols = ['progress','cum_fast','cum_bk_dos','kk_level','ak_level',
             'bk_target_level','bk_irtibat_valve','fast_dosage_valve']
corr = sample[corr_cols + ['bk_level']].corr()['bk_level'].sort_values(ascending=False)
print(corr.to_string())
