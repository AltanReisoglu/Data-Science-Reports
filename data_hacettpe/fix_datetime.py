import pandas as pd

df = pd.DataFrame({
    'timestamp': ['2025-08-13 04:15:21.596000+00:00'],
    'starttime': ['2025-08-13 07:14:50']
})

df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed').dt.tz_localize(None) + pd.Timedelta(hours=3)
df['starttime'] = pd.to_datetime(df['starttime'], format='mixed')
print(df['timestamp'] - df['starttime'])
