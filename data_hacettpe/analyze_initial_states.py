import pandas as pd
import numpy as np

print("Loading train data...")
train = pd.read_csv('aiclubdatathon-26/train.csv')

# Create process_id
train['process_id'] = (train['machineid'].astype(str) + '_' + 
                      train['batchkey'].astype(str) + '_' + 
                      train['commandno'].astype(str))

# Sort
train['timestamp'] = pd.to_datetime(train['timestamp'], format='mixed', utc=True)
train = train.sort_values(['process_id', 'timestamp'])

print("\nAnalyzing initial bk_level per process type:")
first_rows = train.groupby('process_id').first().reset_index()

for cmd in [19, 20, 21, 22]:
    subset = first_rows[first_rows['commandno'] == cmd]
    print(f"\nCommand {cmd}:")
    print(f"  Count: {len(subset)}")
    print(f"  Initial bk_level Mean:   {subset['bk_level'].mean():.2f}")
    print(f"  Initial bk_level Median: {subset['bk_level'].median():.2f}")
    print(f"  Initial bk_level Min:    {subset['bk_level'].min():.2f}")
    print(f"  Initial bk_level Max:    {subset['bk_level'].max():.2f}")
    print(f"  % starting at 0:         {(subset['bk_level'] == 0).mean()*100:.1f}%")
