import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_batch_machine_relations(data_dir):
    print("Loading data...")
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # Process ID creation
    def create_process_id(df):
        return df['machineid'].astype(str) + '_' + \
               df['batchkey'].astype(str) + '_' + \
               df['commandno'].astype(str) + '_' + \
               df['stepno'].astype(str)
               
    train['process_id'] = create_process_id(train)
    train['timestamp'] = pd.to_datetime(train['timestamp'], format='mixed').dt.tz_localize(None) + pd.Timedelta(hours=3)
    train['starttime'] = pd.to_datetime(train['starttime'], format='mixed')
    train['endtime'] = pd.to_datetime(train['endtime'], format='mixed')
    
    with open('batch_analysis.txt', 'w') as f:
        # Machine level stats
        f.write("=== MACHINE LEVEL ===\n")
        f.write(str(train.groupby('machineid')['batchkey'].nunique()) + '\n\n')
        
        # Analyze chronological sequence of processes for a single active machine (e.g., 105)
        machine_105 = train[train['machineid'] == 105]
        
        # Get unique processes sorted by starttime
        processes = machine_105.groupby(['process_id', 'batchkey', 'commandno', 'stepno']).agg({
            'timestamp': ['min', 'max'],
            'bk_level': ['first', 'last'],
            'bk_target_level': 'mean',
            'kk_level': ['first', 'last']
        }).reset_index()
        
        # Flatten columns
        processes.columns = ['_'.join(col).strip('_') for col in processes.columns.values]
        processes = processes.sort_values(by='timestamp_min')
        
        f.write("=== CHRONOLOGICAL PROCESSES ON MACHINE 105 (First 20) ===\n")
        f.write(processes.head(20).to_string() + '\n\n')
        
        # Check if bk_level 'last' of one process matches 'first' of next process (within same batch or across)
        f.write("=== CONTINUITY ANALYSIS ===\n")
        processes['prev_bk_last'] = processes['bk_level_last'].shift(1)
        processes['prev_batch'] = processes['batchkey'].shift(1)
        
        processes['continuity_diff'] = abs(processes['bk_level_first'] - processes['prev_bk_last'])
        
        same_batch_diff = processes[processes['batchkey'] == processes['prev_batch']]['continuity_diff'].mean()
        diff_batch_diff = processes[processes['batchkey'] != processes['prev_batch']]['continuity_diff'].mean()
        
        f.write(f"Mean Difference in bk_level between consecutive processes IN SAME BATCH: {same_batch_diff:.2f}\n")
        f.write(f"Mean Difference in bk_level between consecutive processes ACROSS BATCHES: {diff_batch_diff:.2f}\n")
        
        f.write("\n=== COMMAND DISTRIBUTIONS ===\n")
        f.write(str(train.groupby('commandno').size()) + '\n')

if __name__ == '__main__':
    analyze_batch_machine_relations('aiclubdatathon-26')
