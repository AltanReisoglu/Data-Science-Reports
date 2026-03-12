import pandas as pd
import gc

def check_data():
    sample_df = pd.read_csv('aiclubdatathon-26/train.csv', nrows=100000)
    test_df = pd.read_csv('aiclubdatathon-26/test.csv', nrows=100)
    
    with open('check_data.txt', 'w') as f:
        f.write("TRAIN INFO:\n")
        sample_df.info(buf=f)
        f.write("\n\nTRAIN DESCRIBE:\n")
        f.write(sample_df.describe().to_string())
        f.write("\n\nTRAIN HEAD:\n")
        f.write(sample_df.head().to_string())
        
        f.write("\n\nTEST INFO:\n")
        test_df.info(buf=f)
        f.write("\n\nTEST HEAD:\n")
        f.write(test_df.head().to_string())

if __name__ == '__main__':
    check_data()
