import pandas as pd
import os

def main():
    
    input_path = 'Data/processed_posts_with_prompts.csv'
    train_path = 'Data/processed_posts_with_prompts_train.csv'
    test_path = 'Data/processed_posts_with_prompts_test.csv'
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, sep=';')
    
    # Ensure date column is datetime for correct sorting
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Sort by date for timepoint split
        df = df.sort_values(by='date')
    else:
        print("Warning: 'date' column not found! Splitting sequentially without date sorting.")
    
    # Calculate split index for 15% test set
    split_idx = int(len(df) * 0.85)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Total rows: {len(df)}")
    print(f"Train set (85%): {len(train_df)} rows")
    print(f"Test set (15%): {len(test_df)} rows")
    
    # Save the datasets
    train_df.to_csv(train_path, sep=';', index=False)
    test_df.to_csv(test_path, sep=';', index=False)
    print("Files saved successfully!")

if __name__ == '__main__':
    main()
