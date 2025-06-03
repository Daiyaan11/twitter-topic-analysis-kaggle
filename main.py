import os
import pandas as pd
from src.preprocess import clean_text, extract_hashtags, extract_active_users
from src.visualize import plot_top_hashtags, plot_top_users

def clean_raw_csv(input_csv, output_csv):
    """
    Reads the raw CSV, removes malformed lines, and saves cleaned CSV.
    """
    print("Cleaning raw CSV file...")
    with open(input_csv, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_csv, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Simple heuristic: write lines containing expected number of commas (adjust if needed)
            if line.count(',') >= 10:  # adjust number according to columns expected
                outfile.write(line)
    print(f"Cleaned CSV saved to {output_csv}")

def main():
    raw_csv = 'data/raw/Bitcoin_tweets_dataset_2.csv'
    cleaned_csv = 'data/raw/Bitcoin_tweets_dataset_2_clean.csv'

    # Clean the raw CSV first to avoid parsing errors
    if not os.path.exists(cleaned_csv):
        clean_raw_csv(raw_csv, cleaned_csv)

    print("Loading cleaned data...")
    # Use python engine and skip bad lines (if any remain)
    df = pd.read_csv(cleaned_csv, engine='python', on_bad_lines='skip')
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    print("Preprocessing tweets...")
    # Apply the text cleaning function to the 'text' column
    df['clean_text'] = df['text'].apply(clean_text)

    print("Extracting hashtags and active users...")
    hashtags_df = extract_hashtags(df['hashtags'])
    users_df = extract_active_users(df)

    # Ensure the plots directory exists
    os.makedirs('plots', exist_ok=True)

    print("Plotting top hashtags and users...")
    plot_top_hashtags(hashtags_df)
    plot_top_users(users_df)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
