import pandas as pd
import os

def append_averages():
    files = [
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/scores_tokens.csv',
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/recommend_tokens.csv'
    ]

    for file_path in files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Filter only numeric columns for average calculation
            numeric_df = df.select_dtypes(include=['number'])
            
            if not numeric_df.empty:
                # Calculate mean for each numeric column
                averages = numeric_df.mean()
                
                # Create the average row dictionary
                avg_row = {col: averages[col] for col in numeric_df.columns}
                avg_row['learnerId'] = 'AVERAGE'
                
                # Append the row
                # We use pd.concat because .append is deprecated in newer pandas versions
                avg_df = pd.DataFrame([avg_row])
                df = pd.concat([df, avg_df], ignore_index=True)
                
                # Save back to the same file
                df.to_csv(file_path, index=False)
                print(f"Successfully added AVERAGE row to {file_path}")
            else:
                print(f"Warning: No numeric columns found in {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")

if __name__ == "__main__":
    append_averages()
