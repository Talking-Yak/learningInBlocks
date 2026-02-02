import pandas as pd
import os

def calculate_token_usage():
    # Define file paths and their respective token columns
    group1_files = [
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/selfRefine/selfRefineWT.csv',
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/selfRefine/selfRefineNT.csv',
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/selfConsistency/selfConsistencyNT.csv',
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/selfConsistency/selfConsistencyWT.csv'
    ]
    group1_cols = ['tokens_response1', 'tokens_response2', 'tokens_response3', 'tokens_feedback']

    group2_files = [
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/MAD/hetroMAD/homoMAD_output.csv',
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/MAD/hetroMAD/hetroMAD_output.csv'
    ]
    group2_cols = ['tokens_a_init', 'tokens_b_init', 'tokens_c_init', 'tokens_a_final', 'tokens_b_final', 'tokens_c_final', 'tokens_judge']

    # We'll build a result dictionary starting with the learnerId from the first file
    # Since all files are confirmed to have the same order and count (44 rows)
    first_df = pd.read_csv(group1_files[0])
    result_df = pd.DataFrame({'learnerId': first_df['learnerId']})

    # Process Group 1 (SelfRefine and SelfConsistency)
    for file_path in group1_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            col_name = os.path.splitext(os.path.basename(file_path))[0]
            # Ensure columns are numeric, filling NaNs with 0
            df[group1_cols] = df[group1_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            # Add to result_df directly since orders match
            result_df[col_name] = df[group1_cols].sum(axis=1)
        else:
            print(f"Warning: File not found: {file_path}")

    # Process Group 2 (MAD output files)
    for file_path in group2_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            col_name = os.path.splitext(os.path.basename(file_path))[0]
            # Ensure columns are numeric, filling NaNs with 0
            df[group2_cols] = df[group2_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            # Add to result_df directly since orders match
            result_df[col_name] = df[group2_cols].sum(axis=1)
        else:
            print(f"Warning: File not found: {file_path}")

    # Save to the requested filename
    output_path = '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/scores_tokens.csv'
    result_df.to_csv(output_path, index=False)
    print(f"Successfully saved row-by-row token usage to {output_path}")
    print(f"Output row count: {len(result_df)}")

if __name__ == "__main__":
    calculate_token_usage()
