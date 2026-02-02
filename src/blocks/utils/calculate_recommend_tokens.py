import pandas as pd
import os

def calculate_recommendation_tokens():
    # Define files and their corresponding columns based on their structure
    # Group A: SelfRefine and SelfConsistency
    group_a_files = [
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/MAD/recommendSelfRefine_output.csv',
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/MAD/recommendSelfConsistency_output.csv'
    ]
    group_a_cols = ['tokens_response1', 'tokens_response2', 'tokens_response3']

    # Group B: MAD (homo and hetro)
    group_b_files = [
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/MAD/recommendhomoMAD_output.csv',
        '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/asset/MAD/recommendhetroMAD_output.csv'
    ]
    group_b_cols = ['tokens_a_init', 'tokens_b_init', 'tokens_c_init', 'tokens_a_final', 'tokens_b_final', 'tokens_c_final', 'tokens_judge']

    # Build result dictionary starting with learnerId from the first file (44 rows)
    first_df = pd.read_csv(group_a_files[0])
    result_df = pd.DataFrame({'learnerId': first_df['learnerId']})

    # Process Group A
    for file_path in group_a_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            col_name = os.path.splitext(os.path.basename(file_path))[0]
            # Ensure columns are numeric, filling NaNs with 0
            df[group_a_cols] = df[group_a_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            result_df[col_name] = df[group_a_cols].sum(axis=1)
        else:
            print(f"Warning: File not found: {file_path}")

    # Process Group B
    for file_path in group_b_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            col_name = os.path.splitext(os.path.basename(file_path))[0]
            # Ensure columns are numeric, filling NaNs with 0
            df[group_b_cols] = df[group_b_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            result_df[col_name] = df[group_b_cols].sum(axis=1)
        else:
            print(f"Warning: File not found: {file_path}")

    # Save to the requested filename
    output_path = '/Users/silvester/Documents/Development/Talking_Yak/TYGitFolder/learning_in_blocks/recommend_tokens.csv'
    result_df.to_csv(output_path, index=False)
    print(f"Successfully saved row-by-row recommendation token usage to {output_path}")
    print(f"Output row count: {len(result_df)}")

if __name__ == "__main__":
    calculate_recommendation_tokens()
