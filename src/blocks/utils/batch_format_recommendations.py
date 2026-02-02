import pandas as pd
import json
import re
import os

def load_grammar_mapping(grammar_file_path):
    """
    Loads the grammar skills mapping from the CSV file.
    """
    try:
        df = pd.read_csv(grammar_file_path)
        mapping = pd.Series(df.grammar_skills.values, index=df.skill_no).to_dict()
        return mapping
    except Exception as e:
        print(f"Error loading grammar mapping: {e}")
        return {}

def format_recommendation_markdown(recommendation_str, grammar_mapping):
    """
    Formats the recommendation JSON into the requested Markdown structure.
    """
    if pd.isna(recommendation_str):
        return None

    # Clean the JSON string from Markdown code blocks
    cleaned_res = str(recommendation_str).strip()
    match = re.search(r'```json\s*(.*?)\s*```', cleaned_res, re.DOTALL)
    if match:
        cleaned_res = match.group(1)
    elif cleaned_res.startswith("```") and cleaned_res.endswith("```"):
         cleaned_res = cleaned_res[3:-3].strip()
    
    try:
        data = json.loads(cleaned_res)
        
        grammar_ids = data.get('grammar_skills', [])
        vocab_list = data.get('vocab_skills', [])
        
        # Format Grammar
        grammar_lines = ["GRAMMAR:"]
        for i, gid in enumerate(grammar_ids, 1):
            try:
                # Handle cases where gid might be a string or integer
                skill_id = int(re.search(r'\d+', str(gid)).group()) if re.search(r'\d+', str(gid)) else None
                if skill_id and skill_id in grammar_mapping:
                    gname = grammar_mapping[skill_id]
                    grammar_lines.append(f"{i}. {gname}")
                else:
                    grammar_lines.append(f"{i}. {gid}")
            except:
                grammar_lines.append(f"{i}. {gid}")
        
        # Format Vocab
        vocab_lines = ["VOCABULARY:"]
        for i, vname in enumerate(vocab_list, 1):
            vocab_lines.append(f"{i}. {vname}")
            
        return "\n".join(grammar_lines) + "\n\n___\n\n" + "\n".join(vocab_lines)

    except Exception as e:
        # If it's not valid JSON, just return as is or error
        return f"Error parsing: {str(e)[:50]}"

def process_file(file_path, grammar_mapping):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Processing {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # The key column is 'final_recommendation'
        target_col = 'final_recommendation'
        if target_col not in df.columns:
            print(f"  Column '{target_col}' not found. Skipping.")
            return
            
        df['formatted_recommendation'] = df[target_col].apply(lambda x: format_recommendation_markdown(x, grammar_mapping))
        
        df.to_csv(file_path, index=False)
        print(f"  Saved {file_path} with 'formatted_recommendation' column.")
    except Exception as e:
        print(f"  Error processing {file_path}: {e}")

def main():
    # File paths
    grammar_csv_path = 'asset/grammar_flow.csv'
    files_to_process = [
        'asset/MAD/recommendSelfRefine_output.csv',
        'asset/MAD/recommendSelfConsistency_output.csv',
        # 'asset/MAD/recommendhomoMAD_output.csv',
        # 'asset/MAD/recommendhetroMAD_output.csv'
    ]

    # Load grammar mapping
    print("Loading grammar mapping...")
    grammar_mapping = load_grammar_mapping(grammar_csv_path)

    # Process each file
    for file_path in files_to_process:
        process_file(file_path, grammar_mapping)

    print("All tasks complete.")

if __name__ == "__main__":
    main()
