import pandas as pd
import json
import re
import os

def load_grammar_mapping(grammar_file_path):
    """
    Loads the grammar skills mapping from the CSV file.
    Returns a dictionary mapping skill_no (int) to grammar_skills (str).
    """
    try:
        df = pd.read_csv(grammar_file_path)
        # Create map of skill_no -> grammar_skills
        mapping = pd.Series(df.grammar_skills.values, index=df.skill_no).to_dict()
        return mapping
    except Exception as e:
        print(f"Error loading grammar mapping: {e}")
        return {}

def clean_json_string(s):
    """
    Cleans the JSON string from Markdown code blocks or other artifacts.
    """
    if pd.isna(s):
        return None
    s = str(s).strip()
    # Remove markdown code blocks if present
    match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
    if match:
        s = match.group(1)
    elif s.startswith("```") and s.endswith("```"):
         s = s[3:-3].strip()
    
    return s

def format_recommendation_markdown(row, grammar_mapping):
    """
    Formats the recommendation JSON into the requested Markdown structure.
    """
    # Prefer qwen_judge_response, then final_recommendation
    response_str = row.get('qwen_judge_response')
    if pd.isna(response_str):
        response_str = row.get('final_recommendation')
    
    if pd.isna(response_str):
        return "GRAMMAR:\n\n___\n\nVOCABULARY:"

    cleaned_response = clean_json_string(response_str)
    
    try:
        data = json.loads(cleaned_response)
        
        grammar_ids = data.get('grammar_skills', [])
        vocab_list = data.get('vocab_skills', [])
        
        # Format Grammar
        grammar_lines = ["GRAMMAR:"]
        for i, gid in enumerate(grammar_ids, 1):
            try:
                gname = grammar_mapping.get(int(gid), f"Unknown ID {gid}")
                grammar_lines.append(f"{i}. {gname}")
            except:
                grammar_lines.append(f"{i}. {gid}")
        
        # Format Vocab
        vocab_lines = ["VOCABULARY:"]
        for i, vname in enumerate(vocab_list, 1):
            vocab_lines.append(f"{i}. {vname}")
            
        return "\n".join(grammar_lines) + "\n\n___\n\n" + "\n".join(vocab_lines)

    except Exception as e:
        return "GRAMMAR:\n\n___\n\nVOCABULARY:"

def main():
    # File paths
    input_csv_path = 'asset/Misc/recommendMAD_output2.csv'
    grammar_csv_path = 'asset/grammar_flow.csv'
    output_csv_path = 'asset/Misc/recommendMAD_formatted.csv'

    # Load grammar mapping
    print("Loading grammar mapping...")
    grammar_mapping = load_grammar_mapping(grammar_csv_path)

    # Load input CSV
    print(f"Loading input CSV from {input_csv_path}...")
    if not os.path.exists(input_csv_path):
        print(f"File not found: {input_csv_path}")
        return
    
    df = pd.read_csv(input_csv_path)

    # Apply formatting
    print("Formatting recommendations...")
    output_df = pd.DataFrame()
    
    if 'learnerId' in df.columns:
        output_df['learnerId'] = df['learnerId']
    
    output_df['formatted_recommendation'] = df.apply(lambda row: format_recommendation_markdown(row, grammar_mapping), axis=1)

    # Save to CSV
    print(f"Saving output to {output_csv_path}...")
    output_df.to_csv(output_csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
