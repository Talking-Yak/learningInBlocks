import os
import pandas as pd
import json
import time
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
cwd = os.getcwd()
print(f"Current Working Directory: {cwd}")
env_path = os.path.join(cwd, '.env')
if os.path.exists(env_path):
    print(f".env file found at {env_path}")
else:
    print(".env file NOT found.")

load_dotenv(dotenv_path=env_path)

class QwenJudgeProcessor:
    def __init__(self):
        # Configure Novita AI API
        novita_key = os.getenv('NOVITA_API_KEY')
        if not novita_key:
            raise ValueError("NOVITA_API_KEY not found in environment variables.")
        
        self.novita_client = OpenAI(
            api_key=novita_key,
            base_url="https://api.novita.ai/openai"
        )
        
        self.judge_model = "qwen/qwen3-30b-a3b-fp8"

        # File paths
        self.input_csv_path = "asset/Misc/recommendMAD_output.csv"
        self.output_csv_path = "asset/Misc/recommendMAD_output2.csv"
        
        self.prompt_dir = "asset/MAD/recommendMAD/prompt/"
        
        self.load_prompts()
        self.load_grammar_mapping()
        self.df = self.initialize_data()

    def load_prompts(self):
        with open(f"{self.prompt_dir}judge.txt", 'r') as f:
            self.judge_prompt_template = f.read()

    def load_grammar_mapping(self):
        grammar_csv_path = 'asset/grammar_flow.csv'
        try:
            df = pd.read_csv(grammar_csv_path)
            self.grammar_mapping = pd.Series(df.grammar_skills.values, index=df.skill_no).to_dict()
        except Exception as e:
            print(f"Error loading grammar mapping: {e}")
            self.grammar_mapping = {}

    def format_recommendation_markdown(self, qwen_json):
        if not qwen_json:
            return None
        
        grammar_ids = qwen_json.get('grammar_skills', [])
        vocab_list = qwen_json.get('vocab_skills', [])
        
        # Format Grammar
        grammar_lines = ["GRAMMAR:"]
        for i, gid in enumerate(grammar_ids, 1):
            try:
                # Handle potential string IDs
                skill_id = int(re.search(r'\d+', str(gid)).group()) if re.search(r'\d+', str(gid)) else None
                if skill_id and skill_id in self.grammar_mapping:
                    gname = self.grammar_mapping[skill_id]
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

    def initialize_data(self):
        if os.path.exists(self.output_csv_path):
            print(f"Resuming from {self.output_csv_path}...")
            df = pd.read_csv(self.output_csv_path)
            # Ensure all columns exist
            new_cols = ['final_recommendation', 'qwen_judge_response', 'tokens_judge', 'formatted_recommendation']
            for col in new_cols:
                if col not in df.columns:
                    df[col] = None
                    df[col] = df[col].astype('object')
            return df
        
        if not os.path.exists(self.input_csv_path):
            raise FileNotFoundError(f"Input CSV not found at {self.input_csv_path}")
        
        print(f"Loading data from {self.input_csv_path}...")
        df = pd.read_csv(self.input_csv_path)
        
        # We will replace/add these columns
        new_cols = ['final_recommendation', 'qwen_judge_response', 'tokens_judge', 'formatted_recommendation']
        for col in new_cols:
            df[col] = None
            df[col] = df[col].astype('object')
            
        gemini_cols = ['gemini_judge_response']
        for col in gemini_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                
        return df

    def call_judge(self, prompt, task_name="Qwen Judge"):
        print(f"  - Requesting {task_name}...")
        attempt = 0
        wait_time = 10
        while True:
            attempt += 1
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert assistant. You must output valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=3000,
                    temperature=0.1,
                    response_format={ "type": "json_object" }
                )
                
                raw_text = response.choices[0].message.content
                # Remove <think>...</think> blocks if present
                raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                
                total_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    total_tokens = response.usage.completion_tokens
                
                return json.loads(raw_text), raw_text, total_tokens
            except Exception as e:
                error_msg = str(e)
                print(f"  - Qwen Judge Error (Attempt {attempt}): {error_msg}")
                
                # Check for rate limit or retryable errors
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    print(f"    Rate limit hit. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    wait_time += 10 # Increase wait time by 10s for next try
                else:
                    # For other errors, maybe wait a bit and retry too, or just wait the same amount
                    print(f"    Unexpected error. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    wait_time += 10

    def process(self):
        total_rows = len(self.df)
        print(f"Processing {total_rows} rows...")
        
        # Columns required for judge
        required_cols = ['rec_a_final', 'rec_b_final', 'rec_c_final', 'conversationHistoryCleaned']
        
        for index, row in self.df.iterrows():
            # Skip already processed
            if pd.notna(row['final_recommendation']):
                continue

            # Basic validation
            if any(pd.isna(row[col]) for col in required_cols):
                print(f"Skipping row {index}: Missing required agent responses or transcript.")
                continue

            print(f"\nProcessing Row {index+1}/{total_rows} (learnerId: {row['learnerId']})")
            
            transcript = row['conversationHistoryCleaned']
            res_a_final = row['rec_a_final']
            res_b_final = row['rec_b_final']
            res_c_final = row['rec_c_final']
            
            p_judge = self.judge_prompt_template.replace("{agent_a_final}", str(res_a_final))\
                                                .replace("{agent_b_final}", str(res_b_final))\
                                                .replace("{agent_c_final}", str(res_c_final))\
                                                .replace("{transcript}", str(transcript))
            
            qwen_json, raw_text, qwen_tokens = self.call_judge(p_judge, "Qwen Consensus")
            
            if qwen_json:
                print(f"   - Recommendation Found.")
                self.df.at[index, 'final_recommendation'] = json.dumps(qwen_json)
                self.df.at[index, 'qwen_judge_response'] = raw_text
                self.df.at[index, 'tokens_judge'] = qwen_tokens
                self.df.at[index, 'formatted_recommendation'] = self.format_recommendation_markdown(qwen_json)
            else:
                print("   - Failed to get consensus from Qwen.")
            
            # Save progress periodically or every row
            self.df.to_csv(self.output_csv_path, index=False)
            print(f"  - Progress saved to {self.output_csv_path}")

        print("\nProcessing Complete.")

if __name__ == "__main__":
    processor = QwenJudgeProcessor()
    processor.process()
