import os
import pandas as pd
import json
import time
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

class RecommendationComparator:
    def __init__(self):
        # Configure Gemini API
        gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
            
        self.client = genai.Client(api_key=gemini_key)
        
        self.model = 'gemini-3-flash-preview' # Commented out as requested
        # self.model = 'gemini-2.5-flash'
        self.thinking_budget = 150

        # File path
        self.csv_path = "asset/recommended.csv"

        # Load data
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Models to compare
        self.comparison_tasks = {
            'selfRefine': 'selfRefine | Recommendations',
            'selfConsistency': 'selfConsistency | Recommendations',
            'homoMAD': 'homoMAD | Recommendations',
            'hetroMAD': 'hetroMAD | Recommendations'
        }
        
        # New columns for match counts
        self.score_cols = {
            'selfRefine': 'selfRefine_MatchCount',
            'selfConsistency': 'selfConsistency_MatchCount',
            'homoMAD': 'homoMAD_MatchCount',
            'hetroMAD': 'hetroMAD_MatchCount'
        }

        # Initialize columns if they don't exist
        for col in list(self.comparison_tasks.keys()) + list(self.score_cols.values()):
            if col not in self.df.columns:
                self.df[col] = None
                self.df[col] = self.df[col].astype('object')

    def call_gemini(self, prompt):
        """
        Calls Gemini with retry logic: 10s, 20s, 30s...
        Returns (LABEL, SCORE)
        """
        wait_time = 10
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget)
                    )
                )
                
                if response.text:
                    # Look for boxed tag with label and score
                    # Expected format: \boxed{LABEL, SCORE} or \boxed{LABEL}, Score: \boxed{SCORE}
                    # We will try to extract LABEL and then SCORE from the text.
                    
                    label_match = re.search(r'\\boxed\{(ACCEPTABLE|UNACCEPTABLE)\}', response.text, re.IGNORECASE)
                    score_match = re.search(r'(score|count):\s*(\d)', response.text, re.IGNORECASE)
                    
                    if not score_match:
                        # try looking inside \boxed if score is there
                        score_match = re.search(r'\\boxed\{.*?(?:,\s*|score:\s*)(\d)\}', response.text, re.IGNORECASE)

                    label = f"\\boxed{{{label_match.group(1).upper()}}}" if label_match else None
                    score = int(score_match.group(score_match.lastindex)) if score_match else None
                    
                    if label and score is not None:
                        return label, score
                    
                    # Fallback parsing
                    if not label:
                        if "ACCEPTABLE" in response.text.upper() and "UNACCEPTABLE" not in response.text.upper():
                            label = "\\boxed{ACCEPTABLE}"
                        elif "UNACCEPTABLE" in response.text.upper():
                            label = "\\boxed{UNACCEPTABLE}"
                    
                    if label and score is not None:
                        return label, score

                print(f"  - Attempt {attempt} failed (Empty or unparsable response: {response.text[:50] if response.text else 'None'}). Retrying in {wait_time}s...")
            except Exception as e:
                print(f"  - Attempt {attempt} failed with error: {e}. Retrying in {wait_time}s...")
            
            time.sleep(wait_time)
            wait_time += 10

    def process(self):
        total_rows = len(self.df)
        print(f"Processing {total_rows} rows...")

        for index, row in self.df.iterrows():
            consensus = row.get('Consensus Responses')
            if pd.isna(consensus):
                continue

            print(f"\n[{index+1}/{total_rows}] Learner: {row.get('learnerId', 'Unknown')}")

            # Process each model comparison one by one
            for target_col, source_col in self.comparison_tasks.items():
                score_col = self.score_cols[target_col]
                
                # Skip if already fully done (label and score)
                if pd.notna(row.get(target_col)) and pd.notna(row.get(score_col)):
                    continue

                rec_to_check = row.get(source_col)
                if pd.isna(rec_to_check):
                    continue

                print(f"  Evaluating {target_col}...")
                
                prompt = f"""
I want you to compare a Set of Recommendations against a Ground Truth Consensus.
Each set contains 2 Grammar skills and 2 Vocabulary skills.

Consensus Recommendation (Ground Truth):
{consensus}

Recommendation Set to Verify:
{rec_to_check}

Rule:
- Count how many of the 4 items (2 Grammar, 2 Vocabulary) in the Set match items in the Consensus Recommendation.
- Matching is based on semantic similarity (same skill or topic name/ID), not exact phrasing.
- If at least 3 out of the 4 items match, the label is "ACCEPTABLE".
- Otherwise, the label is "UNACCEPTABLE".

Format:
Strictly return your answer in this format:
Label: \\boxed{{LABEL}}
Score: [NUMBER_OF_MATCHES]
"""
                
                label, score = self.call_gemini(prompt)
                self.df.at[index, target_col] = label
                self.df.at[index, score_col] = score
                
                # Save after each successful model evaluation
                self.df.to_csv(self.csv_path, index=False)
                print(f"    -> Result: {label}, Score: {score}")
                
                # Rate limit safety
                time.sleep(0.5)

        print("\nAll processing complete.")

if __name__ == "__main__":
    comparator = RecommendationComparator()
    comparator.process()
