import os
import pandas as pd
from datasets import load_dataset
import time
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
import re

# Load environment variables
load_dotenv()

class QwenInference:
    """
    A class for generating outputs using Novita AI's Qwen models and verifying with Gemini.
    """
    
    def __init__(self):
        """
        Initialize with API configurations for Novita and Gemini.
        """
        # Configure Novita AI API
        novita_key = os.getenv('NOVITA_API_KEY')
        if not novita_key:
            raise ValueError("NOVITA_API_KEY not found in environment variables.")
        
        self.novita_client = OpenAI(
            api_key=novita_key,
            base_url="https://api.novita.ai/openai"
        )
        
        # Configure Gemini API
        gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found.")
            
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.gemini_model = 'gemini-2.5-flash'
        
        # Model definitions
        self.model_8b = 'qwen/qwen3-8b-fp8'  # As per existing configuration
        
        # Output configuration
        self.output_file = "qwen_direct_inference_think.csv" # Output file name
        self.columns = [
            'gsm8k_question', 
            'qwen3_8b_response', 
            'qwen3_8b_reasoning',
            'gsm8k_answer', 
            'gemini_verdict',
            'completion_tokens'
        ]
        
        # Load existing results
        if os.path.exists(self.output_file):
            self.results_df = pd.read_csv(self.output_file)
            print(f"Loaded existing results: {len(self.results_df)} records")
        else:
            self.results_df = pd.DataFrame(columns=self.columns)
    
    def load_input_data(self):
        """Load GSM8K dataset"""
        print("Loading GSM8K dataset from HuggingFace...")
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        df = dataset.to_pandas()
        return df.rename(columns={'question': 'input'})
    
    def generate_reasoning_32b(self, input_text, max_retries=3):
        """Generate 100-word step-by-step reasoning using 32B model"""
        prompt = f"{input_text}\n\nProvide the logical steps to solve this problem. Do not calculate the final values. Do not provide the answer. Just describe the method. /no_think"
        
        for attempt in range(max_retries):
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.model_32b,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                return f"<think>\n{content}\n</think>"
            except Exception as e:
                print(f"32B Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        return "<think>\nError generating reasoning\n</think>"

    def generate_answer_8b(self, input_text, max_retries=3):
        """Generate final answer using 8B model"""
        for attempt in range(max_retries):
            try:
                response = self.novita_client.chat.completions.create(
                    model=self.model_8b,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{input_text}\n\n Give only the final answer."}
                    ],
                    max_tokens=5000,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                # Try to get reasoning_content from API field
                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', "")
                if not reasoning_content:
                    # Fallback: check if 'reasoning_content' is in model_extra
                     if hasattr(response.choices[0].message, 'model_extra') and response.choices[0].message.model_extra:
                        reasoning_content = response.choices[0].message.model_extra.get('reasoning_content', "")
                
                # If still empty, check for <think> tags in content
                if not reasoning_content and "<think>" in content:
                    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                    if match:
                        reasoning_content = match.group(1).strip()
                        content = content.replace(match.group(0), "")
                    else:
                        # Handle unclosed tag if necessary, or just simple split
                        parts = content.split("<think>")
                        if len(parts) > 1:
                            reasoning_content = parts[1].strip()
                            content = parts[0].strip()

                # Get completion tokens
                completion_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    completion_tokens = response.usage.completion_tokens
                
                clean_content = content.replace("<think>", "").replace("</think>", "").strip()
                return clean_content, reasoning_content, completion_tokens
            except Exception as e:
                print(f"8B Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        return "", "", 0

    def verify_with_gemini(self, question, model_output, ground_truth):
        """Verify the answer using Gemini"""
        prompt = f"""
        Compare the model's response with the correct answer.
        
        Question: {question}
        Model Response: {model_output}
        Correct Answer: {ground_truth}
        
        Is the model's response correct? Reply with only CORRECT or INCORRECT.
        """
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt
            )
            return response.text.strip().upper()
        except Exception as e:
            print(f"Gemini verification failed: {e}")
            return "ERROR"

    def process_all_inputs(self):
        input_df = self.load_input_data()
        print(f"Loaded {len(input_df)} inputs from GSM8K")
        
        total_tasks = len(input_df)
        completed = 0
        
        for idx, row in input_df.iterrows():
            question = row['input']
            ground_truth = row['answer']
            
            # Check if processed
            # We check if this specific question exists in our results
            if not self.results_df.empty and question in self.results_df['gsm8k_question'].values:
                print(f"Skipping {idx + 1}/{total_tasks} (already processed)")
                completed += 1
                continue
                
            print(f"\nProcessing {idx + 1}/{total_tasks}")
            print(f"Question: {question[:50]}...")
            
            # Step 1: Answering (8B)
            print("Generating answer (8B)...")
            answer_8b, reasoning_8b, tokens_8b = self.generate_answer_8b(question)
            
            # Step 2: Verification (Gemini)
            print("Verifying with Gemini...")
            # Gemini verifies the actual content, not reasoning
            verdict = self.verify_with_gemini(question, answer_8b, ground_truth)
            print(f"Verdict: {verdict}")
            
            # Save
            new_row = pd.DataFrame([{
                'gsm8k_question': question,
                'qwen3_8b_response': answer_8b,
                'qwen3_8b_reasoning': reasoning_8b,
                'gsm8k_answer': ground_truth,
                'gemini_verdict': verdict,
                'completion_tokens': tokens_8b
            }])
            
            self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
            self.results_df.to_csv(self.output_file, index=False)
            print(f"Saved progress to {self.output_file}")
            
            completed += 1
            time.sleep(1)

def main():
    try:
        generator = QwenInference()
        generator.process_all_inputs()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()