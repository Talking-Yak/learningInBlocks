import os
import pandas as pd
from datasets import load_dataset
import time
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

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
        self.model_32b = 'qwen/qwen3-32b-fp8' # Attempting to match the 8B naming pattern
        
        # Output configuration
        self.output_file = "qwen_gsm8k_output.csv"
        self.columns = [
            'gsm8k_question', 
            'qwen3_32b_response', 
            'qwen3_8b_response', 
            'gsm8k_answer', 
            'gemini_verdict'
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
                # Clean tags if model outputted them to avoid duplicates later
                clean_content = content.replace("<think>", "").replace("</think>", "").strip()
                return clean_content
            except Exception as e:
                print(f"32B Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        return "<think>\nError generating reasoning\n</think>"

    def generate_answer_8b(self, input_text, reasoning=None, max_retries=3):
        """Generate final answer using 8B model, optionally using reasoning from 32B"""
        for attempt in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{input_text}. Give only the final answer. /no_think"}
                ]
                
                if reasoning:
                    # Clean reasoning just in case, then wrap
                    clean_reasoning = reasoning.replace("<think>", "").replace("</think>", "").strip()
                    messages.append({"role": "assistant", "content": f"<think>\n{clean_reasoning}\n</think>"})

                response = self.novita_client.chat.completions.create(
                    model=self.model_8b,
                    messages=messages,
                    max_tokens=5000,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                return content.replace("<think>", "").replace("</think>", "").strip()
            except Exception as e:
                print(f"8B Attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        return ""

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
            
            while True:
                try:
                    # Step 1: Reasoning (32B)
                    print("Generating reasoning (32B)...")
                    reasoning = self.generate_reasoning_32b(question)
                    if "Error generating reasoning" in reasoning:
                        raise Exception("Reasoning generation failed")
                    
                    # Step 2: Answering (8B)
                    print("Generating answer (8B)...")
                    answer_8b = self.generate_answer_8b(question, reasoning)
                    if not answer_8b:
                        raise Exception("Answer generation failed")
                    
                    # Step 3: Verification (Gemini)
                    print("Verifying with Gemini...")
                    verdict = self.verify_with_gemini(question, answer_8b, ground_truth)
                    if verdict == "ERROR":
                        raise Exception("Verification failed")
                    
                    # Save
                    new_row = pd.DataFrame([{
                        'gsm8k_question': question,
                        'qwen3_32b_response': reasoning,
                        'qwen3_8b_response': answer_8b,
                        'gsm8k_answer': ground_truth,
                        'gemini_verdict': verdict
                    }])
                    
                    self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
                    self.results_df.to_csv(self.output_file, index=False)
                    print(f"Saved progress to {self.output_file}")
                    
                    completed += 1
                    time.sleep(1)
                    break # Success, move to next row
                    
                except Exception as e:
                    print(f"Error processing row {idx + 1}: {e}")
                    print("Waiting 5 minutes before retrying...")
                    time.sleep(300)

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