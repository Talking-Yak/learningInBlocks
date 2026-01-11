import os
import pandas as pd
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

class ExerciseGenerator:
    """
    A class for generating language learning exercises using Google's Gemini API.
    
    This class handles the creation of various types of language exercises including
    multiple choice questions, sentence completion, word arrangement, sentence reordering,
    multi-choice selection, and conversational exercises.
    """
    
    def __init__(self):
        """
        Initialize the ExerciseGenerator with API configuration and context prompts.
        
        Sets up the Gemini API client, loads context prompt templates from files,
        and initializes the results dataframe for storing generated exercises.
        
        Raises:
            ValueError: If GOOGLE_API_KEY or GEMINI_API_KEY is not found in environment variables.
        """
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables. Please create a .env file with your API key.")
        
        os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client()
        self.model_name = 'gemini-2.5-flash'
        
        # Load context prompts
        self.context_prompts = {}
        for i in range(1, 7):  # Updated to include context5 and context6
            context_file = f"asset/context{i}.txt"
            with open(context_file, 'r', encoding='utf-8') as f:
                self.context_prompts[i] = f.read()
        
        # Initialize results dataframe
        self.results_df = pd.DataFrame()
        
        # Load existing results if they exist
        self.output_file = "generated_exercises.csv"
        if os.path.exists(self.output_file):
            self.results_df = pd.read_csv(self.output_file)
            print(f"Loaded existing results: {len(self.results_df)} records")
    
    def load_english_content(self):
        """
        Load the English content CSV file containing skill definitions and examples.
        
        Returns:
            pandas.DataFrame: DataFrame containing drill_id, CEFR Level, Skill,
                            Skill Description, and Examples columns.
        """
        return pd.read_csv("english_content.csv")
    
    def prepare_prompt(self, context_num, cefr_level, skill_name, skill_description, examples):
        """
        Prepare a context-specific prompt by replacing placeholders with actual skill data.
        
        Takes a context template and substitutes placeholders with the provided skill
        information to create a complete prompt for the AI model.
        
        Args:
            context_num (int): The context number (1-6) corresponding to exercise type
            cefr_level (str): The CEFR level for the skill (A1, A2, B1, B2, C1, C2)
            skill_name (str): Name of the language skill
            skill_description (str): Detailed description of the skill
            examples (str): Example sentences demonstrating the skill
            
        Returns:
            str: Complete prompt with all placeholders replaced
        """
        prompt = self.context_prompts[context_num]
        
        # Replace placeholders
        prompt = prompt.replace("[CEFR_LEVEL]", str(cefr_level))
        prompt = prompt.replace("[SKILL_NAME]", str(skill_name))
        prompt = prompt.replace("[SKILL_DESCRIPTION]", str(skill_description))
        prompt = prompt.replace("[EXAMPLE_SENTENCES]", str(examples))
        
        return prompt
    
    def generate_exercises(self, prompt, drill_id, context_num, max_retries=3):
        """
        Generate exercises using Gemini API with automatic retry logic and JSON parsing.
        
        Sends a prompt to the Gemini API to generate exercises, handles response parsing,
        and implements retry logic for failed attempts. Expects JSON response format
        with a 'questions' key containing exercise data.
        
        Args:
            prompt (str): The complete prompt to send to the API
            drill_id (str): Identifier for the skill being processed
            context_num (int): Context number for the exercise type
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
            
        Returns:
            list: List of exercise dictionaries, empty list if all attempts fail
        """
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=100)
                    )
                )
                
                # Extract JSON from response
                response_text = response.text.strip()
                
                # Try to find JSON in the response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    # Assume the entire response is JSON
                    json_text = response_text
                
                # Parse JSON
                exercises_data = json.loads(json_text)
                
                # Validate structure
                if "questions" not in exercises_data:
                    raise ValueError("Response missing 'questions' key")
                
                return exercises_data["questions"]
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for drill_id {drill_id}, context {context_num}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"Failed to generate exercises after {max_retries} attempts")
                    return []
                time.sleep(2)  # Wait before retry
        
        return []
    
    def save_exercises_to_dataframe(self, exercises, drill_id, context_num):
        """
        Convert generated exercises to DataFrame format and append to results.
        
        Processes exercise data into a standardized DataFrame format with columns
        for drill_id, context, question details, and context-specific options.
        Different exercise types have different column structures.
        
        Args:
            exercises (list): List of exercise dictionaries from API response
            drill_id (str): Identifier for the skill being processed
            context_num (int): Context number determining exercise type and structure
        """
        rows = []
        
        for exercise in exercises:
            row = {
                'drill_id': drill_id,
                'context': context_num,
                'question_no': exercise.get('question_no', 0),
                'question': exercise.get('question', ''),
                'answer': exercise.get('answer', '')
            }
            
            # Add context-specific columns
            if context_num in [1, 2, 5, 6]:  # MCQ, Completion, Multi-choice selection, and Conversational have options
                row['option1'] = exercise.get('option1', '')
                row['option2'] = exercise.get('option2', '')
                if context_num in [1, 5]:  # MCQ and Multi-choice selection have 4 options
                    row['option3'] = exercise.get('option3', '')
                    row['option4'] = exercise.get('option4', '')
                else:  # Completion and Conversational have 2 options
                    row['option3'] = ''
                    row['option4'] = ''
            else:  # Word arrangement and sentence reordering
                row['option1'] = ''
                row['option2'] = ''
                row['option3'] = ''
                row['option4'] = ''
            
            rows.append(row)
        
        # Convert to DataFrame and append
        new_df = pd.DataFrame(rows)
        if not new_df.empty:
            self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)
    
    def save_to_csv(self):
        """
        Save the current results DataFrame to a CSV file.
        
        Writes all accumulated exercise data to the output CSV file and prints
        a confirmation message with the total number of records saved.
        """
        self.results_df.to_csv(self.output_file, index=False)
        print(f"Saved {len(self.results_df)} total records to {self.output_file}")
    
    def is_already_processed(self, drill_id, context_num):
        """
        Check if a specific drill_id and context combination has already been processed.
        
        Searches the existing results DataFrame to determine if exercises have already
        been generated for the given skill and exercise type combination.
        
        Args:
            drill_id (str): Identifier for the skill
            context_num (int): Context number for the exercise type
            
        Returns:
            bool: True if the combination has been processed, False otherwise
        """
        if self.results_df.empty:
            return False
        
        existing = self.results_df[
            (self.results_df['drill_id'] == drill_id) & 
            (self.results_df['context'] == context_num)
        ]
        return not existing.empty
    
    def process_specific_contexts(self, context_numbers=[5, 6]):
        """
        Process exercises for specific context types only.
        
        Generates exercises for all skills but only for the specified context numbers.
        This allows for targeted processing of specific exercise types without
        regenerating all exercise formats.
        
        Args:
            context_numbers (list, optional): List of context numbers to process.
                                             Defaults to [5, 6] for Multi-choice Selection
                                             and Conversational exercises.
        """
        # Load English content
        content_df = self.load_english_content()
        print(f"Loaded {len(content_df)} skills from english_content.csv")
        
        total_tasks = len(content_df) * len(context_numbers)
        completed_tasks = 0
        
        context_names = {1: "MCQ", 2: "Sentence Completion", 3: "Word-to-Sentence", 4: "Sentence Reordering", 5: "Multi-choice Selection", 6: "Conversational"}
        context_list = [context_names.get(c, f"Context {c}") for c in context_numbers]
        
        print(f"Total tasks to process: {total_tasks} ({len(context_numbers)} contexts × {len(content_df)} skills)")
        print(f"Exercise types: {', '.join(context_list)}")
        print(f"Using Gemini 2.5 Flash with thinking budget: 100")
        print("=" * 60)
        
        # Process each drill_id through specified contexts
        for _, row in content_df.iterrows():
            drill_id = row['drill_id']
            cefr_level = row['CEFR Level']
            skill_name = row['Skill']
            skill_description = row['Skill Description (<500 words)']
            examples = row['Examples']
            
            # Process through specified contexts only
            for context_num in context_numbers:
                # Skip if already processed
                if self.is_already_processed(drill_id, context_num):
                    print(f"Skipping {drill_id} - Context {context_num} (already processed)")
                    completed_tasks += 1
                    continue
                
                # Show what we're about to process
                context_name = context_names.get(context_num, f"Context {context_num}")
                
                print(f"\nStarting: {drill_id} - {context_name} ({completed_tasks + 1}/{total_tasks})")
                print(f"   Skill: {skill_name}")
                print(f"   Level: {cefr_level}")
                
                # Prepare prompt
                prompt = self.prepare_prompt(
                    context_num, cefr_level, skill_name, skill_description, examples
                )
                
                # Generate exercises
                print(f"   Generating exercises with Gemini 2.5 Flash...")
                exercises = self.generate_exercises(prompt, drill_id, context_num)
                
                if exercises:
                    # Save to dataframe
                    self.save_exercises_to_dataframe(exercises, drill_id, context_num)
                    
                    # Save to CSV after each context completion
                    self.save_to_csv()
                    
                    print(f"   Generated {len(exercises)} exercises for {drill_id} - {context_name}")
                    print(f"   Results saved to CSV (Total records: {len(self.results_df)})")
                else:
                    print(f"   Failed to generate exercises for {drill_id} - {context_name}")
                
                completed_tasks += 1
                
                # Small delay to be respectful to API
                time.sleep(1)
            
            # Summary after completing specified contexts for this drill_id
            drill_exercises = self.results_df[
                (self.results_df['drill_id'] == drill_id) & 
                (self.results_df['context'].isin(context_numbers))
            ]
            if not drill_exercises.empty:
                print(f"\nCompleted {drill_id}: Generated {len(drill_exercises)} exercises for contexts {context_numbers}")
        
        print(f"\nProcessing complete! Generated exercises for {len(content_df)} skills across contexts {context_numbers}.")
        print(f"Total records in final dataset: {len(self.results_df)}")

    def process_all_exercises(self):
        """
        Main processing function to generate exercises for all skills and contexts.
        
        Loads all English content skills and generates exercises for each skill
        across all 6 context types (MCQ, Sentence Completion, Word-to-Sentence,
        Sentence Reordering, Multi-choice Selection, and Conversational).
        Includes progress tracking and automatic saving.
        """
        # Load English content
        content_df = self.load_english_content()
        print(f"Loaded {len(content_df)} skills from english_content.csv")
        
        total_tasks = len(content_df) * 6  # 6 contexts per drill_id
        completed_tasks = 0
        
        print(f"Total tasks to process: {total_tasks} (6 contexts × {len(content_df)} skills)")
        print(f"Exercise types: MCQ, Sentence Completion, Word-to-Sentence, Sentence Reordering, Multi-choice Selection, Conversational")
        print(f"Using Gemini 2.5 Flash with thinking budget: 100")
        print("=" * 60)
        
        # Process each drill_id through all contexts
        for _, row in content_df.iterrows():
            drill_id = row['drill_id']
            cefr_level = row['CEFR Level']
            skill_name = row['Skill']
            skill_description = row['Skill Description (<500 words)']
            examples = row['Examples']
            
            # Process through each context (1-6)
            for context_num in range(1, 7):
                # Skip if already processed
                if self.is_already_processed(drill_id, context_num):
                    print(f"Skipping {drill_id} - Context {context_num} (already processed)")
                    completed_tasks += 1
                    continue
                
                # Show what we're about to process
                context_names = {1: "MCQ", 2: "Sentence Completion", 3: "Word-to-Sentence", 4: "Sentence Reordering", 5: "Multi-choice Selection", 6: "Conversational"}
                context_name = context_names.get(context_num, f"Context {context_num}")
                
                print(f"\nStarting: {drill_id} - {context_name} ({completed_tasks + 1}/{total_tasks})")
                print(f"   Skill: {skill_name}")
                print(f"   Level: {cefr_level}")
                
                # Prepare prompt
                prompt = self.prepare_prompt(
                    context_num, cefr_level, skill_name, skill_description, examples
                )
                
                # Generate exercises
                print(f"   Generating exercises with Gemini 2.5 Flash...")
                exercises = self.generate_exercises(prompt, drill_id, context_num)
                
                if exercises:
                    # Save to dataframe
                    self.save_exercises_to_dataframe(exercises, drill_id, context_num)
                    
                    # Save to CSV after each context completion
                    self.save_to_csv()
                    
                    print(f"   Generated {len(exercises)} exercises for {drill_id} - {context_name}")
                    print(f"   Results saved to CSV (Total records: {len(self.results_df)})")
                else:
                    print(f"   Failed to generate exercises for {drill_id} - {context_name}")
                
                completed_tasks += 1
                
                # Small delay to be respectful to API
                time.sleep(1)
            
            # Summary after completing all contexts for this drill_id
            drill_exercises = self.results_df[self.results_df['drill_id'] == drill_id]
            if not drill_exercises.empty:
                print(f"\nCompleted {drill_id}: Generated {len(drill_exercises)} total exercises across 4 contexts")
        
        print(f"\nProcessing complete! Generated exercises for {len(content_df)} skills across 6 contexts.")
        print(f"Total records in final dataset: {len(self.results_df)}")

def main():
    """
    Main function to run the exercise generator with command-line argument support.
    
    Supports different execution modes:
    - Default: Process all contexts for all skills
    - --contexts X,Y: Process only specified context numbers
    - --context5-6: Process only contexts 5 and 6
    
    Handles keyboard interrupts gracefully and provides error reporting.
    """
    import sys
    
    try:
        generator = ExerciseGenerator()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--contexts" and len(sys.argv) > 2:
                # Parse context numbers from command line
                context_numbers = [int(x) for x in sys.argv[2].split(',')]
                print(f"Running specific contexts: {context_numbers}")
                generator.process_specific_contexts(context_numbers)
            elif sys.argv[1] == "--context5-6":
                print("Running Context 5 and 6 only")
                generator.process_specific_contexts([5, 6])
            else:
                print("Usage: python main.py [--contexts 5,6] [--context5-6]")
                print("  --contexts 5,6    Run specific contexts (comma-separated)")
                print("  --context5-6      Run context 5 and 6 only")
                return
        else:
            # Run all contexts by default
            generator.process_all_exercises()
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()