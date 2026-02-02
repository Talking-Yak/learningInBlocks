import pandas as pd
import re
import os

def extract_items(text):
    """
    Extracts numbered items from Grammar and Vocabulary sections.
    Normalizes them by lowercasing and stripping whitespace.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # Check for the specific "no values" pattern provided by the user
    # "GRAMMAR:\n\n___\n\nVOCABULARY:"
    # We'll use a regex to be flexible with whitespace
    if re.search(r'GRAMMAR:\s*___\s*VOCABULARY:', text, re.IGNORECASE | re.DOTALL):
        return []

    items = []
    lines = text.split('\n')
    for line in lines:
        # Match lines starting with "1. " or "2. " etc.
        # We allow some leading whitespace
        match = re.search(r'^\s*\d+\.\s*(.*)', line)
        if match:
            # Extract content and normalize
            item = match.group(1).strip().lower()
            # Further normalization: remove extra spaces, trailing punctuation if any
            item = re.sub(r'\s+', ' ', item)
            # Remove trailing periods or commas
            item = item.rstrip('.,')
            if item:
                items.append(item)
    return items

def calculate_matches(consensus_items, model_items):
    """
    Calculates the number of matches between model items and consensus items.
    Uses basic string matching (or substring matching for robustness).
    """
    if not consensus_items or not model_items:
        return 0
    
    match_count = 0
    # Create a set of consensus items for faster lookup
    consensus_set = set(consensus_items)
    
    # Track used consensus items to avoid double matching
    used_consensus = set()
    
    for item in model_items:
        # Check for exact match first
        if item in consensus_set:
            match_count += 1
            used_consensus.add(item)
        else:
            # Fallback: check if the item is semantically similar via substring
            for c_item in consensus_set:
                if c_item in used_consensus:
                    continue
                if item in c_item or c_item in item:
                    # To avoid over-matching (e.g., "a" matching "articles"), 
                    # we only count if length is significant or it's a very close match.
                    if len(item) > 3 and len(c_item) > 3:
                        match_count += 1
                        used_consensus.add(c_item)
                        break
    
    return match_count

def main():
    csv_path = "recommended.csv"
    output_path = "recommended_regex_scored.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    comparison_tasks = {
        'selfRefine WT': 'selfRefine WT | Recommendations',
        'selfConsistency WT': 'selfConsistency WT | Recommendations'
    }
    
    for model_key, source_col in comparison_tasks.items():
        score_col = f"{model_key}_MatchCount"
        label_col = model_key
        
        print(f"Comparing {source_col}...")
        
        match_counts = []
        labels = []
        
        for idx, row in df.iterrows():
            consensus = row.get('Consensus Responses')
            model_rec = row.get(source_col)
            
            if pd.isna(consensus) or pd.isna(model_rec):
                match_counts.append(0)
                labels.append("\\boxed{UNACCEPTABLE}")
                continue
            
            c_items = extract_items(consensus)
            m_items = extract_items(model_rec)
            
            # Specific requirement 3: If no values for Grammar and Vocabulary, default is UNACCEPTABLE
            # Our extract_items returns [] for the empty pattern.
            if not m_items:
                match_counts.append(0)
                labels.append("\\boxed{UNACCEPTABLE}")
                continue

            matches = calculate_matches(c_items, m_items)
            
            # Requirement 2: look for at least 2 matches. Then ACCEPTABLE else UNACCEPTABLE.
            label = "\\boxed{ACCEPTABLE}" if matches >= 2 else "\\boxed{UNACCEPTABLE}"
            
            match_counts.append(matches)
            labels.append(label)
            
        df[score_col] = match_counts
        df[label_col] = labels

    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
