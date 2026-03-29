import csv
import json
import re
from sklearn.metrics import cohen_kappa_score

file_path = "asset/forReview.csv"

def parse_json_safely(text):
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.strip()
        cleaned = re.sub(r'"+', '"', cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return {}

# Store sequences of scores for Kappa calculation
data_sequences = {
    "grammar": {"Reviewer1": [], "Reviewer2": []},
    "vocabulary": {"Reviewer1": [], "Reviewer2": []},
    "interactive_communication": {"Reviewer1": [], "Reviewer2": []}
}

with open(file_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        reviewer1_text = row.get("Reviewer1Comment", "")
        reviewer2_text = row.get("Reviewer2Comment", "")
        
        reviewer1_data = parse_json_safely(reviewer1_text)
        reviewer2_data = parse_json_safely(reviewer2_text)
        
        for key in data_sequences.keys():
            r1_val = reviewer1_data.get(key)
            r2_val = reviewer2_data.get(key)
            
            if r1_val is not None and r2_val is not None:
                data_sequences[key]["Reviewer1"].append(float(r1_val))
                data_sequences[key]["Reviewer2"].append(float(r2_val))

print("--- Inter-Rater Reliability Analysis ---")
for key, sequences in data_sequences.items():
    r1_list = sequences["Reviewer1"]
    r2_list = sequences["Reviewer2"]
    
    if len(r1_list) > 0:
        # Percentage Agreement
        matches = sum(1 for c, n in zip(r1_list, r2_list) if c == n)
        total = len(c_list)
        agreement = (matches / total) * 100
        
        # Quadratic Weighted Kappa
        # Note: We use all ratings present in the data to define the labels
        all_labels = sorted(list(set(c_list + n_list)))
        qwk = cohen_kappa_score(c_list, n_list, labels=all_labels, weights='quadratic')
        
        category_name = key.replace('_', ' ').title()
        print(f"[{category_name}]")
        print(f"  Percentage Agreement: {agreement:.2f}% ({matches}/{total})")
        print(f"  Quadratic Weighted Kappa: {qwk:.4f}")
        print()
    else:
        print(f"[{key.replace('_', ' ').title()}]: No data found.")
