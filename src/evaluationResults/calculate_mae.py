import csv
import json
import re

file_path = "asset/forReviewDOV.csv"

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

dimensions = ["grammar", "vocabulary", "interactive_communication"]
comparison_columns = ["selfRefineNT", "selfRefineWT", "selfConsistencyNT", "selfConsistencyWT", "homoMAD", "hetroMAD"]
results = {col: {dim: {"sum_ae": 0.0, "count": 0} for dim in dimensions} for col in comparison_columns}

with open(file_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gt_data = parse_json_safely(row.get("GroundTruth", ""))
        
        for col in comparison_columns:
            pred_data = parse_json_safely(row.get(col, ""))
            
            for dim in dimensions:
                gt_val = gt_data.get(dim)
                pred_val = pred_data.get(dim)
                
                if gt_val is not None and pred_val is not None:
                    try:
                        ae = abs(float(gt_val) - float(pred_val))
                        results[col][dim]["sum_ae"] += ae
                        results[col][dim]["count"] += 1
                    except (ValueError, TypeError):
                        continue

print("--- Mean Absolute Error (MAE) Analysis ---")
print("Ground Truth: \n")

# Print headers
header_str = f"{'Dimension':<25}"
for col in comparison_columns:
    header_str += f" | {col:<17}"
print(header_str)
print("-" * len(header_str))

for dim in dimensions:
    dim_name = dim.replace('_', ' ').title()
    row_str = f"{dim_name:<25}"
    for col in comparison_columns:
        stats = results[col][dim]
        if stats["count"] > 0:
            mae = stats["sum_ae"] / stats["count"]
            row_str += f" | {mae:<17.4f}"
        else:
            row_str += f" | {'N/A':<17}"
    print(row_str)

# Calculate Macro-MAE (average across dimensions)
print("-" * len(header_str))
macro_row = f"{'Macro-MAE (Average)':<25}"
for col in comparison_columns:
    col_sum_ae = sum(results[col][dim]["sum_ae"] for dim in dimensions)
    col_count = sum(results[col][dim]["count"] for dim in dimensions)
    if col_count > 0:
        macro_mae = col_sum_ae / col_count
        macro_row += f" | {macro_mae:<17.4f}"
    else:
        macro_row += f" | {'N/A':<17}"
print(macro_row)
