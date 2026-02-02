import pandas as pd
import os

def fix_labels():
    csv_path = "asset/recommended.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    print(f"Fixing labels in {csv_path}...")
    df = pd.read_csv(csv_path)

    models = ['selfRefine', 'selfConsistency', 'homoMAD', 'hetroMAD']
    
    for model in models:
        score_col = f"{model}_MatchCount"
        if model in df.columns and score_col in df.columns:
            # Apply the rule: Score >= 3 -> ACCEPTABLE, else UNACCEPTABLE
            # We use a lambda to handle potential NaN scores
            def apply_rule(score):
                if pd.isna(score):
                    return None
                try:
                    s = int(score)
                    return "ACCEPTABLE" if s >= 3 else "UNACCEPTABLE"
                except:
                    return None

            df[model] = df[score_col].apply(apply_rule)
            print(f"  Fixed labels for {model}")

    df.to_csv(csv_path, index=False)
    print("Done. All labels updated based on MatchCount.")

if __name__ == "__main__":
    fix_labels()
