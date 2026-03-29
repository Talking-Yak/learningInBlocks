import pandas as pd
import numpy as np
from scipy import stats

# Load your data
cohort1 = pd.read_csv('asset/FinalAIEDData-Cohort1.csv')
cohort2 = pd.read_csv('asset/FinalAIEDData-Cohort2.csv')  
cohort3 = pd.read_csv('asset/FinalAIEDData-Cohort3.csv')

def analyze_cohort(df, name):
    dimensions = ['Gram', 'Vocab', 'Conv']
    results = []
    
    for dim in dimensions:
        week2 = df[f'Week2_{dim}']
        week8 = df[f'Week8_{dim}']
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(week8, week2)
        
        # Cohen's d (paired)
        diff = week8 - week2
        d = diff.mean() / diff.std()
        
        # Mean gain and CI
        mean_gain = diff.mean()
        ci = stats.t.interval(0.95, len(diff)-1, 
                             loc=mean_gain, 
                             scale=stats.sem(diff))
        
        results.append({
            'Cohort': name,
            'Dimension': dim,
            'Mean_Gain': f"{mean_gain:.2f}",
            'CI_95': f"[{ci[0]:.2f}, {ci[1]:.2f}]",
            't': f"{t_stat:.2f}",
            'p': f"{p_val:.2e}",
            "Cohen's d": f"{d:.2f}"
        })
    
    return pd.DataFrame(results)

# Run for all cohorts
results = pd.concat([
    analyze_cohort(cohort1, 'C1 (Self Consistency - Score & Feedback)'),
    analyze_cohort(cohort2, 'C2 (HeteroMAD - Score & Feedback)'),
    analyze_cohort(cohort3, 'C3 (Learning in Blocks)')
])

print("=== Within-Cohort Paired t-tests ===")
print(results.to_string(index=False))
print("\n")

# Calculate gain scores for between-cohort analysis
cohort1['Gram_gain'] = cohort1['Week8_Gram'] - cohort1['Week2_Gram']
cohort1['Vocab_gain'] = cohort1['Week8_Vocab'] - cohort1['Week2_Vocab']
cohort1['Conv_gain'] = cohort1['Week8_Conv'] - cohort1['Week2_Conv']

cohort2['Gram_gain'] = cohort2['Week8_Gram'] - cohort2['Week2_Gram']
cohort2['Vocab_gain'] = cohort2['Week8_Vocab'] - cohort2['Week2_Vocab']
cohort2['Conv_gain'] = cohort2['Week8_Conv'] - cohort2['Week2_Conv']

cohort3['Gram_gain'] = cohort3['Week8_Gram'] - cohort3['Week2_Gram']
cohort3['Vocab_gain'] = cohort3['Week8_Vocab'] - cohort3['Week2_Vocab']
cohort3['Conv_gain'] = cohort3['Week8_Conv'] - cohort3['Week2_Conv']

print("=== Between-Cohort Comparisons (ANOVA on Gain Scores) ===")
dimensions = ['Gram', 'Vocab', 'Conv']

for dim in dimensions:
    # ANOVA
    f_stat, p_anova = stats.f_oneway(
        cohort1[f'{dim}_gain'],
        cohort2[f'{dim}_gain'],
        cohort3[f'{dim}_gain']
    )
    
    print(f"\n{dim}:")
    print(f"  ANOVA: F(2,177) = {f_stat:.2f}, p = {p_anova:.2e}")
    
    # Post-hoc pairwise comparisons
    print("  Post-hoc pairwise t-tests:")
    
    # C3 vs C1
    t31, p31 = stats.ttest_ind(cohort3[f'{dim}_gain'], cohort1[f'{dim}_gain'])
    d31 = (cohort3[f'{dim}_gain'].mean() - cohort1[f'{dim}_gain'].mean()) / \
          np.sqrt((cohort3[f'{dim}_gain'].std()**2 + cohort1[f'{dim}_gain'].std()**2) / 2)
    print(f"    C3 vs C1: t = {t31:.2f}, p = {p31:.2e}, d = {d31:.2f}")
    
    # C3 vs C2
    t32, p32 = stats.ttest_ind(cohort3[f'{dim}_gain'], cohort2[f'{dim}_gain'])
    d32 = (cohort3[f'{dim}_gain'].mean() - cohort2[f'{dim}_gain'].mean()) / \
          np.sqrt((cohort3[f'{dim}_gain'].std()**2 + cohort2[f'{dim}_gain'].std()**2) / 2)
    print(f"    C3 vs C2: t = {t32:.2f}, p = {p32:.2e}, d = {d32:.2f}")
    
    # C2 vs C1
    t21, p21 = stats.ttest_ind(cohort2[f'{dim}_gain'], cohort1[f'{dim}_gain'])
    d21 = (cohort2[f'{dim}_gain'].mean() - cohort1[f'{dim}_gain'].mean()) / \
          np.sqrt((cohort2[f'{dim}_gain'].std()**2 + cohort1[f'{dim}_gain'].std()**2) / 2)
    print(f"    C2 vs C1: t = {t21:.2f}, p = {p21:.2e}, d = {d21:.2f}")