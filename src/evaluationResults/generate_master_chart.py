import matplotlib.pyplot as plt
import numpy as np

# Set global parameters for high quality
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.unicode_minus'] = False

# Data Setup
bins = ["<70", "75", "80", "85", "90", "95", "100"]
bin_centers = np.arange(len(bins))

data = {
    "CG1": {
        "W1": {"Grammar": [32, 15, 8, 3, 2, 0, 0], "Vocabulary": [7, 3, 9, 6, 11, 11, 13], "Communication": [14, 9, 13, 8, 4, 10, 2]},
        "W8": {"Grammar": [32, 13, 8, 4, 2, 1, 0], "Vocabulary": [7, 2, 9, 4, 11, 13, 14], "Communication": [13, 10, 15, 7, 4, 8, 3]}
    },
    "CG2": {
        "W1": {"Grammar": [37, 7, 5, 5, 6, 0, 0], "Vocabulary": [9, 1, 4, 5, 8, 14, 19], "Communication": [24, 4, 7, 8, 6, 8, 3]},
        "W8": {"Grammar": [24, 8, 14, 9, 4, 1, 0], "Vocabulary": [5, 3, 1, 7, 8, 9, 27], "Communication": [8, 4, 17, 12, 8, 6, 5]}
    },
    "TG": {
        "W1": {"Grammar": [47, 4, 4, 3, 1, 1, 0], "Vocabulary": [13, 8, 6, 3, 11, 11, 8], "Communication": [32, 4, 5, 6, 3, 7, 3]},
        "W8": {"Grammar": [0, 0, 1, 8, 5, 14, 32], "Vocabulary": [0, 0, 0, 3, 18, 19, 20], "Communication": [0, 0, 0, 1, 9, 20, 30]}
    }
}

# Colors and darker variants for lines
blue_shades = ["#2E86C1", "#5DADE2", "#AED6F1"]
dark_blue_line = "#1B4F72"

colors_w8 = {
    "CG1": {"bars": ["#E67E22", "#F39C12", "#F8C471"], "line": "#873600"},
    "CG2": {"bars": ["#F1C40F", "#F4D03F", "#F9E79F"], "line": "#7D6608"},
    "TG":  {"bars": ["#27AE60", "#52BE80", "#A9DFBF"], "line": "#0E6251"}
}

fig, axes = plt.subplots(3, 2, figsize=(7, 9), sharey=True)

cohorts = ["CG1", "CG2", "TG"]
titles_w2 = ["Cohort 1: Self Consistency - Score & Feedback (Week 2)", 
             "Cohort 2: Self Consistency - Score & Feedback (Week 2)", 
             "Cohort 3: Self Consistency - Score & Feedback (Week 2)"]
titles_w8 = ["Cohort 1: Self Consistency - Score & Feedback (Week 8)", 
             "Cohort 2: HeteroMAD - Score & Feedback (Week 8)", 
             "Cohort 3: Learning in Blocks (Week 8)"]
skills = ["Grammar", "Vocabulary", "Communication"]

x = np.arange(len(bins))
width = 0.25

# Dynamically calculate global maximum for shared y-axis ceiling
all_values = []
for cohort in data.values():
    for week in cohort.values():
        for skill_vals in week.values():
            all_values.extend(skill_vals)
global_max = max(all_values)
y_ceiling = int(np.ceil(global_max / 5.0) * 5) # Round up to nearest 5 for a clean look

for row, cohort in enumerate(cohorts):
    # Week 2
    ax_w1 = axes[row, 0]
    for i, skill in enumerate(skills):
        vals = data[cohort]["W1"][skill]
        ax_w1.bar(x + (i-1)*width, vals, width, label=skill, color=blue_shades[i], alpha=0.7)
    
    # Trend line - Linear Regression (Straight line)
    all_vals_w1 = np.array([data[cohort]["W1"][s] for s in skills])
    avg_vals_w1 = np.mean(all_vals_w1, axis=0)
    z1 = np.polyfit(x, avg_vals_w1, 1)
    p1 = np.poly1d(z1)
    ax_w1.plot(x, p1(x), color=dark_blue_line, linestyle='-', linewidth=1.2, alpha=0.6, label='Trend Line')

    ax_w1.set_title(titles_w2[row], fontsize=8, fontweight='bold', pad=5)
    ax_w1.set_xticks(x)
    if row == 2:
        ax_w1.set_xticklabels(bins, rotation=0, fontsize=8)
    else:
        ax_w1.set_xticklabels([])
        
    ax_w1.set_ylabel("Student Count", fontsize=9)
    ax_w1.grid(axis='y', linestyle=':', alpha=0.6)
    ax_w1.legend(prop={'size': 7.5}, loc='upper right', frameon=True, borderpad=0.5)
    ax_w1.set_ylim(0, y_ceiling)

    # Week 8
    ax_w8 = axes[row, 1]
    for i, skill in enumerate(skills):
        vals = data[cohort]["W8"][skill]
        ax_w8.bar(x + (i-1)*width, vals, width, label=skill, color=colors_w8[cohort]["bars"][i], alpha=0.7)
    
    # Trend line - Linear Regression
    all_vals_w8 = np.array([data[cohort]["W8"][s] for s in skills])
    avg_vals_w8 = np.mean(all_vals_w8, axis=0)
    z8 = np.polyfit(x, avg_vals_w8, 1)
    p8 = np.poly1d(z8)
    ax_w8.plot(x, p8(x), color=colors_w8[cohort]["line"], linestyle='-', linewidth=1.2, alpha=0.6, label='Trend Line')

    ax_w8.set_title(titles_w8[row], fontsize=8, fontweight='bold', pad=5)
    ax_w8.set_xticks(x)
    if row == 2:
        ax_w8.set_xticklabels(bins, rotation=0, fontsize=8)
    else:
        ax_w8.set_xticklabels([])
        
    ax_w8.grid(axis='y', linestyle=':', alpha=0.6)
    ax_w8.legend(prop={'size': 7.5}, loc='upper right', frameon=True, borderpad=0.5)
    ax_w8.set_ylim(0, y_ceiling)

import os

plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=0.5)
# Save with 300 DPI in the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_pdf = os.path.join(script_dir, "master_progress_chart_highres.pdf")
output_png = os.path.join(script_dir, "master_progress_chart_highres.png")

plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"High-resolution chart generated: {output_pdf} (300 DPI, Times New Roman)")
