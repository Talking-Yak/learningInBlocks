# _Learning in Blocks_

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Venue](https://img.shields.io/badge/Accepted-AIED_2026-brightgreen.svg)](https://www.aied-conference.org/2026/program/main-track)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:
**"_Learning in Blocks_: A Multi Agent Debate Assisted Personalized Adaptive Learning Framework for Language Learning"**
_Accepted as a Main Paper at the 27th International Conference on AI in Education (AIED 2026), Seoul, Republic of Korea._

---

## 📝 Abstract

Most digital language learning curricula rely on discrete-item quizzes that test recall rather than applied conversational proficiency. When progression is driven by quiz performance, learners can advance despite persistent gaps in using grammar and vocabulary during interaction. Recent work on LLM-based judging suggests a path toward scoring open-ended conversations, but using interaction evidence to drive progression and review requires scoring protocols that are reliable and validated.

We introduce **_Learning in Blocks_**, a framework that grounds progression in demonstrated conversational competence evaluated using CEFR-aligned rubrics. The framework employs **Heterogeneous Multi-Agent Debate (HeteroMAD)** in two stages:

1. **Scoring Stage**: Role-specialized agents independently evaluate Grammar, Vocabulary, and Interactive Communication, engage in debate to address conflicting judgments, and a judge synthesizes consensus scores.
2. **Recommendation Stage**: Identifies specific grammar skills and vocabulary topics for targeted review.

Progression requires demonstrating 70% mastery, and spaced review targets identified weaknesses to counter skill decay. We benchmark four scoring and recommendation methods on CEFR A2 conversations annotated by ESL experts. HeteroMAD achieves a superior score agreement with a 0.23 degree of variation and recommendation acceptability of 90.91%. An 8-week study with 180 CEFR A2 learners demonstrates that combining rubric-aligned scoring and recommendation with spaced review and mastery-based progression produces better learning outcomes than feedback alone.

---

## 🚀 Key Features

- **HeteroMAD (Heterogeneous Multi-Agent Debate)**: A novel multi-agent architecture for scoring and recommendations.
- **CEFR-Aligned Rubrics**: Evaluation grounded in International Standards (Common European Framework of Reference for Languages).
- **Two-Stage Architecture**: Decoupled scoring and recommendation for higher accuracy and interpretability.
- **Mastery-Based Progression**: 70% mastery threshold for advancing to new blocks.
- **Spaced Review**: Personalized reinforcement based on identified conversational gaps.

---

## 📂 Project Structure

```text
├── src/                         # Source code
│   ├── blocks/                  # Core framework implementation
│   │   ├── feedback/            # Scoring module (HeteroMAD, HomoMAD, Self-Refine, etc.)
│   │   └── recommend/           # Recommendation module (targeted review paths)
│   └── evaluationResults/       # Scripts for benchmarking and data analysis
│       ├── calculate_agreement.py # Computes Inter-Rater Reliability (Cohen's Kappa)
│       ├── calculate_mae.py     # Mean Absolute Error analysis
│       ├── generate_master_chart.py # Visualizes learning progression (Week 2 vs. Week 8)
│       └── stats.py             # Statistical significance tests (t-tests, ANOVA)
├── asset/                       # Data assets (prompts, CSVs, vocab files)
├── requirements.txt             # Project dependencies
├── LICENSE                      # Project license
└── README.md                    # This file
```

---

## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nicyscaria/learning_in_blocks.git
   cd learning_in_blocks
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Evaluation & Results

The scripts in `src/evaluationResults/` can be used to replicate the results reported in the paper:

- **Inter-Rater Reliability**: `python src/evaluationResults/calculate_agreement.py`
- **Mean Absolute Error**: `python src/evaluationResults/calculate_mae.py`
- **Progress Visualization**: `python src/evaluationResults/generate_master_chart.py`
- **Statistical Analysis**: `python src/evaluationResults/stats.py`

---

## 🔗 Citation

> The final version of this paper and its preprint are not yet available online. This section will be updated once the official links are released.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
