# BHEL_Engineering_Drawing_Analysis_Tool
Developed a Python-based analysis tool during my internship at BHEL to process engineering drawing review data. Automated data cleaning, statistical analysis, and visualization using pandas, numpy, seaborn, and matplotlib, helping identify bottlenecks and improve workflow efficiency.




This repository contains my internship project at **BHEL**: an analysis tool for engineering drawing data.  
It processes raw logs, performs exploratory analysis, and generates visuals/metrics to highlight trends and bottlenecks.

> Main script: `BHEL_ENGINEERING_DRAWING_ANALYSIS_TOOL.py`

---

## ✨ Features
- Data loading and preprocessing
- Summary statistics and group-wise analysis
- Visualizations (charts/plots) saved to `outputs/`
- Configurable paths and lightweight dependencies

---

## 🧰 Tech Stack
- Python 3.9+
- Libraries: see `requirements.txt`

---

## 🚀 Quickstart

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/BHEL_Engineering_Drawing_Analysis_Tool.git
cd BHEL_Engineering_Drawing_Analysis_Tool

# 2) Create & activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run
python BHEL_ENGINEERING_DRAWING_ANALYSIS_TOOL.py
```

> Place your input dataset in the project root (or adjust the script path). Outputs (charts/reports) will be saved under `outputs/` if the script creates them.

---

## 🗂️ Repository Structure

```
BHEL_Engineering_Drawing_Analysis_Tool/
├── BHEL_ENGINEERING_DRAWING_ANALYSIS_TOOL.py   # Main analysis script
├── requirements.txt                            # Dependencies (auto-derived from imports)
├── .gitignore                                  # Ignore cache & generated files
└── README.md                                   # This file
```

---

## 📎 Notes
- This is a learning/analysis tool from my internship; it is not a production application.
- If any library is missing at runtime, please install it and consider opening a PR to update `requirements.txt`.

---

## 📜 License
MIT (or choose a license you prefer)
