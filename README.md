# CSV to Interactive Dashboard (Streamlit)

> Synent Technologies Data Science Internship | Task 4 (Basic level) | Author: Hassan Butt

## Problem statement
Turn any CSV into an interactive dashboard without writing code.

## Dataset details
- **Dataset:** Iris flower dataset
- **Source:** https://www.kaggle.com/datasets/uciml/iris
- **Size used:** 150 rows x 5 columns

## Approach
1. Streamlit app with CSV upload and a built-in sample dataset
2. Automatic detection of numeric and categorical columns
3. Dynamic filters, six chart types (Plotly), correlation heatmap and summary statistics
4. Download of the filtered data

## Results
| Metric | Value |
|---|---|
| App file | app.py |
| Syntax check | passed |
| Launch command | streamlit run app.py |

**Key insights**

- Upload any CSV and the dashboard detects numeric and categorical columns automatically.
- Sidebar filters, six chart types, a correlation heatmap and CSV download update the view instantly.
- A sample dataset (Iris) loads by default so the dashboard works before any upload.

**Other output files:** `app.py`, `data/sample_iris.csv`

## How to run
```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

## Project structure
```
main.py            # full analysis / pipeline
app.py             # Streamlit app
requirements.txt
data/              # dataset (see data/README.md)
outputs/           # figures, tables and reports
```

## Academic integrity
All analysis code and write-ups in this repository were written by the author. The dataset belongs to its original creators (see link above).
