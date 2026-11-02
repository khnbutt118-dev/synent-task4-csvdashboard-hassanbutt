"""Synent Data Science Internship - Task 4: CSV to Interactive Dashboard (Streamlit)
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'iris': {'builtin': True,
          'files': ['Iris.csv', 'iris.csv'],
          'kaggle': 'uciml/iris',
          'name': 'Iris flower dataset',
          'page': 'https://www.kaggle.com/datasets/uciml/iris'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def task4_run(repo_dir):
    """Prepares a sample CSV for the dashboard and verifies app.py is valid Python."""
    try:
        sample = get_data("iris", repo_dir)
    except Exception:
        sample = _builtin("iris")
    sample.to_csv(Path(repo_dir) / "data" / "sample_iris.csv", index=False)
    app = Path(repo_dir) / "app.py"
    ok = False
    if app.exists():
        compile(app.read_text(encoding="utf-8"), "app.py", "exec")
        ok = True
    insights = [
        "Upload any CSV and the dashboard detects numeric and categorical columns automatically.",
        "Sidebar filters, six chart types, a correlation heatmap and CSV download update the view instantly.",
        "A sample dataset (Iris) loads by default so the dashboard works before any upload.",
    ]
    return {"headline": {"App file": "app.py", "Syntax check": "passed" if ok else "app.py missing",
                         "Launch command": "streamlit run app.py"},
            "insights": insights, "dataset_shape": [int(sample.shape[0]), int(sample.shape[1])],
            "files": ["app.py", "data/sample_iris.csv"]}



if __name__ == "__main__":
    res = task4_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
