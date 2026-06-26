# F1 Driver Analysis

A Python data science project for analyzing Formula 1 driver performance and statistics.

## Project Structure

```
F1_Driver_Analysis/
├── data/                 # Data storage
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code modules
├── results/              # Output files (plots, reports)
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Setup

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start working:**
   - Place raw data in `data/` folder
   - Create analysis notebooks in `notebooks/`
   - Add helper functions in `src/`
   - Save outputs to `results/`

## Key Libraries

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** & **seaborn** - Visualization
- **scikit-learn** - Machine learning
- **jupyter** - Interactive notebooks

## Getting Started

Open a Jupyter notebook:
```bash
jupyter notebook
```

Then create a new notebook in the `notebooks/` folder to start your analysis.

## Dashboard

An interactive Streamlit dashboard visualizes the Australia 2025 qualifying
telemetry (classification, per-driver traces, an interactive track map, driver
comparison, and corner analysis):

```bash
streamlit run dashboard/app.py
```

It loads the processed data via `src/data_io.py` (reads the committed
`.csv.gz` directly — no decompression needed). If the processed file is
missing, generate it first with `python process_telemetry.py`.
