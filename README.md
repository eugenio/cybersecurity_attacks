# Cybersecurity Attacks ML Classifier

MSc group project (DSTI) — a machine learning pipeline that classifies network traffic into three cyber attack types: **DDoS**, **Intrusion**, and **Malware**. Includes a Streamlit web app for interactive predictions.

## Team

Eugenio La Cava, Otmane Qorchi, Janagam Vasantha, Elly Smagghe, Kaloina Rakotobe, Sanchana Krishna Kumar, Siham Eldjouher

## Quick Start

Requires [Pixi](https://pixi.sh/) (recommended) or Python 3.12.

```bash
# Run the Streamlit app
pixi run run-app

# Run the full ML training pipeline
pixi run run-pipeline
```

Without Pixi:

```bash
pip install -r requirements.txt
streamlit run app.py
python pipeline.py
```

### Pipeline Options

```bash
python pipeline.py --no-figures     # Skip figure rendering
python pipeline.py --sequential     # Low-RAM sequential mode
python pipeline.py --model-only     # Show only model figures
python pipeline.py --monitor        # Live memory TUI
python pipeline.py --mem-limit 16   # Cap memory at 16 GB
```

## Models

Three classifiers trained on ~40k rows of network traffic data:

| Model | Algorithm | Notes |
|---|---|---|
| Logistic Regression | `solver=lbfgs`, `max_iter=1000` | Baseline model |
| Random Forest | `RandomizedSearchCV` (30 iter, 3-fold CV, F1-macro) | Tuned hyperparameters |
| Extra Trees | `n_estimators=500`, `max_depth=20` | Port-focused feature set with target encoding |

Models are stored on [HuggingFace Hub](https://huggingface.co/uge84/cybersecurity-models) and downloaded automatically at runtime.

## Streamlit App

The app provides three prediction modes:

- **CSV Upload** — batch predictions on uploaded data with downloadable results
- **Pick a Row** — select from the held-out test set, compare prediction vs actual
- **Manual Input** — 25-field form with live IP geolocation and interactive Folium map

Sidebar shows model selector, accuracy metric, confusion matrix, and ROC curve.

Deployed on **Streamlit Cloud** (pipeline re-run disabled in cloud; models loaded from HuggingFace).

## Pipeline

1. **EDA** — column renaming, date feature extraction (day bins, weekday, hour buckets, month), IP geolocation via MaxMind GeoLite2, User-Agent parsing
2. **Feature Engineering** — crosstab encoding, daily aggregates, categorical binary encoding, port target encoding
3. **Training** — 90/10 train/test split, model fitting, metrics computation (confusion matrix, ROC, PR curves)
4. **Export** — models saved as `.pkl` dicts (model + label encoder + features + figures) and uploaded to HuggingFace

## Project Structure

```
cybersecurity_attacks                                                                                                                                                                                                                                                                   
├── app.py                                                                                                                                                                                                                                                                              
├── data                                                                                                                                                                                                                                                                                
│   └── cybersecurity_attacks.csv
├── docs
│   ├── download.jpg
│   └── README.md
├── EDA graphs
│   ├── Action Taken.png
│   ├── Alerts_Warnings.png
│   ├── Anomaly Scores.png
│   ├── Attack Signature.png
│   ├── Browser family.png
│   ├── date H.png
│   ├── date M.png
│   ├── date MW.png
│   ├── date WD.png
│   ├── date.png
│   ├── Destination Port.png
│   ├── Malware Indicators.png
│   ├── OS family.png
│   ├── Packet Length.png
│   ├── Packet Type.png
│   ├── Protocol.png
│   ├── Severity Level.png
│   ├── Source & Dest IPs.png
│   └── Source Port.png
├── generate_ascii_dir_repr.ps1
├── geolite2_db
│   ├── GeoLite2-ASN.mmdb
│   ├── GeoLite2-City.mmdb
│   ├── GeoLite2-Country.mmdb
│   └── readme.txt
├── LICENSE
├── pipeline.py
├── pixi.lock
├── pixi.toml
├── README.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── eda_pipeline.py
    ├── EDA.py
    ├── modelling.py
    ├── ports_pipeline.py
    ├── upload_models.py
    └── utilities
        ├── __init__.py
        ├── config.py
        ├── data_preparation.py
        ├── diagrams.py
        ├── download_files.py
        ├── feature_engineering.py
        ├── helpers.py
        ├── mem_monitor.py
        ├── payload_analyzer.py
        ├── statistical_analysis.py
        ├── time_series.py
        ├── utils.py
        ├── validation.py
        └── visualization.py
```

## Key Dependencies

scikit-learn, pandas, numpy, plotly, streamlit, folium, geoip2, spacy, huggingface_hub, joblib

## License

See [LICENSE](LICENSE) file.
