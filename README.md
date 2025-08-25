# TecloCustomerChain — Telecom Churn Analysis with ML Pipelines

[![Releases](https://img.shields.io/badge/Releases-Download-blue?logo=github&style=for-the-badge)](https://github.com/amroant/TecloCustomerChain/releases)

[Releases / download the artifact](https://github.com/amroant/TecloCustomerChain/releases)

<img src="https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1200&q=80" alt="Telecom network" style="width:100%;max-height:300px;object-fit:cover;margin:16px 0;" />

- Topics: eda, gradient-boosting, logistic-regression, machine-learning, matplotlib, numpy, pandas, random-forest, scikit-learn, seaborn

Table of contents
- About the project
- Key goals
- Dataset and sources
- Project pipeline
- Model suite
- Evaluation metrics
- How to run (release file)
- Local setup
- Example usage
- Visuals and plots
- Results and interpretation
- Project structure
- Contributing
- License
- Contact

About the project
This repository holds a reproducible pipeline for analyzing customer churn in a telecom operator. The repo bundles data processing, exploratory data analysis (EDA), feature engineering, model training, and model evaluation. The goal: find the main drivers of churn and produce actionable signals for retention teams.

Key goals
- Identify features that predict churn.
- Build stable models for binary classification.
- Provide visual reports that explain model decisions.
- Offer artifact(s) that run key steps end to end.

Dataset and sources
The project uses a customer-level dataset with the usual telecom fields:
- customer_id, tenure, contract_type, payment_method
- monthly_charges, total_charges
- services: phone, internet, streaming, tech_support
- demographic fields: gender, senior_citizen, partner, dependents
- churn flag (target)

The dataset includes derived columns for usage patterns and engagement. Where needed, synthetic samples fill gaps to illustrate pipelines. All data transforms appear in the notebooks and scripts.

Project pipeline
The pipeline follows structured steps:

1. EDA
   - Univariate distributions
   - Categorical frequency tables
   - Missing data map
   - Correlation heatmap
2. Feature engineering
   - Binning tenure
   - Usage rate and relative spend
   - Interaction terms (contract × payment_method)
   - Aggregates per region and plan
3. Encoding and scaling
   - One-hot for low-cardinality categoricals
   - Target encoding for high-cardinality keys
   - Standard scaling for numeric inputs
4. Modeling
   - Baseline: logistic regression
   - Tree-based: random forest
   - Boosting: gradient boosting (XGBoost / LightGBM)
5. Evaluation
   - AUC-ROC, precision-recall
   - Calibration plots
   - SHAP or permutation importances for explainability
6. Deployment artifact
   - Saved model and a runner script in releases

Model suite
- Logistic Regression
  - Fast baseline
  - Coefficients show direction per feature
- Random Forest
  - Robust to outliers
  - Provides feature importances
- Gradient Boosting
  - Best predictive power in most tests
  - Use early stopping and cross-validation

Evaluation metrics
- ROC AUC — rank ordering of predictions
- Precision and recall at selected thresholds
- F1-score for balance between precision and recall
- Confusion matrix for operational thresholds
- Lift and gain charts for campaign targeting

How to run (download and execute)
The release page contains prebuilt artifacts and runner scripts. Download the release artifact from the releases page and run the provided runner.

- Visit the releases page: https://github.com/amroant/TecloCustomerChain/releases
- Download the release asset named like teclo-runner-vX.Y.tar.gz or similar.
- Extract and execute the runner script:
  - Linux / macOS:
    - tar -xzf teclo-runner-vX.Y.tar.gz
    - cd teclo-runner
    - ./run_pipeline.sh
  - Windows:
    - Unpack archive
    - Run run_pipeline.bat

The released file contains a runnable pipeline for EDA, training, and report generation. The script reads config.yaml, trains the chosen model, and writes outputs to ./output.

Local setup
Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost or lightgbm
- matplotlib, seaborn
- shap (optional)

Install
- Create a virtualenv
  - python -m venv venv
  - source venv/bin/activate  (or venv\Scripts\activate on Windows)
- Install packages
  - pip install -r requirements.txt

Configuration
- config.yaml holds paths and model settings
- data/raw should hold raw CSVs
- data/processed will hold processed features
- models/ saves trained models
- output/ contains reports and figures

Example usage
Run core notebook
- Start Jupyter: jupyter lab
- Open notebooks/01-eda.ipynb for exploratory analysis
- Run notebooks/02-features.ipynb to build features
- Run notebooks/03-modeling.ipynb to train models and produce plots

Run script
- python scripts/train.py --config config.yaml --model xgb
- python scripts/predict.py --model models/xgb.pkl --input data/sample.csv

Visuals and plots
The project produces a set of visuals to aid decisions:
- Churn rate by tenure bucket
- Monthly charges distribution by churn flag
- Contract type vs churn heatmap
- Feature importance bar chart
- SHAP summary plot for top drivers

Sample embedded images
- Churn by tenure:  
  ![Churn by tenure](https://raw.githubusercontent.com/amroant/TecloCustomerChain/main/docs/images/churn_by_tenure.png)
- Feature importance:  
  ![Feature importance](https://raw.githubusercontent.com/amroant/TecloCustomerChain/main/docs/images/feature_importance.png)

Results and interpretation
- Contract duration strongly affects churn. Short-term contracts show higher churn.
- High monthly charges correlate with churn, but interaction with services matters.
- Non-electronic payment methods link to higher risk.
- Usage drops precede churn in a two-month window for many customers.

Operational guidance
- Target high-risk customers on month 10–14 of tenure for retention offers.
- Offer incentives for customers on month-to-month plans to switch to annual.
- Flag sudden usage drops for proactive outreach.

Project structure
- data/ — raw and processed data
- notebooks/ — step-by-step analysis
- scripts/ — training and prediction scripts
- models/ — saved model files
- docs/ — static images and reports
- requirements.txt — package list
- config.yaml — pipeline configuration

Contributing
- Fork the repo
- Create a feature branch
- Add tests for new transforms
- Open a pull request with clear motivation and description
- Use the existing code style and testing approach

License
- MIT License. See LICENSE file.

Contact
- Report issues and feature requests via GitHub Issues.
- Releases page with runnable artifact: https://github.com/amroant/TecloCustomerChain/releases

<img src="https://raw.githubusercontent.com/amroant/TecloCustomerChain/main/docs/images/architecture_diagram.png" alt="Pipeline diagram" style="width:100%;max-height:300px;object-fit:cover;margin:16px 0;" />