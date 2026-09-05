## Home Credit Default Risk — Streamlit App


Machine Learning project for predicting the probability
of repayment difficulties using the Home Credit dataset.

## Files

app.py — Streamlit application

home_credit_production_pipeline.pkl — saved preprocessing + LightGBM pipeline

requirements.txt — deployment dependencies

## Local run

conda activate homecredit
pip install -r requirements.txt
streamlit run app.py


## Model

- LightGBM
- Class imbalance handling with class weighting
- ColumnTransformer
- Pipeline
- OneHotEncoder
- Median imputation
- SHAP explainability
- ROC-AUC evaluation

## Streamlit

The deployed application allows users to enter applicant
and historical credit information and receive a predicted
default probability.

## Important

This application is an educational / portfolio project
and should not be used as the sole basis for real financial
or lending decisions.