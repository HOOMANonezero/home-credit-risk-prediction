import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st



st.set_page_config(
    page_title="Home Credit | Default Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "home_credit_production_pipeline.pkl"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
DECISION_THRESHOLD = 0.13

NUMERIC_FEATURES = [
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_MIN",
    "age",
    "YEARS_EMPLOYED",
    "APPROVED_RATIO",
    "REGION_RATING_CLIENT_W_CITY",
    "REFUSED_COUNT",
    "ACTIVE_LOAN_COUNT",
    "CREDIT_GOODS_RATIO",
    "LATE_RATIO",
    "ACTIVE_RATIO",
    "DAYS_ID_PUBLISH",
    "LOG_INCOME",
    "LOG_CREDIT",
    "year_take",
    "TOTAL_PAYMENT",
    "AMT_ANNUITY",
]

CATEGORICAL_FEATURES = [
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "ORGANIZATION_TYPE",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


SUITE_OPTIONS = [
    "Unaccompanied",
    "Family",
    "Spouse / partner",
    "Children",
    "Group of people",
]

INCOME_OPTIONS = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Student",
    "Businessman",
    "Maternity leave",
    "Unemployed",
]

EDUCATION_OPTIONS = [
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]

FAMILY_OPTIONS = [
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated",
    "Widow",
]

HOUSING_OPTIONS = [
    "House / apartment",
    "Rented apartment",
    "With parents",
    "Municipal apartment",
    "Office apartment",
    "Co-op apartment",
]

ORGANIZATION_OPTIONS = {
    "Business": "Business Entity Type 3",
    "Government": "Government",
    "Self-employed": "Self-employed",
    "Education": "School",
    "Medicine": "Medicine",
    "Construction": "Construction",
    "Transport": "Transport: type 3",
    "Trade": "Trade: type 3",
    "Industry": "Industry: type 3",
    "Bank": "Bank",
    "Insurance": "Insurance",
    "Security": "Security",
    "Hotel": "Hotel",
    "Restaurant": "Restaurant",
    "Housing / real estate": "Housing",
    "Services": "Services",
    "Other": "Other",
}


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2.0rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #eef6ff 0%, #f8fbff 100%);
            border: 1px solid #d8e9ff;
            margin-bottom: 1.25rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.05rem;
            line-height: 1.15;
            color: #12355b;
        }

        .hero p {
            margin: .55rem 0 0 0;
            color: #52657a;
            font-size: 1rem;
        }

        .section-card {
            border: 1px solid #e4eaf1;
            border-radius: 16px;
            padding: 1.0rem 1.2rem;
            background: #ffffff;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #12355b;
            margin: 0;
        }

        .section-subtitle {
            font-size: .92rem;
            color: #66788a;
            margin-top: .2rem;
            margin-bottom: .85rem;
        }

        .result-card {
            border-radius: 18px;
            border: 1px solid #dbe6f1;
            padding: 1.2rem;
            background: #fbfdff;
            margin-top: .8rem;
        }

        .result-score {
            font-size: 2.5rem;
            font-weight: 850;
            color: #12355b;
            line-height: 1;
        }

        .result-band {
            font-size: 1.25rem;
            font-weight: 800;
            margin-top: .35rem;
        }

        .muted {
            color: #66788a;
            font-size: .88rem;
        }

/* Prediction result cards */
div[data-testid="stMetric"] {
    border: 1px solid #2f5f8f;
    border-radius: 16px;
    padding: 1rem 1.1rem;
    background: #102a43 !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
}

div[data-testid="stMetricLabel"] {
    color: #b8cce0 !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricDelta"] {
    color: #d9e7f5 !important;
}
/* Prediction messages */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid #2f5f8f !important;
    background: #173b5e !important;
}

div[data-testid="stAlert"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stApp {
    background: #0b1726 !important;
}

        button[kind="primary"] {
            min-height: 3.0rem;
            font-size: 1.05rem;
            font-weight: 750;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_FILENAME} was not found in the same folder as app1.py."
        )
    return joblib.load(MODEL_PATH)



def risk_band(score: float) -> tuple[str, str]:
    """Dynamic presentation band based on the CURRENT prediction score."""
    if score < DECISION_THRESHOLD:
        return (
            "Lower risk",
            "Below the project's 13% decision threshold.",
        )
    if score < 0.30:
        return (
            "Elevated risk",
            "Above the decision threshold, but below the app's high-risk band.",
        )
    return (
        "High risk",
        "30% or above on this model-risk-score presentation scale.",
    )


def validate_form(values: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    positive_fields = {
        "Annual income": values["income"],
        "Requested credit amount": values["credit"],
        "Regular loan payment": values["annuity"],
    }
    for label, value in positive_fields.items():
        if float(value) <= 0:
            errors.append(f"{label} must be greater than zero.")

    ratio_fields = {
        "Average external credit score": values["ext_mean"],
        "Highest external credit score": values["ext_max"],
        "Lowest external credit score": values["ext_min"],
        "Previous applications approved": values["approved_ratio"],
        "Payments made late": values["late_ratio"],
        "Previous loans still active": values["active_ratio"],
    }
    for label, value in ratio_fields.items():
        value = float(value)
        if not 0.0 <= value <= 1.0:
            errors.append(f"{label} must be between 0% and 100%.")

    if float(values["ext_min"]) > float(values["ext_max"]):
        errors.append(
            "Lowest external credit score cannot be higher than the highest score."
        )

    if float(values["ext_mean"]) < float(values["ext_min"]) or float(values["ext_mean"]) > float(values["ext_max"]):
        errors.append(
            "Average external credit score should be between the lowest and highest scores."
        )

    if float(values["credit_goods_ratio"]) <= 0:
        errors.append("Credit-to-purchase-value ratio must be greater than zero.")

    if int(values["refused_count"]) < 0 or int(values["active_loan_count"]) < 0:
        errors.append("Loan/application counts cannot be negative.")

    return errors


def build_model_input(values: dict[str, Any]) -> pd.DataFrame:
    income = float(values["income"])
    credit = float(values["credit"])
    annuity = float(values["annuity"])

    row = {
        "EXT_SOURCE_MEAN": float(values["ext_mean"]),
        "EXT_SOURCE_MAX": float(values["ext_max"]),
        "EXT_SOURCE_MIN": float(values["ext_min"]),
        "age": float(values["age"]),
        "YEARS_EMPLOYED": float(values["years_employed"]),
        "APPROVED_RATIO": float(values["approved_ratio"]),
        "REGION_RATING_CLIENT_W_CITY": float(values["region_rating"]),
        "REFUSED_COUNT": int(values["refused_count"]),
        "ACTIVE_LOAN_COUNT": int(values["active_loan_count"]),
        "CREDIT_GOODS_RATIO": float(values["credit_goods_ratio"]),
        "LATE_RATIO": float(values["late_ratio"]),
        "ACTIVE_RATIO": float(values["active_ratio"]),
        "DAYS_ID_PUBLISH": float(values["days_id_publish"]),
        "LOG_INCOME": float(np.log1p(income)),
        "LOG_CREDIT": float(np.log1p(credit)),
        "year_take": float(credit / annuity),
        "TOTAL_PAYMENT": float(values["total_payment"]),
        "AMT_ANNUITY": annuity,
        "CODE_GENDER": values["gender"],
        "NAME_CONTRACT_TYPE": values["contract_type"],
        "FLAG_OWN_CAR": values["own_car"],
        "FLAG_OWN_REALTY": values["own_realty"],
        "NAME_TYPE_SUITE": values["suite"],
        "NAME_INCOME_TYPE": values["income_type"],
        "NAME_EDUCATION_TYPE": values["education"],
        "NAME_FAMILY_STATUS": values["family_status"],
        "NAME_HOUSING_TYPE": values["housing"],
        "ORGANIZATION_TYPE": values["organization_type"],
    }

    return pd.DataFrame([row], columns=ALL_FEATURES)


st.markdown(
    """
    <div class="hero">
        <h1>🏦 Home Credit — Default Risk Predictor</h1>
        <p>
            A portfolio-ready interface for the LightGBM credit-risk model trained
            in <b>Home_credit_risk.ipynb</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    pipeline = load_model()
except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(str(exc))
    st.info(
        f"Place '{MODEL_FILENAME}' next to app1.py and make sure the required "
        "Python packages are installed."
    )
    st.stop()

with st.expander("ℹ️ About this model", expanded=False):
    st.write(
        "The saved production pipeline contains preprocessing plus the final LightGBM model. "
        "The interface below collects the values needed to recreate the 28 input columns "
        "expected by that pipeline."
    )
    st.write(f"**Pipeline file:** `{MODEL_FILENAME}`")
    st.write(f"**Model input columns:** {len(ALL_FEATURES)}")
    st.caption(
        "The score shown by this demo is a model risk score. It should not be interpreted "
        "as a perfectly calibrated real-world probability of default."
    )


with st.form("risk_form", clear_on_submit=False):
    st.markdown(
        '<div class="section-title">1. Applicant profile</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Basic personal, employment, family and housing information.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.number_input(
            "Age (years)",
            min_value=18.0,
            max_value=100.0,
            value=35.0,
            step=1.0,
        )
        years_employed = st.number_input(
            "Years of employment",
            min_value=0.0,
            max_value=60.0,
            value=5.0,
            step=0.5,
        )
        own_car = st.selectbox("Owns a car?", ["N", "Y"])

    with c2:
        contract_type = st.selectbox(
            "Loan type",
            ["Cash loans", "Revolving loans"],
        )
        own_realty = st.selectbox("Owns a home / property?", ["Y", "N"])
        income_type = st.selectbox("Main source of income", INCOME_OPTIONS)
        education = st.selectbox("Education level", EDUCATION_OPTIONS)

    with c3:
        family_status = st.selectbox("Marital / family status", FAMILY_OPTIONS)
        housing = st.selectbox("Current housing type", HOUSING_OPTIONS)
        suite = st.selectbox(
            "Who accompanied the applicant?",
            SUITE_OPTIONS,
            help="Select who, if anyone, came with the applicant during the application process.",
        )
        organization_label = st.selectbox(
            "Main work sector",
            list(ORGANIZATION_OPTIONS.keys()),
            help="Simplified job/organization categories. Several similar dataset categories are grouped together here.",
        )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">2. Credit history & payment behaviour</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Answer what you know about the applicant’s previous borrowing and repayment record. Percentage fields use a 0–100% scale.</div>',
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        ext_mean_pct = st.slider(
            "Average external credit score",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Summary score from external credit-information sources. Enter it as a percentage from 0 to 100. Higher generally indicates a stronger credit profile.",
        )
        ext_mean = ext_mean_pct / 100.0
        ext_max_pct = st.slider(
            "Highest external credit score",
            min_value=0,
            max_value=100,
            value=60,
            step=1,
            help="Highest score available from the external credit-information sources. Enter it as a percentage from 0 to 100.",
        )
        ext_max = ext_max_pct / 100.0
        ext_min_pct = st.slider(
            "Lowest external credit score",
            min_value=0,
            max_value=100,
            value=40,
            step=1,
            help="Lowest score available from the external credit-information sources. Enter it as a percentage from 0 to 100.",
        )
        ext_min = ext_min_pct / 100.0

    with h2:
        approved_ratio_pct = st.slider(
            "Previous applications approved",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            help="Approximate share of previous credit applications that were approved. Enter a percentage from 0 to 100.",
        )
        approved_ratio = approved_ratio_pct / 100.0
        refused_count = st.number_input(
            "Previous applications refused",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="Number of previous credit applications that were refused.",
        )
        active_loan_count = st.number_input(
            "Currently active loans",
            min_value=0,
            max_value=100,
            value=1,
            step=1,
            help="Number of credit accounts/loans currently active in the historical data.",
        )

    with h3:
        late_ratio_pct = st.slider(
            "Payments made late",
            min_value=0,
            max_value=100,
            value=5,
            step=1,
            help="Approximate share of recorded payments that were late. Enter a percentage from 0 to 100.",
        )
        late_ratio = late_ratio_pct / 100.0
        active_ratio_pct = st.slider(
            "Previous loans still active",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Approximate share of previous loans that are still active. Enter a percentage from 0 to 100.",
        )
        active_ratio = active_ratio_pct / 100.0
        region_rating = st.selectbox(
            "Regional risk rating",
            [1.0, 2.0, 3.0],
            index=1,
            format_func=lambda x: {
                1.0: "1 — lower relative risk",
                2.0: "2 — medium / typical",
                3.0: "3 — higher relative risk",
            }[x],
            help="Simplified display of the regional rating feature used by the model.",
        )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">3. Loan & financial information</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Enter the applicant’s main financial values. The model’s logarithmic features are calculated automatically.</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        income = st.number_input(
            "Annual income",
            min_value=1.0,
            max_value=100_000_000.0,
            value=180_000.0,
            step=5_000.0,
            help="Annual income in the same currency units used by the original dataset.",
        )
        credit = st.number_input(
            "Requested credit amount",
            min_value=1.0,
            max_value=100_000_000.0,
            value=500_000.0,
            step=5_000.0,
            help="Amount of credit requested by the applicant.",
        )

    with f2:
        annuity = st.number_input(
            "Regular loan payment",
            min_value=1.0,
            max_value=10_000_000.0,
            value=20_000.0,
            step=500.0,
            help="Regular annuity/payment amount.",
        )
        total_payment = st.number_input(
            "Total historical payments",
            min_value=0.0,
            max_value=100_000_000.0,
            value=100_000.0,
            step=1_000.0,
            help="Total payment amount represented in the applicant’s aggregated payment history.",
        )

    with f3:
        credit_goods_ratio = st.number_input(
            "Credit amount / purchase value",
            min_value=0.01,
            max_value=20.0,
            value=1.20,
            step=0.01,
            help="Ratio used by the trained model. 1.20 means the credit amount is about 20% above the financed purchase value.",
        )
        days_id_publish_age = st.number_input(
            "Days since ID information was updated",
            min_value=0.0,
            max_value=10_000.0,
            value=1_500.0,
            step=1.0,
            help="Enter the number of days since the ID information was updated. The original model feature stores this as a negative number, so the app converts it automatically.",
        )

    submitted = st.form_submit_button(
        "🔎 Calculate default risk",
        type="primary",
        use_container_width=True,
    )


if submitted:
    values = {
        "gender": gender,
        "age": age,
        "years_employed": years_employed,
        "own_car": own_car,
        "contract_type": contract_type,
        "own_realty": own_realty,
        "income_type": income_type,
        "education": education,
        "family_status": family_status,
        "housing": housing,
        "suite": suite,
        "organization_type": ORGANIZATION_OPTIONS[organization_label],
        "ext_mean": ext_mean,
        "ext_max": ext_max,
        "ext_min": ext_min,
        "approved_ratio": approved_ratio,
        "refused_count": refused_count,
        "active_loan_count": active_loan_count,
        "late_ratio": late_ratio,
        "active_ratio": active_ratio,
        "region_rating": region_rating,
        "income": income,
        "credit": credit,
        "annuity": annuity,
        "total_payment": total_payment,
        "credit_goods_ratio": credit_goods_ratio,
        "days_id_publish": -float(days_id_publish_age),
    }

    errors = validate_form(values)

    if errors:
        for error in errors:
            st.error(error)
    else:
        input_df = build_model_input(values)

        try:
            with st.spinner("Analyzing the application..."):
                score = float(pipeline.predict_proba(input_df)[0, 1])
        except Exception as exc:
            st.error("Prediction failed.")
            with st.expander("Technical details"):
                st.exception(exc)
        else:

            st.session_state["latest_prediction"] = {
                "score": score,
                "input_df": input_df.copy(),
                "organization_label": organization_label,
            }

if "latest_prediction" in st.session_state:
    result = st.session_state["latest_prediction"]
    score = float(result["score"])
    band, band_note = risk_band(score)
    above_threshold = score >= DECISION_THRESHOLD

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Prediction result</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "The result below always represents the most recent form submission. "
        "Submit the form again to analyze another applicant."
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Model risk score", f"{score:.2%}")
    with r2:
        st.metric("Risk band", band)
    with r3:
        st.metric("Project threshold", f"{DECISION_THRESHOLD:.0%}")

    st.progress(min(max(score, 0.0), 1.0))

    if above_threshold:
        st.warning(
            f"Current model score: {score:.2%}. This is above the project's "
            f"13% decision threshold and would be classified as higher-risk "
            "under the project's decision rule."
        )
    else:
        st.success(
            f"Current model score: {score:.2%}. This is below the project's "
            f"13% decision threshold and would be classified as lower-risk "
            "under the project's decision rule."
        )

    st.info(f"**Risk band:** {band}. {band_note}")

    st.caption(
        "Important: this is a model risk score, not a guaranteed real-world probability. "
        "The final model uses class weighting (`scale_pos_weight`)."
    )

    with st.expander("View the exact data sent to the model"):
        display_df = result["input_df"].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if st.button("🧹 Clear current result", use_container_width=True):
        del st.session_state["latest_prediction"]
        st.rerun()

