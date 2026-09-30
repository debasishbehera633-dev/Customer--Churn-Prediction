# Python code only
# Customer Churn Prediction Dashboard - core implementation

from pathlib import Path
import sqlite3
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# CONFIGURATION
# =========================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATABASE_DIR = ROOT / "database"

DATA_FILE = DATA_DIR / "customers.csv"
MODEL_FILE = MODEL_DIR / "churn_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.csv"
DATABASE_FILE = DATABASE_DIR / "customer_churn.db"

RANDOM_STATE = 42


# =========================
# DATA GENERATION
# =========================

def generate_customer_data(
    n_customers: int = 3000,
    output_file: Path = DATA_FILE,
) -> pd.DataFrame:

    rng = np.random.default_rng(RANDOM_STATE)

    tenure = rng.integers(1, 73, n_customers)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_customers,
        p=[0.55, 0.28, 0.17],
    )

    monthly_charges = np.clip(
        rng.normal(110, 35, n_customers),
        25,
        350,
    )

    complaints = rng.poisson(0.7, n_customers)
    late_payments = rng.poisson(0.8, n_customers)
    support_tickets = rng.poisson(1.8, n_customers)

    last_login_days = np.clip(
        rng.normal(18, 15, n_customers),
        0,
        120,
    ).round().astype(int)

    satisfaction = np.clip(
        rng.normal(7, 1.8, n_customers),
        1,
        10,
    ).round().astype(int)

    usage = np.clip(
        rng.normal(130, 55, n_customers),
        10,
        400,
    )

    age = rng.integers(18, 76, n_customers)

    gender = rng.choice(
        ["Male", "Female", "Other"],
        n_customers,
        p=[0.48, 0.48, 0.04],
    )

    payment_method = rng.choice(
        ["Electronic", "Card", "Bank Transfer", "Cash"],
        n_customers,
        p=[0.45, 0.25, 0.20, 0.10],
    )

    internet_service = rng.choice(
        ["Fiber", "DSL", "None"],
        n_customers,
        p=[0.50, 0.35, 0.15],
    )

    total_charges = (
        monthly_charges
        * tenure
        * rng.uniform(0.85, 1.15, n_customers)
    )

    churn_score = (
        -2.0
        + 0.025 * monthly_charges
        - 0.035 * tenure
        + 0.35 * complaints
        + 0.22 * late_payments
        + 0.025 * last_login_days
        - 0.30 * (satisfaction - 5)
        + 0.65 * (contract_type == "Month-to-month")
        + 0.25 * (payment_method == "Electronic")
    )

    churn_probability = 1 / (1 + np.exp(-churn_score))

    churn = rng.binomial(
        1,
        np.clip(churn_probability, 0.02, 0.90),
    )

    data = pd.DataFrame(
        {
            "customer_id": [
                f"C{i:05d}"
                for i in range(1, n_customers + 1)
            ],
            "age": age,
            "gender": gender,
            "tenure_months": tenure,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "payment_method": payment_method,
            "internet_service": internet_service,
            "support_tickets": support_tickets,
            "complaints": complaints,
            "late_payments": late_payments,
            "avg_monthly_usage": usage.round(2),
            "last_login_days": last_login_days,
            "customer_satisfaction": satisfaction,
            "churn": churn,
        }
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_file,
        index=False,
    )

    return data


# =========================
# PREPROCESSING
# =========================

CATEGORICAL_FEATURES = [
    "gender",
    "contract_type",
    "payment_method",
    "internet_service",
]

NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "complaints",
    "late_payments",
    "avg_monthly_usage",
    "last_login_days",
    "customer_satisfaction",
]


def create_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


# =========================
# MODEL TRAINING
# =========================

def train_models():

    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        columns=[
            "customer_id",
            "churn",
        ]
    )

    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = {}
    trained_models = {}

    for model_name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        results[model_name] = {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities,
            ),
        }

        trained_models[model_name] = pipeline

    metrics = pd.DataFrame(results).T

    best_model_name = metrics[
        "roc_auc"
    ].idxmax()

    best_model = trained_models[
        best_model_name
    ]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_FILE,
    )

    metrics.to_csv(
        METRICS_FILE
    )

    print(
        f"Best model: {best_model_name}"
    )

    print(metrics)

    return best_model


# =========================
# SQL DATABASE
# =========================

def create_database():

    df = pd.read_csv(
        DATA_FILE
    )

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(
        DATABASE_FILE
    ) as connection:

        df.to_sql(
            "customers",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_customer_contract
            ON customers(contract_type)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_customer_churn
            ON customers(churn)
            """
        )


# =========================
# CUSTOMER PREDICTION
# =========================

def predict_customer(customer_data):

    model = joblib.load(
        MODEL_FILE
    )

    customer_data = customer_data.drop(
        columns=["customer_id"],
        errors="ignore",
    )

    probability = float(
        model.predict_proba(
            customer_data
        )[0][1]
    )

    prediction = int(
        probability >= 0.50
    )

    if probability >= 0.70:
        risk = "High Risk"
    elif probability >= 0.40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "risk": risk,
    }


# =========================
# SQL ANALYTICS
# =========================

def run_sql_queries():

    queries = {

        "Total Customers":
            """
            SELECT COUNT(*) AS total_customers
            FROM customers;
            """,

        "Churn Rate":
            """
            SELECT
                ROUND(
                    100.0 * AVG(churn),
                    2
                ) AS churn_rate
            FROM customers;
            """,

        "Churn By Contract":
            """
            SELECT
                contract_type,
                COUNT(*) AS customers,
                ROUND(
                    100.0 * AVG(churn),
                    2
                ) AS churn_rate
            FROM customers
            GROUP BY contract_type
            ORDER BY churn_rate DESC;
            """,

        "Churn By Payment":
            """
            SELECT
                payment_method,
                COUNT(*) AS customers,
                ROUND(
                    100.0 * AVG(churn),
                    2
                ) AS churn_rate
            FROM customers
            GROUP BY payment_method
            ORDER BY churn_rate DESC;
            """,

        "High Complaint Customers":
            """
            SELECT *
            FROM customers
            WHERE complaints >= 3
            ORDER BY complaints DESC
            LIMIT 20;
            """,

        "Average Charges By Churn":
            """
            SELECT
                churn,
                ROUND(
                    AVG(monthly_charges),
                    2
                ) AS avg_monthly_charges
            FROM customers
            GROUP BY churn;
            """,
    }

    with sqlite3.connect(DATABASE_FILE) as connection:
     for name, sql in queries.items():
        result = pd.read_sql_query(
            sql,
            connection
        )

        st.subheader(name)
        st.code(sql, language="sql")
        st.dataframe(result, use_container_width=True)


# =========================
# STREAMLIT DASHBOARD
# =========================

def run_dashboard():

    import streamlit as st

    st.set_page_config(
        page_title="Customer Churn Dashboard",
        page_icon="📉",
        layout="wide",
    )

    if not DATA_FILE.exists():

        generate_customer_data()

    if not MODEL_FILE.exists():

        train_models()

    if not DATABASE_FILE.exists():

        create_database()

    df = pd.read_csv(
        DATA_FILE
    )

    st.title(
        "📉 Customer Churn Prediction Dashboard"
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Customer Analysis",
            "Churn Prediction",
            "Model Performance",
            "SQL Insights",
        ],
    )

    if page == "Overview":

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Customers",
            f"{len(df):,}",
        )

        col2.metric(
            "Churn Rate",
            f"{df.churn.mean() * 100:.1f}%",
        )

        col3.metric(
            "Average Tenure",
            f"{df.tenure_months.mean():.1f} months",
        )

        col4.metric(
            "Average Monthly Charges",
            f"₹{df.monthly_charges.mean():,.0f}",
        )

        st.subheader(
            "Churn Distribution"
        )

        st.bar_chart(
            df["churn"]
            .value_counts()
            .rename(
                index={
                    0: "Stayed",
                    1: "Churned",
                }
            )
        )

    elif page == "Customer Analysis":

        st.subheader(
            "Churn By Contract Type"
        )

        contract_analysis = pd.crosstab(
            df["contract_type"],
            df["churn"],
            normalize="index",
        ) * 100

        st.bar_chart(
            contract_analysis
        )

        st.subheader(
            "Churn By Payment Method"
        )

        payment_analysis = pd.crosstab(
            df["payment_method"],
            df["churn"],
            normalize="index",
        ) * 100

        st.bar_chart(
            payment_analysis
        )

        st.subheader(
            "Complaints vs Churn"
        )

        complaints_analysis = pd.crosstab(
            df["complaints"],
            df["churn"],
            normalize="index",
        ) * 100

        st.bar_chart(
            complaints_analysis
        )

    elif page == "Churn Prediction":

        st.subheader(
            "Predict Customer Churn"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=90,
            value=30,
        )

        gender = st.selectbox(
            "Gender",
            df["gender"].unique(),
        )

        tenure = st.number_input(
            "Tenure Months",
            min_value=0,
            max_value=120,
            value=12,
        )

        contract = st.selectbox(
            "Contract Type",
            df["contract_type"].unique(),
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=100.0,
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1200.0,
        )

        payment = st.selectbox(
            "Payment Method",
            df["payment_method"].unique(),
        )

        internet = st.selectbox(
            "Internet Service",
            df["internet_service"].unique(),
        )

        tickets = st.number_input(
            "Support Tickets",
            min_value=0,
            max_value=30,
            value=2,
        )

        complaints = st.number_input(
            "Complaints",
            min_value=0,
            max_value=15,
            value=0,
        )

        late = st.number_input(
            "Late Payments",
            min_value=0,
            max_value=20,
            value=0,
        )

        usage = st.number_input(
            "Average Monthly Usage",
            min_value=0.0,
            value=100.0,
        )

        login = st.number_input(
            "Last Login Days Ago",
            min_value=0,
            max_value=365,
            value=10,
        )

        satisfaction = st.slider(
            "Customer Satisfaction",
            min_value=1,
            max_value=10,
            value=7,
        )

        if st.button(
            "Predict Churn"
        ):

            customer = pd.DataFrame(
                [
                    {
                        "age": age,
                        "gender": gender,
                        "tenure_months": tenure,
                        "contract_type": contract,
                        "monthly_charges": monthly,
                        "total_charges": total,
                        "payment_method": payment,
                        "internet_service": internet,
                        "support_tickets": tickets,
                        "complaints": complaints,
                        "late_payments": late,
                        "avg_monthly_usage": usage,
                        "last_login_days": login,
                        "customer_satisfaction": satisfaction,
                    }
                ]
            )

            result = predict_customer(
                customer
            )

            if result["prediction"]:

                st.error(
                    "Customer is likely to churn."
                )

            else:

                st.success(
                    "Customer is likely to stay."
                )

            st.metric(
                "Churn Probability",
                f"{result['churn_probability']:.1%}",
            )

            st.info(
                f"Risk Category: {result['risk']}"
            )

    elif page == "Model Performance":

        metrics = pd.read_csv(
            METRICS_FILE
        )

        st.dataframe(
            metrics,
            use_container_width=True,
        )

    elif page == "SQL Insights":

        run_sql_queries()


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    if not DATA_FILE.exists():
        generate_customer_data()

    if not MODEL_FILE.exists():
        train_models()

    if not DATABASE_FILE.exists():
        create_database()
        
    run_dashboard()    
   