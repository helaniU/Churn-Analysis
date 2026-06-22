import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="ChurnSense AI", page_icon="📊", layout="wide")


# -----------------------------
# STYLES
# -----------------------------
st.markdown(
    """
    <style>
        .css-1lcbmhc.e1fqkh3o3 {background-color: #0b0f14;}
        .stApp {
            background-color: #0b0f14;
            color: #E6EEF3;
        }
        .stButton>button {
            background-color: #0ea5a4;
            color: white;
        }
        div[data-testid="metric-container"] {
            background-color: #0f1724;
            border-radius: 8px;
            padding: 8px 12px;
        }
        .stSidebar {
            background-color: #071024;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📊 ChurnSense AI Dashboard")
st.caption("Smart Customer Churn Analytics & Prediction System")


# -----------------------------
# SIDEBAR: Upload + Navigation
# -----------------------------
with st.sidebar:
    st.header("Dataset & Navigation")
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    menu = st.radio("Navigation", ["Overview", "Analytics", "Prediction"])
    st.markdown("---")
    st.write("Tip: CSV should contain a `Churn` column with Yes/No or 1/0 values.")


if uploaded_file is None:
    st.info("Please upload a CSV file to begin. Example: Telco customer churn dataset.")
    st.stop()


# -----------------------------
# Load data
# -----------------------------
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()


def find_column_case_insensitive(df, name):
    for c in df.columns:
        if c.strip().lower() == name.strip().lower():
            return c
    return None


churn_col = find_column_case_insensitive(df, "churn")
if churn_col is None:
    st.error("'Churn' column not found (case-insensitive). Please include a Churn column.")
    st.stop()


def compute_churn_rate(series):
    if series.dtype == object:
        return series.eq("Yes").mean() * 100
    numeric = pd.to_numeric(series, errors="coerce")
    if pd.isna(numeric.mean()):
        return 0.0
    return numeric.mean() * 100


churn_rate = compute_churn_rate(df[churn_col])


# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", int(df.shape[0]))
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Features", int(df.shape[1]))
col4.metric("Missing Values", int(df.isnull().sum().sum()))


# -----------------------------
# Data preparation & model training
# -----------------------------
def prepare_model(df, churn_col):
    df2 = df.copy()

    # Normalize churn to numeric 0/1
    df2[churn_col] = df2[churn_col].replace({"Yes": 1, "No": 0})
    df2[churn_col] = pd.to_numeric(df2[churn_col], errors="coerce")
    df2 = df2.dropna(subset=[churn_col])

    encoders = {}

    # Encode categorical, coerce numeric
    for col in df2.columns:
        if col == churn_col:
            continue
        if not is_numeric_dtype(df2[col]):
            le = LabelEncoder()
            df2[col] = df2[col].astype(str).fillna("__NA__")
            df2[col] = le.fit_transform(df2[col])
            encoders[col] = le
        else:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

    # Fill numeric NaNs with median (excluding target)
    for col in df2.select_dtypes(include=[np.number]).columns:
        if col == churn_col:
            continue
        df2[col] = df2[col].fillna(df2[col].median())

    X = df2.drop(columns=[churn_col])
    y = df2[churn_col].astype(int)

    if X.shape[0] < 10 or X.shape[1] == 0:
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return {
        "model": model,
        "encoders": encoders,
        "X_cols": X.columns.tolist(),
        "accuracy": acc,
        "processed": df2,
    }


model_info = prepare_model(df, churn_col)


if menu == "Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Data Summary")
    with st.expander("Descriptive statistics"):
        st.write(df.describe(include="all").transpose())


if model_info is None:
    st.warning("Not enough data to train a dependable model. Upload a larger/cleaner dataset.")


if menu == "Analytics" and model_info is not None:
    st.subheader("Churn Distribution")
    fig1 = px.pie(df, names=churn_col, hole=0.4, title="Churn Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    # Tenure (case-insensitive lookup)
    tenure_col = find_column_case_insensitive(df, "tenure")
    if tenure_col is not None:
        st.subheader("Tenure vs Churn")
        fig2 = px.histogram(df, x=tenure_col, color=churn_col, barmode="overlay", title="Tenure vs Churn")
        st.plotly_chart(fig2, use_container_width=True)

    # Contract
    contract_col = find_column_case_insensitive(df, "contract")
    if contract_col is not None:
        st.subheader("Contract Type vs Churn")
        fig3 = px.histogram(df, x=contract_col, color=churn_col, barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        st.subheader("Correlation Heatmap")
        corr_fig = px.imshow(numeric_df.corr(), text_auto=True)
        st.plotly_chart(corr_fig, use_container_width=True)

    st.markdown(f"**Model accuracy (test set):** {model_info['accuracy']:.2%}")

    # -----------------------------
    # Bulk churn predictions (Customers at Risk)
    # -----------------------------
    st.subheader("📋 Customers at Risk of Churn")

    # Use processed data that matches training preprocessing
    X_all = model_info["processed"].drop(columns=[churn_col], errors="ignore")
    try:
        preds_all = model_info["model"].predict(X_all)
        # attach readable labels back to original df (preserve index)
        df.loc[X_all.index, "Churn_Predicted"] = preds_all
        df["Churn_Predicted"] = df["Churn_Predicted"].map({1: "Will Churn ⚠", 0: "Will Stay ✅"})

        st.write("### High Risk Customers")
        risk_df = df[df["Churn_Predicted"] == "Will Churn ⚠"]
        st.dataframe(risk_df, use_container_width=True)

        fig = px.histogram(
            df,
            x="Churn_Predicted",
            color="Churn_Predicted",
            title="Predicted Customer Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Bulk prediction failed: {e}")


if menu == "Prediction" and model_info is not None:
    st.subheader("Predict Customer Churn")

    model = model_info["model"]
    encoders = model_info["encoders"]
    X_cols = model_info["X_cols"]
    processed = model_info["processed"]

    # Optional: customer selector with search to autofill form
    id_candidates = ["customerid", "customer id", "customer_id", "customer", "id"]
    id_col = None
    for cand in id_candidates:
        id_col = find_column_case_insensitive(df, cand)
        if id_col:
            break

    customer_row = None
    if id_col is not None:
        id_options = df[id_col].dropna().astype(str).unique().tolist()
        id_options = sorted(id_options)
        # use a session-backed selectbox so we can detect changes
        selected = st.selectbox(f"Select {id_col}", ["-- Manual Entry --"] + id_options, key="selected_customer")

        if selected and selected != "-- Manual Entry --":
            matches = df[df[id_col].astype(str) == selected]
            if not matches.empty:
                customer_row = matches.iloc[0]
            else:
                customer_row = None
        else:
            customer_row = None

    # If the selected customer changed, populate session_state defaults
    prev = st.session_state.get("selected_customer_prev", None)
    curr = st.session_state.get("selected_customer", None)
    if curr and curr != "-- Manual Entry --" and curr != prev and id_col is not None:
        matches = df[df[id_col].astype(str) == curr]
        if not matches.empty:
            customer_row = matches.iloc[0]
            for col in X_cols:
                key = f"input_{col}"
                if col in encoders:
                    # prefer the raw string value from the original df
                    val = str(customer_row[col]) if col in customer_row.index else ""
                    opts = sorted(df[col].dropna().astype(str).unique().tolist())
                    st.session_state[key] = val if val in opts else (opts[0] if opts else "")
                else:
                    num = pd.to_numeric(customer_row[col], errors="coerce") if col in customer_row.index else None
                    default = float(processed[col].median()) if col in processed.columns else 0.0
                    st.session_state[key] = float(num) if num is not None and not pd.isna(num) else default
            st.session_state["selected_customer_prev"] = curr

    # Initialize session state defaults if missing
    for col in X_cols:
        key = f"input_{col}"
        if key not in st.session_state:
            if col in encoders:
                opts = sorted(df[col].dropna().astype(str).unique().tolist())
                st.session_state[key] = opts[0] if opts else ""
            else:
                st.session_state[key] = float(processed[col].median()) if col in processed.columns else 0.0

    # Build form using session_state-backed widgets (so they update when we set session_state)
    with st.form("predict_form"):
        for col in X_cols:
            key = f"input_{col}"
            if col in encoders:
                opts = sorted(df[col].dropna().astype(str).unique().tolist())
                if opts:
                    _ = st.selectbox(col, opts, key=key)
                else:
                    _ = st.text_input(col, value=st.session_state[key], key=key)
            else:
                _ = st.number_input(col, value=st.session_state[key], format="%.2f", key=key)

        submitted = st.form_submit_button("Run Prediction 🚀")

    if submitted:
        # Build input record matching training preprocessing
        record = {}
        for col in X_cols:
            key = f"input_{col}"
            if col in encoders:
                val = str(st.session_state.get(key, ""))
                le = encoders[col]
                if val in le.classes_:
                    record[col] = int(le.transform([val])[0])
                else:
                    modes = df[col].mode()
                    if len(modes) > 0:
                        mode_val = str(modes.iloc[0])
                        if mode_val in le.classes_:
                            record[col] = int(le.transform([mode_val])[0])
                        else:
                            record[col] = 0
                    else:
                        record[col] = 0
            else:
                num = pd.to_numeric(st.session_state.get(key, None), errors="coerce")
                if pd.isna(num):
                    num = processed[col].median() if col in processed.columns else 0.0
                record[col] = float(num)

        input_df = pd.DataFrame([record])[X_cols]

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        if pred == 1:
            st.error("⚠️ Predicted: Customer WILL CHURN")
        else:
            st.success("✅ Predicted: Customer will NOT churn")

        if proba is not None:
            st.info(f"Estimated probability of churn: {proba:.2%}")


st.markdown("---")
st.caption("Built with ChurnSense AI — clean, professional Streamlit UI")
