
import streamlit as st
import pandas as pd
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")

model = load_model()

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer information to estimate the likelihood "
    "of leaving the telecom service."
)

st.info(
    "The prediction will appear only after you click "
    "'Predict Churn'."
)

st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Personal Details")

        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        st.markdown("#### Phone and Internet")

        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        st.markdown("#### Services")

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        st.markdown("#### Contract and Billing")

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        st.markdown("#### Charges")

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        # Approximation only; actual billing history may differ.
        estimated_total = round(tenure * monthly_charges, 2)

        st.caption(
            f"Estimated Total Charges: ${estimated_total:,.2f}"
        )

        total_charges = st.number_input(
            "Actual Total Charges ($)",
            min_value=0.0,
            value=float(estimated_total),
            step=10.0,
            help=(
                "Use the customer's actual accumulated charges "
                "when available. Otherwise, use the estimate above."
            )
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )

# IMPORTANT: Predictions happen only after form submission.
if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        prediction = model.predict(customer)[0]
        probabilities = model.predict_proba(customer)[0]

        classes = model.named_steps["classifier"].classes_
        churn_probability = (
            probabilities[list(classes).index("Yes")] * 100
        )

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "Yes":
            st.error("⚠️ The model predicts that this customer may churn.")
        else:
            st.success("✅ The model predicts that this customer may stay.")

        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.2f}%"
        )

        st.progress(int(round(churn_probability)))

        if churn_probability >= 50:
            st.warning(
                "The model estimates a relatively high churn probability."
            )
        else:
            st.info(
                "The model estimates a lower churn probability."
            )

        with st.expander("View submitted customer details"):
            st.dataframe(customer, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption(
            "Check that the model file and its scikit-learn version "
            "are compatible."
        )
