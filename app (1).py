import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction")
st.sidebar.title("About")

st.sidebar.info("""
This application predicts whether a customer
is likely to churn using a Machine Learning model.
""")

model=joblib.load("model.pkl")
preprocessor=joblib.load("preprocessor.pkl")

c1,c2=st.columns(2)
st.header("Customer Information")
with c1:
    gender=st.selectbox("Gender",["Male","Female"])
    senior=st.selectbox("Senior Citizen",[0,1])
    partner=st.selectbox("Partner",["Yes","No"])
    dependents=st.selectbox("Dependents",["Yes","No"])
    st.header("Services")
    phone=st.selectbox("Phone Service",["Yes","No"])
    multiple=st.selectbox("Multiple Lines",["Yes","No","No phone service"])
    internet=st.selectbox("Internet Service",["DSL","Fiber optic","No"])
    security=st.selectbox("Online Security",["Yes","No","No internet service"])
    backup=st.selectbox("Online Backup",["Yes","No","No internet service"])
    device=st.selectbox("Device Protection",["Yes","No","No internet service"])
with c2:
    tech=st.selectbox("Tech Support",["Yes","No","No internet service"])
    tv=st.selectbox("Streaming TV",["Yes","No","No internet service"])
    movies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])
    st.header("Billing")
    contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
    paperless=st.selectbox("Paperless Billing",["Yes","No"])
    payment=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    tenure=st.number_input("Tenure Months",0,100,12)
    monthly=st.number_input("Monthly Charges",0.0,500.0,70.0)
    total=st.number_input("Total Charges",0.0,100000.0,1000.0)
    cltv=st.number_input("CLTV",0,10000,3000)
if st.button("Predict"):

    data = pd.DataFrame({
        "Gender": [gender],
        "Senior Citizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "Phone Service": [phone],
        "Multiple Lines": [multiple],
        "Internet Service": [internet],
        "Online Security": [security],
        "Online Backup": [backup],
        "Device Protection": [device],
        "Tech Support": [tech],
        "Streaming TV": [tv],
        "Streaming Movies": [movies],
        "Contract": [contract],
        "Paperless Billing": [paperless],
        "Payment Method": [payment],
        "Tenure Months": [tenure],
        "Monthly Charges": [monthly],
        "Total Charges": [total],
        "CLTV": [cltv],
        "Count": [1]
    })

    X = preprocessor.transform(data)

    pred = model.predict(X)[0]

    prob = model.predict_proba(X)[0]

    if pred == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.subheader("Prediction Probability")

    st.write(f"🟢 Stay Probability: {prob[0]:.2%}")
    st.write(f"🔴 Churn Probability: {prob[1]:.2%}")

    st.progress(int(prob[1] * 100))

    st.metric(
        "Churn Probability",
        f"{prob[1] * 100:.1f}%"
    )

