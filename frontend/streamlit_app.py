import streamlit as st
import requests

st.set_page_config(page_title="Credit Scoring API", page_icon="💳", layout="wide")
st.title("💳 Advanced Credit Scoring Dashboard")

st.markdown("Enter the applicant's financial and personal details to evaluate credit default risk.")

col1, col2 = st.columns(2)

with col1:
    person_age = st.slider("Age", 18, 100, 25)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=5.0, step=0.5)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=3)
    cb_person_default_on_file = st.selectbox("Historical Default on File", ["Y", "N"])

with col2:
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

# Calculate Loan Percent Income automatically to provide better UX
loan_percent_income = loan_amnt / (person_income if person_income > 0 else 1)
st.info(f"Calculated Loan Percent Income: **{loan_percent_income:.2f}**")

if st.button("Check Credit Score", type="primary"):
    url = "http://127.0.0.1:5000/predict"

    data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        prediction = result.get("prediction")
        probability = result.get("probability", None)
        
        # 0 = No Default (Good), 1 = Default (Bad)
        if prediction == 0:
            st.success("✅ **Approved (Low Risk)**")
            if probability is not None:
                st.write(f"Confidence (Probability of Default): {probability:.2%}")
        else:
            st.error("❌ **Rejected (High Risk / Default)**")
            if probability is not None:
                st.write(f"Confidence (Probability of Default): {probability:.2%}")
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend API. Is the Flask app running? Details: {e}")