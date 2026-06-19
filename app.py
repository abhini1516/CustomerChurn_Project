

st.set_page_config(page_title="Customer Churn Intelligence Platform", layout="wide")

# 1. LOAD SERIALIZED MODEL ARTIFACTS
@st.cache_resource
def load_model():
    with open("data/churn_model.pkl", "rb") as f:
        preprocessor, model, feature_names = pickle.load(f)
    return preprocessor, model, feature_names

try:
    preprocessor, model, feature_names = load_model()
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure data/churn_model.pkl exists.")
    st.stop()

# 2. WEB APP LAYOUT
st.title("📊 Customer Churn Intelligence Platform")
st.markdown("An enterprise-grade optimization tool designed to analyze customer risk metrics and protect contract revenue streams.")
st.markdown("---")

# 3. INTERACTIVE SIDEBAR FOR CUSTOMER PROFILE INPUT
st.sidebar.header("Customer Profile Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
cltv = st.sidebar.slider("Customer Lifetime Value (CLTV)", 2000, 6500, 3500)

# Static geographic midpoints for runtime validation
latitude = 34.0522
longitude = -118.2437
total_charges = tenure * monthly_charges

# 4. CONSTRUCT DATAFRAME MATCHING ORIGINAL TRAINING SCHEMA
input_data = pd.DataFrame([{
    'City': 'Los Angeles', 'Zip Code': 90001, 'Lat Long': '34.0522, -118.2437',
    'Latitude': latitude, 'Longitude': longitude, 'Gender': gender, 
    'Senior Citizen': senior_citizen, 'Partner': partner, 'Dependents': dependents, 
    'Tenure Months': tenure, 'Phone Service': phone_service, 'Multiple Lines': multiple_lines, 
    'Internet Service': internet_service, 'Online Security': online_security, 'Online Backup': online_backup, 
    'Device Protection': device_protection, 'Tech Support': tech_support, 'Streaming TV': streaming_tv, 
    'Streaming Movies': streaming_movies, 'Contract': contract, 'Paperless Billing': paperless_billing, 
    'Payment Method': payment_method, 'Monthly Charges': monthly_charges, 'Total Charges': total_charges, 
    'CLTV': cltv
}])

# 5. EXECUTE PIPELINE TRANSFORMATIONS & LIVE INFERENCE
processed_input = preprocessor.transform(input_data)
risk_probability = model.predict_proba(processed_input)[0][1]
prediction = model.predict(processed_input)[0]

# 6. RENDER ANALYSIS DASHBOARD
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Risk Inference Output")
    metric_color = "red" if risk_probability > 0.5 else "green"
    st.markdown(f"### Attrition Risk Score: <span style='color:{metric_color}'>{risk_probability*100:.2f}%</span>", unsafe_allow_html=True)

    if prediction == 1:
        st.error(" HIGH RISK ACTION REQUIRED: This customer exhibits behavioral attributes heavily correlated with near-term cancellation.")
    else:
        st.success("STABLE ACCOUNT: This customer profile tracks closely with historical baseline customer retention.")

with col2:
    st.subheader("Financial Impact Assessment")
    estimated_loss = cltv if prediction == 1 else 0.0
    st.metric(label="At-Risk Revenue Value (CLTV)", value=f"${cltv:,.2f}", delta=f"${estimated_loss:,.2f}" if prediction == 1 else "$0.00", delta_color="inverse")
    st.caption("Financial prioritization matrix: Flag accounts where high risk score intersects with top-tier CLTV values.")
