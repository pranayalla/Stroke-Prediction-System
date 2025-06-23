
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df.dropna()
    df = df.drop(columns=['id'])
    df['gender'] = df['gender'].replace('Other', 'Male')
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

df = load_data()

# Prepare data
X = df.drop("stroke", axis=1)
y = df["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("🧠 Stroke Prediction System")
st.markdown("Predict stroke risk based on patient information.")

with st.form("stroke_form"):
    age = st.slider("Age", 0, 100, 45)
    hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", [0, 1])
    ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
    work_type = st.selectbox("Work Type", ['Govt_job', 'Never_worked', 'Private', 'Self-employed'])
    residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
    avg_glucose = st.slider("Avg Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes'])

    submit = st.form_submit_button("Predict")

if submit:
    # Manual encoding for input
    input_data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'ever_married_Yes': 1 if ever_married == 'Yes' else 0,
        'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
        'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
        'work_type_Private': 1 if work_type == 'Private' else 0,
        'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
        'Residence_type_Urban': 1 if residence_type == 'Urban' else 0,
        'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
        'smoking_status_never smoked': 1 if smoking_status == 'never smoked' else 0,
        'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
        'gender_Male': 1  # default gender since 'Other' was dropped
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = lr.predict(input_scaled)[0]
    probability = lr.predict_proba(input_scaled)[0][1]

    st.success(f"Prediction: {'🚨 Stroke Risk' if prediction == 1 else '✅ No Stroke Risk'}")
    st.info(f"Probability of Stroke: {probability:.2%}")
