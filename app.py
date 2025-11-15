import streamlit as st
import numpy as np
import joblib

# Load model pipeline (StandardScaler + LogisticRegression)
model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction App")

st.markdown("""
### 📌 Model Summary  
This is a **Logistic Regression–based Diabetes Prediction Model** built using the **CDC BRFSS dataset**.

- **Accuracy:** 0.7458  
- **F1 Score:** 0.7503  
- **Algorithm:** Logistic Regression  
- **Preprocessing:** StandardScaler  
- **Notes:** This is the baseline model (no PCA, no feature selection).
- **Dataset used to train the model : https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data?       select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv
                                        
- **By Pasan S.Dharmadasa

---
""")

st.write("Provide the following details:")

# INPUT FIELDS (21 features)
HighBP = st.selectbox("High Blood Pressure (0 = No, 1 = Yes)", [0, 1])
HighChol = st.selectbox("High Cholesterol (0 = No, 1 = Yes)", [0, 1])
CholCheck = st.selectbox("Cholesterol Check in last 5 yrs (0 = No, 1 = Yes)", [0, 1])
BMI = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)
Smoker = st.selectbox("Smoker (0 = No, 1 = Yes)", [0, 1])
Stroke = st.selectbox("Stroke History (0 = No, 1 = Yes)", [0, 1])
HeartDiseaseorAttack = st.selectbox("Heart Disease or Heart Attack (0 = No, 1 = Yes)", [0, 1])
PhysActivity = st.selectbox("Physical Activity (0 = No, 1 = Yes)", [0, 1])
Fruits = st.selectbox("Consumes Fruits Daily (0 = No, 1 = Yes)", [0, 1])
Veggies = st.selectbox("Consumes Vegetables Daily (0 = No, 1 = Yes)", [0, 1])
HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption (0 = No, 1 = Yes)", [0, 1])
AnyHealthcare = st.selectbox("Has Healthcare Access (0 = No, 1 = Yes)", [0, 1])
NoDocbcCost = st.selectbox("Avoided Doctor Due to Cost (0 = No, 1 = Yes)", [0, 1])
GenHlth = st.slider("General Health (1 = Best, 5 = Worst)", 1, 5, 3)
MentHlth = st.slider("Bad Mental Health Days (0–30)", 0, 30, 0)
PhysHlth = st.slider("Bad Physical Health Days (0–30)", 0, 30, 0)
DiffWalk = st.selectbox("Difficulty Walking (0 = No, 1 = Yes)", [0, 1])
Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
Age = st.slider("Age Group (1 = 18-24 ... 13 = 80+)", 1, 13, 4)
Education = st.slider("Education (1–6)", 1, 6, 4)
Income = st.slider("Income Level (1–8)", 1, 8, 4)

# FEATURE ARRAY SHAPE (1,21)
features = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                      HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                      HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
                      MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]])

if st.button("Predict Diabetes Risk"):
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    st.write("---")
    st.write("### 🔍 Prediction Result")

    if pred == 1:
        st.error(f"**High Risk of Diabetes** (Probability: {prob:.2f})")
    else:
        st.success(f"**Low Risk of Diabetes** (Probability: {prob:.2f})")
