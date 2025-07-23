import streamlit as st
import pandas as pd
import joblib

# Load the trained XGBoost pipeline
xgb_pipeline = joblib.load("xgb_pipeline.pkl")  # Make sure this file is in the same folder

# Features used during training
feature_names = [
    'Hours_Studied', 'Attendance', 'Extracurricular_Activities', 'Previous_Scores',
    'Tutoring_Sessions', 'Family_Income', 'Peer_Influence', 'Distance_from_Home',
    'Parental_Involvement_Low', 'Parental_Involvement_Medium', 'Access_to_Resources_Low',
    'Access_to_Resources_Medium', 'Motivation_Level_Low', 'Motivation_Level_Medium',
    'Parental_Education_Level_High School', 'Parental_Education_Level_Postgraduate',
    'School_Type_Public', 'Teacher_Quality_Low', 'Teacher_Quality_Medium', 'Gender_Male',
    'Motivation_Level_Score', 'Study_Effort', 'Engagement_Index', 'Attendance_Tutoring_Interaction'
]

st.title("📚 Student Performance Prediction App")
st.write("Predict a student's academic performance using socio-academic features.")

# Create input fields
user_input = {}
for feature in feature_names:
    if any(x in feature for x in ['Low', 'Medium', 'High', 'Public', 'Male']):
        user_input[feature] = st.checkbox(f"{feature}", value=False)
    else:
        user_input[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=100.0, step=0.1, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("🔮 Predict Performance"):
    prediction = xgb_pipeline.predict(input_df)[0]
    st.success(f"🎯 Predicted Category: **{prediction}**")
