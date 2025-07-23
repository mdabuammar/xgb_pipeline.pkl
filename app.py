import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load pipeline
xgb_pipeline = joblib.load("xgb_pipeline.pkl")

# Feature list (must match training)
features = [
    'Hours_Studied', 'Attendance', 'Extracurricular_Activities', 'Previous_Scores',
    'Tutoring_Sessions', 'Family_Income', 'Peer_Influence', 'Distance_from_Home',
    'Parental_Involvement_Low', 'Parental_Involvement_Medium', 'Access_to_Resources_Low',
    'Access_to_Resources_Medium', 'Motivation_Level_Low', 'Motivation_Level_Medium',
    'Parental_Education_Level_High School', 'Parental_Education_Level_Postgraduate',
    'School_Type_Public', 'Teacher_Quality_Low', 'Teacher_Quality_Medium', 'Gender_Male',
    'Motivation_Level_Score', 'Study_Effort', 'Engagement_Index', 'Attendance_Tutoring_Interaction'
]

# Streamlit UI
st.title("🎓 Student Performance Predictor with SHAP Insights")
st.write("Predict performance and get top suggestions for improvement.")

# Input widgets
user_input = {}
for col in features:
    if any(key in col for key in ['Low', 'Medium', 'High', 'Male', 'Public']):
        user_input[col] = st.checkbox(f"{col}", value=False)
    else:
        user_input[col] = st.number_input(f"{col}", 0.0, 100.0, step=0.1, value=0.0)

input_df = pd.DataFrame([user_input])

# Prediction and SHAP
if st.button("🔍 Predict and Explain"):
    pred = xgb_pipeline.predict(input_df)[0]
    st.success(f"🎯 Predicted Category: **{pred}**")

    # SHAP Explanation
    explainer = shap.TreeExplainer(xgb_pipeline.named_steps['xgb'])
    shap_values = explainer.shap_values(input_df)

    st.subheader("📊 SHAP Feature Importance (Bar Plot)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    st.pyplot(fig)

    # Prescription
    def generate_prescription(index, X_sample, shap_values, top_k=3):
        student_data = X_sample.iloc[index]
        shap_vals = shap_values[index]
        impact_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'Value': student_data.values,
            'SHAP_Impact': shap_vals
        })
        impact_df['Abs_Impact'] = impact_df['SHAP_Impact'].abs()
        top_features = impact_df.sort_values(by='Abs_Impact', ascending=False).head(top_k)

        st.subheader("🧠 Prescription for the Student")
        for _, row in top_features.iterrows():
            direction = "increase" if row['SHAP_Impact'] < 0 else "maintain/improve"
            st.markdown(f"📌 **{row['Feature']}**: *{direction}* (Current: `{row['Value']}`) — Impact: `{row['SHAP_Impact']:.2f}`")

    generate_prescription(0, input_df, shap_values)
