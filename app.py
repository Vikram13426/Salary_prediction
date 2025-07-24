import streamlit as st
import pandas as pd
import numpy as np
import joblib
from difflib import get_close_matches
import plotly.express as px

# Cache data loading for performance
@st.cache_data
def load_data():
    try:
        salary_df = pd.read_csv("Salary Data.csv")
        unique_job_titles = sorted(salary_df["Job Title"].dropna().unique())
        return salary_df, unique_job_titles
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

# Set page configuration
st.set_page_config(page_title="💼 Smart Salary Estimator", page_icon="💰", layout="centered")

# Load model and preprocessors
try:
    model = joblib.load("salary_predictor_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_gender = joblib.load("le_gender.pkl")
    te = joblib.load("target_encoder.pkl")
    education_mapping = joblib.load("education_mapping.pkl")
    seniority_mapping = joblib.load("seniority_mapping.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or preprocessors: {str(e)}")
    st.stop()

# Load dataset
salary_df, unique_job_titles = load_data()

# Header
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color:#2E86C1;'>💼 Smart Salary Estimator</h1>
        <p style='font-size:18px;'>Enter your profile details to estimate your salary or compare with the market average.</p>
    </div>
""", unsafe_allow_html=True)

# Input Form
with st.container():
    st.markdown("---")
    st.subheader("👤 Enter Your Details")

    col1, col2 = st.columns(2)
    with col1:
        job_title = st.selectbox("💼 Job Title", options=[""] + unique_job_titles)
        education_level = st.selectbox("🎓 Education Level", options=["", "Bachelor's", "Master's", "PhD"])
        gender = st.selectbox("🧍 Gender", options=["", "Male", "Female"])
    with col2:
        years_experience = st.slider("🛠️ Years of Experience", 0.0, 25.0, 0.0, step=0.1)
        age = st.slider("📅 Age", 18, 65, 22)

    st.markdown("---")

    # Center buttons with CSS
    st.markdown("""
    <style>
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .stButton > button {
            width: 200px;
            height: 40px;
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_button = st.button("🔮 Predict Salary")
        with col_btn2:
            avg_salary_button = st.button("📊 Get Average Salary")
        st.markdown('</div>', unsafe_allow_html=True)

# Seniority Level function
def get_seniority_level(title):
    title = title.lower()
    if 'junior' in title or 'entry' in title:
        return 'Junior'
    elif 'senior' in title:
        return 'Senior'
    elif 'manager' in title or 'lead' in title:
        return 'Manager'
    elif 'director' in title or 'vp' in title or 'chief' in title or 'ceo' in title:
        return 'Executive'
    else:
        return 'Mid'

# Predict Salary Logic
if predict_button:
    if not job_title or not education_level or not gender:
        st.error("⚠️ Please fill all the fields.")
    else:
        try:
            if job_title not in unique_job_titles:
                closest = get_close_matches(job_title, unique_job_titles, n=1, cutoff=0.6)
                if closest:
                    job_title = closest[0]
                    st.warning(f"🔍 Using closest job title match: **{job_title}**")
                else:
                    st.error("🚫 Invalid Job Title.")
                    st.stop()

            # Create input DataFrame
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Education Level': [education_level],
                'Job Title': [job_title],
                'Years of Experience': [years_experience],
                'Seniority_Level': [get_seniority_level(job_title)],
                'Weighted_Exp': [years_experience ** 2],
                'Log_Exp': [np.log1p(years_experience)]
            })

            # Preprocess input
            input_data['Gender'] = le_gender.transform(input_data['Gender'])
            input_data['Education Level'] = input_data['Education Level'].map(education_mapping)
            input_data['Seniority_Level'] = input_data['Seniority_Level'].map(seniority_mapping)
            input_data['Job Title'] = te.transform(input_data['Job Title'])
            input_data[['Age']] = scaler.transform(input_data[['Age']])

            # Predict
            pred_log = model.predict(input_data)
            predicted_salary = np.expm1(pred_log)[0]

            # Display prediction
            st.markdown(f"""
            <div style='text-align:center; margin-top:30px;'>
                <h2 style='color:#28a745;'>🤑 Predicted Salary</h2>
                <p style='font-size:36px; font-weight:bold;'>₹ {predicted_salary:,.2f} / month</p>
            </div>
            """, unsafe_allow_html=True)

            

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Average Salary Logic
if avg_salary_button:
    if not job_title:
        st.error("⚠️ Please select a Job Title.")
    else:
        try:
            if job_title not in unique_job_titles:
                closest = get_close_matches(job_title, unique_job_titles, n=1, cutoff=0.6)
                if closest:
                    job_title = closest[0]
                    st.warning(f"🔍 Using closest job title match: **{job_title}**")
                else:
                    st.error("🚫 Invalid Job Title.")
                    st.stop()

            avg_salary = salary_df[salary_df["Job Title"] == job_title]["Salary"].mean()
            if pd.isna(avg_salary):
                st.error("🚫 No salary data available for this job title.")
            else:
                st.markdown(f"""
                <div style='text-align:center; margin-top:30px;'>
                    <h2 style='color:#1F618D;'>📊 Average Salary for <span style='color:#154360'>{job_title}</span></h2>
                    <p style='font-size:36px; font-weight:bold;'>₹ {avg_salary:,.2f} / month</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Feedback Form
with st.container():
    st.markdown("---")
    st.subheader("📬 Feedback")
    with st.form("feedback_form"):
        name = st.text_input("Name (optional)")
        feedback = st.text_area("Your Feedback or Suggestions")
        submit_feedback = st.form_submit_button("Submit Feedback")
        if submit_feedback:
            if feedback:
                st.success("✅ Thank you for your feedback!")
                # In a production app, save feedback to a database or file
            else:
                st.error("⚠️ Please provide feedback before submitting.")

# Footer with larger social media icons
st.markdown("""
<hr>
<div style='text-align:center; color:gray;'>
    <p>Built by Vikram with ❤️ using <a href='https://streamlit.io/' target='_blank'>Streamlit</a></p>
    <p>
        <a href='https://www.linkedin.com/in/your-linkedin-profile' target='_blank'>
            <img src='https://img.icons8.com/color/36/000000/linkedin.png' alt='LinkedIn' width='36' height='36'/>
        </a>
        <a href='https://github.com/your-github-profile' target='_blank'>
            <img src='https://img.icons8.com/color/36/000000/github.png' alt='GitHub' width='36' height='36'/>
        </a>
        <a href='https://x.com/your-twitter-profile' target='_blank'>
            <img src='https://img.icons8.com/color/36/000000/twitter.png' alt='Twitter' width='36' height='36'/>
        </a>
    </p>
</div>
""", unsafe_allow_html=True)