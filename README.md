
# 💼 Smart Salary Estimator

An interactive **Streamlit web application** that predicts and compares salaries based on user profiles, leveraging **Machine Learning** for accurate salary insights.

---

## 🚀 Features

* **🔮 Salary Prediction** – Estimate your monthly salary based on job title, experience, education, and other profile details.
* **📊 Market Comparison** – Compare your salary with the **average salary** for the same job title.
* **🧠 Machine Learning Powered** – Uses a trained Random Forest model with preprocessing (scaling, encoding, mappings).
* **💡 Intelligent Suggestions** – Automatically matches similar job titles if the input is slightly off.
* **📬 Feedback System** – Users can submit feedback directly from the app.

---

## 🖼️ Demo Preview

*(Add screenshots of your app interface here)*

```
📌 Example:
Predicted Salary: ₹ 85,000 / month
Average Salary: ₹ 82,500 / month
```

---

## 📦 Installation & Setup

Follow the steps below to run the app locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/smart-salary-estimator.git
cd smart-salary-estimator
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add Required Files

Make sure the following files are in the project folder:

* **Salary Data.csv** – Dataset with job titles and salaries
* **salary\_predictor\_rf\_model.pkl** – Trained Random Forest model
* **scaler.pkl, le\_gender.pkl, target\_encoder.pkl** – Preprocessing encoders
* **education\_mapping.pkl, seniority\_mapping.pkl** – Custom mappings for feature transformation

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then, open your browser at **[http://localhost:8501](http://localhost:8501)** 🎉

---

## 🧩 Project Structure

```
smart-salary-estimator/
│
├── app.py                         # Main Streamlit application
├── Salary Data.csv                 # Dataset
├── salary_predictor_rf_model.pkl   # Trained ML model
├── scaler.pkl                      # Scaler for numeric features
├── le_gender.pkl                   # Label encoder for gender
├── target_encoder.pkl              # Encoder for job titles
├── education_mapping.pkl           # Mapping for education levels
├── seniority_mapping.pkl           # Mapping for seniority levels
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 📊 Tech Stack

* **Frontend:** Streamlit, HTML/CSS (Custom Styling)
* **Backend:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Visualization:** Plotly Express
* **Storage:** Local CSV & Pickle files

---

## 📈 How It Works

1. **User Input** → Enter Job Title, Gender, Age, Experience, and Education Level.
2. **Data Preprocessing** → Encodes categorical variables & scales numerical features.
3. **Prediction** → ML model predicts **log-salary**, then converts it back to real salary.
4. **Average Salary** → Computes mean salary for the selected job title from the dataset.

---

## 📬 Feedback & Suggestions

You can submit your feedback directly in the app or connect via:

* [LinkedIn]
* [GitHub]
* [Twitter]

---

## ⭐ Acknowledgements

* Built with ❤️ by **Vikram** using [Streamlit](https://streamlit.io/)
* Icons by [Icons8](https://icons8.com/)




