# 🏏 IPL Match Winner Predictor

An interactive machine learning app that predicts the winner of an IPL cricket match based on inputs like teams, toss decisions, and match city. The app also includes insightful visualizations like team-wise win percentage, season trends, toss impact, and more!

---

## 📌 Features

- ✅ Predict the winner of a match using XGBoost
- 🎯 Input form with team, toss, and city selection
- 📈 Interactive Plotly charts for historical insights
- 🧠 Team-wise win percentage & seasonal trends
- 🌆 Venue, toss decision, and home/away impact analysis
- 🎨 IPL-inspired color theme and logos

---

## 📁 Project Structure

ipl-match-predictor/
├── app/
│ └── streamlit_app.py # Streamlit app code
├── data/
│ └── ipl_data.xlsx # IPL dataset (replace with real cleaned data)
├── model/
│ ├── xgb_model.pkl # Trained XGBoost model
│ ├── feature_encoders.pkl # Label encoders for input features
│ └── winner_encoder.pkl # Label encoder for match_winner
├── notebooks/
│ └── train_model.py # Model training script
├── utils/
│ └── preprocessing.py # (Optional) Custom functions
├── README.md # This file
└── requirements.txt # Python dependencies


---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas** & **NumPy** for data processing
- **XGBoost** for classification model
- **Plotly** for visualizations
- **Streamlit** for web app interface

---

## 🚀 How to Run the App

   
✅ Step 1: Install Dependencies

pip install -r requirements.txt

✅ Step 2: Train the Model 

python notebooks/train_model.py

✅ Step 3: Launch the Streamlit App

streamlit run app/streamlit_app.py
