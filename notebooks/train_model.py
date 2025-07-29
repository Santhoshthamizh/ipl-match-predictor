import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load dataset
df = pd.read_excel("data/ipl_data.xlsx")
df = df.head(1100)  # Use more rows later for better accuracy

# Fill nulls
df['win_by_runs'] = df['win_by_runs'].fillna(0)
df['win_by_wickets'] = df['win_by_wickets'].fillna(0)
df['city'] = df['city'].fillna("Unknown")

# Encode categorical columns
categorical_cols = ['team1', 'team2', 'toss_winner', 'toss_decision', 'city', 'match_winner']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Final features (no venue, no win_by_runs/wickets)
X = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'city']]
y = df['match_winner']
winner_encoder = encoders['match_winner']

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(n_estimators=10, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)

# Save model & encoders
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, f"{model_dir}/xgb_model.pkl")
joblib.dump(encoders, f"{model_dir}/feature_encoders.pkl")
joblib.dump(winner_encoder, f"{model_dir}/winner_encoder.pkl")

print("🎉 Model and encoders saved successfully!")
