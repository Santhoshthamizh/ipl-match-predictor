import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load model and encoders
model = joblib.load("../model/xgb_model.pkl")
encoders = joblib.load("../model/feature_encoders.pkl")
winner_encoder = joblib.load("../model/winner_encoder.pkl")

# Load dataset for charts
df_chart = pd.read_excel("data/ipl_data.xlsx")

df_chart['win_by_runs'] = df_chart['win_by_runs'].fillna(0)
df_chart['win_by_wickets'] = df_chart['win_by_wickets'].fillna(0)
df_chart['city'] = df_chart['city'].fillna("Unknown")
df_chart = df_chart[df_chart['match_winner'].notna()]

# Page title
st.set_page_config(page_title="🏏 IPL Match Predictor", layout="wide")
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: white;
            background-color: #0b1c2c;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .logo-row {
            text-align: center;
            margin-bottom: 20px;
        }
        .logo-row img {
            margin: 6px 8px;
            vertical-align: middle;
            border-radius: 4px;
        }
    </style>

    <div class="main-title">🏏 IPL Match Winner Predictor</div>

    <div class="logo-row">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQTMDeAC-Pse1QMyrvyvW4vLwxCNVfZhUXZg&s" width="50">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/c/cd/Mumbai_Indians_Logo.svg/800px-Mumbai_Indians_Logo.svg.png" width="50">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/5/5c/This_is_the_logo_for_Rajasthan_Royals%2C_a_cricket_team_playing_in_the_Indian_Premier_League_%28IPL%29.svg/640px-This_is_the_logo_for_Rajasthan_Royals%2C_a_cricket_team_playing_in_the_Indian_Premier_League_%28IPL%29.svg.png" width="50">
        <img src="https://images.seeklogo.com/logo-png/34/1/royal-challengers-bengaluru-logo-png_seeklogo-349568.png" width="50">
        <img src="https://c8.alamy.com/comp/2T6TGX3/kolkata-knight-riders-logo-indian-professional-cricket-club-vector-illustration-abstract-editable-image-2T6TGX3.jpg" width="50">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/a/a9/Lucknow_Super_Giants_IPL_Logo.svg/640px-Lucknow_Super_Giants_IPL_Logo.svg.png" width="50">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/2/2f/Delhi_Capitals.svg/800px-Delhi_Capitals.svg.png" width="50">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQsST4vmjlVXg96I0-4FKwlp3kNhd5QZJ-acA&s" width="50">
        <img src="https://upload.wikimedia.org/wikipedia/en/thumb/0/09/Gujarat_Titans_Logo.svg/800px-Gujarat_Titans_Logo.svg.png" width="50">
        <img src="https://www.punjabkingsipl.in/static-assets/waf-images/a3/63/8a/16-9/ItTZtPrSIp.jpg" width="50">
    </div>
""", unsafe_allow_html=True)


# Tabs
tabs = st.tabs(["🧠 Predict Winner", "📊 Insights & Charts"])

# ------------------------ Tab 1: Prediction ------------------------
with tabs[0]:
    st.markdown("### 🧠 Enter Match Details")

    teams = encoders['team1'].classes_
    cities = encoders['city'].classes_
    toss_decisions = encoders['toss_decision'].classes_

    col1, col2 = st.columns(2)

    with col1:
        team1 = st.selectbox("Select Team 1", teams)
        team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])

    with col2:
        toss_winner = st.selectbox("Select Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", toss_decisions)
        city = st.selectbox("Match City", cities)

    st.markdown("")

    if st.button("⚡ Predict Winner"):
        try:
            input_data = np.array([[
                encoders['team1'].transform([team1])[0],
                encoders['team2'].transform([team2])[0],
                encoders['toss_winner'].transform([toss_winner])[0],
                encoders['toss_decision'].transform([toss_decision])[0],
                encoders['city'].transform([city])[0]
            ]])
            pred_encoded = model.predict(input_data)[0]
            winner = winner_encoder.inverse_transform([pred_encoded])[0]
            st.success(f"🏆 Predicted Match Winner: **{winner}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------------ Tab 2: Insights ------------------------
with tabs[1]:
    st.markdown("### 📊 Match Insights")
    st.markdown("Get in-depth visualizations from IPL match data to analyze team performance and trends.")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🥇 Team-wise Win Count")
        win_counts = df_chart['match_winner'].value_counts().reset_index()
        win_counts.columns = ['Team', 'Wins']
        fig1 = px.bar(win_counts, x='Team', y='Wins', title='Team-wise Win Count', height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 🧢 Toss Decision Impact")
        toss_impact = df_chart.groupby('toss_decision')['match_winner'].count().reset_index()
        toss_impact.columns = ['Toss Decision', 'Wins']
        fig2 = px.pie(toss_impact, names='Toss Decision', values='Wins', title='Toss Decision Impact')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🏠 Home vs Away Matches")
        df_chart['home_team'] = df_chart.apply(
            lambda row: row['team1'] if row['city'] in row['venue'] else "Away", axis=1
        )
        home_away = df_chart['home_team'].value_counts().reset_index()
        home_away.columns = ['Type', 'Matches']
        fig3 = px.pie(home_away, names='Type', values='Matches', title='Home vs Away Matches')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### 📈 Matches per Season")
        season_trend = df_chart.groupby('season_id')['match_winner'].count().reset_index()
        season_trend.columns = ['Season', 'Matches']
        fig4 = px.line(season_trend, x='Season', y='Matches', markers=True, title='Matches per Season', height=350)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Row 3: Team-wise Win %
    st.markdown("### 📌 Team-wise Win Percentage")
    team_matches = df_chart.groupby(['team1']).size().add(df_chart.groupby(['team2']).size(), fill_value=0)
    team_wins = df_chart['match_winner'].value_counts()
    team_win_pct = (team_wins / team_matches).dropna().reset_index()
    team_win_pct.columns = ['Team', 'Win %']
    team_win_pct['Win %'] = (team_win_pct['Win %'] * 100).round(2)
    fig5 = px.bar(team_win_pct, x='Team', y='Win %', title='Team Win Percentage', color='Win %', height=400)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # Row 4: Year-wise Wins
    st.markdown("### 📆 Wins by Teams Over Seasons")
    year_team = df_chart.groupby(['season_id', 'match_winner']).size().reset_index(name='Wins')
    fig6 = px.bar(year_team, x='season_id', y='Wins', color='match_winner', title='Wins by Teams Over Seasons', height=450)
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown(" ")
