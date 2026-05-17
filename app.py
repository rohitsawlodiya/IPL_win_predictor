import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
pipe = pickle.load(open("pipe.pkl", "rb"))

st.title("IPL Win Predictor")

# Team options
teams = [
    'Royal Challengers Bangalore',
    'Punjab Kings',
    'Chennai Super Kings',
    'Sunrisers Hyderabad',
    'Delhi Capitals',
    'Mumbai Indians',
    'Kolkata Knight Riders',
    'Rajasthan Royals',
    'Lucknow Super Giants',
    'Gujarat Titans'
]

cities = [
    'Mumbai',
    'Kolkata',
    'Delhi',
    'Chennai',
    'Hyderabad',
    'Bangalore',
    'Jaipur',
    'Chandigarh',
    'Pune',
    'Dubai',
    'Ahmedabad',
    'Abu Dhabi',
    'Sharjah',
    'Lucknow',
    'Visakhapatnam',
    'Durban',
    'Dharamsala',
    'Centurion',
    'Rajkot',
    'Indore',
    'Navi Mumbai',
    'Mohali'
]

# Layout columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Batting Team", teams)
    city = st.selectbox("City", cities)
    runs_left = st.number_input("Runs Left")

with col2:
    bowling_team = st.selectbox("Bowling Team", teams)
    balls_left = st.number_input("Balls Left")
    wickets_left = st.number_input("Wickets Left")

current_rr = st.number_input("Current Run Rate")
required_rr = st.number_input("Required Run Rate")

# Prediction button
if st.button("Predict Probability"):

    input_df = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "city": city,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": wickets_left,
        "current_run_rate": current_rr,
        "required_run_rate": required_rr
    }])

    result = pipe.predict_proba(input_df)

    loss_prob = result[0][0]
    win_prob = result[0][1]

    st.write(
        f"{batting_team} need {runs_left} runs in "
        f"{balls_left} balls with {wickets_left} wickets left."
    )

    st.subheader(f"{batting_team} Win Probability")
    st.progress(float(win_prob))
    st.write(f"{round(win_prob * 100)}%")

    st.subheader(f"{bowling_team} Win Probability")
    st.progress(float(loss_prob))
    st.write(f"{round(loss_prob * 100)}%")
