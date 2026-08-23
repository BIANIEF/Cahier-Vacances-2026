import streamlit as st
import pandas as pd
import joblib

from utils import get_current_form

# Charger les données
df = pd.read_csv("data/results.csv")

# Charger le modèle et le scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "diff_avg_points",
    "diff_avg_goals_scored",
    "diff_avg_goals_conceded",
    "is_neutral",
    "is_friendly",
]


def predict_match(team_a, team_b):

    form_a = get_current_form(df, team_a)
    form_b = get_current_form(df, team_b)

    features = pd.DataFrame(
        [[
            form_a["avg_points"] - form_b["avg_points"],
            form_a["avg_goals_scored"] - form_b["avg_goals_scored"],
            form_a["avg_goals_conceded"] - form_b["avg_goals_conceded"],
            1,
            0
        ]],
        columns=FEATURES
    )

    features_scaled = scaler.transform(features)

    probability = model.predict_proba(features_scaled)
    proba_a_wins = probability[0][1]

    if proba_a_wins >= 0.5:
        return team_a, float(proba_a_wins)

    return team_b, float(1 - proba_a_wins)


# -------------------------
# INTERFACE STREAMLIT
# -------------------------

st.title("⚽ Football Match Predictor")

teams = sorted(
    set(df["home_team"]) | set(df["away_team"])
)

team_a = st.selectbox("Équipe A", teams)
team_b = st.selectbox("Équipe B", teams)


if st.button("Prédire"):

    if team_a == team_b:
        st.error("Choisis deux équipes différentes.")

    else:

        # Récupérer la forme récente
        form_a = get_current_form(df, team_a)
        form_b = get_current_form(df, team_b)

        # -------------------------
        # FORME RÉCENTE
        # -------------------------

        st.subheader("📊 Forme récente")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {team_a}")

            st.metric(
                "Points moyens",
                f"{form_a['avg_points']:.2f}"
            )

            st.metric(
                "Buts marqués",
                f"{form_a['avg_goals_scored']:.2f}"
            )

            st.metric(
                "Buts encaissés",
                f"{form_a['avg_goals_conceded']:.2f}"
            )

        with col2:
            st.markdown(f"### {team_b}")

            st.metric(
                "Points moyens",
                f"{form_b['avg_points']:.2f}"
            )

            st.metric(
                "Buts marqués",
                f"{form_b['avg_goals_scored']:.2f}"
            )

            st.metric(
                "Buts encaissés",
                f"{form_b['avg_goals_conceded']:.2f}"
            )

        # -------------------------
        # PRÉDICTION
        # -------------------------

        winner, probability = predict_match(team_a, team_b)

        st.subheader("🔮 Prédiction")

        st.success(f"🏆 Vainqueur prédit : {winner}")

        st.metric(
            "Probabilité",
            f"{probability:.2%}"
        )