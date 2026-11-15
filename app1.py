import os

import joblib as jb
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------
# Configuration de la page (doit être la première commande Streamlit)
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered",
)

# ---------------------------------------------------------------
# Fichiers issus de l'entraînement (à placer dans le dépôt GitHub à côté de app.py)
# ---------------------------------------------------------------
MODEL_PATH = "DNN_model.h5"
ENCODERS_PATH = "encoders.joblib"
SCALER_PATH = "scaler.joblib"
UNIQUES_PATH = "uniques.joblib"


# ---------------------------------------------------------------
# Chargement des ressources
# @st.cache_resource : chargées une seule fois, puis réutilisées
# ---------------------------------------------------------------
@st.cache_resource
def load_resources():
    for path in [MODEL_PATH, ENCODERS_PATH, SCALER_PATH, UNIQUES_PATH]:
        if not os.path.exists(path):
            st.error(f"Fichier introuvable : {path}. Vérifiez qu'il est bien dans le dépôt GitHub.")
            st.stop()

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    encoders = jb.load(ENCODERS_PATH)
    scaler = jb.load(SCALER_PATH)
    uniques = jb.load(UNIQUES_PATH)
    return model, encoders, scaler, uniques


# ---------------------------------------------------------------
# Prétraitement + prédiction (identique à la version Gradio)
# ---------------------------------------------------------------
def predict(model, encoders, scaler, values):
    (age, job, marital, education, default, housing, loan, contact,
     month, day_of_week, duration, campaign, pdays, previous, poutcome) = values

    # Encodage des variables catégorielles
    job = encoders[0].transform([job])[0]
    marital = encoders[1].transform([marital])[0]
    education = encoders[2].transform([education])[0]
    default = encoders[3].transform([default])[0]
    housing = encoders[4].transform([housing])[0]
    loan = encoders[5].transform([loan])[0]
    contact = encoders[6].transform([contact])[0]
    month = encoders[7].transform([month])[0]
    day_of_week = encoders[8].transform([day_of_week])[0]
    poutcome = encoders[9].transform([poutcome])[0]

    # Vecteur des valeurs (même ordre qu'à l'entraînement)
    x_new = np.array([
        age, job, marital, education, default, housing, loan, contact,
        month, day_of_week, duration, campaign, pdays, previous, poutcome,
    ]).reshape(1, -1)

    # Normalisation
    x_new = scaler.transform(x_new)

    # Prédiction : probabilité de souscription
    proba = float(model.predict(x_new, verbose=0).ravel()[0])
    return proba


# ---------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------
st.title("🏦 Bank Marketing Prediction")
st.write("Renseignez les informations du client pour prédire s'il va souscrire à l'offre.")

model, encoders, scaler, uniques = load_resources()

# Variables numériques et catégorielles, réparties sur deux colonnes
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("age", min_value=18, max_value=100, value=40, step=1)
    job = st.selectbox("job", list(uniques[0]))
    marital = st.selectbox("marital", list(uniques[1]))
    education = st.selectbox("education", list(uniques[2]))
    default = st.selectbox("default", list(uniques[3]))
    housing = st.selectbox("housing", list(uniques[4]))
    loan = st.selectbox("loan", list(uniques[5]))
    contact = st.selectbox("contact", list(uniques[6]))

with col2:
    month = st.selectbox("month", list(uniques[7]))
    day_of_week = st.selectbox("day_of_week", list(uniques[8]))
    duration = st.number_input("duration", min_value=0, value=100, step=1)
    campaign = st.number_input("campaign", min_value=1, value=1, step=1)
    pdays = st.number_input("pdays", min_value=0, value=999, step=1)
    previous = st.number_input("previous", min_value=0, value=0, step=1)
    poutcome = st.selectbox("poutcome", list(uniques[9]))

if st.button("Prédire", type="primary"):
    values = [age, job, marital, education, default, housing, loan, contact,
              month, day_of_week, duration, campaign, pdays, previous, poutcome]

    with st.spinner("Analyse en cours..."):
        proba = predict(model, encoders, scaler, values)

    st.subheader("Souscription")
    if proba >= 0.5:
        st.success(f"**Souscrire** ({proba:.1%})")
    else:
        st.error(f"**Ne souscrira pas** ({1 - proba:.1%})")

    st.write(f"Souscrire : {proba:.1%}")
    st.progress(proba)
    st.write(f"Ne souscrira pas : {1 - proba:.1%}")
    st.progress(1 - proba)

st.caption("Modèle DNN · Bank Marketing")
