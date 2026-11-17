import os

import joblib as jb
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Prédictions DNN", page_icon="🏦", layout="centered")

# ---------------------------------------------------------------
# Style : palette bleu nuit + safran, typographie Sora / Manrope
# ---------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600&family=Sora:wght@500;700;800&display=swap');

:root {
  --ink: #14213D;
  --ink-soft: #3B4A6B;
  --saffron: #F2A541;
  --mint: #1F9D7A;
  --rose: #D64550;
  --paper: #EEF2F7;
  --card: #FFFFFF;
  --line: #D9E0EB;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--paper);
  font-family: 'Manrope', sans-serif;
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 820px; }

h1, h2, h3, .hero h1 { font-family: 'Sora', sans-serif; letter-spacing: -0.02em; color: var(--ink); }

/* Bandeau d'en-tête */
.hero {
  background: linear-gradient(135deg, #14213D 0%, #23386B 70%);
  border-radius: 20px;
  padding: 2rem 2rem 1.8rem;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
}
.hero::after {
  content: "";
  position: absolute; right: -60px; top: -60px;
  width: 220px; height: 220px; border-radius: 50%;
  background: var(--saffron); opacity: .9;
}
.hero h1 { color: #fff !important; font-size: 2rem; margin: 0 0 .4rem; position: relative; z-index: 1; }
.hero p  { color: #C9D3EA; margin: 0; max-width: 480px; position: relative; z-index: 1; }

/* Onglets */
[data-baseweb="tab-list"] { gap: 6px; border-bottom: none; }
[data-baseweb="tab"] {
  background: #E1E8F2; border-radius: 12px 12px 0 0;
  padding: 10px 18px; font-weight: 600; color: var(--ink-soft);
}
[data-baseweb="tab"][aria-selected="true"] { background: var(--card); color: var(--ink); }
[data-baseweb="tab-highlight"] { background: var(--saffron) !important; height: 3px; }

/* Cartes de saisie */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card); border: 1px solid var(--line) !important;
  border-radius: 16px; box-shadow: 0 1px 2px rgba(20,33,61,.05);
}
.section-title {
  font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem;
  margin: 0 0 .2rem; color: var(--ink);
}
.section-sub { color: var(--ink-soft); font-size: .88rem; margin: 0 0 .8rem; }

/* Champs */
label, [data-testid="stWidgetLabel"] p { font-weight: 600; color: var(--ink-soft); font-size: .88rem; }
[data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 10px !important; }

/* Bouton principal */
.stButton > button {
  width: 100%; border: none; border-radius: 12px;
  background: var(--ink); color: #fff;
  font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1rem;
  padding: .75rem 1rem; transition: background .15s ease, transform .1s ease;
}
.stButton > button:hover { background: #23386B; color: #fff; }
.stButton > button:active { transform: scale(.99); }
.stButton > button:focus-visible { outline: 3px solid var(--saffron); outline-offset: 2px; }

/* Résultat */
.result {
  display: flex; align-items: center; gap: 1.6rem;
  background: var(--card); border: 1px solid var(--line);
  border-radius: 18px; padding: 1.4rem 1.6rem; margin-top: 1rem;
}
.gauge {
  width: 130px; height: 130px; border-radius: 50%; flex: none;
  display: grid; place-items: center;
}
.gauge-inner {
  width: 98px; height: 98px; border-radius: 50%; background: var(--card);
  display: grid; place-items: center;
  font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.5rem;
}
.verdict { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.3rem; margin: 0 0 .3rem; }
.detail  { color: var(--ink-soft); margin: 0; font-size: .95rem; }

.bar-row { display: flex; align-items: center; gap: .8rem; margin: .45rem 0; font-size: .92rem; }
.bar-name { width: 110px; font-weight: 600; color: var(--ink-soft); }
.bar-track { flex: 1; height: 10px; background: #E3E9F2; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-pct { width: 56px; text-align: right; font-weight: 600; color: var(--ink); }

.footer { text-align: center; color: var(--ink-soft); font-size: .8rem; margin-top: 1.5rem; }

@media (max-width: 600px) {
  .result { flex-direction: column; text-align: center; }
  .hero h1 { font-size: 1.6rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def section(title, subtitle=""):
    st.markdown(
        f'<p class="section-title">{title}</p><p class="section-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def gauge(pct, color):
    """Jauge circulaire, pct entre 0 et 1."""
    deg = int(pct * 360)
    return (
        f'<div class="gauge" style="background: conic-gradient({color} {deg}deg, #E3E9F2 0);">'
        f'<div class="gauge-inner" style="color:{color}">{pct:.0%}</div></div>'
    )


# ---------------------------------------------------------------
# Chemins des fichiers
# ---------------------------------------------------------------
BANK_MODEL_PATH = "DNN_model.h5"
BANK_ENCODERS_PATH = "encoders.joblib"
BANK_SCALER_PATH = "scaler.joblib"
BANK_UNIQUES_PATH = "uniques.joblib"

STUDENT_MODEL_PATH = "DNNRe.h5"
STUDENT_ENCODER_PATH = "encoders2.joblib"
STUDENT_SCALER_PATH = "scaler2.joblib"

CIFAR_MODEL_PATH = "cnn_model.h5"
CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR_CLASSES_FR = {
    "airplane": "Avion", "automobile": "Automobile", "bird": "Oiseau",
    "cat": "Chat", "deer": "Cerf", "dog": "Chien", "frog": "Grenouille",
    "horse": "Cheval", "ship": "Bateau", "truck": "Camion",
}


def check_files(paths):
    for path in paths:
        if not os.path.exists(path):
            st.error(f"Fichier introuvable : {path}. Vérifiez qu'il est bien dans le dépôt GitHub.")
            st.stop()


@st.cache_resource
def load_bank_resources():
    check_files([BANK_MODEL_PATH, BANK_ENCODERS_PATH, BANK_SCALER_PATH, BANK_UNIQUES_PATH])
    model = tf.keras.models.load_model(BANK_MODEL_PATH, compile=False)
    encoders = jb.load(BANK_ENCODERS_PATH)
    scaler = jb.load(BANK_SCALER_PATH)
    uniques = jb.load(BANK_UNIQUES_PATH)
    return model, encoders, scaler, uniques


@st.cache_resource
def load_student_resources():
    check_files([STUDENT_MODEL_PATH, STUDENT_ENCODER_PATH, STUDENT_SCALER_PATH])
    model = tf.keras.models.load_model(STUDENT_MODEL_PATH, compile=False)
    encoder = jb.load(STUDENT_ENCODER_PATH)
    scaler = jb.load(STUDENT_SCALER_PATH)
    return model, encoder, scaler


@st.cache_resource
def load_cifar_resources():
    check_files([CIFAR_MODEL_PATH])
    return tf.keras.models.load_model(CIFAR_MODEL_PATH, compile=False)


# ---------------------------------------------------------------
# Fonctions de prédiction (inchangées)
# ---------------------------------------------------------------
def predict_bank(model, encoders, scaler, values):
    (age, job, marital, education, default, housing, loan, contact,
     month, day_of_week, duration, campaign, pdays, previous, poutcome) = values

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

    x_new = np.array([
        age, job, marital, education, default, housing, loan, contact,
        month, day_of_week, duration, campaign, pdays, previous, poutcome,
    ]).reshape(1, -1)

    x_new = scaler.transform(x_new)
    return float(model.predict(x_new, verbose=0).ravel()[0])


def predict_student(model, encoder, scaler, values):
    hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers = values
    extracurricular = encoder.transform([extracurricular])[0]

    x_new = pd.DataFrame([[
        hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers
    ]], columns=[
        'Hours Studied',
        'Previous Scores',
        'Extracurricular Activities',
        'Sleep Hours',
        'Sample Question Papers Practiced',
    ])

    x_new = scaler.transform(x_new)
    return float(model.predict(x_new, verbose=0)[0][0])


def predict_cifar(model, image):
    """Retourne le vecteur des 10 probabilités."""
    image = image.convert("RGB").resize((32, 32))
    arr = np.array(image, dtype="float32") / 255.0
    arr = arr.reshape(1, 32, 32, 3)
    return model.predict(arr, verbose=0)[0]


# ---------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Prédictions par réseau de neurones</h1>
      <p>Estimez la souscription d'un client à une offre bancaire,
      l'indice de performance d'un étudiant, ou la classe d'une image.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_bank, tab_student, tab_cifar = st.tabs(
    ["🏦 Souscription bancaire", "🎓 Performance étudiante", "🖼️ Images CIFAR-10"]
)

# ======================= BANK MARKETING =======================
with tab_bank:
    bank_model, bank_encoders, bank_scaler, uniques = load_bank_resources()

    with st.container(border=True):
        section("Profil du client", "Qui est la personne contactée ?")
        a, b = st.columns(2)
        with a:
            age = st.number_input("Âge", min_value=18, max_value=100, value=40, step=1)
            marital = st.selectbox("Situation familiale", list(uniques[1]))
        with b:
            job = st.selectbox("Profession", list(uniques[0]))
            education = st.selectbox("Niveau d'études", list(uniques[2]))

    with st.container(border=True):
        section("Situation financière", "Crédits et défauts de paiement")
        a, b, c = st.columns(3)
        with a:
            default = st.selectbox("Défaut de paiement", list(uniques[3]))
        with b:
            housing = st.selectbox("Crédit immobilier", list(uniques[4]))
        with c:
            loan = st.selectbox("Crédit personnel", list(uniques[5]))

    with st.container(border=True):
        section("Contact", "Comment et quand le client a été joint")
        a, b, c, d = st.columns(4)
        with a:
            contact = st.selectbox("Canal", list(uniques[6]))
        with b:
            month = st.selectbox("Mois", list(uniques[7]))
        with c:
            day_of_week = st.selectbox("Jour", list(uniques[8]))
        with d:
            poutcome = st.selectbox("Campagne précédente", list(uniques[9]))

    with st.container(border=True):
        section("Historique de campagne", "Durées et nombre de contacts")
        a, b = st.columns(2)
        with a:
            duration = st.number_input("Durée du dernier appel (s)", min_value=0, value=100, step=1)
            pdays = st.number_input("Jours depuis le dernier contact", min_value=0, value=999, step=1)
        with b:
            campaign = st.number_input("Contacts cette campagne", min_value=1, value=1, step=1)
            previous = st.number_input("Contacts avant cette campagne", min_value=0, value=0, step=1)

    if st.button("Lancer la prédiction", key="btn_bank"):
        values = [age, job, marital, education, default, housing, loan, contact,
                  month, day_of_week, duration, campaign, pdays, previous, poutcome]

        with st.spinner("Analyse en cours..."):
            proba = predict_bank(bank_model, bank_encoders, bank_scaler, values)

        if proba >= 0.5:
            color, verdict = "#1F9D7A", "Le client va souscrire"
            detail = f"Probabilité de souscription : {proba:.1%}"
            shown = proba
        else:
            color, verdict = "#D64550", "Le client ne souscrira pas"
            detail = f"Probabilité de refus : {1 - proba:.1%} (souscription : {proba:.1%})"
            shown = 1 - proba

        st.markdown(
            f'<div class="result">{gauge(shown, color)}'
            f'<div><p class="verdict" style="color:{color}">{verdict}</p>'
            f'<p class="detail">{detail}</p></div></div>',
            unsafe_allow_html=True,
        )

# ===================== STUDENT PERFORMANCE =====================
with tab_student:
    stu_model, stu_encoder, stu_scaler = load_student_resources()

    with st.container(border=True):
        section("Habitudes de l'étudiant", "Travail, sommeil et activités")
        a, b = st.columns(2)
        with a:
            hours_studied = st.number_input("Heures d'étude", min_value=0, max_value=24, value=5, step=1)
            previous_scores = st.number_input("Notes précédentes", min_value=0, max_value=100, value=70, step=1)
            extracurricular = st.selectbox("Activités extrascolaires", stu_encoder.classes_.tolist())
        with b:
            sleep_hours = st.number_input("Heures de sommeil", min_value=0, max_value=24, value=7, step=1)
            sample_papers = st.number_input("Sujets d'examen pratiqués", min_value=0, value=5, step=1)

    if st.button("Prédire l'indice de performance", key="btn_student"):
        values = [hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]

        with st.spinner("Prédiction en cours..."):
            result = predict_student(stu_model, stu_encoder, stu_scaler, values)

        pct = min(max(result / 100, 0), 1)
        color = "#1F9D7A" if result >= 60 else "#F2A541" if result >= 40 else "#D64550"

        st.markdown(
            f'<div class="result">{gauge(pct, color)}'
            f'<div><p class="verdict" style="color:{color}">Indice prédit : {result:.1f} / 100</p>'
            f'<p class="detail">Estimation fournie par le modèle DNN.</p></div></div>',
            unsafe_allow_html=True,
        )

# ======================== CIFAR-10 (CNN) =======================
with tab_cifar:
    cifar_model = load_cifar_resources()

    with st.container(border=True):
        section("Votre image", "Formats acceptés : PNG, JPG, JPEG, WEBP")
        uploaded = st.file_uploader(
            "Choisir une image", type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed", key="cifar_upload",
        )
        if uploaded is not None:
            image = Image.open(uploaded)
            st.image(image, caption="Image téléversée", use_container_width=True)

    if uploaded is not None and st.button("Classer l'image", key="btn_cifar"):
        with st.spinner("Analyse en cours..."):
            probs = predict_cifar(cifar_model, image)

        top = np.argsort(probs)[::-1][:4]
        best = int(top[0])
        best_p = float(probs[best])
        color = "#1F9D7A" if best_p >= 0.6 else "#F2A541" if best_p >= 0.4 else "#D64550"

        st.markdown(
            f'<div class="result">{gauge(best_p, color)}'
            f'<div><p class="verdict" style="color:{color}">{CIFAR_CLASSES_FR[CIFAR_CLASSES[best]]}</p>'
            f'<p class="detail">Classe prédite : {CIFAR_CLASSES[best]} ({best_p:.1%})</p></div></div>',
            unsafe_allow_html=True,
        )

        rows = ""
        for i in top:
            p = float(probs[i])
            rows += (
                f'<div class="bar-row"><span class="bar-name">{CIFAR_CLASSES_FR[CIFAR_CLASSES[i]]}</span>'
                f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:{p*100:.1f}%; background:{color if i == best else "#23386B"}"></div></div>'
                f'<span class="bar-pct">{p:.1%}</span></div>'
            )
        st.markdown(
            f'<div class="result" style="display:block">'
            f'<p class="section-title">Top 4 des classes</p>{rows}</div>',
            unsafe_allow_html=True,
        )

st.markdown('<p class="footer">Modèles DNN et CNN · Bank Marketing, Student Performance et CIFAR-10</p>', unsafe_allow_html=True)