"""Streamlit frontend. All predictions are requested from the API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Penguin Species Predictor", page_icon="🐧")
st.title("🐧 Penguin Species Predictor")
st.caption("Enter body measurements and classify the Palmer penguin species.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])
        bill_length = st.number_input("Bill length (mm)", 25.0, 70.0, 47.5, 0.1)
        flipper_length = st.number_input("Flipper length (mm)", 150.0, 250.0, 220.0, 1.0)
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        bill_depth = st.number_input("Bill depth (mm)", 10.0, 30.0, 15.2, 0.1)
        body_mass = st.number_input("Body mass (g)", 2_000.0, 7_000.0, 5_000.0, 25.0)
    submitted = st.form_submit_button("Predict species", type="primary", use_container_width=True)

if submitted:
    payload = {
        "island": island,
        "sex": sex,
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper_length,
        "body_mass_g": body_mass,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted species: **{result['prediction']}**")
        st.subheader("Class probabilities")
        st.bar_chart(result["probabilities"])
    except requests.RequestException as exc:
        st.error(f"The prediction API is unavailable: {exc}")
