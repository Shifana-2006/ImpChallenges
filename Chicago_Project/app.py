import streamlit as st
import joblib
import numpy as np

# ---------------- LOAD MODEL ----------------
data = joblib.load("C:/Users/user/Desktop/Chicago_Project/chicago_system.pkl")

model_clf = data["classification_model"]
model_reg = data["regression_model"]
threshold = data["threshold"]
features = data["features"]

# ---------------- UI ----------------
st.set_page_config(page_title="Risk Prediction System", layout="centered")

st.title("🚦 Chicago Risk Prediction System")
st.write("Enter details to predict risk level and severity score")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
posted_speed_limit = st.number_input("Posted Speed Limit")
traffic_control_device = st.number_input("Traffic Control Device")
device_condition = st.number_input("Device Condition")
weather_condition = st.number_input("Weather Condition")
lighting_condition = st.number_input("Lighting Condition")
first_crash_type = st.number_input("First Crash Type")
trafficway_type = st.number_input("Trafficway Type")
alignment = st.number_input("Alignment")
roadway_surface_cond = st.number_input("Road Surface Condition")
road_defect = st.number_input("Road Defect")
crash_type = st.number_input("Crash Type")
prim_cause = st.number_input("Primary Contributory Cause")
sec_cause = st.number_input("Secondary Contributory Cause")
num_units = st.number_input("Number of Units")
crash_hour = st.number_input("Crash Hour")
crash_day = st.number_input("Crash Day of Week")
crash_month = st.number_input("Crash Month")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")
is_night = st.selectbox("Is Night?", [0, 1])
is_weekend = st.selectbox("Is Weekend?", [0, 1])

# ---------------- PREDICTION ----------------
if st.button("Predict Risk 🚀"):

    input_data = np.array([[
        posted_speed_limit,
        traffic_control_device,
        device_condition,
        weather_condition,
        lighting_condition,
        first_crash_type,
        trafficway_type,
        alignment,
        roadway_surface_cond,
        road_defect,
        crash_type,
        prim_cause,
        sec_cause,
        num_units,
        crash_hour,
        crash_day,
        crash_month,
        latitude,
        longitude,
        is_night,
        is_weekend
    ]])

    # Classification
    prob = model_clf.predict_proba(input_data)[0][1]
    pred = 1 if prob >= threshold else 0

    st.markdown("---")
    st.subheader("🚨 Risk Prediction Result")

    if pred == 1:
        st.error("RISK - Emergency Response Needed")
    else:
        st.success("SAFE - Normal Handling")

    st.write("Risk Probability:", round(prob, 4))

    # Regression output
    severity = model_reg.predict(input_data)[0]
    st.subheader("📊 Severity Score")
    st.info(round(severity, 2))