import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Read Dataset
df = pd.read_csv("data/employee_salary.csv")

# Encode Education
encoder = LabelEncoder()
df["Education"] = encoder.fit_transform(df["Education"])

# Features and Target
X = df[
    [
        "Experience",
        "Age",
        "Education",
        "SkillScore"
    ]
]

y = df["Salary"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("Employee Salary Prediction System")

experience = st.number_input(
    "Experience",
    min_value=0
)

age = st.number_input(
    "Age",
    min_value=18
)

education = st.selectbox(
    "Education",
    [
        "BCA",
        "BSc",
        "BTech",
        "MCA",
        "MTech"
    ]
)

skill_score = st.slider(
    "Skill Score",
    0,
    100
)

if st.button("Predict Salary"):

    edu_value = encoder.transform(
        [education]
    )[0]

    prediction = model.predict(
        [[
            experience,
            age,
            edu_value,
            skill_score
        ]]
    )

    st.success(
        f"Predicted Salary: ₹{prediction[0]:,.0f}"
    )