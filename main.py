import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read Dataset
df = pd.read_csv("data/employee_salary.csv")

# Encode Education Column
encoder = LabelEncoder()

df["Education"] = encoder.fit_transform(df["Education"])

# Features
X = df[
    [
        "Experience",
        "Age",
        "Education",
        "SkillScore"
    ]
]

# Target
y = df["Salary"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

print("Actual Salary:")
print(y_test)

print("\nPredicted Salary:")
print(prediction)

print("\nR2 Score:")
print(r2_score(y_test, prediction))