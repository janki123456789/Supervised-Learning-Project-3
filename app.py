import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression

st.title("Risk Alert Classifier")

# Dataset Load
df = pd.read_csv("Risk_Alert_Classifier_Dataset.csv")

# Encoding
le_gender = LabelEncoder()
le_region = LabelEncoder()
le_emp = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["region"] = le_region.fit_transform(df["region"].astype(str))
df["employment_type"] = le_emp.fit_transform(df["employment_type"].astype(str))

# Features and Target
X = df.drop(['risk_status', 'last_transaction_date'], axis=1)
y = df['risk_status']

# Missing Value Handling
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# Model Training
model = LogisticRegression(max_iter=5000)
model.fit(X_imputed, y)

st.header("Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
region = st.selectbox("Region", df["region"].unique())
employment_type = st.selectbox(
    "Employment Type",
    df["employment_type"].unique()
)

# Example for remaining columns
income = st.number_input("Income")
transaction_count = st.number_input("Transaction Count")

if st.button("Predict Risk"):

    gender_value = le_gender.transform([gender])[0]

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender_value],
        "region": [region],
        "employment_type": [employment_type],
        "income": [income],
        "transaction_count": [transaction_count]
    })

    input_data = pd.DataFrame(
        imputer.transform(input_data),
        columns=input_data.columns
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk Customer")
    else:
        st.success("Low Risk Customer")