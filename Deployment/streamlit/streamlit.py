import requests
import streamlit as st

# Streamlit app
def main():
    st.title("Churn Prediction")
    st.write("Enter customer data to predict churn:")

    # Input form
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["France", "Spain", "Germany"])
    age_category = st.selectbox("Age Category", ["Young", "Mature"])
    active_member = st.selectbox("Active Member", ["Yes", "No"])
    tenure = st.number_input("Tenure ", min_value=0, max_value=100, value=0)
    balance = st.number_input("Balance", min_value=0.0, max_value=10000000000000.0, value=0.0)
    credit_score = st.number_input("Credit Score", min_value=0.0, max_value=1000.0, value=0.0)
    submit_button = st.button("Predict Churn")

    if submit_button:
        # Prepare data payload
        data = {
            "gender_Male": 1 if gender == "Male" else 0,
            "gender_Female": 1 if gender == "Female" else 0,
            "country_France": 1 if country == "France" else 0,
            "country_Spain": 1 if country == "Spain" else 0,
            "country_Germany": 1 if country == "Germany" else 0,
            "Age_young": 1 if age_category == "Young" else 0,
            "Age_mature": 1 if age_category == "Mature" else 0,
            "active_member": 1 if active_member == "Yes" else 0,
            "tenure": tenure,
            "balance": balance,
            "credit_score": credit_score
        }

        # Make POST request to FastAPI server
        try:
            response = requests.post("http://localhost:8000/prediction", json=data)
            response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
            result = response.json()
            prediction = result["prediction is"]
            st.success(f"Churn Prediction: {prediction}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to make prediction. Error: {e}")

if __name__ == "__main__":
    main()
