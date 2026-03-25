import requests
import time

# Give the server a second to start
time.sleep(2)

url = "http://127.0.0.1:5000/predict"
data = {
    "person_age": 25,
    "person_income": 50000,
    "person_home_ownership": "RENT",
    "person_emp_length": 5.0,
    "loan_intent": "EDUCATION",
    "loan_grade": "A",
    "loan_amnt": 10000,
    "loan_int_rate": 8.0,
    "loan_percent_income": 0.20,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 3
}

try:
    print(f"Sending POST to {url}")
    response = requests.post(url, json=data)
    response.raise_for_status()
    print("Success! Response from API:")
    print(response.json())
except Exception as e:
    print(f"Error testing API: {e}")
