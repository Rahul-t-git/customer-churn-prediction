import joblib

model = joblib.load("customer_churn_model.pkl")

print("Model loaded successfully!")
print(type(model))