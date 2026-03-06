import joblib

model = joblib.load("ml/titanic_model.pkl")

sample = [[2, 30]]  # pclass=2, age=30
prediction = model.predict(sample)
probability = model.predict_proba(sample)

print("Prediction:", prediction)
print("Probability:", probability)