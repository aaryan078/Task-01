import joblib
model=joblib.load('Salary_model.pkl')



num=float(input("years of experience:"))
predict=model.predict([[num]])
print(predict)

