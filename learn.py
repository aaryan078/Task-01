import pandas as pd 
import numpy as np

dataframe= pd.read_csv('Salary_Data.csv')  
x = dataframe['YearsExperience'].values.reshape(30,1) 
y = dataframe['Salary']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y) 

import joblib
joblib.dump(model,'Salary_model.pkl')





