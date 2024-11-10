
#Data were downloaded from https://ourworldindata.org/coronavirus/country/france.

import pandas as pd
veriseti=pd.read_excel("veri_seti.xlsx")

X=veriseti.iloc[:,5].values
y=veriseti.iloc[:,6].values
X=X.reshape(len(X),1)

#Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=6)

#Fitting the training set according to linear regression
from sklearn.linear_model import LinearRegression
model_Regresyon=LinearRegression()
model_Regresyon.fit(X_train, y_train)

#Estimation of test set results
y_pred=model_Regresyon.predict(X_test)

# Plotting the training set results
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model_Regresyon.predict(X_train), color='blue')
plt.title("Average Number of Covid-19 Cases and Deaths per Week in France (Education Data Set)")
plt.xlabel('Number of Cases per Week')
plt.ylabel('Weekly Number of Deaths')
plt.show()

#Test seti sonuçlarının grafiğinin çizdirilmesi
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, model_Regresyon.predict(X_train), color='blue')
plt.title("Average Number of Covid-19 Cases and Deaths per Week in France (Test Data Set)")
plt.xlabel('Number of Cases per Week')
plt.ylabel('Weekly Number of Deaths')
plt.show()

#Regression Equation
print("y=%0.2f"%model_Regresyon.coef_+"x+%0.2f"%model_Regresyon.intercept_)

#Performance of Test Data Set
from sklearn.metrics import median_absolute_error, r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
print("R-Kare: ", r2_score(y_test, y_pred))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("MedAE: ", median_absolute_error(y_test, y_pred))
print("EVS: ", explained_variance_score(y_test, y_pred))













