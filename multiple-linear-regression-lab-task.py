import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml

my_new_data = pd.read_csv("Car-details-v3.csv")
print(my_new_data.head())

print("-------------my_new_data---------------")
print("----------------------------")

target_price = my_new_data["engine"]
print(target_price)
#year = my_new_data['year']
engine = my_new_data['engine']
mileage = my_new_data['mileage']
ses = my_new_data['ses']

print("-------------target_price---------------")
print("----------------------------")


#histogram plotting
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(my_new_data['engine'], bins=10)
plt.show()



#correlation plotting
correlation_matrix = my_new_data.corr().round(2)
print(correlation_matrix)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

X = pd.DataFrame(np.c_[my_new_data['mileage'], my_new_data['ses']], columns = ['mileage','ses'])
y = my_new_data['enginer']
print(X)
print(X.dtypes)
print(y.dtypes)

#splitting test and train
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

from sklearn.metrics import r2_score

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
testPred = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (mean_squared_error(Y_test, y_test_predict))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#plot to see
plt.scatter(X_train['mileage'], Y_train,color='g') 
plt.scatter(X_train['ses'], Y_train,color='b') 
plt.plot(X_train['mileage'], y_train_predict,color='k') 
plt.show()

plt.scatter(X_test['mileage'],Y_test)
plt.scatter(X_test['ses'],y_test_predict)
plt.show()

plt.scatter(mileage, target_price)
plt.scatter(engine, target_price)
plt.show()