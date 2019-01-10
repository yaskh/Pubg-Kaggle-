import pandas as pd 

dataset = pd.read_csv("sample.csv")


X = dataset.iloc[:,4:26].values

y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)


from sklearn import tree
dt = tree.DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)