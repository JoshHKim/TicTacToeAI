import numpy as np
import os
from os.path import dirname, join
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

current_dir = dirname(__file__)
file_path = join(current_dir, "datasets/tictac_multi.txt")

A = np.loadtxt(file_path)
np.random.shuffle(A)
X = A[:, :9]
y = A[:, 9:]

knn = KNeighborsRegressor(n_neighbors=5)
mlp = MLPRegressor(random_state=42, max_iter=1000)
lin = MultiOutputRegressor(LinearRegression())

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nFor final board optimal moves regression")
print("----------------------------------------")

print("\nFor KNN Regression \n--------------------------------")
knn_scores = cross_val_score(knn, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-knn_scores)
print("\nRoot mean squared error across folds: ")
print(rmse_scores)
print(f"\nAverage root mean squared error: {np.mean(rmse_scores):.4f}")

print("\nFor Linear Regression \n--------------------------------")
lin_scores = cross_val_score(lin, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-lin_scores)
print("\nRoot mean squared error across folds: ")
print(rmse_scores)
print(f"\nAverage root mean squared error: {np.mean(rmse_scores):.4f}")

print("\nFor Multi Layer Perceptron Regression \n--------------------------------")
mlp_scores = cross_val_score(mlp, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mlp_scores)
print("\nRoot mean squared error across folds: ")
print(rmse_scores)
print(f"\nAverage root mean squared error: {np.mean(rmse_scores):.4f}")


#-----------------------------------------------------------------------#
#                 Testing with 10% of the data                          #
#-----------------------------------------------------------------------#

print("\nFor final board optimal moves regression with 10% data")
print("-----------------------------------------------------")

np.random.shuffle(A)
X = A[:len(A)//10, :9]
y = A[:len(A)//10, 9:]
print("\nFor KNN Regression \n--------------------------------")
knn_scores = cross_val_score(knn, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-knn_scores)
print("\nRoot mean squared error across folds: ")
print(rmse_scores)
print(f"\nAverage root mean squared error: {np.mean(rmse_scores):.4f}")

print("\nFor Linear Regression \n--------------------------------")
lin_scores = cross_val_score(lin, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-lin_scores)
print("\nRoot mean squared error across folds: ")
print(rmse_scores)
print(f"\nAverage root mean squared error: {np.mean(rmse_scores):.4f}")

print("\nFor Multi Layer Perceptron Regression \n--------------------------------")
mlp_scores = cross_val_score(mlp, X, y, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mlp_scores)
print("\nRoot mean squared error across folds: ")
print(rmse_scores)
print(f"\nAverage root mean squared error: {np.mean(rmse_scores):.4f}")

#-----------------------------------------------------------------------#
#         Testing with Normal Equations for Linear Regression           #
#-----------------------------------------------------------------------#

print("\nFor Linear Regression with Normal Equations\n--------------------------------")

np.random.shuffle(A)
X = A[:, :9]
y = A[:, 9:]

def fit(X, y):
    one = np.ones(X.shape[0])
    X = np.c_[one, X]
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y) 
    return theta

def predict(X, theta):
    x = np.array(X)
    one = np.ones(x.shape[0])
    x = np.c_[one,x]
    res = np.dot(x, theta)
    return np.ravel(res)

def rmse(y, y_pred):
    diff = y_pred - y
    squared_diff = diff ** 2
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    return rmse


def crossVal(X, y):
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        thetas = []
        rmses = []
        for i in range(9):
            y_single = y_train[:, i:i+1]
            thetas.append(fit(X_train, y_single))

            y_pred = []
            for j in range(len(X_test)):
                y_pred.append(predict(X_test[j:j+1, :], thetas[i]))
            rmses.append(rmse(y_test[:, i:i+1], y_pred))

    return np.asarray(rmses)

np.set_printoptions(precision=3)
lnr_scores = crossVal(X,y)
print("\nRoot mean squared error across folds: ")
print(lnr_scores)
print(f"\nAverage root mean squared error: {np.mean(lnr_scores):.4f}")