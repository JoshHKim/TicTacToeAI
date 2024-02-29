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

kf = KFold(n_splits=10, shuffle=True, random_state=42)

print("\nFor KNN Regression \n --------------------------------")
knn_scores = cross_val_score(knn, X, y, cv=5)
print("\nAccuracy across folds: ")
print(knn_scores)
print(f"\nAverage Accuracy: {np.mean(knn_scores):.4f}")

print("\nFor Linear Regression \n --------------------------------")
lin_scores = cross_val_score(lin, X, y, cv=5)
print("\nAccuracy across folds: ")
print(lin_scores)
print(f"\nAverage Accuracy: {np.mean(lin_scores):.4f}")

print("\nFor Multi Layer Perceptron Regression \n --------------------------------")
mlp_scores = cross_val_score(mlp, X, y, cv=5)
print("\nAccuracy across folds: ")
print(mlp_scores)
print(f"\nAverage Accuracy: {np.mean(mlp_scores):.4f}")


#-----------------------------------------------------------------------#
#                 Testing with 10% of the data                          #
#-----------------------------------------------------------------------#

np.random.shuffle(A)
X = A[:len(A)//10, :9]
y = A[:len(A)//10, 9:]
print("\nFor KNN Regression \n --------------------------------")
knn_scores = cross_val_score(knn, X, y, cv=5)
print("\nAccuracy across folds: ")
print(knn_scores)
print(f"\nAverage Accuracy: {np.mean(knn_scores):.4f}")

print("\nFor Linear Regression \n --------------------------------")
lin_scores = cross_val_score(lin, X, y, cv=5)
print("\nAccuracy across folds: ")
print(lin_scores)
print(f"\nAverage Accuracy: {np.mean(lin_scores):.4f}")

print("\nFor Multi Layer Perceptron Regression \n --------------------------------")
mlp_scores = cross_val_score(mlp, X, y, cv=5)
print("\nAccuracy across folds: ")
print(mlp_scores)
print(f"\nAverage Accuracy: {np.mean(mlp_scores):.4f}") 