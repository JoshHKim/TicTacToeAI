import numpy as np
import os
from os.path import dirname, join
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

current_dir = dirname(__file__)
file_path = join(current_dir, "datasets/tictac_single.txt")

A = np.loadtxt(file_path)
np.random.shuffle(A)
X = A[:, :9]
y = A[:, 9:]
y=y.ravel()

svm = SVC(kernel='linear', C=1.0, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
mlp = MLPClassifier(random_state=42, max_iter=1000)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc_svm = []
cm_svm = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

acc_knn = []
cm_knn = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

acc_mlp = []
cm_mlp = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the models
    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    # Make predictions
    y_pred_svm = svm.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_mlp = mlp.predict(X_test)
    
    # Compute the confusion matrix for this fold and add it to the aggregate confusion matrix
    cm_svm += confusion_matrix(y_test, y_pred_svm)
    cm_knn += confusion_matrix(y_test, y_pred_knn)
    cm_mlp += confusion_matrix(y_test, y_pred_mlp)

    # Calculate and store the accuracy for this fold
    acc_svm.append(accuracy_score(y_test, y_pred_svm))
    acc_knn.append(accuracy_score(y_test, y_pred_knn))
    acc_mlp.append(accuracy_score(y_test, y_pred_mlp))

print("\nFor Support Vector Machine \n --------------------------------")
for i, accuracy in enumerate(acc_svm, 1):
    print(f"Accuracy for Fold {i}: {accuracy*100:.2f}%")
overall_accuracy = np.trace(cm_svm) / np.sum(cm_svm)
print(f"\nOverall Accuracy: {overall_accuracy*100:.2f}%")
print(f"\nAggregate Confusion Matrix:\n{cm_svm}")


print("\nFor K-Nearest Neighbors \n --------------------------------")
for i, accuracy in enumerate(acc_knn, 1):
    print(f"Accuracy for Fold {i}: {accuracy*100:.2f}%")
overall_accuracy = np.trace(cm_knn) / np.sum(cm_knn)
print(f"\nOverall Accuracy: {overall_accuracy*100:.2f}%")
print(f"\nAggregate Confusion Matrix:\n{cm_knn}")


print("\nFor Multi-Layer Perceptron \n --------------------------------")
for i, accuracy in enumerate(acc_mlp, 1):
    print(f"Accuracy for Fold {i}: {accuracy*100:.2f}%")
overall_accuracy = np.trace(cm_mlp) / np.sum(cm_mlp)
print(f"\nOverall Accuracy: {overall_accuracy*100:.2f}%")
print(f"\nAggregate Confusion Matrix:\n{cm_mlp}")
