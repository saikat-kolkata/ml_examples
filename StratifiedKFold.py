import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Define the model
model = LogisticRegression(max_iter=200)

# Define StratifiedKFold
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Arrays to store the results
train_accuracies = []
val_accuracies = []

# Perform Stratified Cross-Validation
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate on the training set
    train_preds = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_preds)
    train_accuracies.append(train_accuracy)
    
    # Predict and evaluate on the validation set
    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    val_accuracies.append(val_accuracy)

    # Perform Stratified Cross-Validation using cross_val_score
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    # Print the results
    print(f"Cross-validation accuracies: {scores}")
    print(f"Mean cross-validation accuracy: {np.mean(scores)}")

# Print the results
print(f"Training accuracies: {train_accuracies}")
print(f"Validation accuracies: {val_accuracies}")
print(f"Mean training accuracy: {np.mean(train_accuracies)}")
print(f"Mean validation accuracy: {np.mean(val_accuracies)}")
