#Iris dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Problem definintion: Classify iris species based on features (sepal length, sepal width, petal length, petal width)

# Step 1: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train the model
model = SVC()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#---------------------------------------------------------
#Test the model with a sample input
#---------------------------------------------------------
# Define a sample input (reshape to 2D since model expects batches)
sample_input1 = np.array([[5.1, 1.5, 2.5, 0.2]])
sample_input2 = np.array([[5.1, 2.0, 1.4, 0.2]])

# Apply the same scaler used for training
sample_input_scaled1 = scaler.transform(sample_input1)
sample_input_scaled2 = scaler.transform(sample_input2)
# Predict the class
predicted_class = model.predict(sample_input_scaled1)[0]
print("Predicted class_sample1:", iris.target_names[predicted_class])
predicted_class = model.predict(sample_input_scaled2)[0]
print("Predicted class_sample2:", iris.target_names[predicted_class])