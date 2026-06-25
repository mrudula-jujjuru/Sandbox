#Digits data set Prediction

# Import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2 
import matplotlib.pyplot as plt


# Load the Digits dataset
digits = datasets.load_digits()
X = digits.data  # Features (pixel values)
y = digits.target  # Labels (digit values)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))

param_grid = {
    'svc__C': [0.1, 0.4, 0.8,1],
    'svc__gamma': [0.001, 0.01, 0.1, 0.5]
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)


# Predict the labels for the test data
y_pred = grid.predict(X_test)

# Evaluate the model by checking its accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the result
print(f"Accuracy: {accuracy * 100:.2f}%")


image_path = 'data/6.jpg'  # image of handwritten digit '6' in data folder
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

# Apply CLAHE for local contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
image = clahe.apply(image)


# Make square
h, w = image.shape
size = max(h, w)
square = np.full((size, size), 255, dtype=np.uint8)
square[(size - h)//2:(size - h)//2 + h, (size - w)//2:(size - w)//2 + w] = image

# Resize to 8x8
resized = cv2.resize(square, (8, 8), interpolation=cv2.INTER_AREA)

# Invert (black digit on white)
inverted = 255 - resized

# Normalize to 0–16 like sklearn
normalized = (inverted / 255.0) * 16
normalized = normalized.astype(np.float64)

print("Normalized image shape:", normalized.shape)
print("Min/Max pixel values:", normalized.min(), normalized.max())

# Flatten
flattened = normalized.flatten().reshape(1, -1)

print("Flattened image shape:", flattened.shape)

# ------------------ Step 4: Predict ------------------
prediction = grid.best_estimator_.predict(flattened)

print("Prediction complete")
print(f"Predicted digit: {prediction[0]}")


