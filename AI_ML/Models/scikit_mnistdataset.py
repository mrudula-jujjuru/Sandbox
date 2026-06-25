#Digits data set

# Import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#problem definition: Classify handwritten digits (0-9) based on pixel values    

# Step 1Load the Digits dataset
digits = datasets.load_digits()
X = digits.data  # Features (pixel values)
y = digits.target  # Labels (digit values)

# Step 2: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3 : Scale the features
# Create the SVM model with an RBF kernel
model = SVC(kernel='rbf', gamma=0.5, C=1)

# Step 4:Train the model using the training data
model.fit(X_train, y_train)


#step 3: Create a pipeline that first scales the data then applies SVC
pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
# Define the parameter grid for GridSearchCV
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.001, 0.01, 0.1, 1]
}

#step 4: Perform Grid Search with Cross-Validation to find the best hyperparameters
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

# Step 5:Predict the labels for the test data
y_pred_old = model.predict(X_test)
y_pred = grid.predict(X_test)

# Step 6: Evaluate the model by checking its accuracy
acc1 = accuracy_score(y_test, y_pred_old)
accuracy = accuracy_score(y_test, y_pred)

# Print the result
print(f"accuracy without hyperparameter tuning: {acc1 * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
