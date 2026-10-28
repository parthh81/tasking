"python -m pip install pandas scikit-learn"

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_iris_svm_classification():
    # 1. LOAD THE BUILT-IN DATASET
    print("Loading the Iris dataset...")
    iris = load_iris()
    
    # Create a DataFrame for better visibility
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    flower_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

    print("\n--- DATA PREVIEW ---")
    print(X.head())
    print(f"Target classes: {flower_names}")
    print("-" * 50)

    # 2. SPLIT THE DATA
    # 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. FEATURE SCALING
    # SVMs are highly sensitive to the scale of the data. 
    # We MUST scale features so one measurement (like petal length) doesn't dominate another.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. INITIALIZE AND TRAIN THE SVM MODEL
    print("\nTraining the Support Vector Machine (SVM) model...")
    # Using a 'linear' kernel. Other popular options are 'rbf' or 'poly'
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # 5. MAKE PREDICTIONS ON THE TEST SET
    y_pred = svm_model.predict(X_test_scaled)

    # 6. EVALUATE THE MODEL
    print("\n" + "=" * 50)
    print("--- MODEL EVALUATION ---")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, columns=flower_names, index=flower_names))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=flower_names))

    # 7. PREDICT A NEW FLOWER
    print("-" * 50)
    print("--- PREDICTING A NEW FLOWER ---")
    # A new flower with measurements: Sepal Length, Sepal Width, Petal Length, Petal Width
    # We pass it as a DataFrame to avoid the "UserWarning" about missing feature names!
    new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
    
    # Scale the new flower
    new_flower_scaled = scaler.transform(new_flower)
    
    # Predict
    prediction_num = svm_model.predict(new_flower_scaled)[0]
    predicted_species = flower_names[prediction_num]
    
    print(f"Measurements: {new_flower.iloc[0].to_dict()}")
    print(f"Predicted Species: {predicted_species.upper()}")

if __name__ == "__main__":
    run_iris_svm_classification()