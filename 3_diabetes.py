"python -m pip install pandas scikit-learn"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run_diabetes_prediction():
    # 1. LOAD THE DATA
    # Using the standard Pima Indians Diabetes Dataset hosted publicly
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    print("Downloading dataset...")
    
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully!\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your internet connection.")
        return

    print("--- DATA PREVIEW ---")
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("-" * 50)

    # 2. SEPARATE FEATURES (X) AND TARGET (y)
    # 'Outcome' is 1 for Diabetes, 0 for No Diabetes
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 3. SPLIT INTO TRAINING AND TESTING SETS
    # 80% of data used to train the model, 20% kept hidden to test it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. FEATURE SCALING
    # Logistic regression relies on math that performs better when features are on the same scale.
    # (Notice we fit ONLY on the training data, then transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. INITIALIZE AND TRAIN THE MODEL
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # 6. MAKE PREDICTIONS ON THE UNSEEN TEST DATA
    y_pred = model.predict(X_test_scaled)

    # 7. EVALUATE THE MODEL
    print("\n" + "=" * 50)
    print("--- MODEL EVALUATION ---")
    
    # Overall Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    # Confusion Matrix (shows true positives, false positives, etc.)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, 
                       columns=['Predicted: No Diabetes', 'Predicted: Diabetes'], 
                       index=['Actual: No Diabetes', 'Actual: Diabetes']))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

    # 8. MAKE A PREDICTION FOR A NEW PATIENT
    print("-" * 50)
    print("--- PREDICTING FOR A NEW PATIENT ---")
    # Patient profile: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age
    new_patient = [[2, 140, 72, 35, 0, 33.6, 0.627, 45]]
    
    # Must scale the new patient's data using the SAME scaler
    new_patient_scaled = scaler.transform(new_patient)
    prediction = model.predict(new_patient_scaled)
    probability = model.predict_proba(new_patient_scaled)[0][1]

    result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    print(f"Prediction for new patient: {result}")
    print(f"Probability of having diabetes: {probability * 100:.1f}%")

if __name__ == "__main__":
    run_diabetes_prediction()