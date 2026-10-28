"python -m pip install pandas numpy"

import pandas as pd
import numpy as np

def run_preprocessing_analysis():
    # ---------------------------------------------------------
    # 1. GENERATE MESSY DATASET
    # ---------------------------------------------------------
    data = {
        'Employee_ID': [101, 102, 103, 104, 104, 105, 106, 107], # Contains a duplicate (104)
        'Age': [25, np.nan, 28, 35, 35, 40, 22, 55],             # Contains missing value
        'Salary': [50000, 60000, np.nan, 80000, 80000, 120000, 45000, 95000], # Contains missing value
        'Department': ['HR', 'IT', 'IT', 'Finance', 'Finance', np.nan, 'HR', 'IT'], # Missing categorical
        'Performance_Score': ['Low', 'Medium', 'High', 'Medium', 'Medium', 'High', 'Low', 'High'],
        'Join_Date': ['2020-01-15', '2021-03-22', '2019-11-05', '2018-07-12', 
                      '2018-07-12', '2015-05-30', '2023-01-10', '2017-08-21']
    }
    
    df = pd.DataFrame(data)
    
    print("--- ORIGINAL DATASET ---")
    print(df)
    print("\nDataset Info:")
    print(df.info())
    print("-" * 50)

    # ---------------------------------------------------------
    # 2. HANDLING DUPLICATES
    # ---------------------------------------------------------
    print("\n--- STEP 1: REMOVING DUPLICATES ---")
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicate row(s).")
    
    # ---------------------------------------------------------
    # 3. HANDLING MISSING VALUES (IMPUTATION)
    # ---------------------------------------------------------
    print("\n--- STEP 2: HANDLING MISSING VALUES ---")
    print("Missing values before:\n", df.isnull().sum())
    
    # Fill numerical missing values with the median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Salary'] = df['Salary'].fillna(df['Salary'].median())
    
    # Fill categorical missing values with the mode (most frequent)
    df['Department'] = df['Department'].fillna(df['Department'].mode()[0])
    
    print("\nMissing values after imputation:\n", df.isnull().sum())

    # ---------------------------------------------------------
    # 4. DATA TYPE CONVERSION
    # ---------------------------------------------------------
    print("\n--- STEP 3: DATA TYPE CONVERSION ---")
    # Convert string dates to actual datetime objects
    df['Join_Date'] = pd.to_datetime(df['Join_Date'])
    
    # Extract a new feature from the date (e.g., Year)
    df['Join_Year'] = df['Join_Date'].dt.year
    print("Converted 'Join_Date' to datetime and extracted 'Join_Year'.")

    # ---------------------------------------------------------
    # 5. CATEGORICAL ENCODING
    # ---------------------------------------------------------
    print("\n--- STEP 4: ENCODING CATEGORICAL VARIABLES ---")
    
    # A. Ordinal Encoding for 'Performance_Score' (Order matters)
    score_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Performance_Encoded'] = df['Performance_Score'].map(score_mapping)
    
    # B. One-Hot Encoding for 'Department' (Nominal data - no intrinsic order)
    df = pd.get_dummies(df, columns=['Department'], prefix='Dept', dtype=int)
    
    print("Applied Ordinal Mapping to Performance and One-Hot Encoding to Department.")

    # ---------------------------------------------------------
    # 6. FEATURE SCALING (NORMALIZATION & STANDARDIZATION)
    # ---------------------------------------------------------
    print("\n--- STEP 5: FEATURE SCALING ---")
    
    # Min-Max Normalization (Scaling values between 0 and 1)
    df['Salary_Normalized'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
    
    # Z-Score Standardization (Mean = 0, Standard Deviation = 1)
    df['Age_Standardized'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
    
    print("Normalized 'Salary' (0 to 1) and Standardized 'Age' (Z-score).")

    # ---------------------------------------------------------
    # 7. FINAL ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("--- FINAL PROCESSED DATASET ---")
    # Dropping original columns that we've encoded/transformed for a cleaner view
    final_columns_to_show = ['Employee_ID', 'Age_Standardized', 'Salary_Normalized', 
                             'Join_Year', 'Performance_Encoded', 'Dept_Finance', 'Dept_HR', 'Dept_IT']
    
    df_final = df[final_columns_to_show]
    print(df_final.round(3))  # Rounding for readability
    print("\nFinal Descriptive Statistics:")
    print(df_final[['Age_Standardized', 'Salary_Normalized']].describe().round(2))

if __name__ == "__main__":
    run_preprocessing_analysis()