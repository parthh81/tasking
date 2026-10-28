"python -m pip install pandas scikit-learn"


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def run_scaling_demo():
    # 1. CREATE SAMPLE DATA
    # Notice the massive difference in scale between Age (20s-50s) and Salary (thousands)
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 32, 47, 51, 22],
        'Salary': [50000, 75000, 120000, 150000, 42000]
    }
    df = pd.DataFrame(data)
    
    print("--- ORIGINAL DATA ---")
    print(df)
    print("\nOriginal Statistics:")
    print(df[['Age', 'Salary']].describe().round(2))
    print("-" * 50)

    # 2. INITIALIZE SCALERS
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # We only scale the numerical features, not the text (Name)
    features_to_scale = ['Age', 'Salary']

    # 3. APPLY NORMALIZATION (Min-Max Scaling)
    # Transforms features by scaling each feature to a given range, usually (0, 1)
    normalized_data = min_max_scaler.fit_transform(df[features_to_scale])
    
    # Add back to DataFrame with new column names for comparison
    df[['Age_Norm', 'Salary_Norm']] = normalized_data

    # 4. APPLY STANDARDIZATION (Z-Score Scaling)
    # Transforms features so they have a mean of 0 and a standard deviation of 1
    standardized_data = standard_scaler.fit_transform(df[features_to_scale])
    
    # Add back to DataFrame with new column names
    df[['Age_Std', 'Salary_Std']] = standardized_data

    # 5. VIEW RESULTS
    print("\n--- FINAL DATASET WITH SCALED FEATURES ---")
    # Reordering columns for easier side-by-side comparison
    final_df = df[['Name', 'Age', 'Age_Norm', 'Age_Std', 'Salary', 'Salary_Norm', 'Salary_Std']]
    print(final_df.round(3))
    
    print("\n--- PROOF OF SCALING ---")
    print("\nNormalization Proof (Min is 0, Max is 1):")
    print(df[['Age_Norm', 'Salary_Norm']].describe().round(2).loc[['min', 'max']])
    
    print("\nStandardization Proof (Mean is ~0, Std Dev is ~1):")
    # Note: scikit-learn's StandardScaler uses population standard deviation (ddof=0), 
    # while pandas describe() uses sample standard deviation (ddof=1). 
    # The mean will be exactly 0, and the std will be very close to 1.
    print(df[['Age_Std', 'Salary_Std']].describe().round(2).loc[['mean', 'std']])

if __name__ == "__main__":
    run_scaling_demo()