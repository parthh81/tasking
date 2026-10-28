"python -m pip install pandas scikit-learn"

import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def run_kmeans_clustering():
    # 1. GENERATE SYNTHETIC UNLABELED DATA
    # We will create 300 data points secretly grouped into 4 clusters
    print("Generating synthetic customer data...")
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    
    # Put it in a Pandas DataFrame (Imagine these are customer spending habits)
    df = pd.DataFrame(X, columns=['Annual_Spend', 'Store_Visits'])
    
    print("\n--- DATA PREVIEW (Unlabeled) ---")
    print(df.head())
    print("-" * 50)

    # 2. FEATURE SCALING
    # K-Means groups data by measuring the physical distance between points.
    # If we don't scale, a feature with large numbers (Annual Spend) will overpower smaller ones (Visits).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # 3. INITIALIZE AND RUN K-MEANS
    print("\nApplying K-Means Clustering (Instructing it to find 4 groups)...")
    # n_init='auto' suppresses a common scikit-learn background warning
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    
    # Notice we ONLY pass X_scaled. There is no 'y' target variable! 
    # The algorithm must figure out the groups blindly.
    kmeans.fit(X_scaled)

    # 4. EXTRACT RESULTS
    # Get the cluster number (0, 1, 2, or 3) assigned to each data point
    df['Assigned_Cluster'] = kmeans.labels_
    
    print("\n--- CLUSTERING RESULTS ---")
    print(df.head(10).round(2))
    
    print("\nCalculated Cluster Centers (Centroids):")
    # We invert the scaling so the centers make sense in our original numbers
    original_scale_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(original_scale_centers, columns=['Annual_Spend', 'Store_Visits'])
    centers_df.index.name = 'Cluster'
    print(centers_df.round(2))

    # 5. EVALUATE THE CLUSTERING
    print("\n" + "=" * 50)
    print("--- MODEL EVALUATION ---")
    
    # Silhouette Score ranges from -1 to 1. 
    # Closer to 1 means clusters are dense and perfectly separated from each other.
    score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"Silhouette Score: {score:.3f}")
    
    if score > 0.7:
        print("Conclusion: Excellent clustering! The groups are highly distinct.")
    elif score > 0.5:
        print("Conclusion: Good clustering, but some groups might overlap slightly.")
    else:
        print("Conclusion: Weak clustering. The data might not naturally group well.")

if __name__ == "__main__":
    run_kmeans_clustering()