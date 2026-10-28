"python -m pip install scikit-learn matplotlib"

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def run_handwritten_digit_recognition():
    # 1. LOAD THE DATA
    print("Loading handwritten digits dataset...")
    digits = load_digits()
    
    # X contains the raw image pixels, y contains the actual number (0-9)
    X = digits.data
    y = digits.target

    print(f"Dataset loaded! Total images: {len(X)}")
    print(f"Each image is an 8x8 grid of pixels (flattened to {X.shape[1]} numbers).")
    print("-" * 50)

    # 2. PREPROCESS / SCALE THE DATA
    # The pixels range from 0 (white) to 16 (black). 
    # Neural networks learn much faster when numbers are between 0 and 1.
    X_scaled = X / 16.0 

    # 3. SPLIT THE DATA
    # 80% to train the neural network, 20% to test it on unseen handwriting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. BUILD AND TRAIN THE NEURAL NETWORK
    print("\nBuilding Neural Network (Multi-Layer Perceptron)...")
    # We are creating a network with two hidden layers: 
    # Layer 1 has 64 "neurons", Layer 2 has 32 "neurons".
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), 
                             max_iter=1000, 
                             random_state=42,
                             verbose=False) # Set to True to see the math happening in real-time
    
    print("Training the network (this might take a few seconds)...")
    nn_model.fit(X_train, y_train)
    print("Training complete!")

    # 5. TEST THE NEURAL NETWORK
    print("\n" + "=" * 50)
    print("--- NEURAL NETWORK EVALUATION ---")
    y_pred = nn_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Confusion Matrix (Rows=Actual, Columns=Predicted):")
    # This shows us exactly which numbers confused the AI (e.g., mistaking a 3 for an 8)
    print(confusion_matrix(y_test, y_pred))

    # 6. VISUALIZE A SPECIFIC PREDICTION
    print("-" * 50)
    print("--- VISUALIZING A TEST IMAGE ---")
    print("A popup window will now open showing an image from the test set.")
    print("Check your taskbar if you don't see it right away!")
    
    # Pick a random image from the test set (e.g., image index 50)
    test_index = 50
    test_image_pixels = X_test[test_index]
    actual_number = y_test[test_index]
    predicted_number = y_pred[test_index]
    
    # Reshape the 64 flattened pixels back into an 8x8 image grid so we can see it
    image_grid = test_image_pixels.reshape(8, 8)
    
    # Use Matplotlib to draw the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_grid, cmap=plt.cm.gray_r)
    plt.title(f"AI Predicted: {predicted_number}\nActual Number: {actual_number}", 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # This pauses the script and opens the image window
    plt.show()

if __name__ == "__main__":
    run_handwritten_digit_recognition()