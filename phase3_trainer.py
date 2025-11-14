import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Trainer script started.")

# Define file names
DATASET_FILE = 'asl_dataset.csv'
MODEL_FILE = 'asl_model.joblib'

# 1. Load the dataset
print(f"Loading dataset from {DATASET_FILE}...")
try:
    data = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print(f"ERROR: Dataset file not found at {DATASET_FILE}")
    print("Please run the data collector script first.")
    exit()

# Handle potential NaN values (if any)
data = data.dropna()

if data.empty:
    print("ERROR: The dataset is empty. Please collect some data first.")
    exit()

# 2. Separate features (X) and labels (y)
print("Separating data into features (X) and labels (y)...")
X = data.drop('label', axis=1)  # All columns EXCEPT 'label'
y = data['label']              # Only the 'label' column

# 3. Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 4. Initialize and Train the Model
print("Training the Random Forest model... This may take a moment.")
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

print("Model training complete.")

# 5. Evaluate the Model
print("Evaluating model accuracy on the test set...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'----------------------------------------')
print(f'Model Accuracy on Test Data: {accuracy * 100:.2f}%')
print(f'----------------------------------------')

if accuracy < 0.8:
    print("WARNING: Model accuracy is low. Consider collecting more data for each letter.")

# 6. Save the Trained Model
print(f"Saving the trained model to {MODEL_FILE}...")
joblib.dump(model, MODEL_FILE)

print(f"Model saved successfully to {MODEL_FILE}")
print("Trainer script finished.")
