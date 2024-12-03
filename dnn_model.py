import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os


# --- Utility Functions ---
def load_and_preprocess_data(file_path):
    """Load and preprocess data for ML model training."""
    print("Loading dataset...")
    df = pd.read_pickle(file_path)
    print("Encoding categorical features...")

    # One-hot encode 'fuelmod' feature
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    fuelmod_encoded = encoder.fit_transform(df[['fuelmod']])
    fuelmod_feature_names = encoder.get_feature_names_out(['fuelmod'])
    fuelmod_df = pd.DataFrame(fuelmod_encoded, columns=fuelmod_feature_names)

    # Standardize numerical features
    scaler = StandardScaler()
    numeric_features = ['temp', 'rain', 'rhum', 'wind', 'sw', 'elevation', 'slope', 'aspect']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Concatenate processed features
    processed_df = pd.concat([df, fuelmod_df], axis=1)
    processed_df.drop('fuelmod', axis=1, inplace=True)

    # Define feature columns and target
    feature_columns = numeric_features + list(fuelmod_feature_names)
    X = processed_df[feature_columns]
    y = processed_df['label']

    return X, y, feature_columns


def create_dnn_model(input_dim):
    """Build and compile a DNN model."""
    print("Creating DNN model...")
    model = Sequential([
        InputLayer(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_dnn_model(model, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
    """Train the DNN model with early stopping."""
    print("Training DNN model...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=False
    )
    print("Training completed.")
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and visualize performance metrics."""
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test).flatten()
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='green', lw=2, label=f'DNN ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.close()

def save_model(model, file_path):
    """Save the trained model to disk."""
    print(f"Saving model to {file_path}...")
    model.save(file_path)
    print("Model saved.")

def load_trained_model(file_path):
    """Load a previously trained model."""
    print(f"Loading model from {file_path}...")
    return load_model(file_path)

def integrated_gradients(model, baseline, input_data, steps=50):
    """
    Compute Integrated Gradients for a model and input data.

    Args:
        model (keras.Model): The trained model.
        baseline (np.ndarray): Baseline input to compare against (e.g., zeros or mean values).
        input_data (np.ndarray): Input data for which to compute gradients.
        steps (int): Number of steps for the IG approximation.

    Returns:
        np.ndarray: Integrated gradients for each feature.
    """
    # Generate scaled inputs
    scaled_inputs = np.linspace(baseline, input_data, steps).reshape(steps, -1)

    # Convert inputs to tensors
    scaled_inputs = tf.convert_to_tensor(scaled_inputs, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(scaled_inputs)
        predictions = model(scaled_inputs)

    # Get gradients
    gradients = tape.gradient(predictions, scaled_inputs)

    # Compute average gradients
    avg_gradients = tf.reduce_mean(gradients, axis=0).numpy()

    # Compute integrated gradients
    int_grad_comp = (input_data - baseline) * avg_gradients

    return int_grad_comp


def interpret_features_class_specific(model, X, y, feature_columns, steps=50):
    """
    Perform class-specific Integrated Gradients analysis.

    Args:
        model (keras.Model): The trained model.
        X (pd.DataFrame): Feature matrix (scaled).
        y (pd.Series): Target labels.
        feature_columns (list): List of feature names.
        steps (int): Number of steps for IG approximation. Default is 50.

    Returns:
        dict: Weighted feature importances.
    """
    # Convert X to NumPy array for direct row access
    X_array = X.to_numpy()
    y_array = y.to_numpy()

    # Separate fire and non-fire samples
    fire_indices = y_array == 1
    non_fire_indices = y_array == 0

    X_fire = X_array[fire_indices]
    X_non_fire = X_array[non_fire_indices]

    # Baselines (mean of the non-fire and fire samples)
    baseline_non_fire = np.mean(X_non_fire, axis=0)
    baseline_fire = np.mean(X_fire, axis=0)

    print("Computing Integrated Gradients...")

    def compute_ig_for_class(baseline, inputs):
        igs = [integrated_gradients(model, baseline, x, steps=steps) for x in inputs]
        return np.mean(igs, axis=0)

    # Compute IG for each class
    print("Computing IG for fire samples...")
    ig_fire = compute_ig_for_class(baseline_fire, X_fire)

    print("Computing IG for non-fire samples...")
    ig_non_fire = compute_ig_for_class(baseline_non_fire, X_non_fire)

    # Combine using class weights
    fire_weight = len(fire_indices) / len(y)
    non_fire_weight = len(non_fire_indices) / len(y)

    weighted_importance = fire_weight * ig_fire + non_fire_weight * ig_non_fire

    # Display and visualize results
    feature_importances = dict(zip(feature_columns, weighted_importance))
    print("\nWeighted Feature Importances:")
    for feature, importance in feature_importances.items():
        print(f"{feature}: {importance:.4f}")

    # Sort for visualization
    sorted_features = sorted(feature_importances, key=feature_importances.get, reverse=True)
    sorted_importances = [feature_importances[feature] for feature in sorted_features]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_features, sorted_importances)
    plt.xticks(rotation=45, ha='right')
    plt.title("Weighted Feature Importances via Integrated Gradients")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("weighted_feature_importances.png")
    plt.show()

    return feature_importances

#"""Main function to train, evaluate, and analyze the model."""
# Main Execution
if __name__ == "__main__":
    data_path = 'processed_data.pkl'
    model_path = 'dnn_model.keras'
    evaluate = False

    # Load data
    X, y, feature_columns = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load or train model
    if os.path.exists(model_path):
        print("Pre-trained model found. Loading...")
        model = load_trained_model(model_path)
    else:
        print("No pre-trained model found. Training a new model...")
        model = create_dnn_model(input_dim=X_train.shape[1])
        train_dnn_model(model, X_train, y_train)
        save_model(model, model_path)

    # Evaluate model
    if evaluate:
        evaluate_model(model, X_test, y_test)

    # Interpret features
    interpret_features_class_specific(model, X_train, y_train, feature_columns)



