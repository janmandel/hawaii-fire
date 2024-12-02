import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.input_modifiers import Normalize
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
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

    # --- Correlation Matrix ---
    plt.figure(figsize=(10, 8))

    # Compute the correlation matrix
    corr_matrix = X.corr()

    # Round the correlation values for display
    corr_matrix_rounded = corr_matrix.round(2)

    # Create the heatmap with rounded values and cleaner axes
    sns.heatmap(
        corr_matrix_rounded,
        annot=True,  # Display the correlation values
        fmt=".2f",  # Format the annotation to 2 decimal places
        cmap='coolwarm',  # Use a clean colormap
        cbar=True,  # Include a color bar
        square=True,  # Make the heatmap square-shaped
        linewidths=0.5  # Add small lines between the cells for separation
    )

    # Update the axes for clarity
    plt.xticks(rotation=45, ha="right")  # Rotate and align x-axis labels
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.title('Feature Correlation Matrix', fontsize=16)  # Add a title
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig("correlation_matrix.png")  # Save the plot to a file
    plt.close()  # Close the figure to free memory
    print("Correlation matrix saved to 'correlation_matrix.png'")

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
    y_pred = (y_pred_proba >= 0.5).astype(int)

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


def visualize_with_tf_keras_vis(model, X_sample, feature_columns, output_path):
    """
    Visualize feature importance using tf-keras-vis Integrated Gradients and save the visualization.

    Args:
        model: Trained Keras model.
        X_sample: Sample data for explanation (numpy array).
        feature_columns: List of feature column names.
        output_path: File path to save the visualization.
    """
    print("Visualizing feature importance with tf-keras-vis...")

    # Define the score function
    score = CategoricalScore([1])  # Focus on the class of interest (fire occurrence)

    # Initialize Integrated Gradients
    saliency = Saliency(model, model_modifier=None, clone=True)

    # Generate saliency maps
    saliency_map = saliency(score, X_sample, smooth_samples=20, smooth_noise=0.1)

    # Aggregate the saliency maps for all features
    aggregated_saliency = np.mean(saliency_map[0], axis=0)

    # Create a bar plot for feature attributions
    feature_attributions = dict(zip(feature_columns, aggregated_saliency))
    plt.figure(figsize=(10, 6))
    plt.bar(feature_attributions.keys(), feature_attributions.values(), color='skyblue')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylabel("Saliency Attribution")
    plt.title("Feature Importances (Saliency)", fontsize=14)
    plt.tight_layout()

    # Save the visualization
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Visualization saved to {output_path}")

    # Print attributions for logging
    print("\nFeature Importance (Saliency):")
    for feature, attribution in feature_attributions.items():
        print(f"{feature}: {attribution:.4f}")


def save_model(model, file_path):
    """Save the trained model to disk."""
    print(f"Saving model to {file_path}...")
    model.save(file_path)
    print("Model saved.")


def load_trained_model(file_path):
    """Load a previously trained model."""
    print(f"Loading model from {file_path}...")
    return load_model(file_path)


# --- Main Function ---
def main():
    """Main function to train, evaluate, or analyze the model."""
    data_path = 'processed_data.pkl'
    model_path = 'dnn_model.hd5'

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
    evaluate_model(model, X_test, y_test)

    # Use a single sample for visualization
    X_sample = X_train.iloc[:1].values  # Extract the first example from the training set
    visualize_with_tf_keras_vis(model, X_sample, feature_columns, output_path="tf_keras_vis_visualization.png")


if __name__ == "__main__":
    main()
