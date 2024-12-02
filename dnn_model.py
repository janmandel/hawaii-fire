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
from vis.visualization import visualize_saliency
from vis.utils import utils
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


def generate_saliency_map(model_path, X_data, sample_index, layer_name, class_index, output_path):
    """
    Generate and save a saliency map for a given sample.

    Parameters:
    - model_path (str): Path to the trained model file.
    - X_data (np.ndarray): Dataset from which the sample is selected.
    - sample_index (int): Index of the sample to visualize.
    - layer_name (str): Name of the layer to analyze.
    - class_index (int): Index of the class for which saliency is computed.
    - output_path (str): Path to save the generated saliency map.

    Returns:
    - None
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print(f"Preparing layer modifications for saliency visualization...")
    layer_idx = utils.find_layer_idx(model, layer_name)

    # Optional: Swap activation to linear for better gradient calculation
    model.layers[layer_idx].activation = None
    model = utils.apply_modifications(model)

    print(f"Generating saliency map for sample {sample_index}, class {class_index}...")
    sample_input = X_data[sample_index].reshape(1, -1)
    saliency_map = visualize_saliency(model, layer_idx, filter_indices=class_index, seed_input=sample_input)

    print(f"Plotting and saving saliency map to {output_path}...")
    plt.figure(figsize=(10, 8))
    plt.title(f'Saliency Map for Sample {sample_index}', fontsize=16)
    plt.imshow(saliency_map, cmap='viridis')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saliency map saved to {output_path}.")


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
    saliency_output_path = 'saliency_map_sample_0.png'

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

    # Generate saliency map for a specific sample
    generate_saliency_map(
        model_path=model_path,
        X_data=X_test,
        sample_index=0,  # Example: First test sample
        layer_name='dense_2',  # Adjust based on your architecture
        class_index=1,  # Example: Fire class
        output_path=saliency_output_path
    )

    if __name__ == "__main__":
        main()