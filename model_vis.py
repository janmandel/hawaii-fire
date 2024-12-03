import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from keras.models import Sequential, load_model
import os
import rasterio
from rasterio.plot import show
from pyproj import Transformer

# --- Utility Functions ---
def load_trained_model(file_path):
    """Load a previously trained model."""
    print(f"Loading model from {file_path}...")
    return load_model(file_path)

def load_and_process_data(file_path, model):
    """Load, preprocess data, and evaluate probabilities for ML model."""
    print("Loading dataset...")
    df = pd.read_pickle(file_path)

    # Encoding categorical features
    print("Encoding categorical features...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    fuelmod_encoded = encoder.fit_transform(df[['fuelmod']])
    fuelmod_feature_names = encoder.get_feature_names_out(['fuelmod'])
    fuelmod_df = pd.DataFrame(fuelmod_encoded, columns=fuelmod_feature_names)

    # Standardize numerical features
    print("Standardizing numerical features...")
    scaler = StandardScaler()
    numeric_features = ['temp', 'rain', 'rhum', 'wind', 'sw', 'elevation', 'slope', 'aspect']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Concatenate processed features
    print("Concatenating processed features...")
    processed_df = pd.concat([df, fuelmod_df], axis=1)
    processed_df.drop('fuelmod', axis=1, inplace=True)

    # Define feature columns
    feature_columns = numeric_features + list(fuelmod_feature_names)
    X = processed_df[feature_columns]

    # Evaluate probabilities using the model
    print("Predicting probabilities for fire occurrence...")
    probabilities = model.predict(X).flatten()

    # Add probabilities to the original DataFrame
    df['fire_probability'] = probabilities

    # Compute statistics
    print(df['fire_probability'].describe())

    print("Probabilities added to the DataFrame.")
    return df


def plot_fire_occurrences(fire_df, raster_path, output_path):
    """
    Plots fire occurrence points from the DataFrame on a raster map.

    Parameters:
    - df_fire_samples: DataFrame with fire samples containing 'lon' and 'lat' columns.
    - raster_path: Path to a raster file for the background (e.g., slope or elevation).
    - output_path: Path to save the output plot.
    """
    # Load the raster to get the island outline
    with rasterio.open(raster_path) as raster:
        island_data = raster.read(1)  # Read raster data
        raster_transform = raster.transform
        raster_crs = raster.crs
        island_nodata = raster.nodata

    # Mask NoData values to only show the island
    island_data_masked = np.ma.masked_where(island_data == island_nodata, island_data)

    # Transform fire occurrence points (longitude and latitude)
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    fire_lon = fire_df['lon'].values  # Longitude from your fire dataframe
    fire_lat = fire_df['lat'].values  # Latitude from your fire dataframe
    raster_x, raster_y = transformer.transform(fire_lon, fire_lat)

    # Plot the island outline and fire occurrence points
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot the island shape
    show(
        island_data_masked,
        transform=raster_transform,
        ax=ax,
        cmap='Greys',  # Grey colormap for the island
        title="Fire Occurrence Map with Island Outline"
    )

    # Overlay fire points
    ax.scatter(raster_x, raster_y, color='red', s=10, label="Fire Points")

    # Add title, labels, and legend
    ax.set_title("Fire Occurrence Points on Hawai'i Island from 2011-2024", fontsize=16)
    ax.legend(fontsize=12, loc='lower right')

    # Save and show the plot
    plt.savefig(output_path, dpi=300)
    plt.show()

#"""Main function to train, evaluate, and analyze the model."""
# Main Execution
if __name__ == "__main__":
    data_path = 'processed_data.pkl'
    model_path = 'dnn_model.keras'

    # Load model
    if os.path.exists(model_path):
        print("Pre-trained model found. Loading...")
        model = load_trained_model(model_path)
    else:
        print("Model not found. Please train or provide a model.")
        exit()

    # Load data, evaluate model and add to dataframe the probabilities a fire occurred
    df_prob_path = 'processed_data_with_probabilities.pkl'
    save = True
    if os.path.exists(df_prob_path):
        print("Required dataframe found, loading...")
        df_prob = pd.read_pickle(df_prob_path)
        # Compute statistics
        print(df_prob['fire_probability'].describe())
    else:
        print("Creating the required dataframe...")
        df_prob = load_and_process_data(data_path, model)
        if save:
            # Save the updated DataFrame for further use
            df_prob.to_pickle(df_prob_path)
            print(f"Updated DataFrame saved to {df_prob_path}")

    # Create the fire inventory map
    fire_map_path = 'fire_map.png'
    base_dir = os.path.join('/', 'home', 'spearsty', 'p', 'data')
    raster_path = os.path.join(base_dir, 'feat', 'landfire', 'top', 'LF2020_SlpP_220_HI', 'LH20_SlpP_220.tif')
    if os.path.exists(fire_map_path):
        print("Fire inventory map exists in current directory, moving on...")
    else:
        print("Creating the fire inventory map...")
        df_fire = df_prob[df_prob['label'] == 1]
        plot_fire_occurrences(df_fire, raster_path, fire_map_path)
