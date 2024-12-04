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
from scipy.interpolate import griddata
from dbfread import DBF

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


def add_geographic_ticks(ax, raster_crs, extent, num_ticks=6, fontsize=14):
    """
    Add geographic ticks (latitude and longitude) to a plot.

    Args:
        ax (matplotlib.axes.Axes): The axis to add the geographic ticks.
        raster_crs (str): The CRS of the raster, e.g., 'EPSG:32633'.
        extent (list): The extent of the raster in the format [xmin, xmax, ymin, ymax].
        num_ticks (int): Number of ticks to generate along each axis. Default is 6.
        fontsize (int): Font size for tick labels. Default is 14.
    """
    xmin, xmax, ymin, ymax = extent
    transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

    # Generate tick positions in raster CRS
    xticks_raster = np.linspace(xmin, xmax, num=num_ticks)
    yticks_raster = np.linspace(ymin, ymax, num=num_ticks)

    # Transform tick positions to WGS84
    xticks_geo, _ = transformer.transform(xticks_raster, np.full_like(xticks_raster, ymin))
    _, yticks_geo = transformer.transform(np.full_like(yticks_raster, xmin), yticks_raster)

    # Set ticks and labels on the axis
    ax.set_xticks(xticks_raster)
    ax.set_xticklabels([f"{lon:.2f}°" for lon in xticks_geo], fontsize=fontsize)
    ax.set_yticks(yticks_raster)
    ax.set_yticklabels([f"{lat:.2f}°" for lat in yticks_geo], fontsize=fontsize)

    # Add axis labels
    ax.set_xlabel('Longitude', fontsize=fontsize + 4)
    ax.set_ylabel('Latitude', fontsize=fontsize + 4)

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
        raster_extent = [raster.bounds.left, raster.bounds.right, raster.bounds.bottom, raster.bounds.top]

    # Mask NoData values to only show the island
    island_data_masked = np.ma.masked_where(island_data == island_nodata, island_data)

    # Transform fire occurrence points (longitude and latitude)
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    fire_lon = fire_df['lon'].values  # Longitude from your fire dataframe
    fire_lat = fire_df['lat'].values  # Latitude from your fire dataframe
    raster_x, raster_y = transformer.transform(fire_lon, fire_lat)

    # Plot the island outline and fire occurrence points
    fig, ax = plt.subplots(figsize=(20, 15))

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

    # Add geographic ticks
    add_geographic_ticks(ax, raster_crs, raster_extent, num_ticks=6, fontsize=16)

    # Add title, labels, and legend
    ax.set_title("Fire Occurrence Points on Hawai'i Island from 2011-2024 (n = 1976)", fontsize=30)
    ax.legend(fontsize=14, loc='lower right')

    # Save and show the plot
    plt.savefig(output_path, dpi=300)
    print(f"The Fire inventory map was saved as {output_path}")
    plt.show()

def create_fire_susceptibility_map(df_prob, raster_path, fuelmod_path, fuelvat_path, output_image_path, interpolate_missing=False, single_hour=True):
    """
    Create and save a fire susceptibility map overlaid on the island outline,
    with optional interpolation for cells without assigned probabilities.
    Interpolation is only performed over valid land areas defined by the fuelmod raster.

    Parameters:
    - df_prob: DataFrame containing 'lon', 'lat', 'fire_probability', samples used to build model).
    - raster_path: Path to a raster file for background (e.g., slope or elevation).
    - fuelmod_path: Path to the fuelmod raster file.
    - fuelvat_path: Path to the VAT file associated with the fuelmod raster.
    - output_image_path: Path to save the output fire susceptibility PNG image.
    - interpolate_missing: Boolean flag to interpolate probabilities for cells without assigned probabilities.
    """

    # Load the background raster to get affine transformation and dimensions
    with rasterio.open(raster_path) as src:
        background_data = src.read(1)  # Read raster data
        raster_transform = src.transform
        raster_crs = src.crs
        raster_width = src.width
        raster_height = src.height
        background_nodata = src.nodata
        raster_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Load the fuelmod raster
    with rasterio.open(fuelmod_path) as fuelmod_dataset:
        fuelmod_data = fuelmod_dataset.read(1)
        fuelmod_nodata = fuelmod_dataset.nodata

    # Replace nodata values with NaN
    if fuelmod_nodata is not None:
        fuelmod_data = np.where(fuelmod_data == fuelmod_nodata, np.nan, fuelmod_data)

    # Load VAT file and map pixel values to vegetation classes
    vat_df = pd.DataFrame(iter(DBF(fuelvat_path)))
    value_to_class = dict(zip(vat_df['VALUE'], vat_df['FBFM13']))

    # Define a mapping function that handles NaN values
    def map_value_to_class(value):
        if np.isnan(value):
            return 'NoData'
        else:
            return value_to_class.get(value, 'Unknown')

    # Apply the mapping function to the fuelmod data
    map_func = np.vectorize(map_value_to_class)
    fuel_classes = map_func(fuelmod_data)

    # Define invalid fuel categories
    invalid_fuel_categories = ['Barren', 'Water', 'Urban', 'Fill-NoData', 'NoData', 'Unknown']

    # Create the valid land mask
    valid_land_mask = ~np.isin(fuel_classes, invalid_fuel_categories)

    # Identify the date with the maximum samples
    date_sample_counts = df_prob.groupby('date').size()
    max_samples_date = date_sample_counts.idxmax()
    max_samples_count = date_sample_counts.max()

    print(f"Using data from {max_samples_date} with {max_samples_count} samples for mapping.")

    # Filter the data for the identified date
    df_prob_filtered = df_prob[df_prob['date'] == max_samples_date]

    if single_hour:
        df_prob = df_prob_filtered

    # Transform lon/lat to raster coordinates
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    x_coords, y_coords = transformer.transform(df_prob['lon'].values, df_prob['lat'].values)

    # Invert the affine transformation to get pixel indices
    inv_transform = ~raster_transform
    col_indices, row_indices = inv_transform * (x_coords, y_coords)
    col_indices = col_indices.astype(int)
    row_indices = row_indices.astype(int)

    # Ensure indices are within the raster dimensions
    valid_indices_mask = (
        (row_indices >= 0) & (row_indices < raster_height) &
        (col_indices >= 0) & (col_indices < raster_width)
    )
    if not valid_indices_mask.all():
        print("Warning: Some row or column indices are out of bounds and will be ignored.")
    # Filter the data to valid indices
    row_indices = row_indices[valid_indices_mask]
    col_indices = col_indices[valid_indices_mask]
    probabilities = df_prob['fire_probability'].values[valid_indices_mask]

    # Create a DataFrame with row, col, and probabilities
    df_cells = pd.DataFrame({
        'row': row_indices,
        'col': col_indices,
        'fire_probability': probabilities
    })

    # Aggregate probabilities for each cell by averaging
    df_cell_probs = df_cells.groupby(['row', 'col'])['fire_probability'].mean().reset_index()

    # Create an empty array to hold the susceptibility values
    susceptibility_array = np.full((raster_height, raster_width), np.nan, dtype=np.float32)

    # Assign aggregated probabilities to the array
    susceptibility_array[df_cell_probs['row'], df_cell_probs['col']] = df_cell_probs['fire_probability']

    # Create a mask for valid land cells without assigned probabilities
    mask_to_interpolate = valid_land_mask & np.isnan(susceptibility_array)

    if interpolate_missing:
        print("Interpolating missing probabilities for valid land cells...")
        # Prepare data for interpolation
        known_points = np.array((df_cell_probs['col'], df_cell_probs['row'])).T
        known_values = df_cell_probs['fire_probability'].values

        # Points to interpolate
        grid_x, grid_y = np.meshgrid(np.arange(raster_width), np.arange(raster_height))
        unknown_points = np.array((grid_x[mask_to_interpolate], grid_y[mask_to_interpolate])).T

        # Perform interpolation using the specified method
        interpolated_values = griddata(
            known_points,
            known_values,
            unknown_points,
            method='cubic'  # Options: 'nearest', 'linear', 'cubic'
        )

        # Assign interpolated values back to the susceptibility array
        susceptibility_array[mask_to_interpolate] = interpolated_values

    # Mask out areas outside valid land
    susceptibility_array = np.ma.masked_where(~valid_land_mask, susceptibility_array)

    # Report statistics
    total_valid_cells = valid_land_mask.sum()
    cells_with_probabilities = np.isfinite(susceptibility_array).sum()
    cells_without_probabilities = total_valid_cells - cells_with_probabilities
    print(f"Total valid land cells: {total_valid_cells}")
    print(f"Cells with assigned probabilities: {cells_with_probabilities}")
    print(f"Cells without assigned probabilities: {cells_without_probabilities}")

    # Plot and save the fire susceptibility map overlaid on the background raster
    fig, ax = plt.subplots(figsize=(20, 15))

    if single_hour:
        plot_title = f"Fire Susceptibility Map for Hawai'i Island (Date: {max_samples_date})"
    else:
        plot_title = "Fire Susceptibility Map over Hawai'i Island"

    # Plot the background raster
    show(
        background_data,
        transform=raster_transform,
        ax=ax,
        cmap='Greys',
        title=plot_title
    )

    # Overlay the susceptibility map with transparency
    cax = ax.imshow(
        susceptibility_array,
        extent=raster_extent,
        origin='upper',
        cmap='hot',
        alpha=0.8,
        interpolation='none'
    )

    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label('Fire Probability', fontsize=20)

    # Add geographic ticks
    add_geographic_ticks(ax, raster_crs, raster_extent, num_ticks=6, fontsize=16)

    # Save and show the plot
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Fire susceptibility map saved as {output_image_path}")
    plt.show()


#"""Main function to implement the model."""
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

    # Create the fire susceptibility map
    susceptibility_map_path = 'fire_susceptibility_map.png'
    # Paths to the fuelmod raster and VAT file
    fuelmod_path = os.path.join(base_dir, 'feat', 'landfire', 'afbfm', 'LF2022_FBFM13_230_HI', 'LH22_F13_230.tif')
    fuelvat_path = os.path.join(base_dir, 'feat', 'landfire', 'afbfm', 'LF2022_FBFM13_230_HI', 'LH22_F13_230.tif.vat.dbf')

    if os.path.exists(susceptibility_map_path):
        print("Fire susceptibility map exists in current directory, hello world!")
    else:
        print("Creating the fire susceptibility map...")
        create_fire_susceptibility_map(
            df_prob,
            raster_path,
            fuelmod_path,
            fuelvat_path,
            susceptibility_map_path,
            interpolate_missing=True,  # Set to True if you want to interpolate missing values
            single_hour=True
        )
