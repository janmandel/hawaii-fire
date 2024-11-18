# The file paths
## Define the base directory (main)
main = osp.join('C:/', 'Users', 'T-Spe', 'OneDrive', 'School', "Fall '25", "Master's Project", 'test')
tifdata = osp.join(main, 'tifdata')

## topography paths
slope_path = osp.join(main, 'LH20_SlpP_220.tif')
elevation_path = osp.join(main, 'LH20_Elev_220.tif')
aspect_path = osp.join(main, 'LH20_Asp_220.tif')


## vegetation paths
fuelmod_path = osp.join(main, 'LH20_F13_200.tif')
fuelvat_path = osp. join(main, 'LH20_F13_200.tif.vat.dbf')

# meteorology paths (wrf outputs)
wrfout_path = osp.join(main, 'wrfout_d02_2020-08-29_14-00-00')

# Extraction of the data from the files
wrfdata = nc.Dataset(wrfout_path)

# Extract projection parameters
CEN_LAT = wrfdata.getncattr('CEN_LAT')
CEN_LON = wrfdata.getncattr('CEN_LON')
TRUELAT1 = wrfdata.getncattr('TRUELAT1')
TRUELAT2 = wrfdata.getncattr('TRUELAT2')
STAND_LON = wrfdata.getncattr('STAND_LON')
MAP_PROJ = wrfdata.getncattr('MAP_PROJ')
MAP_PROJ_CHAR = wrfdata.getncattr('MAP_PROJ_CHAR')

print(f"CEN_LAT: {CEN_LAT}")
print(f"CEN_LON: {CEN_LON}")
print(f"TRUELAT1: {TRUELAT1}")
print(f"TRUELAT2: {TRUELAT2}")
print(f"STAND_LON: {STAND_LON}")
print(f"MAP_PROJ: {MAP_PROJ}")
print(f"MAP_PROJ_CHAR: {MAP_PROJ_CHAR}")

# Step 2: Define the target CRS
if MAP_PROJ == 1:
    target_crs = CRS.from_dict({
        'proj': 'lcc',
        'lat_1': TRUELAT1,
        'lat_2': TRUELAT2,
        'lat_0': CEN_LAT,
        'lon_0': STAND_LON,
        'x_0': 0,
        'y_0': 0,
        'units': 'm',
        'datum': 'WGS84',
        'no_defs': True
    })
else:
    raise ValueError("Unsupported MAP_PROJ value")

# Step 3: Reproject raster datasets
def reproject_raster(src_path, dst_path, dst_crs):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs.to_wkt(),
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
    print(f"Reprojected {src_path} to {dst_path}")

# Create a directory for reprojected files
reprojected_dir = osp.join(main, 'reprojected')
os.makedirs(reprojected_dir, exist_ok=True)

# Reproject each dataset
for name, path in datasets.items():
    reprojected_filename = f"{name}_reproj.tif"
    reprojected_path = osp.join(reprojected_dir, reprojected_filename)
    reproject_raster(path, reprojected_path, target_crs)
    # Update the path to the reprojected file in the datasets dictionary
    datasets[name] = reprojected_path