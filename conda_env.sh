conda create -n gdal_env python=3.11 gdal -c conda-forge 
conda activate gdal_env
conda install -y netCDF4 pandas matplotlib rasterio dbfread  pygrib -c conda-forge
