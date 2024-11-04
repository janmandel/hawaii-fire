# purpose: map from grid given by longitude and latitude to indices i,j
# usage:
#
# from lonlat_interp import Coord_to_inde
#
# interp = Coord_to_index(degree-3)
# interp.build(
# interp.build(lon_grid, lat_grid)
# ia,ja = interp.evaluate(lon_array,lat_array)
# 


import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt

debug_print = False

def print_var(name,var):
    if debug_print:
        print(f"{name} shape {var.shape}\n{var}")

class Interpolator:
    def __init__(self, degree):
        self.degree = degree  # Degree of Legendre polynomials
        self.coefficients = None  # To store calculated coefficients

    def _normalize_points(self, points, bounds):
        # Normalize points to the range [-1, 1] based on the provided bounds
        min_vals, max_vals = bounds
        return 2 * (points - min_vals) / (max_vals - min_vals) - 1

    def _compute_basis_matrix(self, points, degree):
        # Generate the basis matrix for tensor product Legendre polynomials
        n = points.shape[0]
        basis_matrix = np.ones((n, (degree + 1) ** 2))
        
        for i in range(degree + 1):
            for j in range(degree + 1):
                legendre_i = Legendre.basis(i)
                legendre_j = Legendre.basis(j)
                basis_matrix[:, i * (degree + 1) + j] = legendre_i(points[:, 0]) * legendre_j(points[:, 1])
                
        return basis_matrix

    def build(self, x_points, f_values):
        print_var('Interpolator: build: x_points',x_points)
        print_var('Interpolator: build: f_values',f_values)
       # print(np.min(x_points, axis=0))
        bounds = np.array([np.min(x_points, axis=0), np.max(x_points, axis=0)])
        # bounds = np.array([x_points[:,0].min(), axis=0), np.max(x_points, axis=0)])
        print_var('bounds',bounds)
        normalized_points = self._normalize_points(x_points, bounds)
        
        # Compute basis matrix for the tensor product of Legendre polynomials
        basis_matrix = self._compute_basis_matrix(normalized_points, self.degree)
        
        # Solve least-squares problem to get coefficients to store
        self.coefficients, residuals, _, _ = np.linalg.lstsq(basis_matrix, f_values, rcond=None)
        self.bounds = bounds  # Store bounds for normalization in evaluation

    def evaluate(self, x_points):
        print_var('Interpolator: evaluate: x_points',x_points)
        if self.coefficients is None:
            raise ValueError("Model has not been built. Please call build() first.")
        
        normalized_points = self._normalize_points(x_points, self.bounds)
        basis_matrix = self._compute_basis_matrix(normalized_points, self.degree)
        ret_val = basis_matrix @ self.coefficients
        print_var('Interpolator: evaluate: ret_val',ret_val)
        return ret_val

def error_metrics(a, b):
    residuals = a - b
    mrs = np.mean(residuals ** 2)
    amax = np.max(np.abs(residuals))
    print("Mean Square Error (MRS):", mrs)
    print("Maximum Absolute Error (amax):", amax)
    return mrs, amax

def generate_test_grid(imax, jmax, xsize=1, ysize=1, deformation=0):
    """ 
    Generate a synthetic test grid with perturned lat/lon values within specified bounds.
    """
    print('generate_test_grid imax',imax,'jmax',jmax,'xsize',xsize,'ysize',ysize,'deformation',deformation)
    lon = np.linspace(0, xsize, imax) 
    lat = np.linspace(0, ysize, jmax)
    lat_grid, lon_grid = np.meshgrid(lat, lon)
    for i in range(imax):
        for j in range(jmax):
            lon_grid[i, j] = lon_grid[i, j] + xsize * deformation * np.cos(2 * j / jmax)
            lat_grid[i, j] = lat_grid[i, j] + ysize * deformation * np.sin(2 * i / imax)

    return lon_grid, lat_grid

def plot_lon_lat_grid(lon, lat):
    """
    Plots a 2D grid with given longitude and latitude coordinates.
    
    Parameters:
    lon (2D array): Array of longitudes
    lat (2D array): Array of latitudes
    """
    # Plotting the grid
    plt.figure(figsize=(10, 5))
    plt.scatter(lon, lat, s=5, color='blue')  # Scatter plot of grid points
    
    # Labeling the plot
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('2D Grid of Longitude and Latitude Coordinates')
    plt.grid(True)
    plt.show(block=False)
    plt.pause(0.1) 

def test_reproduce_smooth_values(degree = 3, xsize=1, ysize=1, imax = 10, jmax=16, deformation=0.05):

    print('Test reproducing a smooth function on a slightly smoothly deformed grid, degree=',degree)
    # Generate a test grid of x_points using the provided function
    lon_grid, lat_grid = generate_test_grid(imax, jmax, xsize=xsize, ysize=ysize, deformation=deformation)
    plot_lon_lat_grid(lon_grid, lat_grid)

    # Define a smooth f values on this grid
    f_grid = np.sin(np.pi * lon_grid/xsize) * np.cos(np.pi * lat_grid/ysize)

    # Reshape as columns
    x_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    f_values = f_grid.ravel()
    print_var('x_points',x_points)
    print_var('f_values',f_values)

    # Create the Interpolator object and build the model
    interp = Interpolator(degree)
    interp.build(x_points, f_values)

    # Evaluate on the same points
    f_approx = interp.evaluate(x_points)

    # Calculate and print error metrics on the test data
    return error_metrics(f_values, f_approx)

class Coord_to_index:
    # provide inverse of the mapping [i,j] -> [ lon_grid[i,j], lon_grid[i,j] ] 
    def __init__(self, degree=4, ):
        self.degree = degree  # Degree of Legendre polynomials

    def build(self, lon_grid, lat_grid):
        print_var('Coord_to_index: build: lon_grid',lon_grid)
        print_var('Coord_to_index: build: lat_grid',lat_grid)
        assert lon_grid.shape == lat_grid.shape, "The arrays lon_grid and lat_grid must have the same shape."
        imax, jmax = lon_grid.shape
        self.grid_shape = lon_grid.shape
        j_grid, i_grid = np.meshgrid(np.arange(jmax), np.arange(imax))
        print_var('Coord_to_index: build: i_grid',i_grid)
        print_var('Coord_to_index: build: j_grid',j_grid)
         # Reshape as columns
        x_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
        i_values = i_grid.ravel()
        j_values = j_grid.ravel() 
        print_var('Coord_to_index: build: x_points',x_points)
        print_var('Coord_to_index: build: i_values',i_values)
        print_var('Coord_to_index: build: j_values',j_values)
        self.i_interp = Interpolator(degree=self.degree)
        self.j_interp = Interpolator(degree=self.degree)
        self.i_interp.build(x_points,i_values)
        self.j_interp.build(x_points,j_values)
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid

    def evaluate(self, lon_array, lat_array):
        # Stack the raveled lon and lat arrays into a 2D array of coordinate pairs
        x_points = np.column_stack((lon_array.ravel(), lat_array.ravel()))
        print_var('Coord_to_index: evaluate: lon_array',lon_array)
        print_var('Coord_to_index: evaluate: lat_array',lat_array)
        print_var('Coord_to_index: evaluate: x_points',x_points)
        
        # Evaluate interpolated indices
        i_flat = self.i_interp.evaluate(x_points)
        j_flat = self.j_interp.evaluate(x_points)
        print_var('Coord_to_index: evaluate: i_flat',i_flat)
        print_var('Coord_to_index: evaluate: j_flat',j_flat)

        # Reshape the flat results back to the original grid shape        
        i_ret = i_flat.reshape(lon_array.shape)
        j_ret = j_flat.reshape(lat_array.shape)
        print_var('Coord_to_index: evaluate: i_ret',i_ret)
        print_var('Coord_to_index: evaluate: j_ret',j_ret)

        return i_ret, j_ret

    def err_reproduce(self,graphics=False):
        # compute the reproducing error of Coord_to_index interpolator
    
        imax, jmax = self.lon_grid.shape
        
        # Evaluate on the same points
        i_approx, j_approx = self.evaluate(self.lon_grid, self.lat_grid)
    
        # Corrected i_grid and j_grid generation to match (imax, jmax)
        i_grid, j_grid = np.meshgrid(np.arange(imax), np.arange(jmax), indexing='ij')


        # Calculate and print error metrics on the test data
        print("Error metrics for i_approx vs. i_grid:")
        imrs, iamax = error_metrics(i_approx,i_grid)
        print_var('i_approx',i_approx)
        print_var('i_grid',i_grid)
        print("Error metrics for j_approx vs. j_grid:")
        jmrs, jamax = error_metrics(j_approx,j_grid)
        print_var('j_approx',j_approx)
        print_var('j_grid',j_grid)

        if graphics:
            d = np.sqrt((i_approx-i_grid)**2 + (j_approx-j_grid)**2)
            plt.figure(figsize=(10, 6))
            color_map = plt.pcolormesh(self.lon_grid, self.lat_grid, d, shading='auto', cmap='viridis')
            plt.colorbar(color_map, label='Difference')  # Add color bar with label
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Difference between original and approximated indices (i,j)')
            plt.show()
            plt.pause(0.1)
        
        return max(imrs,jmrs), max(iamax,jamax)

def test_reproduce_smooth_grid(degree = 6, xsize=1, ysize=1, imax = 10, jmax=12, deformation=0.05):
    print('Test reproducing a smoothly deformed grid, degree',degree,'xsize',xsize,'ysize',ysize,
          'imax',imax,'jmax',jmax,'deformation',deformation)
    
    # Generate a test grid of x_points using the provided function
    lon_grid, lat_grid = generate_test_grid(imax, jmax, xsize=xsize, ysize=ysize, deformation=deformation)
    plot_lon_lat_grid(lon_grid, lat_grid)
    print_var('lat_grid', lat_grid)
    print_var('lon_grid', lon_grid)

    # Create interpolator
    interp = Coord_to_index(degree)
    interp.build(lon_grid, lat_grid)

    # compute the interpolator reproducing error
    mrs, amax = interp.err_reproduce()

if __name__ == "__main__":

    do_print_var = False
    
    test_reproduce_smooth_values( degree=5, xsize = 2, ysize = 5, imax = 10, jmax=12, deformation=0.05)

    test_reproduce_smooth_grid( degree=4, xsize = 1, ysize = 2, imax = 30, jmax=20, deformation=0.05)
    
    input('Press Enter to continue...')
