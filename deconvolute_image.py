"""
Signal Strength Analysis and Visualization

This module processes signal strength data collected across azimuth and altitude coordinates.
It creates visualizations, performs interpolation, fits an Airy pattern, and applies
deconvolution to enhance signal resolution.

Usage as a module:
    from signal_processor import process_signal_data
    
    # Process from file
    results = process_signal_data(file_path='sample_data.tsv', deconv_iterations=10, roi_width=15)
    
    # Or process from arrays
    results = process_signal_data(
        data_arrays=(azimuth_array, altitude_array, signal_array),
        deconv_iterations=10, 
        roi_width=15
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import j1
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy
from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator
import os

def load_psf_parameters(file_path):
    """
    Load PSF parameters from a text file.
    
    Parameters:
    -----------
    file_path : str
        Path to the saved PSF parameters file
        
    Returns:
    --------
    dict or float
        Either a dictionary containing all parameters or just the FWHM value
    """
    try:
        # Load the data
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Check if it's a simple file with just one value (old format)
        if len(lines) == 1:
            try:
                fwhm = float(lines[0].strip())
                print(f"Loaded simple Airy disk diameter: FWHM = {fwhm:.6f} degrees")
                return fwhm
            except ValueError:
                pass  # Not a simple float, continue to complex parsing
        
        # Parse the more complex format with multiple parameters
        params = {}
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            # Try to extract parameter name and value
            parts = line.split(':')
            param_name = parts[0].strip().lower()
            param_value_str = ':'.join(parts[1:]).strip()
            
            # Try to convert to float if possible
            try:
                # Handle special cases first
                if "center" in param_name:
                    # Parse the center coordinates from format like "(174.59, 43.09) degrees"
                    coords_str = param_value_str.split('degrees')[0].strip()
                    coords_str = coords_str.strip('()')
                    az_str, alt_str = coords_str.split(',')
                    params['center_az'] = float(az_str.strip())
                    params['center_alt'] = float(alt_str.strip())
                elif "kernel shape" in param_name:
                    # Parse kernel shape from format like "(123, 123)"
                    shape_str = param_value_str.strip('()')
                    shape_parts = shape_str.split(',')
                    params['kernel_shape'] = tuple(int(p.strip()) for p in shape_parts)
                elif "radius" in param_name:
                    params['radius'] = float(param_value_str.split()[0])
                elif "amplitude" in param_name:
                    params['amplitude'] = float(param_value_str)
                elif "background" in param_name:
                    params['background'] = float(param_value_str)
                else:
                    # Try generic float conversion
                    try:
                        params[param_name] = float(param_value_str)
                    except ValueError:
                        # If not a float, store as string
                        params[param_name] = param_value_str
            except ValueError:
                # If conversion fails, store the raw string
                params[param_name] = param_value_str
        
        # Check if we successfully parsed any parameters
        if params:
            print(f"Loaded PSF parameters from file:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            # Return all parsed parameters
            return params
        else:
            print(f"Warning: Could not parse any parameters from {file_path}")
            return None
            
    except Exception as e:
        print(f"Error loading PSF parameters: {str(e)}")
        return None
    
def load_data_from_file(filename):
    """Load and parse the signal strength data file."""
    data = np.loadtxt(filename, delimiter=' ')
    azimuth = data[:, 0]
    altitude = data[:, 1]
    signal_strength = data[:, 2]
    
    # Get unique coordinate values
    unique_az = np.sort(np.unique(azimuth))
    unique_alt = np.sort(np.unique(altitude))
    
    return azimuth, altitude, signal_strength, unique_az, unique_alt


def create_signal_grid(azimuth, altitude, signal_strength, unique_az, unique_alt):
    """Create a 2D grid from the scattered signal strength measurements."""
    grid = np.full((len(unique_az), len(unique_alt)), np.nan)
    
    # Fill the grid with signal strength values
    for i, (az, alt, sig) in enumerate(zip(azimuth, altitude, signal_strength)):
        az_idx = np.where(unique_az == az)[0][0]
        alt_idx = np.where(unique_alt == alt)[0][0]
        grid[az_idx, alt_idx] = sig
    
    return grid


def fill_missing_values(grid):
    """Fill any NaN values in the grid using nearest neighbor interpolation."""
    # Convert NaN to masked array
    grid_masked = np.ma.masked_invalid(grid)
    
    # Check if there are any NaN values
    if np.ma.is_masked(grid_masked):
        print("Filling in NaN values with nearest neighbor interpolation")
        
        # Fill missing values with nearest valid value
        xx, yy = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))
        valid_mask = ~np.ma.getmaskarray(grid_masked)
        coords = np.array(np.nonzero(valid_mask)).T
        values = grid[valid_mask]
        
        interp = NearestNDInterpolator(coords, values)
        grid_filled = interp(np.vstack((xx.flatten(), yy.flatten())).T).reshape(grid.shape)
        return grid_filled
    else:
        print("No NaN values found in the grid")
        return grid


def plot_signal_map(grid, extent, title, cbar_label, filename, cmap='CMRmap', save_dir=None):
    """Create and save a visualization of the signal strength map."""
    plt.figure(figsize=(12, 10))
    im = plt.imshow(
        grid.T,  # Transpose to match original orientation (azimuth on x-axis, altitude on y-axis)
        extent=extent,
        origin='lower',
        cmap=cmap,
        aspect='equal',
        interpolation='nearest'
    )
    plt.colorbar(im, label=cbar_label)
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Altitude (degrees)')
    plt.title(title)
    plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
    else:
        filepath = filename
        
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    return im


def interpolate_grid(grid, unique_az, unique_alt, interp_factor=3, margin=0.5):
    """Create a higher resolution interpolated grid."""
    az_min, az_max = min(unique_az), max(unique_az)
    alt_min, alt_max = min(unique_alt), max(unique_alt)
    
    # Create new high-resolution coordinates with margin
    interp_az = np.linspace(az_min - margin, az_max + margin, int(len(unique_az) * interp_factor))
    interp_alt = np.linspace(alt_min - margin, alt_max + margin, int(len(unique_alt) * interp_factor))
    
    # Create the interpolation function using cubic spline
    interp_func = RectBivariateSpline(unique_az, unique_alt, grid, kx=3, ky=3)
    
    # Generate the interpolated grid
    interp_grid = interp_func(interp_az, interp_alt)
    
    return interp_grid, interp_az, interp_alt


def airy_pattern(xy_array, amplitude, x0, y0, radius, background):
    """
    2D radially symmetric Airy pattern (jinc function)
    
    Parameters:
    xy_array: Array of shape (n, 2) containing x,y coordinates
    amplitude: Peak amplitude
    x0, y0: Center position
    radius: Characteristic radius (related to telescope aperture)
    background: Background level
    
    Returns:
    Array of function values
    """
    x = xy_array[:, 0]
    y = xy_array[:, 1]
    
    # Calculate radial distance from center
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    
    # Handle the case where r = 0
    mask_zero = (r < 1e-10)
    r[mask_zero] = 1e-10
    
    # Jinc function = J₁(x)/x where J₁ is the first order Bessel function of the first kind
    arg = np.pi * r / radius
    result = amplitude * (2 * j1(arg) / arg)**2 + background
    
    # At r = 0, the value should be amplitude + background
    result[mask_zero] = amplitude + background
    
    return result


def find_signal_maximum(interp_grid, interp_az, interp_alt, search_bounds=None):
    """Find the location of maximum signal strength within specified bounds."""
    if search_bounds is not None:
        search_az_min, search_az_max, search_alt_min, search_alt_max = search_bounds
        
        # Get indices for search region
        search_az_indices = np.where((interp_az >= search_az_min) & (interp_az <= search_az_max))[0]
        search_alt_indices = np.where((interp_alt >= search_alt_min) & (interp_alt <= search_alt_max))[0]
        
        # Extract search region from interpolated grid
        if len(search_az_indices) > 0 and len(search_alt_indices) > 0:
            search_grid = interp_grid[np.ix_(search_az_indices, search_alt_indices)]
            search_az = interp_az[search_az_indices]
            search_alt = interp_alt[search_alt_indices]
            
            # Find maximum within the search region
            max_idx_search = np.unravel_index(np.nanargmax(search_grid), search_grid.shape)
            center_az = search_az[max_idx_search[0]]
            center_alt = search_alt[max_idx_search[1]]
            max_signal = np.nanmax(search_grid)
            
            print(f"Maximum found within search range at: Az={center_az:.2f}°, Alt={center_alt:.2f}°, Value={max_signal:.2f}")
            return center_az, center_alt, max_signal
    
    # If no search bounds provided or search failed, use global maximum
    max_idx = np.unravel_index(np.nanargmax(interp_grid), interp_grid.shape)
    center_az = interp_az[max_idx[0]]
    center_alt = interp_alt[max_idx[1]]
    max_signal = np.nanmax(interp_grid)
    
    print(f"Global maximum at: Az={center_az:.2f}°, Alt={center_alt:.2f}°, Value={max_signal:.2f}")
    return center_az, center_alt, max_signal


def extract_roi(interp_grid, interp_az, interp_alt, center_az, center_alt, roi_width=15):
    """Extract a region of interest (ROI) around a center point."""
    roi_az_min = center_az - roi_width
    roi_az_max = center_az + roi_width
    roi_alt_min = center_alt - roi_width
    roi_alt_max = center_alt + roi_width
    
    # Get indices for ROI on the interpolated grid
    roi_az_indices = np.where((interp_az >= roi_az_min) & (interp_az <= roi_az_max))[0]
    roi_alt_indices = np.where((interp_alt >= roi_alt_min) & (interp_alt <= roi_alt_max))[0]
    
    # Extract ROI from interpolated grid
    if len(roi_az_indices) > 0 and len(roi_alt_indices) > 0:
        roi_grid = interp_grid[np.ix_(roi_az_indices, roi_alt_indices)]
        roi_az = interp_az[roi_az_indices]
        roi_alt = interp_alt[roi_alt_indices]
        return roi_grid, roi_az, roi_alt, (roi_az_min, roi_az_max, roi_alt_min, roi_alt_max)
    else:
        print("ROI selection failed. Using full interpolated grid.")
        return interp_grid, interp_az, interp_alt, None


def fit_airy_pattern(roi_grid, roi_az, roi_alt, center_az, center_alt):
    """Fit an Airy pattern to the region of interest."""
    # Prepare data for fitting
    X, Y = np.meshgrid(roi_az, roi_alt)
    xy_data = np.column_stack([X.flatten(), Y.flatten()])
    z_data = roi_grid.T.flatten()
    
    # Remove NaN values before fitting
    valid_indices = ~np.isnan(z_data)
    xy_data_valid = xy_data[valid_indices]
    z_data_valid = z_data[valid_indices]
    
    # Initial parameter guesses
    max_signal = np.nanmax(roi_grid)
    background = np.nanmin(roi_grid)
    amplitude_guess = max_signal - background
    x0_guess = center_az
    y0_guess = center_alt
    radius_guess = 3.0  # initial guess for characteristic radius in degrees
    
    initial_guess = [amplitude_guess, x0_guess, y0_guess, radius_guess, background]
    
    # Set bounds for parameters
    roi_bounds = [min(roi_az), max(roi_az), min(roi_alt), max(roi_alt)]
    lower_bounds = [0, roi_bounds[0], roi_bounds[2], 0.1, 0]
    upper_bounds = [amplitude_guess*2, roi_bounds[1], roi_bounds[3], 20, max_signal]
    
    try:
        # Fit the 2D Airy pattern to the data
        params, covariance = curve_fit(
            airy_pattern, 
            xy_data_valid, 
            z_data_valid, 
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000  # Increase maximum number of function evaluations
        )
        
        # Extract fitted parameters
        amplitude, x0, y0, radius, background = params
        
        # Calculate errors from the diagonal of the covariance matrix
        errors = np.sqrt(np.diag(covariance))
        
        # Calculate FWHM (Full Width at Half Maximum) from radius
        # For an Airy disk, FWHM ≈ 1.03 * radius
        fwhm = 1.03 * radius
        
        # Print results
        print("\nAiry Pattern Fit Results:")
        print(f"Amplitude: {amplitude:.2f} ± {errors[0]:.2f}")
        print(f"Center (Az, Alt): ({x0:.2f} ± {errors[1]:.2f}, {y0:.2f} ± {errors[2]:.2f}) degrees")
        print(f"Characteristic radius: {radius:.2f} ± {errors[3]:.2f} degrees")
        print(f"Background: {background:.2f} ± {errors[4]:.2f}")
        print(f"FWHM: {fwhm:.2f} degrees")
        print(f"Angular resolution: {fwhm:.2f} degrees")
        
        return params, errors, fwhm
        
    except Exception as e:
        print(f"Fitting failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def create_psf_kernel(radius, interp_factor):
    """
    Create a Point Spread Function kernel based on the Airy pattern parameters.
    
    Returns:
    --------
    ndarray
        Normalized PSF kernel
    """
    # Create a grid for the PSF kernel (odd dimensions work better for convolution)
    kernel_size = int(np.ceil(6 * radius * interp_factor))  # Scaled for interpolated resolution
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    
    # Create coordinate grids for the kernel
    x_kernel = np.linspace(-kernel_size//2, kernel_size//2, kernel_size) / interp_factor
    y_kernel = np.linspace(-kernel_size//2, kernel_size//2, kernel_size) / interp_factor
    X_kernel, Y_kernel = np.meshgrid(x_kernel, y_kernel)
    
    # Calculate distances from center
    R_kernel = np.sqrt(X_kernel**2 + Y_kernel**2)
    
    # Create the Airy pattern kernel (without amplitude or background)
    eps = 1e-10  # Avoid division by zero
    arg_kernel = np.pi * R_kernel / radius
    psf_kernel = np.ones_like(arg_kernel)
    
    # Calculate J1(x)/x for all non-zero values
    non_zero = arg_kernel > eps
    psf_kernel[non_zero] = (2 * j1(arg_kernel[non_zero]) / arg_kernel[non_zero])**2
    
    # Normalize the kernel to sum to 1 (for proper deconvolution)
    psf_kernel = psf_kernel / np.sum(psf_kernel)
    
    # Print kernel info
    print(f"Created PSF kernel with shape {psf_kernel.shape} and sum {np.sum(psf_kernel):.10f}")
    print(f"Kernel center value: {psf_kernel[kernel_size//2, kernel_size//2]:.6f}")
    
    return psf_kernel


def perform_deconvolution(interp_grid, psf_kernel, background, num_iterations=10):
    """
    Perform Richardson-Lucy deconvolution to enhance signal resolution.
    
    Parameters:
    -----------
    interp_grid : ndarray
        The interpolated grid to deconvolve
    psf_kernel : ndarray
        The Point Spread Function kernel
    background : float
        Background level to subtract before deconvolution
    num_iterations : int, default=10
        Number of Richardson-Lucy iterations
        
    Returns:
    --------
    ndarray
        Deconvolved image
    """
    # Ensure PSF is normalized
    psf_kernel = psf_kernel / np.sum(psf_kernel)
    
    # Check if kernel has odd dimensions (required for proper boundary conditions)
    for i, dim in enumerate(psf_kernel.shape):
        if dim % 2 == 0:
            print(f"Warning: PSF kernel dimension {i} is even ({dim}). Padding to odd dimension.")
            pad_width = [(0, 0), (0, 0)]
            pad_width[i] = (0, 1)
            psf_kernel = np.pad(psf_kernel, pad_width, mode='constant')
            print(f"New kernel shape: {psf_kernel.shape}")
            # Re-normalize after padding
            psf_kernel = psf_kernel / np.sum(psf_kernel)
    
    # Prepare the image for deconvolution by removing background and handling edges
    image_for_deconv = np.copy(interp_grid.T)  # Need to transpose to match original code
    image_for_deconv[np.isnan(image_for_deconv)] = background
    
    # Subtract background (improves deconvolution performance)
    image_detrended = image_for_deconv - background
    
    # Apply a small amount of smoothing to reduce noise amplification
    image_smoothed = gaussian_filter(image_detrended, sigma=0.5)
    
    # Ensure non-negativity (important for Richardson-Lucy)
    image_smoothed = np.maximum(image_smoothed, 0)
    
    # Richardson-Lucy deconvolution
    deconvolved = richardson_lucy(
        image_smoothed, 
        psf_kernel, 
        num_iter=num_iterations, 
        clip=False
    )
    
    # Add the background back
    deconvolved += background
    
    # Ensure non-negativity (physical constraint)
    deconvolved = np.maximum(deconvolved, 0)
    
    return deconvolved  # Returns with altitude as rows, azimuth as columns


def load_psf_parameters(file_path):
    """
    Load PSF parameters from a text file.
    
    Parameters:
    -----------
    file_path : str
        Path to the saved PSF parameters file
        
    Returns:
    --------
    dict or float
        Either a dictionary containing all parameters or just the FWHM value
    """
    try:
        # Load the data
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Check if it's a simple file with just one value (old format)
        if len(lines) == 1:
            try:
                fwhm = float(lines[0].strip())
                print(f"Loaded simple Airy disk diameter: FWHM = {fwhm:.6f} degrees")
                return fwhm
            except ValueError:
                pass  # Not a simple float, continue to complex parsing
        
        # Parse the more complex format with multiple parameters
        params = {}
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            # Try to extract parameter name and value
            parts = line.split(':')
            param_name = parts[0].strip().lower()
            param_value_str = ':'.join(parts[1:]).strip()
            
            # Try to convert to float if possible
            try:
                # Handle special cases first
                if "center" in param_name:
                    # Parse the center coordinates from format like "(174.59, 43.09) degrees"
                    coords_str = param_value_str.split('degrees')[0].strip()
                    coords_str = coords_str.strip('()')
                    az_str, alt_str = coords_str.split(',')
                    params['center_az'] = float(az_str.strip())
                    params['center_alt'] = float(alt_str.strip())
                elif "kernel shape" in param_name:
                    # Parse kernel shape from format like "(123, 123)"
                    shape_str = param_value_str.strip('()')
                    shape_parts = shape_str.split(',')
                    params['kernel_shape'] = tuple(int(p.strip()) for p in shape_parts)
                elif "radius" in param_name:
                    params['radius'] = float(param_value_str.split()[0])
                elif "amplitude" in param_name:
                    params['amplitude'] = float(param_value_str)
                elif "background" in param_name:
                    params['background'] = float(param_value_str)
                else:
                    # Try generic float conversion
                    try:
                        params[param_name] = float(param_value_str)
                    except ValueError:
                        # If not a float, store as string
                        params[param_name] = param_value_str
            except ValueError:
                # If conversion fails, store the raw string
                params[param_name] = param_value_str
        
        # Check if we successfully parsed any parameters
        if params:
            print(f"Loaded PSF parameters from file:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            # Return all parsed parameters
            return params
        else:
            print(f"Warning: Could not parse any parameters from {file_path}")
            return None
            
    except Exception as e:
        print(f"Error loading PSF parameters: {str(e)}")
        return None

def process_signal_data(file_path=None, data_arrays=None, deconv_iterations=10, 
                       roi_width=15, interp_factor=3, search_bounds=None,
                       save_plots=True, output_dir=None, verbose=True,
                       external_psf=None, export_psf=False):
    """
    Process signal strength data from a file or provided arrays.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the input file (space-separated values without headers).
        First column should be azimuth, second altitude, third signal strength.
    
    data_arrays : tuple, optional
        Tuple of (azimuth_array, altitude_array, signal_strength_array) as numpy arrays.
        If provided, file_path is ignored.
    
    deconv_iterations : int, default=10
        Number of iterations for Richardson-Lucy deconvolution.
    
    roi_width : float, default=15
        Width (in degrees) of the region of interest around the maximum.
    
    interp_factor : int, default=3
        Factor by which to increase resolution during interpolation.
    
    search_bounds : tuple, optional
        Search bounds for finding maximum signal as (az_min, az_max, alt_min, alt_max).
        If None, the global maximum is used.
    
    save_plots : bool, default=True
        Whether to save visualization plots.
    
    output_dir : str, optional
        Directory to save output files. If None, files are saved in current directory.
    
    verbose : bool, default=True
        Whether to print progress and results.
        
    external_psf : ndarray or str, optional
        External PSF kernel to use for deconvolution instead of fitting an Airy pattern.
        If str, should be a path to a PSF parameters file.
        
    export_psf : bool, default=False
        Whether to export the PSF kernel to a numpy file.
    
    Returns:
    --------
    dict
        Dictionary containing results including:
        - 'grid': Original signal strength grid
        - 'interp_grid': Interpolated grid
        - 'deconvolved': Deconvolved grid (if successful)
        - 'params': Fitted Airy pattern parameters (if successful)
        - 'errors': Parameter fitting errors (if successful)
        - 'center': (center_az, center_alt) coordinates of maximum
        - 'coordinate_data': All coordinate information
        - 'psf_kernel': The PSF kernel used for deconvolution
    """
    # Create results dictionary
    results = {
        'success': False,
        'grid': None,
        'interp_grid': None,
        'deconvolved': None,
        'params': None,
        'errors': None,
        'center': None,
        'coordinate_data': None,
        'psf_kernel': None
    }
    
    try:
        # Set printing based on verbose flag
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
                
        # Load data
        if data_arrays is not None:
            # Unpack data arrays
            azimuth, altitude, signal_strength = data_arrays
            # Get unique coordinate values
            unique_az = np.sort(np.unique(azimuth))
            unique_alt = np.sort(np.unique(altitude))
            vprint("Using provided data arrays")
        elif file_path is not None:
            # Load from file
            azimuth, altitude, signal_strength, unique_az, unique_alt = load_data_from_file(file_path)
            vprint(f"Loaded data from: {file_path}")
        else:
            raise ValueError("Either file_path or data_arrays must be provided")
            
        # Create signal grid
        grid = create_signal_grid(azimuth, altitude, signal_strength, unique_az, unique_alt)
        vprint(f"Created grid with shape: {grid.shape}")
        
        # Fill missing values
        grid = fill_missing_values(grid)
        results['grid'] = grid
            
        # Plot original data if requested
        if save_plots:
            extent_orig = [min(unique_az)-0.5, max(unique_az)+0.5, min(unique_alt)-0.5, max(unique_alt)+0.5]
            plot_signal_map(
                grid,
                extent_orig,
                'Original Signal Strength Map',
                'Signal Strength',
                'original_signal_strength.png',
                save_dir=output_dir
            )
            vprint("Saved original signal strength map")
        
        # Create interpolated grid
        interp_grid, interp_az, interp_alt = interpolate_grid(
            grid,
            unique_az,
            unique_alt,
            interp_factor
        )
        vprint(f"Created interpolated grid with shape: {interp_grid.shape}")
        results['interp_grid'] = interp_grid
        
        # Save coordinate data for further use
        results['coordinate_data'] = {
            'unique_az': unique_az, 
            'unique_alt': unique_alt,
            'interp_az': interp_az,
            'interp_alt': interp_alt
        }
            
        # Plot interpolated data if requested
        if save_plots:
            extent_interp = [min(interp_az), max(interp_az), min(interp_alt), max(interp_alt)]
            plot_signal_map(
                interp_grid,
                extent_interp,
                f'Interpolated Signal Strength Map ({interp_factor}x)',
                'Signal Stregth',
                'interpolated_signal_strength.png',
                save_dir=output_dir
            )
            vprint("Saved interpolated signal strength map")
            
        # Find maximum signal
        center_az, center_alt, max_signal = find_signal_maximum(
            interp_grid,
            interp_az,
            interp_alt,
            search_bounds
        )
        results['center'] = (center_az, center_alt)
        
        # Extract region of interest
        roi_grid, roi_az, roi_alt, roi_bounds = extract_roi(
            interp_grid,
            interp_az,
            interp_alt,
            center_az,
            center_alt,
            roi_width
        )
        vprint(f"Extracted ROI with dimensions: {roi_grid.shape}")
        
        # Process PSF kernel
        psf_kernel = None
        params = None
        errors = None
        fwhm = None
        background = None
        do_fit_airy = True  # Flag to control whether to fit the Airy pattern
        
        # If external PSF is a string (file path), try to load parameters from it
        if isinstance(external_psf, str):
            # Load parameters from file
            loaded_params = load_psf_parameters(external_psf)
            vprint(f"Loaded external PSF parameters from: {external_psf}")
            
            if loaded_params is not None:
                # Check if it's a dictionary of parameters or just FWHM value
                if isinstance(loaded_params, dict):
                    # We have a full parameter set
                    if 'radius' in loaded_params:
                        # Use radius directly if available
                        radius = loaded_params['radius']
                    elif 'fwhm' in loaded_params:
                        # Calculate radius from FWHM
                        fwhm = loaded_params['fwhm']
                        radius = fwhm / 1.03
                    else:
                        # No radius info, will need to fit
                        vprint("Warning: No radius or FWHM found in parameters file")
                        do_fit_airy = True
                        radius = None
                    
                    if 'background' in loaded_params:
                        background = loaded_params['background']
                    else:
                        # Estimate background from the minimum value in ROI
                        background = np.nanmin(roi_grid)
                        vprint(f"Using estimated background value: {background:.6f}")
                    
                    if radius is not None:
                        # Create PSF kernel from the loaded radius
                        psf_kernel = create_psf_kernel(radius, interp_factor)
                        do_fit_airy = False
                        vprint(f"Created PSF kernel from loaded parameters with radius={radius:.6f}")
                        
                        # Create synthetic params for results
                        # [amplitude, x0, y0, radius, background]
                        if 'amplitude' in loaded_params:
                            amplitude = loaded_params['amplitude']
                        else:
                            amplitude = np.nanmax(roi_grid) - background
                            
                        results['params'] = [
                            amplitude,
                            center_az,  # Use current center
                            center_alt,
                            radius,
                            background
                        ]
                
                elif isinstance(loaded_params, (int, float)):
                    # Just a FWHM value
                    fwhm = loaded_params
                    
                    # Calculate radius from FWHM
                    radius = fwhm / 1.03
                    
                    # Estimate background
                    background = np.nanmin(roi_grid)
                    vprint(f"Using estimated background value: {background:.6f}")
                    
                    # Create PSF kernel
                    psf_kernel = create_psf_kernel(radius, interp_factor)
                    do_fit_airy = False
                    vprint(f"Created PSF kernel from loaded FWHM={fwhm:.6f}, calculated radius={radius:.6f}")
                    
                    # Create synthetic params for results
                    # [amplitude, x0, y0, radius, background]
                    amplitude = np.nanmax(roi_grid) - background
                    results['params'] = [
                        amplitude,
                        center_az,  # Use current center
                        center_alt,
                        radius,
                        background
                    ]
        
        # If external PSF is a numpy array, use it directly
        elif external_psf is not None and hasattr(external_psf, 'shape'):
            psf_kernel = external_psf
            vprint("Using provided external PSF kernel array")
            
            # Estimate the background
            background = np.nanmin(roi_grid)
            vprint(f"Using estimated background value: {background:.6f}")
            
            # Skip Airy pattern fitting
            do_fit_airy = False
        
        # If we should fit the Airy pattern (no external PSF or loading failed)
        if do_fit_airy:
            vprint("Performing Airy pattern fitting on ROI...")
            
            # Fit Airy pattern
            params, errors, fwhm = fit_airy_pattern(roi_grid, roi_az, roi_alt, center_az, center_alt)
            
            if params is not None:
                # Extract parameters
                amplitude, x0, y0, radius, background = params
                
                # Create PSF kernel
                psf_kernel = create_psf_kernel(radius, interp_factor)
                
                # Update results
                results['params'] = params
                results['errors'] = errors
        
        # If we have a PSF kernel, perform deconvolution
        if psf_kernel is not None:
            # Save the PSF kernel in results
            results['psf_kernel'] = psf_kernel
            
            # Export PSF if requested
            if export_psf:
                # Save the parameters
                if output_dir is None:
                    params_file = 'psf_parameters.txt'
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    params_file = os.path.join(output_dir, 'psf_parameters.txt')
                
                with open(params_file, 'w') as f:
                    f.write("PSF Parameters:\n")
                    
                    # If we have params from fitting or synthetic
                    if results['params'] is not None:
                        params = results['params']
                        f.write(f"Amplitude: {params[0]:.6f}\n")
                        f.write(f"Center (Az, Alt): ({params[1]:.6f}, {params[2]:.6f}) degrees\n")
                        f.write(f"Characteristic radius: {params[3]:.6f} degrees\n")
                        f.write(f"Background: {params[4]:.6f}\n")
                    
                    # Always write PSF kernel info
                    if psf_kernel is not None:
                        f.write(f"PSF kernel shape: {psf_kernel.shape}\n")
                        f.write(f"PSF kernel sum: {np.sum(psf_kernel):.10f}\n")
                
                vprint(f"Saved PSF parameters to {params_file}")
            
            # Make sure we have a background value
            if background is None:
                background = 0
                vprint("Warning: No background value available, using 0")
            
            # Perform deconvolution
            deconvolved = perform_deconvolution(
                interp_grid,
                psf_kernel,
                background,
                deconv_iterations
            )
            results['deconvolved'] = deconvolved
            
            # Export deconvoluted data as space-separated text file
            if output_dir is None:
                deconv_txt_file = 'deconvoluted_data.txt'
            else:
                os.makedirs(output_dir, exist_ok=True)
                deconv_txt_file = os.path.join(output_dir, 'deconvoluted_data.txt')
            
            # Create the text file data
            interp_az = results['coordinate_data']['interp_az']
            interp_alt = results['coordinate_data']['interp_alt']
            
            # Open text file for writing
            with open(deconv_txt_file, 'w') as f:
                # Write header
                f.write("azimuth altitude signal\n")
                
                # Write data rows with correct index mapping (swapped i,j from previous version)
                for i in range(len(interp_az)):         # Azimuth dimension
                    for j in range(len(interp_alt)):    # Altitude dimension
                        az_val = interp_az[i]
                        alt_val = interp_alt[j]
                        signal_val = deconvolved[j, i]  # Using corrected indices
                        f.write(f"{az_val:.6f} {alt_val:.6f} {signal_val:.6f}\n")
            
            vprint(f"Exported deconvoluted data to {deconv_txt_file}")
            
            # Plot deconvolved data if requested
            if save_plots:
                #plot the log of the signal data as it is clearer
                deconv_log = np.log(deconvolved)

                extent_interp = [min(interp_az), max(interp_az), min(interp_alt), max(interp_alt)]
                plot_signal_map(
                    deconv_log.T,  # Transpose back to original orientation
                    extent_interp,
                    f'Deconvolved Signal Strength Map ({deconv_iterations} iterations)',
                    'Log Signal Strength',
                    'deconvolved_signal_strength.png',
                    save_dir=output_dir
                )
                vprint("Saved deconvolved signal strength map")
                
                # Plot the PSF kernel as well
                plt.figure(figsize=(8, 8))
                plt.imshow(psf_kernel, origin='lower', cmap='viridis')
                plt.colorbar(label='Intensity')
                plt.title('PSF Kernel')
                
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(os.path.join(output_dir, 'psf_kernel.png'), dpi=300)
                else:
                    plt.savefig('psf_kernel.png', dpi=300)
                plt.close()
                vprint("Saved PSF kernel visualization")
            
            results['success'] = True
            
    except Exception as e:
        vprint(f"Processing failed with error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
    
    return results


def run_from_command_line():
    """Run the signal processing from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process signal strength data.')
    parser.add_argument('file_path', help='Path to input file (space-separated values)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of deconvolution iterations')
    parser.add_argument('--roi', type=float, default=15, help='Region of interest width in degrees')
    parser.add_argument('--interp', type=int, default=3, help='Interpolation factor')
    parser.add_argument('--search', type=float, nargs=4, 
                        metavar=('AZ_MIN', 'AZ_MAX', 'ALT_MIN', 'ALT_MAX'),
                        help='Search bounds for finding maximum signal')
    parser.add_argument('--save-plots', action='store_true', help='Do not save plots')
    parser.add_argument('--output-dir', help='Directory to save output files')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    parser.add_argument('--external-psf', help='Path to external PSF parameters file (.txt)')
    parser.add_argument('--export-psf', action='store_true', help='Export the PSF parametes to a txt file')
    
    args = parser.parse_args()
    
    # Process the data
    results = process_signal_data(
        file_path=args.file_path,
        deconv_iterations=args.iterations,
        roi_width=args.roi,
        interp_factor=args.interp,
        search_bounds=args.search,
        save_plots= args.save_plots,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        external_psf=args.external_psf,
        export_psf=args.export_psf
    )
    
    # Return success status for command line
    return 0 if results['success'] else 1


# Example usage when run directly
if __name__ == "__main__":
    run_from_command_line()