from imports import *


def normalize_tiff(tiff):
    '''
    Normalizes a TIFF image based on percentile scaling to enhance contrast.

    Parameters:
        tiff (numpy.ndarray): The input TIFF image as a NumPy array, typically 2D or 3D.

    Returns:
        numpy.ndarray: The normalized image with intensity values scaled between 0 and 1 
        using the 1st and 99.8th percentiles.

    Notes:
        - This is useful for preparing images for intensity-based measurements or visualization.
        - Percentile normalization reduces the impact of extreme outlier values.
        - Normalization is applied along the spatial axes (axis=(0, 1)).
    '''

    img_norm = normalize(tiff, pmin=1, pmax=99.8, axis=(0, 1))


    return img_norm



def measure_blob_intensity(image, center, radius, print_info=False):
    """
    Measures mean intensity inside a circular mask in a single image.

    Parameters:
        image (2D ndarray): Grayscale cropped image.
        center (tuple): (x, y) center of the circle in pixel coordinates.
        radius (float): Radius of the circular mask.

    Returns:
        float: Mean intensity inside the circle, or None if center/radius invalid.
    """
    
    if print_info:
        print("center shape:", center.shape)
        print("center:",center)
        print("radius shape:",radius.shape)
        print("radius:",radius)


    if center is None or center.shape != (1, 2):
        return None

    h, w = image.shape
    cx, cy = center[0]
    r = radius[0]

    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r**2

    values = image[mask]
    return np.mean(values) if values.size > 0 else None


def measure_all_blob_intensities(img_boxes, all_coordinates, all_radii):

    intensities = [
        measure_blob_intensity(image, center, radius)
        for image, center, radius in zip(img_boxes, all_coordinates, all_radii)
    ]

    return intensities


