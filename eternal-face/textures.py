"""
textures.py
Texture synthesis functions based on Efros & Freeman (2001).
"""
import math
import random

import cv2 
import numpy as np

import boundary as meb

def generate_patch(source, window):
    """
    Selects random seed patch from filled in region of image.

    Args:
        source (np.array): Image to subset for patch.
        window (int): Width and height of patch.

    Returns:
        patch (np.array): window x window subset of source.
    """
    k = (window - 1) / 2
    init_row = random.randint(k, source.shape[0] - k - 1)
    init_col = random.randint(k, source.shape[1] - k - 1)
    patch = source[init_row-k:init_row+k+1, init_col-k:init_col+k+1]

    return patch

def find_matches(source, up, left, window):
    """
    Scans an image in raster order, determining L2 Euclidean distance between
    each possible patch of specified window size ("candidates") and pair of up/left neighbors.

    Args:
        source (np.array): Image to search for matches.
        up (np.array): Neighbor patch directly above candidate patch for comparison.
            If None, will only compare with left patch.
        left (np.array): Neighbor patch directly to the left of candidate patch for comparison.
            If None, will only compare with top patch.
        window (int): Width and height of patch.

    Returns:
        candidates (np.array): Pixel values of each patch candidate.
        errors (np.array): Up and left errors for each patch candidate.
    """
    # Initialize upper and lower pixel bounds for patch search based on window size
    # and source image dimensions.
    k = (window - 1) / 2
    upper_row, upper_col = source.shape[0] - k - 1, source.shape[1] - k - 1
    if upper_row == k:
        upper_row += 1
    if upper_col == k:
        upper_col += 1

    # Initialize empty array storing pixel values for each possible patch.
    candidates = np.zeros(((upper_row - k) * (upper_col - k), window, window, 3))
    # Intiialize empty array storing errors compared to top and left neighbors.
    errors = np.zeros(((upper_row - k) * (upper_col - k), window*2, window/6))

    # Loop through every possible patch of specified window size.
    for row in range(k, upper_row):
        for col in range(k, upper_col):
            patch = source[row-k:row+k+1, col-k:col+k+1]
            idx = (row - k) * (upper_col - k) + col - k
            candidates[idx] = patch

            # Calculates overlap error between current patch and relevant neighbors.
            if up is not None:
                errors[idx, :window, :] = meb.overlap_error(np.rot90(up), np.rot90(patch), window/6)
            if left is not None:
                errors[idx, window:, :] = meb.overlap_error(left, patch, window/6)

    return candidates, errors

def good_match(candidates, errors, correspondence=None, alpha=0.5, min_err=0.05):
    """
    Returns random patch candidate within min_err range of the minimum total distance.

    Args:
        candidates (np.array): Pixel values of each patch candidate.
        errors (np.array): Up and left errors for each patch candidate.
        window (int): Width and height of patch.
        min_error (float): Acceptable error tolerance threshold.

    Returns:
        candidates[match_index] (np.array): Subset of source image selected as
            acceptable match for region based on neighbor values.
        errors (np.array): Errors in overlap region with neighbor(s).
    """
    # Calculates total error for each candidate.
    tot_errors = np.sum(errors, axis=(1, 2))

    # If correspondence map is provided, take squared pixel differences
    # and include in total error assessment.
    if correspondence is not None:
        intensities = np.average(candidates, axis=3, weights=[0.114, 0.587, 0.299])
        corr_errors = np.power(intensities - correspondence, 2)
        tot_errors = alpha * tot_errors + (1 - alpha) * np.sum(corr_errors, axis=(1, 2))

    # Randomly selects a candidate within error tolerance
    match_values = np.where(tot_errors <= np.min(tot_errors) * (1 + min_err))
    match_index = match_values[0][random.randint(0, match_values[0].shape[0] - 1)]

    return candidates[match_index], errors[match_index]


def fill_patch(source, dest, window, coord, correspondence=None):
    """
    Finds acceptable patch and stitches with neighbor(s) along minimum error boundary.

    Args:
        source (np.array): Image to scan for patches.
        dest (np.array): Working image containing current progress and neighbors.
        window (int): Window size of texture patches.
        directions (tuple): Booleans for whether up and left neighbors exist, respectively.
        coord (tuple): Row, col coordinates of center of patch in destination image.

    Returns:
        stitched (np.array): Image of window size with overlap regions stitched
            along minimum error boundary with neighbor(s).
    """
    k = (window - 1) / 2

    # If specified, return neighbor above and to the left of active patch region.
    up_n, left_n = None, None
    if coord[0] > k:
        up_n = dest[coord[0] - (window - window/6) - k:coord[0] - (window - window/6) + k + 1,
                    coord[1] - k:coord[1] + k + 1]
    if coord[1] > k:
        left_n = dest[coord[0] - k:coord[0] + k + 1,
                      coord[1] - (window - window/6) - k:coord[1] - (window - window/6) + k + 1]
    if correspondence is not None:
        correspondence = correspondence[coord[0] - k:coord[0] + k + 1,
                                        coord[1] - k:coord[1] + k + 1]

    # Scan source image and find a patch that satisfies error conditions.
    candidates, errors = find_matches(source, up_n, left_n, window)
    stitched, patch_errors = good_match(candidates, errors, correspondence)

    # For each neighbor if specified, calculate minimum error boundary between acceptable
    # patch and neighbor, then stitch together patch with each neighbor along boundary.
    if coord[0] > k:
        h_boundary = meb.minimum_error_boundary(patch_errors[:window])
        stitched = meb.stitch_images(up_n, stitched,
                                     h_boundary, window/6, 'horizontal')
    if coord[1] > k:
        v_boundary = meb.minimum_error_boundary(patch_errors[window:])
        stitched = meb.stitch_images(left_n, stitched,
                                     v_boundary, window/6, 'vertical')

    return stitched

def fill_image(source, dims, correspondence=None, window=25):
    """
    Fills in image using texture algorithm from Efros & Freeman (2001).

    Args:
        source (np.array): Image to derive texture from.
        dims (tuple): Integer dimensions of desired output.
        window (int): Window size of texture patches.

    Returns:
        filled[:dims[0] - pad_row, :dims[1] - pad_col] (np.array):
            Image fully filled with texture.
    """
    # Initialize empty array of appropriate size, padding for additional rows/cols
    # if desired dimensions are not fully divisible by window size.
    pad_row = (dims[0] / (window - window/6) + 1) * (window - window/6) + window/6 - dims[0]
    pad_col = (dims[1] / (window - window/6) + 1) * (window - window/6) + window/6 - dims[1]
    filled = np.zeros((dims[0] + pad_row, dims[1] + pad_col, dims[2]))
    filled[:window, :window] = generate_patch(source, window)

    # Fill each patch with an acceptable candidate.
    k = (window - 1) / 2
    for row in range(k, dims[0] + 1, window - window/6):
        for col in range(k, dims[1] + 1, window - window/6):
            if row > k or col > k:
                stitched = fill_patch(source, filled, window, (row, col), correspondence)
                filled[row - k:row + k + 1, col - k:col + k + 1] = stitched

    # Return filled image with padding removed.
    return filled[:dims[0] - pad_row, :dims[1] - pad_col]
