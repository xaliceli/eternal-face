"""
boundary.py
Computes minimum error boundaries and combines images along boundary.
"""

import cv2
import numpy as np

def overlap_error(image_1, image_2, col_overlap):
    """
    Return absolute value of difference between two images where they overlap.
    """
    overlap_img1 = np.float32(image_1[:, -col_overlap:])
    overlap_img2 = np.float32(image_2[:, :col_overlap])
    error = cv2.cvtColor(abs(overlap_img1 - overlap_img2), cv2.COLOR_BGR2GRAY)
    return error

def minimum_errors(error):
    """
    Returns array containing cumulative error of minimum cost boundary
    up to each coordinate.
    """
    overlap_rows, overlap_cols = error.shape[:2]
    cumulative_error = np.empty((overlap_rows, overlap_cols, 2), dtype='object')
    cumulative_error[:, :, 0] = error.copy()

    for row in range(1, overlap_rows):
        for col in range(overlap_cols):
            if col == 0:
                subtotal = min(cumulative_error[row - 1, col : col + 2, 0])
                slice_index = np.argmin(cumulative_error[row - 1, col : col + 2, 0])
            elif col == overlap_cols - 1:
                subtotal = min(cumulative_error[row - 1, col - 1 : col + 1, 0])
                slice_index = np.argmin(cumulative_error[row - 1, col - 1 : col + 1, 0]) + col - 1
            else:
                subtotal = min(cumulative_error[row - 1, col - 1 : col + 2, 0])
                slice_index = np.argmin(cumulative_error[row - 1, col - 1 : col + 2, 0]) + col - 1
            cumulative_error[row, col, 0] += subtotal
            cumulative_error[row, col, 1] = (row - 1, slice_index)

    return cumulative_error

def minimum_error_boundary(error):
    """
    Returns coordinates of minimum error boundary across overlap region.
    """
    rows = error.shape[0]
    error_map = minimum_errors(error)
    cumulative_errors = error_map[-1, :, 0]
    least_index = np.argmin(cumulative_errors)
    coords = [(rows - 1, least_index)]

    while len(coords) < rows:
        current = coords[-1]
        parent = error_map[current[0], current[1], 1]
        coords.append(parent)

    return coords

def overlap_mask(mask_rows, mask_cols, boundary):
    """
    Returns mask for overlapping blend region along boundary.
    """
    mask = np.zeros((mask_rows, mask_cols))
    for row in range(1, len(boundary) + 1):
        boundary_col = boundary[-row][1]
        mask[row - 1, boundary_col:] = 1
    return mask

def stitch_images(image_1, image_2, boundary, overlap, direction):
    """
    Returns two images stitched into one according to mask.
    """
    if direction == 'horizontal':
        image_1, image_2 = np.rot90(image_1), np.rot90(image_2)
    stitched_img = np.zeros((image_1.shape))
    stitched_img[:, :overlap] = image_1[:, -overlap:]
    # for layer in range(image_1.shape[2]):
    for row in range(1, len(boundary) + 1):
        boundary_col = boundary[-row][1]
        stitched_img[row - 1, boundary_col:, :] = image_2[row - 1, boundary_col:, :]
    if direction == 'horizontal':
        stitched_img = np.rot90(stitched_img, 3)

    return stitched_img
