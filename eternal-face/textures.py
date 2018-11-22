"""
textures.py
Texture synthesis functions.
"""

import math
import random

import cv2 
import numpy as np

import boundary as meb


def generate_patch(source, window):
    """
    Selects random seed patch from filled in region of image.
    """
    k = (window - 1) / 2
    init_row = random.randint(k, source.shape[0] - k - 1)
    init_col = random.randint(k, source.shape[1] - k - 1)
    patch = source[init_row-k:init_row+k+1, init_col-k:init_col+k+1]

    return patch


def find_matches(source, up, left, window, min_err=0.05):
    """
    Returns random texture patch from source image within tolerance threshold
    of L2 overlap error with region to be extended.
    """
    tot_rows, tot_cols = source.shape[:2]
    k = (window - 1) / 2
    upper_row, upper_col = tot_rows - k - 1, tot_cols - k - 1
    if upper_row == k:
        upper_row += 1
    if upper_col == k:
        upper_col += 1
    candidates = np.zeros(((upper_row - k) * (upper_col - k), window, window, 3))
    errors = np.zeros(((upper_row - k) * (upper_col - k), window*2, window/6))
    for row in range(k, upper_row):
        for col in range(k, upper_col):
            patch = source[row-k:row+k+1, col-k:col+k+1]
            idx = (row - k) * (upper_col - k) + col - k
            candidates[idx] = patch

            if up is not None:
                up, rot_patch = np.rot90(up), np.rot90(patch)
                errors[idx, :window, :] = meb.overlap_error(up, rot_patch, window/6)
            if left is not None:
                errors[idx, window:, :] = meb.overlap_error(left, patch, window/6)

    tot_errors = np.sum(errors, axis=(1, 2))
    err_threshold = np.min(tot_errors) * (1 + min_err)
    match_values = np.where(tot_errors <= err_threshold)
    if match_values[0].shape > 1:
        match_index = match_values[0][random.randint(0, match_values[0].shape[0] - 1)]
    else:
        match_index = match_values[0][0]

    return candidates[match_index], errors[match_index, :window, :], errors[match_index, window:, :]

def fill_image(source, dims, window=25):
    """
    Fills in image using texture algorithm.
    """

    pad_row, pad_col = dims[0] % window, dims[1] % window
    filled = np.zeros((dims[0] + pad_row, dims[1] + pad_col, dims[2]))
    filled[:window, :window] = generate_patch(source, window)

    k = (window - 1) / 2
    for iter_row in range(k, dims[0] + 1, window):
        for iter_col in range(k, dims[1] + 1, window):
            row, col = iter_row, iter_col
            if row > k:
                row -= window/6 * (row/window)
            if col > k:
                col -= window/6 * (col/window)
            up_neighbor = filled[row - (window - window/6) - k:row - (window - window/6) + k + 1,
                                 col - k:col + k + 1]
            left_neighbor = filled[row - k:row + k + 1,
                                   col - (window - window/6) - k:col - (window - window/6) + k + 1]
            if iter_row == k:
                up_neighbor = None
            if iter_col == k:
                left_neighbor = None
            if iter_row > k or iter_col > k:
                best_patch, h_error, v_error = find_matches(source, up_neighbor,
                                                            left_neighbor, window)
                stitched = best_patch.copy()
                if iter_row > k:
                    h_boundary = meb.minimum_error_boundary(h_error)
                    stitched = meb.stitch_images(up_neighbor, stitched,
                                                 h_boundary, window/6, 'horizontal')
                if iter_col > k:
                    v_boundary = meb.minimum_error_boundary(v_error)
                    stitched = meb.stitch_images(left_neighbor, stitched,
                                                 v_boundary, window/6, 'vertical')
                left, right, top, bottom = col - k, col + k + 1, row - k, row + k + 1
                filled[top:bottom, left:right] = stitched
                cv2.imshow('filled', np.uint8(filled))
                cv2.waitKey(25)

    return filled[:dims[0] - pad_row, :dims[1] - pad_col]
