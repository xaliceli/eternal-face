"""
morph.py
Morphs series of images together.
"""

import numpy as np
import cv2 

def triangle_warp(img1, tri1, tri2):
    """
    Returns vertices of triangle in bounding rectangle.
    """
    # Find rectangles bounding triangle in source and destination.
    rect1, rect2 = cv2.boundingRect(np.float32(tri1)), cv2.boundingRect(np.float32(tri2))
    tri1_crop, tri2_crop = [], []

    # Return coordinates of triangle vertices in bounding rectangle.
    for i in range(3):
        tri1_crop.append(((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri2_crop.append(((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))
    tri1_crop, tri2_crop = np.float32(tri1_crop), np.float32(tri2_crop)

    # Calculate warp matrix.
    warp_matrix = cv2.getAffineTransform(tri1_crop, tri2_crop)

    # Apply warp to image 1 cropped to bounding rectangle of triangle.
    img1_crop = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_crop = cv2.warpAffine(img1_crop, warp_matrix,
                               (rect2[2], rect2[3]), None,
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Create mask of triangle region.
    mask = np.zeros((rect2[3], rect2[2], 3), dtype='float32')
    cv2.fillConvexPoly(mask, np.int32(tri2_crop),
                       (1, 1, 1), cv2.LINE_AA)

    # Apply mask to boundary rectangle in destination image.
    img2_crop = img2_crop * mask

    # Generated warped destination image spanning entire canvas region with only triangle filled.
    img2 = np.zeros(img1.shape)
    # top, bottom, left, right = rect2[1], rect2[1]+rect2[3], rect2[0], rect2[0]+rect2[2]
    img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2_crop

    return img2
