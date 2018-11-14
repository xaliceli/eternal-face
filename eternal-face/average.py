"""
average.py
Computes n choose k averages of input images.
"""

import cv2
import numpy as np
import morph
from face import Face

class AverageFace(Face):
    """
    Average of n faces.
    """

    def __init__(self, images):
        Face.__init__(self, np.zeros(images[0].shape))
        self._images = images
        self._all_features = np.zeros((68, len(images), 2))

    def features_in(self):
        """
        Calculates features for each input image.
        """
        for num, face in enumerate(self._images):
            self._all_features[:, num, :] = face.get_features()

    def features_out(self):
        """
        Takes in array of shape n x m x 2 where:
        n rows = number of feature points
        m cols = number of images to average
        2 coordinates = x and y for each feature
        Returns array with coordinates of feature points in final image).
        """
        self._features = np.average(self._all_features, axis=0)

    def generate_avg(self):
        """
        Generates image as average of inputs.
        """
        target_pts = self.get_delaunay_points()
        alpha = 1/len(self._images)
        for img in self._images:
            source_pts = img.get_delaunay_points()
            for num, triangle in enumerate(source_pts):
                transform = cv2.getAffineTransform(triangle, target_pts[num])
                mask = np.zeros(img.shape)
                cv2.fillConvexPoly(mask, triangle, 1)
                masked_img = img * mask
                warped = cv2.warpAffine(masked_img, transform, borderMode=cv2.BORDER_REFLECT_101)
            self._image += alpha * warped

    def get_avg(self):
        """
        Returns averaged face.
        """
        return self._image
