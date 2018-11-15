"""
average.py
Computes n choose k averages of input images.
"""

import os
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
        self._images = [Face(img) for img in images]
        self._all_features = np.zeros((68, len(images), 2))

    def features_in(self):
        """
        Calculates features and Delaunay triangles for each input image.
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
        self._features = np.average(self._all_features, axis=1)

    def get_delaunay_mapping(self, image):
        """
        Because Delaunay triangulation is not guaranteed to return
        vertices in the same order, calling get_delaunay_points()
        on both source and target images results in distortions in warps.
        So, instead we generate Delaunay points in the target,
        averaged image by mapping corresponding features instead.
        """
        triangles = image.get_delaunay_points()
        source_features = image.get_features()
        delaunay_mapping = []
        for triangle in triangles:
            target_triangle = []
            for point in triangle:
                x_match = np.argwhere(source_features == point[0])
                for coords in x_match:
                    if coords[1] == 0:
                        coords[1] += 1
                        if source_features[tuple(coords)] == point[1]:
                            target_triangle.append((self._features[coords[0], 0],
                                                    self._features[coords[0], 1]))
            delaunay_mapping.append(target_triangle)
        return delaunay_mapping

    def generate_avg(self):
        """
        Generates image as average of inputs.
        """
        if np.sum(self._all_features) == 0:
            self.features_in()
        if np.sum(self._features) == 0:
            self.features_out()
        alpha = 1.0/len(self._images)
        for img in self._images:
            source_pts = img.get_delaunay_points()
            target_pts = self.get_delaunay_mapping(img)
            for num, triangle in enumerate(source_pts[:len(target_pts)]):
                warped = morph.triangle_warp(img.get_image(), triangle, target_pts[num])
                self._image += alpha * warped

    def get_avg(self):
        """
        Returns averaged face.
        """
        if np.sum(self._image) == 0:
            self.generate_avg()
        return self._image
