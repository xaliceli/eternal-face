"""
average.py
Computes n choose k averages of input images.
"""

import os

import cv2
import numpy as np

from face import Face
from morph import triangle_warp
from textures import fill_image

class AverageFace(Face):
    """
    Average of n faces.
    """

    def __init__(self, images):
        Face.__init__(self, np.zeros(images[0].shape))
        self._images = [Face(img) for img in images]
        self._textures = []
        self._frames = []
        self._all_features = np.zeros((74, len(images), 2))

    def features_in(self):
        """
        Calculates features and Delaunay triangles for each input image.
        """
        for num, face in enumerate(self._images):
            self._all_features[:, num, :] = face.get_features()

    def features_out(self, weights=None):
        """
        Takes in array of shape n x m x 2 where:
        n rows = number of feature points
        m cols = number of images to average
        2 coordinates = x and y for each feature
        Returns array with coordinates of feature points in final image).
        """
        self._features = np.int32(np.average(self._all_features, axis=1, weights=weights))

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
        for img_num, img in enumerate(self._images):
            source_pts = img.get_delaunay_points()
            target_pts = self.get_delaunay_mapping(img)
            image_warped = np.zeros(self._dims)
            for num, triangle in enumerate(source_pts):
                tri_warped, _ = triangle_warp(img.get_image(), triangle, target_pts[num])
                image_warped = np.where(tri_warped > 0, tri_warped, image_warped)
            if img_num > 0:
                empty = ~image_warped.any(axis=2)
                empty_mat = np.dstack((empty, empty, empty))
                image_warped = np.where(empty_mat, self._image / (img_num * alpha), image_warped)
            self._image += alpha * image_warped

    def generate_morph(self, frames=100):
        """
        Generates morph frames between inputs.
        """
        if np.sum(self._all_features) == 0:
            self.features_in()
        alpha = 1.0/frames
        for frame in range(frames + 1):
            frame_img = np.zeros(self._dims)
            self.features_out(weights=[1-frame*alpha, frame*alpha])
            for img_num, img in enumerate(self._images):
                source_pts = img.get_delaunay_points()
                target_pts = self.get_delaunay_mapping(img)
                image_warped = np.zeros(self._dims)
                for num, triangle in enumerate(source_pts):
                    tri_warped, _ = triangle_warp(img.get_image(), triangle, target_pts[num])
                    image_warped = np.where(tri_warped > 0, tri_warped, image_warped)
                if img_num == 0:
                    frame_img += 1 - alpha * frame * image_warped
                else:
                    frame_img += alpha * frame * image_warped
            self._frames.append(frame_img)

    def generate_textures(self):
        """
        Generates textures of individual regions.
        """
        if np.sum(self._all_features) == 0:
            self.features_in()
        if np.sum(self._features) == 0:
            self.features_out()
        for img in self._images:
            source_pts = img.get_delaunay_points()
            target_pts = self.get_delaunay_points()
            for num, triangle in enumerate(source_pts):
                tri_warped, bounded = triangle_warp(img.get_image(),
                                                    triangle, target_pts[num], mask=False)
                window = min(bounded.shape[:2])
                if window % 2 == 0:
                    window -= 1
                cv2.imshow('bounded', np.uint8(bounded))
                cv2.waitKey(25)
                filled = fill_image(bounded, tri_warped.shape, window)
                cv2.imshow('filled', np.uint8(filled))
                cv2.waitKey(25)
                out_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'output', 'set_1')
                out_path = os.path.join(out_dir, 'texture{0:04d}.png'.format(num))
                cv2.imwrite(out_path, filled)
                self._textures.append(filled)

    def get_avg(self):
        """
        Returns averaged face.
        """
        if np.sum(self._image) == 0:
            self.generate_avg()
        return self._image

    def get_morph(self, num_frames):
        """
        Returns morph frames.
        """
        if not self._frames:
            self.generate_morph(num_frames)
        return self._frames

    def get_texture(self):
        """
        Returns distorted facial textures.
        """
        if not self._textures:
            self.generate_textures()
        return self._textures
