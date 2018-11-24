"""
morph_face.py
Performs averaging, morphing, and texturizing operations on input faces.
"""

import os

import cv2
import numpy as np

from face import Face
from textures import fill_image
from warp import triangle_warp

class MorphFace(Face):
    """
    Performs averaging, morphing, and texturizing operations on input faces.

    Attributes:
        images: List of Face class objects.
        textures: List of textures generated from warp triangles.
        frames: List of frames generated in morph sequence.
        all_features: Array of feature points from all images.
    """

    def __init__(self, images, write=None):
        Face.__init__(self, np.zeros(images[0].shape))
        self._images = [Face(img) for img in images]
        self._textures = []
        self._frames = []
        self._all_features = np.zeros((74, len(images), 2))
        self._write = write
        if self._write:
            if not os.path.exists(self._write):
                os.makedirs(self._write)

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
        Calculates array with new feature point coordinates
        as averages of each input image's coordinates.

        Args:
            weights (list): If None, weights all images equally.
                Otherwise, applies weight values to each image accordingly.
        """
        self._features = np.int32(np.average(self._all_features, axis=1, weights=weights))

    def get_delaunay_mapping(self, image):
        """
        Calculates new Delaunay triangle vertices as mapping of
        individual image's feature points and Delaunay triangles.

        Because Delaunay triangulation is not guaranteed to return
        vertices in the same order, calling get_delaunay_points()
        on both source and target images results in distortions in warps.
        So, instead we generate Delaunay points in the target,
        averaged image by mapping corresponding features instead.

        Args:
            image (Face): Image of class Face.

        Returns:
            delaunay_mapping (list): List of corresponding
                triangle vertices.
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

    def generate_morph(self, frames=10):
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
                    frame_img += (1 - alpha * frame) * image_warped
                else:
                    frame_img += alpha * frame * image_warped
            self._frames.append(frame_img)

    def generate_textures(self, min_window=20):
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
                window = min(bounded.shape[:2])/2
                if window > min_window:
                    print('Generating texture from triangle ' + str(num))
                    if window % 2 == 0:
                        window -= 1
                    filled = fill_image(bounded, tri_warped.shape, None, window)
                    self._textures.append(filled)
                    if self._write:
                        out_path = os.path.join(self._write, 'texture{0:04d}.png'.format(num))
                        cv2.imwrite(out_path, filled)

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