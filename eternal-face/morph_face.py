"""
morph_face.py
Performs averaging, morphing, and texturizing operations on input faces.
"""

import os

import cv2
import numpy as np

from face import Face
from warp import triangle_warp

class MorphFace(Face):
    """
    Performs averaging, morphing, and texturizing operations on input faces.

    Attributes:
        images: List of Face class objects.
        textures: List of textures generated from warp triangles.
        frames: List of frames generated in morph sequence.
        all_features: Array of feature points from all images.
        write: Optional directory to write outputs to.
    """

    def __init__(self, images, feature_points=None, write=None):
        Face.__init__(self, np.zeros(images[0].shape))
        self.images = [Face(img, feature_points) for img in images]
        self.textures = []
        self.frames = []
        self.all_features = np.zeros((74, len(images), 2))
        self.write = write
        if self.write:
            if not os.path.exists(self.write):
                os.makedirs(self.write)

    def features_in(self, detect=True):
        """
        Calculates features and Delaunay triangles for each input image.
        """
        for num, face in enumerate(self.images):
            if detect:
                if np.sum(face.get_features()):
                    self.all_features[:, num, :] = face.get_features()
            else:
                divided = np.zeros((self.all_features.shape[0], 2))
                w_step = face.dims[1] / 7
                h_step = face.dims[0] / 10
                for row in range(1, 11):
                    for col in range(1, 8):
                        idx = (row - 1) * 7 + (col - 1)
                        divided[idx, :] = (col * w_step, row * h_step)
                        if col * w_step == face.dims[1]:
                            divided[idx, :][0] -= 1
                        if row * h_step == face.dims[0]:
                            divided[idx, :][1] -= 1
                divided[70] = [0, 0]
                divided[71] = [0, face.dims[0] - 1]
                divided[72] = [face.dims[1] - 1, 0]
                divided[73] = [face.dims[1] - 1, face.dims[0] - 1]
                face.set_features(divided)
                self.all_features[:, num, :] = divided
        self.all_features = self.all_features[:, ~np.all(self.all_features == 0, axis=(0, 2)), :]
        # TODO: Return to this -- why negative feature values? off frame?
        self.all_features = self.all_features[:, ~np.any(self.all_features < 0, axis=(0, 2)), :]

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
        self.features = np.int32(np.round(np.average(self.all_features, axis=1, weights=weights)))

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
                match = np.where((source_features[:, 0] == point[0]) & \
                                 (source_features[:, 1] == point[1]))[0][0]
                target_triangle.append((self.features[match, 0],
                                        self.features[match, 1]))
            delaunay_mapping.append(target_triangle)
        return delaunay_mapping

    def generate_avg(self):
        """
        Generates image as average of inputs.
        """
        if np.sum(self.all_features) == 0:
            self.features_in()
        if np.sum(self.features) is None:
            self.features_out()
        alpha = 1.0/self.all_features.shape[1]
        for img_num, img in enumerate(self.images):
            if np.sum(img.get_features()) and np.all(img.get_features() >= 0) and \
               np.all(img.get_features() < img.get_image().shape[0]):
                source_pts = img.get_delaunay_points()
                target_pts = self.get_delaunay_mapping(img)
                image_warped = np.zeros(self.dims)
                for num, triangle in enumerate(source_pts):
                    tri_warped, _ = triangle_warp(img.get_image(), triangle, target_pts[num])
                    image_warped = np.where(tri_warped > 0, tri_warped, image_warped)
                if img_num > 0:
                    empty = ~image_warped.any(axis=2)
                    empty_mat = np.dstack((empty, empty, empty))
                    image_warped = np.where(empty_mat, self.image / (img_num * alpha), image_warped)
                self.image += alpha * image_warped

    def generate_morph(self, frames=10):
        """
        Generates morph frames between inputs.

        Args:
            frames (int): Number of frames between images.
        """
        if np.sum(self.all_features) == 0:
            self.features_in()
        alpha = 1.0/frames
        for frame in range(frames + 1):
            frame_img = np.zeros(self.dims)
            self.features_out(weights=[1-frame*alpha, frame*alpha])
            for img_num, img in enumerate(self.images):
                source_pts = img.get_delaunay_points()
                target_pts = self.get_delaunay_mapping(img)
                image_warped = np.zeros(self.dims)
                for num, triangle in enumerate(source_pts):
                    tri_warped, _ = triangle_warp(img.get_image(), triangle, target_pts[num])
                    image_warped = np.where(tri_warped > 0, tri_warped, image_warped)
                if img_num == 0:
                    frame_img += (1 - alpha * frame) * image_warped
                else:
                    frame_img += alpha * frame * image_warped
            self.frames.append(frame_img)

    def generate_regions(self, detect, min_window=25):
        """
        Generates individual warp regions for texture synthesis.

        Args:
            min_window (int): Minimum width of region to save.
            detect (bool): Detect features. If false, splits image
                into equal regions.
        """
        if np.sum(self.all_features) == 0:
            self.features_in(detect)
        if np.sum(self.features) is None:
            self.features_out()
        for img_num, img in enumerate(self.images):
            source_pts = img.get_delaunay_points()
            target_pts = self.get_delaunay_points()
            for num, triangle in enumerate(source_pts):
                _, bounded = triangle_warp(img.get_image(), triangle, target_pts[num], mask=False)
                window = min(bounded.shape[:2])
                if window > min_window:
                    if self.write:
                        bounded_path = os.path.join(self.write,
                                                    'warp' + str(img_num) + str(num) + '.png')
                        cv2.imwrite(bounded_path, bounded)
                    self.textures.append(bounded)

    def get_avg(self):
        """
        Returns averaged face.
        """
        if np.sum(self.image) == 0:
            self.generate_avg()
        return self.image

    def get_morph(self, num_frames):
        """
        Returns morph frames.
        """
        if not self.frames:
            self.generate_morph(num_frames)
        return self.frames

    def get_texture_regions(self, detect=True):
        """
        Returns distorted facial textures.
        """
        if not self.textures:
            self.generate_regions(detect)
        return self.textures
