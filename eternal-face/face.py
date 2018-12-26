"""
face.py
Defines face class.
"""

import os

import cv2
import dlib
import numpy as np

class Face():
    """
    Represents an image of a face.

    Attributes:
        image: Image of the face.
        delaunay: Face image with Delaunay triangles drawn.
        delaunay_pts: List of Delaunay triangle vertices.
        dims: Tuple of image dimensions.
        features: Array of coordinates of facial features.
    """

    def __init__(self, image, feature_points=None):
        self.image = image
        self.delaunay = self.image.copy()
        self.delaunay_pts = []
        self.dims = self.image.shape
        if feature_points is not None:
            print('Features initialized from specified points.')
            self.features = feature_points
        else:
            self.features = None

    def detect_features(self, auto_border=True, model_name='predictor_68.dat'):
        """
        Detects facial features from image.
        Modified from dlib: http://dlib.net/face_landmark_detection.py.html

        Args:
            auto_border (bool): If True, adds six additional points along border of image.
            model_name (str): Name of predictor model file.
        """
        detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(os.path.split(os.getcwd())[0], 'data', model_name)
        predictor = dlib.shape_predictor(predictor_path)

        # Ask the detector to find the bounding boxes of face after upsampling once.
        detected = detector(self.image, 1)
        if detected:
            points = predictor(self.image, detected[0])

            self.features = np.zeros((68, 2))
            for num, point in enumerate(points.parts()):
                self.features[num, :] = [int(point.x), int(point.y)]

            # Adds points at edges of image.
            if auto_border:
                border_pts = np.array([[0, 0],
                                       [0, self.dims[0]/2],
                                       [0, self.dims[0] - 1],
                                       [self.dims[1] - 1, 0],
                                       [self.dims[1] - 1, self.dims[0]/2],
                                       [self.dims[1] - 1, self.dims[0] - 1]])
                self.features = np.insert(self.features, border_pts.shape[0], border_pts, axis=0)

    def calc_delaunay(self, color=(255, 0, 0)):
        """
        Calculates Delaunay points and draws delaunay triangle diagram.

        Args:
            color (tuple): RGB value of lines for triangles to be drawn.
        """
        rect = (0, 0, self.dims[1], self.dims[0])
        subdiv = cv2.Subdiv2D(rect)

        for point in self.features:
            subdiv.insert((point[0], point[1]))

        triangles = subdiv.getTriangleList()

        for tri in triangles:
            pt1 = (tri[0], tri[1])
            pt2 = (tri[2], tri[3])
            pt3 = (tri[4], tri[5])

            if all(0 <= x <= self.dims[1] for x in [pt1[0], pt2[0], pt3[0]]):
                if all(0 <= y <= self.dims[0] for y in [pt1[1], pt2[1], pt3[1]]):
                    self.delaunay_pts.append([pt1, pt2, pt3])
                    cv2.line(self.delaunay, pt1, pt2, color, 1, cv2.LINE_AA, 0)
                    cv2.line(self.delaunay, pt2, pt3, color, 1, cv2.LINE_AA, 0)
                    cv2.line(self.delaunay, pt3, pt1, color, 1, cv2.LINE_AA, 0)

    def get_features(self):
        """
        Returns array of features.
        """
        if np.sum(self.features) is None:
            print("Detecting features.")
            self.detect_features()
        return self.features

    def set_features(self, features):
        """
        Sets array of features.
        """
        self.features = features

    def get_delaunay_points(self):
        """
        Returns list of Delaunay triangle vertex coordinates.
        """
        if np.sum(self.features) is None:
            print("Detecting features.")
            self.detect_features()
        if not self.delaunay_pts:
            print("Calculating Delaunay triangle vertices.")
            self.calc_delaunay()
        return self.delaunay_pts

    def get_image(self):
        """
        Return image.
        """
        return self.image
