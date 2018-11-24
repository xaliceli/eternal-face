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

    def __init__(self, image):
        self._image = image
        self._delaunay = self._image.copy()
        self._delaunay_pts = []
        self._dims = self._image.shape
        self._features = np.zeros((68, 2))

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
        det = detector(self._image, 1)[0]
        points = predictor(self._image, det)

        for num, point in enumerate(points.parts()):
            self._features[num, :] = [int(point.x), int(point.y)]

        # Adds points at edges of image.
        if auto_border:
            border_pts = np.array([[0, 0],
                                   [0, self._dims[1]/2],
                                   [0, self._dims[1] - 1],
                                   [self._dims[0] - 1, 0],
                                   [self._dims[0] - 1, self._dims[1]/2],
                                   [self._dims[0] - 1, self._dims[1] - 1]])
            self._features = np.insert(self._features, border_pts.shape[0], border_pts, axis=0)

    def calc_delaunay(self, color=(255, 0, 0)):
        """
        Calculates Delaunay points and draws delaunay triangle diagram.

        Args:
            color (tuple): RGB value of lines for triangles to be drawn.
        """
        rect = (0, 0, self._dims[1], self._dims[0])
        subdiv = cv2.Subdiv2D(rect)

        for point in self._features:
            subdiv.insert((point[0], point[1]))

        triangles = subdiv.getTriangleList()

        for tri in triangles:
            pt1 = (tri[0], tri[1])
            pt2 = (tri[2], tri[3])
            pt3 = (tri[4], tri[5])

            if all(0 <= x <= self._dims[1] for x in [pt1[0], pt2[0], pt3[0]]):
                if all(0 <= y <= self._dims[0] for y in [pt1[1], pt2[1], pt3[1]]):
                    self._delaunay_pts.append([pt1, pt2, pt3])
                    cv2.line(self._delaunay, pt1, pt2, color, 1, cv2.LINE_AA, 0)
                    cv2.line(self._delaunay, pt2, pt3, color, 1, cv2.LINE_AA, 0)
                    cv2.line(self._delaunay, pt3, pt1, color, 1, cv2.LINE_AA, 0)

    def get_features(self):
        """
        Returns array of features.
        """
        if np.sum(self._features) == 0:
            print("Detecting features.")
            self.detect_features()
        return self._features

    def get_delaunay_points(self):
        """
        Returns list of Delaunay triangle vertex coordinates.
        """
        if np.sum(self._features) == 0:
            print("Detecting features.")
            self.detect_features()
        if not self._delaunay_pts:
            print("Calculating Delaunay triangle vertices.")
            self.calc_delaunay()
        return self._delaunay_pts

    def get_image(self):
        """
        Return image.
        """
        return self._image
