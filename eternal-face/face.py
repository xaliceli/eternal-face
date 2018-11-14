"""
face.py
Define face class.
"""

import os
import dlib
import cv2
import numpy as np

class Face():
    """
    Represents an image of a face.
    """

    def __init__(self, image):
        self._image = image
        self._delaunay = self._image.copy()
        self._delaunay_pts = []
        self._cols, self._rows = self._image.shape[1], self._image.shape[0]
        self._features = np.zeros((68, 1, 2))

    def detect_features(self, model_name='predictor_68.dat'):
        """
        Detect facial features from image.
        Modified from dlib: http://dlib.net/face_landmark_detection.py.html
        """
        detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(os.path.split(os.getcwd())[0], 'data', model_name)
        predictor = dlib.shape_predictor(predictor_path)

        # Ask the detector to find the bounding boxes of face after upsampling once.
        det = detector(self._image, 1)[0]
        points = predictor(self._image, det)

        for num, point in enumerate(points.parts()):
            self._features[num, 0, :] = [int(point.x), int(point.y)]

    def calc_delaunay(self, color=(255, 0, 0)):
        """
        Draw delaunay triangle diagram.
        """
        rect = (0, 0, self._cols, self._rows)
        subdiv = cv2.Subdiv2D(rect)

        for point in self._features:
            subdiv.insert((point[0][0], point[0][1]))

        triangles = subdiv.getTriangleList()

        for tri in triangles:
            pt1 = (tri[0], tri[1])
            pt2 = (tri[2], tri[3])
            pt3 = (tri[4], tri[5])

            if all(0 <= x <= self._cols for x in [pt1[0], pt2[0], pt3[0]]):
                if all(0 <= y <= self._rows for y in [pt1[1], pt2[1], pt3[1]]):
                    self._delaunay_pts.append((pt1, pt2, pt3))
                    cv2.line(self._delaunay, pt1, pt2, color, 1, cv2.LINE_AA, 0)
                    cv2.line(self._delaunay, pt2, pt3, color, 1, cv2.LINE_AA, 0)
                    cv2.line(self._delaunay, pt3, pt1, color, 1, cv2.LINE_AA, 0)

    def get_features(self):
        """
        Returns array of features.
        """
        if np.sum(self._features) == 0:
            self.detect_features()
        return self._features

    def get_delaunay_points(self):
        """
        Returns list of Delaunay triangle vertex coordinates.
        """
        if not self._delaunay_pts:
            self.calc_delaunay()
        return self._delaunay_pts
