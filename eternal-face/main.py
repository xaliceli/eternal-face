"""
main.py
Run scripts to generate output video.
"""

from glob import glob
from itertools import combinations
from random import shuffle
import sys
import os
import cv2
import scipy
from average_face import AverageFace

def read_faces(image_dir, downsample=2):
    """
    Read in all images from directory. and return randomized list.
    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]
    images = [img[::downsample, ::downsample] for img in images]
    shuffle(images)

    return images


def average_faces(faces, k=None, limit=None):
    """
    Computes k averages from n faces.
    """
    if not k:
        k = len(faces) - 1
    avg_sets = list(combinations(faces, k))
    if not limit:
        limit = len(avg_sets)

    averages = []
    for num, combination in enumerate(avg_sets[:limit]):
        print('Generating average ' + str(num) + '...')
        avg = AverageFace(images=combination).get_avg()
        averages.append(avg)

    return averages


if __name__ == '__main__':

    set_dir = sys.argv[2]
    image_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'input', set_dir)
    out_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'output', set_dir)
    images_in = read_faces(image_dir)

    if sys.argv[1] == 'avg':
        images_out = average_faces(images_in)

    for num, face in enumerate(images_out):
        out_path = os.path.join(out_dir, 'avg{0:04d}.png'.format(num))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_path, face)
