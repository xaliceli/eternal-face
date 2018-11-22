"""
main.py
Run scripts to generate output video.
"""

from glob import glob
from itertools import combinations
import os
import random
import sys

import cv2

from average_face import AverageFace


def read_faces(image_dir, downsample=2):
    """
    Reads in all images from directory and returns randomized list.

    Args:
        image_dir (str): Full path of directory where images are stored.
        downsample (int): Downsample factor to resize images by.

    Returns:
        images (list): List of downsampled images as numpy arrays.

    Raises: @TODO
        TBD: Error if directory does not exist.
    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]
    images = [img[::downsample, ::downsample] for img in images]
    random.shuffle(images)

    return images


def average_faces(faces, out_dir, k=None, limit=None):
    """
    Computes k averages from n faces.

    Args:
        faces (list): A list of images representing faces.
        k (int): Number of images to include in each average.
            Default is one less than the number of faces.
        limit (int): Number of averages to produce.
            Default is len(faces) choose k.

    Returns:
        averages (list): List of averaged faces as numpy arrays.
    """
    if not k:
        k = len(faces) - 1
    avg_sets = list(combinations(faces, k))
    if not limit:
        limit = len(avg_sets)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    avgs = []
    for num, combination in enumerate(avg_sets[:limit]):
        print('Generating average ' + str(num) + '...')
        img = AverageFace(images=combination).get_avg()
        avgs.append(img)
        if out_dir:
            out_path = os.path.join(out_dir, 'avg{0:04d}.png'.format(num))
            cv2.imwrite(out_path, img)
    return avgs


def morph_averages(avgs, out_dir, frames=100):
    """
    """
    writer = cv2.VideoWriter(os.path.join(out_dir, 'video.avi'), -1, 1, avgs[0].shape)
    for avg_num, avg in enumerate(avgs):
        next_num = avg_num + 1
        if next_num == len(avgs):
            next_num = 0
        morph_frames = AverageFace(images=[avg, avgs[next_num]]).get_morph(frames)
        for frame in morph_frames:
            writer.write(frame)
    writer.release()


def create_textures(faces, out_dir):
    """
    Computes k averages from n faces.

    Args:
        faces (list): A list of images representing faces.
        k (int): Number of images to include in each average.
            Default is one less than the number of faces.
        limit (int): Number of averages to produce.
            Default is len(faces) choose k.

    Returns:
        averages (list): List of averaged faces as numpy arrays.
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    textures = AverageFace(images=faces[random.randint(0, len(faces) - 1)]).get_texture()
    for text_num, texture in enumerate(textures):
        out_path = os.path.join(out_dir, 'texture{0:04d}.png'.format(text_num))
        cv2.imwrite(out_path, texture)


if __name__ == '__main__':

    set_dir = sys.argv[2]
    image_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'input', set_dir)
    out_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'output', set_dir)
    images_in = read_faces(image_dir)

    if sys.argv[1] == 'avg':
        average_faces(images_in, out_dir)
    elif sys.argv[1] == 'video':
        averages = average_faces(images_in, out_dir)
        morph_averages(averages, out_dir)
    elif sys.argv[1] == 'texture':
        create_textures(images_in, os.path.join(out_dir, 'textures'))

