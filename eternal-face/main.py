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
import numpy as np

from morph_face import MorphFace


def read_images(image_dir, randomize=True, downsample=2):
    """
    Reads in all images from directory and returns randomized list.

    Args:
        image_dir (str): Full path of directory where images are stored.
        randomize (bool): If True, randomly shuffles images before returning.
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
    if randomize:
        random.shuffle(images)

    return images


def average_faces(faces, out_dir, k=None, limit=None):
    """
    Computes k averages from n faces.

    Args:
        faces (list): A list of images representing faces.
        out_dir (str): Directory to save images to. If None,
            images are not saved.
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
        img = MorphFace(images=combination).get_avg()
        avgs.append(img)
        if out_dir:
            out_path = os.path.join(out_dir, 'avg{0:04d}.png'.format(num))
            cv2.imwrite(out_path, img)
    return avgs


def morph_averages(images, out_dir, f_rate=10, frames=40):
    """
    Creates and saves video of images morphing.

    Args:
        images (list): List containing images to morph together.
        out_dir (str): Directory to save video to.
        f_rate (int): Frames per second.
        frames (int): Number of frames to generate between images.
    """
    writer = cv2.VideoWriter(os.path.join(out_dir, 'video.avi'),
                             cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             f_rate, (images[0].shape[1], images[0].shape[0]))
    for avg_num, avg in enumerate(images):
        next_num = avg_num + 1
        if next_num == len(images):
            next_num = 0
        morph_frames = MorphFace(images=[avg, images[next_num]]).get_morph(frames)
        for frame in morph_frames:
            writer.write(np.uint8(frame))
    writer.release()


def create_textures(faces, out_dir):
    """
    Computes textures from intentionally distorted warp triangles.

    Args:
        faces (list): A list of images representing faces.
        out_dir (str): Directory to save textures to.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Choose one random face out of all available faces for texturization.
    MorphFace(images=[faces[random.randint(0, len(faces) - 1)]], write=out_dir).get_texture()
    # for text_num, texture in enumerate(textures):
    #     out_path = os.path.join(out_dir, 'texture{0:04d}.png'.format(text_num))
    #     cv2.imwrite(out_path, texture)


def main(action, set_dir):
    """
    Runs operations based on arguments passed in terminal.

    Args:
        action (str): Action to perform.
        set_dir (str): Directory name where input images are stored.
    """
    image_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'input', set_dir)
    out_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'output', set_dir)
    images_in = read_images(image_dir)

    if action == 'avg':
        average_faces(images_in, out_dir)
    elif action == 'video':
        images = read_images(out_dir, downsample=1)
        if len(images) < 2:
            images = average_faces(images_in, out_dir)
        morph_averages(images, out_dir)
    elif action == 'texture':
        create_textures(images_in, os.path.join(out_dir, 'textures'))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
