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
import imageio
import numpy as np

from face import Face
from morph_face import MorphFace
from texture import Texture


def read_images(image_dir, randomize=True, downsample=1, crop=0):
    """
    Reads in all images from directory and returns randomized list.

    Args:
        image_dir (str): Full path of directory where images are stored.
        randomize (bool): If True, randomly shuffles images before returning.
        downsample (int): Downsample factor to resize images by.
        crop (int): Percent of existing dimensions to crop image by.

    Returns:
        images (list): List of downsampled images as numpy arrays.
    """
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    images = [cv2.imread(f) for f in image_files]
    crop_row, crop_col = int(images[0].shape[0]*crop/2), int(images[0].shape[1]*crop/2)
    images = [img[crop_row:img.shape[0]-crop_row:downsample,
                  crop_col:img.shape[1]-crop_col:downsample] for img in images]
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
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not k:
        k = len(faces)
        if k > 2:
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


def morph_gif(images, out_dir, features=None, f_rate=20, frames=40):
    """
    Creates and saves video of images morphing.

    Args:
        images (list): List containing images to morph together.
        out_dir (str): Directory to save video to.
        features (np.array): If none, calculate features automatically.
            Otherwise, use features provided.
        f_rate (int): Frames per second.
        frames (int): Number of frames to generate between images.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    frame_imgs = []
    for num, img in enumerate(images):
        next_num = num + 1
        if next_num == len(images):
            next_num = 0
        morph_frames = MorphFace(images=[img, images[next_num]],
                                 feature_points=features).get_morph(frames)
        for frame in morph_frames:
            frame_imgs.append(cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2RGB))
    imageio.mimsave(os.path.join(out_dir, 'morph.gif'), frame_imgs, fps=f_rate)
    # os.system('gifsicle -b -O3 --colors 256 ' + os.path.join(out_dir, 'morph.gif'))


def morph_video(images, out_dir, features=None, f_rate=24, frames=40):
    """
    Creates and saves video of images morphing.

    Args:
        images (list): List containing images to morph together.
        out_dir (str): Directory to save video to.
        features (np.array): If none, calculate features automatically.
            Otherwise, use features provided.
        f_rate (int): Frames per second.
        frames (int): Number of frames to generate between images.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    file_num = str(len(os.listdir(out_dir)))
    writer = cv2.VideoWriter(os.path.join(out_dir, 'morph' + file_num + '.avi'),
                             cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                             f_rate, (images[0].shape[1], images[0].shape[0]))
    for num, img in enumerate(images):
        next_num = num + 1
        if next_num == len(images):
            next_num = 0
        morph_frames = MorphFace(images=[img, images[next_num]],
                                 feature_points=features).get_morph(frames)
        for frame in morph_frames:
            writer.write(np.uint8(frame))
    writer.release()


def seed_textures(faces, out_dir):
    """
    Warps facial regions for texture generation.

    Args:
        faces (list): A list of images representing faces.
        out_dir (str): Directory to save warps to.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seeds = MorphFace(images=faces, write=out_dir).get_texture_regions()
    return seeds


def generate_textures(seeds, dims, out_dir, window=None, rotate=3):
    """
    Computes textures from intentionally distorted warp triangles.

    Args:
        faces (list): A list of images representing faces.
        dims (tuple): Desired dimensions of output image.
        out_dir (str): Directory to save textures to.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    textures = []
    for num, seed in enumerate(seeds):
        if not window:
            window = min(seed.shape[:2])/2
        if window % 2 == 0:
            window -= 1
        texture = Texture(dims, window).get_texture(seed, None, rotate)
        out_path = os.path.join(out_dir, 'texture{0:04d}.png'.format(num))
        cv2.imwrite(out_path, texture)
        textures.append(texture)
    return textures


def transfer_texture(textures, intensities, out_dir, window=None, rotate=3, iterations=0):
    """
    Computes textures from intentionally distorted warp triangles.

    Args:
        faces (list): A list of images representing faces.
        out_dir (str): Directory to save textures to.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    transfers = []
    for num, texture in enumerate(textures):
        if (min(texture.shape[:2]) * (.67) ** (iterations + 1))/6 > 1:
            for iter_num in range(iterations+1):
                if window is None and iter_num == 0:
                    w_size = min(texture.shape[:2])*2/3
                elif iter_num == 0:
                    w_size = window
                elif iter_num > 0:
                    w_size = w_size - w_size/3
                if w_size % 2 == 0:
                    w_size -= 1
                transfer = Texture(intensities.shape, w_size).get_texture(texture,
                                                                          intensities,
                                                                          rotate)
                out_path = os.path.join(out_dir, 'transfer' + str(num) + str(iter_num) + '.png')
                cv2.imwrite(out_path, transfer)
                transfers.append(transfer)
        else:
            print('Skipping -- source image cannot support specified number of iterations.')

    return transfers


def main(action, set_dir, source='inputs'):
    """
    Runs operations based on arguments passed in terminal.

    Args:
        action (str): Action to perform.
        set_dir (str): Directory name where input images are stored.
    """
    image_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'input', set_dir)
    out_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'output', set_dir)
    images_in = read_images(image_dir, downsample=2)

    if action == 'avg':
        average_faces(images_in, os.path.join(out_dir, 'averages'))
    elif action == 'morph_video':
        if source == 'transfers':
            feature_source = random.choice(images_in)
            features = Face(feature_source).get_features()
            images = read_images(os.path.join(out_dir, 'transfers'), randomize=False)
        elif source == 'inputs':
            images = images_in
            features = None
        elif source == 'averages':
            images = read_images(os.path.join(out_dir, 'averages'), randomize=True)
            features = None
        morph_video(images, os.path.join(out_dir, 'morphs'), features)
    elif action == 'morph_gif':
        if source == 'transfers':
            feature_source = random.choice(images_in)
            features = Face(feature_source).get_features()
            images = read_images(os.path.join(out_dir, 'transfers'), randomize=False)
        elif source == 'inputs':
            images = images_in
            features = None
        elif source == 'averages':
            images = read_images(os.path.join(out_dir, 'averages'), randomize=True)
            features = None
        morph_gif(images, os.path.join(out_dir, 'morphs'), features)
    elif action == 'seed_textures':
        if source == 'averages':
            images = read_images(os.path.join(out_dir, 'averages'))
        elif source == 'inputs':
            images = images_in
        seed_textures(images, os.path.join(out_dir, 'warps'))
    elif action == 'generate_textures':
        seeds = read_images(os.path.join(out_dir, 'warps'))
        generate_textures(seeds, images_in[0].shape, os.path.join(out_dir, 'textures'))
    elif action == 'transfer_from_warp':
        if source == 'averages':
            images = read_images(os.path.join(out_dir, 'averages'))
        elif source == 'inputs':
            images = images_in
        textures = read_images(os.path.join(out_dir, 'warps'))
        transfer_texture(textures, random.choice(images),
                         os.path.join(out_dir, 'transfers'), iterations=3)
    elif action == 'transfer_from_texture':
        if source == 'averages':
            images = read_images(os.path.join(out_dir, 'averages'))
        elif source == 'inputs':
            images = images_in
        textures = read_images(os.path.join(out_dir, 'textures'), crop=0.8)
        transfer_texture(textures, random.choice(images),
                         os.path.join(out_dir, 'transfers'), iterations=3)
    elif action == 'full':
        seeds = seed_textures(images_in, os.path.join(out_dir, 'warps'))
        if source == 'random':
            seeds = random.sample(seeds, len(seeds)/10)
        textures = generate_textures(seeds, images_in[0].shape, os.path.join(out_dir, 'textures'))
        transfers = transfer_texture(seeds, random.choice(images_in),
                                     os.path.join(out_dir, 'transfers'), iterations=3)
        feature_source = random.choice(images_in)
        features = Face(feature_source).get_features()
        for set_num in range(0, len(transfers), 4):
            morph_video(transfers[set_num:set_num + 4], os.path.join(out_dir, 'morphs'), features)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
