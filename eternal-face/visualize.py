"""
visualize.py
Visualizations for live A/V.
"""

from glob import glob
import os
import math
import random
import sys
import time

import cv2
import numpy as np
import pygame


def np_to_surface(array):
    new_array = array.copy()
    new_array[:, :, 0] = array[:, :, 2]
    new_array[:, :, 2] = array[:, :, 0]
    surface = pygame.surfarray.make_surface(np.flip(np.rot90(np.uint8(new_array)), 0))
    return surface

def set_squares(base, offsets, images, prev=None, pct_blank=0):
    dim = images[0].shape[0]
    bounds = [(offsets[1], offsets[1] + dim, offsets[0], offsets[0] + dim),
              (offsets[1]/4, offsets[1]/4 + dim, offsets[0]/4, offsets[0]/4 + dim),
              (offsets[1]/4, offsets[1]/4 + dim, -(dim + offsets[0]/4), -offsets[0]/4),
              (-(dim + offsets[1]/4), -offsets[1]/4, offsets[0]/4, offsets[0]/4 + dim),
              (-(dim + offsets[1]/4), -offsets[1]/4, -(dim + offsets[0]/4), -offsets[0]/4)]
    if random.random() < pct_blank:
        idx = random.randint(0, len(images) - 1)
        if np.sum(prev) is None:
            images[idx] = np.zeros(images[0].shape)
        else:
            images[idx] = prev[bounds[idx][0]:bounds[idx][1], bounds[idx][2]:bounds[idx][3]]

    for loc, square in enumerate(images):
        base[bounds[loc][0]:bounds[loc][1], bounds[loc][2]:bounds[loc][3]] = square

    return base

def add_texture(base, v_block, h_block, w_size, texture):
    patch = texture[v_block*w_size:v_block*w_size + w_size,
                    h_block*w_size:h_block*w_size + w_size]
    base[v_block*w_size:v_block*w_size + w_size,
         h_block*w_size:h_block*w_size + w_size] = patch

    return base

def progress_texture(base, w_size, texture):
    frames = []
    for v_block in range(base.shape[0] / w_size + 1):
        for h_block in range(base.shape[1] / w_size + 1):
            patch = texture[v_block*w_size:v_block*w_size + w_size,
                            h_block*w_size:h_block*w_size + w_size]
            base[v_block*w_size:v_block*w_size + w_size,
                 h_block*w_size:h_block*w_size + w_size] = patch
            frames.append(base.copy())

    return frames

def resize_warps(warps, dim):

    adj_warps = []
    for warp in warps:
        row_factor, col_factor = 1, 1
        if warp.shape[0] < dim:
            row_factor = math.ceil(float(dim) / warp.shape[0])
        if warp.shape[1] < dim:
            col_factor = math.ceil(float(dim) / warp.shape[1])
        factor = max(row_factor, col_factor)
        if factor > 1:
            warp = cv2.resize(warp, fx=factor, fy=factor, dsize=None, interpolation=cv2.INTER_CUBIC)
        resized = warp[(warp.shape[0] - dim)/2:(warp.shape[0] - dim)/2 + dim,
                       (warp.shape[1] - dim)/2:(warp.shape[1] - dim)/2 + dim]
        adj_warps.append(resized)

    return adj_warps

def animation(warps, texture, target_dim, obj_dim, w_size):

    output = np.zeros(target_dim)
    offset_y = (target_dim[0] - obj_dim)/2
    offset_x = (target_dim[1] - obj_dim)/2
    if offset_y < 0:
        offset_y = 0
    if offset_x < 0:
        offset_x = 0

    pygame.init()
    screen = pygame.display.set_mode((target_dim[1], target_dim[0]))
    pygame.mouse.set_visible(0)

    phase = 1

    new = set_squares(output, (offset_x, offset_y), random.sample(warps, 5))
    show = np_to_surface(new)

    timer = pygame.USEREVENT + 1
    timer_value = 1000
    pygame.time.set_timer(timer, timer_value)
    rand_factor = 1
    idx = -1

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                phase += 1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                timer_value = int(1.1 * timer_value)
                pygame.time.set_timer(timer, timer_value)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                timer_value = int(.9 * timer_value)
                pygame.time.set_timer(timer, timer_value)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                if rand_factor >= .1:
                    rand_factor -= .1
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                if rand_factor <= .9:
                    rand_factor += .1

        if phase == 1:
            screen.blit(show, (0, 0))
            if pygame.event.get(timer):
                new = set_squares(output, (offset_x, offset_y), random.sample(warps, 5),
                                  None, rand_factor)
                show = np_to_surface(new)
                pygame.display.update()

        if phase == 2:
            screen.blit(show, (0, 0))
            if pygame.event.get(timer):
                idx += 1
                h_range = output.shape[1] / w_size + 1
                v_block, h_block = idx / h_range, idx % h_range
                texture_frame = add_texture(output, v_block, h_block, w_size, texture)
                new = set_squares(texture_frame, (offset_x, offset_y), random.sample(warps, 5),
                                  new.copy(), rand_factor)
                show = np_to_surface(new)
                pygame.display.update()

        if phase == 3:
            rand_factor = 0
            screen.blit(show, (0, 0))
            if pygame.event.get(timer):
                new = set_squares(np.zeros(target_dim), (offset_x, offset_y), random.sample(warps, 5),
                                  None, rand_factor)
                show = np_to_surface(new)
                pygame.display.update()


def visualize(set_name):

    image_dir = os.path.join(os.path.split(os.getcwd())[0], 'images', 'output', set_name, 'vis')

    texture_img = cv2.imread(os.path.join(image_dir, 'texture.png'))

    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, 'warp*.' + ext) for ext in extensions]
    image_paths = sorted(sum(map(glob, search_paths), []))
    warps = [cv2.imread(f) for f in image_paths]

    dim = max((texture_img.shape[1]/8, texture_img.shape[0]/8))
    adj_warps = resize_warps(warps, dim)

    if min(warps[0].shape[:2]) > 50:
        w_size = int(min(warps[0].shape[:2]) * .5)
    else:
        w_size = int(min(warps[0].shape[:2]) * .75)

    animation(adj_warps, texture_img, texture_img.shape, dim, w_size)


if __name__ == '__main__':
    visualize(sys.argv[1])
