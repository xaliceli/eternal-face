"""
visualize.py
Visualizations for live A/V.
"""
import argparse
from glob import glob
import os
import math
import random
import sys

import cv2
import numpy as np

def set_squares(base, offsets, images, prev=None, pct_blank=0):
    dim = images[0].shape[0]
    bounds = [(offsets[1], offsets[1] + dim, offsets[0], offsets[0] + dim),
              (offsets[1]/4, offsets[1]/4 + dim, offsets[0]/4, offsets[0]/4 + dim),
              (offsets[1]/4, offsets[1]/4 + dim, -(dim + offsets[0]/4), -offsets[0]/4),
              (-(dim + offsets[1]/4), -offsets[1]/4, offsets[0]/4, offsets[0]/4 + dim),
              (-(dim + offsets[1]/4), -offsets[1]/4, -(dim + offsets[0]/4), -offsets[0]/4)]
    if random.random() < pct_blank:
        idx = random.randint(0, len(images) - 1)
        if prev is None:
            images[idx] = np.zeros(images[0].shape)
        else:
            images[idx] = prev[idx]

    for loc, square in enumerate(images):
        base[bounds[loc][0]:bounds[loc][1], bounds[loc][2]:bounds[loc][3]] = square

    return base

def melt(base, amount, offsets, dim, iterations):
    # bounds = [(offsets[1], offsets[1] + dim, offsets[0], offsets[0] + dim),
    #           (offsets[1]/4, offsets[1]/4 + dim, offsets[0]/4, offsets[0]/4 + dim),
    #           (offsets[1]/4, offsets[1]/4 + dim, -(dim + offsets[0]/4), -offsets[0]/4),
    #           (-(dim + offsets[1]/4), -offsets[1]/4, offsets[0]/4, offsets[0]/4 + dim),
    #           (-(dim + offsets[1]/4), -offsets[1]/4, -(dim + offsets[0]/4), -offsets[0]/4)]
    bounds = [(offsets[1], offsets[1] + dim[0], offsets[0], offsets[0] + dim[1])]
    if amount % 2 > 0:
        amount -= 1
    base = cv2.GaussianBlur(base, (amount - 1, amount - 1), 0)
    for iteration in range(iterations):
        for bound in bounds:
            row = random.randint(bound[0] + amount, bound[1] - amount)
            col = random.randint(bound[2], bound[3])
            melted = base[row - (amount - 2):row + 1, col - amount/2:col + amount/2]
            # melted = cv2.GaussianBlur(base[row - (amount - 2):row + 1, col - amount/2:col + amount/2], (9, 9), 0)
            base[row + 1:row + amount, col - amount/2: col + amount/2] = melted
            # base = cv2.GaussianBlur(base, (amount - 1, amount - 1), 0)
        yield base.copy()

def vertical_distort(base, factor, iterations):
    for iteration in range(iterations):
        for col in range(factor*2, base.shape[1] - factor*4):
            tot_height = base.shape[0]
            prev_height = 0
            for group in range(10):
                if group < 9:
                    height = random.randint(0, tot_height - prev_height)
                else:
                    height = tot_height - prev_height
                adj = random.randint(col - factor, col + factor)
                base[prev_height:height + prev_height, col-factor:col + factor] = base[prev_height:height + prev_height, adj-factor:adj+factor]
                prev_height += height
        yield base.copy()

def add_texture(base, v_block, h_block, w_size, texture):
    patch = texture[v_block*w_size:v_block*w_size + w_size,
                    h_block*w_size:h_block*w_size + w_size]
    # current = base[v_block*w_size:v_block*w_size + w_size,
    #                h_block*w_size:h_block*w_size + w_size]

    # abs_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # seam_coords = meb.minimum_error_boundary(abs_patch)
    # if np.sum(current) == 0:
    #     stitched = meb.stitch_images(patch, current, seam_coords, w_size, 'vertical')
    # else:
    #     stitched = meb.stitch_images(current, patch, seam_coords, w_size, 'vertical')

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

def resize_image(image, dims, resize=True):
    """
    Resizes image to supplied dimensions, preserving aspect ratio.
    """
    if resize:
        row_factor = float(dims[0]) / image.shape[0]
        col_factor = float(dims[1]) / image.shape[1]
        factor = np.float(max(row_factor, col_factor))
        if factor > 1:
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    resized = image[(image.shape[0] - dims[0])/2:(image.shape[0] - dims[0])/2 + dims[0],
                    (image.shape[1] - dims[1])/2:(image.shape[1] - dims[1])/2 + dims[1]]
    return resized

class Visualize():

    def __init__(self, set_name, out_dim):
        self.in_dir = os.path.join(os.path.split(os.getcwd())[0],
                                   'images', 'output', set_name, 'vis')
        self.out_dir = os.path.join(os.path.split(os.getcwd())[0],
                                    'images', 'output', set_name, 'vis', 'videos')
        self.out_dim = out_dim
        self.warps = None
        self.read_images()
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def read_images(self):
        extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                      'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']
        search_paths = [os.path.join(self.in_dir, 'warp*.' + ext) for ext in extensions]
        image_paths = sorted(sum(map(glob, search_paths), []))
        self.warps = [cv2.imread(f) for f in image_paths]

    def calc_offsets(self, obj_dim):
        offset_y = (self.out_dim[0] - obj_dim)/2
        offset_x = (self.out_dim[1] - obj_dim)/2
        if offset_y < 0:
            offset_y = 0
        if offset_x < 0:
            offset_x = 0
        return offset_y, offset_x

    def flicker_grid(self, obj_dim, duration, frate, start_blank=True):
        adj_warps = []
        for warp in self.warps:
            adj = resize_image(warp, (obj_dim, obj_dim))
            adj_warps.append(adj)
        offset_y, offset_x = self.calc_offsets(obj_dim)
        squares = random.sample(adj_warps, 5)
        frames_generated = 0
        writer = cv2.VideoWriter(os.path.join(self.out_dir, 'flicker_grid.avi'),
                                 cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 frate, (self.out_dim[1], self.out_dim[0]))
        while frames_generated < duration * frate:
            if start_blank and frames_generated == 0:
                frame = np.zeros(self.out_dim)
            else:
                frame = set_squares(np.zeros(self.out_dim), (offset_x, offset_y),
                                    squares, None, 1)
            writer.write(np.uint8(frame))
            frames_generated += 1
        writer.release()

    def melt_image(self, obj_dim, duration, frate, start_blank=False):
        adj_warps = []
        for warp in self.warps:
            adj = resize_image(warp, (obj_dim, obj_dim))
            adj_warps.append(adj)
        offset_y, offset_x = self.calc_offsets(obj_dim)
        frames_generated = 0
        writer = cv2.VideoWriter(os.path.join(self.out_dir, 'melt.avi'),
                                 cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 frate, (self.out_dim[1], self.out_dim[0]))
        while frames_generated < duration * frate:
            square = random.sample(adj_warps, 1)[0]
            frame = np.zeros(self.out_dim)
            if frames_generated > 0 or not start_blank:
                reflected = cv2.copyMakeBorder(square, 0, 0, offset_x, offset_x,
                                               cv2.BORDER_REFLECT_101)
                frame[offset_y:offset_y+obj_dim, :, :] = reflected
                melt_frames = melt(frame, 50, (0, offset_y), reflected.shape, 1000)
                for iteration in melt_frames:
                    writer.write(np.uint8(iteration))
                    frames_generated += 1
        writer.release()

    def vdist_image(self, obj_dim, duration, frate, dfactor, out_name='v_distort.avi', start_blank=False):
        adj_warps = []
        for warp in self.warps:
            adj = resize_image(warp, (obj_dim, obj_dim))
            adj_warps.append(adj)
        offset_y, offset_x = self.calc_offsets(obj_dim)
        frames_generated = 0
        writer = cv2.VideoWriter(os.path.join(self.out_dir, out_name),
                                 cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),
                                 frate, (self.out_dim[1], self.out_dim[0]))
        while frames_generated < duration * frate:
            square = random.sample(adj_warps, 1)[0]
            frame = np.zeros(self.out_dim)
            if frames_generated > 0 or not start_blank:
                reflected = cv2.copyMakeBorder(square, 0, 0, offset_x, offset_x,
                                               cv2.BORDER_REFLECT_101)
                frame[offset_y:offset_y+obj_dim, :, :] = reflected
                distort_frames = vertical_distort(frame, 8, random.randint(dfactor[0], dfactor[1]))
                for iteration in distort_frames:
                    writer.write(np.uint8(iteration))
                    frames_generated += 1
        writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate texture visualization.')
    parser.add_argument('set', help='Set name for directory reference.')
    parser.add_argument('--outdim', help='Dimensions of output frames.', default=(1080, 1920, 3))
    args = parser.parse_args()
    vis = Visualize(args.set, args.outdim)
    vis.vdist_image(int(1080*.8), 10, 30, (0, 10), 'v_distort3.avi')
    # vis.vdist_image(int(1080*.8), 120, 40, (10, 100), 'v_distort2.avi')
