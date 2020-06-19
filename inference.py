# -*- coding: utf-8 -*-

import numpy as np
from high_dim_filter_loader import spatial_high_dim_filter, bilateral_high_dim_filter
from PIL import Image
import matplotlib.pyplot as plt
from cv2 import cv2

def unary_from_labels(labels, n_labels, gt_prob, zero_unsure=True):
    """
    Simple classifier that is 50% certain that the annotation is correct.
    (same as in the inference example).


    Parameters
    ----------
    labels: numpy.array
        The label-map, i.e. an array of your data's shape where each unique
        value corresponds to a label.
    n_labels: int
        The total number of labels there are.
        If `zero_unsure` is True (the default), this number should not include
        `0` in counting the labels, since `0` is not a label!
    gt_prob: float
        The certainty of the ground-truth (must be within (0,1)).
    zero_unsure: bool
        If `True`, treat the label value `0` as meaning "could be anything",
        i.e. entries with this value will get uniform unary probability.
        If `False`, do not treat the value `0` specially, but just as any
        other class.
    """
    assert 0 < gt_prob < 1, "`gt_prob must be in (0,1)."

    labels = labels.flatten()

    n_energy = -np.log((1.0 - gt_prob) / (n_labels - 1))
    p_energy = -np.log(gt_prob)

    # Note that the order of the following operations is important.
    # That's because the later ones overwrite part of the former ones, and only
    # after all of them is `U` correct!
    U = np.full((n_labels, len(labels)), n_energy, dtype='float32')
    U[labels - 1 if zero_unsure else labels, np.arange(U.shape[1])] = p_energy

    # Overwrite 0-labels using uniform probability, i.e. "unsure".
    if zero_unsure:
        U[:, labels == 0] = -np.log(1.0 / n_labels)

    return U

def _diagonal_compatibility(shape):
    return np.eye(shape[0], shape[1], dtype=np.float32)

def _potts_compatibility(shape):
    return -1 * _diagonal_compatibility(shape)

def _softmax(x):
    e_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def inference(image, unary, num_classes, theta_alpha, theta_beta, theta_gamma, spatial_compat, bilateral_compat, num_iterations):
    spatial_weights = spatial_compat * _diagonal_compatibility((num_classes, num_classes))
    bilateral_weights = bilateral_compat * _diagonal_compatibility((num_classes, num_classes))
    compatibility_matrix = _potts_compatibility((num_classes, num_classes))

    height, width, _ = image.shape
    all_ones = np.ones((height, width, num_classes), dtype=np.float32)

    spatial_norm_vals = spatial_high_dim_filter(all_ones, theta_gamma)
    bilateral_norm_vals = bilateral_high_dim_filter(all_ones, image, theta_alpha, theta_beta)

    # Initialize Q
    Q = _softmax(-unary)

    for i in range(num_iterations):
        print('iter {}'.format(i))
        tmp1 = -unary

        # Message passing - spatial
        spatial_out = spatial_high_dim_filter(Q, theta_gamma)
        spatial_out /= spatial_norm_vals

        # Message passing - bilateral
        bilateral_out = bilateral_high_dim_filter(Q, image, theta_alpha, theta_beta)
        bilateral_out /= bilateral_norm_vals

        # Message passing
        message_passing = spatial_out.reshape((-1, num_classes)).dot(spatial_weights) + bilateral_out.reshape((-1, num_classes)).dot(bilateral_weights)
        
        # Compatibility transform
        pairwise = message_passing.dot(compatibility_matrix)
        pairwise = pairwise.reshape((height, width, num_classes))
        
        # Local update
        tmp1 -= pairwise
        
        # Normalize
        Q = _softmax(tmp1)

    return Q


if __name__ == "__main__":
    img = cv2.imread('examples/im3.png')
    anno_rgb = cv2.imread('examples/anno3.png').astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    colors, labels = np.unique(anno_lbl, return_inverse=True)

    HAS_UNK = 0 in colors
    if HAS_UNK:
        print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    unary = unary_from_labels(labels, n_labels, 0.7, HAS_UNK)
    unary = np.rollaxis(unary.reshape(n_labels, *img.shape[:2]), 0, 3)

    pred = inference(img, unary, n_labels, theta_alpha=80., theta_beta=13., theta_gamma=3., spatial_compat=3., bilateral_compat=10., num_iterations=10)
    
    np.savez('pred.npz', pred)
    MAP = np.argmax(pred, axis=-1)
    plt.imshow(MAP)
    plt.show()