# -*- coding: utf-8 -*-

import numpy as np
from permutohedral_lattice import permutohedral_lattice_filter

def _compute_spatial_kernel(height, width, theta_gamma):
    positions = np.zeros((height, width, 2), dtype='float32')
    for r in range(height):
        for c in range(width):
            positions[r, c, 0] = c / theta_gamma
            positions[r, c, 1] = r / theta_gamma

    return positions

def _compute_bilateral_kernel(img, theta_alpha, theta_beta):
    height, width = img.shape[0], img.shape[1]
    positions = np.zeros((height, width, 5), dtype='float32')
    for r in range(height):
        for c in range(width):
            positions[r, c, 0] = c / theta_alpha
            positions[r, c, 1] = r / theta_alpha
            positions[r, c, 2] = img[r, c, 0] / theta_beta
            positions[r, c, 3] = img[r, c, 1] / theta_beta
            positions[r, c, 4] = img[r, c, 2] / theta_beta

    return positions

def spatial_high_dim_filter(inp, theta_gamma):
    height, width, _ = inp.shape
    
    print('Computing spatial kernel...')
    positions = _compute_spatial_kernel(height, width, theta_gamma)
    print('Spatial kernel computed.')
    
    print('High order filtering...')
    out = permutohedral_lattice_filter(inp, positions)
    print('High order filtered.')
    
    return out

def bilateral_high_dim_filter(inp, img, theta_alpha, theta_beta):
    print('Computing bilateral kernel...')
    positions = _compute_bilateral_kernel(img, theta_alpha, theta_beta)
    print('Bilateral kernel filtered.')

    print('High order filtering...')
    out = permutohedral_lattice_filter(inp, positions)
    print('High order filtered.')
    
    return out