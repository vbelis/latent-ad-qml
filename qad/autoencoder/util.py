import numpy as np


def get_mean(data):
    """data shape (N, num_particles, features)"""
    mean = np.nanmean(data, axis=(0, 1))
    return mean


def get_std(data):
    """data shape (N, num_particles, features)"""
    std = np.nanstd(data, axis=(0, 1))
    std[std == 0.0] = 0.001
    return std
