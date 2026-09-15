# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 ByteDance and/or its affiliates.
# Copyright (c) 2026 Aureka AI Research
import numpy as np
from scipy.spatial.transform import Rotation


def angle_3p(a, b, c, *, eps: float = 0.0) -> float:
    """Calculate the angle between three points in a 2D space.

    Args:
        a (list or array-like): The coordinates of the first point.
        b (list or array-like): The coordinates of the second point.
        c (list or array-like): The coordinates of the third point.

    Returns:
        float: The angle in degrees (0, 180) between the vectors
               from point a to point b and point b to point c.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    ab = b - a
    bc = c - b

    dot_product = np.dot(ab, bc)

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)

    denom = norm_ab * norm_bc
    if eps == 0.0 and np.isclose(denom, 0.0):
        return 0.0

    cos_theta = np.clip(dot_product / (denom + eps), -1, 1)
    theta_radians = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta_radians)
    return theta_degrees


def random_transform(
    points: np.ndarray,
    max_translation: float = 1.0,
    apply_augmentation: bool = False,
    centralize: bool = True,
    *,
    translation_before_rotation: bool = False,
) -> np.ndarray:
    """Randomly transform a set of 3D points.

    Args:
        points (numpy.ndarray): The points to be transformed, shape=(N, 3)
        max_translation (float): The maximum translation value. Default is 1.0.
        apply_augmentation (bool): Whether to apply random rotation/translation on
            ref_pos

    Returns:
        numpy.ndarray: The transformed points.
    """
    if centralize:
        points = points - points.mean(axis=0)
    if not apply_augmentation:
        return points
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    R = Rotation.random().as_matrix()
    if translation_before_rotation:
        return np.dot(points + translation, R.T)
    transformed_points = np.dot(points, R.T) + translation
    return transformed_points
