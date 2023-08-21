import math

import numpy as np


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_points(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def angle_relative_to_180(angle):
    if angle < 0:
        return -180 - angle
    else:
        return 180 - angle


def angle(pt1,pt2):
    m1 = (pt1.getY() - pt1.getY())/1
    m2 = (pt2.getY() - pt1.getY())/(pt2.getX()-pt1.getX())

    tnAngle = (m1-m2)/(1+(m1*m2))
    return math.atan(tnAngle)