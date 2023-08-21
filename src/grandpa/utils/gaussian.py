import math
from copy import deepcopy

import cv2
import numpy as np

from . import bounding_box


def gaussian_2d():
    """
    Create a 2-dimensional isotropic Gaussian map.
    :return: a 2D Gaussian map. 1000x1000.
    """
    mean = 0
    radius = 1.4
    a = 1.
    x0, x1 = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01))
    x = np.append([x0.reshape(-1)], [x1.reshape(-1)], axis=0).T

    m0 = (x[:, 0] - mean) ** 2
    m1 = (x[:, 1] - mean) ** 2
    gaussian_map = a * np.exp(-0.5 * (m0 + m1) / (radius ** 2))
    gaussian_map = gaussian_map.reshape(len(x0), len(x1))

    max_prob = np.max(gaussian_map)
    min_prob = np.min(gaussian_map)
    gaussian_map = (gaussian_map - min_prob) / (max_prob - min_prob)
    gaussian_map = np.clip(gaussian_map, 0., 1.)
    return gaussian_map


class GaussianGenerator:
    def __init__(self):
        self.gaussian_img = gaussian_2d()
        self.gaussian_maps = {1000: self.gaussian_img}

    @staticmethod
    def perspective_transform(src, dst_shape, dst_points):
        """
        Perspective Transform
        :param src: Image to transform.
        :param dst_shape: Tuple of 2 intergers(rows and columns).
        :param dst_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        :return: Image after perspective transform.
        """
        img = src.copy()
        h, w = img.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32(dst_points)
        perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)
        perspective_mat = np.array(perspective_mat)
        dst = cv2.warpPerspective(img, perspective_mat, (dst_shape[1], dst_shape[0]),
                                  borderValue=0, borderMode=cv2.BORDER_CONSTANT)
        dst = np.array(dst)
        return dst

    @staticmethod
    def normalize(img: np.ndarray):
        factor = 1 / img.max()
        return img * factor

    def get_gaus_map(self, shape):
        gaus_map_size = max(shape)
        if gaus_map_size not in self.gaussian_maps:
            gaus_map = self.normalize(cv2.resize(deepcopy(self.gaussian_img),
                                                                          (gaus_map_size, gaus_map_size)))
            self.gaussian_maps[gaus_map_size] = np.array(gaus_map)
        ret_value = cv2.resize(deepcopy(self.gaussian_maps[gaus_map_size]), shape, interpolation=cv2.INTER_LINEAR)
        ret_value = np.array(ret_value)
        return ret_value

    def gen(self, score_shape, points_list):
        score_map = np.zeros(score_shape, dtype=np.float32)
        # TODO: Use polygons
        boxes = [bounding_box.four_point_bb_to_2_point(points, round_values=True) for points in points_list]
        for box in boxes:
            point1 = [int(box[0]), int(box[1])]
            point2 = [int(box[2]), int(box[3])]
            for i in range(2):
                if point1[i] < 0:
                    point1[i] = 0
                if point2[i] < 0:
                    point2[i] = 0

            # assure not out of bounds and not identical start and end points
            point1[0] = np.clip(point1[0], 0, score_map.shape[0] - 1)
            point1[1] = np.clip(point1[1], 0, score_map.shape[1] - 1)
            point2[0] = np.clip(point2[0], 0, score_map.shape[0] - 1)
            point2[1] = np.clip(point2[1], 0, score_map.shape[1] - 1)

            x_points = np.array([point1[0], point2[0]])
            y_points = np.array([point1[1], point2[1]])
            if np.all(x_points >= score_map.shape[0]-2) or np.all(x_points <= 0) \
                    or np.all(y_points >= score_map.shape[1]-2) or np.all(y_points <= 0)\
                    or x_points[0] == x_points[1] or y_points[0] == y_points[1]:
                continue

            if min((point2[0] - point1[0], point2[1] - point1[1])) <= 3:
                if point1[0] - 1 > 0:
                    point1[0] -= 1
                if point1[1] - 1 > 0:
                    point1[1] -= 1
                if point2[0] + 1 < score_shape[0]:
                    point2[0] += 1
                if point2[1] + 1 < score_shape[1]:
                    point2[1] += 1

            # do gaussian transform
            img_section = score_map[point1[0]:point2[0], point1[1]:point2[1]]
            temp_gaus_map = self.get_gaus_map((img_section.shape[1], img_section.shape[0]))
            score_map[point1[0]:point2[0], point1[1]:point2[1]] = np.where(
                temp_gaus_map > img_section,
                temp_gaus_map,
                img_section
            )
        score_map = np.clip(score_map, 0, 1.)
        return score_map


def cal_quadrangle_area(points):
    """
    Calculate the area of quadrangle.
    :return: The area of quadrangle.
    """
    points = reorder_points(points)
    p1, p2, p3, p4 = points
    s1 = cal_triangle_area(p1, p2, p3)
    s2 = cal_triangle_area(p3, p4, p1)
    s3 = cal_triangle_area(p2, p3, p4)
    s4 = cal_triangle_area(p4, p1, p2)
    if s1 + s2 == s3 + s4:
        return s1 + s2
    else:
        return 0


def reorder_points(point_list):
    """
    Reorder points of quadrangle.
    (top-left, top-right, bottom right, bottom left).
    :param point_list: List of point. Point is (x, y).
    :return: Reorder points.
    """
    # Find the first point which x is minimum.
    if len(point_list) == 3:
        point_list = bounding_box.convert_triangle_to_quadrangle(point_list)
    elif len(point_list) == 2:
        point_list = bounding_box.two_point_bb_to_4_point(point_list)
    elif len(point_list) > 4:
        point_list = bounding_box.convert_polygon_box_to_4_point_bb(point_list)
    elif len(point_list) < 2:
        raise ValueError('Box passed to reorder points must have at least 2 points.')
    ordered_point_list = sorted(point_list, key=lambda x: (x[0], x[1]))
    first_point = ordered_point_list[0]

    # Find the third point. The slope is middle.
    slope_list = [[cal_slope(first_point, p), p] for p in ordered_point_list[1:]]
    ordered_slope_point_list = sorted(slope_list, key=lambda x: x[0])
    first_third_slope, third_point = ordered_slope_point_list[1]

    # Find the second point which is above the line between the first point and the third point.
    # All that's left is the fourth point.
    if above_line(ordered_slope_point_list[0][1], third_point, first_third_slope):
        second_point = ordered_slope_point_list[0][1]
        fourth_point = ordered_slope_point_list[2][1]
        reverse_flag = False
    else:
        second_point = ordered_slope_point_list[2][1]
        fourth_point = ordered_slope_point_list[0][1]
        reverse_flag = True

    # Find the top left point.
    second_fourth_slope = cal_slope(second_point, fourth_point)
    if first_third_slope < second_fourth_slope:
        if reverse_flag:
            reorder_point_list = [fourth_point, first_point, second_point, third_point]
        else:
            reorder_point_list = [second_point, third_point, fourth_point, first_point]
    else:
        reorder_point_list = [first_point, second_point, third_point, fourth_point]

    return reorder_point_list


def cal_triangle_area(p1, p2, p3):
    """
    Calculate the area of triangle.
    S = |(x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1)| / 2
    :param p1: (x, y)
    :param p2: (x, y)
    :param p3: (x, y)
    :return: The area of triangle.
    """
    [x1, y1], [x2, y2], [x3, y3] = p1, p2, p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2


def cal_slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-5)


def above_line(p, start_point, slope):
    y = (p[0] - start_point[0]) * slope + start_point[1]
    return p[1] < y


def calc_affinities_from_batch(batch):
    return [get_img_aff_boxes(batch, img_num) for img_num in range(len(batch))]


def get_img_aff_boxes(batch, img_num):
    img_aff_boxes = []
    boxes = batch[img_num]["annotations"]
    hierarchy = batch[img_num]["hierarchy"]
    for word in hierarchy:
        char_boxes = get_boxes_by_id(boxes, word['to'])
        if len(char_boxes) < 2:
            continue
        aff_boxes = cal_affinity_boxes(char_boxes)
        img_aff_boxes.extend(aff_boxes)
    return img_aff_boxes


def get_boxes_by_id(boxes, ids):
    boxes_new = []
    for b_id in ids:
        for box in boxes:
            if box['id'] == b_id:
                boxes_new.append(bounding_box.convert_polygon_box_to_4_point_bb(box['box']))
                break
    return boxes_new


def cal_affinity_boxes(char_boxes):
    ordered_char_boxes = [reorder_points(cb) for cb in char_boxes]
    ordered_char_boxes = reorder_box(ordered_char_boxes)

    fix_box_dimensions(ordered_char_boxes)

    point_pairs_list = [cal_point_pairs(bbox) for bbox in ordered_char_boxes]
    aff_boxes = []
    for i in range(len(point_pairs_list) - 1):
        affinity_box = cal_affinity_box(point_pairs_list[i], point_pairs_list[i + 1])
        aff_boxes.append(affinity_box)

    return aff_boxes


def fix_box_dimensions(ordered_char_boxes):
    """
    Makes sure the box is at least 1 pixel in each direction to prevent bugs.

    Args:
        ordered_char_boxes: List of ordered char boxes to check.

    Returns:
        None, changes original list.
    """
    for box in ordered_char_boxes:
        if all([b[0] == box[0][0] for b in box]):
            box[1][0] += 1
            box[2][0] += 1
        if all([b[1] == box[0][1] for b in box]):
            box[2][1] += 1
            box[3][1] += 1


def reorder_box(box_list):
    """
    Reorder character boxes.
    :param box_list: List of box. Box is a list of point. Point is (x, y).
    :return: Reorder boxes.
    """
    # Calculate the minimum distance between any two boxes.
    box_count = len(box_list)
    distance_mat = np.zeros((box_count, box_count), dtype=np.float32)
    for i in range(box_count):
        box1 = box_list[i]
        for j in range(i + 1, box_count):
            box2 = box_list[j]
            distance = cal_min_box_distance(box1, box2)
            distance_mat[i][j] = distance
            distance_mat[j][i] = distance

    # Find the boxes on the both ends.
    end_box_index = np.argmax(distance_mat)
    nan = distance_mat[end_box_index // box_count, end_box_index % box_count] + 1
    for i in range(box_count):
        distance_mat[i, i] = nan
    last_box_index = start_box_index = end_box_index // box_count
    last_box = box_list[start_box_index]

    # reorder box.
    reordered_box_list = [last_box]
    for i in range(box_count - 1):
        distance_mat[:, last_box_index] = nan
        closest_box_index = np.argmin(distance_mat[last_box_index])
        reordered_box_list.append(box_list[closest_box_index])
        last_box_index = closest_box_index

    return reordered_box_list


def cal_point_pairs(points):
    intersection = intersection_of_diagonals(points)
    p1, p2, p3, p4 = points
    point_pairs = [[cal_center_point([p1, p2, intersection]), cal_center_point([p3, p4, intersection])],
                   [cal_center_point([p2, p3, intersection]), cal_center_point([p4, p1, intersection])]]
    return point_pairs


def cal_min_box_distance(box1, box2):
    box_distance = [math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2) for p1 in box1 for p2 in box2]
    return np.min(box_distance)


def intersection_of_diagonals(points):
    """
    Calculate the intersection of diagonals.
    x=[(x3-x1)(x4-x2)(y2-y1)+x1(y3-y1)(x4-x2)-x2(y4-y2)(x3-x1)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)]
    y=(y3-y1)[(x4-x2)(y2-y1)+(x1-x2)(y4-y2)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)]+y1
    :param points: (x1, y1), (x2, y2), (x3, y3), (x4, y4).
    :return: (x, y).
    """
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = points
    x = ((x3 - x1) * (x4 - x2) * (y2 - y1) + x1 * (y3 - y1) * (x4 - x2) - x2 * (y4 - y2) * (x3 - x1)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + 1e-5)
    y = (y3 - y1) * ((x4 - x2) * (y2 - y1) + (x1 - x2) * (y4 - y2)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + 1e-5) + y1
    return [x, y]


def cal_center_point(points):
    points = np.array(points)
    return [round(np.average(points[:, 0])), round(np.average(points[:, 1]))]


def cal_affinity_box(point_pairs1, point_pairs2):
    areas = [cal_quadrangle_area([p1, p2, p3, p4]) for p1, p2 in point_pairs1 for p3, p4 in point_pairs2]
    max_area_index = np.argmax(areas)
    affinity_box = [point_pairs1[max_area_index // 2][0],
                    point_pairs1[max_area_index // 2][1],
                    point_pairs2[max_area_index % 2][0],
                    point_pairs2[max_area_index % 2][1]]
    return np.int32(affinity_box)