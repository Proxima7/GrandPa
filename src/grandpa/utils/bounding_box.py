import numpy as np
from enum import Enum
import cv2

from .math import angle_between_points, angle_relative_to_180
from .standard import tree_search, cal_distance


class BboxFormat(Enum):
    """
    Class used to reference the bounding box bbox_format.
    """
    FOUR_POINT = 0
    TWO_POINT = 1
    XY_WIDTH_HEIGHT = 2


def cut_out_image_area(image, area_bbox):
    """
    Cuts out the specified area from an image and returns it.

    Args:
        image: The original image.
        area_bbox: The area to cut out.

    Returns:
        Cut out area as image.
    """
    max_height, max_width, ordered_bbox = get_area_dimensions(area_bbox)
    # now that we have the dimensions of the new image, construct the set of destination points to obtain a
    # "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right,
    # bottom-right, and bottom-left order
    destination_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    transform_matrix = cv2.getPerspectiveTransform(ordered_bbox, destination_points)
    transform_matrix = np.array(transform_matrix)
    cut_out_image = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    cut_out_image = np.array(cut_out_image)
    # return the cut out image
    return cut_out_image


def get_area_dimensions(area_bbox):
    """
    Gets the area dimension for the given area bbox.

    Args:
        area_bbox: 4 point bbox.

    Returns:
        max_height: Maximum height of the area.
        max_width: Maximum width of the area.
        ordered_bbox: Reordered bounding box as [TL, TR, BR, BL]
    """
    # obtain a consistent order of the points and unpack them individually
    area_bbox = np.array(area_bbox)
    ordered_bbox = order_points(area_bbox)
    (tl, tr, br, bl) = ordered_bbox
    # compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-bbox
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_bottom), int(width_top))
    # compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
    # y-bbox or the top-left and bottom-left y-bbox
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_right), int(height_left))
    return max_height, max_width, ordered_bbox


def four_point_bb_to_2_point(bbox, bbox_format="y1x1y2x2", round_values=False):
    """
    Converts a four point bbox to a two point bbox.

    Args:
        bbox: Bounding box in four point format.
        bbox_format: The format of the bbox as string (for example "y1x1y2x2").

    Returns:
        Two point bbox.
    """
    min_x = np.min([point[0] for point in bbox])
    min_y = np.min([point[1] for point in bbox])
    max_x = np.max([point[0] for point in bbox])
    max_y = np.max([point[1] for point in bbox])

    if round:
        min_x = round(min_x)
        min_y = round(min_y)
        max_x = round(max_x)
        max_y = round(max_y)

    if bbox_format == "y1x1y2x2":
        return [min_y, min_x, max_y, max_x]
    return [min_x, min_y, max_x, max_y]


def two_point_bb_to_4_point(bbox):
    """
    Converts a two point bounding box to a four point bounding box.

    Args:
        bbox: Two point bounding box.

    Returns:
        Four point bbox.
    """
    # convert from (x1,y1,x2,y2) to (x1y1, x2y2)
    if len(bbox) == 4:
        bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]

    # conversion matches for both formats (x1y1..) also for y1x1y2x2
    x1y1 = bbox[0]
    x4y4 = bbox[1]

    x2y2 = [x4y4[0], x1y1[1]]
    x3y3 = [x1y1[0], x4y4[1]]
    return [x1y1, x2y2, x3y3, x4y4]


def xy_width_height_to_two_point(bbox: list):
    """
    Converts a bounding box with shape [x1,y2,width,height] to a two point bounding box.

    Args:
        bbox: bbox with shape [x1,y2,width,height].

    Returns: bbox with shape [x1y1, x2y2].
    """
    x1y1 = [bbox[0], bbox[1]]
    x2y2 = [bbox[0] + bbox[2], bbox[1] + bbox[3]]
    return [x1y1, x2y2]


def xy_width_height_to_four_point(bbox: list):
    """
    Converts a bounding box with shape [x1,y2,width,height] to a four point bounding box.

    Args:
        bbox: bbox with shape [x1,y2,width,height].

    Returns: four point bbox.
    """
    bbox = xy_width_height_to_two_point(bbox)
    return two_point_bb_to_4_point(bbox)


def bbox_conversion(bbox, bbox_format: BboxFormat, target_bbox_format: BboxFormat) -> np.array:
    """
    This method converts a bounding box to a different bounding box bbox_format.

    Args:
        bbox: the bounding box with the bbox_format of bbox_format
        bbox_format: bbox_format of the bounding box
        target_bbox_format: bbox_format to convert the bounding box to

    Returns: the converted bounding box

    """
    # no conversion necessary
    if bbox_format == target_bbox_format:
        return np.array(bbox)
    # two point to four point
    if bbox_format == BboxFormat.TWO_POINT and target_bbox_format == BboxFormat.FOUR_POINT:
        return np.array(two_point_bb_to_4_point(bbox))
    # point + width, height to four point
    if bbox_format == BboxFormat.XY_WIDTH_HEIGHT and target_bbox_format == BboxFormat.FOUR_POINT:
        return np.array(xy_width_height_to_four_point(bbox))
    # four point to two point
    if bbox_format == BboxFormat.FOUR_POINT and target_bbox_format == BboxFormat.TWO_POINT:
        return np.array(four_point_bb_to_2_point(bbox))


def rotate_bounding_box(bbox, angle):
    """
    Method to rotate the bounding box by a certain angle.

    Args:
        bbox: bounding box in either two point or four point format.
        angle: Angle by which the bounding box should be rotated.

    Returns:
        List containing the 4 points of the bounding box [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    bbox, center = get_4_point_bbox_with_center(bbox)
    return get_rotated_bbox(angle, bbox, center)


def get_rotated_bbox(angle, bbox, center):
    """
    Rotates the coordinates by the given angle around the given center.

    Args:
        angle: Angle to rotate by.
        bbox: bounding box.
        center: Center of the rotation as (x, y).

    Returns:
        Rotated bounding box.
    """
    x_center, y_center = center
    rotated_bbox = []
    for point in bbox:
        temp_x = point[0] - x_center
        temp_y = point[1] - y_center
        rotated_x = temp_x * np.cos(angle) - temp_y * np.sin(angle)
        rotated_y = temp_x * np.sin(angle) - temp_y * np.cos(angle)
        final_x = rotated_x + x_center
        final_y = rotated_y + y_center
        rotated_bbox.append([final_x, final_y])
    return rotated_bbox


def get_4_point_bbox_with_center(bbox, center=True):
    """
    Gets the 4 point bbox and its center from the bbox.

    Args:
        bbox: bbox in either two or four point format.
        center: Whether to calculate the center or not (to save performance).

    Returns:
        Four_point_bbox: BBox in four point format.
        Center: Center of the four point bbox if center=True else None.
    """
    x_max, x_min, y_max, y_min = get_x_y_min_max(bbox)
    four_point_bbox = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    if center:
        x_center = ((x_max - x_min) / 2) + x_min
        y_center = ((y_max - y_min) / 2) + y_min
        return four_point_bbox, (x_center, y_center)
    else:
        return four_point_bbox, None


def get_x_y_points(bbox):
    """
    Gets the x and y coordinates of the bbox.

    Args:
        bbox: Bounding box in any [(x, y), (x, y), ...] format.

    Returns:
        x_points: X points of the bbox.
        y_points: Y points of the bbox.
    """
    x_points = []
    y_points = []
    for coord in bbox:
        x_points.append(coord[0])
        y_points.append(coord[1])
    return x_points, y_points


def convert_polygon_box_to_4_point_bb(bbox):
    """
    Method to convert a polygon box to a 4 point bbox.

    Args:
        bbox: [[x1, y1], [x2, y2], ..., [xn, yn]]

    Returns: 4-point BBox
    """
    bbox, _ = get_4_point_bbox_with_center(bbox, center=False)
    return bbox

def get_x_y_min_max(bbox):
    """
    Gets the x and y min and max values of the bbox.

    Args:
        bbox: Bounding box in format [[x1, y2], [x2, y2], ..., [xn, yn]]

    Returns:
        x_max: max x value
        x_min: min x value
        y_max: max y value
        y_min: min y value
    """
    x_points, y_points = get_x_y_points(bbox)
    x_max = np.max(x_points)
    x_min = np.min(x_points)
    y_max = np.max(y_points)
    y_min = np.min(y_points)
    return x_max, x_min, y_max, y_min

def flip_bbox(bbox, img_shape, direction):
    """
    Flips a 4 point bounding box in a given direction
    Args:
        bbox: source bounding box
        img_shape: shape of the associated image
        direction: left to right as leftright, top to bottom as topbottom, both as both

    Returns: flipped bounding box

    """
    h, w = img_shape[:2]
    center_x, center_y = int(w/2), int(h/2)
    if direction == "leftright":
        for i in range(len(bbox)):
            bbox[i][0] -= 2 * (bbox[i][0] - center_x)
    elif direction == "topbottom":
        for i in range(len(bbox)):
            bbox[i][1] -= 2 * (bbox[i][1] - center_y)
    elif direction == "both":
        bbox = flip_bbox(bbox, img_shape, "leftright")
        bbox = flip_bbox(bbox, img_shape, "topbottom")
    else:
        raise ValueError("Direction is not supported.")

    return bbox


def to_radian(angle):
    """
    Helper function to convert degree angles into radians.
    Args:
        angle: degree angle

    Returns: radian angle

    """
    return (np.math.pi / 180) * angle


def rotate_bbox_around_center(bbox, cx, cy, angle):
    """
    Rotates a 4 point bounding box around a center point.
    Args:
        bbox: bounding box to rotate
        cx: center x
        cy: center y
        angle: rotation angle in degrees

    Returns: rotated 4 point bounding box as int32

    """
    for i in range(len(bbox)):
        radian = to_radian(angle)
        s = np.math.sin(radian)
        c = np.math.cos(radian)
        px, py = bbox[i]
        xnew = cx + (px - cx) * c - (py - cy) * s
        ynew = cx + (px - cx) * s + (py - cy) * c
        bbox[i] = [xnew, ynew]
    return np.array(bbox).astype(np.int32)


def rescale_bbox(bbox, scale=(1.0, 1.0), offset=(0, 0), skip_none=False):
    """
    Custom function to rescale bounding box.
    Args:
        bbox: source bounding box
        scale: scale factor (xy tuple)
        offset: optional offset for bounding boxes (xy tuple)
        skip_none: skip the function if bbox is none

    Returns: rescaled bounding boxes

    """
    if skip_none and bbox is None:
        return None
    return np.add(np.array(bbox) * scale, offset)


def get_matching_bounding_boxes(bboxes, w, h):
    """
    Removes bounding boxes which are out of range.
    Args:
        bboxes: List of bounding boxes
        w: new image width
        h: new image height

    Returns: prepared list of bboxes

    """
    res = []
    for bbox in bboxes:
        x0 = np.amin([b[0] for b in bbox])
        x1 = np.amax([b[0] for b in bbox])
        y0 = np.amin([b[1] for b in bbox])
        y1 = np.amax([b[1] for b in bbox])

    return np.array([bb for bb in bboxes
                     if np.amin([b[0] for b in bb]) >= 0
                     and np.amax([b[0] for b in bb]) <= w
                     and np.amin([b[1] for b in bb]) >= 0
                     and np.amax([b[1] for b in bb]) <= h])


def convert_triangle_to_quadrangle(box):
    two_point_box = four_point_bb_to_2_point(box)
    return two_point_bb_to_4_point(two_point_box)


def order_points(polygon, clockwise=True, always_return=True):
    if type(polygon) == np.ndarray:
        polygon = polygon.tolist()
    sums = [sum(p) for p in polygon]
    min_v = min(sums)
    min_index = sums.index(min_v)
    set_points = [polygon[min_index]]
    del polygon[min_index]
    if clockwise:
        return tree_search(polygon, set_points, overall_angle, -360, always_return)
    else:
        return tree_search(polygon, set_points, overall_angle, 360, always_return)


def overall_angle(box):
    angles = [angle_relative_to_180(angle_between_points(box[i-1], box[i], box[i+1])) for i in range(len(box[:-1]))]
    angles.append(angle_relative_to_180(angle_between_points(box[-2], box[-1], box[0])))
    return round(sum(angles))


def point_in_box(point, box):
    x_max, x_min, y_max, y_min = get_x_y_min_max(box)
    if point[0] < x_min or point[0] > x_max or point[1] < y_min or point[1] > y_max:
        return False

    box = np.array(order_points(box, always_return=True))
    point = np.array(point)

    # Point calc
    tr1 = 0.5 * cal_distance(box[0], box[3]) * np.linalg.norm(np.cross(box[3] - box[0], box[0] - point)) / \
          np.linalg.norm(box[3] - box[0])
    tr2 = 0.5 * cal_distance(box[3], box[2]) * np.linalg.norm(np.cross(box[2] - box[3], box[3] - point)) / \
          np.linalg.norm(box[2] - box[3])
    tr3 = 0.5 * cal_distance(box[2], box[1]) * np.linalg.norm(np.cross(box[1] - box[2], box[2] - point)) / \
          np.linalg.norm(box[1] - box[2])
    tr4 = 0.5 * cal_distance(box[1], box[0]) * np.linalg.norm(np.cross(box[0] - box[1], box[1] - point)) / \
          np.linalg.norm(box[0] - box[1])

    avg_len = (cal_distance(box[1], box[0]) + cal_distance(box[3], box[2])) / 2
    avg_height = (cal_distance(box[3], box[0]) + cal_distance(box[2], box[1])) / 2
    point = np.array([box[0][0] + avg_len, box[0][1] + avg_height])
    tra1 = 0.5 * cal_distance(box[0], box[3]) * np.linalg.norm(np.cross(box[3] - box[0], box[0] - point)) / \
          np.linalg.norm(box[3] - box[0])
    tra2 = 0.5 * cal_distance(box[3], box[2]) * np.linalg.norm(np.cross(box[2] - box[3], box[3] - point)) / \
          np.linalg.norm(box[2] - box[3])
    tra3 = 0.5 * cal_distance(box[2], box[1]) * np.linalg.norm(np.cross(box[1] - box[2], box[2] - point)) / \
          np.linalg.norm(box[1] - box[2])
    tra4 = 0.5 * cal_distance(box[1], box[0]) * np.linalg.norm(np.cross(box[0] - box[1], box[1] - point)) / \
          np.linalg.norm(box[0] - box[1])

    quadrangle_area = sum([tra1, tra2, tra3, tra4])
    calc_p_area = sum([tr1, tr2, tr3, tr4])

    return quadrangle_area >= calc_p_area


def move_in_any_box(point, boxes):
    point = [point[0], point[1]]
    distances = [get_4_point_bbox_with_center(box)[1] for box in boxes]
    box = boxes[distances.index(min(distances))]
    x_max, x_min, y_max, y_min = get_x_y_min_max(box)
    if point[0] < x_min or point[0] > x_max:
        point[0] = x_min if abs(x_min - point[0]) < abs(x_max - point[0]) else x_max
    if point[1] < y_min or point[1] > y_max:
        point[1] = y_min if abs(y_min - point[1]) < abs(y_max - point[1]) else y_max
    return point