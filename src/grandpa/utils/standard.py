import base64
import glob
import inspect
import io
import json
import math
import os
import pickle
import random
import re
import sys
from copy import deepcopy

import cv2
import numpy as np
from imageio import imread
from tqdm import tqdm


def unify_string(string: str) -> str:
    """
    Converts the input string to lowercase and removes all special characters.

    Args:
        string: string to convert

    Returns: String in lowercase with all special characters removed.
    """
    lower = string.lower()
    return re.sub(r"[^a-z0-9]", "", lower)


def unify_path(path: str) -> str:
    """
    OpenCV may have problems opening images on a mixed slashes path.
    This function replaces forward slashes with double backwards slashes on windows systems
    and backwards on a unix system.
    Args:
        path: path string

    Returns: Path string in the format of the current operating system.

    """
    if os.name == "nt":
        return re.sub(r"/", "\\\\", path)
    else:
        return re.sub(r"\\", "/", path)


def isnan(arr):
    """
    Checks if the input is nan or infinite.

    Args:
        arr: Input to check

    Returns:
        True if the input is nan or infinite, else False.
    """
    if isinstance(arr, list):
        return np.any([isnan(x) for x in arr])
    return np.isnan(np.sum(arr)) or np.isinf(np.sum(arr))


def to_num(arr):
    """
    Replaces nan with zero and infinite with a high finite number.

    Args:
        arr: Iterable list type or single attribute.

    Returns:
        Input with all nan and inf values replaced.
    """
    if isinstance(arr, list):
        return [to_num(x) for x in arr]
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=1, neginf=0.0)


def load_json_from_file(path: str):
    """
    Loads and parses a json file as a dictionary.

    Args:
        path: location of json file

    Returns: dictionary or none if json error or file does not exist.
    """
    try:
        return json.load(open(path))
    except Exception as e:
        print("Load json: " + str(path) + " failed: " + str(e))
        return None


def get_next_numeric_folder_in_dir(path: str, alt_folder_name: str):
    """
    Gets the path to a folder that can be created by appending an increasing number to alt_folder_name.

    Args:
        path: Path where the folder should be created.
        alt_folder_name: Name of the folder.

    Returns:
        Path of a folder that can be created, with the given name and a number appended to it.
    """
    sub_dir = (
        path
        + alt_folder_name
        + str(len(glob.glob(path + alt_folder_name + '*')))
        + '\\'
    )
    i = 0
    while True:
        if os.path.exists(sub_dir):
            sub_dir = path + alt_folder_name + str(i) + '\\'
            i += 1
        else:
            return sub_dir


def weighted_choice(gens: list):
    """
    Weighted random choice based on propability.
    Args:
        gens: list of elements to be chosen of

    Returns: element of given list based on their probability

    """
    x = random.random()
    for i, elmt in enumerate(gens):
        if x <= gens[i].get_probability():
            return elmt
        x -= gens[i].get_probability()


def rand_minmax(val_min=0, val_max=1, size=1):
    """
    uniform random value between val_min and val_max

    Args:
        val_min: min random value
        val_max: max random value
        size: number of random values

    Returns: random floating point value
        (list of random floating point values if size > 1)

    """
    x = val_min + np.random.rand(size) * (val_max - val_min)
    if size == 1:
        x = x[0]
    return x


def blocks(files, size=65536):
    """
    Calculates the number of lines of a large text file.
    (source: https://stackoverflow.com/questions/9629179/python-counting-lines-in-a-huge-10gb-file-as-fast-as-possible)
    Args:
        files: source file
        size: block size to be read

    Returns: number of lines

    """
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def print_warning(txt: str):
    """
    Prints a warning in the color yellow to the console.

    Args:
        txt: String to print.

    Returns:
        None
    """
    print("\033[93m WARNING: " + txt + '\033[0m')


def convert_image_to_255(image):
    """
    Converts the image from 0-1 to 0-255 if the image is in the 0-1 format, otherwise returns the image as it is.

    Args:
        image: image as array

    Returns:
        image as array with a number range from 0-255
    """
    if np.max(image) > 1 and not np.max(image) > 255:
        return image
    elif not np.max(image) > 1:
        return image * 255
    else:
        print_warning(
            f'Utils.convert_image_to_255 received image with max value {np.max(image)}. '
            f'Normalizing image to 255.'
        )
        return normalize_image(image, 255)


def convert_images_to_255(images):
    """
    Converts a list of images to the 0-255 format.

    Args:
        images: List of images.

    Returns:
        List of images convert to 255.
    """
    for i in range(len(images)):
        images[i] = convert_image_to_255(images[i])
    return images


def normalize_image(image, value):
    """
    Normalizes the image to the given value.

    Args:
        image: image as array
        value: the value to normalize to (usually 255 or 1)

    Returns:
        Image as array normalized to the given value.
    """
    return image / (value / np.max(image))


def get_funcs_in_modules(module, submodule_depth=0) -> list:
    """
    Returns the functions declared as from xxx import xxx in the module. Can explore submodules via depth param
    Args:
        module: the module to look in for functions
        submodule_depth: how many submodules to explore

    Returns: A list of functions in the submodules
    """
    funcs = inspect.getmembers(module, inspect.isfunction)
    for i in range(submodule_depth):
        submods = inspect.getmembers(module, inspect.ismodule)
        for submod in submods:
            subfuncs = get_funcs_in_modules(submod, submodule_depth - 1)
            funcs.extend(subfuncs)
    return funcs


def keys_in_dict(dictionary: dict, keys: list):
    """
    Checks if the given list of keys is contained in the dictionary.

    Args:
        dictionary: Dict to check.
        keys: Keys to check for.

    Returns:
        True if all keys are contained in the dict, else false.
    """
    return all([key in dictionary for key in keys])


def keys_in_dicts(dicts: list, keys: list):
    """
    Checks if all keys are contained in all dicts.

    Args:
        dicts: List of dicts to check.
        keys: List of keys to check for.

    Returns:
        True if all keys are contained in all dicts, else false.
    """
    return all([keys_in_dict(dictionary, keys) for dictionary in dicts])


def get_value_in_nested_dict(d: dict, keys: list, not_found_return=None):
    """
    Returns a value in a nested dict if value is in dict else None
    Args:
        d: dictionairy
        keys: a list of keys e.g. ['value1','value'] equals d['value1]['value2']
    Returns: value if in dict else None
    """
    for key in keys:
        try:
            d = d[key]
        except KeyError:
            return not_found_return
    return d


def nested_set(dic, keys, value, create_missing=True):
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dic
    if keys[-1] in d or create_missing:
        d[keys[-1]] = value
    return dic


def do_import(class_path: str):
    """
    Imports a class from a string. Modules can either be separated by a '.' or a '/'.
    Example: class_path = data_processing/dp_trees/tree/Tree will import the tree class.

    Args:
        class_path: Path to import

    Returns:
        The imported class
    """
    class_path = class_path.replace('/', '.')
    module, import_class = class_path.rsplit('.', 1)
    return getattr(__import__(module, fromlist=[import_class]), import_class)


def convert_base_64_to_ndarray(base64img: str):
    '''
    Converts a Base64 encoded image (as returned by our DB) to a ndarray grayscale image.

    Args:
        base64img: Base64 encoded image
        color: Color of the output image

    Returns:
        ndarray grayscale image
    '''
    if base64img is None:
        print_warning("recieved None as image")
        return None

    encoded_img = base64img.split(',')[1]
    decoded_img = np.array(imread(io.BytesIO(base64.b64decode(encoded_img))))
    swapped = decoded_img[..., [2, 1, 0]].copy()
    return swapped


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


def get_all_line_coords(point1, point2, resolution=1):
    """
    Args:
        point1: Point to start the line from
        point2: Point at which the line should end
        resolution: Resolution of the points. Higher resolution results in better accuracy, lower resolution will make
        computation less expensive.

    Returns:
        Points representing a line between the given points.
    """
    dist = int(cal_distance(point1, point2) * resolution)
    if dist == 0:
        return []
    x_diff = (point1[0] - point2[0]) / dist
    y_diff = (point1[1] - point2[1]) / dist
    line_points = [
        [point1[0] - x_diff * i, point1[1] - y_diff * i] for i in range(dist)
    ]
    return line_points


def get_min_distance(point, points):
    min_dist = [None, float('inf')]
    for p in points:
        dist = cal_distance(point, p)
        if dist < min_dist[1]:
            min_dist = [p, dist]
    return min_dist


def get_conversion_matrix_boxes(src_points, target_points):
    assert len(src_points) == len(
        target_points
    ), "Target and source point count need to match."
    assert len(src_points) % 2 == 0, "There must be an even number of source points."
    src_boxes = []
    target_boxes = []
    for i in range(1, int(len(src_points) / 2)):
        points = [
            src_points[i * 2 - 2],
            src_points[i * 2],
            src_points[i * 2 + 1],
            src_points[i * 2 - 1],
        ]
        src_boxes.append(points)
        target_boxes.append(
            [
                target_points[i * 2 - 2],
                target_points[i * 2],
                target_points[i * 2 + 1],
                target_points[i * 2 - 1],
            ]
        )
    return np.float32(src_boxes), np.float32(target_boxes)


def center_to_zero(box):
    return np.array(
        [
            [0, 0],
            [box[1][0] - box[0][0], 0],
            [box[2][0] - box[3][0], box[2][1] - box[1][1]],
            [0, box[3][1] - box[0][1]],
        ],
        dtype="float32",
    )


def warp_polygon_box(img, src_points):
    src_boxes, target_boxes = calc_poly_warp(src_points)
    dst = warp_with_boxes(img, src_boxes, target_boxes)
    return dst, target_boxes, src_boxes


def calc_poly_warp(src_points):
    point_pairs = [[src_points[0], src_points[-1]]]
    forward_index = 1
    reverse_index = -2
    while len(src_points) + 1 > forward_index - reverse_index:
        bottom_line = get_all_line_coords(
            src_points[reverse_index], src_points[reverse_index + 1]
        )
        top_line = get_all_line_coords(
            src_points[forward_index], src_points[forward_index - 1]
        )
        closest_top_line_point, min_top_line_distance = get_min_distance(
            src_points[reverse_index], top_line
        )
        closest_bottom_line_point, min_bottom_line_distance = get_min_distance(
            src_points[forward_index], bottom_line
        )
        if min_top_line_distance < min_bottom_line_distance:
            point_pairs.append([src_points[reverse_index], closest_top_line_point])
            if all(
                np.array(closest_top_line_point, dtype="int32")
                == src_points[forward_index]
            ):
                forward_index += 1
            reverse_index -= 1
        else:
            point_pairs.append([src_points[forward_index], closest_bottom_line_point])
            if all(
                np.array(closest_bottom_line_point, dtype="int32")
                == src_points[reverse_index]
            ):
                reverse_index -= 1
            forward_index += 1
    average_height = int(
        sum([cal_distance(pp[0], pp[1]) for pp in point_pairs]) / len(point_pairs)
    )
    rel_x = 0
    rel_y = 0
    target_points = [[rel_x, rel_y], [rel_x, rel_y + average_height]]
    source_points = [point_pairs[0][0], point_pairs[0][1]]
    distance_to_0 = 0
    for i in range(1, len(point_pairs)):
        x_coord = (
            rel_x
            + distance_to_0
            + (
                int(
                    (
                        cal_distance(point_pairs[i][0], point_pairs[i - 1][0])
                        + cal_distance(point_pairs[i][1], point_pairs[i - 1][1])
                    )
                    / 2
                )
            )
        )
        target_points.append([x_coord, rel_y])
        target_points.append([x_coord, rel_y + average_height])

        # Make sure the boxes are in the correct order
        if point_pairs[i][0][1] < point_pairs[i][1][1]:
            source_points.append(np.int32(point_pairs[i][0]))
            source_points.append(np.int32(point_pairs[i][1]))
        else:
            source_points.append(np.int32(point_pairs[i][1]))
            source_points.append(np.int32(point_pairs[i][0]))
        distance_to_0 += int(cal_distance(point_pairs[i][0], point_pairs[i - 1][0]))
    src_boxes, target_boxes = get_conversion_matrix_boxes(source_points, target_points)
    return src_boxes, target_boxes


def warp_with_boxes(img, src_boxes, target_boxes):
    dst = None
    for src_box, target_box in zip(src_boxes, target_boxes):
        if any(
            [
                cal_distance(src_box[i1], src_box[i2]) <= 0
                for i1, i2 in [(0, 1), (2, 3), (1, 2), (0, 3)]
            ]
        ):
            continue
        target_box = center_to_zero(target_box)
        width = round(
            (
                cal_distance(target_box[0], target_box[1])
                + cal_distance(target_box[2], target_box[3])
            )
            / 2
        )
        height = round(
            (
                cal_distance(target_box[1], target_box[2])
                + cal_distance(target_box[3], target_box[0])
            )
            / 2
        )
        perspective_mat = cv2.getPerspectiveTransform(src=src_box, dst=target_box)
        perspective_mat = np.array(perspective_mat)
        if dst is None:
            dst = cv2.warpPerspective(img, perspective_mat, (width, height))
            dst = np.array(dst)
        else:
            warped = cv2.warpPerspective(img, perspective_mat, (width, height))
            warped = np.array(warped)
            dst = np.append(dst, warped, axis=1)
    return dst


def crop_image(src, points, dst_height=None):
    """
    Crop heat map with points.
    :param src: 8-bit single-channel image (map).
    :param points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    :return: dst_heat_map: Cropped image. 8-bit single-channel image (map) of heat map.
             src_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
             dst_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    src_image = src.copy()
    src_points = np.float32(points)
    width = round(
        (cal_distance(points[0], points[1]) + cal_distance(points[2], points[3])) / 2
    )
    height = round(
        (cal_distance(points[1], points[2]) + cal_distance(points[3], points[0])) / 2
    )
    if dst_height is not None:
        if width == 0:
            width += 1
        if height == 0:
            height += 1
        ratio = dst_height / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
    crop_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=crop_points)
    perspective_mat = np.array(perspective_mat)
    dst_heat_map = cv2.warpPerspective(
        src_image,
        perspective_mat,
        (width, height),
        borderValue=0,
        borderMode=cv2.BORDER_CONSTANT,
    )
    dst_heat_map = np.array(dst_heat_map)
    return dst_heat_map, src_points, crop_points


def pad_to_min_shape(img: np.array, min_shape):
    ## (4) pad image to have same size
    padding = np.array([0, 0, 0, 0])
    # pad image to have minimum shape of min_shape
    pad_pixels = np.array(min_shape) - np.array(img.shape)[:-1]
    pad_pixels[pad_pixels < 0] = 0
    if np.any(pad_pixels > 0):
        top, bottom = pad_pixels[0] // 2, pad_pixels[0] - (pad_pixels[0] // 2)
        left, right = pad_pixels[1] // 2, pad_pixels[1] - (pad_pixels[1] // 2)
        padding = np.array([left, top, bottom, right])
        img = cv2.copyMakeBorder(
            img,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        img = np.array(img)
    return img, padding


def crop_polygon(img, points):
    pts = np.float32(points)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    rect = np.array(rect)
    x, y, w, h = rect
    cropped = img[y : y + h, x : x + w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts.astype("int32")], -1, (1, 1, 1), -1, cv2.LINE_AA)
    mask = np.array(mask)

    ## (3) do bit-op
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    dst = np.array(dst)

    return dst


def tree_search(
    search_list, set_value_list, check_func, target_value, always_return=False
):
    """
    Greedy function that is called recursively to create and execute a basic depth first search tree.

    Args:
        search_list: The list of possible solutions for this tree node.
        set_value_list: The list of values set by previous nodes, for final execution and check.
        check_func: Function used for checking the result. Gets set_value_list as arg.
        target_value: Expected result of the cehck_func.
        always_return: Will return the set_value_list extended by the search list if no solution was found.
    Returns:
        First found correct solution or none, if no solution was found.
    """
    if len(search_list) == 1:
        local_set_value_list = deepcopy(set_value_list)
        local_set_value_list.append(search_list[0])
        if check_func(local_set_value_list) == target_value:
            return local_set_value_list
    else:
        for option in search_list:
            local_search_list = deepcopy(search_list)
            local_set_value_list = deepcopy(set_value_list)
            local_set_value_list.append(option)
            local_search_list.remove(option)
            res = tree_search(
                local_search_list, local_set_value_list, check_func, target_value
            )
            if res is not None:
                return res
    if always_return:
        set_value_list.extend(search_list)
        return set_value_list


gettrace = getattr(sys, 'gettrace', None)


def check_debug():
    return os.getenv("project_x_debug") == "true"


def save_as_pickle(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)
    return data


def filter_kwargs(func, kwargs):
    filtered_call_params = {}
    for k, v in kwargs.items():
        if k in inspect.signature(func).parameters or any(
            str(p.kind) == 'VAR_KEYWORD'
            for p in inspect.signature(func).parameters.values()
        ):
            filtered_call_params[k] = v
    return filtered_call_params


def img_normalize_craft(src):
    """
    Normalize a RGB image.
    :param src: Image to normalize. Must be RGB order.
    :return: Normalized Image
    """
    NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    NORMALIZE_VARIANCE = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    img = src.copy().astype(np.float32)

    img -= NORMALIZE_MEAN
    img /= NORMALIZE_VARIANCE
    return img
