import glob
import os.path

import cv2

from grandpa.decorators import Component, Node, Workflow


def sum_array(arr):
    total = 0
    for item in arr:
        total += item
    return total


@Node("sum_array_node")
def sum_array_node(arr):
    total = 0
    for item in arr:
        total += item
    return total


def run_sum_array():
    test_array = [i for i in range(1, 1000000)]
    result = sum_array(test_array)
    return result


@Workflow("sum_array_workflow")
def sum_array_workflow():
    test_array = [i for i in range(1, 1000000)]
    result = sum_array_node(test_array)
    return result


def process_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    os.makedirs(f'gray_{os.path.dirname(os.path.relpath(image_path))}', exist_ok=True)
    cv2.imwrite(f'gray_{os.path.relpath(image_path)}', gray_img)


def process_images():
    image_paths = glob.glob('images/**/*.jpg')
    for path in image_paths:
        process_image(path)


@Node("process_image")
def process_image_node(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    os.makedirs(f'gray_{os.path.dirname(os.path.relpath(image_path))}', exist_ok=True)
    cv2.imwrite(f'gray_{os.path.relpath(image_path)}', gray_img)


@Node("result")
def result_node(*results):
    return results


@Workflow("process_images")
def process_images_workflow():
    image_paths = glob.glob('images/**/*.jpg')
    path_nodes = [process_image_node(path) for path in image_paths]
    return result_node(*path_nodes)
