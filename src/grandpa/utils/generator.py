import cv2
import numpy as np
from PIL import Image, ImageChops

from . import bounding_box
from .standard import unify_path


def get_mask(shape, bbox: list) -> np.array:
    """
    Returns a black image with the bounding box in white
    Args:
        shape: shape of the new image
        bbox: bounding box locations

    Returns:
        np.array all values are zero, bounding box values are 1
    """
    mask_img = np.zeros(shape, dtype="uint8")
    for bb in bbox:
        if len(bb) == 4:
            bb = bounding_box.order_points(bb)
        mask_img = cv2.fillPoly(
            mask_img, [np.array(bb).astype("int32")], color=255
        ).astype("float32")
        mask_img = np.array(mask_img)
    return mask_img


def read_img_as_cv2(img_path: str) -> np.array:
    """
    Reads an image from disk into cv2 bbox_format
    Args:
        img_path: the path to the image on disk

    Returns:
        the loaded image as np.array
    """
    try:
        img = cv2.imread(unify_path(img_path), cv2.IMREAD_COLOR)
        if img is None:
            img = np.array(Image.open(img_path).convert("RGB"))
            img = cv2.cvtColor(img, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Problems with  img: " + img_path + " Error: " + str(e))
        img = None
    return img


def resize_img(img: np.ndarray, dim=(512, 512)) -> np.ndarray:
    """
    Resizes an image by absolute dimensions or scale between 0.0 and 1.0.
    Args:
        img: image to resize
        dim: tuple - absolute scaling - (x,y)
             float - relative scaling - 0.0 < x <= 1.0

    Returns: resized image

    """
    if type(dim) is tuple:
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    elif type(dim) is float:
        w = int(img.shape[1] * dim)
        h = int(img.shape[0] * dim)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        raise TypeError("Dimension type not supported. Provide tuple or float instead.")


def invert_img(img: np.ndarray) -> np.ndarray:
    """
    Inverts an image.
    Args:
        img: image to invert

    Returns: inverted image

    """
    return cv2.bitwise_not(img)


def gaussian_noise(img, mean=0, var=0.1):
    """
    Applies a gaussian noise effect to an image.
    Args:
        img: source
        mean: TODO: add mean description
        var: strength of noise

    Returns: image with noise

    """
    row, col = img.shape[:2]
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, 3))
    gauss = gauss.reshape((row, col, 3))
    noisy = (img + gauss).astype(np.uint8)
    return noisy


def smoothen(img):
    """
    Smoothes an image.
    Args:
        img: source

    Returns: smoothed image

    """
    return cv2.GaussianBlur(img, (5, 5), 0)


def auto_crop_content(img):
    """
    Crops an image.
    Args:
        img: image (numpy array)

    Returns: cropped image (numpy array)

    """
    pil = True
    if type(img) is np.ndarray:
        pil = False
        im = Image.fromarray(img)
    else:
        im = img.copy()
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        bbox = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
        return im.crop(bbox)
    if pil:
        return im
    else:
        return np.array(im)


def get_wh(bbox):
    """
    Calculated width and height from bounding box.
    Args:
        bbox: two point bounding box

    Returns: width and height

    """
    if len(bbox) > 2:
        # 4-point or poly
        return bbox[1][0] - bbox[0][0], bbox[2][1] - bbox[0][1]
    # 2-point
    return bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]


def get_transformation(img: np.ndarray, output_shape, bbox: list) -> np.ndarray:
    """
    Based on the bounding box that image part is cut out and then resized to output_shape
    Args:
        img: Image as numpy array
        output_shape: The shape the cutout template gets resized to (height, width)
        bbox: the bounding box for the part of the image

    Returns:
        the resized cutout
    """
    raw_image = bounding_box.cut_out_image_area(img, bbox[0])
    return cv2.resize(raw_image, tuple(output_shape), interpolation=cv2.INTER_CUBIC)
