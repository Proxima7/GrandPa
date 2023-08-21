import cv2
import numpy as np
from numpy import random

from . import standard
from . import generator
from .bounding_box import rescale_bbox

def color_fix(image: np.array, target_shape):
    """
    Converts the the image into the color format given by the target shape.

    Args:
        image: Image to convert.
        target_shape: Target shape for the image.

    Returns:
        Image converted into the target color format.
    """
    # RGB to Gray
    if target_shape[-1] == 1 and len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # RGBA to Gray
    elif target_shape[-1] == 1 and len(image.shape) == 3 and image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    # GRAY to RGB
    elif target_shape[-1] == 3 and (len(image.shape) != 3 or image.shape[-1] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # gray no channel to gray with channel
    elif target_shape[-1] == 1 and len(image.shape) == 2:
        image = image.reshape(target_shape)
    # RGBA to RGB
    elif target_shape[-1] == 3 and (len(image.shape) == 3 and image.shape[-1] == 4):
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = np.array(image)
    return image


def preprocess_image(image: np.array, target_shape) -> np.array:
    """
    This method rescales the image to target images,
    makes sure that all batches have the same color bbox_format (of target shape)
    Replaces nan values
    And normalizes the image to 0 to 1 by divding all values by 255

    Args:
        image: the image to process
        target_shape: the output shape and bbox_format

    Returns:
        The preprocessed image as np.array
    """
    image = np.array(image)
    # rescale image
    image_size = target_shape[:-1]
    image_size_array = np.array(image_size)
    origin_image_size = np.array(image.shape)
    scale = image_size_array / origin_image_size[:2]
    image = cv2.resize(image, dsize=tuple(image_size[::-1]), interpolation=cv2.INTER_CUBIC)
    image = np.array(image)
    # fixing errors
    image = standard.to_num(image)
    # to flaot
    image = image.astype(np.float32)
    # divide by 255 to normalize image
    #NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    #NORMALIZE_VARIANCE = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    #image = image.copy().astype(np.float32)
    image /= 255
    #image -= NORMALIZE_MEAN
    #image /= NORMALIZE_VARIANCE
    # image bbox_format
    image = color_fix(image, target_shape)

    return image, scale


def get_and_preprocess_template(image, template_shape, bbox):
    """
    Gets and preprocesses a template, for example for target tracking.

    Args:
        image: the image to extract the template from
        template_shape: the target shape to which the template is resized to
        bbox: the bounding box of the template that gets extracted

    Returns:
        A preprocessed template cut out of the image
    """
    output_hw = template_shape[:-1][::-1]
    template = generator.get_transformation(image, output_hw[:2], bbox['bounding_box'].tolist())
    template = preprocess_image(template, template_shape[1:])[0]
    return template


def preprocess_images(images: list, target_shape, return_scale=True) -> np.array:
    """
    wrapper for preprocess_image for multiple images
    Args:
        images: a list of images as np.arrays
        target_shape: the shape the images shall have
        return_scale: Whether to return the reshaping scale (True) or the actual reshaped image (False).
    Returns:
        A list of preprocessed images or the scales the images need to be resized by if return_scale == True.
    """
    images_scales = np.array([preprocess_image(img, target_shape) for img in images], dtype=object)
    if return_scale:
        return images_scales
    return np.array([im_sc[0] for im_sc in images_scales])


def rescale_bounding_boxes(target_shape, image_shapes, bounding_boxes):
    for i in range(len(image_shapes)):
        image_shapes[i] = image_shapes[i] if len(image_shapes[i]) == 2 else image_shapes[i][:2]
    scales = target_shape[:2] / np.array(image_shapes)
    rescaled_bboxes = rescale_bounding_boxes_by_scales(bounding_boxes, scales)
    return rescaled_bboxes


def rescale_bounding_boxes_by_scales(boxes, scales):
    """

    Args:
        boxes: Boxes to rescale.
        scales: Scales as single tuple (x, y) or list of tuples [(x,y)..]
            If a single tuple is given all boxes are rescaled by the given factors.
            If a list of tuples is given, each box gets individually scaled by the given factor.

    Returns:
        Rescaled bounding boxes.
    """
    # No boxes. Return
    if not boxes or len(boxes) == 0 or (len(boxes[0]) == 0 and len(boxes) <= 1):
        return boxes

    # single scale which is applied to all
    scales = np.array(scales)
    if scales.shape == (2,):
        scales = np.array([scales for i in range(len(boxes))])

    # Note: Changed it from boxes[0][0] to boxes[0] if you have to change it again please contact Matthias.
    # Change was required because bounding box list is not nested anymore

    # Do it for four point bounding boxes
    if np.array(boxes[0]).shape == (4, 2):
        # tile scales to four point and then easy matrix multiplication
        scales = np.tile(scales, 4).reshape(len(scales), 4, 2)

    rescaled_boxes = [[np.array(box) * [scale[1], scale[0]] for box in img_boxes] for scale, img_boxes in zip(scales, boxes)]
    return rescaled_boxes

    # Two point bounding box
    #return [np.array(boxes[i]) * np.tile(scales[i], 2).T for i in range(len(boxes))]


def process_inputs(batch, target_shape):
    """
    Rescales the images in batch[0] and the corresponding bounding boxes.

    Args:
        batch: Batch returned by generator.
        target_shape: Target shape for the images.

    Returns:
        images: Rescaled images.
        rescaled_boxes: Rescaled bounding boxes.
    """
    image_scale = preprocess_images(batch[0], target_shape)
    images = np.array([isc[0] for isc in image_scale])
    # calculate scalses
    scales = np.array([isc[1] for isc in image_scale])
    rescaled_boxes = []
    for i, entry in enumerate(batch[1]):
        scale = [scales[i][1], scales[i][0]]
        boxes_for_entry = []
        for bbox in entry['bounding_box']:
            rescaled_bbox = rescale_bbox([bbox], scale)
            boxes_for_entry.append(rescaled_bbox)
        rescaled_boxes.append(boxes_for_entry)

    return images, rescaled_boxes


def preprocess_true_boxes(true_boxes, anchors, num_classes, input_shape, output_shapes):
    # Todo: Damit die Methode funktioniert muss herausgefunden werden, was die output shapes tun. Meines Wissens nach kommen diese aus den Outputs des Model, müssen also per routing aus der networkfactory gezogen werden.
    """
    Preprocess true boxes to training input bbox_format

    Args:
        true_boxes: Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: (height, width), multiples of 32
        anchors: shape=(N, 2), (width, height)
        num_classes: Number of classes

    Returns:
        y_true: list of array, shape like yolo_outputs, xywh are relative values
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]

    # Transform box info to (x_center, y_center, box_width, box_height, cls_id)
    # and image relative coordinate.
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape[:-1], dtype='int32')

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]

    # grid_shapes = [input_shape // ]
    # TODO: Akzeptiert gerade nur die letzten num_layers als factor.
    #  Besser mehr unterstützen oder parametrisieren
    max_down_sample_factor = input_shape[0] // output_shapes[0][0]
    grid_shapes = [
        input_shape // {0: max_down_sample_factor, 1: max_down_sample_factor//2, 2: max_down_sample_factor//4}[l]
        for l in range(num_layers)
    ]

    yolo_boxes = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch_size):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        best_anchor = get_best_anchor(anchor_maxes, anchor_mins, anchors, wh)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    yolo_boxes[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    yolo_boxes[l][b, j, i, k, 4] = 1
                    yolo_boxes[l][b, j, i, k, 5 + c] = 1

    return yolo_boxes


def get_best_anchor(anchor_maxes, anchor_mins, anchors, wh):
    """
    Get the best yolo anchor for specific bounding boxes.

    Args:
        anchor_maxes: Max anchor values.
        anchor_mins: Min anchor values.
        anchors: Yolo anchors.
        wh: width and height of the boxes.

    Returns:
        The best anchor for the boxes.
    """
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)

    box_maxes = wh / 2.
    box_mins = -box_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)

    return best_anchor


def weighted_choice(items: list):
    """
    Weighted random choice based on probability. An item must have the attribute probability to be used

    Args:
        items: list of elements to be chosen of

    Returns:
        Element of given list based on their probability
    """
    if len(items) == 1:
        return items[0]

    x = random.random() * sum([item.probability for item in items])
    for i, element in enumerate(items):
        if x <= items[i].probability:
            return element
        x -= items[i].probability


def get_layer_shapes(layer_datapipe_mapping):
    """
    Gets the shapes of the layers.

    Args:
        layer_datapipe_mapping: List of neuron reference wrappings.

    Returns:
        Shapes of the neurons as func_dict with keys 0, 1, ..., n.
    """
    shapes = {}
    for layername in layer_datapipe_mapping:
        if layername.deref().shape not in shapes.items():
            shapes[len(shapes.keys())] = layername.deref().shape
    return shapes


def rescale_images(images, target_shape):
    """
    Rescales the images and returns the scales they were rescaled by.

    Args:
        images: Images to rescale.
        shapes: List of target images as func_dict with keys 0, 1, ..., n.

    Returns:
        images: Rescaled images as func_dict with the image shape as key.
        scales: The scales the images where rescaled by as func_dict with the image images as key.
    """
    images = [color_fix(img, (img.shape[0], img.shape[1], 3)) for img in images]
    images = preprocess_images(images, target_shape, return_scale=False)
    return images


def rescale_image(image, target_shape):
    """
    Rescales the images and returns the scales they were rescaled by.

    Args:
        images: Images to rescale.
        shapes: List of target images as func_dict with keys 0, 1, ..., n.

    Returns:
        images: Rescaled images as func_dict with the image shape as key.
        scales: The scales the images where rescaled by as func_dict with the image images as key.
    """
    image = color_fix(image, (image.shape[0], image.shape[1], 3))
    image, _ = preprocess_image(image, target_shape)
    return image


def filter_annotations_for_class(class_name):
    def _filter_annotations_for_class(batch):
        filtered = []
        for x in batch:
            filtered.append([ann for ann in x['annotations'] if ann['type'] == class_name])
        return filtered

    return _filter_annotations_for_class


def get_annotation_vals(query):
    def _f(annotations):
        anns = []
        for x in annotations:
            if type(query) == list:
                anns.append([
                    standard.get_value_in_nested_dict(ann, query) for ann in x
                ])
            else:
                anns.append([ann[query] for ann in x])
        return anns
    return _f


def get_reshaped_masks(target_shape, rescaled_bounding_boxes):
    """
    Creates masks and reshapes them.

    Args:
        batch: Batch for predictions.
        bbox_scales: Scales the boxes need to be rescaled by to match the original rescaled image.
        key: Neuron the masks should be created for.
        key_ref: Dict containing reference keys and the corresponding image images.
        kwargs: Additional arguments.
        images: Images the masks are created for.

    Returns:
        reshaped_masks: N Masks, N depending on the batch size.
    """
    masks = [generator.get_mask(target_shape[:2], rescaled_img_bounding_boxes)
             for rescaled_img_bounding_boxes in rescaled_bounding_boxes]
    return masks


def store_data_in_inputs_and_outputs(neurons, data, key_refs, inputs=None, outputs=None):
    """
    Stores the data in the input and/or output func_dict and returns them.

    Args:
        neurons: Neurons the function should be executed for.
        data: Data to store in inputs / outputs.
        key_refs: Keys referring to the shapes.
        inputs: Inputs passed to the network.
        outputs: Outputs of the network.

    Returns:
       inputs: Inputs passed to the network.
        outputs: Outputs of the network.
    """
    for neuron in neurons:
        for key in key_refs:
            shape = key_refs[key]
            if neuron.deref().shape == shape:
                if inputs is not None:
                    inputs[neuron] = data[key]
                if outputs is not None:
                    outputs[neuron] = data[key]
    return inputs, outputs


def get_template_dict(batch, shapes, targets):
    """
    Gets a dictionary containing the templates with a key ranging from 0...n.

    Args:
        batch: Batch for predictions.
        shapes: Shapes the templates should be created for.
        targets: Targets for the template generation (exactly one bounding box per image).

    Returns:
        Dict containing keys from 0...n and templates for each image and each shape.
    """
    results = {}
    for key in shapes:
        templates = []
        template_shape = shapes[key]
        for i in range(len(batch[0])):
            template = get_and_preprocess_template(batch[0][i], template_shape, targets[i])
            templates.append(np.array(template))
        results[key] = np.array(templates)
    return results


def convert_bb_to_yolov4bb(bounding_boxes, input_shapes, output_shapes, anchors=None):
    """
    Converts the standard bounding boxes to YoloV4 bounding boxes.

    Args:
        model: The keras model of the neural network
        neurons: The reference wrappings of the processing tree.
        bounding_boxes: The bounding boxes to convert to YoloV4 format.

    Returns:
        Bounding boxes in YoloV4 format.
    """
    # prepare to bbox_format where multiple boxes could be per image --> 0 is the class
    bbx = np.array([np.array([np.append(b, 0) for b in image]) for image in bounding_boxes])
    bbx = [image.reshape([1, image.shape[0], image.shape[1]]) for image in bbx]
    yolo_bounding_boxes = [preprocess_true_boxes(image, anchors, 1, input_shapes, output_shapes) for image in bbx]
    n1 = np.array([x[0][0] for x in yolo_bounding_boxes])
    n2 = np.array([x[1][0] for x in yolo_bounding_boxes])
    n3 = np.array([x[2][0] for x in yolo_bounding_boxes])
    # fixing nan erros in preprocessing
    yolo_bounding_boxes = [n1, n2, n3]
    yolo_bounding_boxes = standard.to_num(yolo_bounding_boxes)
    return yolo_bounding_boxes
