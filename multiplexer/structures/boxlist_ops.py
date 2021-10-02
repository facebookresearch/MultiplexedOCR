# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from d2ocr.structures.rotated_box_list import RotatedBoxList
from d2ocr.structures.segmentation_mask import SegmentationMask

# from d2ocr.layers import nms as _box_nms
from multiplexer.structures.bounding_box import BoxList


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat
    avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def _cat_mask(masks):
    polygons_cat = []
    size = masks[0].size
    for mask in masks:
        polygons = mask.get_polygons()
        polygons_cat.extend(polygons)
    masks_cat = SegmentationMask(polygons_cat, size)
    return masks_cat


def _cat_rotated_box(rotated_boxes):
    return RotatedBoxList(torch.cat([rotated_box.tensor for rotated_box in rotated_boxes]))


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    # if bboxes is None:
    #     return None
    # if bboxes[0] is None:
    #     bboxes = [bboxes[1]
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        if field == "masks":
            data = _cat_mask([bbox.get_field(field) for bbox in bboxes])
        elif field == "rotated_boxes_5d":
            data = _cat_rotated_box([bbox.get_field(field) for bbox in bboxes])
        else:
            data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def cat_boxlist_gt(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    # bboxes[1].set_size(size)
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    if bboxes[0].bbox.sum().item() == 0:
        cat_boxes = BoxList(bboxes[1].bbox, size, mode)
    else:
        cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        if bboxes[0].bbox.sum().item() == 0:
            if field == "masks":
                data = _cat_mask([bbox.get_field(field) for bbox in bboxes[1:]])
            elif field == "rotated_boxes_5d":
                data = _cat_rotated_box([bbox.get_field(field) for bbox in bboxes[1:]])
            else:
                data = _cat([bbox.get_field(field) for bbox in bboxes[1:]], dim=0)
        else:
            if field == "masks":
                data = _cat_mask([bbox.get_field(field) for bbox in bboxes])
            elif field == "rotated_boxes_5d":
                data = _cat_rotated_box([bbox.get_field(field) for bbox in bboxes])
            else:
                data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
