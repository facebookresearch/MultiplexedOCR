# Copyright (c) Facebook, Inc. and its affiliates.
import math
import random

import cv2
import numpy as np
import pyclipper
import torch
from shapely.geometry import Polygon
from torch import nn

from multiplexer.structures.bounding_box import BoxList
from multiplexer.structures.boxlist_ops import cat_boxlist, cat_boxlist_gt
from multiplexer.structures.segmentation_mask import SegmentationMask

from .build import PROPOSAL_GENERATOR_REGISTRY


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class SEGHead(nn.Module):
    """
    Adds a simple SEG Head with pixel-level prediction
    """

    def __init__(self, in_channels, cfg):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SEGHead, self).__init__()
        self.cfg = cfg
        ndim = 256
        self.fpn_out5 = nn.Sequential(
            conv3x3(ndim, 64), nn.Upsample(scale_factor=8, mode="nearest")
        )
        self.fpn_out4 = nn.Sequential(
            conv3x3(ndim, 64), nn.Upsample(scale_factor=4, mode="nearest")
        )
        self.fpn_out3 = nn.Sequential(
            conv3x3(ndim, 64), nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.fpn_out2 = conv3x3(ndim, 64)
        self.seg_out = nn.Sequential(
            conv3x3_bn_relu(in_channels, 64, 1),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid(),
        )
        if self.cfg.MODEL.SEG.USE_PPM:
            # PPM Module
            pool_scales = (2, 4, 8)
            fc_dim = 256
            self.ppm_pooling = []
            self.ppm_conv = []
            for scale in pool_scales:
                self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
                self.ppm_conv.append(
                    nn.Sequential(
                        nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                    )
                )
            self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
            self.ppm_conv = nn.ModuleList(self.ppm_conv)
            self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, ndim, 1)
            self.ppm_conv.apply(self.weights_init)
            self.ppm_last_conv.apply(self.weights_init)
        self.fpn_out5.apply(self.weights_init)
        self.fpn_out4.apply(self.weights_init)
        self.fpn_out3.apply(self.weights_init)
        self.fpn_out2.apply(self.weights_init)
        self.seg_out.apply(self.weights_init)

    def forward(self, x):
        if self.cfg.MODEL.SEG.USE_PPM:
            conv5 = x[-2]
            input_size = conv5.size()
            ppm_out = [conv5]
            for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
                ppm_out.append(
                    pool_conv(
                        nn.functional.interpolate(
                            pool_scale(conv5),
                            (input_size[2], input_size[3]),
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                )
            ppm_out = torch.cat(ppm_out, 1)
            f = self.ppm_last_conv(ppm_out)
        else:
            if self.cfg.MODEL.FPN.USE_PRETRAINED:
                f = x["p5"]
            else:
                f = x[-2]
        # p5 = self.fpn_out5(x[-2])
        p5 = self.fpn_out5(f)
        if self.cfg.MODEL.FPN.USE_PRETRAINED:
            p4 = self.fpn_out4(x["p4"])
            p3 = self.fpn_out3(x["p3"])
            p2 = self.fpn_out2(x["p2"])
        else:
            p4 = self.fpn_out4(x[-3])
            p3 = self.fpn_out3(x[-4])
            p2 = self.fpn_out2(x[-5])
        fuse = torch.cat((p5, p4, p3, p2), 1)
        out = self.seg_out(fuse)

        # if torch.isnan(out).any():
        #     print("[Debug] NaN detected in out: {}".format(out))
        #     print("[Debug] x = \n{}".format(x))
        #     for name, param in self.seg_out.named_parameters():
        #         print("[Debug] Parameter {} = \n{}".format(name, param))
        #         if torch.isnan(param).any():
        #             print("[Debug] NaN detected in {}".format(name))
        #     raise Exception("NaN detected in segmentation.py!")

        return out, fuse

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)


class SEGPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(self, cfg, is_train):
        super(SEGPostProcessor, self).__init__()

        self.top_n = cfg.MODEL.SEG.TOP_N_TRAIN if is_train else cfg.MODEL.SEG.TOP_N_TEST
        self.binary_thresh = cfg.MODEL.SEG.BINARY_THRESH
        self.box_thresh = cfg.MODEL.SEG.BOX_THRESH
        self.min_size = cfg.MODEL.SEG.MIN_SIZE
        self.cfg = cfg
        expand_method_map = {"constant": 0, "log_a": 1}
        self.expand_method = expand_method_map[cfg.MODEL.SEG.EXPAND_METHOD]

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        # device = proposals[0].bbox.
        if (
            self.cfg.MODEL.SEG.USE_SEG_POLY
            or self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE
            or self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE
        ):
            gt_boxes = [target.copy_with_fields(["masks"]) for target in targets]
        else:
            gt_boxes = [target.copy_with_fields([]) for target in targets]
        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        # for gt_box in gt_boxes:
        #     gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))
        proposals = [
            cat_boxlist_gt([proposal, gt_box]) for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def aug_tensor_proposals(self, boxes):
        # boxes: N * 4
        boxes = boxes.float()
        N = boxes.shape[0]
        device = boxes.device
        aug_boxes = torch.zeros((4, N, 4), device=device)
        aug_boxes[0, :, :] = boxes.clone()
        xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        for i in range(3):
            choice = random.random()
            if choice < 0.5:
                # shrink or expand
                ratio = (torch.randn((N,), device=device) * 3 + 1) / 2.0
                height = height * ratio
                ratio = (torch.randn((N,), device=device) * 3 + 1) / 2.0
                width = width * ratio
            else:
                move_x = width * (torch.randn((N,), device=device) * 4 - 2)
                move_y = height * (torch.randn((N,), device=device) * 4 - 2)
                x_center += move_x
                y_center += move_y
            boxes[:, 0] = x_center - width / 2
            boxes[:, 2] = x_center + width / 2
            boxes[:, 1] = y_center - height / 2
            boxes[:, 3] = y_center + height / 2
            aug_boxes[i + 1, :, :] = boxes.clone()
        return aug_boxes.reshape((-1, 4))

    def forward_for_single_feature_map(self, pred, image_shapes):
        """
        Arguments:
            pred: tensor of size N, 1, H, W
        """
        device = pred.device
        # torch.cuda.synchronize()
        # start_time = time.time()
        bitmap = self.binarize(pred)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('binarize time:', end_time - start_time)
        N, height, width = pred.shape[0], pred.shape[2], pred.shape[3]
        # torch.cuda.synchronize()
        # start_time = time.time()
        bitmap_numpy = bitmap.cpu().numpy()  # The first channel
        pred_map_numpy = pred.cpu().numpy()
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('gpu2numpy time:', end_time - start_time)
        boxes_batch = []
        rotated_boxes_batch = []
        polygons_batch = []
        scores_batch = []
        # torch.cuda.synchronize()
        # start_time = time.time()
        for batch_index in range(N):
            image_shape = image_shapes[batch_index]
            boxes, scores, rotated_boxes, polygons = self.boxes_from_bitmap(
                pred_map_numpy[batch_index], bitmap_numpy[batch_index], width, height
            )
            boxes = boxes.to(device)
            if self.training and self.cfg.MODEL.SEG.AUG_PROPOSALS:
                boxes = self.aug_tensor_proposals(boxes)
            if boxes.shape[0] > self.top_n:
                boxes = boxes[: self.top_n, :]
                # _, top_index = scores.topk(self.top_n, 0, sorted=False)
                # boxes = boxes[top_index, :]
                # scores = scores[top_index]
            # boxlist = BoxList(boxes, (width, height), mode="xyxy")
            boxlist = BoxList(boxes, (image_shape[1], image_shape[0]), mode="xyxy")
            if (
                self.cfg.MODEL.SEG.USE_SEG_POLY
                or self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE
                or self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE
            ):
                masks = SegmentationMask(polygons, (image_shape[1], image_shape[0]))
                boxlist.add_field("masks", masks, check_consistency=False)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = remove_small_boxes(boxlist, self.min_size)
            boxes_batch.append(boxlist)
            rotated_boxes_batch.append(rotated_boxes)
            polygons_batch.append(polygons)
            scores_batch.append(scores)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('loop time:', end_time - start_time)
        return boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch

    def forward(self, seg_output, image_shapes, targets=None):
        """
        Arguments:
            seg_output: list[tensor]

        Returns:
            boxlists (list[BoxList]): bounding boxes
        """
        sampled_boxes = []
        (
            boxes_batch,
            rotated_boxes_batch,
            polygons_batch,
            scores_batch,
        ) = self.forward_for_single_feature_map(seg_output, image_shapes)
        if not self.training:
            return boxes_batch, rotated_boxes_batch, polygons_batch, scores_batch
        sampled_boxes.append(boxes_batch)

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)
        return boxlists

    # def select_over_all_levels(self, boxlists):
    #     num_images = len(boxlists)
    #     # different behavior during training and during testing:
    #     # during training, post_nms_top_n is over *all* the proposals combined, while
    #     # during testing, it is over the proposals for each image
    #     # TODO resolve this difference and make it consistent. It should be per image,
    #     # and not per batch
    #     if self.training:
    #         objectness = torch.cat(
    #             [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
    #         )
    #         box_sizes = [len(boxlist) for boxlist in boxlists]
    #         post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
    #         _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
    #         inds_mask = torch.zeros_like(objectness, dtype=torch.uint8)
    #         inds_mask[inds_sorted] = 1
    #         inds_mask = inds_mask.split(box_sizes)
    #         for i in range(num_images):
    #             boxlists[i] = boxlists[i][inds_mask[i]]
    #     else:
    #         for i in range(num_images):
    #             objectness = boxlists[i].get_field("objectness")
    #             post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
    #             _, inds_sorted = torch.topk(
    #                 objectness, post_nms_top_n, dim=0, sorted=True
    #             )
    #             boxlists[i] = boxlists[i][inds_sorted]
    #     return boxlists

    def binarize(self, pred):
        if self.cfg.MODEL.SEG.USE_MULTIPLE_THRESH:
            binary_maps = []
            for thre in self.cfg.MODEL.SEG.MULTIPLE_THRESH:
                binary_map = pred > thre
                binary_maps.append(binary_map)
            return torch.cat(binary_maps, dim=1)
        else:
            return pred > self.binary_thresh

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        """
        # assert _bitmap.size(0) == 1
        # bitmap = _bitmap[0]  # The first channel
        pred = pred[0]
        height, width = bitmap.shape[1], bitmap.shape[2]
        boxes = []
        scores = []
        rotated_boxes = []
        polygons = []
        contours_all = []
        for i in range(bitmap.shape[0]):
            try:
                _, contours, _ = cv2.findContours(
                    (bitmap[i] * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            except BaseException:
                contours, _ = cv2.findContours(
                    (bitmap[i] * 255).astype(np.uint8),
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE,
                )
            contours_all.extend(contours)
        for contour in contours_all:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.reshape((-1, 2))
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points)
            if not self.training and self.box_thresh > score:
                continue
            if polygon.shape[0] > 2:
                polygon = self.unclip(
                    polygon,
                    expand_ratio=self.cfg.MODEL.SEG.EXPAND_RATIO,
                    expand_method=self.expand_method,
                    shrink_ratio=self.cfg.MODEL.SEG.SHRINK_RATIO,
                )
                if len(polygon) > 1:
                    continue
            else:
                continue
            # polygon = polygon.reshape(-1, 2)
            polygon = polygon.reshape(-1)
            box = self.unclip(
                points,
                expand_ratio=self.cfg.MODEL.SEG.BOX_EXPAND_RATIO,
                expand_method=self.expand_method,
                shrink_ratio=self.cfg.MODEL.SEG.SHRINK_RATIO,
            ).reshape(-1, 2)
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            min_x, min_y = min(box[:, 0]), min(box[:, 1])
            max_x, max_y = max(box[:, 0]), max(box[:, 1])
            horizontal_box = torch.from_numpy(np.array([min_x, min_y, max_x, max_y]))
            boxes.append(horizontal_box)
            scores.append(score)
            rotated_box, _ = self.get_mini_boxes(box.reshape(-1, 1, 2))
            rotated_box = np.array(rotated_box)
            rotated_boxes.append(rotated_box)
            polygons.append([polygon])
        if len(boxes) == 0:
            boxes = [torch.from_numpy(np.array([0, 0, 0, 0]))]
            scores = [0.0]
            polygons = [[np.array([[0, 0], [0, 0], [0, 0]])]]
            rotated_boxes = [np.array([[0, 0], [0, 0], [0, 0], [0, 0]])]

        boxes = torch.stack(boxes)
        scores = torch.from_numpy(np.array(scores))
        return boxes, scores, rotated_boxes, polygons

    def aug_proposals(self, box):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        x_center = int((xmin + xmax) / 2.0)
        y_center = int((ymin + ymax) / 2.0)
        width = xmax - xmin
        height = ymax - ymin
        choice = random.random()
        if choice < 0.5:
            # shrink or expand
            ratio = (random.random() * 3 + 1) / 2.0
            height = height * ratio
            ratio = (random.random() * 3 + 1) / 2.0
            width = width * ratio
        else:
            move_x = width * (random.random() * 4 - 2)
            move_y = height * (random.random() * 4 - 2)
            x_center += move_x
            y_center += move_y
        xmin = int(x_center - width / 2)
        xmax = int(x_center + width / 2)
        ymin = int(y_center - height / 2)
        ymax = int(y_center + height / 2)
        return [xmin, ymin, xmax, ymax]

    def unclip(self, box, expand_ratio=1.5, expand_method=1, shrink_ratio=0.4):
        # expand_method:
        # 0: constant
        # 1: log_a
        poly = Polygon(box)
        poly_len = poly.length
        poly_area = poly.area
        if poly_area <= 1.0 or poly_len <= 1.0:
            # polygon area/length is too small to apply the log_a method
            expand_method = 0

        if expand_method == 1:
            if shrink_ratio == 0.3:
                # ~87.22% within 0.2 diff
                a = [-45.598, 4.145, -2.596, 21.623, 21.029]
            elif shrink_ratio == 0.4:
                # ~92.16% within 0.2 diff
                a = [-24.109, 2.338, -1.395, 12.446, 10.626]
            elif shrink_ratio == 0.5:
                # ~96.42% within 0.2 diff
                a = [-6.607, 1.102, -0.589, 3.919, 2.657]
            elif shrink_ratio == 0.6:
                # ~98.77% within 0.2 diff
                a = [2.546, 0.406, -0.169, -0.976, -1.2]
            elif shrink_ratio == 0.7:
                # ~100% within 0.2 diff
                a = [4.091, 0.105, -0.014, -1.881, -1.747]
            else:
                raise NotImplementedError(
                    f"Unsupported shrink ratio {shrink_ratio} for expand_method {expand_method}"
                )
            log_len = math.log(poly_len)
            log_area = math.log(poly_area)
            expand_ratio = (
                a[0]
                + a[1] * log_len
                + a[2] * log_area
                + a[3] * (log_len / log_area)
                + a[4] * (log_area / log_len)
            )
            if expand_ratio < 0.1:
                expand_ratio = 0.1
            elif expand_ratio > 10.0:
                expand_ratio = 10.0
        distance = poly_area * expand_ratio / poly_len
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score(self, bitmap, box):
        """
        naive version of box score computation,
        only for helping principle understand.
        """
        mask = np.zeros_like(bitmap, dtype=np.uint8)
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap, mask)[0]

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, 4, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


class SEGLossComputation(object):
    """
    This class computes the SEG loss.
    """

    def __init__(self, cfg):
        self.eps = 1e-6
        self.cfg = cfg
        self.loss_name = cfg.MODEL.SEG.LOSS

        if self.loss_name == "Dice":
            self.loss_function = self.dice_loss
        elif self.loss_name == "BCE":
            self.torch_bce_loss = torch.nn.BCELoss()
            self.loss_function = self.bce_loss
        elif self.loss_name == "Mixed":
            self.torch_bce_loss = torch.nn.BCELoss()
            self.loss_function = self.mixed_loss
        else:
            raise ValueError(f"Unknown loss name for segmentation module: {self.loss_name}")

        self.counter = 0

    def __call__(self, preds, targets):
        """
        Arguments:
            preds (Tensor)
            targets (list[Tensor])
            masks (list[Tensor])
        Returns:
            seg_loss (Tensor)
        """
        image_size = (preds.shape[2], preds.shape[3])
        segm_targets, masks = self.prepare_targets(targets, image_size)
        device = preds.device
        segm_targets = segm_targets.float().to(device)
        masks = masks.float().to(device)
        seg_loss = self.loss_function(preds, segm_targets, masks)
        return seg_loss

    def mixed_loss(self, pred, gt, m):
        dc_wt = 0.1 * int(self.counter / 10000)
        if dc_wt > 1.0:
            dc_wt = 1.0
        bc_wt = 1 - dc_wt
        loss = bc_wt * self.bce_loss(pred, gt, m) + dc_wt * self.dice_loss(pred, gt, m)
        self.counter += 1
        return loss

    def bce_loss(self, pred, gt, m):
        # if self.counter % 1000 == 0:
        #     from virtual_fs.virtual_io import open
        #     dbg_info = {
        #         "pred": pred,
        #         "gt": gt,
        #         "m": m,
        #     }
        #     dbg_file = f"manifold://ocr_vll/tree/tmp/dbg_bce_{self.counter // 1000}.pt"
        #     with open(dbg_file, "wb") as buffer:
        #         torch.save(dbg_info, buffer)

        #     print(f"[Debug] Saved bce_loss debug info to {dbg_file}.")

        #     if self.counter >= 5000:
        #         self.counter = 0
        # self.counter += 1

        loss = self.torch_bce_loss(pred, gt)
        return loss

    def dice_loss(self, pred, gt, m):
        intersection = torch.sum(pred * gt * m)
        union = torch.sum(pred * m) + torch.sum(gt * m) + self.eps
        loss = 1 - 2.0 * intersection / union

        if torch.isnan(loss):
            print("[Debug] NaN detected:")
            print("[Debug] pred = {}".format(pred))
            print("[Debug] gt = {}".format(gt))
            print("[Debug] m = {}".format(m))
            print("[Debug] intersection = {}".format(intersection))
            print("[Debug] union = {}".format(union))
            raise Exception("NaN detected!")
        return loss

    def project_masks_on_image(self, mask_polygons, labels, shrink_ratio, image_size):
        seg_map, training_mask = mask_polygons.convert_seg_map(
            labels, shrink_ratio, image_size, self.cfg.MODEL.SEG.IGNORE_DIFFICULT
        )
        return torch.from_numpy(seg_map), torch.from_numpy(training_mask)

    def prepare_targets(self, targets, image_size):
        segms = []
        training_masks = []
        for target_per_image in targets:
            segmentation_masks = target_per_image.get_field("masks")
            labels = target_per_image.get_field("labels")
            seg_maps_per_image, training_masks_per_image = self.project_masks_on_image(
                segmentation_masks, labels, self.cfg.MODEL.SEG.SHRINK_RATIO, image_size
            )
            segms.append(seg_maps_per_image)
            training_masks.append(training_masks_per_image)
        return torch.stack(segms), torch.stack(training_masks)


@PROPOSAL_GENERATOR_REGISTRY.register()
class SPN(nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and RPN
    proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(SPN, self).__init__()
        self.cfg = cfg.clone()

        self.head = SEGHead(in_channels=cfg.MODEL.BACKBONE.OUT_CHANNELS, cfg=cfg)
        self.box_selector_train = SEGPostProcessor(cfg=cfg, is_train=True)
        self.box_selector_test = SEGPostProcessor(cfg=cfg, is_train=False)
        self.loss_evaluator = SEGLossComputation(cfg)

        if cfg.MODEL.SEG.FROZEN:
            print("[Info] Freezing SEG head.")
            if not cfg.MODEL.SEG.BN_FROZEN:
                print("[Warning] BatchNorm in SEG head is not frozen.")
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Tensor): fused feature from FPN
            targets (Tensor): segmentaion gt map

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        preds, fuse_feature = self.head(features)
        # anchors = self.anchor_generator(images, features)
        image_shapes = images.get_sizes()
        if self.training:
            return self._forward_train(preds, targets, image_shapes), [fuse_feature]
        else:
            return self._forward_test(preds, image_shapes), [fuse_feature]

    def _forward_train(self, preds, targets, image_shapes):
        # Segmentation map must be transformed into boxes for detection.
        # sampled into a training batch.
        if self.cfg.MODEL.TRAIN_DETECTION_ONLY:
            # boxes would not be used when TRAIN_DETECTION_ONLY == True,
            # so we can skip the post processor for it.
            boxes = None
        else:
            with torch.no_grad():
                boxes = self.box_selector_train(preds, image_shapes, targets)

        loss_seg = self.loss_evaluator(preds, targets)
        losses = {"loss_seg": loss_seg}
        return boxes, losses

    def _forward_test(self, preds, image_shapes):
        # torch.cuda.synchronize()
        # start_time = time.time()
        boxes, rotated_boxes, polygons, scores = self.box_selector_test(preds, image_shapes)

        if self.cfg.MODEL.ROI_HEADS.NAME == "MaskROIHead":
            # when there's no box head, add the scores field
            for boxes_per_image, scores_per_image in zip(boxes, scores):
                boxes_per_image.add_field("scores", scores_per_image)

        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('post time:', end_time - start_time)
        seg_results = {
            "rotated_boxes": rotated_boxes,
            "polygons": polygons,
            "preds": preds,
            "scores": scores,
        }
        return boxes, seg_results
