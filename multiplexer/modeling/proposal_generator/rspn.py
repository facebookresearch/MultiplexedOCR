import cv2
import numpy as np
import torch
from torch import nn

from multiplexer.structures.bounding_box import BoxList
from multiplexer.structures.boxlist_ops import cat_boxlist, cat_boxlist_gt
from multiplexer.structures.rotated_box_list import RotatedBoxList
from multiplexer.structures.segmentation_mask import SegmentationMask

from .build import PROPOSAL_GENERATOR_REGISTRY
from .spn import SPN, SEGHead, SEGLossComputation, SEGPostProcessor


class RotatedSEGPostProcessor(SEGPostProcessor):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(self, cfg, is_train):
        super(RotatedSEGPostProcessor, self).__init__(cfg, is_train)

    def add_gt_proposals(self, proposals, targets, device):
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
            for gt_box in gt_boxes:
                polygons = gt_box.extra_fields["masks"].polygons
                rotated_boxes_5d = [
                    self.cast_pts_to_xywha(polygon.polygons[0]) for polygon in polygons
                ]
                rotated_boxes_5d = torch.tensor(rotated_boxes_5d, device=device)
                gt_box.add_field("rotated_boxes_5d", RotatedBoxList(rotated_boxes_5d))
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
        rotated_boxes_5d = []
        polygons = []
        contours_all = []
        for i in range(bitmap.shape[0]):
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
            points, sside, rotated_box_5d = self.get_mini_boxes(contour)
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
            rotated_box, _, rotated_box_5d = self.get_mini_boxes(box.reshape(-1, 1, 2))
            rotated_box = np.array(rotated_box)
            rotated_boxes.append(rotated_box)
            polygons.append([polygon])
            rotated_boxes_5d.append(rotated_box_5d)
        if len(boxes) == 0:
            boxes = [torch.from_numpy(np.array([0, 0, 0, 0]))]
            scores = [0.0]
            polygons = [[np.array([[0, 0], [0, 0], [0, 0]])]]
            rotated_boxes = [np.array([[0, 0], [0, 0], [0, 0], [0, 0]])]
            rotated_boxes_5d = [torch.tensor([0, 0, 0, 0, 0])]

        boxes = torch.stack(boxes)
        scores = torch.from_numpy(np.array(scores))
        rotated_boxes_5d = torch.stack(rotated_boxes_5d)
        return boxes, scores, rotated_boxes, polygons, rotated_boxes_5d

    # Assuming input is a list of points, containing at least
    # 4 points (8 coords), but can be more
    # More detail on usage is shown via a test case: https://fburl.com/diffusion/x7ulz68b
    def cast_pts_to_xywha(self, pts):
        # Cast the input points coords to approx. rotated rectangle
        # Uses the OPENCV implementation that computer outer rectangle with
        # minimum area for a given set of points.
        num_pts = len(pts)
        # make sure there're at least 4 points, and come in paired coords
        assert num_pts >= 8 and num_pts % 2 == 0
        pts = np.int0(np.around(pts)).reshape([-1, 2])
        rect = cv2.minAreaRect(pts)

        x, y, w, h, a = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]

        # Orientation is determined by the first two points in the annotation
        a_o = math.atan2((pts[1][1] - pts[0][1]), (pts[1][0] - pts[0][0]))
        a_o = a_o * 180 / np.pi
        # choose the rotated rectangle angle as the one close to polygon orientation
        # Generate the 4 angles of rotated rectangle
        rec_or = np.ones(4) * a + np.arange(4) * 90.0
        rec_or = -360 * (rec_or > 180) + rec_or  # convert range to -180 - 180
        a_ind = np.argmin(np.absolute(rec_or - a_o))
        a = rec_or[a_ind]
        if a_ind == 1 or a_ind == 3:
            # Since the orientaion is changed, flip the rectangle
            w, h = h, w

        # -a to be consistent with our representation :
        # https://fburl.com/diffusion/t8oty9r8
        return [x, y, w, h, -a]

    def convert_cv_rotated_box_to_torch_tensor(self, rbox5):
        cnt_x, cnt_y, w, h, angle = (
            rbox5[0][0],
            rbox5[0][1],
            rbox5[1][0],
            rbox5[1][1],
            -rbox5[2],
        )
        if w * 1.5 < h:
            t = w
            w = h
            h = t
            angle = angle - 90
        angle = (angle + 90) % 180 - 90
        return torch.tensor([cnt_x, cnt_y, w, h, angle])

    def forward(self, seg_output, image_shapes, targets=None):
        """
        Arguments:
            seg_output: list[tensor]

        Returns:
            boxlists (list[BoxList]): bounding boxes
        """
        device = seg_output.device
        (
            boxes_batch,
            rotated_boxes_batch,
            polygons_batch,
            scores_batch,
        ) = self.forward_for_single_feature_map(seg_output, image_shapes)
        if not self.training:
            return (
                boxes_batch,
                rotated_boxes_batch,
                polygons_batch,
                scores_batch,
            )

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxes_batch = self.add_gt_proposals(boxes_batch, targets, device=device)
        return boxes_batch

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
            (boxes, scores, rotated_boxes, polygons, rotated_boxes_5d,) = self.boxes_from_bitmap(
                pred_map_numpy[batch_index], bitmap_numpy[batch_index], width, height
            )
            boxes = boxes.to(device)
            rotated_boxes_5d = rotated_boxes_5d.to(device)
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
                # print(rotated_boxes_5d)
                rotated_boxes_5d_list = RotatedBoxList(
                    rotated_boxes_5d, (image_shape[1], image_shape[0])
                )
                boxlist.add_field("masks", masks)
                boxlist.add_field("rotated_boxes_5d", rotated_boxes_5d_list)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # boxlist = remove_small_boxes(boxlist, self.min_size)
            boxes_batch.append(boxlist)
            rotated_boxes_batch.append(rotated_boxes)
            polygons_batch.append(polygons)
            scores_batch.append(scores)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print('loop time:', end_time - start_time)
        return (
            boxes_batch,
            rotated_boxes_batch,
            polygons_batch,
            scores_batch,
        )

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
        return (
            box,
            min(bounding_box[1]),
            self.convert_cv_rotated_box_to_torch_tensor(bounding_box),
        )


@PROPOSAL_GENERATOR_REGISTRY.register()
class RSPN(SPN):
    """
    Rotated SPN, using RotatedSEGPostProcessor instead of SEGPostProcessor
    """

    def __init__(self, cfg):
        super(RSPN, self).__init__(cfg)

    def init_box_selectors(self):
        self.box_selector_train = RotatedSEGPostProcessor(cfg=self.cfg, is_train=True)
        self.box_selector_test = RotatedSEGPostProcessor(cfg=self.cfg, is_train=False)


#     def forward(self, images, features, targets=None):
#         """
#         Arguments:
#             images (ImageList): images for which we want to compute the predictions
#             features (Tensor): fused feature from FPN
#             targets (Tensor): segmentaion gt map

#         Returns:
#             boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
#                 image.
#             losses (dict[Tensor]): the losses for the model during training. During
#                 testing, it is an empty dict.
#         """
#         preds, fuse_feature = self.head(features)
#         # anchors = self.anchor_generator(images, features)
#         image_shapes = images.get_sizes()
#         if self.training:
#             return self._forward_train(preds, targets, image_shapes), [fuse_feature]
#         else:
#             return self._forward_test(preds, image_shapes), [fuse_feature]

#     def _forward_train(self, preds, targets, image_shapes):
#         # Segmentation map must be transformed into boxes for detection.
#         # sampled into a training batch.
#         if self.cfg.MODEL.TRAIN_DETECTION_ONLY:
#             # boxes would not be used when TRAIN_DETECTION_ONLY == True,
#             # so we can skip the post processor for it.
#             boxes = None
#         else:
#             with torch.no_grad():
#                 boxes = self.box_selector_train(preds, image_shapes, targets)

#         loss_seg = self.loss_evaluator(preds, targets)
#         losses = {"loss_seg": loss_seg}
#         return boxes, losses

#     def _forward_test(self, preds, image_shapes):
#         # torch.cuda.synchronize()
#         # start_time = time.time()
#         boxes, rotated_boxes, polygons, scores = self.box_selector_test(preds, image_shapes)

#         if self.cfg.MODEL.ROI_HEADS.NAME == "MaskROIHead":
#             # when there's no box head, add the scores field
#             for boxes_per_image, scores_per_image in zip(boxes, scores):
#                 boxes_per_image.add_field("scores", scores_per_image)

#         # torch.cuda.synchronize()
#         # end_time = time.time()
#         # print('post time:', end_time - start_time)
#         seg_results = {
#             "rotated_boxes": rotated_boxes,
#             "polygons": polygons,
#             "preds": preds,
#             "scores": scores,
#         }
#         return boxes, seg_results
