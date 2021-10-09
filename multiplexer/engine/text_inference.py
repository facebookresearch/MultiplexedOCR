import logging

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from multiplexer.structures.word_result import WordResult
from virtual_fs import virtual_os as os

logger = logging.getLogger(__name__)


def compute_result_logs(
    prediction_dict,
    cfg,
    img,
    polygon_format,
    use_seg_poly,
):
    # polygons = []

    width, height = img.size

    word_result_list = prediction_dict["word_result_list"]
    scores = prediction_dict["scores"]
    rotated_boxes_5d = prediction_dict["rotated_boxes_5d"]
    final_word_result_list = []

    for k, box in enumerate(prediction_dict["boxes"]):
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            continue

        box = list(map(int, box))
        if not use_seg_poly:
            mask = prediction_dict["masks"][k, 0, :, :]
            m2p = mask2polygon
            # m2p = (
            #     mask2polygon_cpp_op
            #     if cfg.TEST.MASK2POLYGON_OP == "cpp"
            #     else mask2polygon
            # )
            polygon = m2p(
                mask,
                box,
                img.size,
                threshold=0.5,
                polygon_format=polygon_format,
            )
        else:
            polygon = list(prediction_dict["masks"][k].get_polygons()[0].cpu().numpy())
            if polygon_format == "rect":
                polygon = polygon2rbox(polygon, height, width)

        if polygon is None:
            continue

        # polygons.append(polygon)

        if cfg.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX:
            word_result_list[k].det_score = scores[k]
        else:
            word_result_list[k].det_score = 1.0

        word_result_list[k].box = [int(x * 1.0) for x in box[:4]]
        if len(polygon) != 8:
            if len(polygon) < 8:
                logger.warning("Polygon {} has fewer than 4 points!".format(polygon))
            rbox = polygon2rbox(polygon, height, width)
            word_result_list[k].rotated_box = rbox
            msg = "Polygon {} is also saved in rbox: {}".format(polygon, rbox)
            logger.info(msg)
        word_result_list[k].polygon = polygon

        if rotated_boxes_5d is None:
            poly = np.array(polygon).reshape((-1, 2))
            rect = cv2.minAreaRect(poly)
            x, y, w, h, a = rect[0][0], rect[0][1], rect[1][0], rect[1][1], -rect[2]
            word_result_list[k].rotated_box_5d = [x, y, w, h, a]
        else:
            word_result_list[k].rotated_box_5d = rotated_boxes_5d[k].tolist()

        final_word_result_list.append(word_result_list[k])

    return {
        "result_logs": final_word_result_list,
    }


def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y
    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]


def load_image(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f).convert("RGB")
    return img


def mask2polygon(mask, box, im_size, threshold=0.5, polygon_format="polygon"):
    # mask 32*128
    image_width, image_height = im_size[0], im_size[1]
    box_h = box[3] - box[1]
    box_w = box[2] - box[0]
    cls_polys = (mask * 255).astype(np.uint8)
    poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
    poly_map = poly_map.astype(np.float32) / 255
    poly_map = cv2.GaussianBlur(poly_map, (3, 3), sigmaX=3)
    ret, poly_map = cv2.threshold(poly_map, threshold, 1, cv2.THRESH_BINARY)
    if polygon_format == "polygon":
        SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        poly_map = cv2.erode(poly_map, SE1)
        poly_map = cv2.dilate(poly_map, SE1)
        poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
        try:
            _, contours, _ = cv2.findContours(
                (poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        except Exception:
            contours, _ = cv2.findContours(
                (poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        if len(contours) == 0:
            # print(contours)
            # print(len(contours))
            return None
        max_area = 0
        max_cnt = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt
        # perimeter = cv2.arcLength(max_cnt, True)
        epsilon = 0.01 * cv2.arcLength(max_cnt, True)
        approx = cv2.approxPolyDP(max_cnt, epsilon, True)
        pts = approx.reshape((-1, 2))
        pts[:, 0] = pts[:, 0] + box[0]
        pts[:, 1] = pts[:, 1] + box[1]
        polygon = list(pts.reshape((-1,)))
        polygon = list(map(int, polygon))
        if len(polygon) < 6:
            return None
    else:
        assert polygon_format == "rect", f"Unknown polygon format: {polygon_format}"
        SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        poly_map = cv2.erode(poly_map, SE1)
        poly_map = cv2.dilate(poly_map, SE1)
        poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
        idy, idx = np.where(poly_map == 1)
        xy = np.vstack((idx, idy))
        xy = np.transpose(xy)
        hull = cv2.convexHull(xy, clockwise=True)
        # reverse order of points.
        if hull is None:
            return None
        hull = hull[::-1]
        # find minimum area bounding box.
        rect = cv2.minAreaRect(hull)
        corners = cv2.boxPoints(rect)
        corners = np.array(corners, dtype="int")
        pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
        polygon = [x * 1.0 for x in pts]
        polygon = list(map(int, polygon))
    return polygon


def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2))
    try:
        rect = cv2.minAreaRect(poly)
    except Exception:
        print(f"cv2.minAreaRect failed for polygon {polygon}")
        return None
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = list(map(int, pts))
    return pts


def render_box_multi_text(cfg, image, result_logs_dict, resize_ratio, det_thresh=0.2):
    word_result_list = result_logs_dict["result_logs"]

    draw = ImageDraw.Draw(image, "RGBA")

    color_green = (0, 255, 0, 192)

    renderer = render_box_multi_text

    if not hasattr(renderer, "fonts"):
        renderer.fonts = {0: ImageFont.load_default()}
        print("[Info] fonts initiated.")

    font_path = "/checkpoint/jinghuang/fonts/Arial-Unicode-Regular.ttf"
    if not os.path.exists(font_path):
        use_default_font = True
        logger.warning("[Warning] Font {} doesn't exist, using default.".format(font_path))
    else:
        use_default_font = False

    for word_result in word_result_list:
        # Use line width to illustrate boxes with low confidence
        score_det = word_result.det_score
        score_rec_seq = word_result.seq_score

        if score_det < det_thresh:
            continue

        line_width = 6

        if score_det < 0.05:
            line_width -= 2

        if score_rec_seq < 0.9:
            line_width -= 1

        polygon = word_result.polygon

        min_x = min(polygon[0::2])
        min_y = min(polygon[1::2])
        height = min(word_result.rotated_box_5d[2], word_result.rotated_box_5d[3])

        if use_default_font:
            font_size = 0
        else:
            font_size = max(2, min(10, int(round(0.3 * height / 5)))) * 5

        if font_size in renderer.fonts:
            font = renderer.fonts[font_size]
        else:
            with open(font_path, "rb") as f:
                font = ImageFont.truetype(f, font_size)
            renderer.fonts[font_size] = font

        # r = randint(0, 255)
        # g = randint(0, 255)
        # b = randint(0, 255)

        # draw.polygon(polygon, outline=(0,0,255,128), fill=(0,255,0,64))
        # draw.polygon(polygon, outline=(0,0,255,128), fill=(r, g, b, 96))
        vertices = polygon + [polygon[0], polygon[1]]  # append the 1st vertex
        draw.line(vertices, width=line_width, fill=(255, 0, 0, 255))

        word = f"{word_result.seq_word} [{int(round(score_det*100))}%,{word_result.language}"

        if word_result.language_enabled != word_result.language:
            word += f",{word_result.language_enabled}"

        word += "]"

        draw_loc = (min_x, min_y - font_size)
        text_size = draw.textsize(word, font=font)
        rec_pad = 0
        rec_min = (draw_loc[0] - rec_pad, draw_loc[1] - rec_pad)
        rec_max = (
            draw_loc[0] + text_size[0] + rec_pad,
            draw_loc[1] + text_size[1] + rec_pad,
        )
        draw.rectangle([rec_min, rec_max], fill=color_green)
        draw.text((min_x, min_y - font_size), word, font=font, fill=(0, 0, 0, 255))


def run_model(model, images, cfg):
    prediction_dict = {
        "global_prediction": None,
    }

    cpu_device = torch.device("cpu")

    with torch.no_grad():
        predictions, proposals, seg_results_dict = model(images)
        if cfg.MODEL.TRAIN_DETECTION_ONLY:
            prediction_dict["global_prediction"] = [o.to(cpu_device) for o in predictions]
            assert len(seg_results_dict["scores"]) == 1
            prediction_dict["scores"] = seg_results_dict["scores"][0].to(cpu_device).tolist()
            # Add dummy word result list
            word_result_list = []
            for _ in range(len(prediction_dict["scores"])):
                word_result = WordResult()
                word_result.seq_word = ""
                word_result_list.append(word_result)
            prediction_dict["word_result_list"] = word_result_list
        else:
            prediction_dict["global_prediction"] = [o.to(cpu_device) for o in predictions[0]]

            prediction_dict["word_result_list"] = predictions[1]["word_result_list"]

    return prediction_dict
