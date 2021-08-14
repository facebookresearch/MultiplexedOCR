# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import torch
import time
from PIL import Image

from multiplexer.config import cfg
from multiplexer.config.parser import override_cfg_from_arg_opts
from multiplexer.checkpoint import DetectionCheckpointer
from multiplexer.data import make_data_loader
from multiplexer.data.transforms import build_transforms
from multiplexer.engine.text_inference import compute_result_logs, load_image, render_box_multi_text
from multiplexer.modeling import build_model
from multiplexer.structures.image_list import to_image_list

from virtual_fs import virtual_os as os


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Multiplexer Demo")
    parser.add_argument(
        "--config-file", type=str, default="configs/seg_rec_poly_fuse_feature_once.yaml"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="A file or directory to save output visualizations. "
    )
    parser.add_argument(
        "opts",
        help="See config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    print("Args: {}".format(args))

    cfg.merge_from_file(args.config_file)
    
    override_cfg_from_arg_opts(cfg, args)
    
    opts = []
    
    if cfg.MODEL.DEVICE == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] cuda not available, using cpu as device")
            opts += ["MODEL.DEVICE", "cpu"]
            
    cfg.merge_from_list(opts)
    
    print(cfg)
    
    print(f"[Info] Using weight: {cfg.MODEL.WEIGHT}")
    
    model = build_model(cfg)
    model = model.to(cfg.MODEL.DEVICE)
    
    checkpointer = DetectionCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    
    model.eval()
    
    transforms = build_transforms(cfg, False)
    
    print("Loading input image ...")
    start = time.time()

    image = load_image(args.input)
    
    img, _ = transforms(image, None)
    
    images = to_image_list(img, cfg.DATALOADER.SIZE_DIVISIBILITY)

    images = images.to(cfg.MODEL.DEVICE)
    print("Total time loading the image: {:.3f} s".format(time.time() - start))
    print(f"Running model on {args.input} ...")
    start = time.time()
    with torch.no_grad():
        prediction_dict = model(images)
    print("Total time running the model: {:.3f} s".format(time.time() - start))


    global_prediction = prediction_dict["global_prediction"][0]
    test_image_width, test_image_height = global_prediction.size
    # print(test_image_width, test_image_height)

    img = load_image(args.input)
    width, height = img.size
    resize_ratio = float(height) / test_image_height
    global_prediction = global_prediction.resize((width, height))
    prediction_dict["rotated_boxes_5d"] = None
    prediction_dict["boxes"] = global_prediction.bbox.tolist()
    use_seg_poly = cfg.MODEL.SEG.USE_SEG_POLY
    if cfg.MODEL.TRAIN_DETECTION_ONLY:
        prediction_dict["scores"] = [1.0 for _ in range(len(global_prediction))]  # dummy
    #     if not cfg.MODEL.SEG.USE_SEG_POLY:
    #         masks = global_prediction.get_field("mask").cpu().numpy()
    #     else:
        use_seg_poly = True
        prediction_dict["masks"] = global_prediction.get_field("masks").get_polygons()
    else:
        if cfg.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX:
            prediction_dict["scores"] = global_prediction.get_field("scores").tolist()
        if not use_seg_poly:
            prediction_dict["masks"] = global_prediction.get_field("mask").cpu().numpy()
        else:
            prediction_dict["masks"] = global_prediction.get_field("masks").get_polygons()


    polygon_format = "polygon"
    result_logs_dict = compute_result_logs(
        prediction_dict=prediction_dict,
        cfg=cfg,
        img=img,
        polygon_format=polygon_format,
        use_seg_poly=use_seg_poly,
    )
    result_logs = result_logs_dict["result_logs"]

    img_vis = img.copy()
    render_box_multi_text(
        cfg=cfg,
        image=img_vis,
        result_logs_dict=result_logs_dict,
        resize_ratio=resize_ratio,
    )
    if os.path.isdir(args.output):
        out_filename = os.path.join(args.output, "vis_" + os.path.basename(args.input))
    else:
        out_filename = args.output
        
    print(f"[Info] Saving visualization to {out_filename}")
    img_vis.save(out_filename)



