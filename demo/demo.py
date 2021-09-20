import argparse
import os, sys

from multiplexer.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Multiplexer Demo")
    parser.add_argument("--config-file", type=str, default='configs/seg_rec_poly_fuse_feature_once.yaml')
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
    print(cfg)