import math, os, sys
# multiplexer_dir = os.path.abspath('../multiplexer/multiplexer')
multiplexer_dir = "/private/home/jinghuang/code/ocr/multiplexer/"
# the multiplexer_dir could already be there if the kernel was not restarted,
# and we run this cell again
if multiplexer_dir not in sys.path:
    sys.path.append(multiplexer_dir)

from tools.train_net import detectron2_launch, parse_args

# args = parse_args("--config-file /checkpoint/jinghuang/multiplexer/flow/20201111/ocr.1k4fu05s/config.yaml".split())
args = parse_args("--config-file /checkpoint/jinghuang/multiplexer/configs/multiplexer_v1.yaml".split())

args.num_gpus = 8
args.workers = 4
args.dist_url = "auto"
args

# detectron2_launch(args)

# import submitit

# # executor is the submission interface (logs are dumped in the folder)
# executor = submitit.AutoExecutor(folder="/checkpoint/jinghuang/tmp/20211011a")

# nodes = math.ceil(args.num_gpus / 8)
# print(nodes)

detectron2_launch(args)
