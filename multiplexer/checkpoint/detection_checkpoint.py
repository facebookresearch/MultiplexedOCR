# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging

import torch
from torch.nn import functional as F

from multiplexer.model_zoo import cache_url

# from multiplexer.utils.imports import import_file
from multiplexer.utils.languages import get_language_config
from virtual_fs import virtual_os as os
from virtual_fs.virtual_io import open

from .c2_model_loading import load_c2_format
from .model_serialization import load_state_dict, strip_prefix_if_present


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with open(save_file, "wb") as buffer:
            torch.save(data, buffer)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, resume=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if resume:
            if "optimizer" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))
                self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
            if "scheduler" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        with open(f, "rb") as buffer:
            return torch.load(buffer, map_location=torch.device("cpu"))
        # temp_dir = tempfile.mkdtemp(prefix="/tmp/ufs/")
        # local_file_path = os.path.join(temp_dir, os.path.basename(f))
        # shutil.copy2(f, local_file_path)

        # return torch.load(local_file_path, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectionCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectionCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        # if f.startswith("catalog://"):
        #     paths_catalog = import_file(
        #         "d2ocr.config.paths_catalog", self.cfg.PATHS_CATALOG, True
        #     )
        #     catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
        #     self.logger.info("{} points to {}".format(f, catalog_f))
        #     f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native checkpoint
        loaded = super(DetectionCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def enforce_shape(self, model, name, new_shape, backup=None):
        if name not in model:
            if backup is None:
                self.logger.info("{} doesn't exist in the checkpoint, skipped.".format(name))
                return
            elif backup not in model:
                self.logger.info(
                    "{}, as well as backup {}, doesn't exist in the checkpoint, skipped.".format(
                        name, backup
                    )
                )
                return
            else:
                self.logger.info(
                    "{} doesn't exist in the checkpoint, clone from backup {}.".format(name, backup)
                )
                model[name] = model[backup].clone().detach()

        old_shape = list(model[name].shape)

        assert len(old_shape) == len(new_shape), "\n".join(
            [
                "Dimensions for the shapes for {} should be the same!".format(name),
                "len(old_shape): {}".format(len(old_shape)),
                "len(new_shape): {}".format(len(new_shape)),
            ]
        )

        if new_shape == old_shape:
            return

        pad_mode = "constant"  # alternatives: 'replicate', etc.
        dim = len(new_shape)

        # narrow one shrinking dimension at a time
        for d in range(dim):
            if new_shape[d] < old_shape[d]:
                middle_shape = old_shape.copy()
                middle_shape[d] = new_shape[d]

                self.logger.info(
                    " ".join(
                        [
                            "[Checkpointer] Shape of {} changed:".format(name),
                            "Narrowing from {}".format(old_shape),
                            "to {}".format(middle_shape),
                        ]
                    )
                )

                model[name] = torch.narrow(input=model[name], dim=d, start=0, length=new_shape[d])
                old_shape = middle_shape

        if new_shape == old_shape:
            return
        # pad all expanding dimensions at once
        middle_shape = old_shape.copy()
        pad = [0] * (dim * 2)  # each dimension has two padding directions
        for d in range(dim):
            if new_shape[d] > old_shape[d]:
                middle_shape[d] = new_shape[d]
                # the padding dimension is reversed, thus dim - d - 1 below:
                pad[(dim - d - 1) * 2 + 1] = new_shape[d] - old_shape[d]

        self.logger.info(
            " ".join(
                [
                    "[Checkpointer] Shape of {} changed:".format(name),
                    "Padding from {}".format(old_shape),
                    "to {}".format(middle_shape),
                ]
            )
        )
        model[name] = F.pad(input=model[name], pad=tuple(pad), mode=pad_mode, value=0)

    def enforce_seq_shape(self, checkpoint_model, language):

        language_config = get_language_config(self.cfg, language)
        arch = language_config.ARCH
        C = language_config.NUM_CHAR + 2
        E = language_config.EMBED_SIZE
        H = language_config.HIDDEN_SIZE
        Width = int(self.cfg.SEQUENCE.RESIZE_WIDTH / 2)
        Height = int(self.cfg.SEQUENCE.RESIZE_HEIGHT / 2)

        if arch.startswith("ctc_"):
            lstm0_c = 256

            if arch.startswith("ctc_lstm"):
                # ctc_lstm (aka lstm-only) doesn't need seq_encoder
                # ctc_lstm_res4 uses res_layer4 as seq_encoder
                pre_lstm_kernel_height = 3
                pre_lstm_conv_c = 512

                if arch == "ctc_lstm_a":
                    lstm0_c = 512
            elif arch == "ctc_transformer":
                pre_lstm_kernel_height = 3
                pre_lstm_conv_c = 512
            else:
                pre_lstm_kernel_height = self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_H // 8
                pre_lstm_conv_c = 256
                # for ctc heads, the first layer of seq_encoder does not involve hidden_size
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.seq_encoder.0.weight".format(language),
                    new_shape=[256, 256, 3, 3],
                    backup="roi_heads.mask.predictor.seq.seq_encoder.0.weight",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.seq_encoder.0.bias".format(language),
                    new_shape=[256],
                    backup="roi_heads.mask.predictor.seq.seq_encoder.0.bias",
                )

            # pre_lstm_conv
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.pre_lstm_conv.weight".format(language),
                new_shape=[lstm0_c, pre_lstm_conv_c, pre_lstm_kernel_height, 1],
                backup="roi_heads.mask.predictor.seq.pre_lstm_conv.weight",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.pre_lstm_conv.bias".format(language),
                new_shape=[lstm0_c],
                backup="roi_heads.mask.predictor.seq.pre_lstm_conv.bias",
            )

            if arch.startswith("ctc_lstm"):
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.weight_ih_l0".format(language),
                    new_shape=[4 * H, lstm0_c],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.weight_ih_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.weight_hh_l0".format(language),
                    new_shape=[4 * H, H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.weight_hh_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.bias_ih_l0".format(language),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.bias_ih_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.bias_hh_l0".format(language),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.bias_hh_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.weight_ih_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H, lstm0_c],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.weight_ih_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.weight_hh_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H, H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.weight_hh_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.bias_ih_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.bias_ih_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.rnn.bias_hh_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.rnn.bias_hh_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.fc.weight".format(language),
                    new_shape=[H, 2 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.fc.weight",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.0.fc.bias".format(language),
                    new_shape=[H],
                    backup="roi_heads.mask.predictor.seq.lstm.0.fc.bias",
                )

                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.weight_ih_l0".format(language),
                    new_shape=[4 * H, H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.weight_ih_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.weight_hh_l0".format(language),
                    new_shape=[4 * H, H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.weight_hh_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.bias_ih_l0".format(language),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.bias_ih_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.bias_hh_l0".format(language),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.bias_hh_l0",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.weight_ih_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H, H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.weight_ih_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.weight_hh_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H, H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.weight_hh_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.bias_ih_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.bias_ih_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.rnn.bias_hh_l0_reverse".format(
                        language
                    ),
                    new_shape=[4 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.rnn.bias_hh_l0_reverse",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.fc.weight".format(language),
                    new_shape=[C - 1, 2 * H],
                    backup="roi_heads.mask.predictor.seq.lstm.1.fc.weight",
                )
                self.enforce_shape(
                    model=checkpoint_model,
                    name="roi_heads.mask.predictor.seq_{}.lstm.1.fc.bias".format(language),
                    new_shape=[C - 1],
                    backup="roi_heads.mask.predictor.seq.lstm.1.fc.bias",
                )
        else:
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.embedding.weight".format(
                    language
                ),
                new_shape=[C, E],
                backup="roi_heads.mask.predictor.seq.seq_decoder.embedding.weight",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.word_linear.weight".format(
                    language
                ),
                new_shape=[H, E],
                backup="roi_heads.mask.predictor.seq.seq_decoder.word_linear.weight",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.word_linear.bias".format(
                    language
                ),
                new_shape=[H],
                backup="roi_heads.mask.predictor.seq.seq_decoder.word_linear.bias",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.attn.v".format(language),
                new_shape=[H],
                backup="roi_heads.mask.predictor.seq.seq_decoder.attn.v",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.attn.attn.weight".format(
                    language
                ),
                new_shape=[H, 2 * H + Width + Height],
                backup="roi_heads.mask.predictor.seq.seq_decoder.attn.attn.weight",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.attn.attn.bias".format(language),
                new_shape=[H],
                backup="roi_heads.mask.predictor.seq.seq_decoder.attn.attn.bias",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.rnn.weight_ih".format(language),
                new_shape=[3 * H, 2 * H + Width + Height],
                backup="roi_heads.mask.predictor.seq.seq_decoder.rnn.weight_ih",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.rnn.weight_hh".format(language),
                new_shape=[3 * H, H],
                backup="roi_heads.mask.predictor.seq.seq_decoder.rnn.weight_hh",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.rnn.bias_ih".format(language),
                new_shape=[3 * H],
                backup="roi_heads.mask.predictor.seq.seq_decoder.rnn.bias_ih",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.rnn.bias_hh".format(language),
                new_shape=[3 * H],
                backup="roi_heads.mask.predictor.seq.seq_decoder.rnn.bias_hh",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.out.weight".format(language),
                new_shape=[C, H],
                backup="roi_heads.mask.predictor.seq.seq_decoder.out.weight",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_decoder.out.bias".format(language),
                new_shape=[C],
                backup="roi_heads.mask.predictor.seq.seq_decoder.out.bias",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_encoder.0.weight".format(language),
                new_shape=[H, 256, 3, 3],
                backup="roi_heads.mask.predictor.seq.seq_encoder.0.weight",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.seq_encoder.0.bias".format(language),
                new_shape=[H],
                backup="roi_heads.mask.predictor.seq.seq_encoder.0.bias",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.x_onehot.weight".format(language),
                new_shape=[Width, Width],
                backup="roi_heads.mask.predictor.seq.x_onehot.weight",
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.seq_{}.y_onehot.weight".format(language),
                new_shape=[Height, Height],
                backup="roi_heads.mask.predictor.seq.y_onehot.weight",
            )

    def _load_model(self, checkpoint):
        checkpoint_model = checkpoint.pop("model")

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        checkpoint_model = strip_prefix_if_present(checkpoint_model, prefix="module.")

        cfg = self.cfg

        C = cfg.SEQUENCE.NUM_CHAR + 2
        E = cfg.SEQUENCE.EMBED_SIZE
        H = cfg.SEQUENCE.HIDDEN_SIZE
        L = cfg.MODEL.LANGUAGE_HEAD.NUM_CLASSES
        M = cfg.MODEL.ROI_MASK_HEAD.MASK_FCN_INPUT_DIM
        Width = int(cfg.SEQUENCE.RESIZE_WIDTH / 2)
        Height = int(cfg.SEQUENCE.RESIZE_HEIGHT / 2)

        # assert C == cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES + 1

        # language head
        lang_arch = cfg.MODEL.LANGUAGE_HEAD.PREDICTOR

        input_h = cfg.MODEL.LANGUAGE_HEAD.INPUT_H
        input_w = cfg.MODEL.LANGUAGE_HEAD.INPUT_W
        input_c = cfg.MODEL.LANGUAGE_HEAD.INPUT_C
        conv1_c = cfg.MODEL.LANGUAGE_HEAD.CONV1_C
        conv2_c = cfg.MODEL.LANGUAGE_HEAD.CONV2_C

        if lang_arch == "V1LanguagePredictor":
            h_divisible = 8
            w_divisible = 8
        elif lang_arch == "V2LanguagePredictor":
            h_divisible = 3
            w_divisible = 8
        elif lang_arch in [
            "V3LanguagePredictor",
            "V4LanguagePredictor",
            "V5LanguagePredictor",
        ]:
            h_divisible = 3
            w_divisible = 4

        assert input_h % h_divisible == 0
        assert input_w % w_divisible == 0

        fc1_in = (input_h // h_divisible) * (input_w // w_divisible) * conv2_c

        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.language_predictor.fc1.weight",
            new_shape=[64, fc1_in],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.language_predictor.fc2.weight",
            new_shape=[L, 64],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.language_predictor.fc2.bias",
            new_shape=[L],
        )

        if lang_arch in [
            "V2LanguagePredictor",
            "V3LanguagePredictor",
            "V4LanguagePredictor",
            "V5LanguagePredictor",
        ]:
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.language_predictor.conv1.weight",
                new_shape=[conv1_c, input_c, 3, 2],
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.language_predictor.conv1.bias",
                new_shape=[conv1_c],
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.language_predictor.conv2.weight",
                new_shape=[conv2_c, conv1_c, 1, 2],
            )
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.language_predictor.conv2.bias",
                new_shape=[conv2_c],
            )

        # mask head
        if cfg.MODEL.ROI_MASK_HEAD.CONV5_ARCH.startswith("conv"):
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.conv5_mask.weight",
                new_shape=[M, 256, 1, 1],
            )
        elif cfg.MODEL.ROI_MASK_HEAD.CONV5_ARCH.startswith("transpose"):
            self.enforce_shape(
                model=checkpoint_model,
                name="roi_heads.mask.predictor.conv5_mask.weight",
                new_shape=[M, 256, 2, 2],
            )

        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.conv5_mask.bias",
            new_shape=[M],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.mask_fcn_logits.weight",
            new_shape=[1, M, 1, 1],
        )

        # rec head
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.embedding.weight",
            new_shape=[C, E],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.word_linear.weight",
            new_shape=[H, E],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.word_linear.bias",
            new_shape=[H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.attn.v",
            new_shape=[H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.attn.attn.weight",
            new_shape=[H, 2 * H + Width + Height],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.attn.attn.bias",
            new_shape=[H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.rnn.weight_ih",
            new_shape=[3 * H, 2 * H + Width + Height],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.rnn.weight_hh",
            new_shape=[3 * H, H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.rnn.bias_ih",
            new_shape=[3 * H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.rnn.bias_hh",
            new_shape=[3 * H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.out.weight",
            new_shape=[C, H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_decoder.out.bias",
            new_shape=[C],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_encoder.0.weight",
            new_shape=[H, 256, 3, 3],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.seq.seq_encoder.0.bias",
            new_shape=[H],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.char_mask_fcn_logits.bias",
            new_shape=[C - 1],
        )
        self.enforce_shape(
            model=checkpoint_model,
            name="roi_heads.mask.predictor.char_mask_fcn_logits.weight",
            new_shape=[C - 1, 256, 1, 1],
        )

        # individual rec heads
        for language in cfg.SEQUENCE.LANGUAGES_ENABLED:
            self.enforce_seq_shape(checkpoint_model=checkpoint_model, language=language)

        load_state_dict(self.model, checkpoint_model)
