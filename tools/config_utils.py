# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import ast
import datetime
import json

import yaml

from virtual_fs import virtual_os as os
from virtual_fs import virtual_tempfile as tempfile
from virtual_fs.virtual_io import open


def create_random_remote_work_dir(base_work_dir, name="ocr"):
    date_string = datetime.date.today().strftime("%Y%m%d")
    base_date_folder = os.path.join(base_work_dir, date_string)
    if not os.path.exists(base_date_folder):
        os.makedirs(base_date_folder)
    prefix = os.path.join(base_date_folder, f"{name}_")
    work_dir = tempfile.mkdtemp(prefix=prefix)
    os.system('chmod 777 "%s"' % work_dir)
    return work_dir


class ExtendedJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return super(ExtendedJsonEncoder, self).default(obj)


def get_dataset_config(dataset_str, dataset_ratios_str):
    if dataset_str is None:
        return {}

    dataset_groups = {
        "mlt19_all": ["mlt19", "mlt19_val"],
    }

    dataset_config = {}
    dataset_config["IGNORE_DIFFICULT"] = False

    group_list = dataset_str.split(":")
    if dataset_ratios_str is not None:
        group_ratios = [float(ratio_str) for ratio_str in dataset_ratios_str.split(":")]
        assert len(group_list) == len(group_ratios)
    else:
        group_ratios = []
        for i in range(len(group_list)):
            if "*" in group_list[i]:
                ds_ratio = group_list[i].split("*")
                assert len(ds_ratio) == 2
                group_list[i] = ds_ratio[0]
                group_ratios.append(float(ds_ratio[1]))
            else:
                group_ratios.append(1.0)

    dataset_ratio_dict = {}

    for i, group_name in enumerate(group_list):
        if group_name in dataset_groups:
            dataset_group = dataset_groups[group_name]
        else:
            # single dataset group not listed in dataset_groups
            dataset_group = [group_name]

        for dataset in dataset_group:
            if dataset in dataset_ratio_dict:
                dataset_ratio_dict[dataset] += group_ratios[i]
            else:
                dataset_ratio_dict[dataset] = group_ratios[i]

    dataset_config["TRAIN"] = []
    dataset_config["TEST"] = []
    dataset_config["RATIOS"] = []

    for dataset in dataset_ratio_dict:
        # if dataset not in dataset_groups["xxx"]:
        #     # datasets in the "xxx" group don't have training sets
        if True:
            dataset_config["TRAIN"].append(dataset + "_train")
            dataset_config["RATIOS"].append(dataset_ratio_dict[dataset])
        dataset_config["TEST"].append(dataset + "_test")

    return dataset_config


def get_language_head_list(language_heads_str):
    if language_heads_str == "none":
        return []
    else:
        return ["{}".format(name) for name in language_heads_str.split(":")]


def get_single_job_cfg_value(value_str, args, decoder="job=value"):
    # Example 1:
    # value_str = "fcc_en=2:nfi_en=3+nfi_fr=2+nfi_nl=1",
    # args.job_list = ['en', 'fr', 'nl'],
    # args.job_id = 1,
    # decoder = "raw"
    # return: "nfi_fr=2"
    #
    # Example 2:
    # value_str = "default=1:en=3",
    # args.job_list = ['en', 'fr', 'nl'],
    # args.job_id = 0,
    # decoder = "job=value"
    # return: "3"
    #
    # Example 3:
    # value_str = "default=1:en=3",
    # args.job_list = ['en', 'fr', 'nl'],
    # args.job_id = 2,
    # decoder = "job=value"
    # return: "1"

    if "+" in value_str:
        value_list = value_str.split("+")
        if decoder == "job=value":
            key_value_dict = {}
            for key_value_pair in value_list:
                key_value_list = key_value_pair.split("=")
                if len(key_value_list) == 1:
                    key = key_value_list[0]
                    key_value_dict[key] = key  # special case: key == value
                    assert key in args.job_list, f"Unknown job-key in {value_str}: {key}"
                else:
                    assert len(key_value_list) == 2, f"Invalid key_value_pair: {key_value_pair}"
                    key = key_value_list[0]
                    key_value_dict[key] = key_value_list[1]
                    if key != "default":
                        assert key in args.job_list, f"Unknown job-key in {value_str}: {key}"

            if args.job_list[args.job_id] in key_value_dict:
                return key_value_dict[args.job_list[args.job_id]]
            else:
                return key_value_dict["default"]
        else:
            value = value_list[args.job_id]
            assert decoder == "raw", f"Unknown cfg value decoder: {decoder}"
            assert len(value_list) == args.job_num
        return value
    else:
        return value_str


def decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    if v == "":
        return v

    try:
        if type(ast.parse(v).body[0].value) is ast.BinOp:
            # This is to avoid cases like
            # ast.literal_eval('2019-10-10')
            # will return 1999 instead of a string
            # See https://bugs.python.org/issue31778
            print("Binary op found in argument value {}, treating it as string".format(v))
            return v
        # Try to interpret `v` as a:
        # string, number, tuple, list, dict, boolean, or None
        v = ast.literal_eval(v)
    except ValueError:
        # ast.literal_eval('2019-06-06')
        pass
    except SyntaxError:
        # ast.literal_eval('2019-06-06')
        pass

    return v


def override_cfg_from_arg_opts(cfg, args):
    # Override config with key-value pairs in arg list (e.g., from command line).
    arg_list = args.opts
    assert len(arg_list) % 2 == 0, str(arg_list)
    for i in range(0, len(arg_list), 2):
        key_str = arg_list[i]
        value_str = get_single_job_cfg_value(arg_list[i + 1], args)
        key_list = key_str.split(".")
        d = cfg  # start from root
        for subkey in key_list[:-1]:
            if subkey not in d:
                d[subkey] = {}
            d = d[subkey]
        subkey = key_list[-1]
        value = decode_cfg_value(value_str)
        d[subkey] = value


def merge_a_into_b(a, b):
    # merge dict a into dict b. values in a will overwrite b
    for k, v in a.items():
        if isinstance(v, dict) and k in b:
            assert isinstance(b[k], dict), "Cannot inherit key '{}' from base!".format(k)
            merge_a_into_b(v, b[k])
        else:
            b[k] = v


def load_yaml_config_recursively(current_yaml):
    with open(current_yaml, "r") as f:
        yaml_string = f.read()
        cfg = yaml.load(yaml_string, Loader=yaml.SafeLoader)

    BASE_KEY = "_BASE_"
    if BASE_KEY in cfg:
        base_yaml = cfg[BASE_KEY]
        if base_yaml.startswith("~"):
            base_yaml = os.path.expanduser(base_yaml)
        if not base_yaml.startswith("/"):
            # the path to base cfg is relative to the config file itself
            base_yaml = os.path.join(os.path.dirname(current_yaml), base_yaml)
        base_cfg = load_yaml_config_recursively(base_yaml)
        del cfg[BASE_KEY]
        merge_a_into_b(cfg, base_cfg)
        return base_cfg
    else:
        return cfg


def create_config_file(args):
    # Load default config from default yaml
    default_yaml = os.path.join(args.yaml_dir, args.yaml)
    cfg = load_yaml_config_recursively(default_yaml)

    # Override the parameters
    if args.dataset is not None:
        dataset_config = get_dataset_config(
            dataset_str=get_single_job_cfg_value(args.dataset, args),
            dataset_ratios_str=args.dataset_ratios,
        )
        cfg["DATASETS"].update(dataset_config)

    if hasattr(args, "min_size_train") and args.min_size_train is not None:
        min_size_train = get_single_job_cfg_value(args.min_size_train, args)
        cfg["INPUT"]["MIN_SIZE_TRAIN"] = [int(size) for size in min_size_train.split(":")]

    if hasattr(args, "solver_steps") and args.solver_steps is not None:
        solver_steps = get_single_job_cfg_value(args.solver_steps, args)
        cfg["SOLVER"]["STEPS"] = [int(step) for step in solver_steps.split(":")]

    if args.language_heads is not None:
        language_heads = get_single_job_cfg_value(args.language_heads, args)
        language_head_list = get_language_head_list(language_heads)
        cfg["SEQUENCE"]["LANGUAGES"] = language_head_list
        cfg["SEQUENCE"]["LANGUAGES_ENABLED"] = language_head_list
        cfg["SEQUENCE"]["NUM_SEQ_HEADS"] = len(language_head_list)
        cfg["MODEL"]["LANGUAGE_HEAD"]["NUM_CLASSES"] = len(language_head_list)

    if args.language_heads_enabled is not None:
        language_heads_enabled = get_single_job_cfg_value(args.language_heads_enabled, args)
        enabled_language_head_list = get_language_head_list(language_heads_enabled)
        cfg["SEQUENCE"]["LANGUAGES_ENABLED"] = enabled_language_head_list
        cfg["SEQUENCE"]["NUM_SEQ_HEADS"] = len(enabled_language_head_list)
        # for lang in enabled_language_head_list:
        #     assert (
        #         lang in language_head_list
        #     ), "Enabled language {} not found in the language head list {}!".format(
        #         lang, language_head_list
        #     )

    if args.unfreezed_seq_heads is not None:
        unfreezed_seq_heads = get_single_job_cfg_value(args.unfreezed_seq_heads, args)
        unfreezed_seq_head_list = get_language_head_list(unfreezed_seq_heads)
        cfg["SEQUENCE"]["LANGUAGES_UNFREEZED"] = unfreezed_seq_head_list

    override_cfg_from_arg_opts(cfg, args)

    if hasattr(args, "train_from_scratch") and args.train_from_scratch:
        cfg["MODEL"]["WEIGHT"] = ""
        print("MODEL.WEIGHT is set to empty to train from scratch.")

    if "OUTPUT_DIR" not in cfg or cfg["OUTPUT_DIR"] == "":
        cfg["OUTPUT_DIR"] = args.work_dir

    # Save updated yaml
    cfg_yaml = os.path.join(args.work_dir, "config.yaml")
    with open(cfg_yaml, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    print("Final YAML config: {}".format(cfg_yaml))
    return cfg_yaml
