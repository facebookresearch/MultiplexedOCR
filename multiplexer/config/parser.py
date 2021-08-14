# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import ast

def get_single_job_cfg_value(value_str, args, decoder="job=value"):
    # Example 1:
    # value_str = "ds1_en=2:ds2_en=3+ds1_fr=2+ds1_nl=1",
    # args.job_list = ['en', 'fr', 'nl'],
    # args.job_id = 1,
    # decoder = "raw"
    # return: "ds1_fr=2"
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
                    assert (
                        key in args.job_list
                    ), f"Unknown job-key in {value_str}: {key}"
                else:
                    assert (
                        len(key_value_list) == 2
                    ), f"Invalid key_value_pair: {key_value_pair}"
                    key = key_value_list[0]
                    key_value_dict[key] = key_value_list[1]
                    if key != "default":
                        assert (
                            key in args.job_list
                        ), f"Unknown job-key in {value_str}: {key}"

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
            print(
                "Binary op found in argument value {}, treating it as string".format(v)
            )
            return v
        # Try to interpret `v` as a:
        # string, number, tuple, list, dict, boolean, or None
        v = ast.literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
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