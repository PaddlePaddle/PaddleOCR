import numpy as np
import os
import subprocess
import json
import argparse


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--gt_file", type=str, default="")
    parser.add_argument("--log_file", type=str, default="")
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def run_shell_command(cmd):
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()

    if p.returncode == 0:
        return out.decode('utf-8')
    else:
        return None


def parser_results_from_log_by_name(log_path, names_list):
    if not os.path.exists(log_path):
        raise ValueError("The log file {} does not exists!".format(log_path))

    if names_list is None or len(names_list) < 1:
        return []

    parser_results = {}
    for name in names_list:
        cmd = "grep {} {}".format(name, log_path)
        outs = run_shell_command(cmd)
        outs = outs.split("\n")[0]
        result = outs.split("{}".format(name))[-1]
        result = json.loads(result)
        parser_results[name] = result
    return parser_results


def load_gt_from_file(gt_file):
    if not os.path.exists(gt_file):
        raise ValueError("The log file {} does not exists!".format(gt_file))
    with open(gt_file, 'r') as f:
        data = f.readlines()
        f.close()
    parser_gt = {}
    for line in data:
        image_name, result = line.strip("\n").split("\t")
        result = json.loads(result)
        parser_gt[image_name] = result
    return parser_gt


def testing_assert_allclose(dict_x, dict_y, atol=1e-7, rtol=1e-7):
    for k in dict_x:
        np.testing.assert_allclose(
            np.array(dict_x[k]), np.array(dict_y[k]), atol=atol, rtol=rtol)


if __name__ == "__main__":
    # Usage:
    # python3.7 tests/compare_results.py --gt_file=./det_results_gpu_fp32.txt  --log_file=./test_log.log

    args = parse_args()
    gt_dict = load_gt_from_file(args.gt_file)
    key_list = list(gt_dict.keys())

    pred_dict = parser_results_from_log_by_name(args.log_file, key_list)

    testing_assert_allclose(gt_dict, pred_dict, atol=args.atol, rtol=args.rtol)
