#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import re
from collections import OrderedDict

from absl import flags
from six.moves import zip
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("task", "regression", "Task to evaluate.")


def parse_log(dataset, log_path):
    with open(log_path, "r") as f:
        text = f.read()
    float_p = r"[+-]?(?:[0-9]*[.])?[0-9]+"
    final_str = text.strip().split("\n")[-1].strip()
    m = re.match(r">> Test.* = ({}), .* = ({})".format(float_p, float_p),
                 final_str)
    test_rmse = float(m.group(1))
    test_ll = float(m.group(2))
    return test_rmse, test_ll


def eval_regression():
    results = OrderedDict()
    root_dir = os.path.join("results", "regression")
    datasets = sorted(os.listdir(root_dir))
    for dataset in datasets:
        results[dataset] = OrderedDict()
        dataset_dir = os.path.join(root_dir, dataset)
        methods = sorted(os.listdir(dataset_dir))
        for method in methods:
            try:
                method_dir = os.path.join(dataset_dir, method)
                sub_dirs = os.listdir(method_dir)
                if sub_dirs[0].startswith("run"):
                    runs = sorted(sub_dirs, key=lambda s: int(s.split("_")[1]))
                    tmp = []
                    for i, run in enumerate(runs):
                        run_dir = os.path.join(method_dir, run)
                        log_path = os.path.join(run_dir, "log")
                        tmp.append(parse_log(dataset, log_path))
                        # if i == 9:
                        #     break
                    test_rmses, test_lls = list(zip(*tmp))
                else:
                    test_rmse, test_ll = parse_log(
                        dataset, os.path.join(method_dir, "log"))
                    test_rmses = [test_rmse]
                    test_lls = [test_ll]
                results[dataset][method] = {}
                results[dataset][method]["test_rmse"] = np.array(test_rmses)
                results[dataset][method]["test_ll"] = np.array(test_lls)
            except Exception:
                continue
    return results


def main():
    results = eval_regression()
    for dataset in results:
        print(dataset)
        for method in results[dataset]:
            rmses = results[dataset][method]["test_rmse"]
            lls = results[dataset][method]["test_ll"]
            print("{}: rmse={:.3f}+-{:.3f}, ll={:.3f}+-{:.3f}"
                  .format(method,
                          np.mean(rmses), np.std(rmses) / np.sqrt(rmses.size),
                          np.mean(lls), np.std(lls) / np.sqrt(lls.size)))
            print(rmses)


if __name__ == "__main__":
    main()
