#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging


def setup_logger(name, src, result_path, filename="log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(result_path, filename)
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    info_file_handler = logging.FileHandler(log_path)
    info_file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler)
    logger.info(src)
    with open(src) as f:
        logger.info(f.read())
    return logger
