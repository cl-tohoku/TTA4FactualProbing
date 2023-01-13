from __future__ import annotations
from typing import Optional
import sys
import os
import logging

def init_logging(logger_name: str, log_dir:str, filename = "info.log"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers == []:

        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # file handler
        fh = logging.FileHandler(str(log_dir) + "/" + filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.WARNING)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.propagate = False

    return logger

def split_list(sequence: list, num_cols = None, num_rows = None):
    assert sequence != None
    sequence_len = len(sequence)
    if num_cols != None and num_rows != None:
        assert num_cols * num_rows == sequence_len, "need num_cols * num_rows == sequence_len"
    if num_cols == None:
        assert num_rows != None, "at least one of num_cols or num_rows need to be set"
        assert sequence_len % num_rows == 0, "sequence length not multiple of num_rows"
        num_cols = int(sequence_len / num_rows)
        
    return [sequence[i:i+num_cols] for i in range(0, sequence_len, num_cols)]

