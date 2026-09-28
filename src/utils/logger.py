# coding: utf-8
# @email: enoche.chow@gmail.com

"""MMRec-style logging shared by backbone and publication pipelines."""

import logging
from pathlib import Path

from utils.utils import get_local_time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = ROOT / "log"


def _level_from_config(config):
    state = config["state"]
    if state is None:
        return logging.INFO
    return getattr(logging, str(state).upper(), logging.INFO)


def init_logger(config, log_name=None, log_dir=None, reset=True, run_id=None):
    """Initialize one console handler and one file handler.

    The default call remains compatible with upstream MMRec/MSCA. Pipeline
    callers may provide a custom log prefix while reusing one root logger for
    the complete run.
    """
    log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = log_name or "{}-{}".format(config["model"], config["dataset"])
    log_tag = str(run_id) if run_id else get_local_time()
    log_path = log_dir / "{}-{}.log".format(prefix, log_tag)

    level = _level_from_config(config)
    root = logging.getLogger()
    root.setLevel(level)

    if reset:
        for handler in list(root.handlers):
            try:
                handler.flush()
                handler.close()
            finally:
                root.removeHandler(handler)

    fileformatter = logging.Formatter(
        "%(asctime)-15s %(levelname)s %(message)s", "%a %d %b %Y %H:%M:%S"
    )
    streamformatter = logging.Formatter(
        "%(asctime)-15s %(levelname)s %(message)s", "%d %b %H:%M"
    )

    fh = logging.FileHandler(log_path, "w", "utf-8")
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(streamformatter)

    root.addHandler(sh)
    root.addHandler(fh)
    root._second_paper_log_path = str(log_path.resolve())
    return str(log_path.resolve())


def current_log_path():
    return getattr(logging.getLogger(), "_second_paper_log_path", None)
