import os
import os.path as osp
import sys
import time

__all__ = ["Logger", "setup_logger"]


class Logger:
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(osp.dirname(fpath), exist_ok=True)
            self.file = open(fpath, "w", encoding="utf-8")

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.file is not None and not self.file.closed:
            self.file.close()


def setup_logger(output=None):
    if output is None:
        return
    fpath = output if output.endswith((".txt", ".log")) else osp.join(output, "log.txt")
    if osp.exists(fpath):
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")
    sys.stdout = Logger(fpath)
