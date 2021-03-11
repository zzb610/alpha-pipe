import os
import datetime as dt
from loguru import logger

def display_info(func):
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        ppid = os.getppid()
        begin_time = dt.datetime.now()
        func(*args, **kwargs)
        end_time = dt.datetime.now()
        pass_time = end_time - begin_time
        logger.info("ppid:{}, pid {}, 函数:{}, 运行时间{}".format(ppid, pid, func.__name__, pass_time))
    return wrapper
