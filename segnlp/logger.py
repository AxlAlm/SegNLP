import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def get_log_file(suffix:str):
    # if 'LOG_FILE' in os.environ and os.environ['LOG_FILE']:
    #     log_file = os.environ['LOG_FILE']
    # else:
    if not os.path.exists('/tmp/segnlp/'):
        os.makedirs('/tmp/segnlp/')
    log_file = f'/tmp/segnlp/segnlp{suffix}.log'

    return log_file

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler(log_file:str):
    file_handler = TimedRotatingFileHandler(log_file, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler

def get_logger(logger_name, suffix:str="", logging_level=logging.INFO):
    log_file = get_log_file(suffix=suffix)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(log_file=log_file))
    logger.propagate = False
    return logger
