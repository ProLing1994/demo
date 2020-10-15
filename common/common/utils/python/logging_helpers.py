import logging
import os


def setup_logger(log_file, name):
    """
    setup logger for logging training messages
    :param log_file: the location of log file
    :param name: the name of logger
    :return: a logger object
    """
    dirname = os.path.dirname(log_file)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # stream handler
    h1 = logging.StreamHandler()
    h1.setLevel(logging.DEBUG)
    h1.setFormatter(formatter)
    logger.addHandler(h1)

    # file handler
    h2 = logging.FileHandler(log_file)
    h2.setLevel(logging.DEBUG)
    h2.setFormatter(formatter)
    logger.addHandler(h2)

    return logger