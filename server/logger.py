import logging


def setup_logger(name="ragbot"):

    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch=logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # formatter

    formatter=logging.Formatter("[%(asctime)s] [%(levelname)s] -  %(message)s ")
    ch.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)

    return logger





logger=setup_logger()