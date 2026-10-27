import logging

from hisim.env import Envs


def get_logger(name: str = "hisim") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level = Envs.log_level()

        logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger
