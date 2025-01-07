import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                            logging.StreamHandler()
                        ]
                    )

logger = logging.getLogger("mat_logger")
logger.info(f"Completed configuring logger")

def get_logger() -> logging.Logger:
    """
    Returns a logger object
    :return: logger
    """

    return logger
