import logging

# OUTPUT_LOG = "debug.log"

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                            # logging.FileHandler(OUTPUT_LOG),
                            logging.StreamHandler()
                        ]
                    )

logger = logging.getLogger("mat_logger")
logger.info(f"Completed configuring logger")