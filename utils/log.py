import logging


def Log2Txt(logpath):
    logger = logging.getLogger(__name__)

    logger.setLevel(level = logging.INFO)

    handler = logging.FileHandler(logpath)

    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    # self.logger = logger
    return  logger

    logger.info("Start print log")

    logger.debug("Do something")

    logger.warning("Something maybe fail.")

    logger.info("Finish")