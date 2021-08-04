import logging
logger = logging.getLogger()


def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter(
        "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)

setup_logger(logger)

if __name__ == '__main__':
    logger.info('lr=%f', 0.00000000000000001)