import logging
import os
from  datetime import datetime


def setup_logging():
    """
    Setting up logger
    """

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log")
    log_path = os.path.join(log_dir, log_file)


    logging.basicConfig(
        #sets logging level to INFO
        level=logging.INFO,
        format='[%(asctime)s: %(levelname)s: %(name)s]: %(message)s',# log format
        handlers=[
            logging.FileHandler(log_path), # log file handler
            logging.StreamHandler() # log stream handler outputs to console
        ]
    )