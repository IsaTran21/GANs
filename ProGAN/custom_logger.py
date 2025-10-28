import logging.config
import logging
import time
import json
# Custom Formatter
class DictFormatter(logging.Formatter):
    # Override the converter to use GMT+7
    # make converter a staticmethod so self is not inserted
    converter = staticmethod(lambda ts: time.gmtime(ts + 7*3600))
    def format(self, record):
        #  Create the dict from the record
        log_dict = {
            "level": record.levelname,
            "name": record.name, # The name of the logger
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        # Only add keys if it exists
        if hasattr(record, "save_path"):
             log_dict["save_path"] = record.save_path



        return json.dumps(log_dict)
# Log config
LOGGER_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'save_formatter': {
            "()": DictFormatter # Create an instance of the custom DictFormatter
        },
    },
    'handlers': {
        "save_handler":
            {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "save_log.json",
                "formatter": "save_formatter",
                "mode": "a",  # explicitly append
                "maxBytes": 10485760 * 5,  # 50 MB = 10 * 1024 * 1024 * 5
                "backupCount": 50  # number of rotated log files to keep
            },
        "data_handler":
            {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "data_log.json",
                "formatter": "save_formatter",
                "mode": "a",  # explicitly append
                "maxBytes": 10485760 * 5,  # 50 MB = 10 * 1024 * 1024 * 5
                "backupCount": 50  # number of rotated log files to keep
            },

    },
    'loggers': {

        'save_logger': {
            "level": "INFO",
            "handlers": ["save_handler"],
            "propagate": False
        },
        'data_logger': {
            "level": "INFO",
            "handlers": ["data_handler"],
            "propagate": False
        },
    },
}

