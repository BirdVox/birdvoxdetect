import logging
import loggin.config

from .cli import main

# configurate logger according to external config file
logging.config.fileConfig('logging.conf')

# call the CLI handler when the module is executed as `python -m birdvoxdetect`
main()
