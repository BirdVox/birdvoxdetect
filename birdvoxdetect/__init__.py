import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelno)s:%(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

from .version import version as __version__
from .core import process_file
