import logging

logging._warn_preinit_stderr = 0
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
)
logger = logging.getLogger(__name__)

from .version import version as __version__
from .core import process_file
