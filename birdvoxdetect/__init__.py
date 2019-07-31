import absl.logging
import logging

formatter = logging.Formatter("%(message)s")
absl.logging.get_absl_handler().setFormatter(formatter)
absl.logging._warn_preinit_stderr = 0

from .version import version as __version__
from .core import process_file
