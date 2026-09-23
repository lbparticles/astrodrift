import warnings

from ..util.config import __config__

_SHOW_WARNINGS = __config__.getboolean("warnings", "verbose")


class galpyWarning(Warning):
    pass


# galpy warnings only shown if verbose = True in the configuration
class galpyWarningVerbose(galpyWarning):
    pass


def _warning(
    message, category=galpyWarning, filename="", lineno=-1, file=None, line=None
):
    if issubclass(category, galpyWarning):
        pass  # off the traced path


warnings.showwarning = _warning






_TINY = 0.000000001






