import configparser
import copy
import os
import os.path

# The default configuration
default_configuration = {
    "normalization": {"ro": "8.", "vo": "220."},
    "astropy": {"astropy-units": "False", "astropy-coords": "True"},
    "plot": {"seaborn-bovy-defaults": "False"},
    "warnings": {"verbose": "False"},
    "version-check": {
        "do-check": "True",
        "check-non-interactive": "True",
        "check-non-interactive-every": "1",
        "last-non-interactive-check": "2000-01-01",
    },
}
default_filename = os.path.join(os.path.expanduser("~"), ".galpyrc")








# Read the configuration file
__config__ = configparser.ConfigParser()
cfilename = __config__.read(".galpyrc")
if not cfilename:
    cfilename = __config__.read(default_filename)
# Store a version of the config in case we need to re-write parts of it,
# but don't want to apply changes that we don't want to re-write
configfilename = cfilename[-1]
__orig__config__ = copy.deepcopy(__config__)


# Set configuration variables on the fly


