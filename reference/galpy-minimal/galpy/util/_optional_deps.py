# Central place to process optional dependencies.
#
# This cut-down galpy requires astropy, and nothing on the ISO/GMC pathway
# reaches the numba, jax or numexpr code that upstream probes for here, so the
# only flags left are the astropy ones the surrounding code still reads.
from ..util.config import __config__

_APY_UNITS = __config__.getboolean("astropy", "astropy-units")
_APY_LOADED = True

