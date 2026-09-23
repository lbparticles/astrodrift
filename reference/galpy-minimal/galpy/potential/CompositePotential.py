###############################################################################
#   CompositePotential.py: class that represents a combination of potentials
###############################################################################
from ..util.conversion import physical_compatible
from .baseCompositePotential import baseCompositePotential
from .Potential import Potential, _check_c, _isNonAxi, flatten


class CompositePotential(baseCompositePotential, Potential):
    """Class that represents a combination of potentials and allows them to be
    called with method functions in the same way as individual potentials."""

    def __init__(self, *args, ro=None, vo=None):
        """
        Initialize a CompositePotential.

        Parameters
        ----------
        *args : Force, Potential, or list of such instances
            Forces/Potentials to combine. Can be individual forces and potentials, lists, or nested lists.
        ro : float or Quantity, optional
            Physical distance scale (in kpc or as Quantity). Default is from the first Force.
        vo : float or Quantity, optional
            Physical velocity scale (in km/s or as Quantity). Default is from the first Force.

        Notes
        -----
        - 2024-11-24 - Written - Bovy (UofT)

        """
        # Flatten the input arguments into a list of potentials
        if len(args) == 1 and isinstance(args[0], list):
            pot_list = args[0]
        else:
            pot_list = list(args)
        # Flatten nested lists
        self._potlist = flatten(pot_list)

        # Check that all potentials are 3D

        # Check that unit systems of all forces are compatible
        if len(self._potlist) > 1:
            for pot in self._potlist[1:]:
                assert physical_compatible(self._potlist[0], pot), (
                    """Physical unit conversion parameters (ro,vo) are not """
                    """compatible between potentials to be combined"""
                )

        # Get ro/vo and _roSet/_voSet from first potential (standard behavior)
        first_pot = self._potlist[0] if len(self._potlist) > 0 else None
        if ro is None and first_pot is not None:
            ro = first_pot._ro
            roSet = first_pot._roSet
        if vo is None and first_pot is not None:
            vo = first_pot._vo
            voSet = first_pot._voSet

        # Initialize with amp=1.0 (amplitude is in individual potentials)
        Potential.__init__(self, amp=1.0, ro=ro, vo=vo)

        # Override _roSet/_voSet based on first potential's settings
        # (unless explicitly provided)
        self._roSet = roSet
        self._voSet = voSet

        # Set properties based on constituent potentials using existing functions
        self.isNonAxi = _isNonAxi(self._potlist)
        # Set dimensionality to 3 (already checked above)
        self.dim = 3
        # Use _check_c to determine C support based on constituent potentials
        self.hasC = _check_c(self._potlist)
        self.hasC_dxdv = _check_c(self._potlist, dxdv=True)
        self.hasC_dens = _check_c(self._potlist, dens=True)
        return None

    def __add__(self, other):
        """
        Add another potential or CompositePotential to this one.

        Parameters
        ----------
        other : Potential or CompositePotential
            Potential(s) to add.

        Returns
        -------
        CompositePotential or planarCompositePotential
            New CompositePotential with combined potentials, or
            planarCompositePotential if adding to a planar potential.

        """
        from .Force import Force
        from .planarForce import planarForce

        # Check type first before checking unit compatibility
        if not isinstance(other, (Force, CompositePotential, planarForce)):
            raise TypeError(
                "Can only add Potential, CompositePotential, or planarForce to CompositePotential"
            )

        # Check unit compatibility
        assert physical_compatible(self, other), (
            """Physical unit conversion parameters (ro,vo) are not """
            """compatible between potentials to be combined"""
        )

        # If adding a planarForce, convert this CompositePotential to planar

        if isinstance(other, CompositePotential):
            return CompositePotential(self._potlist + other._potlist)
        else:  # isinstance(other, Force)
            return CompositePotential(self._potlist + [other])














