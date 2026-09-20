

from ..util.conversion import (
    physical_conversion,
    potential_physical_input,
)
from .planarForce import planarForce
from .Potential import (
    Potential,
    PotentialError,
    _check_potential_list_and_deprecate,
    potential_list_of_potentials_input,
    potential_positional_arg,
)


class planarPotential(planarForce):
    r"""Class representing 2D (R,\phi) potentials"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        planarForce.__init__(self, amp=amp, ro=ro, vo=vo)













class planarAxiPotential(planarPotential):
    """Class representing axisymmetric planar potentials"""

    def __init__(self, amp=1.0, ro=None, vo=None):
        planarPotential.__init__(self, amp=amp, ro=ro, vo=vo)
        self.isNonAxi = False









class planarPotentialFromRZPotential(planarAxiPotential):
    """Class that represents an axisymmetic planar potential derived from a
    RZPotential"""

    def __init__(self, RZPot):
        """
        Initialize.

        Parameters
        ----------
        RZPot : RZPotential instance
            RZPotential instance.

        Returns
        -------
        planarAxiPotential instance

        Notes
        -----
        - 2010-07-13 - Written - Bovy (NYU)

        """
        planarAxiPotential.__init__(self, amp=1.0, ro=RZPot._ro, vo=RZPot._vo)
        # Also transfer roSet and voSet
        self._roSet = RZPot._roSet
        self._voSet = RZPot._voSet
        self._Pot = RZPot
        self.hasC = RZPot.hasC
        self.hasC_dxdv = RZPot.hasC_dxdv
        self.hasC_dens = RZPot.hasC_dens
        return None



    def _Rforce(self, R, phi=0.0, t=0.0):
        """
        Evaluate the radial force.

        Parameters
        ----------
        R : float
            Galactocentric radius.
        phi : float, optional
            Azimuth (default: 0.0).
        t : float, optional
            Time (default: 0.0).

        Returns
        -------
        float
            Radial force at (R, phi, t).

        Notes
        -----
        - Written on 2010-07-13 by Bovy (NYU).
        """
        return self._Pot.Rforce(R, 0.0, t=t, use_physical=False)





def toPlanarPotential(Pot):
    """
    Convert an Potential to a planarPotential in the mid-plane (z=0).

    Parameters
    ----------
    Pot : Potential instance or a combined potential formed using addition (pot1+pot2+…)
        Existing planarPotential instances are just copied to the output.

    Returns
    -------
    planarPotential, planarCompositePotential, or planarDissipativeForce
        instance(s)

    Notes
    -----
    - 2016-06-11: Written - Bovy (UofT)
    - 2024-11-27: Updated to return planarCompositePotential for multiple potentials - Copilot

    """
    from .CompositePotential import CompositePotential
    from .planarCompositePotential import planarCompositePotential

    Pot = _check_potential_list_and_deprecate(Pot)
    if isinstance(Pot, CompositePotential):
        out = []
        for pot in Pot:
            if isinstance(pot, Potential):
                out.append(planarPotentialFromRZPotential(pot))
            else:  # pragma: no cover
                # Can't get here, because there can't be something that's not a proper
                # potential/force in a CompositePotential, but leaving it in case this ever changes
                raise PotentialError(
                    "Input to 'toPlanarPotential' is neither an Potential-instance or a list of such instances"
                )
        # If we get a CompositePotential, always return a planarCompositePotential,
        # even if only one component
        return planarCompositePotential(out)






@potential_positional_arg
@potential_physical_input
@physical_conversion("force", pop=True)
@potential_list_of_potentials_input
def evaluateplanarRforces(Pot, R, phi=None, t=0.0, v=None):
    """
    Evaluate the cylindrical radial force of a planarPotential instance or a combined potential formed using addition (pot1+pot2+…).

    Parameters
    ----------
    Pot : planarPotential instance or a combined potential formed using addition (pot1+pot2+…)
        The potential(s) to evaluate.
    R : float or Quantity
        Cylindrical radius.
    phi : float or Quantity, optional
        Azimuth (default: None).
    t : float or Quantity, optional
        Time (default: 0.0).
    v : numpy.ndarray or Quantity, optional
        Current velocity in cylindrical coordinates (default: None).
        Required when including dissipative forces.

    Returns
    -------
    float or Quantity
        The cylindrical radial force F_R(R, phi, t).

    Notes
    -----
    - 2010-07-13 - Written - Bovy (NYU)
    - 2023-05-29 - Added velocity input for dissipative forces - Bovy (UofT)
    - 2024-11-28 - Updated to use planarCompositePotential internally - Copilot

    """
    if not isinstance(Pot, planarForce):
        raise PotentialError(
            "Input to 'evaluateplanarRforces' is neither a planarForce-instance or a combination of such instances"
        )

    return _evaluateplanarRforces(Pot, R, phi=phi, t=t, v=v)


def _evaluateplanarRforces(Pot, R, phi=None, t=0.0, v=None):
    """Raw, undecorated function for internal use."""
    return Pot._Rforce_nodecorator(R, phi=phi, t=t)










