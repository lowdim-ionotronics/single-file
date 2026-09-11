"""
Physical constants and pore-geometry setup for the single-file pore model.

Unit conventions
----------------
Length   : Angstroms [Å]
Energy   : eV  (or kBT where noted)
Charge   : elementary charge e
Temp     : Kelvin

Key output units
----------------
Charge density  : µC/cm²
Capacitance     : µF/cm²
"""

import math

# SI constants (CODATA 2018 exact values)
k_B      = 1.380649e-23     # J/K
e_charge = 1.602176634e-19  # C
k_e      = 8.9875517923e9   # N m²/C²  (Coulomb constant)
J_to_eV  = 1.0 / e_charge   # eV/J  (exact: 1 eV = e_charge J)

# Geometric factors from the cylindrical pore analytical solution
_BV0_PREFACTOR = 3.08   # bV0 = 3.08 * lB / r_pore
_RR0_DIVISOR   = 2.4    # rr0 = r_pore / 2.4


def kBT_eV(T):
    """Thermal energy kT in eV."""
    return k_B * T * J_to_eV


def lB_A(T, epsr):
    """Bjerrum length in Å."""
    return k_e * e_charge**2 / (epsr * k_B * T) * 1e10


def bV0(T, epsr, r_pore_A, eshift):
    """Interaction parameter beta*V(r=0) for the cylindrical pore.

    Uses the image-charge screening-surface radius (r_pore_A - eshift),
    not the bare wall-atom-position radius.
    """
    return _BV0_PREFACTOR * lB_A(T, epsr) / (r_pore_A - eshift)


def rr0_A(r_pore_A, eshift):
    """Reference length scale rr0 = (r_pore - eshift) / 2.4  [Å]."""
    return (r_pore_A - eshift) / _RR0_DIVISOR


def d_max_ren(ions_radii_A, rr0):
    """Dimensionless max hard-sphere diameter: 2*R_max / rr0."""
    return 2.0 * max(ions_radii_A) / rr0


def q_factor(r_accessible_A, r_ion_max_A):
    """Prefactor: dimensionless 1D charge density -> µC/cm².
    Q [µC/cm²] = q_factor * charge_au

    r_accessible_A must be the *accessible* pore radius (r_pore_A minus the
    wall-atom radius, i.e. the radius of the cylindrical surface where ion
    charge actually resides), not the nominal radius to the wall-atom
    centres. Using the nominal radius here (rather than in bV0/rr0, which
    correctly use the screening-surface radius r_pore_A - eshift) distorts
    the shape of the charging curve even when the close-packing limit is
    unaffected.
    """
    r_pore_m = r_accessible_A * 1e-10
    r_ion_m  = r_ion_max_A * 1e-10
    return e_charge / (2.0 * math.pi * r_pore_m * 2.0 * r_ion_m) * 100.0


def C_factor(T, r_accessible_A, r_ion_max_A):
    """Prefactor: d(charge_au)/d(u_kBT) -> µF/cm².
    C [µF/cm²] = C_factor * delta_charge_au / delta_u_kBT

    r_accessible_A must be the *accessible* pore radius — see q_factor().
    """
    r_pore_m = r_accessible_A * 1e-10
    r_ion_m  = r_ion_max_A * 1e-10
    return e_charge**2 / (2.0 * math.pi * r_pore_m * 2.0 * r_ion_m * k_B * T) * 100.0
