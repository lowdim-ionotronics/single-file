"""
Free energy shift for an ion entering a single-file cylindrical pore.

calc_psi() returns the shift (in eV) accounting for:
  - image charge attraction at the dielectric cylinder boundary (via mplib)
  - Lennard-Jones attraction to the tube wall (eps_lj > 0)

This is the 3D -> 1D dimensional-reduction factor: the Boltzmann-weighted
integral of the transverse potential over the pore cross-section.

Three radii describe the pore geometry (all measured from the tube axis):
  pore_R      : nominal pore radius, i.e. the radius to the centre of the
                wall atoms (e.g. carbon nuclei for a CNT).
  pore_R - wall_atom_r - ion_r (= rmax below)
              : accessible radius — the maximum radial position of an ion's
                centre, once both the wall atom's and the ion's own size are
                excluded.
  pore_R - eshift
              : image-charge screening-surface radius — the effective
                conducting/dielectric surface that produces image charges.
                It sits slightly inside the wall-atom centres because the
                screening electron density does not coincide with the
                nuclear positions (eshift ~ 0.8 A for a CNT).

References:
  Kondrat & Kornyshev, J. Phys.: Condens. Matter 23 (2011) 022201
    https://doi.org/10.1088/0953-8984/23/2/022201
  Verkholyak, Kuzmak & Kondrat, J. Chem. Phys. 155 (2021) 174702 — psi as Eq. (9)
    https://doi.org/10.1063/5.0066786
"""
import math
import numpy as np
from scipy.integrate import quad
from scipy import integrate as _sci_int
import mplib_ctypes as mplib

try:
    from scipy.integrate import simpson as _simpson
except ImportError:
    from scipy.integrate import simps as _simpson   # type: ignore

_kB_eV = 8.617333262e-5   # Boltzmann constant [eV/K]

# Unit conversion for image-charge potential:
# mplib returns 1/Å; multiply by k_e*e^2 in kcal·Å/mol and then convert kcal/mol -> K.
_KE_KCAL_A = 332.0636   # k_e * e^2  [kcal·Å/mol / e^2]
_KCAL_TO_K  = 503.2166  # 1 kcal/mol in K  (= 4184 J/mol / k_B / N_A)


def _integrand_att(phi, r, d, R):
    """LJ-6 attractive integrand for the cylindrical wall at angle phi."""
    d6 = d**6
    a  = R*R + r*r - 2.0 * R * r * math.cos(phi)
    return -3.0 * d6 / (8.0 * a**2.5)


def _uLJ_att(r, eps_lj, d, R):
    """LJ attractive ion-wall potential at radial position r [eV].

    d = ion_r + wall_atom_r  (contact distance)
    R = pore radius
    """
    if eps_lj == 0.0:
        return 0.0
    result = quad(_integrand_att, 0.0, 2.0 * math.pi, args=(r, d, R))
    return 4.0 * math.pi * R * eps_lj * result[0]


def calc_psi(pore_R, ion_r, eshift, wall_atom_r, eps_r, eps_lj, T, q=1):
    """Free energy shift [eV] for placing an ion in a cylindrical pore.

    Parameters
    ----------
    pore_R      : float  nominal pore radius [Å], i.e. radius to the centre
                  of the wall atoms (e.g. carbon nuclei for a CNT)
    ion_r       : float  ion hard-sphere radius [Å]
    eshift      : float  image-charge screening-surface inward shift [Å],
                  relative to pore_R (0.8 for CNT); accounts for the
                  screening electron density sitting inside the wall-atom
                  centres, not at the nuclei themselves
    wall_atom_r : float  wall atom radius [Å] (1.685 for CNT carbon)
    eps_r       : float  dielectric constant inside pore
    eps_lj      : float  LJ epsilon for ion-wall interaction [eV] (0 = disabled)
    T           : float  temperature [K]
    q           : float  ion charge [e] (default 1)

    Returns
    -------
    psi : float  free energy shift [eV]  (negative = favourable)
    """
    R    = pore_R - eshift               # image-charge surface radius [Å]
    rmax = pore_R - ion_r - wall_atom_r  # max radial position of ion centre [Å]

    if rmax <= 0:
        raise ValueError(
            f"Ion (R={ion_r} Å) does not fit in pore "
            f"(R={pore_R} Å, wall_atom_r={wall_atom_r} Å)")

    r_arr, U1_raw = mplib.cyl_u1_calc_array(R, rmax, 100)
    r_arr  = np.asarray(r_arr)
    conv   = q * q * _KE_KCAL_A * _KCAL_TO_K / eps_r
    U1_K   = conv * np.asarray(U1_raw)   # image-charge potential [K]

    d_wall = ion_r + wall_atom_r
    kBT    = _kB_eV * T
    Uatt_K = np.array([T * _uLJ_att(ri, eps_lj, d_wall, pore_R) / kBT
                       for ri in r_arr])  # LJ potential [K]

    U_tot = U1_K + Uatt_K
    U0    = U_tot[0]
    f     = r_arr * np.exp(-(U_tot - U0) / T)

    F = 2.0 * math.pi * float(_simpson(f, x=r_arr))
    if F <= 0:
        raise ValueError("Partition function F <= 0; check pore/ion parameters")

    return T * _kB_eV * (-U0 / T + math.log(F))
