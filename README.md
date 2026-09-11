# single-file

Analytical (generalized Tonks-gas) solver for ions confined in a **single-file cylindrical
nanopore** — an infinitely long pore narrow enough that ions cannot pass each other. Given a
pore geometry and per-ion bulk chemical potentials, it computes in-pore ion densities,
accumulated charge, and differential capacitance as a function of electrode potential (or bulk
chemical potential).

The physics follows Verkholyak, Kuzmak & Kondrat, *J. Chem. Phys.* **155**, 174702 (2021),
[doi:10.1063/5.0066786](https://doi.org/10.1063/5.0066786): the lateral degrees of freedom of an
electrolyte confined to a single-file pore are integrated out, yielding an effective
one-dimensional continuum description with hard-sphere exclusion and exponentially screened
electrostatic interactions along the pore axis. The resulting partition function is solved
exactly via a transfer-matrix formalism, giving closed-form ion densities.

The original code was written by Taras Verkholyak. See [AUTHORS](AUTHORS) for who developed the
model, and [LICENSE](LICENSE) (GPLv3).

## Contents

- `calc_pore.py` — the solver. Reads a pore-configuration JSON, scans either the electrode
  voltage or the bulk chemical potential, and writes out ion densities, charge, and differential
  capacitance. Pure `numpy`/`scipy`, no other dependencies.
- `lib/tonks.py` — the generalized Tonks-gas solver (`Ion`, `f_density`, `f_charge`).
- `lib/pore.py` — physical constants and pore-geometry helper functions.
- `make_pore_cfg.py` + `lib/func_psi.py` — optional helper that builds a pore-configuration
  JSON from *bulk* per-ion chemical potentials, applying the 3D→1D confinement/image-charge free
  energy shift. Requires [mplib](https://github.com/lowdim-ionotronics/mplib) (`mplib_ctypes`).
- `examples/example_emimBF4.json` — a minimal two-ion (EMIM+/BF4-) example.
- `examples/jcp2026/` — inputs reproducing the single-file curves of `fig:1D_vs_2D` from our
  2026 *J. Chem. Phys.* paper on nanopore charging across dimensionalities (citation to be added
  once published).

## Usage

```
python3 calc_pore.py -i examples/example_emimBF4.json
```

### Pore-configuration JSON format

```json
{
  "T": 300,
  "epsr": 5.0,
  "r_pore_A": 3.75,
  "eshift_A": 0.8,
  "ions": [
    {"q": -1, "R": 2.5, "mu_eV": 0.0, "name": "An"},
    {"q":  1, "R": 2.5, "mu_eV": 0.0, "name": "Cat"}
  ],
  "mode": "voltage",
  "scan_min": -1.0,
  "scan_max": 1.0,
  "scan_step": 0.01
}
```

- `r_pore_A` — nominal pore radius, i.e. the radius to the centre of the wall atoms (e.g. carbon
  nuclei for a CNT).
- `eshift_A` — inward shift, from `r_pore_A`, of the image-charge screening surface: the
  effective conducting/dielectric surface used for the ion-ion electrostatic interaction is
  `r_pore_A - eshift_A`, since the screening electron density sits slightly inside the wall-atom
  centres rather than exactly at them.
- `wall_atom_r_A` (optional, default 0) — physical wall-atom radius. `r_pore_A - wall_atom_r_A`
  is the **accessible** pore radius: the radius of the cylindrical surface where ion charge
  actually resides. This — not the nominal `r_pore_A` — is what the charge/capacitance surface
  normalization uses internally; conflating the two silently distorts the shape of the charging
  curve even though the steric close-packing limit (and hence the saturation charge) looks fine.
  You can give `r_accessible_A` directly instead of (or alongside) `r_pore_A` — matching how pore
  width is usually reported in the literature — and `calc_pore.py` derives whichever is missing.
  If you give both, they must agree with `wall_atom_r_A` or `calc_pore.py` raises an error rather
  than silently picking one.
- `ions[].mu_eV` — the bulk (or already pore-corrected) chemical potential of each ion.
- `mode` — `"voltage"` scans the electrode potential `u` [V] at fixed chemical potentials;
  `"mu"` scans a common shift to all ions' chemical potentials [eV] at a fixed voltage `u_V`.

Run `python3 calc_pore.py -h` for the full CLI, and see the module docstrings in `lib/` for the
underlying physics and unit conventions.

### Building a pore config from bulk chemical potentials

If you have per-ion bulk chemical potentials (e.g. from an SPT+MSA bulk electrolyte
calculation) rather than a ready-made pore JSON, `make_pore_cfg.py` applies the 3D→1D
confinement free energy shift to get pore-corrected `mu_eV` values:

```
python3 make_pore_cfg.py -i bulk_mus.json --r-pore 3.75 -o pore.json
python3 calc_pore.py -i pore.json
```

Give `--r-pore-accessible` instead of `--r-pore` if you have the accessible radius rather than
the nominal one (matching how pore width is usually reported in the literature). If you give
both, they must agree with `--wall-atom-r` (default 1.685 Å, the CNT carbon radius) or the tool
errors out.

This requires [mplib](https://github.com/lowdim-ionotronics/mplib) to be installed
(`mplib_ctypes`).

## Requirements

```
pip install -r requirements.txt
```

`calc_pore.py` itself only needs `numpy` and `scipy`. `make_pore_cfg.py` additionally needs
`mplib`.
