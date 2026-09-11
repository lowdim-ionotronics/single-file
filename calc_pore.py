#!/usr/bin/env python3
"""
Compute ion densities, charge, and capacitance for a multicomponent
single-file pore (analytical generalised Tonks gas model).

Two scan modes:
  mu      : scan bulk chemical potential mu [eV] (same shift for all ions),
            fixed electrode voltage u_V
  voltage : scan electrode voltage u [V], fixed chemical potentials

Usage:
  calc_pore.py -i pore.json
  calc_pore.py -i pore.json --mode voltage --scan-min -1 --scan-max 1

JSON format:
  {
    "T": 300,
    "epsr": 5.0,
    "r_pore_A": 3.75,
    "eshift_A": 0.8,
    "wall_atom_r_A": 1.685,
    "ions": [
      {"q": -1, "R": 2.5, "mu_eV": 0.0},
      {"q":  1, "R": 2.5, "mu_eV": 0.0}
    ],
    "mode": "mu",
    "u_V": 0.0,
    "scan_min": -0.4,
    "scan_max":  0.8,
    "scan_step": 0.01,
    "output": "result.dat"
  }

Three radii describe the pore geometry (see lib/func_psi.py for the full
picture):
  r_pore_A is the nominal pore radius, i.e. the radius to the centre of the
    wall atoms (e.g. carbon nuclei for a CNT).
  r_pore_A - eshift_A is the image-charge screening-surface radius: the
    effective conducting/dielectric surface used in bV0/rr0 (the ion-ion
    electrostatic interaction and length scale), since the screening
    electron density sits slightly inside the wall-atom centres rather
    than exactly at them.
  r_pore_A - wall_atom_r_A is the accessible pore radius, i.e. the radius
    of the cylindrical surface where ion charge actually resides once the
    physical extent of the wall atoms is excluded. This (not the nominal
    r_pore_A) is what the charge/capacitance surface-area normalization
    (q_factor/C_factor in lib/pore.py) must use to convert the dimensionless
    in-pore charge into a physical µC/cm² surface density; using the
    nominal radius there systematically distorts the shape of the
    charging curve even when bV0/rr0 (and hence the steric close-packing
    limit) are correct. wall_atom_r_A defaults to 0 if omitted, in which
    case the accessible and nominal radii coincide.

Ion mu_eV values are the individual bulk chemical potentials.
In "mu" mode a common shift w is added to all ions at each scan step.
"""
import sys
import os
import json
import argparse
import numpy as np
from math import log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.tonks import Ion, f_density, f_charge, f_charge_full
from lib import pore as p

DU_DIFF = 1e-5   # voltage step for numerical dC/du  [V]


def ion_labels(ion_cfgs):
    return [d.get('name', str(i)) for i, d in enumerate(ion_cfgs)]


def ion_header_lines(ion_cfgs):
    lines = []
    for i, d in enumerate(ion_cfgs):
        name = d.get('name', '')
        label = f"{name}  " if name else ''
        lines.append(f"# ion[{i}] {label}q={d['q']:+g}  R={d['R']}A  mu={d['mu_eV']}eV\n")
    return lines


def find_pzc(ions, bv0, rr0, dmax, kBTeV, u_range=3.0):
    """Find PZC [V] by solving Q(u)=0 with the pore model. Returns None if not bracketed."""
    from scipy.optimize import brentq

    def q_at(u_V):
        for ion in ions:
            ion.set_elchempot(u_V / kBTeV)
        dens, _ = f_density(ions, bv0, rr0, dmax)
        return f_charge(dens, ions)

    try:
        pzc = brentq(q_at, -u_range, u_range, xtol=1e-6)
    except ValueError:
        pzc = None
    # restore ions to neutral state (u=0)
    for ion in ions:
        ion.set_elchempot(0.0)
    return pzc


def build_ions(ion_cfgs, kBTeV, rr0):
    ions, base_mus = [], []
    for ic in ion_cfgs:
        mu_kBT = ic['mu_eV'] / kBTeV
        ions.append(Ion(ic['q'], ic['R'], mu_kBT, rr0))
        base_mus.append(mu_kBT)
    return ions, base_mus


def run_mu_scan(cfg, ions, base_mus, bv0, rr0, dmax, qf, Cf, kBTeV, outfile):
    u_V   = cfg.get('u_V', 0.0)
    u_kBT = u_V / kBTeV
    scan  = np.arange(cfg['scan_min'], cfg['scan_max'] + cfg['scan_step'] * 0.5, cfg['scan_step'])
    N     = len(ions)

    labels = ion_labels(cfg['ions'])
    N = len(ions)
    pzc = find_pzc(ions, bv0, rr0, dmax, kBTeV)
    pzc_str = f"{pzc:.4f} V" if pzc is not None else "not found in [-3,3] V"
    with open(outfile, 'w') as f:
        f.write(f"# mode=mu  T={cfg['T']}K  epsr={cfg['epsr']}  r_pore={cfg['r_pore_A']}A  u={u_V}V\n")
        f.write(f"# PZC = {pzc_str}\n")
        for line in ion_header_lines(cfg['ions']): f.write(line)
        f.write(f"# (1)mu[eV]  (2)rho_tot  (3)Q[uC/cm2]  (4)C[uF/cm2]\n")
        f.write("# " + "  ".join(f"({5+i})rho[{lbl}]"        for i, lbl in enumerate(labels)) + "\n")
        f.write("# " + "  ".join(f"({5+N+i})mu_el[{lbl}][eV]" for i, lbl in enumerate(labels)) + "\n")

        sv_prev = None
        for mu_eV in scan:
            delta_kBT = mu_eV / kBTeV
            for ion, base_mu in zip(ions, base_mus):
                ion.set_chempot(base_mu + delta_kBT)
                ion.set_elchempot(u_kBT)
            mu_els   = [(ion.elchempot - log(ion.rr0_A)) * kBTeV for ion in ions]
            dens, sv = f_density(ions, bv0, rr0, dmax, sv_hint=sv_prev)
            Q0_au    = f_charge(dens, ions)
            Q1_au    = f_charge_full(ions, bv0, rr0, (u_V + DU_DIFF) / kBTeV, dmax,
                                     sv_hint=sv if sv > 0 else sv_prev)
            # restore elchempot after f_charge_full side effect
            for ion in ions: ion.set_elchempot(u_kBT)
            sv_prev  = sv if sv > 0 else sv_prev
            rho_tot  = sum(dens)
            Q        = qf * Q0_au
            C        = Cf * (Q1_au - Q0_au) / (DU_DIFF / kBTeV)
            row      = (f"{mu_eV}\t{rho_tot}\t{Q}\t{C}\t"
                        + "\t".join(str(d) for d in dens) + "\t"
                        + "\t".join(str(m) for m in mu_els))
            f.write(row + "\n")
            print(f"mu={mu_eV:+.3f} eV  sv={sv:.3e}  Q={Q:+.4f} µC/cm²  C={C:.4f} µF/cm²")


def run_voltage_scan(cfg, ions, bv0, rr0, dmax, qf, Cf, kBTeV, outfile):
    scan = np.arange(cfg['scan_min'], cfg['scan_max'] + cfg['scan_step'] * 0.5, cfg['scan_step'])
    N    = len(ions)

    labels = ion_labels(cfg['ions'])
    N = len(ions)
    pzc = find_pzc(ions, bv0, rr0, dmax, kBTeV)
    pzc_str = f"{pzc:.4f} V" if pzc is not None else "not found in [-3,3] V"
    with open(outfile, 'w') as f:
        f.write(f"# mode=voltage  T={cfg['T']}K  epsr={cfg['epsr']}  r_pore={cfg['r_pore_A']}A\n")
        f.write(f"# PZC = {pzc_str}\n")
        for line in ion_header_lines(cfg['ions']): f.write(line)
        f.write(f"# (1)u[V]  (2)Q[uC/cm2]  (3)C[uF/cm2]  (4)rho_tot\n")
        f.write("# " + "  ".join(f"({5+i})rho[{lbl}]"        for i, lbl in enumerate(labels)) + "\n")
        f.write("# " + "  ".join(f"({5+N+i})mu_el[{lbl}][eV]" for i, lbl in enumerate(labels)) + "\n")

        sv_prev = None
        for u_V in scan:
            u_kBT    = u_V / kBTeV
            for ion in ions: ion.set_elchempot(u_kBT)
            mu_els   = [(ion.elchempot - log(ion.rr0_A)) * kBTeV for ion in ions]
            dens, sv = f_density(ions, bv0, rr0, dmax, sv_hint=sv_prev)
            Q0_au    = f_charge(dens, ions)
            Q1_au    = f_charge_full(ions, bv0, rr0, (u_V + DU_DIFF) / kBTeV, dmax,
                                     sv_hint=sv if sv > 0 else sv_prev)
            sv_prev  = sv if sv > 0 else sv_prev
            rho_tot  = sum(dens)
            Q        = qf * Q0_au
            C        = Cf * (Q1_au - Q0_au) / (DU_DIFF / kBTeV)
            row      = (f"{u_V}\t{Q}\t{C}\t{rho_tot}\t"
                        + "\t".join(str(d) for d in dens) + "\t"
                        + "\t".join(str(m) for m in mu_els))
            f.write(row + "\n")
            print(f"u={u_V:+.3f} V  sv={sv:.3e}  Q={Q:+.4f} µC/cm²  C={C:.4f} µF/cm²")


def main():
    parser = argparse.ArgumentParser(
        description='Single-file pore: ion densities, charge, capacitance.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('-i', '--input', required=True, metavar='FILE',
                        help='JSON file with system description')
    parser.add_argument('--mode', choices=['mu', 'voltage'],
                        help='Scan mode override')
    parser.add_argument('--scan-min',  type=float, metavar='VAL')
    parser.add_argument('--scan-max',  type=float, metavar='VAL')
    parser.add_argument('--scan-step', type=float, metavar='VAL')
    parser.add_argument('-o', '--out', metavar='FILE',
                        help='Output data file')
    parser.add_argument('--epsr-pore', type=float, default=None, metavar='EPS',
                        help='Override dielectric constant in pore (overrides JSON epsr)')
    parser.add_argument('-s', '--save', metavar='FILE', default='pore.json',
                        help='Save effective config to JSON (default: pore.json)')

    args = parser.parse_args()

    with open(args.input) as f:
        cfg = json.load(f)

    if args.mode:       cfg['mode']      = args.mode
    if args.scan_min  is not None: cfg['scan_min']  = args.scan_min
    if args.scan_max  is not None: cfg['scan_max']  = args.scan_max
    if args.scan_step is not None: cfg['scan_step'] = args.scan_step
    if args.epsr_pore is not None: cfg['epsr']      = args.epsr_pore

    mode          = cfg.get('mode', 'mu')
    T             = cfg['T']
    epsr          = cfg['epsr']
    r_pore        = cfg['r_pore_A']
    eshift        = cfg['eshift_A']
    wall_atom_r   = cfg.get('wall_atom_r_A', 0.0)
    r_accessible  = r_pore - wall_atom_r

    cfg.setdefault('scan_min',  -0.4 if mode == 'mu' else -1.0)
    cfg.setdefault('scan_max',   0.8 if mode == 'mu' else  1.0)
    cfg.setdefault('scan_step',  0.01)

    default_out = os.path.splitext(args.input)[0] + '.dat'
    outfile = args.out or cfg.get('output', default_out)

    kBTeV  = p.kBT_eV(T)
    rr0    = p.rr0_A(r_pore, eshift)
    bv0    = p.bV0(T, epsr, r_pore, eshift)
    r_max  = max(d['R'] for d in cfg['ions'])
    dmax   = p.d_max_ren([d['R'] for d in cfg['ions']], rr0)
    qf     = p.q_factor(r_accessible, r_max)
    Cf     = p.C_factor(T, r_accessible, r_max)

    print(f"# T={T}K  epsr={epsr}  r_pore={r_pore}Å  eshift={eshift}Å  wall_atom_r={wall_atom_r}Å  r_accessible={r_accessible}Å  lB={p.lB_A(T,epsr):.3f}Å  bV0={bv0:.4f}  rr0={rr0:.4f}Å")
    print(f"# mode={mode}  scan=[{cfg['scan_min']}, {cfg['scan_max']}, {cfg['scan_step']}]")
    print(f"# kBT={kBTeV:.5f}eV  dmax={dmax:.4f}  q_factor={qf:.4e}  C_factor={Cf:.4e}")

    ions, base_mus = build_ions(cfg['ions'], kBTeV, rr0)

    if mode == 'mu':
        run_mu_scan(cfg, ions, base_mus, bv0, rr0, dmax, qf, Cf, kBTeV, outfile)
    else:
        run_voltage_scan(cfg, ions, bv0, rr0, dmax, qf, Cf, kBTeV, outfile)

    with open(args.save, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"# config saved to {args.save},  results saved to {outfile}")


if __name__ == '__main__':
    main()
