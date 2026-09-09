#!/usr/bin/env python3
"""
Build a pore configuration JSON for calc_pore.py.

Reads a bulk ions JSON produced by  chempot/calc_mu.py -o  and applies the
confinement free energy (3D -> 1D reduction, image charge + LJ wall) to each
ion's chemical potential:

       psi_i = calc_psi(pore_R, ion_r, eshift, wall_atom_r, epsr_pore, eps_lj, T, q)

Total pore chemical potential:
       mu_pore_i = mu_bulk_i + psi_i

Pore geometry (see lib/func_psi.py for the full picture): r_pore_A is the
nominal pore radius, i.e. the radius to the centre of the wall atoms
(e.g. carbon nuclei for a CNT). wall_atom_r is the physical radius of those
wall atoms, subtracted (together with the ion radius) to get the radius
accessible to an ion's centre. eshift is the additional inward shift, from
r_pore_A, of the image-charge screening surface — the effective
conducting/dielectric surface is not exactly at the wall-atom centres
because the screening electron density sits slightly inside them.

Input JSON format (from chempot/calc_mu.py -o):
  {
    "T": 298, "epsr": 12.0,
    "ions": [
      {"name": "EMIM+", "q": 1, "R": 2.79, "mu_eV": -0.523},
      ...
    ]
  }

Usage:
  make_pore_cfg.py -i mix_mus.json --r-pore 3.5 -o pore.json
  make_pore_cfg.py -i mix_mus.json --r-pore 3.5 --epsr-pore 2.0 --eps-lj 0.3 -o pore.json
"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.func_psi import calc_psi

_DEFAULT_WALL_ATOM_R = 3.37 / 2.0 # Å  (graphene / CNT carbon atom radius)
_DEFAULT_ESHIFT      = 0.8        # Å  (image-charge surface inward shift)
_DEFAULT_EPS_LJ      = 0.0        # eV (no LJ wall interaction)
_MU_EXCLUDED_eV      = -50.0      # mu_pore when ion does not fit (exp(-50/kBT) ~ 0)


def build_pore_cfg(bulk_cfg, r_pore_A, epsr_pore=None,
                   wall_atom_r=_DEFAULT_WALL_ATOM_R, eshift=_DEFAULT_ESHIFT,
                   eps_lj=_DEFAULT_EPS_LJ, v_min=-1.0, v_max=1.0, v_step=0.01):
    T = bulk_cfg['T']
    if epsr_pore is None:
        epsr_pore = bulk_cfg.get('epsr', 1.0)

    pore_ions = []
    excluded  = []
    for ion in bulk_cfg['ions']:
        q, R       = ion['q'], ion['R']
        mu_bulk_eV = ion['mu_eV']

        rmax = r_pore_A - R - wall_atom_r
        if rmax <= 0:
            mu_pore_eV = _MU_EXCLUDED_eV
            psi_eV     = None
            excluded.append(ion)
        else:
            psi_eV     = calc_psi(pore_R=r_pore_A, ion_r=R, eshift=eshift,
                                  wall_atom_r=wall_atom_r, eps_r=epsr_pore, eps_lj=eps_lj,
                                  T=T, q=q)
            mu_pore_eV = mu_bulk_eV + psi_eV

        d = {'q': q, 'R': R,
             'mu_eV': mu_pore_eV, 'mu_bulk_eV': mu_bulk_eV, 'psi_eV': psi_eV}
        if 'name' in ion:
            d['name'] = ion['name']
        pore_ions.append(d)

    if len(excluded) == len(bulk_cfg['ions']):
        for ion in excluded:
            name  = ion.get('name', f"q={ion['q']:+g}")
            r_min = ion['R'] + wall_atom_r
            print(f"ERROR: ion '{name}' (R={ion['R']} A) does not fit; "
                  f"min r_pore = {r_min:.4f} A", file=sys.stderr)
        sys.exit(1)

    return {
        '_units': {
            'T': 'K',
            'r_pore_A': 'Angstrom (nominal pore radius: radius to the centre '
                        'of the wall atoms, e.g. carbon nuclei for a CNT)',
            'wall_atom_r_A': 'Angstrom (physical radius of a wall atom; '
                             'r_pore_A - wall_atom_r_A - ion.R is the radius '
                             "accessible to an ion's centre)",
            'eshift_A': 'Angstrom (inward shift of the image-charge screening '
                        'surface relative to r_pore_A; the screening electron '
                        'density sits slightly inside the wall-atom centres, '
                        'so the effective conducting surface radius used for '
                        'image-charge/interionic electrostatics is '
                        'r_pore_A - eshift_A, not r_pore_A itself)',
            'eps_lj_eV': 'eV (LJ epsilon, 0 = off)',
            'ions.R': 'Angstrom', 'ions.q': 'e',
            'ions.mu_eV':      'eV  (= mu_bulk + psi)',
            'ions.mu_bulk_eV': 'eV  (bulk chemical potential from input JSON)',
            'ions.psi_eV':     'eV  (3D->1D confinement + image charge free energy)',
            'scan_min/max/step': 'V (voltage mode) or eV (mu mode)',
        },
        'T': T, 'epsr': epsr_pore,
        'r_pore_A': r_pore_A,
        'wall_atom_r_A': wall_atom_r, 'eshift_A': eshift, 'eps_lj_eV': eps_lj,
        'ions': pore_ions,
        'mode': 'voltage',
        'scan_min': v_min, 'scan_max': v_max, 'scan_step': v_step,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build pore configuration JSON for calc_pore.py.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('-i', '--input', required=True, metavar='FILE',
                        help='Bulk ions JSON with mu_eV (from chempot/calc_mu.py -o)')
    parser.add_argument('--r-pore', type=float, required=True, metavar='A',
                        help='Pore radius [Å]')
    parser.add_argument('--epsr-pore', type=float, default=None, metavar='EPS',
                        help='Dielectric constant inside pore '
                             '(default: same as epsr in input JSON)')
    parser.add_argument('--wall-atom-r', type=float, default=_DEFAULT_WALL_ATOM_R, metavar='A',
                        help=f'Wall atom radius [Å] (default: {_DEFAULT_WALL_ATOM_R:.4f}, CNT carbon)')
    parser.add_argument('--eshift', type=float, default=_DEFAULT_ESHIFT, metavar='A',
                        help='Inward shift of the image-charge screening surface '
                             f'relative to r_pore [Å] (default: {_DEFAULT_ESHIFT})')
    parser.add_argument('--eps-lj', type=float, default=_DEFAULT_EPS_LJ, metavar='EV',
                        help=f'LJ epsilon for ion-wall interaction [eV] '
                             f'(default: {_DEFAULT_EPS_LJ} = off)')
    parser.add_argument('--voltage', metavar='MIN,MAX,STEP', default=None,
                        help='Voltage scan range [V]: min,max,step (default: -1,1,0.01)')
    parser.add_argument('-o', '--out', metavar='FILE', default=None,
                        help='Output pore JSON (default: <input_stem>_r<rpore>.json)')

    args = parser.parse_args()

    with open(args.input) as f:
        bulk_cfg = json.load(f)

    if 'ions' not in bulk_cfg:
        parser.error("Input JSON must have an 'ions' key with mu_eV per ion. "
                     "Run  chempot/calc_mu.py -i <mix.json> -o <mus.json>  first.")

    if args.voltage:
        parts = args.voltage.split(',')
        if len(parts) != 3:
            parser.error('--voltage must be MIN,MAX,STEP')
        v_min, v_max, v_step = float(parts[0]), float(parts[1]), float(parts[2])
    else:
        v_min, v_max, v_step = -1.0, 1.0, 0.01

    outfile = args.out
    if outfile is None:
        stem    = os.path.splitext(os.path.basename(args.input))[0]
        outfile = f"{stem}_r{args.r_pore}.json"

    pore_cfg = build_pore_cfg(
        bulk_cfg, args.r_pore,
        epsr_pore=args.epsr_pore,
        wall_atom_r=args.wall_atom_r,
        eshift=args.eshift,
        eps_lj=args.eps_lj,
        v_min=v_min, v_max=v_max, v_step=v_step,
    )

    epsr_pore = args.epsr_pore if args.epsr_pore is not None else bulk_cfg.get('epsr', 1.0)
    print(f"# input:   {args.input}  T={bulk_cfg['T']}K", file=sys.stderr)
    print(f"# pore:    r_pore={args.r_pore}A  epsr={epsr_pore}  "
          f"wall_atom_r={args.wall_atom_r:.4f}A  eshift={args.eshift}A  "
          f"eps_lj={args.eps_lj}eV", file=sys.stderr)
    for ion in pore_cfg['ions']:
        name = ion.get('name', f"q={ion['q']:+g}")
        psi_str = f"{ion['psi_eV']:+.4f}" if ion['psi_eV'] is not None else "excluded"
        print(f"#   {name:<12} mu_bulk={ion['mu_bulk_eV']:+.4f}  "
              f"psi={psi_str}  ->  mu_pore={ion['mu_eV']:+.4f} eV",
              file=sys.stderr)

    with open(outfile, 'w') as f:
        json.dump(pore_cfg, f, indent=2)
    print(f"# pore JSON saved to {outfile}", file=sys.stderr)
    print(f"# run:  calc_pore.py -i {outfile}", file=sys.stderr)

    for ion in pore_cfg['ions']:
        if ion['psi_eV'] is None:
            name  = ion.get('name', f"q={ion['q']:+g}")
            r_min = ion['R'] + args.wall_atom_r
            print(f"# WARNING: ion '{name}' (R={ion['R']} A) excluded from pore "
                  f"(mu_pore={_MU_EXCLUDED_eV} eV); min r_pore = {r_min:.4f} A",
                  file=sys.stderr)


if __name__ == '__main__':
    main()
