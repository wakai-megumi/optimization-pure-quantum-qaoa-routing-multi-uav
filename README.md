# QUBO/QAOA benchmarking for single-UAV routing under budget constraints

Code accompanying the manuscript *Quantum-Enhanced Multi-Objective Route Planning for UAV Swarms under SORA Airspace Risk and Environmental Constraints* (Tarun Sharma, Keshav Singh Rawat; submitted to Quantum Machine Intelligence).

The code is an offline benchmark of QUBO encodings and QAOA variants on small single-UAV routing instances (at most 5 nodes). All quantum results come from noiseless statevector simulation. Nothing here runs on quantum hardware or controls a UAV.

## What is implemented

- QUBO for a time-window-constrained tour with a composite arc cost (travel, time, risk, energy) and a single penalty weight.
- Budget-augmented QUBOs: binary slack registers for an energy budget, a risk budget, or both.
- QAOA variants at depth p = 2: Standard, CVaR (alpha = 0.25), and Warm-Start with epsilon in {1e-4, 0.1, 0.25}. Optimizer: COBYLA.
- Classical baselines: greedy, simulated annealing, budget-matched random-restart local search (RRLS), relaxation rounding, uniform sampling.
- Instance families S (synthetic), D (derived from public Dumas et al. instances), H (harder, small cost gap), R (risk-binding twins of S07-S10) and J (joint energy and risk).

## Installation

```
pip install -r requirements.txt
```

Python 3.10 or newer. The self-test was checked with Python 3.13, NumPy 2.5, SciPy 1.18. Run all scripts from this folder: they import each other by file name.

## Files

| File | Role |
|---|---|
| `gusrn_qaoa_patched.py` | Base module: statevector engine, Standard/CVaR/Warm-Start QAOA classes, decoder, classical baselines. Also contains earlier instance templates not used for the reported results. |
| `gusrn_v2.py` | Main pipeline: families S and D, QUBO builder, corrected warm-start gradient, sampling, baselines, reports. |
| `harder_instances.py` | Family H generator and runner. |
| `sweep_highp.py` | Extended-depth sweep (Standard, CVaR, Warm-Start). |
| `csweep_and_transfer.py` | Cost-weight sweep and penalty-scale margin sweep. |
| `budget_qubo.py` | Budget-augmented QUBO (energy or risk), slack calibration, QAOA on S07-S10. Imported by the joint scripts. |
| `risk_binding.py` | Path-dependent risk model, family R twins, fast slack-projected exactness check. |
| `risk_qaoa.py` | QAOA on family R. |
| `combined_binding.py` | Helpers for two slack registers, imported by `joint_family.py`. |
| `joint_family.py` | Family J generation, strict calibration, robustness tiers, hierarchy check. |
| `joint_qaoa.py` | QAOA on family J. |
| `joint_qaoa_normalized.py` | Secondary, post hoc analysis with H divided by its standard deviation. |

## Reproducing the results

Each stage writes to its own `*_results/` folder and can be resumed.

```
python gusrn_v2.py --selftest
python gusrn_v2.py --preset standard --stage all                  # families S, D
python gusrn_v2.py --stage report
python harder_instances.py --n-instances 10 --min-feasible 6 --max-gap 0.08   # family H
python sweep_highp.py --depths 1,2,4,6,8,12,16,20
python csweep_and_transfer.py --stage all
python budget_qubo.py --mode full --preset quick --only S07       # repeat for S08, S09, S10
python risk_binding.py --stage all                                # family R twins
python risk_qaoa.py --mode full --preset quick
python joint_family.py --stage generate --n-instances 4
python joint_family.py --stage select
python joint_family.py --stage robustness
python joint_family.py --stage hierarchy
python joint_qaoa.py --mode full --preset quick --tier k1
python joint_qaoa_normalized.py --mode full --preset quick
```

The commands above show the interface. The exact arguments behind each table in the paper (instance counts, seed ranges, presets) are listed in the paper's experimental-setup section.

## Resources

Statevector simulation needs 2^n complex amplitudes. The budget-augmented and joint instances reach 26 to 30 qubits. Those runs need several GB of RAM and, for some instances, hours per variant and seed. The base instances (up to 19 qubits) run in seconds to minutes.

## Notes on reproducibility

- Seeds are fixed in the code, but results depend on library versions (SciPy's COBYLA in particular).
- `joint_family.py`, `joint_qaoa.py` and `joint_qaoa_normalized.py` store a hash of their own source (`code_sha`) with every result. These files are the comment-free release, so the hash differs from the one in result files produced by the original commented sources.
- `budget_qubo.py` still reads and writes `./budget_qubo_v3_results/` so that earlier cached runs stay compatible.

## Citation

See `CITATION.cff`. Please cite the paper and the Zenodo archive (DOI to be added after release).

## License

MIT, see `LICENSE`.
