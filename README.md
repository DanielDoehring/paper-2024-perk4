# paper-2024-perk4
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/863491454.svg)](https://doi.org/10.5281/zenodo.13899246)

This repository contains information and code to reproduce the results presented in the article
```bibtex
@online{doehring2024fourth,
  title={Fourth-Order Paired-Explicit Runge-Kutta Methods},
  author={Doehring, Daniel and Christmann, Lars and Schlottke-Lakemper, Michael and Gassner, Gregor J.
          and Torrilhon, Manuel},
  year={2024},
  eprint={2408.05470},
  eprinttype={arxiv},
  eprintclass={math.NA},
  url={https://arxiv.org/abs/2408.05470},
  journal={arXiv preprint arXiv:2408.05470},
  doi={10.48550/arXiv.2408.05470}
}
```

If you find these results useful, please cite the article mentioned above. If you use the implementations provided here, please also cite this repository as
```bibtex
@misc{doehring2024fourthRepro,
  title={Reproducibility repository for "Fourth-Order Paired-Explicit Runge-Kutta Methods"},
  author={Doehring, Daniel and Christmann, Lars and Schlottke-Lakemper, Michael and Gassner, Gregor J.
          and Torrilhon, Manuel},
  year={2024},
  howpublished={\url{https://github.com/DanielDoehring/paper-2024-perk4}},
  doi={https://doi.org/10.5281/zenodo.13899247}
}
```

## Abstract

In this paper, we extend the Paired-Explicit Runge-Kutta (P-ERK) schemes by Vermeire et. al. [[1](https://doi.org/10.1016/j.jcp.2019.05.014),[2](https://doi.org/10.1016/j.jcp.2022.111470)] to fourth-order of consistency.
Based on the order conditions for partitioned Runge-Kutta methods we motivate a specific form of the Butcher arrays which leads to a family of fourth-order accurate methods.
The employed form of the Butcher arrays results in a special structure of the stability polynomials, which needs to be adhered to for an efficient optimization of the domain of absolute stability.
The P-ERK schemes enable multirate time-integration with no changes in the spatial discretization methodology, making them readily implementable in existing codes that employ a method-of-lines approach.

We demonstrate that the constructed fourth-order P-ERK methods satisfy linear stability, internal consistency, designed order of convergence, and conservation of linear invariants.
At the same time, these schemes are seamlessly coupled for codes employing a method-of-lines approach, in particular without any modifications of the spatial discretization.
We demonstrate speedup for single-threaded program executions, shared-memory parallelism, i.e., multi-threaded executions and distributed-memory parallelism with MPI.

We apply the multirate P-ERK schemes to inviscid and viscous problems with locally varying wave speeds, which may be induced by non-uniform grids or multiscale properties of the governing partial differential equation.
Compared to state-of-the-art optimized standalone methods, the multirate P-ERK schemes allow significant reductions in right-hand-side evaluations and wall-clock time, ranging from 66% up to factors greater than four.

A reproducibility repository is provided which enables the reader to examine all results presented in this work.
## Reproducing the results

### Installation

To download the code using `git`, use 

```bash
git clone git@github.com:DanielDoehring/paper-2024-perk4.git
``` 

If you do not have git installed you can obtain a `.zip` and unpack it:
```bash
wget https://github.com/DanielDoehring/paper-2024-perk4/archive/refs/heads/main.zip
unzip main.zip -d paper-2024-perk4
```

To instantiate the Julia environment execute the following two commands:
```bash
cd paper-2024-perk4/paper-2024-perk4-main
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Note that the results are obtained using Julia 1.10.4, which is also set in the `Manifest.toml`.
Thus, you might need to install the [old Julia 1.10.4 release](https://julialang.org/downloads/oldreleases/) first
and *replace* the `julia` calls from this README with
`/YOUR/PATH/TO/julia-1.10.4/bin/julia`

### Project initialization

If you installed Trixi.jl this way, you always have to start Julia with the `--project` flag set to your `paper-2024-perk4` directory, e.g.,
```bash
julia --project=.
```
if already inside the `paper-2024-perk4` directory.

If you do not execute from the `paper-2024-perk4` directory, you have to call `julia` with
```bash
julia --project=/YOUR/PATH/TO/paper-2024-perk4
```

### Running the code

The scripts for validations and applications are located in the `5_Validation` and `6_Applications` directory, respectively.

To execute them provide the respective path:

```bash
julia --project=. ./5_Validation/5_1_LinearStability/elixir_advection_linear_stability.jl
```

For all cases in the `applications` directory the solution has been computed using a specific number of 
threads.
To specify the number of threads the [`--threads` flag](https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads) needs to be specified, i.e., 
```bash
julia --project=. --threads 24 ./6_Applications/6_4_SD7003Airfoil/Integrators_PERK4.jl
```
The number of threads used for the examples are given in the `README.md` in `6_Applications`.

## Authors

* [Daniel Doehring](https://www.acom.rwth-aachen.de/the-lab/team-people/name:daniel_doehring) (Corresponding Author)
* [Lars Christmann](https://github.com/lchristm)
* [Michael Schlottke-Lakemper](https://lakemper.eu/)
* [Gregor J. Gassner](https://www.mi.uni-koeln.de/NumSim/gregor-gassner/)
* [Manuel Torrilhon](https://www.acom.rwth-aachen.de/the-lab/team-people/name:manuel_torrilhon)

Note that the Trixi authors are listed separately [here](https://github.com/trixi-framework/paper-2024-amr-paired-rk/blob/main/Trixi.jl-v0.5.42%2Bmod/AUTHORS.md).

## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
