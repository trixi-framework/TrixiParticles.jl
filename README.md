# TrixiParticles.jl

[![Docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://trixi-framework.github.io/TrixiParticles.jl/stable)
[![Docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://trixi-framework.github.io/TrixiParticles.jl/dev)
[![Slack](https://img.shields.io/badge/chat-slack-e01e5a)](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)
[![Youtube](https://img.shields.io/youtube/channel/views/UCpd92vU2HjjTPup-AIN0pkg?style=social)](https://www.youtube.com/@trixi-framework)
[![CI](https://github.com/trixi-framework/TrixiParticles.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/trixi-framework/TrixiParticles.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/trixi-framework/TrixiParticles.jl/branch/main/graph/badge.svg?token=RDZXYbij0b)](https://codecov.io/github/trixi-framework/TrixiParticles.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10797541.svg)](https://zenodo.org/doi/10.5281/zenodo.10797541)

**TrixiParticles.jl** is a numerical simulation framework designed for particle-based numerical methods, with an emphasis on multiphysics applications, written in [Julia](https://julialang.org).
A primary goal of the framework is to be user-friendly for engineering, science, and educational purposes. In addition to its extensible design and optimized implementation, we prioritize the user experience, including installation, pre- and postprocessing.
Its features include:

## Features
- Incompressible Navier-Stokes
  - Methods: Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH), Entropically Damped Artificial Compressibility (EDAC)
  - Models: Surface Tension
- Solid-body mechanics
  - Methods:  Total Lagrangian SPH (TLSPH), Discrete Element Method (DEM)
- Fluid-Structure Interaction
- Output formats:
  - VTK

## Examples
We provide several example simulation setups in the `examples` folder (which can be accessed from Julia via `examples_dir()`).

<table align="center" border="0">
  <tr>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/683e9363-5705-49cc-9a5c-3b47d73ea4b8" style="width: 80% !important;"/><br><figcaption>2D Dam Break</figcaption>
    </td>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/c10faddf-0400-47c9-b225-f5d286a8ecb8" style="width: 80% !important;"/><br><figcaption>Moving Wall</figcaption>
    </td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/e05ace63-e330-441a-a391-eda3d2764074" style="width: 80% !important;"/><br><figcaption>Oscillating Beam</figcaption>
    </td>
    <td align="center">
      <img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/ada0d554-e0ba-44ed-923d-2b77ef252258" style="width: 80% !important;"/><br><figcaption>Dam Break with Elastic Plate</figcaption>
    </td>
  </tr>
</table>


## Installation
If you have not yet installed Julia, please [follow the instructions for your
operating system](https://julialang.org/downloads/platform/). TrixiParticles.jl works
with Julia v1.9 and newer. We recommend using the latest stable release of Julia.

### For users
TrixiParticles.jl is a registered Julia package.
You can install TrixiParticles.jl,
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) (used for time integration)
and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) by executing the following commands
in the Julia REPL:
```julia
julia> using Pkg

julia> Pkg.add(["TrixiParticles", "OrdinaryDiffEq", "Plots"])
```

### For developers
If you plan on editing TrixiParticles.jl itself, you can download TrixiParticles.jl
to a local folder and use the code from the cloned directory:
```bash
git clone git@github.com:trixi-framework/TrixiParticles.jl.git
cd TrixiParticles.jl
mkdir run
julia --project=run -e 'using Pkg; Pkg.develop(PackageSpec(path="."))' # Add TrixiParticles.jl to `run` project
julia --project=run -e 'using Pkg; Pkg.add("OrdinaryDiffEq", "Plots")' # Add additional packages
```

If you installed TrixiParticles.jl this way, you always have to start Julia with the
`--project` flag set to your `run` directory, e.g.,
```bash
julia --project=run
```
from the TrixiParticles.jl root directory.
Further details can be found in the [documentation](https://trixi-framework.github.io/TrixiParticles.jl/stable).

## Usage

In the Julia REPL, first load the package TrixiParticles.jl.
```jldoctest getting_started
julia> using TrixiParticles
```

Then start the simulation by executing
```jldoctest getting_started; filter = r".*"s
julia> trixi_include(joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"))
```

This will open a new window with a 2D visualization of the final solution:
<img src="https://github.com/trixi-framework/TrixiParticles.jl/assets/44124897/95821154-577d-4323-ba57-16ef02ea24e0" width="400">

Further details can be found in the [documentation](https://trixi-framework.github.io/TrixiParticles.jl/stable).

## Documentation

You can find the documentation for the latest release
[here](https://trixi-framework.github.io/TrixiParticles.jl/stable).

## Publications

## Cite Us

If you use TrixiParticles.jl in your own research or write a paper using results obtained
with the help of TrixiParticles.jl, please cite it as
```bibtex
@misc{trixiparticles,
  title={{T}rixi{P}articles.jl: {P}article-based multiphysics simulations in {J}ulia},
  author={Erik Faulhaber and Niklas Neher and Sven Berger and
          Michael Schlottke-Lakemper and Gregor Gassner},
  year={2024},
  howpublished={\url{https://github.com/trixi-framework/TrixiParticles.jl}},
  doi={10.5281/zenodo.10797541}
}
```

## Authors
Erik Faulhaber (University of Cologne) and Niklas Neher (HLRS) implemented the foundations
for TrixiParticles.jl and are principal developers along with Sven Berger (hereon).
The project was started by Michael Schlottke-Lakemper (University of Augsburg)
and Gregor Gassner (University of Cologne), who provide scientific direction and technical advice.
The full list of contributors can be found in [AUTHORS.md](AUTHORS.md).

## License and contributing
TrixiParticles.jl is licensed under the MIT license (see [LICENSE.md](LICENSE.md)). Since TrixiParticles.jl is
an open-source project, we are very happy to accept contributions from the
community. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
Note that we strive to be a friendly, inclusive open-source community and ask all members
of our community to adhere to our [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
To get in touch with the developers,
[join us on Slack](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)
or [create an issue](https://github.com/trixi-framework/TrixiParticles.jl/issues/new).

## Acknowledgments
<p align="center">
  <img align="middle" src="https://github.com/trixi-framework/TrixiParticles.jl/assets/44124897/05132bf1-180f-4228-b30a-37dfb6e36ed5" width=20%/>&nbsp;&nbsp;&nbsp;
  <img align="middle" src="https://github.com/trixi-framework/TrixiParticles.jl/assets/44124897/ae2a91d1-7c10-4e0f-8b92-6ed1c43ddc28" width=20%/>&nbsp;&nbsp;&nbsp;
</p>

The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).
