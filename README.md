# TrixiParticles.jl

[![CI](https://github.com/trixi-framework/TrixiParticles.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/trixi-framework/TrixiParticles.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/trixi-framework/TrixiParticles.jl/branch/main/graph/badge.svg?token=RDZXYbij0b)](https://codecov.io/github/trixi-framework/TrixiParticles.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

[![Youtube](https://img.shields.io/youtube/channel/views/UCpd92vU2HjjTPup-AIN0pkg?style=social)](https://www.youtube.com/@trixi-framework)
[![Slack](https://img.shields.io/badge/chat-slack-e01e5a)](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)
[![Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/TrixiParticles)](https://pkgs.genieframework.com?packages=TrixiParticles)


TrixiParticles.jl is a code that implements particle based numerical methods for multiphysics applications.

## Features
- Incompressible Navier-Stokes
  - Methods: WCSPH, EDAC
- Solid-body mechanics
  - Methods: TLSPH
- Fluid-Structure Interaction
- Output formats:
  - VTK
 
## Examples
 
## Installation
If you have not yet installed Julia, please [follow the instructions for your
operating system](https://julialang.org/downloads/platform/). TrixiParticles.jl works
with Julia v1.9 and newer. We recommend using the latest stable release of Julia.

### For users
TrixiParticles.jl is a registered Julia package. Hence, you
can install TrixiParticles.jl and OrdinaryDiffEq.jl (used by the examples) by executing the following commands in the Julia REPL:
```julia
julia> using Pkg

julia> Pkg.add(["OrdinaryDiffEq", "TrixiParticles"])
```

### For developers
If you plan on editing TrixiParticles.jl itself, you can download TrixiParticles.jl locally and use the
code from the cloned directory:
```bash
git clone git@github.com:trixi-framework/TrixiParticles.jl.git
cd TrixiParticles.jl
mkdir run
cd run
julia --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path=".."))' # Install locally
julia --project=. -e 'using Pkg; Pkg.add("OrdinaryDiffEq")' # Install additional packages
```
**Note:** OrdinaryDiffEq is only necessary to run examples

If you installed TrixiParticles.jl this way, you always have to start Julia with the `--project`
flag set to your `run` directory, e.g.,
```bash
julia --project=.
```
## Documentation

## Publications

## Cite Us

## Authors
Erik Faulhaber (University of Cologne) and Niklas Neher (HLRS) implemented the foundations for TrixiParticles.jl and are principal developers along with Sven Berger (hereon). The project was started by Michael Schlottke-Lakemper (RWTH Aachen University/HLRS, Germany) and Gregor Gassner (University of Cologne, Germany), who provide scientific direction and technical advice.
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
  <img align="middle" src="https://github.com/svchb/TrixiParticles.jl/assets/10238714/e6c56282-b905-4499-8b88-e3b2c8f54545.jpg" width=20%/>&nbsp;&nbsp;&nbsp;
  <img align="middle" src="https://github.com/svchb/TrixiParticles.jl/assets/10238714/06aac9fa-a446-4e7b-91e7-4faeeacb9813.jpg" width=20%/>&nbsp;&nbsp;&nbsp;
</p>

The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).
