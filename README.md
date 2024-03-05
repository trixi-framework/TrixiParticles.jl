# TrixiParticles.jl

[![CI](https://github.com/trixi-framework/TrixiParticles.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/trixi-framework/TrixiParticles.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/trixi-framework/TrixiParticles.jl/branch/main/graph/badge.svg?token=RDZXYbij0b)](https://codecov.io/github/trixi-framework/TrixiParticles.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

[![Youtube](https://img.shields.io/youtube/channel/views/UCpd92vU2HjjTPup-AIN0pkg?style=social)](https://www.youtube.com/@trixi-framework)
[![Slack](https://img.shields.io/badge/chat-slack-e01e5a)](https://join.slack.com/t/trixi-framework/shared_invite/zt-sgkc6ppw-6OXJqZAD5SPjBYqLd8MU~g)

**TrixiParticles.jl** is a numerical simulation framework designed for particle-based numerical methods, with an emphasis on multiphysics applications, written in [Julia](https://julialang.org).
A primary goal of the framework is to be user-friendly for engineering, science, and educational purposes. In addition to its extensible design and optimized implementation, we prioritize the user experience, including installation, pre- and postprocessing.
Its features include:

## Features
- Incompressible Navier-Stokes
  - Methods: Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH), Entropically Damped Artificial Compressibility (EDAC)
- Solid-body mechanics
  - Methods:  Total Lagrangian SPH (TLSPH)
- Fluid-Structure Interaction
- Output formats:
  - VTK

## Examples

## Installation
If you have not yet installed Julia, please [follow the instructions for your
operating system](https://julialang.org/downloads/platform/). TrixiParticles.jl works
with Julia v1.9 and newer. We recommend using the latest stable release of Julia.

### For users
<!--
TrixiParticles.jl is a registered Julia package. Hence, you
can install TrixiParticles.jl and OrdinaryDiffEq.jl (used by the examples) by executing the following commands in the Julia REPL:
```julia
julia> using Pkg

julia> Pkg.add(["OrdinaryDiffEq", "TrixiParticles"])
```
-->

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
**Note:** OrdinaryDiffEq is only necessary to run examples.

If you installed TrixiParticles.jl this way, you always have to start Julia with the `--project`
flag set to your `run` directory, e.g.,
```bash
julia --project=.
```
## Documentation

## Publications

## Cite Us

## Authors
Erik Faulhaber (University of Cologne) and Niklas Neher (HLRS) implemented the foundations for TrixiParticles.jl and are principal developers along with Sven Berger (hereon). The project was started by Michael Schlottke-Lakemper (RWTH Aachen University/HLRS) and Gregor Gassner (University of Cologne), who provide scientific direction and technical advice.
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
  <img align="middle" src="https://private-user-images.githubusercontent.com/10238714/278250248-50843ea0-6ce7-4849-b59c-5be0c55ea6f6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk2NDExMzQsIm5iZiI6MTcwOTY0MDgzNCwicGF0aCI6Ii8xMDIzODcxNC8yNzgyNTAyNDgtNTA4NDNlYTAtNmNlNy00ODQ5LWI1OWMtNWJlMGM1NWVhNmY2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA1VDEyMTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA4NGQ1NDAxYTViNjcxZTAxYWE3OGQxMDcwYmY1MzBiZDg2ZjY3MzdkYTkxOWEyN2Y2YzhlMmRlMWE4ZDNjOWUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.QIm_e6lzanFgC_KSx8a6Iu5w0mZomqBgZuz_uZ57lNw" width=20%/>&nbsp;&nbsp;&nbsp;
  <img align="middle" src="https://private-user-images.githubusercontent.com/10238714/278250497-a9791026-75a6-48fd-9fef-633e58b80725.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk2NDExMzQsIm5iZiI6MTcwOTY0MDgzNCwicGF0aCI6Ii8xMDIzODcxNC8yNzgyNTA0OTctYTk3OTEwMjYtNzVhNi00OGZkLTlmZWYtNjMzZTU4YjgwNzI1LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA1VDEyMTM1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZkY2I3NWI0OGE3MzMzMTJhMjlmYmE3NTZhNThkMjU2MDRjMTI3MzQ3NTIxZWQxMTVhMzhiYmM3ZTRlMTU2ODcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.9wG_a378-DBm6sbxfU_jd7YksMlU9QkMhqOjkekekmA" width=20%/>&nbsp;&nbsp;&nbsp;
</p>

The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).
