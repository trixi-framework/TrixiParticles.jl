# Installation

## Setting up Julia
If you have not yet installed Julia, please [follow the instructions for your
operating system](https://julialang.org/downloads/platform/). TrixiParticles.jl works
with Julia v1.9 and newer. We recommend using the latest stable release of Julia.

## Installation for Users


## Installation for Developers
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

If you installed TrixiParticles.jl this way, you always have to start Julia with the `--project`
flag set to your `run` directory, e.g.,
```bash
julia --project=.
```

## Optional Packages
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) -- a Julia package of Ordinary Differential Equation solvers that is used in the examples
- [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) -- Plotting library that is used in some examples for plotting
- [ParaView](https://www.paraview.org/) -- ParaView can be used for visualization of results

## Common Issues