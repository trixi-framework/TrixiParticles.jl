# [Installation](@id installation)

## Setting up Julia
If you have not installed Julia yet, please [follow the instructions on the
official website](https://julialang.org/downloads/). TrixiParticles.jl works
with Julia v1.10 and newer. We recommend using the latest stable release of Julia.

## For users
TrixiParticles.jl is a registered Julia package.
You can install TrixiParticles.jl,
time integration sub-packages of [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
(for example `OrdinaryDiffEqLowStorageRK`), and [Plots.jl](https://github.com/JuliaPlots/Plots.jl)
by executing the following commands
in the Julia REPL:
```julia
julia> using Pkg

julia> Pkg.add(["TrixiParticles", "OrdinaryDiffEqLowStorageRK", "OrdinaryDiffEqSymplecticRK", "Plots"])
```

## [For developers](@id for-developers)
If you plan on editing TrixiParticles.jl itself, you can download TrixiParticles.jl
to a local folder and use the code from the cloned directory:
```bash
git clone git@github.com:trixi-framework/TrixiParticles.jl.git
cd TrixiParticles.jl
mkdir run
julia --project=run -e 'using Pkg; Pkg.develop(PackageSpec(path="."))' # Add TrixiParticles.jl to `run` project
julia --project=run -e 'using Pkg; Pkg.add(["OrdinaryDiffEqLowStorageRK", "OrdinaryDiffEqSymplecticRK", "Plots"])' # Add additional packages
```

If you installed TrixiParticles.jl this way, you always have to start Julia with the
`--project` flag set to your `run` directory, e.g.,
```bash
julia --project=run
```
from the TrixiParticles.jl root directory.

The advantage of using a separate `run` directory is that you can also add other
related packages (e.g., sub-packages of OrdinaryDiffEq.jl, see above) to the project
in the `run` folder
and always have a reproducible environment at hand to share with others.

## Optional software/packages
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) -- A Julia package of ordinary differential equation solvers; examples in TrixiParticles.jl use sub-packages such as `OrdinaryDiffEqLowStorageRK` and `OrdinaryDiffEqSymplecticRK`
- [Plots.jl](https://github.com/JuliaPlots/Plots.jl) -- Julia Plotting library that is used in some examples
- [PythonPlot.jl](https://github.com/JuliaPy/PythonPlot.jl) -- Plotting library that can be used instead of Plots.jl
- [ParaView](https://www.paraview.org/) -- Visualization software for simulation results

## [Common issues](@id installation-issues)

If you followed the [installation instructions for developers](@ref for-developers) and run
into package issues after pulling the latest version of TrixiParticles.jl, start Julia with
the project in the `run` folder,
```bash
   julia --project=run
```
then update packages, resolve dependency conflicts, and install new dependencies:
```julia
julia> using Pkg

julia> Pkg.update()

julia> Pkg.resolve()

julia> Pkg.instantiate()
```
