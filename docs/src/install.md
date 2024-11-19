# [Installation](@id installation)

## Setting up Julia
If you have not yet installed Julia, please [follow the instructions on the
official website](https://julialang.org/downloads/). TrixiParticles.jl works
with Julia v1.9 and newer. We recommend using the latest stable release of Julia.

## For users
TrixiParticles.jl is a registered Julia package.
You can install TrixiParticles.jl,
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) (used for time integration)
and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) by executing the following commands
in the Julia REPL:
```julia
julia> using Pkg

julia> Pkg.add(["TrixiParticles", "OrdinaryDiffEq", "Plots"])
```

## [For developers](@id for-developers)
If you plan on editing TrixiParticles.jl itself, you can download TrixiParticles.jl
to a local folder and use the code from the cloned directory:
```bash
git clone git@github.com:trixi-framework/TrixiParticles.jl.git
cd TrixiParticles.jl
mkdir run
julia --project=run -e 'using Pkg; Pkg.develop(PackageSpec(path="."))' # Add TrixiParticles.jl to `run` project
julia --project=run -e 'using Pkg; Pkg.add(["OrdinaryDiffEq", "Plots"])' # Add additional packages
```

If you installed TrixiParticles.jl this way, you always have to start Julia with the
`--project` flag set to your `run` directory, e.g.,
```bash
julia --project=run
```
from the TrixiParticles.jl root directory.

The advantage of using a separate `run` directory is that you can also add other
related packages (e.g., OrdinaryDiffEq.jl, see above) to the project in the `run` folder
and always have a reproducible environment at hand to share with others.

## Optional software/packages
- [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) -- A Julia package of ordinary differential equation solvers that is used in the examples
- [Plots.jl](https://github.com/JuliaPlots/Plots.jl) -- Julia Plotting library that is used in some examples
- [PythonPlot.jl](https://github.com/JuliaPy/PythonPlot.jl) -- Plotting library that can be used instead of Plots.jl
- [ParaView](https://www.paraview.org/) -- Software that can be used for visualization of results

## [Common issues](@id installation-issues)

If you followed the [installation instructions for developers](@ref for-developers) and you
run into any problems with packages when pulling the latest version of TrixiParticles.jl,
start Julia with the project in the `run` folder,
```bash
   julia --project=run
```
update all packages in that project, resolve all conflicts in the project, and install all
new dependencies:
```julia
julia> using Pkg

julia> Pkg.update()

julia> Pkg.resolve()

julia> Pkg.instantiate()
```
