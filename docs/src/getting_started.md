# [Getting started](@id getting_started)
If you have not installed TrixiParticles.jl, please follow the instructions given [here](install.md).

In the following sections, we will give a short introduction. For a more thorough discussion, take a look at our [Tutorials](tutorial.md).

## Running an Example
The easiest way to run a simulation is to run one of our predefined example files.
We will run the file `examples/fluid/hydrostatic_water_column_2d.jl`, which simulates a fluid resting in a rectangular tank.
Since TrixiParticles.jl uses multithreading, you should start Julia with the flag `--threads auto` (or, e.g. `--threads 4` for 4 threads).

In the Julia REPL, first load the package TrixiParticles.jl.
```jldoctest getting_started
julia> using TrixiParticles
```

Then start the simulation by executing
```jldoctest getting_started; filter = r".*"s
julia> trixi_include(joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"))
```

This will result in the following:
![image](https://github.com/svchb/TrixiParticles.jl/assets/10238714/f8d2c249-fd52-4958-bc8b-265bbadc49f2)

To visualize the results, see [Visualization](visualization.md).

## Running other Examples
You can find a list of our other predefined examples under [Examples](examples.md).
Execute them as follows from the Julia REPL by replacing `subfolder` and `example_name`
```julia
julia> trixi_include(joinpath(examples_dir(), "subfolder", "example_name.jl"))
```

## Modifying an example
You can pass keyword arguments to the function `trixi_include` to overwrite assignments in the file.

With `trixi_include`, we can overwrite variables defined in the example file to run a different simulation without modifying the example file.
```jldoctest getting_started; filter = r".*"s
julia> trixi_include(joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), initial_fluid_size=(1.0, 0.5))
```
This for example, will change the fluid size from ``(0.9, 1.0)`` to ``(1.0, 0.5)``.

To understand why, take a look into the file `hydrostatic_water_column_2d.jl` in the subfolder `fluid` inside the examples directory, which is the file that we executed earlier.
You can see that the initial size of the fluid is defined in the variable `initial_fluid_size`, which we could overwrite with the `trixi_include` call above.
Another variable that is worth experimenting with is `fluid_particle_spacing`, which controls the resolution of the simulation in this case.
A lower value will increase the resolution and the runtime.

## Set up you first simulation from scratch
See [Set up your first simulation](tutorials/tut_setup.md).

Find an overview over the available tutorials under [Tutorials](tutorial.md).
