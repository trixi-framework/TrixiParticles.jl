# Geting started  
If you have not installed TrixiParticles.jl, please follow the instructions given [here](install.md).

In the following examples we will run through some easy first steps to get a more thorough discussion take a look at our [tutorial section](tutorial.md).

## Running an Example
The easiest way to run a simulation is to run one of our pre-defined example files.
We will run the file `examples/fluid/rectangular_tank_2d.jl`, which simulates a fluid resting in a rectangular tank.
Since TrixiParticles.jl uses multithreading, you should start Julia with the flag `--threads auto` or `--threads 4` (for 4 threads).
In the Julia REPL, first load the package TrixiParticles.jl.

```julia
julia> using TrixiParticles
```

Then start the simulation by executing
```julia
julia> trixi_include(joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"))
```

This will result in the following:
![image](https://github.com/svchb/TrixiParticles.jl/assets/10238714/f8d2c249-fd52-4958-bc8b-265bbadc49f2)


To visualize the results, see our [visualization](visualization.md) page.

## Running other Examples
To run pick one of our examples [see also](examples.md) and execute them as follows from Julia REPL by replacing `folder` and `example_name`

```julia
julia> trixi_include(joinpath(examples_dir(), "folder", "example_name.jl"))
```

## Modifying an example
You can pass keyword arguments to the function `trixi_include` to overwrite assignments in the file.

With `trixi_include`, we can overwrite this variable to run a different simulation without modifying the example file.
```julia
julia> trixi_include(joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"), initial_fluid_size=(1.0, 0.5))
```
This for example, will change the fluid size from (1.0, 1.0) to (1.0, 0.5) and will show up as half the height.

To explore further you can take a look into the file `examples/fluid/rectangular_tank_2d.jl` that we executed earlier,
you can see that the initial size of the fluid is defined in the variable `initial_fluid_size`. 
A variable that also yields it self for experimentation is `fluid_particle_spacing`, which controls the resolution of the simulation in this case.
A lower value will increase the resolution and the runtime.

## Setup you first simulation from scratch
Please see the following [Setup your first simulation](tutorials/tut_setup.md). 

For an overview over the available [tutorials](tutorial.md).
