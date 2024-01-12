# Get Started

If you have not installed **TrixiParticles.jl** follow the instructions given [here](install.md).

In the following examples we will run through some easy first steps to get a more thorough discussion take a look at our [tutorial section](tutorial.md).

## Running an Example
To run pick one of our examples [see also](examples.md) and execute as follows:

```bash
cd path/to/TrixiParticles.jl/
julia --project=. examples/group/example.jl
```

or in the REPL

```bash
cd path/to/TrixiParticles.jl/
julia --project=. 
include("examples/group/example.jl")
```

Per default the results will be written to a folder "out". Depending on the example you are trying to run you will find at least one "*.pvd", which can be opened using ParaView (see example below for pictures)

### Running multithreaded
To run an example with multithreading add the flag "-t n" with "n" being the number of threads.
See also the example below:

```bash
cd path/to/TrixiParticles.jl/
julia -t 4 --project=. examples/group/example.jl
```

or in the REPL

```bash
cd path/to/TrixiParticles.jl/
julia -t 4 --project=. 
include("examples/group/example.jl")
```


## Running fluid/dam\_break\_2d
In the following we will run the example "fluid/dam\_break\_2d to provide some more consistent details

```bash
cd path/to/TrixiParticles.jl/
julia -t 4 --project=. examples/fluid/dam_break_2d.jl
```

This will result in the following:
![image](https://github.com/svchb/TrixiParticles.jl/assets/10238714/f8d2c249-fd52-4958-bc8b-265bbadc49f2)

Afterwards you will find the following files in the "out" directory:
- fluid_1_x.vtu -- Solution files of the *first* fluid at output number 'x'
- boundary_1_x.vtu -- Solution files of the *first* boundary at output number 'x'
- fluid_1.pvd -- Collection of the first fluid's solution files
- boundary_1.pvd -- Collection of the first boundaries solution files

We can now view these files by opening them in **ParaView**:

1. Click file open
2. Navigate to the out directory
3. Open both "boundary_1.pvd" and "fluid_1.pvd"
4. Click "Apply" which by default is on the left pane below the "Pipeline Browser"
5. To move the solution around **hold the left mouse button**

You will now see the following:
![image](https://github.com/svchb/TrixiParticles.jl/assets/10238714/45c90fd2-984b-4eee-b130-e691cefb33ab)

To now view the result variables **first** make sure you have "fluid_1.pvd" highlighted in the "Pipeline Browser" then select them in the variable selection combo box (see picture below).
Lets, for example pick "density". To now view the time progression of the result hit the "play button" (see picture below).
![image](https://github.com/svchb/TrixiParticles.jl/assets/10238714/7565a13f-9532-4a69-9f81-e79505400b1c)


## Modifying an Example
Open the "dam\_break\_2d.jl" in you favorite text editor and lets make some edits:
- increase time span from `tspan = (0.0, 5.7 / sqrt(gravity))` to `tspan = (0.0, 3.0)` to simulate a longer time span
- increase the height of the initial fluid from `initial_fluid_size = (2.0, 1.0)` to `initial_fluid_size = (2.0, 2.0)`

Run the simulation as before. The original files will be overwritten and will appear in **ParaView** as they are generated.


## Setup you first simulation from scratch
Please see the following [tutorial](tutorials/tut_setup.md). 

For more information follow the [tutorial series](tutorial.md)
