# Get Started

If you have not installed TrixiParticles.jl follow the instructions given [here](install.md).

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



## Modifying an Example


## Setup you first simulation from scratch


For more information follow the [tutorial series](tutorial.md)