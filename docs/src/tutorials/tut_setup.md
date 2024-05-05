# Setting up your simulation from scratch

In this tutorial, we will guide you through the general structure of simulation files.
We will set up a simulation similar to the example simulation
[`examples/fluid/hydrostatic_water_column_2d.jl`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/examples/fluid/hydrostatic_water_column_2d.jl),
which is one of our simplest example simulations.
In the second part of this tutorial, we will show how to replace components
of TrixiParticles.jl by custom implementations from within a simulation file,
without ever cloning the repository.

For different setups and physics, have a look at [our other example files](@ref examples).

## Resolution

At the beginning of most simulation files, we define the numerical resolution,
so that it can easily be found and changed.
First, we import TrixiParticles.jl and
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), which we will
use at the very end for the time integration.
```jldoctest tut_setup; output = false
using TrixiParticles
using OrdinaryDiffEq

# output

```
Now, we define the particle spacing, which is our numerical resolution.
We also set the number of boundary layers, which need to be sufficiently
large, depending on the smoothing kernel and smoothing length, so that
the compact support of the smoothing kernel is fully sampled with particles
for a fluid particle close to a boundary.
```jldoctest tut_setup; output = false
fluid_particle_spacing = 0.05
boundary_layers = 3

# output
3
```

## Experiment setup

We want to simulate a water column resting under hydrostatic pressure inside
a rectangular tank.
First, we define the physical parameters gravitational acceleration,
initial fluid size, tank size, fluid density, and simulation time.
```jldoctest tut_setup; output = false
gravity = 9.81
tspan = (0.0, 1.0)
initial_fluid_size = (1.0, 0.9)
tank_size = (1.0, 1.0)
fluid_density = 1000.0

# output
1000.0
```

In order to have the initial particle mass and density correspond to the
hydrostatic pressure gradient, we need to define a state equation, which
relates the fluid density to pressure.
Note that we could also skip this part here and define the state equation
later when we define the fluid system, but then the fluid would be initialized
with constant density, which would cause it to oscillate under gravity.
```jldoctest tut_setup; output = false
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

# output
StateEquationCole{Float64, false}(10.0, 7.0, 1000.0, 0.0)
```
The speed of sound here is numerical and not physical.
We artificially lower the speed of sound, since the physical speed of sound
in water would lead to prohibitively small time steps.
The speed of sound in Weakly Compressible SPH should be chosen as small as
possible for numerical efficiency, but large enough to limit density fluctuations
to about 1%.

TrixiParticles.jl requires the initial particle positions and quantities in
form of an [`InitialCondition`](@ref).
Instead of manually defining particle positions, you can work with our
pre-defined setups.
Among others, we provide setups for rectangular shapes, circles, and spheres.
Initial conditions can also be combined with common set operations.
See [this page](@ref initial_condition) for a list of pre-defined setups
and details on set operations on initial conditions.

Here, we use the [`RectangularTank`](@ref) setup, which generates a rectangular
fluid inside a rectangular tank, and supports a hydrostatic pressure gradient
by passing a gravitational acceleration and a state equation (see above).
```jldoctest tut_setup; output = false, filter = r"RectangularTank.*"
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size,
                       fluid_density, n_layers=boundary_layers,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# output
RectangularTank{2, 4, Float64}(InitialCondition{Float64}(...))
```

## Fluid system

To model the water column, we use the Weakly Compressible Smoothed Particle
Hydrodynamics (WCSPH) method.
This method requires a smoothing kernel and a corresponding smoothing length,
which should be chosen in relation to the particle spacing.
```jldoctest tut_setup; output = false
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# output
SchoenbergCubicSplineKernel{2}()
```
You can find an overview over smoothing kernels and corresponding smoothing
lengths [here](@ref smoothing_kernel).

For stability, we need numerical dissipation in form of an artificial viscosity
term.
```jldoctest tut_setup; output = false
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

# output
ArtificialViscosityMonaghan{Float64}(0.02, 0.0, 0.01)
```
We choose the parameters as small as possible to avoid visible viscosity,
but as large as possible to stabilize the simulation.

The WCSPH method can either compute the particle density by a kernel summation
over all neighboring particles (see [`SummationDensity`](@ref)) or by making
the particle density a variable in the ODE system and integrating it over time.
We choose the latter approach here by using the density calculator
[`ContinuityDensity`](@ref).
```jldoctest tut_setup; output = false
fluid_density_calculator = ContinuityDensity()
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ WeaklyCompressibleSPHSystem{2}                                                                   │
│ ══════════════════════════════                                                                   │
│ #particles: ………………………………………………… 360                                                              │
│ density calculator: …………………………… ContinuityDensity                                                │
│ correction method: ……………………………… Nothing                                                          │
│ state equation: ……………………………………… StateEquationCole                                                │
│ smoothing kernel: ………………………………… SchoenbergCubicSplineKernel                                      │
│ viscosity: …………………………………………………… ArtificialViscosityMonaghan{Float64}(0.02, 0.0, 0.01)            │
│ density diffusion: ……………………………… nothing                                                          │
│ acceleration: …………………………………………… [0.0, -9.81]                                                     │
│ source terms: …………………………………………… Nothing                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Boundary system

In order to define a boundary system, we first have to choose a boundary model,
which defines how the fluid interacts with boundary particles.
We will use the [`BoundaryModelDummyParticles`](@ref) with
[`AdamiPressureExtrapolation`](@ref), which generally produces the best results
of the implemented methods.
See [here](@ref boundary_models) for a comprehensive overview over boundary models.
```jldoctest tut_setup; output = false
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length)
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ BoundarySPHSystem{2}                                                                             │
│ ════════════════════                                                                             │
│ #particles: ………………………………………………… 276                                                              │
│ boundary model: ……………………………………… BoundaryModelDummyParticles(AdamiPressureExtrapolation, Nothing) │
│ movement function: ……………………………… nothing                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Semidiscretization

The key component of every simulation is the [`Semidiscretization`](@ref),
which couples all systems of the simulation.
All methods in TrixiParticles.jl are semidiscretizations, which discretize
the equations in time to provide an ordinary differential equation that still
has to be solved in time.
By providing a simulation time span, we can call [`semidiscretize`](@ref),
which returns an `ODEProblem` that can be solved with a time integration method.
```jldoctest tut_setup; output = false, filter = r"ODEProblem with .*"s
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

# output
ODEProblem with uType RecursiveArrayTools.ArrayPartition...
```

## Time integration

We use the methods provided by
[OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl),
but note that other packages or custom implementations can also be used.

OrdinaryDiffEq.jl supports callbacks, which are executed during the simulation.
For this simulation, we use the [`InfoCallback`](@ref), which prints
information about the simulation setup at the beginning of the simulation,
information about the current simulation time and runtime during the simulation,
and a performance summary at the end of the simulation.
We also want to save the current solution in regular intervals in terms of
simulation time as VTK, so that we can look at the solution in ParaView.
The [`SolutionSavingCallback`](@ref) provides this functionality.
To pass the callbacks to OrdinaryDiffEq.jl, we have to bundle them into a
`CallbackSet`.
```jldoctest tut_setup; output = false, filter = r"CallbackSet.*"
info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

# output
CallbackSet{Tuple{}, ...}
```

Finally, we can start the simulation by solving the `ODEProblem`.
We use the method `RDPK3SpFSAL35` of OrdinaryDiffEq.jl, which is a Runge-Kutta
method with automatic (error based) time step size control.
This method is usually a good choice for prototyping, since we do not have to
worry about choosing a stable step size and can just run the simulation.
For better performance, it might be beneficial to tweak the tolerances
of this method or choose a different method that is more efficient for the
respective simulation.
You can find both approaches in our [example files](@ref examples).
Here, we just use the method with the default parameters, and only disable
`save_everystep` to avoid expensive saving of the solution in every time step.
```jldoctest tut_setup; output = false, filter = r".*"s
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks);

# output

```

See [Visualization](@ref) for how to visualize the solution.

## Replacing components with custom implementations

If we would like to use an implementation of a component that is not available
in TrixiParticles.jl, we can implement it ourselves within the simulation file,
without ever cloning the TrixiParticles.jl repository.
A good starting point is to check out the available implementations in
TrixiParticles.jl, then copy the relevant functions to the simulation file
and modify them as needed.

### Custom smoothing kernel

To implement a custom smoothing kernel, we define a struct extending
`TrixiParticles.SmoothingKernel`.
This abstract struct has a type parameter for the number of dimensions,
which we set to 2 in this case.

```jldoctest tut_setup; output = false
struct MyGaussianKernel <: TrixiParticles.SmoothingKernel{2} end

# output

```
This kernel is going to be an implementation of the Gaussian kernel with
a cutoff for compact support.
Note that the same kernel in a more optimized version is already implemented
in TrixiParticles.jl as [`GaussianKernel`](@ref).

In order to use our new kernel, we have to define three functions.
`TrixiParticles.kernel`, which is the kernel function itself,
`TrixiParticles.kernel_deriv`, which is the derivative of the kernel function,
and `TrixiParticles.compact_support`, which defines the compact support of the
kernel in relation to the smoothing length.
The latter is relevant for determining the search radius of the neighborhood search.

```jldoctest tut_setup; output = false
function TrixiParticles.kernel(kernel::MyGaussianKernel, r, h)
    q = r / h

    if q < 2
        return 1 / (pi * h^2) * exp(-q^2)
    end

    return 0.0
end

function TrixiParticles.kernel_deriv(kernel::MyGaussianKernel, r, h)
    q = r * h

    if q < 2
        return 1 / (pi * h^2)  * (-2 * q) * exp(-q^2) / h
    end

    return 0.0
end

TrixiParticles.compact_support(::MyGaussianKernel, h) = 3 * h

# output

```

This is all we need to use our custom kernel implementation in a simulation.
We only need to replace the definition above by
```jldoctest tut_setup; output = false
smoothing_kernel = MyGaussianKernel()

# output
MyGaussianKernel()
```
and run the simulation file again.

In order to use our kernel in a pre-defined example file, we can use the function
[`trixi_include`](@ref) to replace the definition of the variable `smoothing_kernel`.
The following will run the example simulation
`examples/fluid/hydrostatic_water_column_2d.jl` with our custom kernel.
```jldoctest tut_setup; output = false, filter = r".*"s
julia> trixi_include(joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
                     smoothing_kernel=MyGaussianKernel());
```
