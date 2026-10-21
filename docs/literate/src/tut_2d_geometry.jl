# # [Setting up a 2D simulation from geometry files](@id tut_2d_geometry)

# In this tutorial, we build two 2D setups from geometry files:
# 1. a curved pipe, where one geometry file defines the outer wall and another
#    defines the channel cut out of it,
# 2. a dam-break basin with a coastline profile, where one geometry file defines
#    the coastline wall and the seawall on the right.
#
# We use 2D geometry formats such as `.asc` or `.dxf` for 2D setups.
# STL files are supported only for 3D geometries.

# First, we import TrixiParticles.jl together with
# `OrdinaryDiffEqLowStorageRK` of
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
# and [Plots.jl](https://docs.juliaplots.org/stable/).
using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using Plots

# ## Building the geometry

# We use the same particle spacing for the fluid and for the wall geometries.
particle_spacing = 0.03
fluid_density = 1000.0
gravity = 9.81
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)
nothing # hide

# ### Loading the 2D geometry files

# The following helper loads a closed 2D geometry file and samples particles in its interior:
# 1. load the polygon with [`load_geometry`](@ref),
# 2. fill the polygon with [`ComplexShape`](@ref).
#
# This creates a filled 2D solid region rather than particles only along the polygon edges.
# The half-spacing `grid_offset` shifts the sampling grid in each coordinate direction
# to avoid placing particles directly on the grid-aligned edges of these geometries.
function solid_from_geometry_file(file; particle_spacing, density)
    geometry = load_geometry(file)
    solid = ComplexShape(geometry; particle_spacing, density,
                         grid_offset=0.5particle_spacing)

    return (; geometry, solid)
end
nothing # hide

# ### A curved pipe from two filled geometries

# The pipe wall is an L-shaped solid region with a channel cut out of it:
# 1. one geometry file describes the outer pipe envelope,
# 2. one geometry file describes the empty channel,
# 3. `setdiff` subtracts the channel from the solid envelope.
pipe_outer_file = pkgdir(TrixiParticles, "examples", "preprocessing", "data",
                         "curved_pipe_outer_2d.asc")
pipe_channel_file = pkgdir(TrixiParticles, "examples", "preprocessing", "data",
                           "curved_pipe_channel_2d.asc")

pipe_outer = solid_from_geometry_file(pipe_outer_file; particle_spacing,
                                      density=fluid_density)
pipe_channel = load_geometry(pipe_channel_file)

pipe_setup = (; wall=setdiff(pipe_outer.solid, pipe_channel),
              outer_geometry=pipe_outer.geometry,
              channel_geometry=pipe_channel)

p_pipe = plot(pipe_setup.wall, label="wall", size=(450, 450), showaxis=false,
              xlims=(-0.03, 1.23), ylims=(-0.03, 1.23))
plot!(p_pipe, dpi=300) # hide
savefig(p_pipe, "tut_2d_geometry_pipe.png"); # hide
# ![Curved pipe wall particles](tut_2d_geometry_pipe.png)

# ### A dam-break basin with a coastline profile

# In the second setup, one 2D geometry file defines the coastline wall:
# the beach profile, a finite wall thickness below it, and the seawall on the right.
coast_file = pkgdir(TrixiParticles, "examples", "preprocessing", "data",
                    "coastline_profile_2d.asc")
coast = solid_from_geometry_file(coast_file; particle_spacing, density=fluid_density)
nothing # hide

# The geometry file defines the coastline bed and right wall as a solid region.
# We add the left wall as a rectangular particle block and place a rectangular
# dam-break water column beside it.
left_wall = RectangularShape(particle_spacing, (5, 50), (0.0, -0.12),
                             density=fluid_density)
reservoir = RectangularShape(particle_spacing, (28, 42), (0.15, 0.03),
                             acceleration=(0.0, -gravity),
                             state_equation=state_equation)
coast_setup = (; geometry=coast.geometry,
               wall=union(coast.solid, left_wall),
               fluid=setdiff(reservoir, coast.geometry))

p_coast = plot(coast_setup.fluid, coast_setup.wall, labels=["fluid" "wall"],
               size=(700, 360), showaxis=false, xlims=(0.0, 2.75), ylims=(-0.15, 1.35))
plot!(p_coast, dpi=200) # hide
savefig(p_coast, "tut_2d_geometry_coast.png"); # hide
# ![Coastline dam-break initial conditions](tut_2d_geometry_coast.png)

# ## Running a coastline simulation

# We continue with the coastline setup using the systems, semidiscretization, and time
# integration described in [Setting up your simulation from scratch](@ref tut_setup).
setup = coast_setup
tspan = (0.0, 1.0)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_density_calculator = ContinuityDensity()
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(setup.fluid;
                                           density_calculator=fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity))

boundary_model = BoundaryModelDummyParticles(setup.wall.density, setup.wall.mass,
                                             state_equation=state_equation,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length)
boundary_system = WallBoundarySystem(setup.wall, boundary_model)

semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

callbacks = CallbackSet(InfoCallback(interval=10))

sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks)

plot(sol, title="Coastline dam break at t = $(sol.t[end])",
     size=(700, 360), showaxis=false, legend=:topleft)
plot!(dpi=200) # hide
savefig("tut_2d_geometry_solution.png"); # hide
# ![Coastline dam-break final result](tut_2d_geometry_solution.png)

# For more accurate body-fitted particles around sharper features, you can also
# apply the [particle packing workflow](@ref tut_packing) to the 2D geometry files
# before starting the simulation.
