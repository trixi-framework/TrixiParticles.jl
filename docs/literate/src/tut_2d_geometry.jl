# # [Setting up a 2D simulation from geometry files](@id tut_2d_geometry)

# In this tutorial, we build two genuine 2D setups from geometry files:
# 1. a curved pipe, where one geometry file defines the outer wall envelope and a second
#    one defines the empty channel cut out of it,
# 2. a dam-break basin with a coastline profile, where one geometry file defines the
#    filled coastline wall together with the seawall on the right.
#
# For a real 2D setup, we use 2D geometry formats such as `.asc` or `.dxf`.
# STL files are surface meshes and therefore naturally lead to thin 3D setups instead.

# First, we import TrixiParticles.jl together with
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl)
# and [Plots.jl](https://docs.juliaplots.org/stable/).
using TrixiParticles
using OrdinaryDiffEq
using Plots

# ## Resolution

# We use the same particle spacing for the fluid and for the wall geometries.
particle_spacing = 0.03
fluid_density = 1000.0
gravity = 9.81
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)
nothing # hide

# ## Loading 2D geometry files

# The following helper loads a closed 2D geometry file and samples particles in its interior:
# 1. load the polygon with [`load_geometry`](@ref),
# 2. fill the polygon with [`ComplexShape`](@ref).
#
# This creates a true 2D solid region instead of a hollow shell around the polygon edges.
function solid_from_geometry_file(file; particle_spacing, density)
    geometry = load_geometry(file)
    solid = ComplexShape(geometry; particle_spacing, density,
                         grid_offset=0.5particle_spacing)

    return (; geometry, solid)
end

# ## A curved pipe from two filled geometries

# The pipe wall is a solid L-shaped region with a channel cut out of it:
# 1. one geometry file describes the outer pipe envelope,
# 2. one geometry file describes the empty channel,
# 3. the `setdiff` operation subtracts the channel from the solid envelope.
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

# ## A dam-break basin with a coastline profile

# In the second setup, a single 2D geometry file defines a filled coastline wall:
# the beach profile on top, a finite wall thickness below it, and the seawall on the right.
coast_file = pkgdir(TrixiParticles, "examples", "preprocessing", "data",
                    "coastline_profile_2d.asc")
coast = solid_from_geometry_file(coast_file; particle_spacing, density=fluid_density)

# The geometry file gives the coastline bed and the right wall as a solid region.
# We add the left wall explicitly as a rectangular particle block and place a
# 1.5x taller rectangular dam-break water column next to it.
left_wall = RectangularShape(particle_spacing, (5, 50), (0.0, -0.12),
                             density=fluid_density)
reservoir = RectangularShape(particle_spacing, (28, 42), (0.15, 0.03),
                             acceleration=(0.0, -gravity),
                             state_equation=state_equation)
coast_setup = (; geometry=coast.geometry,
               wall=union(coast.solid, left_wall),
               fluid=setdiff(reservoir, coast.geometry))

p_pipe = plot(pipe_setup.wall, label="wall", title="Curved pipe",
              markerstrokewidth=0, markersize=4)
plot!(p_pipe, showaxis=false, aspect_ratio=:equal,
      xlims=(-0.03, 1.23), ylims=(-0.03, 1.23))

p_coast = plot(coast_setup.fluid, coast_setup.wall,
               labels=["fluid" "wall"], title="Coastline dam break",
               markerstrokewidth=0, markersize=3)
plot!(p_coast, showaxis=false, aspect_ratio=:equal,
      xlims=(0.0, 2.75), ylims=(-0.15, 1.35))

plot(p_pipe, p_coast, layout=(1, 2), size=(900, 360))
savefig("tut_2d_geometry_plot.png"); # hide
# ![2D geometry based initial conditions](tut_2d_geometry_plot.png)

# ## Building the simulation systems

# To keep the example focused, we continue with the coastline setup.
# From this point on, the simulation setup is the same as in other 2D simulation files.
setup = coast_setup
tspan = (0.0, 0.03)
nothing # hide

# We define the state equation, smoothing kernel, and viscosity for a
# weakly compressible SPH simulation.
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_density_calculator = ContinuityDensity()
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(setup.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity))
nothing # hide

# For the wall, we reuse the combined solid wall particles created above.
boundary_model = BoundaryModelDummyParticles(setup.wall.density, setup.wall.mass,
                                             state_equation=state_equation,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length)
boundary_system = WallBoundarySystem(setup.wall, boundary_model)
nothing # hide

# ## Semidiscretization

# With fluid and wall particles defined, we can build the
# [`Semidiscretization`](@ref TrixiParticles.Semidiscretization) exactly as in other tutorials.
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)
nothing # hide

# ## Time integration

# The setup is now complete.
# To start the simulation, run for example
# ```julia
# callbacks = CallbackSet(InfoCallback(interval=10))
# sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks)
# ```
# This is the same final step as in [the basic setup tutorial](@ref tut_setup).
callbacks = CallbackSet(InfoCallback(interval=10))
nothing # hide

sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks) #!md

# For more accurate body-fitted particles around sharper features, you can also
# apply the [particle packing workflow](@ref tut_packing) to the 2D geometry files
# before starting the simulation.
