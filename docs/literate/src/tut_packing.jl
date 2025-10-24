# # Particle Packing Tutorial

# In this tutorial, we will guide you through the complete particle packing pipeline.
# The algorithmic background is explained in [Particle Packing](@ref particle_packing).
# Throughout this tutorial, we will refer to the initially sampled particles as the "initial configuration",
# and the configuration after packing as the "packed configuration".
# The particles, created by an inside–outside segmentation of the geometry,
# are referred to as "interior particles", whereas the particles on the surface of the geometry
# are called "boundary particles".

# ## Loading the geometry

# As a first step, we will load a geometry.
# Supported file formats are described [in the documentation](@ref read_geometries_from_file).
using TrixiParticles
using Plots

file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "potato.asc")
geometry = load_geometry(file)

# To get an overview, we can visualize this geometry.
plot(geometry, showaxis=false, label=nothing, color=:black)

# As shown in the plot, the 2D geometry (`TrixiParticles.Polygon`) is represented by its edges.
# Similarly, 3D geometries (`TrixiParticles.TriangleMesh`), are represented by triangles.
# In this tutorial, we will stay with a 2D geometry, but the steps are identical for 3D geometries.

# ## Creating a signed distance field

# For the actual packing process, this geometry representation is not used directly.
# What we need is only the distance to the surface of the geometry.
# To obtain this, we create a signed distance field (SDF).
# The SDF is constructed within a band around the geometry's surface.
# The "thickness" of this band is controlled by `max_signed_distance`.
# For example, a `max_signed_distance` of 0.1 means that the SDF is created
# up to 0.1 units inside and 0.1 units outside of the geometry interface.
# It is a good practice to choose `max_signed_distance` equal to or larger than
# the compact support of the [smoothing kernel](@ref smoothing_kernel) that will be used in the later simulation.
# This ensures that particles near the geometry interface have full kernel support,
# maintaining accurate interpolation during the packing process.
# The resolution of the SDF is defined by `particle_spacing`, which should ideally
# be identical to the `particle_spacing` of the initial configuration.
particle_spacing = 0.05
boundary_thickness = 3 * particle_spacing

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            max_signed_distance=boundary_thickness)

# We can also visualize the SDF by simply creating an `InitialCondition` from the sampled points and plot it.
# The color coding represents the signed distance to the geometry surface.
sdf_ic = InitialCondition(; coordinates=stack(signed_distance_field.positions),
                          density=1.0, particle_spacing=particle_spacing)

plot(sdf_ic, zcolor=signed_distance_field.distances, label=nothing, color=:coolwarm)
plot!(geometry, linestyle=:dash, label=nothing, showaxis=false, color=:black,
      seriestype=:path, linewidth=2)
plot!(right_margin=5Plots.mm) #hide

# Since we will later also pack boundary particles, we need to extend the SDF to the outside.
# For that, we set `use_for_boundary_packing=true`.
signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

# We can see in the plot that the SDF has been extended outwards to twice `max_signed_distance`.
sdf_ic = InitialCondition(; coordinates=stack(signed_distance_field.positions),
                          density=1.0, particle_spacing=particle_spacing)

plot(sdf_ic, zcolor=signed_distance_field.distances, label=nothing, color=:coolwarm)

plot!(geometry, linestyle=:dash, label=nothing, showaxis=false, color=:black,
      seriestype=:path, linewidth=2)
plot!(right_margin=5Plots.mm) #hide

# ## Creating an initial configuration of boundary particles

# To create the initial configuration of the boundary particles,
# we use the sampled points of the SDF whose signed distance lies between 0
# and `boundary_thickness`.
# Here, we need to specify the `density` of the boundary particles.
# As an example, we choose `1.0` for all particles.
# This gives us an [`InitialCondition`](@ref InitialCondition) for the boundary particles.
density = 1.0
boundary_sampled = sample_boundary(signed_distance_field; boundary_density=density,
                                   boundary_thickness)

## Plotting the initial configuration of the boundary particles
plot(boundary_sampled, label=nothing)
plot!(geometry, linestyle=:dash, label=nothing, showaxis=false, color=:black,
      seriestype=:path, linewidth=2)

# ## Creating an initial configuration of interior particles

# Next, we need to create the initial configuration of the interior particles.
# This step is independent of the other steps in the packing pipeline.
# For the later packing, any [`InitialCondition`](@ref InitialCondition) can be used.
# Later more on this.
# Different inside–outside segmentation algorithms can be applied here.
# In this tutorial, we will use a winding number approach.
# See also [Sampling of Geometries](@ref sampling_of_geometries) for details.
# We initialize the winding number algorithm with the geometry.
point_in_geometry_algorithm = WindingNumberJacobson(; geometry)

# The inside–outside segmentation of the geometry is performed by the `ComplexShape` function.
# This function creates an [`InitialCondition`](@ref InitialCondition) for the interior particles.
# Here, we need to specify `particle_spacing` and `density`. For further arguments, please refer to
# the documentation of [ComplexShape](@ref ComplexShape) or [InitialCondition](@ref InitialCondition).
shape_sampled = ComplexShape(geometry; particle_spacing, density=density,
                             point_in_geometry_algorithm)

# If we want to assign the mass of each sampled particle consistently with its density,
# we can adjust it as follows:
shape_sampled.mass .= density * TrixiParticles.volume(geometry) / nparticles(shape_sampled);

# Now we can plot the initial configuration of the interior particles
# together with the boundary particles.
plot(shape_sampled, boundary_sampled, label=nothing)
plot!(geometry, linestyle=:dash, label=nothing, showaxis=false, color=:black,
      seriestype=:path, linewidth=2)

# As shown in the plot, the interface of the geometry surface is not well resolved yet.
# In other words, there is no body-fitted configuration.
# This is where the particle packing will come into play.

# ## Particle packing

# In the following, we will essentially follow the same steps described in the fluid tutorials.
# That means we will generate systems that are then passed to the [`Semidiscretization`](@ref Semidiscretization).
# The difference from a typical physical simulation is that we use [`ParticlePackingSystem`](@ref ParticlePackingSystem),
# which does not represent any physical law. Instead, we only use the simulation framework to time-integrate
# the packing process.

# We first need to import [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).
using OrdinaryDiffEq

# Next, we set a background pressure. This can be chosen arbitrarily.
# A higher value results in smaller time steps, but the final packed state
# will remain the same after running the same number of iterations.
background_pressure = 1.0

# For particle interaction, we select a [smoothing kernel](@ref smoothing_kernel)
# with a suitable smoothing length. Empirically, a factor of `0.8` times the
# particle spacing gives good results [Neher2026](@cite).
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()
smoothing_length = 0.8 * particle_spacing

# Now we can create the packing system. For learning purposes, let’s first try
# passing no signed distance field (SDF) and see what happens.
packing_system = ParticlePackingSystem(shape_sampled;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=nothing, background_pressure)

# We now proceed with the familiar steps
# "Semidiscretization" and "Time integration" from the fluid tutorials.
semi = Semidiscretization(packing_system)

## Use a high `tspan` to guarantee that the simulation runs for at least `maxiters`
tspan = (0, 10000.0)
ode = semidiscretize(semi, tspan)

maxiters = 100
callbacks = CallbackSet(UpdateCallback())
time_integrator = RDPK3SpFSAL35()

sol = solve(ode, time_integrator;
            abstol=1e-7, reltol=1e-4, save_everystep=false, maxiters=maxiters,
            callback=callbacks)

packed_ic = InitialCondition(sol, packing_system, semi)

plot(packed_ic)
plot!(geometry, seriestype=:path, linewidth=2, color=:black, label=nothing)

# As we can see in the plot, the particles are not constrained to the
# geometric surface.

# We therefore add an SDF for the geometry and repeat the same procedure.
packing_system = ParticlePackingSystem(shape_sampled;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field,
                                       background_pressure)

# Again, we follow the same steps for semidiscretization and time integration.
semi = Semidiscretization(packing_system)

tspan = (0, 10000.0)
ode = semidiscretize(semi, tspan)

maxiters = 1000
callbacks = CallbackSet(UpdateCallback())
time_integrator = RDPK3SpFSAL35()

sol = solve(ode, time_integrator;
            abstol=1e-7, reltol=1e-4, save_everystep=false, maxiters=maxiters,
            callback=callbacks)

packed_ic = InitialCondition(sol, packing_system, semi)

## Plotting the final configuration
plot(packed_ic)
plot!(geometry, seriestype=:path, color=:black, label=nothing, linewidth=2)

# We can see that the particles now stay inside the geometry,
# but their distribution near the surface can still be improved by adding boundary particles [Neher2026](@cite).
# Therefore, we set up a dedicated boundary packing system
# by setting `is_boundary = true`.
# For convex geometries, it is useful to slightly compress the
# boundary layer thickness. The background for this is that the true
# geometric volume is often larger than what was assumed when sampling
# the boundary particles, because the mass was computed assuming an
# interior particle volume.
# A `boundary_compress_factor` of `0.8` or `0.9` works well for most shapes.
# Since we have a relatively large particle spacing compared to the
# geometry size in this example, we will choose `0.7`.
boundary_system = ParticlePackingSystem(boundary_sampled;
                                        is_boundary=true,
                                        smoothing_kernel=smoothing_kernel,
                                        smoothing_length=smoothing_length,
                                        boundary_compress_factor=0.7,
                                        signed_distance_field, background_pressure)

# We can now couple the boundary system with the interior system:
semi = Semidiscretization(packing_system, boundary_system)

tspan = (0, 10000.0)
ode = semidiscretize(semi, tspan)

maxiters = 1000
callbacks = CallbackSet(UpdateCallback())
time_integrator = RDPK3SpFSAL35()

sol = solve(ode, time_integrator;
            abstol=1e-7, reltol=1e-4, save_everystep=false, maxiters=maxiters,
            callback=callbacks)

packed_ic = InitialCondition(sol, packing_system, semi)
packed_boundary_ic = InitialCondition(sol, boundary_system, semi)

## Plotting the final configuration
plot(packed_ic, packed_boundary_ic)
plot!(geometry, seriestype=:path, color=:black, linestyle=:dash, linewidth=2, label=nothing)

# ## Multi-body packing

# So far, we have focused on packing a single body.
# However, it is often useful to also pack a surrounding (complex) body in a way
# that the interface between both domains is well represented.
# We will demonstrate this using a simple rectangular domain:
# first we will pack the entire domain, then only a selected (necessary) part of it.

# We will reuse the final configuration of our geometry to create a packing system.
# Because we are satisfied with this particle distribution, we want it to remain unchanged.
# Therefore, we set `fixed_system=true` so that this system is not integrated further
# but instead serves as a static boundary for the packing of other domains.
fixed_system = ParticlePackingSystem(packed_ic;
                                     smoothing_kernel=smoothing_kernel,
                                     smoothing_length=smoothing_length,
                                     signed_distance_field=nothing,
                                     background_pressure,
                                     fixed_system=true)

# Now we define a rectangular domain that we want to pack.
# In practice, you could create any `InitialCondition` that encloses your complex geometry.
tank_domain = RectangularTank(particle_spacing, (4, 4), (0, 0), min_coordinates=(-1, -2),
                              density)

sampled_outer_domain = setdiff(tank_domain.fluid, packed_ic)

# If we plot these two `InitialCondition`s, we can see
# that the geometry interface is not properly represented yet.
plot(sampled_outer_domain, packed_ic)

# Next, we create a packing system for the outer domain.
packing_system = ParticlePackingSystem(sampled_outer_domain;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=nothing,
                                       background_pressure)

# Since we do not want to sample a boundary for the outer domain,
# we can set up a periodic box to ensure that all outer particles have full kernel support.
periodic_box = PeriodicBox(min_corner=[-1.0, -2.0], max_corner=[3.0, 2.0])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box)

semi = Semidiscretization(packing_system, fixed_system; neighborhood_search)

tspan = (0, 10000.0)
ode = semidiscretize(semi, tspan)

maxiters = 1000
callbacks = CallbackSet(UpdateCallback())
time_integrator = RDPK3SpFSAL35()

sol_1 = solve(ode, time_integrator; abstol=1e-7, reltol=1e-4,
              save_everystep=false, maxiters=maxiters, callback=callbacks)

packed_outer_domain = InitialCondition(sol_1, packing_system, semi)

# We see that after packing, the geometry interface is well represented.
plot(packed_outer_domain, packed_ic)

# However, from the plot we can also see that the particles
# near the interface were primarily affected by the packing algorithm,
# while the rest of the domain remains almost unchanged.
# It is often more efficient to pack only a local region instead of the full domain.
# Therefore, we define a rectangle that encloses the complex geometry:
pack_window = TrixiParticles.Polygon(stack([
                                               [-0.5, -1.5],
                                               [2.5, -1.5],
                                               [2.5, 0.5],
                                               [-0.5, 0.5],
                                               [-0.5, -1.5]
                                           ]))

# Then, we extract the particles that fall inside this window
pack_domain = intersect(sampled_outer_domain, pack_window)
# and those outside the window
fixed_domain = setdiff(sampled_outer_domain, pack_window)

plot(pack_domain, fixed_domain, packed_ic)

# We can now treat the particles outside the window, along with the already
# finalized configuration of the complex geometry, as fixed systems:
fixed_system_1 = ParticlePackingSystem(fixed_domain;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=nothing,
                                       background_pressure, fixed_system=true)

fixed_system_2 = ParticlePackingSystem(packed_ic;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=nothing,
                                       background_pressure, fixed_system=true)

# The window that we want to pack is passed to a moving packing system:
packing_system = ParticlePackingSystem(pack_domain;
                                       smoothing_kernel=smoothing_kernel,
                                       smoothing_length=smoothing_length,
                                       signed_distance_field=nothing,
                                       background_pressure)

semi = Semidiscretization(packing_system, fixed_system_1, fixed_system_2)

tspan = (0, 10000.0)
ode = semidiscretize(semi, tspan)

maxiters = 1000
callbacks = CallbackSet(UpdateCallback())
time_integrator = RDPK3SpFSAL35()

sol_2 = solve(ode, time_integrator; abstol=1e-7, reltol=1e-4,
              save_everystep=false, maxiters=maxiters, callback=callbacks)

packed_fluid_domain = InitialCondition(sol_2, packing_system, semi)

# We see that this still gives us a good result
plot(fixed_domain, packed_fluid_domain, packed_ic)

# Finally, we can show the size of our integration arrays:
@show length(sol_1.u[1].x[1]);
@show length(sol_2.u[1].x[1]);

# This shows that we can avoid integrating unnecessary particles by restricting
# the packing to a relevant window.
