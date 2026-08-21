# # [Visualizing particle data with Plots.jl](@id tut_visualization)

# In this tutorial, we run the two-dimensional vortex street from
# [`examples/fluid/vortex_street_2d.jl`](https://github.com/trixi-framework/TrixiParticles.jl/blob/main/examples/fluid/vortex_street_2d.jl)
# and visualize the particle data with [`Plots.jl`](https://github.com/juliaplots/plots.jl).

using TrixiParticles
using Plots
#src # Reset GR's process-wide color table, which can be exhausted by earlier tutorials.
Plots.closeall() # hide

# The example defines the particle spacing as `particle_spacing_factor * cylinder_diameter`.
# We deliberately use a very coarse particle resolution. This makes the distinction between
# the discrete particles and the interpolated field in the next section clear.
# To remove visual clutter, we disable the info callback.
# Since we visualize with Plots.jl, we also disable the saving callback.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "vortex_street_2d.jl");
              particle_spacing_factor=0.2,
              info_callback=nothing, saving_callback=nothing);
nothing # hide

# ## Visualizing discrete particles

# SPH stores the solution on moving particles. The standard plotting recipe provides the
# quickest way to inspect their distribution at the final time. We color the fluid particles
# by the magnitude of the velocity stored on each particle.
v_ode, _ = sol.u[end].x
v_fluid = TrixiParticles.wrap_v(v_ode, fluid_system, semi)

active_particles = TrixiParticles.eachparticle(fluid_system)
particle_velocity = TrixiParticles.current_velocity(v_fluid,
                                                    fluid_system)[:, active_particles]
particle_velocity_magnitude = vec(sqrt.(sum(abs2, particle_velocity; dims=1)))

particle_plot = plot(fluid_system, sol; zcolor=particle_velocity_magnitude, color=:viridis,
                     xlims=(0.25, 1.8), ylims=(0.1, 0.9), legend=false,
                     xlabel="x", ylabel="y", colorbar=true, colorbar_title="|v|",
                     size=(900, 450))
plot!(particle_plot; dpi=200) # hide
savefig(particle_plot, "tut_visualization_particles.png") # hide
nothing # hide

# ![Particle visualization of the vortex street](tut_visualization_particles.png)

# ## Interpolating particle data onto a regular grid

# Smoothed particle hydrodynamics (SPH) represents a continuous (smoothed) field
# by a discrete set of particles. While visualizing individual particles is straightforward
# and often sufficient, in order to visualize the actual field approximation,
# the particle data must be interpolated.
#
# Importantly, interpolation does not add physical resolution: features that are not resolved
# by the particles cannot be recovered by choosing a finer interpolation grid.
# It simply visualizes the SPH approximation instead of only the interpolation points.

# [`interpolate_plane_2d`](@ref) constructs regularly spaced sample points between two corners
# and uses the SPH kernel to reconstruct the requested fields there. The interpolation spacing
# is one quarter of the particle spacing, so the plot contains many more pixels than
# the simulation contains particles.
interpolation_min = [0.0, 0.0]
interpolation_max = domain_size
interpolation_spacing = particle_spacing / 4

interpolated = interpolate_plane_2d(interpolation_min, interpolation_max,
                                    interpolation_spacing, semi, fluid_system, sol)
interpolated_velocity_magnitude = vec(sqrt.(sum(abs2, interpolated.velocity; dims=1)))
nothing # hide

# The returned named tuple also contains `pressure`, `density`, `neighbor_count`, and
# `computed_density`. Here we visualize the magnitude of the interpolated velocity.
interpolated_plot = scatter(interpolated.point_coords[1, :],
                            interpolated.point_coords[2, :];
                            marker_z=interpolated_velocity_magnitude,
                            color=:viridis,
                            marker=:square, markerstrokewidth=0, markersize=2.5,
                            aspect_ratio=:equal, size=(900, 450),
                            xlims=(0.25, 1.8), ylims=(0.1, 0.9), xlabel="x", ylabel="y",
                            label=nothing, colorbar_title="|v|")
plot!(interpolated_plot; dpi=200) # hide
savefig(interpolated_plot, "tut_visualization_interpolated_velocity.png") # hide
nothing # hide

# ![Interpolated velocity magnitude](tut_visualization_interpolated_velocity.png)

# Compared with the visibly discrete particle distribution, the interpolated field shows
# much more detail, representing the continuous SPH approximation of the solution.

# To write the same reconstruction as a VTI image for ParaView, replace the interpolation call
# above with [`interpolate_plane_2d_vtk`](@ref):
interpolate_plane_2d_vtk(interpolation_min, interpolation_max, interpolation_spacing,
                         semi, fluid_system, sol; filename="vortex_street_velocity")
nothing # hide
