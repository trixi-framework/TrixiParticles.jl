# In this tutorial, we will guide you through the complete pipeline of particle packing.
# The algorithmic background is explained in [Particle Packing](@ref particle_packing).
# Throughout this tutorial, we will refer to the initially sampled particles as the "initial configuration",
# and the configuration after packing as the "packed configuration".
# The interior particles, created by an inside–outside segmentation of the geometry,
# are referred to as "interior particles", while the particles on the surface of the geometry
# are called "boundary particles".

# # Load geometry

# As a first step, we will load a geometry.
# Supported file formats are described in [Read geometries from file](@ref read_geometries_from_file).
using TrixiParticles
using Plots

file = pkgdir(TrixiParticles, "examples", "preprocessing", "data", "potato.asc")
geometry = load_geometry(file)

# To get an overview, we can visualize this geometry.
my_xlims = (-0.3, 2.3)
my_ylims = (-1.3, 0.3)

p1 = plot(Plots.Shape, geometry, axis=false, label=nothing, xlims=my_xlims, ylims=my_ylims,
          markersize=5)

# As shown in the plot, the 2D geometry is represented by its edges.
# Similarly, for 3D geometries, the boundaries are represented by triangles.
# In this tutorial, we will stay with a 2D geometry, but the steps are identical for 3D geometries.
# In 2D, the geometry is represented by the `TrixiParticles.Polygon` type,
# and in 3D by the `TrixiParticles.TriangleMesh` type.

# # Create signed distance field

# For the actual packing process, this geometry representation is not used directly.
# What we need is only the information about the surface of the geometry.
# To obtain this, we create a signed distance field (SDF).
# The SDF is constructed within a band around the geometry surface.
# The "thickness" of this band is controlled by `max_signed_distance`.
# For example, a `max_signed_distance` of 0.1 means that the SDF is created
# up to 0.1 units inside and 0.1 units outside of the geometry interface.
# It is a good practice to choose `max_signed_distance` on the same order of magnitude
# as the compact support of the [smoothing kernel](@ref smoothing_kernel) that will be used in the later simulation.
# The resolution of the SDF is defined by `particle_spacing`, which should ideally
# be identical to the `particle_spacing` of the initial configuration.
particle_spacing = 0.05

boundary_thickness = 3 * particle_spacing

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            max_signed_distance=boundary_thickness)

# We can also visualize the SDF by simply creating an `InitialCondition` from the sampled points and plot it.
# The color coding represents the signed distance to the geometry surface.
sdf_1 = InitialCondition(; coordinates=stack(signed_distance_field.positions),
                         density=1.0, particle_spacing=particle_spacing)

p2 = plot(sdf_1, zcolor=signed_distance_field.distances, label=nothing,
          xlims=my_xlims, ylims=my_ylims, color=:coolwarm)
plot!(p2, Plots.Shape, geometry, linestyle=:dash, label=nothing, axis=false, grid=false,
      xlims=my_xlims, ylims=my_ylims)

# Since we will later also pack boundary particles, we need to extend the SDF to the outside.
# For that, we set `use_for_boundary_packing=true`.
signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

# We can see in the plot that the SDF has been extended to twice the distance.
sdf_2 = InitialCondition(; coordinates=stack(signed_distance_field.positions),
                         density=1.0, particle_spacing=particle_spacing)

p3 = plot(sdf_2, zcolor=signed_distance_field.distances, label=nothing,
          xlims=my_xlims, ylims=my_ylims, color=:coolwarm)

plot!(p3, Plots.Shape, geometry, linestyle=:dash, label=nothing, axis=false, grid=false,
      xlims=my_xlims, ylims=my_ylims)

# # Create initial configuration of boundary particles

# To create the initial configuration of the boundary particles,
# we use the sampled points of the SDF whose signed distance lies between 0
# and `boundary_thickness`.
# Here, we need to specify the `density` of the boundary particles.
# As an example, we choose `1.0` for all particles.
# This gives us an [`InitialCondition`](@ref InitialCondition) for the boundary particles.
density = 1.0
boundary_sampled = sample_boundary(signed_distance_field; boundary_density=density,
                                   boundary_thickness)
p4 = plot(boundary_sampled, xlims=my_xlims, ylims=my_ylims, label=nothing)
plot!(p4, Plots.Shape, geometry, linestyle=:dash, label=nothing, axis=false, grid=false,
      xlims=my_xlims, ylims=my_ylims)

# # Create initial configuration of interior particles

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

# Now we can plot the initial configuration of the interior particles
# together with the boundary particles.
p5 = plot(shape_sampled, boundary_sampled, xlims=my_xlims, ylims=my_ylims, label=nothing)
plot!(p5, Plots.Shape, geometry, linestyle=:dash, label=nothing, axis=false, grid=false,
      xlims=my_xlims, ylims=my_ylims)

# As shown in the plot, the interface of the geometry surface is not well resolved yet.
# In other words, there is no body-fitted configuration.
# This is where the particle packing will come into play.
