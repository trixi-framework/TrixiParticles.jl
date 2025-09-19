"""
    ComplexShape(geometry::Union{TriangleMesh, Polygon}; particle_spacing, density,
                 pressure=0.0, mass=nothing, velocity=zeros(ndims(geometry)),
                 point_in_geometry_algorithm=WindingNumberJacobson(; geometry,
                                                                   hierarchical_winding=false,
                                                                   winding_number_factor=sqrt(eps())),
                 grid_offset::Real=0.0, max_nparticles=10^7,
                 pad_initial_particle_grid=2particle_spacing)

Sample a complex geometry with particles. Returns an [`InitialCondition`](@ref).
Note that an initial particle grid is generated inside the bounding box of the geometry.
A `point_in_geometry_algorithm` checks if particles are inside the geometry or not.
For more information about the method see [`WindingNumberJacobson`](@ref) or [`WindingNumberHormann`](@ref).

# Arguments
- `geometry`: Geometry returned by [`load_geometry`](@ref).

# Keywords
- `particle_spacing`:   Spacing between the particles.
- `density`:            Either a function mapping each particle's coordinates to its density,
                        or a scalar for a constant density over all particles.
- `velocity`:           Either a function mapping each particle's coordinates to its velocity,
                        or, for a constant fluid velocity, a vector holding this velocity.
                        Velocity is constant zero by default.
- `mass`:               Either `nothing` (default) to automatically compute particle mass from particle
                        density and spacing, or a function mapping each particle's coordinates to its mass,
                        or a scalar for a constant mass over all particles.
- `pressure`:           Scalar to set the pressure of all particles to this value.
                        This is only used by the [`EntropicallyDampedSPHSystem`](@ref) and
                        will be overwritten when using an initial pressure function in the system.
- `point_in_geometry_algorithm`: Algorithm for sampling the complex geometry with particles.
                                 It basically checks whether a particle is inside an object or not.
                                 For more information see [`WindingNumberJacobson`](@ref) or [`WindingNumberHormann`](@ref)
- `grid_offset`: Offset of the initial particle grid of the bounding box of the `geometry`.
- `max_nparticles`: Maximum number of particles in the initial particle grid.
                    This is only used to avoid accidentally choosing a `particle_spacing`
                    that is too small for the scale of the geometry.
- `pad_initial_particle_grid`: Padding of the initial particle grid.


!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
function ComplexShape(geometry; particle_spacing, density,
                      pressure=0.0, mass=nothing, velocity=zeros(ndims(geometry)),
                      point_in_geometry_algorithm=WindingNumberJacobson(; geometry,
                                                                        hierarchical_winding=true,
                                                                        winding_number_factor=sqrt(eps())),
                      store_winding_number=false, grid_offset::Real=0.0,
                      max_nparticles=10^7, pad_initial_particle_grid=2particle_spacing)
    if ndims(geometry) == 3 && point_in_geometry_algorithm isa WindingNumberHormann
        throw(ArgumentError("`WindingNumberHormann` only supports 2D geometries"))
    end

    if grid_offset < 0.0
        throw(ArgumentError("only a positive `grid_offset` is supported"))
    end

    grid = particle_grid(geometry, particle_spacing; padding=pad_initial_particle_grid,
                         grid_offset, max_nparticles)

    inpoly,
    winding_numbers = point_in_geometry_algorithm(geometry, grid;
                                                  store_winding_number)

    coordinates = stack(grid[inpoly])

    initial_condition = InitialCondition(; coordinates, density, mass, velocity, pressure,
                                         particle_spacing)

    # This is most likely only useful for debugging. Note that this is not public API.
    if store_winding_number
        return (; initial_condition=initial_condition, winding_numbers=winding_numbers,
                grid=grid)
    end

    return initial_condition
end

"""
    sample_boundary(signed_distance_field;
                    boundary_density, boundary_thickness, place_on_shell=true)

Sample boundary particles of a complex geometry by using the [`SignedDistanceField`](@ref)
of the geometry.

# Arguments
- `signed_distance_field`: The signed distance field of a geometry (see [`SignedDistanceField`](@ref)).

# Keywords
- `boundary_thickness`: Thickness of the boundary
- `boundary_density`: Density of each boundary particle.
- `place_on_shell`:     When `place_on_shell=true`, boundary particles will be placed
                        one particle spacing from the surface of the geometry.
                        Otherwise when `place_on_shell=true` (simulating fluid particles),
                        boundary particles will be placed half particle spacing away from the surface.


# Examples
```jldoctest; output = false, setup = :(particle_spacing = 0.03; boundary_thickness = 4 * particle_spacing; file = joinpath(pkgdir(TrixiParticles, "examples", "preprocessing", "data"), "circle.asc"))
geometry = load_geometry(file)

signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                            use_for_boundary_packing=true,
                                            max_signed_distance=boundary_thickness)

boundary_sampled = sample_boundary(signed_distance_field; boundary_density=1.0,
                                   boundary_thickness)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ InitialCondition{Float64}                                                                        │
│ ═════════════════════════                                                                        │
│ #dimensions: ……………………………………………… 2                                                                │
│ #particles: ………………………………………………… 889                                                              │
│ particle spacing: ………………………………… 0.03                                                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
function sample_boundary(signed_distance_field;
                         boundary_density, boundary_thickness, place_on_shell=true)
    (; max_signed_distance, boundary_packing,
     positions, distances, particle_spacing) = signed_distance_field

    if !(boundary_packing)
        throw(ArgumentError("`SignedDistanceField` was not generated with `use_for_boundary_packing`"))
    end

    if boundary_thickness > max_signed_distance
        throw(ArgumentError("`boundary_thickness` is greater than `max_signed_distance` of `SignedDistanceField`. " *
                            "Please generate a `SignedDistanceField` with higher `max_signed_distance`."))
    end

    # Only keep the required part of the signed distance field
    distance_to_boundary = zero(particle_spacing)
    keep_indices = (distance_to_boundary .< distances .<= max_signed_distance)

    boundary_coordinates = stack(positions[keep_indices])
    return InitialCondition(; coordinates=boundary_coordinates, density=boundary_density,
                            particle_spacing)
end

function particle_grid(geometry, particle_spacing;
                       padding=2particle_spacing, grid_offset=0.0, max_nparticles=10^7)
    (; max_corner) = geometry

    min_corner = geometry.min_corner .- grid_offset .- padding

    n_particles_per_dimension = Tuple(ceil.(Int,
                                            (max_corner .- min_corner .+ 2padding) ./
                                            particle_spacing))

    n_particles = prod(n_particles_per_dimension)

    if n_particles > max_nparticles
        throw(ArgumentError("Number of particles of the initial grid ($n_particles) exceeds " *
                            "`max_nparticles` = $max_nparticles"))
    end

    grid = rectangular_shape_coords(particle_spacing, n_particles_per_dimension,
                                    min_corner; place_on_shell=true)
    return reinterpret(reshape, SVector{ndims(geometry), eltype(geometry)}, grid)
end
