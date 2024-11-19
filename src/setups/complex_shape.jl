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
struct ComplexShape{S, IC, ICB, SDF, IG, WN, ELTYPE}
    geometry                   :: S
    initial_condition          :: IC
    initial_condition_boundary :: ICB
    signed_distance_field      :: SDF
    initial_grid               :: IG
    winding_number             :: WN
    particle_spacing           :: ELTYPE
end

function ComplexShape(geometry; particle_spacing, density,
                      pressure=0.0, mass=nothing, velocity=zeros(ndims(geometry)),
                      sample_boundary=false, boundary_thickness=6particle_spacing,
                      create_signed_distance_field=false, tlsph=true,
                      point_in_geometry_algorithm=WindingNumberJacobson(; geometry,
                                                                        hierarchical_winding=false,
                                                                        winding_number_factor=sqrt(eps())),
                      store_winding_number=false, grid_offset::Real=0.0,
                      max_nparticles=10^7)
    if ndims(geometry) == 3 && point_in_geometry_algorithm isa WindingNumberHormann
        throw(ArgumentError("`WindingNumberHormann` only supports 2D geometries"))
    end

    if grid_offset < 0.0
        throw(ArgumentError("only a positive `grid_offset` is supported"))
    end

    padding = sample_boundary ? 2boundary_thickness : 2particle_spacing

    grid = particle_grid(geometry, particle_spacing; padding, grid_offset, max_nparticles)

    inpoly, winding_numbers = point_in_geometry_algorithm(geometry, grid;
                                                          store_winding_number)

    coordinates = stack(grid[inpoly])

    initial_condition = InitialCondition(; coordinates, density, mass, velocity, pressure,
                                         particle_spacing)

    if create_signed_distance_field
        signed_distance_field = SignedDistanceField(geometry, particle_spacing;
                                                    point_grid=grid,
                                                    max_signed_distance=boundary_thickness,
                                                    use_for_boundary_packing=sample_boundary)
    else
        signed_distance_field = nothing
    end

    if sample_boundary
        # Use the particles outside the object as boundary particles.
        (; positions, distances, max_signed_distance) = signed_distance_field

        # Delete unnecessary large signed distance field
        distance_to_boundary = tlsph ? particle_spacing : 0.5 * particle_spacing
        keep_indices = (distance_to_boundary .< distances .<= max_signed_distance)

        boundary_coordinates = stack(positions[keep_indices])
        initial_condition_boundary = InitialCondition(; coordinates=boundary_coordinates,
                                                      density, particle_spacing)
    else
        initial_condition_boundary = nothing
    end

    # This is most likely only useful for debugging. Note that this is not public API.
    if store_winding_number
        initial_grid = stack(grid)
    else
        initial_grid = nothing
    end

    IC = typeof(initial_condition)
    ICB = typeof(initial_condition_boundary)
    IG = typeof(initial_grid)
    WN = typeof(winding_numbers)
    SDF = typeof(signed_distance_field)
    S = typeof(geometry)
    ELTYPE = eltype(initial_condition)

    return ComplexShape{S, IC, ICB, SDF, IG, WN, ELTYPE}(geometry, initial_condition,
                                                         initial_condition_boundary,
                                                         signed_distance_field,
                                                         initial_grid, winding_numbers,
                                                         particle_spacing)
end

Base.ndims(cs::ComplexShape) = ndims(cs.initial_condition)
Base.eltype(cs::ComplexShape) = eltype(cs.initial_condition)

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
                                    min_corner; tlsph=true)
    return reinterpret(reshape, SVector{ndims(geometry), eltype(geometry)}, grid)
end
