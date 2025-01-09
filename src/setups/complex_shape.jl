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
function ComplexShape(geometry::Union{TriangleMesh, Polygon}; particle_spacing, density,
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

    return sample(geometry; particle_spacing, density, pressure, mass, velocity,
                  point_in_geometry_algorithm, store_winding_number, grid_offset,
                  max_nparticles, padding=pad_initial_particle_grid)
end

function sample(geometry; particle_spacing, density, pressure=0.0, mass=nothing,
                velocity=zeros(ndims(geometry)),
                point_in_geometry_algorithm=WindingNumberJacobson(; geometry,
                                                                  hierarchical_winding=false,
                                                                  winding_number_factor=sqrt(eps())),
                store_winding_number=false, grid_offset::Real=0.0, max_nparticles=10^7,
                padding=2particle_spacing)
    grid = particle_grid(geometry, particle_spacing; padding, grid_offset, max_nparticles)

    inpoly, winding_numbers = point_in_geometry_algorithm(geometry, grid;
                                                          store_winding_number)
    coordinates = grid[:, inpoly]

    ic = InitialCondition(; coordinates, density, mass, velocity, pressure,
                          particle_spacing)

    # This is most likely only useful for debugging. Note that this is not public API.
    if store_winding_number
        return (initial_condition=ic, winding_numbers=winding_numbers, grid=grid)
    end

    return ic
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

    return rectangular_shape_coords(particle_spacing, n_particles_per_dimension,
                                    min_corner; tlsph=true)
end

function Base.setdiff(initial_condition::InitialCondition,
                      geometries::Union{Polygon, TriangleMesh}...)
    geometry = first(geometries)

    if ndims(geometry) != ndims(initial_condition)
        throw(ArgumentError("all passed geometries must have the same dimensionality as the initial condition"))
    end

    delete_indices, _ = WindingNumberJacobson(; geometry)(geometry,
                                                          initial_condition.coordinates)

    coordinates = initial_condition.coordinates[:, .!delete_indices]
    velocity = initial_condition.velocity[:, .!delete_indices]
    mass = initial_condition.mass[.!delete_indices]
    density = initial_condition.density[.!delete_indices]
    pressure = initial_condition.pressure[.!delete_indices]

    result = InitialCondition{ndims(initial_condition)}(coordinates, velocity, mass,
                                                        density, pressure,
                                                        initial_condition.particle_spacing)

    return setdiff(result, Base.tail(geometries)...)
end

function Base.intersect(initial_condition::InitialCondition,
                        geometries::Union{Polygon, TriangleMesh}...)
    geometry = first(geometries)

    if ndims(geometry) != ndims(initial_condition)
        throw(ArgumentError("all passed geometries must have the same dimensionality as the initial condition"))
    end

    keep_indices, _ = WindingNumberJacobson(; geometry)(geometry,
                                                        initial_condition.coordinates)

    coordinates = initial_condition.coordinates[:, keep_indices]
    velocity = initial_condition.velocity[:, keep_indices]
    mass = initial_condition.mass[keep_indices]
    density = initial_condition.density[keep_indices]
    pressure = initial_condition.pressure[keep_indices]

    result = InitialCondition{ndims(initial_condition)}(coordinates, velocity, mass,
                                                        density, pressure,
                                                        initial_condition.particle_spacing)

    return intersect(result, Base.tail(geometries)...)
end
