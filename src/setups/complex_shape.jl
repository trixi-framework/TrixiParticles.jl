"""
    ComplexShape(geometry::Union{TriangleMesh, Polygon}; particle_spacing, density,
                 pressure=0.0, mass=nothing, velocity=zeros(ndims(geometry)),
                 point_in_geometry_algorithm=WindingNumberJacobson(; geometry,
                                                                hierarchical_winding=false,
                                                                winding_number_factor=sqrt(eps())),
                 grid_offset::Real=0.0, max_nparticles=10^7,
                 pad_initial_particle_grid=2particle_spacing)

Complex geometry filled with particles. Returns an [`InitialCondition`](@ref).
Note that an initial particle grid is generated inside the bounding box of the geometry.
An `point_in_geometry_algorithm` checks if particles are inside the geometry or not.
For more information about the method see [`WindingNumberJacobson`](@ref) or [`WindingNumberHorman`](@ref).

# Arguments
- `geometry`: Complex geometry returned by [`load_geometry`](@ref).

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
                        Cannot be used together with hydrostatic pressure gradient.
- `point_in_geometry_algorithm`: Algorithm for sampling the complex geometry with particles.
                              It basically checks whether a particle is inside an object or not.
                              For more information see [`WindingNumberJacobson`](@ref) or [`WindingNumberHorman`](@ref)
- `grid_offset`: Offset of the initial particle grid of the bounding box of the `geometry`.
- `max_nparticles`: Maximum number of particles in the initial particle grid.
- `pad_initial_particle_grid`: Padding of the initial particle grid.


!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
function ComplexShape(geometry::Union{TriangleMesh, Polygon}; particle_spacing, density,
                      pressure=0.0, mass=nothing, velocity=zeros(ndims(geometry)),
                      point_in_geometry_algorithm=WindingNumberJacobson(; geometry,
                                                                        hierarchical_winding=false,
                                                                        winding_number_factor=sqrt(eps())),
                      store_winding_number=false, grid_offset::Real=0.0,
                      max_nparticles=10^7, pad_initial_particle_grid=2particle_spacing)
    if ndims(geometry) == 3 && point_in_geometry_algorithm isa WindingNumberHorman
        throw(ArgumentError("`WindingNumberHorman` only supports 2D geometries"))
    end

    return sample(geometry; particle_spacing, density, pressure, mass, velocity,
                  point_in_geometry_algorithm, store_winding_number, grid_offset,
                  max_nparticles,
                  padding=pad_initial_particle_grid)
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
