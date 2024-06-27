"""
    ComplexShape(shape::Shapes; particle_spacing, density, pressure=0.0, mass=nothing,
                 velocity=zeros(ndims(shape)),
                 point_in_shape_algorithm=WindingNumberJacobson(; shape,
                                                                hierarchical_winding=false,
                                                                winding_number_factor=sqrt(eps())),
                 seed=shape.min_corner, max_nparticles=Int(1e7),
                 pad_initial_particle_grid=2particle_spacing)

Complex shape filled with particles. Returns an [`InitialCondition`](@ref).
Note that an initial particle grid is generated inside the bounding box of the shape.
An `point_in_shape_algorithm` checks if particles are inside the shape or not.
For more information about the method see [`WindingNumberJacobson`](@ref) or [`WindingNumberHorman`](@ref).

# Arguments
- `shape`: Complex shape returned by [`load_shape`](@ref).

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
- `point_in_shape_algorithm`: Algorithm for sampling the complex shape with particles.
                              It basically checks whether a particle is inside an object or not.
                              For more information see [`WindingNumberJacobson`](@ref) or [`WindingNumberHorman`](@ref)
- `seed`:               Coordinates of the minimum corner of the initial particle grid (excluding padding).
- `max_nparticles`:     Maximum number of particles in the initial particle grid.
- `pad_initial_particle_grid`: Padding of the initial particle grid.


!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
function ComplexShape(shape::Shapes; particle_spacing, density, pressure=0.0, mass=nothing,
                      velocity=zeros(ndims(shape)),
                      sample_boundary=false, signed_distance_field=nothing,
                      point_in_shape_algorithm=WindingNumberJacobson(; shape,
                                                                     hierarchical_winding=false,
                                                                     winding_number_factor=sqrt(eps())),
                      store_winding_number=false, seed=shape.min_corner,
                      max_nparticles=Int(1e7), pad_initial_particle_grid=2particle_spacing)
    if ndims(shape) == 3 && point_in_shape_algorithm isa WindingNumberHorman
        throw(ArgumentError("`WindingNumberHorman` only supports 2D shapes"))
    end

    if sample_boundary
        if isnothing(signed_distance_field)
            throw(ArgumentError("`SignedDistanceField` must be passed when `sample_boundary=true`"))

        elseif !(signed_distance_field.boundary_packing)
            throw(ArgumentError("`SignedDistanceField` must be generated with " *
                                "`use_for_boundary_packing=true` when `sample_boundary=true`"))
        end

        # Use the particles outside the object as boundary particles.
        (; positions, distances, max_signed_distance) = signed_distance_field

        # Delete unnecessary large signed distance field
        keep_indices = (particle_spacing .< distances .< max_signed_distance)

        boundary_coordinates = stack(positions[keep_indices])
        boundary = InitialCondition(; coordinates=boundary_coordinates, density,
                                    particle_spacing)

        fluid = sample(shape; particle_spacing, density, pressure, mass, velocity,
                       point_in_shape_algorithm, store_winding_number, seed, max_nparticles,
                       pad=pad_initial_particle_grid)

        return (fluid=fluid, boundary=boundary)
    end

    return sample(shape; particle_spacing, density, pressure, mass, velocity,
                  point_in_shape_algorithm, store_winding_number, seed, max_nparticles,
                  pad=pad_initial_particle_grid)
end

function sample(shape::Shapes; particle_spacing, density, pressure=0.0, mass=nothing,
                velocity=zeros(ndims(shape)),
                point_in_shape_algorithm=WindingNumberJacobson(; shape,
                                                               hierarchical_winding=false,
                                                               winding_number_factor=sqrt(eps())),
                store_winding_number=false, seed=shape.min_corner, max_nparticles=Int(1e7),
                pad=2particle_spacing)
    grid = particle_grid(shape, particle_spacing; pad, seed, max_nparticles)

    inpoly, winding_numbers = point_in_shape_algorithm(shape, grid; store_winding_number)
    coordinates = grid[:, inpoly]

    ic = InitialCondition(; coordinates, density, mass, velocity, pressure,
                          particle_spacing)

    # This is most probably only for debugging
    if store_winding_number
        return (initial_condition=ic, winding_numbers=winding_numbers, grid=grid)
    end

    return ic
end

function particle_grid(shape::Shapes, particle_spacing; pad=2particle_spacing,
                       seed=shape.min_corner, max_nparticles=Int(1e7))
    (; max_corner) = shape

    NDIMS = ndims(shape)

    if !(seed isa AbstractVector)
        throw(ArgumentError("`seed` must be of type `AbstractVector`"))
    elseif !(length(seed) == NDIMS)
        throw(ArgumentError("`seed` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    min_corner = seed

    ranges(dim) = (min_corner .- pad)[dim]:particle_spacing:(max_corner .+ pad)[dim]

    ranges_ = ntuple(dim -> ranges(dim), NDIMS)

    n_particles = prod(length.(ranges_))

    if n_particles > max_nparticles
        throw(ArgumentError("`particle_spacing` is too small. Initial particle grid: " *
                            "# particles ($n_particles) > `max_nparticles` ($max_nparticles)"))
    end

    return hcat(collect.(Iterators.product(ranges_...))...)
end
