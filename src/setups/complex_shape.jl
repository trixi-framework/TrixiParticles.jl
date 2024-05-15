function ComplexShape(shape::Shapes; particle_spacing, density, velocity=nothing,
                      pressure=0.0,
                      point_in_shape_algorithm=WindingNumberJacobson(), seed=nothing,
                      hierarchical_winding=false,
                      pad_initial_particle_grid=2particle_spacing, max_nparticles=Int(1e7))
    if point_in_shape_algorithm isa WindingNumberJacobson && hierarchical_winding
        bbox = BoundingBoxTree(eachface(shape), shape.min_corner, shape.max_corner)

        directed_edges = zeros(Int, length(shape.normals_edge))
        construct_hierarchy!(bbox, shape, directed_edges)

        point_in_shape_algorithm = WindingNumberJacobson(;
                                                         winding=HierarchicalWinding(bbox))
    elseif point_in_shape_algorithm isa WindingNumberJacobson
        point_in_shape_algorithm = WindingNumberJacobson(; winding=NaiveWinding())
    end

    if ndims(shape) == 3 && point_in_shape_algorithm isa WindingNumberHorman
        throw(ArgumentError("`WindingNumberHorman` only supports 2D shapes"))
    end

    if velocity isa Nothing
        velocity = zeros(ndims(shape))
    end

    return sample(shape; particle_spacing, density, velocity, point_in_shape_algorithm,
                  pressure, seed, pad=pad_initial_particle_grid, max_nparticles)
end

function sample(shape::Shapes; particle_spacing, density, velocity=zeros(ndims(shape)),
                pressure=0.0, point_in_shape_algorithm=WindingNumberJacobson(),
                pad=2particle_spacing, seed=nothing, max_nparticles=Int(1e6))
    grid = particle_grid(shape, particle_spacing; pad, seed, max_nparticles)

    inpoly = point_in_shape_algorithm(shape, grid)
    coordinates = grid[:, inpoly]

    return InitialCondition(; coordinates, density, velocity, pressure,
                            particle_spacing=particle_spacing)
end

function particle_grid(shape::Shapes, particle_spacing; pad=2particle_spacing, seed=nothing,
                       max_nparticles=Int(1e6))
    NDIMS = ndims(shape)
    (; min_corner, max_corner) = shape

    if seed !== nothing
        if seed isa AbstractVector && length(seed) == NDIMS
            min_corner_ = seed
        else
            throw(ArgumentError("`seed` must be of type $AbstractVector and " *
                                "of length $NDIMS for a $(NDIMS)D problem"))
        end
    else
        min_corner_ = min_corner .- pad
    end

    ranges(dim) = min_corner_[dim]:particle_spacing:(max_corner .+ pad)[dim]

    ranges_ = ntuple(dim -> ranges(dim), NDIMS)

    n_particles = prod(length.(ranges_))

    if n_particles > max_nparticles
        throw(ArgumentError("`particle_spacing` is too small. Initial particle grid: " *
                            "# particles ($n_particles) > `max_nparticles` ($max_nparticles)"))
    end

    return hcat(collect.(Iterators.product(ranges_...))...)
end
