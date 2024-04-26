abstract type Shapes{NDIMS} end

include("../point_in_poly/algorithm.jl")
include("polygon_shape.jl")
include("triangle_mesh.jl")

function ComplexShape(shape; particle_spacing, density, velocity=nothing,
                      pressure=0.0,
                      point_in_shape_algorithm=WindingNumberJacobson(), seed=nothing,
                      pad_initial_particle_grid=2particle_spacing, max_nparticles=Int(1e6))
    if ndims(shape) == 3 && point_in_shape_algorithm isa WindingNumberHorman
        throw(ArgumentError("`WindingNumberHorman` only supports 3D shapes"))
    end

    if velocity isa Nothing
        velocity = zeros(ndims(shape))
    end

    return sample(; shape, particle_spacing, density, velocity, point_in_shape_algorithm,
                  pressure, seed, pad=pad_initial_particle_grid, max_nparticles)
end

function load_shape(filename; scale_factor=nothing, ELTYPE=Float64, skipstart=1)
    if !isa(filename, String)
        throw(ArgumentError("`filename` must be of type String"))
    end

    file_extension = splitext(filename)[end]

    if file_extension == ".asc"
        points = read_in_2d(; filename, scale_factor, ELTYPE, skipstart)
        shape = Polygon(points)
    elseif file_extension == ".stl"
        # TODO: For some reason, this only works on the second run.
        mesh = load(filename)
        shape = TriangleMesh(mesh)
    else
        throw(ArgumentError("Only `.stl` and `.asc` files are supported (yet)."))
    end
end

function read_in_2d(; filename, scale_factor=nothing, ELTYPE=Float64, skipstart=1)

    # Read in the ASCII file as an Tuple containing the coordinates of the points and the
    # header.

    # Either `header=true` which returns a tuple `(data_cells, header_cells)`
    # or ignoring the corresponding number of lines from the input with `skipstart`
    points = readdlm(filename, ' ', ELTYPE, '\n'; skipstart)[:, 1:2]
    if scale_factor isa ELTYPE
        points .*= scale_factor
    elseif scale_factor !== nothing
        throw(ArgumentError("`scale_factor` must be of type $ELTYPE"))
    end

    return copy(points')
end

function particle_grid(shape, particle_spacing; pad=2particle_spacing, seed=nothing,
                       max_nparticles=Int(1e6))
    NDIMS = ndims(shape)
    (; min_box, max_box) = shape

    if seed !== nothing
        if seed isa AbstractVector && length(seed) == NDIMS
            min_corner_ = seed
        else
            throw(ArgumentError("`seed` must be of type $AbstractVector and " *
                                "of length $NDIMS for a $(NDIMS)D problem"))
        end
    else
        min_corner_ = min_box .- pad
    end

    ranges(dim) = min_corner_[dim]:particle_spacing:(max_box .+ pad)[dim]

    ranges_ = ntuple(dim -> ranges(dim), NDIMS)

    n_particles = prod(length.(ranges_))

    if n_particles > max_nparticles
        throw(ArgumentError("`particle_spacing` is too small. Initial particle grid: " *
                            "# particles ($n_particles) > `max_nparticles` ($max_nparticles)"))
    end

    return hcat(collect.(Iterators.product(ranges_...))...)
end

function sample(; shape, particle_spacing, density, velocity=zeros(ndims(shape)),
                pressure=0.0, point_in_shape_algorithm=WindingNumberJacobson(),
                pad=2particle_spacing, seed=nothing, max_nparticles=Int(1e6))
    grid = particle_grid(shape, particle_spacing; pad, seed, max_nparticles)

    inpoly = point_in_shape_algorithm(shape, grid)
    coordinates = grid[:, inpoly]

    return InitialCondition(; coordinates, density, velocity, pressure,
                            particle_spacing=particle_spacing)
end

@inline Base.ndims(::Shapes{NDIMS}) where {NDIMS} = NDIMS

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

@inline point_position(A, shape, i) = extract_svector(A, Val(ndims(shape)), i)