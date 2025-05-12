abstract type Geometry{BACKEND, NDIMS, ELTYPE} end

include("polygon.jl")
include("triangle_mesh.jl")
include("io.jl")

@inline Base.ndims(::Geometry{BACKEND, NDIMS}) where {BACKEND, NDIMS} = NDIMS

@inline Base.eltype(::Geometry{BACKEND, NDIMS, ELTYPE}) where {BACKEND, NDIMS,
                                                               ELTYPE} = ELTYPE

@inline eachface(mesh) = Base.OneTo(nfaces(mesh))

function Base.setdiff(initial_condition::InitialCondition, geometries::Geometry...)
    geometry = first(geometries)

    if ndims(geometry) != ndims(initial_condition)
        throw(ArgumentError("all passed geometries must have the same dimensionality as the initial condition"))
    end

    coords = reinterpret(reshape, SVector{ndims(geometry), eltype(geometry)},
                         initial_condition.coordinates)

    delete_indices, _ = WindingNumberJacobson(; geometry)(geometry, coords)

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

function Base.intersect(initial_condition::InitialCondition, geometries::Geometry...)
    geometry = first(geometries)

    if ndims(geometry) != ndims(initial_condition)
        throw(ArgumentError("all passed geometries must have the same dimensionality as the initial condition"))
    end

    coords = reinterpret(reshape, SVector{ndims(geometry), eltype(geometry)},
                         initial_condition.coordinates)

    keep_indices, _ = WindingNumberJacobson(; geometry)(geometry, coords)

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
