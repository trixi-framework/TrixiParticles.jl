abstract type ProvidesParticleGroup end

struct ParticleGroup{NDIMS, ELTYPE <: Real} <: ProvidesParticleGroup
    coordinates :: Array{ELTYPE, 2}
    velocities  :: Array{ELTYPE, 2}
    densities   :: Vector{ELTYPE}
    masses      :: Vector{ELTYPE}
end

include("circular_shape.jl")
include("rectangular_shape.jl")
include("rectangular_tank.jl")
