include("nhs_faces.jl")
include("signed_distance.jl")
include("system.jl")
include("rhs.jl")

function add_meta_data!(meta_data, system::ParticlePackingSystem)
    return meta_data
end
