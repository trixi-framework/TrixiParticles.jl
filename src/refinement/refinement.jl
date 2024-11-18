include("resize.jl")

struct ParticleRefinement
    n_particles_before_resize :: Ref{Int}
    n_new_particles           :: Ref{Int}
    delete_candidates         :: Vector{Bool}   # length(delete_candidates) == nparticles
end

function ParticleRefinement()
    return ParticleRefinement(Ref(0), Ref(0), Bool[])
end
