struct ParticleRefinement
    n_particles_before_resize :: Ref{Int}
    n_new_particles           :: Ref{Int}
end

function ParticleRefinement()
    return ParticleRefinement(Ref(0), Ref(0))
end
