module TrixiParticlesOrdinaryDiffEqExt

using TrixiParticles

# This is needed because `TrixiParticles.@threaded` translates
# to `PointNeighbors.parallel_foreach`, so `PointNeighbors` must be available.
const PointNeighbors = TrixiParticles.PointNeighbors

using OrdinaryDiffEqCore: @.., @muladd, @cache, OrdinaryDiffEqCore,
                          OrdinaryDiffEqPartitionedAlgorithm, OrdinaryDiffEqMutableCache

struct SymplecticPositionVerlet <: OrdinaryDiffEqPartitionedAlgorithm end

TrixiParticles.SymplecticPositionVerlet() = SymplecticPositionVerlet()

OrdinaryDiffEqCore.default_linear_interpolation(alg::SymplecticPositionVerlet, prob) = true

@cache struct SymplecticPositionVerletCache{uType, rateType, uEltypeNoUnits} <:
              OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    k::rateType
    fsalfirst::rateType
    half::uEltypeNoUnits
end

function OrdinaryDiffEqCore.get_fsalfirstlast(cache::SymplecticPositionVerletCache, u)
    return (cache.fsalfirst, cache.k)
end

# Copied from OrdinaryDiffEqSymplecticRK
#
# provide the mutable uninitialized objects to keep state and derivative in case of mutable caches
# no such objects are required for constant caches
function alloc_symp_state(integrator)
    (integrator.u.x..., integrator.cache.tmp.x...)
end

# load state and derivatives at begin of symplectic iteration steps
function load_symp_state(integrator)
    (integrator.uprev.x..., integrator.fsallast.x...)
end

# store state and derivatives at the end of symplectic iteration steps
function store_symp_state!(integrator, cache, kdu, ku)
    copyto!(integrator.k[1].x[1], integrator.k[2].x[1])
    copyto!(integrator.k[1].x[2], integrator.k[2].x[2])
    copyto!(integrator.k[2].x[2], ku)
    copyto!(integrator.k[2].x[1], kdu)
    nothing
end

function OrdinaryDiffEqCore.alg_cache(alg::SymplecticPositionVerlet, u, rate_prototype,
                                      ::Type{uEltypeNoUnits},
                                      ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                                      uprev,
                                      uprev2, f, t,
                                      dt, reltol, p, calck,
                                      ::Val{true}) where {uEltypeNoUnits,
                                                          uBottomEltypeNoUnits,
                                                          tTypeNoUnits}
    tmp = zero(u)
    k = zero(rate_prototype)
    fsalfirst = zero(rate_prototype)
    half = uEltypeNoUnits(1 // 2)
    SymplecticPositionVerletCache(u, uprev, k, tmp, fsalfirst, half)
end

function OrdinaryDiffEqCore.alg_cache(alg::SymplecticPositionVerlet, u, rate_prototype,
                                      ::Type{uEltypeNoUnits},
                                      ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                                      uprev,
                                      uprev2, f, t,
                                      dt, reltol, p, calck,
                                      ::Val{false}) where {uEltypeNoUnits,
                                                           uBottomEltypeNoUnits,
                                                           tTypeNoUnits}
    error("`SymplecticPositionVerlet` only supports inplace functions")
end

function OrdinaryDiffEqCore.initialize!(integrator,
                                        cache::C) where {C <: SymplecticPositionVerletCache}
    integrator.kshortsize = 2
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast

    duprev, uprev = integrator.uprev.x
    integrator.f.f1(integrator.k[2].x[1], duprev, uprev, integrator.p, integrator.t)
    # verify_f2(integrator.f.f2, integrator.k[2].x[2], duprev, uprev, integrator.p,
    #     integrator.t, integrator, cache)
    OrdinaryDiffEqCore.increment_nf!(integrator.stats, 1)
    integrator.stats.nf2 += 1
end

@muladd function OrdinaryDiffEqCore.perform_step!(integrator,
                                                  cache::SymplecticPositionVerletCache,
                                                  repeat_step=false)
    (; t, dt, f, p) = integrator
    duprev, uprev, _, _ = load_symp_state(integrator)
    du, u, kdu, ku = alloc_symp_state(integrator)

    # update position half step
    half = cache.half
    f.f2(ku, duprev, uprev, p, t)
    @.. broadcast=false u=uprev + dt * half * ku

    # update velocity half step
    f.f1(kdu, duprev, uprev, p, t)
    @.. broadcast=false du=duprev + dt * half * kdu

    # update velocity (add to previous full step velocity)
    f.f1(kdu, du, u, p, t + half * dt)
    # The following is equivalent to `du = duprev + dt * kdu` for the velocity, but when
    # the density is integrated, a different update is used for the density.
    semi = p
    TrixiParticles.foreach_system(semi) do system
        kdu_system = TrixiParticles.wrap_v(kdu, system, semi)
        du_system = TrixiParticles.wrap_v(du, system, semi)
        duprev_system = TrixiParticles.wrap_v(duprev, system, semi)

        update_velocity!(du_system, kdu_system, duprev_system, system, dt)
        update_density!(du_system, kdu_system, duprev_system, system, dt)
    end

    # update position (add to half step position)
    f.f2(ku, du, u, p, t + dt)
    @.. broadcast=false u=u + dt * half * ku

    OrdinaryDiffEqCore.increment_nf!(integrator.stats, 2)
    integrator.stats.nf2 += 2
    store_symp_state!(integrator, cache, kdu, ku)
end

@muladd function update_velocity!(du_system, kdu_system, duprev_system, system, dt)
    @.. broadcast=false du_system=duprev_system + dt * kdu_system
end

@inline function update_density!(du_system, kdu_system, duprev_system, system, dt)
    return du_system
end

@muladd function update_velocity!(du_system, kdu_system, duprev_system,
                                  system::WeaklyCompressibleSPHSystem, dt)
    TrixiParticles.@threaded system for particle in TrixiParticles.each_moving_particle(system)
        for i in 1:ndims(system)
            du_system[i, particle] = duprev_system[i, particle] +
                                     dt * kdu_system[i, particle]
        end
    end
end

@inline function update_density!(du_system, kdu_system, duprev_system,
                                 system::WeaklyCompressibleSPHSystem, dt)
    update_density!(du_system, kdu_system, duprev_system,
                    system.density_calculator, system, dt)
end

@inline function update_density!(du_system, kdu_system, duprev_system,
                                 density_calculator, system, dt)
    return du_system
end

@muladd function update_density!(du_system, kdu_system, duprev_system,
                                 ::ContinuityDensity, system, dt)
    TrixiParticles.@threaded system for particle in TrixiParticles.each_moving_particle(system)
        density_prev = duprev_system[end, particle]
        density_half = du_system[end, particle]
        epsilon = -kdu_system[end, particle] / density_half * dt
        du_system[end, particle] = density_prev * (2 - epsilon) / (2 + epsilon)
    end
end

end # module
