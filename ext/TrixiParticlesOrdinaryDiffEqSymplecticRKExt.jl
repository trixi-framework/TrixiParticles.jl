module TrixiParticlesOrdinaryDiffEqSymplecticRKExt

# This package extension defines the `SymplecticPositionVerlet` scheme from DualSPHysics.
# The scheme is similar to the `LeapfrogDriftKickDrift` scheme, but with a different
# update for the density.
# See https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#372-symplectic-position-verlet-scheme
# and the TrixiParticles.jl docs on time integration for more details.

# We need to load the name `PointNeighbors` because `@threaded` translates
# to `PointNeighbors.parallel_foreach`, so `PointNeighbors` must be available.
using TrixiParticles: TrixiParticles, @threaded, each_integrated_particle,
                      WeaklyCompressibleSPHSystem, ContinuityDensity,
                      PointNeighbors

using OrdinaryDiffEqSymplecticRK: alloc_symp_state, load_symp_state,
                                  store_symp_state!

using OrdinaryDiffEqCore: OrdinaryDiffEqCore, @.., @muladd, @cache,
                          OrdinaryDiffEqPartitionedAlgorithm,
                          OrdinaryDiffEqMutableCache

# Define a new struct for the SymplecticPositionVerlet scheme
struct SymplecticPositionVerlet <: OrdinaryDiffEqPartitionedAlgorithm end

# Overwrite the function in TrixiParticles to use the new scheme
TrixiParticles.SymplecticPositionVerlet() = SymplecticPositionVerlet()

# The following is similar to the definition of the `VerletLeapfrog` scheme
# and the corresponding cache in OrdinaryDiffEq.jl.
OrdinaryDiffEqCore.default_linear_interpolation(alg::SymplecticPositionVerlet, prob) = true

# The first stages are recomputed in every step, so there is no derivative to reuse.
# Treating this method as FSAL would trigger an unnecessary RHS evaluation after callbacks
# that modify the state.
OrdinaryDiffEqCore.isfsal(::SymplecticPositionVerlet) = false

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

function OrdinaryDiffEqCore.alg_cache(alg::SymplecticPositionVerlet, u, rate_prototype,
                                      ::Type{uEltypeNoUnits},
                                      ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                                      uprev, uprev2, f, t, dt, reltol, p, calck,
                                      ::Val{true},
                                      verbose) where {uEltypeNoUnits,
                                                      uBottomEltypeNoUnits,
                                                      tTypeNoUnits}
    tmp = zero(u)
    k = zero(rate_prototype)
    fsalfirst = zero(rate_prototype)
    half = uEltypeNoUnits(1 // 2)
    SymplecticPositionVerletCache(u, uprev, tmp, k, fsalfirst, half)
end

function OrdinaryDiffEqCore.alg_cache(alg::SymplecticPositionVerlet, u, rate_prototype,
                                      ::Type{uEltypeNoUnits},
                                      ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                                      uprev, uprev2, f, t,
                                      dt, reltol, p, calck,
                                      ::Val{false},
                                      verbose) where {uEltypeNoUnits,
                                                      uBottomEltypeNoUnits,
                                                      tTypeNoUnits}
    # We only use inplace functions in TrixiParticles, so there is no point
    # in implementing the non-inplace version.
    error("`SymplecticPositionVerlet` supports only in-place functions")
end

function OrdinaryDiffEqCore.initialize!(integrator,
                                        cache::C) where {C <: SymplecticPositionVerletCache}
    integrator.kshortsize = 2
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast

    duprev, uprev = integrator.uprev.x
    integrator.f.f1(integrator.k[2].x[1], duprev, uprev, integrator.p, integrator.t)
    integrator.f.f2(integrator.k[2].x[2], duprev, uprev, integrator.p, integrator.t)
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
    @.. broadcast=false u=uprev+dt*half*ku

    # update velocity half step
    f.f1(kdu, duprev, uprev, p, t)
    @.. broadcast=false du=duprev+dt*half*kdu

    # update velocity (add to previous full step velocity)
    f.f1(kdu, du, u, p, t + half * dt)

    # The following is equivalent to `du = duprev + dt * kdu` for the velocity, but when
    # the density is integrated, a different update is used for the density.
    semi = p.semi
    TrixiParticles.foreach_system(semi) do system
        kdu_system = TrixiParticles.wrap_v(kdu, system, semi)
        du_system = TrixiParticles.wrap_v(du, system, semi)
        duprev_system = TrixiParticles.wrap_v(duprev, system, semi)

        update_velocity!(du_system, kdu_system, duprev_system, system, semi, dt)
        update_density!(du_system, kdu_system, duprev_system, system, semi, dt)
    end

    # update position (add to half step position)
    f.f2(ku, du, u, p, t + dt)
    @.. broadcast=false u=u+dt*half*ku

    OrdinaryDiffEqCore.increment_nf!(integrator.stats, 2)
    integrator.stats.nf2 += 2
    store_symp_state!(integrator, cache, kdu, ku)
end

@muladd function update_velocity!(du_system, kdu_system, duprev_system, system, semi, dt)
    @.. broadcast=false du_system=duprev_system+dt*kdu_system
end

@inline function update_density!(du_system, kdu_system, duprev_system, system, semi, dt)
    return du_system
end

@muladd function update_velocity!(du_system, kdu_system, duprev_system,
                                  system::WeaklyCompressibleSPHSystem, semi, dt)
    # For WCSPH, only update the first NDIMS components of the velocity.
    # With `ContinuityDensity`, the last component is the density,
    # which is updated separately.
    @threaded semi for particle in each_integrated_particle(system)
        for i in 1:ndims(system)
            du_system[i,
                      particle] = duprev_system[i, particle] + dt * kdu_system[i, particle]
        end
    end
end

@inline function update_density!(du_system, kdu_system, duprev_system,
                                 system::WeaklyCompressibleSPHSystem, semi, dt)
    update_density!(du_system, kdu_system, duprev_system,
                    system.density_calculator, system, semi, dt)
end

@inline function update_density!(du_system, kdu_system, duprev_system,
                                 density_calculator, system, semi, dt)
    # Don't do anything when the density is not integrated.
    # This scheme is then equivalent to the `LeapfrogDriftKickDrift` scheme.
    return du_system
end

@muladd function update_density!(du_system, kdu_system, duprev_system,
                                 ::ContinuityDensity, system, semi, dt)
    @threaded semi for particle in each_integrated_particle(system)
        density_prev = duprev_system[end, particle]
        density_half = du_system[end, particle]
        epsilon = -kdu_system[end, particle] / density_half * dt
        du_system[end, particle] = density_prev * (2 - epsilon) / (2 + epsilon)
    end
end

# `SymplecticPositionVerletWithSorting` is a copy of `SymplecticPositionVerlet` that
# sorts particles before every force evaluation.
struct SymplecticPositionVerletWithSorting <: OrdinaryDiffEqPartitionedAlgorithm end

TrixiParticles.SymplecticPositionVerletWithSorting() = SymplecticPositionVerletWithSorting()

OrdinaryDiffEqCore.default_linear_interpolation(alg::SymplecticPositionVerletWithSorting,
                                                prob) = true

OrdinaryDiffEqCore.isfsal(::SymplecticPositionVerletWithSorting) = false

@cache struct SymplecticPositionVerletWithSortingCache{uType, rateType, uEltypeNoUnits} <:
              OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType
    k::rateType
    fsalfirst::rateType
    half::uEltypeNoUnits
end

function OrdinaryDiffEqCore.get_fsalfirstlast(cache::SymplecticPositionVerletWithSortingCache,
                                              u)
    return (cache.fsalfirst, cache.k)
end

function OrdinaryDiffEqCore.alg_cache(alg::SymplecticPositionVerletWithSorting, u,
                                      rate_prototype, ::Type{uEltypeNoUnits},
                                      ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                                      uprev, uprev2, f, t, dt, reltol, p, calck,
                                      ::Val{true}, verbose) where {uEltypeNoUnits,
                                                                  uBottomEltypeNoUnits,
                                                                  tTypeNoUnits}
    tmp = zero(u)
    k = zero(rate_prototype)
    fsalfirst = zero(rate_prototype)
    half = uEltypeNoUnits(1 // 2)
    SymplecticPositionVerletWithSortingCache(u, uprev, tmp, k, fsalfirst, half)
end

function OrdinaryDiffEqCore.alg_cache(alg::SymplecticPositionVerletWithSorting, u,
                                      rate_prototype, ::Type{uEltypeNoUnits},
                                      ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits},
                                      uprev, uprev2, f, t, dt, reltol, p, calck,
                                      ::Val{false}, verbose) where {uEltypeNoUnits,
                                                                   uBottomEltypeNoUnits,
                                                                   tTypeNoUnits}
    error("`SymplecticPositionVerletWithSorting` supports only in-place functions")
end

function OrdinaryDiffEqCore.initialize!(integrator,
                                        cache::C) where {C <:
                                                          SymplecticPositionVerletWithSortingCache}
    initialize_sorting_integrator!(integrator, cache)

    integrator.kshortsize = 2
    resize!(integrator.k, integrator.kshortsize)
    integrator.k[1] = integrator.fsalfirst
    integrator.k[2] = integrator.fsallast

    duprev, uprev = integrator.uprev.x
    integrator.f.f1(integrator.k[2].x[1], duprev, uprev, integrator.p, integrator.t)
    integrator.f.f2(integrator.k[2].x[2], duprev, uprev, integrator.p, integrator.t)
    OrdinaryDiffEqCore.increment_nf!(integrator.stats, 1)
    integrator.stats.nf2 += 1
end

function initialize_sorting_integrator!(integrator, cache)
    duprev, uprev = integrator.uprev.x
    du, u = integrator.u.x
    tmpdu, tmpu = cache.tmp.x
    lastdu, lastu = integrator.fsallast.x
    firstdu, firstu = integrator.fsalfirst.x
    history_v, history_u = interpolation_history(integrator.uprev2)

    sort_particle_systems!(duprev, uprev, integrator.p.semi,
                           (du, tmpdu, lastdu, firstdu, history_v...),
                           (u, tmpu, lastu, firstu, history_u...); initial=true)

    return integrator
end

interpolation_history(::Nothing) = ((), ())
interpolation_history(uprev2) = ((uprev2.x[1],), (uprev2.x[2],))

function sort_particle_systems!(primary_v, primary_u, semi, extra_v, extra_u;
                                initial=false)
    TrixiParticles.@trixi_timeit TrixiParticles.timer() "sort" begin
        TrixiParticles.foreach_system(semi) do system
            should_sort = system isa TrixiParticles.RequiresSortingSystem ||
                        (initial && system isa TrixiParticles.WallBoundarySystem)
            should_sort || return

            v = TrixiParticles.wrap_v(primary_v, system, semi)
            u = TrixiParticles.wrap_u(primary_u, system, semi)
            nhs = TrixiParticles.get_neighborhood_search(system, semi)
            TrixiParticles.@trixi_timeit TrixiParticles.timer() "permutation" begin
                perm = TrixiParticles.particle_sorting_permutation(system, u, nhs, semi)
            end

            TrixiParticles.@trixi_timeit TrixiParticles.timer() "sort systems" begin
                perm = TrixiParticles.sort_system_with_permutation!(system, v, u, perm,
                                                                    TrixiParticles.buffer(system))
            end

            TrixiParticles.@trixi_timeit TrixiParticles.timer() "sort arrays" begin
                sort_extra_arrays!(extra_v, primary_v, system, semi, perm,
                                TrixiParticles.wrap_v)
                sort_extra_arrays!(extra_u, primary_u, system, semi, perm,
                                TrixiParticles.wrap_u)
            end
        end
    end

    return nothing
end

function sort_extra_arrays!(arrays, primary, system, semi, perm, wrap)
    for i in eachindex(arrays)
        array = arrays[i]
        array === primary && continue

        already_sorted = false
        for j in firstindex(arrays):(i - 1)
            already_sorted |= array === arrays[j]
        end
        already_sorted && continue

        TrixiParticles.sort_particle_array!(wrap(array, system, semi), perm)
    end

    return nothing
end

@muladd function OrdinaryDiffEqCore.perform_step!(
    integrator, cache::SymplecticPositionVerletWithSortingCache, repeat_step=false)
    (; t, dt, f, p) = integrator
    duprev, uprev, _, _ = load_symp_state(integrator)
    du, u, kdu, ku = alloc_symp_state(integrator)
    lastdu, lastu = integrator.fsallast.x
    firstdu, firstu = integrator.fsalfirst.x
    history_v, history_u = interpolation_history(integrator.uprev2)

    # Update position half step.
    half = cache.half
    f.f2(ku, duprev, uprev, p, t)
    @.. broadcast=false u=uprev+dt*half*ku

    # Sort the state and all integrator-owned arrays before the first force evaluation.
    sort_particle_systems!(duprev, uprev, p.semi,
                           (du, kdu, lastdu, firstdu, history_v...),
                           (u, ku, lastu, firstu, history_u...))

    # Update velocity half step.
    f.f1(kdu, duprev, uprev, p, t)
    @.. broadcast=false du=duprev+dt*half*kdu

    # Sort again at the half step, immediately before the second force evaluation.
    sort_particle_systems!(du, u, p.semi,
                           (duprev, kdu, lastdu, firstdu, history_v...),
                           (uprev, ku, lastu, firstu, history_u...))

    # Update velocity (add to previous full step velocity).
    f.f1(kdu, du, u, p, t + half * dt)

    semi = p.semi
    TrixiParticles.foreach_system(semi) do system
        kdu_system = TrixiParticles.wrap_v(kdu, system, semi)
        du_system = TrixiParticles.wrap_v(du, system, semi)
        duprev_system = TrixiParticles.wrap_v(duprev, system, semi)

        update_velocity!(du_system, kdu_system, duprev_system, system, semi, dt)
        update_density!(du_system, kdu_system, duprev_system, system, semi, dt)
    end

    # Update position (add to half step position).
    f.f2(ku, du, u, p, t + dt)
    @.. broadcast=false u=u+dt*half*ku

    OrdinaryDiffEqCore.increment_nf!(integrator.stats, 2)
    integrator.stats.nf2 += 2
    store_symp_state!(integrator, cache, kdu, ku)
end

end # module
