function drift!(du_ode, v_ode, u_ode, p, t)
    (; semi) = p

    @trixi_timeit timer() "drift!" begin
        foreach_system(semi) do system
            du = wrap_u(du_ode, system, semi)
            v = wrap_v(v_ode, system, semi)
            u = wrap_u(u_ode, system, semi)

            set_velocity!(du, v, u, system, semi, t)
        end
    end

    return du_ode
end

# Generic fallback for all systems that don't define this function
function set_velocity!(du, v, u, system, semi, t)
    set_velocity_default!(du, v, u, system, semi, t)
end

# Only set velocity for TLSPH systems if they are integrated
function set_velocity!(du, v, u, system::TotalLagrangianSPHSystem, semi, t)
    if semi.integrate_tlsph[]
        set_velocity_default!(du, v, u, system, semi, t)
    else
        set_zero!(du)
    end

    return du
end

# Solid wall boundary system doesn't integrate the particle positions
function set_velocity!(du, v, u, system::WallBoundarySystem, semi, t)
    # Note that `du` is of length zero, so we don't have to set it to zero
    return du
end

# Fluid systems integrate the particle positions and can have a shifting velocity
function set_velocity!(du, v, u, system::AbstractFluidSystem, semi, t)
    @threaded semi for particle in each_integrated_particle(system)
        delta_v_ = @inbounds delta_v(system, particle)

        for i in 1:ndims(system)
            @inbounds du[i, particle] = v[i, particle] + delta_v_[i]
        end
    end

    return du
end

function set_velocity_default!(du, v, u, system, semi, t)
    @threaded semi for particle in each_integrated_particle(system)
        for i in 1:ndims(system)
            @inbounds du[i, particle] = v[i, particle]
        end
    end

    return du
end

# This defaults to optimized GPU copy that is about 4x faster than the threaded version above
function set_velocity_default!(du::AbstractGPUArray, v, u, system, semi, t)
    indices = CartesianIndices(du)
    copyto!(du, indices, v, indices)
end

function kick!(dv_ode, v_ode, u_ode, p, t)
    (; semi, split_integration_data) = p

    # This is a no-op if no split integration
    # or split integration without stage-coupling is used.
    split_integrate_stage!(v_ode, u_ode, t, split_integration_data)

    @trixi_timeit timer() "kick!" begin
        # Check that the `UpdateCallback` is used if required
        check_update_callback(semi)

        @trixi_timeit timer() "reset ∂v/∂t" set_zero!(dv_ode)

        @trixi_timeit timer() "update systems and nhs" update_systems_and_nhs(v_ode, u_ode,
                                                                              semi, t)

        @trixi_timeit timer() "system interaction" system_interaction!(dv_ode, v_ode, u_ode,
                                                                       semi)

        add_source_terms!(dv_ode, v_ode, u_ode, semi, t)
    end

    return dv_ode
end

# Update the systems and neighborhood searches (NHS) for a simulation
# before calling `interact!` to compute forces.
function update_systems_and_nhs(v_ode, u_ode, semi, t)
    # First update step before updating the NHS
    # (for example for writing the current coordinates in the TLSPH system)
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_positions!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Update NHS
    @trixi_timeit timer() "update nhs" update_nhs!(semi, u_ode)

    # Second update step.
    # This is used to calculate density and pressure of the fluid systems
    # before updating the boundary systems,
    # since the fluid pressure is needed by the Adami interpolation.
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_quantities!(system, v, u, v_ode, u_ode, semi, t)
    end

    update_implicit_sph!(semi, v_ode, u_ode, t)

    # Perform correction and pressure calculation
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_pressure!(system, v, u, v_ode, u_ode, semi, t)
    end

    # This update depends on the computed quantities of the fluid system and therefore
    # needs to be after `update_quantities!`.
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_boundary_interpolation!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Final update step for all remaining systems
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_final!(system, v, u, v_ode, u_ode, semi, t)
    end
end

# Some systems accumulate pairwise interaction state outside `dv_ode`. Reset that state once
# at the beginning of every explicitly assembled interaction pass.
function reset_interaction_caches!(semi::Union{NamedTuple, Semidiscretization})
    foreach_system(semi) do system
        reset_interaction_caches!(system)
    end

    return semi
end

# The `SplitIntegrationCallback` overwrites `semi_wrap` to use a different
# semidiscretization for wrapping arrays.
# `semi_wrap` is the small semidiscretization, `semi` is the large semidiscretization.
# TODO `semi` is not used yet, but will be used when the source terms API is modified
# to match the custom quantities API.
function add_source_terms!(dv_ode, v_ode, u_ode, semi, t; semi_wrap=semi)
    foreach_system_wrapped(semi_wrap, v_ode, u_ode) do system, v, u
        dv = wrap_v(dv_ode, system, semi_wrap)

        # `integrate_tlsph` is extracted from the `semi_wrap`, so that this function
        # can be used in the `SplitIntegrationCallback` as well.
        # In this case, `semi_wrap` will be the small sub-integration semidiscretization.
        add_source_terms!(dv, v, u, system, semi, t, semi_wrap.integrate_tlsph[])
    end

    return dv_ode
end

# This is a no-op by default but can be dispatched by system type
function add_source_terms!(dv, v, u, system, semi, t, integrate_tlsph)
    return dv
end

function add_source_terms!(dv, v, u,
                           system::Union{AbstractFluidSystem, AbstractStructureSystem},
                           semi, t, integrate_tlsph)
    add_source_terms_inner!(dv, v, u, system, semi, t)
end

function add_source_terms!(dv, v, u, system::TotalLagrangianSPHSystem,
                           semi, t, integrate_tlsph)
    if integrate_tlsph
        add_source_terms_inner!(dv, v, u, system, semi, t)
    end

    return dv
end

function add_source_terms_inner!(dv, v, u,
                                 system::Union{AbstractFluidSystem,
                                               AbstractStructureSystem},
                                 semi, t)
    if iszero(system.acceleration) && isnothing(source_terms(system))
        # Nothing to do
        return dv
    end

    @trixi_timeit timer() "source terms" begin
        @threaded semi for particle in each_integrated_particle(system)
            add_acceleration!(dv, system, particle)
            add_source_terms_inner!(dv, v, u, particle, system, source_terms(system), t)
        end
    end

    return dv
end

@inline source_terms(system) = nothing
@inline source_terms(system::Union{AbstractFluidSystem, AbstractStructureSystem}) = system.source_terms

@inline function add_acceleration!(dv, system, particle)
    (; acceleration) = system

    for i in 1:ndims(system)
        @inbounds dv[i, particle] += acceleration[i]
    end

    return dv
end

@propagate_inbounds function add_source_terms_inner!(dv, v, u, particle,
                                                     system::RigidBodySystem,
                                                     source_terms_, t)
    coords = current_coords(u, system, particle)
    velocity = current_velocity(v, system, particle)
    density = system.material_density[particle]
    pressure = 0 # Rigid body systems don't have a pressure, but some source terms might depend on it

    source = source_terms_(coords, velocity, density, pressure, t)

    for i in eachindex(source)
        dv[i, particle] += source[i]
    end

    return dv
end

@inline add_source_terms_inner!(dv, v, u, particle,
                                system::RigidBodySystem,
                                source_terms_::Nothing, t) = dv

@propagate_inbounds function add_source_terms_inner!(dv, v, u, particle, system,
                                                     source_terms_, t)
    coords = current_coords(u, system, particle)
    velocity = current_velocity(v, system, particle)
    density = current_density(v, system, particle)
    pressure = current_pressure(v, system, particle)

    source = source_terms_(coords, velocity, density, pressure, t)

    # Loop over `eachindex(source)`, so that users could also pass source terms for
    # the density when using `ContinuityDensity`.
    for i in eachindex(source)
        dv[i, particle] += source[i]
    end

    return dv
end

@inline add_source_terms_inner!(dv, v, u, particle, system, source_terms_::Nothing, t) = dv

function system_interaction!(dv_ode, v_ode, u_ode, semi)
    reset_interaction_caches!(semi)

    # Call `interact!` for each ordered pair of systems.
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            has_system_interaction(system, neighbor, semi) || return dv_ode

            # Construct string for the interactions timer.
            # Avoid allocations from string construction when no timers are used.
            if timeit_debug_enabled()
                system_index = system_indices(system, semi)
                neighbor_index = system_indices(neighbor, semi)
                timer_str = "$(timer_name(system))$system_index-$(timer_name(neighbor))$neighbor_index"
            else
                timer_str = ""
            end

            interact!(dv_ode, v_ode, u_ode, system, neighbor, semi; timer_str)
        end
    end

    # Finalize systems that need to reduce accumulated interaction data afterward.
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        dv = wrap_v(dv_ode, system, semi)

        finalize_interaction!(system, dv, v, u, dv_ode, v_ode, u_ode, semi)
    end

    return dv_ode
end

# Function barrier to make benchmarking interactions easier.
# One can benchmark, e.g. the fluid-fluid interaction, with:
# dv_ode, du_ode = copy(sol.u[end]).x; v_ode, u_ode = copy(sol.u[end]).x;
# For manual multi-pair interaction assembly, call `reset_interaction_caches!(semi)` once
# before the first direct `interact!` call.
# @btime TrixiParticles.interact!($dv_ode, $v_ode, $u_ode, $fluid_system, $fluid_system, $semi);
@inline function interact!(dv_ode, v_ode, u_ode, system, neighbor, semi; timer_str="")
    dv = wrap_v(dv_ode, system, semi)
    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)

    v_neighbor = wrap_v(v_ode, neighbor, semi)
    u_neighbor = wrap_u(u_ode, neighbor, semi)

    @trixi_timeit timer() timer_str begin
        apply_system_interaction!(dv, v_system, u_system, v_neighbor, u_neighbor,
                                  system, neighbor, semi)
    end

    return dv_ode
end

@inline function apply_system_interaction!(dv, v_system, u_system, v_neighbor,
                                           u_neighbor, system, neighbor, semi; kwargs...)
    interaction = system_interaction(system, neighbor, semi)
    return apply_interaction!(interaction, dv, v_system, u_system, v_neighbor,
                              u_neighbor, system, neighbor, semi; kwargs...)
end

@inline function apply_system_interaction!(dv, v_system, u_system, v_neighbor,
                                           u_neighbor, system::TotalLagrangianSPHSystem,
                                           neighbor, semi;
                                           integrate_tlsph=semi.integrate_tlsph[],
                                           kwargs...)
    integrate_tlsph || return dv

    interaction = system_interaction(system, neighbor, semi)
    return apply_interaction!(interaction, dv, v_system, u_system, v_neighbor,
                              u_neighbor, system, neighbor, semi; kwargs...)
end

@inline function apply_interaction!(interaction::Bool, dv, v_system, u_system,
                                    v_neighbor, u_neighbor, system, neighbor, semi;
                                    kwargs...)
    interaction || return dv
    return interact!(dv, v_system, u_system, v_neighbor, u_neighbor, system, neighbor,
                     semi; kwargs...)
end

@inline function apply_interaction!(interaction, dv, v_system, u_system,
                                    v_neighbor, u_neighbor, system, neighbor, semi;
                                    kwargs...)
    return interaction(dv, v_system, u_system, v_neighbor, u_neighbor, system, neighbor,
                       semi; kwargs...)
end

function check_update_callback(semi)
    foreach_system(semi) do system
        # This check will be optimized away if the system does not require the callback
        if requires_update_callback(system, semi) && !semi.update_callback_used[]
            system_name = system |> typeof |> nameof
            throw(ArgumentError("`UpdateCallback` is required for `$system_name`"))
        end
    end
end
