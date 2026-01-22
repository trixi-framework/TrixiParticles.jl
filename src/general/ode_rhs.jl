function calculate_dt(v_ode, u_ode, cfl_number, semi::Semidiscretization)
    calculate_dt(v_ode, u_ode, cfl_number, semi, semi.integrate_tlsph[])
end

function calculate_dt(v_ode, u_ode, cfl_number, semi::Semidiscretization, integrate_tlsph)
    (; systems) = semi

    return minimum(systems) do system
        if system isa TotalLagrangianSPHSystem && !integrate_tlsph
            # Skip TLSPH systems if they are not integrated
            return Inf
        end
        return calculate_dt(v_ode, u_ode, cfl_number, system, semi)
    end
end

function drift!(du_ode, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "drift!" begin
        @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du_ode)

        @trixi_timeit timer() "velocity" begin
            # Set velocity and add acceleration for each system
              foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
                du = wrap_u(du_ode, system, semi)

                @threaded semi for particle in each_integrated_particle(system)
                    # This can be dispatched per system
                    add_velocity!(du, v, u, particle, system, semi, t)
                end
            end
        end
    end

    return du_ode
end

@inline function add_velocity!(du, v, u, particle, system, semi::Semidiscretization, t)
    add_velocity!(du, v, u, particle, system, t)
end

@inline function add_velocity!(du, v, u, particle, system::TotalLagrangianSPHSystem,
                               semi::Semidiscretization, t)
    # Only add velocity for TLSPH systems if they are integrated
    if semi.integrate_tlsph[]
        add_velocity!(du, v, u, particle, system, t)
    end
end

@inline function add_velocity!(du, v, u, particle, system, t)
    # Generic fallback for all systems that don't define this function
    for i in 1:ndims(system)
        @inbounds du[i, particle] = v[i, particle]
    end

    return du
end

# Solid wall boundary system doesn't integrate the particle positions
@inline add_velocity!(du, v, u, particle, system::WallBoundarySystem, t) = du

@inline function add_velocity!(du, v, u, particle, system::AbstractFluidSystem, t)
    # This is zero unless a shifting technique is used
    delta_v_ = delta_v(system, particle)

    for i in 1:ndims(system)
        @inbounds du[i, particle] = v[i, particle] + delta_v_[i]
    end

    return du
end

function kick!(dv_ode, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "kick!" begin
        # Check that the `UpdateCallback` is used if required
        check_update_callback(semi)

        @trixi_timeit timer() "reset ∂v/∂t" set_zero!(dv_ode)

        @trixi_timeit timer() "update systems and nhs" update_systems_and_nhs(v_ode, u_ode,
                                                                              semi, t)

        @trixi_timeit timer() "system interaction" system_interaction!(dv_ode, v_ode, u_ode,
                                                                       semi)

        @trixi_timeit timer() "source terms" add_source_terms!(dv_ode, v_ode, u_ode,
                                                               semi, t)
    end

    return dv_ode
end

# Update the systems for a simulation before calling `interact!` to compute forces.
function update_systems!(v_ode, u_ode, semi, t;
                         update_nhs=true,
                         update_boundary_interpolation=true,
                         update_inter_system=true)
    # First update step before updating the NHS
    # (for example for writing the current coordinates in the TLSPH system)
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_positions!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Update NHS
    if update_nhs
        @trixi_timeit timer() "update nhs" update_nhs!(semi, u_ode)
    end

    # Second update step.
    # This is used to calculate density and pressure of the fluid systems
    # before updating the boundary systems,
    # since the fluid pressure is needed by the Adami interpolation.
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_quantities!(system, v, u, v_ode, u_ode, semi, t)
    end

    if update_inter_system
        update_inter_system_quantities!(semi, v_ode, u_ode, t)
    end

    # Perform correction and pressure calculation
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_pressure!(system, v, u, v_ode, u_ode, semi, t)
    end

    # This update depends on the computed quantities of the fluid system and therefore
    # needs to be after `update_quantities!`.
    if update_boundary_interpolation
        foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
            update_boundary_interpolation!(system, v, u, v_ode, u_ode, semi, t)
        end
    end

    # Final update step for all remaining systems
    foreach_system_wrapped(semi, v_ode, u_ode) do system, v, u
        update_final!(system, v, u, v_ode, u_ode, semi, t)
    end
end

# Update the systems and neighborhood searches (NHS) for a simulation
# before calling `interact!` to compute forces.
function update_systems_and_nhs(v_ode, u_ode, semi, t)
    update_systems!(v_ode, u_ode, semi, t;
                    update_nhs=true,
                    update_boundary_interpolation=true,
                    update_inter_system=true)
end

# The `SplitIntegrationCallback` overwrites `semi_wrap` to use a different
# semidiscretization for wrapping arrays.
# TODO `semi` is not used yet, but will be used when the source terms API is modified
# to match the custom quantities API.
function add_source_terms!(dv_ode, v_ode, u_ode, semi, t; semi_wrap=semi)
    foreach_system_wrapped(semi, dv_ode, v_ode, u_ode) do system, dv, v, u
        @threaded semi for particle in each_integrated_particle(system)
            # Dispatch by system type to exclude boundary systems.
            # `integrate_tlsph` is extracted from the `semi_wrap`, so that this function
            # can be used in the `SplitIntegrationCallback` as well.
            add_acceleration!(dv, particle, system, semi_wrap.integrate_tlsph[])
            add_source_terms_inner!(dv, v, u, particle, system, source_terms(system), t,
                                    semi_wrap.integrate_tlsph[])
        end
    end

    return dv_ode
end

@inline function add_acceleration!(dv, particle, system, integrate_tlsph)
    add_acceleration!(dv, particle, system)
end

@inline function add_acceleration!(dv, particle, system::TotalLagrangianSPHSystem,
                                   integrate_tlsph)
    integrate_tlsph && add_acceleration!(dv, particle, system)
end

@inline add_acceleration!(dv, particle, system) = dv

@inline function add_acceleration!(dv, particle,
                                   system::Union{AbstractFluidSystem,
                                                 AbstractStructureSystem})
    (; acceleration) = system

    for i in 1:ndims(system)
        dv[i, particle] += acceleration[i]
    end

    return dv
end

@inline function add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t,
                                         integrate_tlsph)
    add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t)
end

@inline function add_source_terms_inner!(dv, v, u, particle,
                                         system::TotalLagrangianSPHSystem,
                                         source_terms_, t, integrate_tlsph)
    integrate_tlsph && add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t)
end

@inline function add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t)
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
    # Call `interact!` for each pair of systems
    foreach_system_indexed(semi) do system_index, system
        foreach_system_indexed(semi) do neighbor_index, neighbor
            # Construct string for the interactions timer.
            # Avoid allocations from string construction when no timers are used.
            if timeit_debug_enabled()
                timer_str = "$(timer_name(system))$system_index-$(timer_name(neighbor))$neighbor_index"
            else
                timer_str = ""
            end

            interact!(dv_ode, v_ode, u_ode, system, neighbor, semi,
                      system_index, neighbor_index; timer_str=timer_str)
        end
    end

    return dv_ode
end

# Function barrier to make benchmarking interactions easier.
# One can benchmark, e.g. the fluid-fluid interaction, with:
# dv_ode, du_ode = copy(sol.u[end]).x; v_ode, u_ode = copy(sol.u[end]).x;
# @btime TrixiParticles.interact!($dv_ode, $v_ode, $u_ode, $fluid_system, $fluid_system, $semi);
@inline function interact!(dv_ode, v_ode, u_ode, system, neighbor, semi; timer_str="")
    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor, semi)

    return interact!(dv_ode, v_ode, u_ode, system, neighbor, semi,
                     system_index, neighbor_index; timer_str=timer_str)
end

@inline function interact!(dv_ode, v_ode, u_ode, system, neighbor, semi,
                           system_index::Integer, neighbor_index::Integer; timer_str="")
    dv = wrap_v(dv_ode, system, semi, system_index)
    v_system = wrap_v(v_ode, system, semi, system_index)
    u_system = wrap_u(u_ode, system, semi, system_index)

    v_neighbor = wrap_v(v_ode, neighbor, semi, neighbor_index)
    u_neighbor = wrap_u(u_ode, neighbor, semi, neighbor_index)

    @trixi_timeit timer() timer_str begin
        interact!(dv, v_system, u_system, v_neighbor, u_neighbor, system, neighbor, semi)
    end
end

function check_update_callback(semi)
    foreach_system(semi) do system
        # This check will be optimized away if the system does not require the callback
        if requires_update_callback(system) && !semi.update_callback_used[]
            system_name = system |> typeof |> nameof
            throw(ArgumentError("`UpdateCallback` is required for `$system_name`"))
        end
    end
end

