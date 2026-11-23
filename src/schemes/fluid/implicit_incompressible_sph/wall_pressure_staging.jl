const IISPHExtrapolatingWall = WallBoundarySystem{<:BoundaryModelDummyParticles{<:Union{AdamiPressureExtrapolation,
                                                                                        BernoulliPressureExtrapolation}}}
const IISPHPressureBoundary = WallBoundarySystem{<:BoundaryModelDummyParticles{<:PressureBoundaries}}
const IISPHPressureUnknown = Union{ImplicitIncompressibleSPHSystem, IISPHPressureBoundary}

@inline configured_iisph_eos_wall(system) = false
@inline configured_iisph_eos_wall(wall::IISPHExtrapolatingWall) = has_pressure_extrapolated_density(wall)

@inline iisph_eos_wall_pair(fluid, wall, semi) = false

@inline function iisph_eos_wall_pair(fluid::ImplicitIncompressibleSPHSystem,
                                     wall::IISPHExtrapolatingWall, semi)
    return configured_iisph_eos_wall(wall) && system_interaction(fluid, wall, semi) === true
end

@inline has_iisph_wall_contact(system) = false
@inline has_iisph_wall_contact(system::ImplicitIncompressibleSPHSystem) = any(system.cache.extrapolating_wall_contact)

@inline is_iisph_fluid(system) = system isa ImplicitIncompressibleSPHSystem
@inline has_iisph_pressure_unknowns(system) = system isa IISPHPressureUnknown

@inline function iisph_fluid_samples_wall(fluids, interactions, wall_index)
    return any(i -> fluids[i] && is_enabled_interaction(interactions[i, wall_index]),
               eachindex(fluids))
end

@inline function iisph_component_workspace(systems)
    return iisph_component_workspace(first(systems), Base.tail(systems))
end

@inline iisph_component_workspace(::Tuple{}) = nothing
@inline iisph_component_workspace(system, remaining) = iisph_component_workspace(remaining)
@inline iisph_component_workspace(system::ImplicitIncompressibleSPHSystem,
                                  remaining) = system.cache.pressure_components

@inline function iisph_component_root(labels, i)
    while labels[i] != i
        i = labels[i]
    end
    return i
end

function label_iisph_pressure_components!(labels, systems, interactions)
    resize!(labels, length(systems))
    unknowns = map(has_iisph_pressure_unknowns, systems)
    eos_walls = map(configured_iisph_eos_wall, systems)
    fluids = map(is_iisph_fluid, systems)
    for i in eachindex(labels)
        # EOS walls join the component of the fluids whose pressures they extrapolate
        # or whose PPE uses their frozen force. Pressure boundaries couple through fluid rows.
        member = unknowns[i] ||
                 (eos_walls[i] && iisph_fluid_samples_wall(fluids, interactions, i))
        labels[i] = member ? i : 0
    end
    for i in eachindex(labels)
        fluids[i] || continue
        for j in eachindex(labels)
            (i == j || iszero(labels[j])) && continue
            if is_enabled_interaction(interactions[i, j]) ||
               is_enabled_interaction(interactions[j, i])
                a = iisph_component_root(labels, i)
                b = iisph_component_root(labels, j)
                labels[max(a, b)] = min(a, b)
            end
        end
    end
    for i in eachindex(labels)
        iszero(labels[i]) || (labels[i] = iisph_component_root(labels, i))
    end
    return labels
end

function check_iisph_wall_configuration(systems, interactions)
    any(configured_iisph_eos_wall, systems) || return nothing
    labels = iisph_component_workspace(systems)
    labels === nothing && return nothing
    label_iisph_pressure_components!(labels, systems, interactions)

    # The EOS is a constitutive law for the walls of this pressure component. Check
    # parameters here rather than letting an invalid inverse fail inside a particle loop.
    for (i, wall) in pairs(systems)
        configured_iisph_eos_wall(wall) || continue
        iszero(labels[i]) && continue
        model = wall.boundary_model
        eos = model.state_equation
        valid_iisph_wall_eos(eos) ||
            throw(ArgumentError("IISPH extrapolating wall $i needs a valid StateEquationCole or StateEquationIdealGas with a positive finite density at zero pressure"))
        if clip_negative_pressure(eos) && !clip_negative_pressure(model)
            throw(ArgumentError("IISPH extrapolating wall $i must clip negative pressure when its equation of state clips it"))
        end

        for (j, fluid) in pairs(systems)
            labels[j] == labels[i] && fluid isa ImplicitIncompressibleSPHSystem || continue
            if is_enabled_interaction(interactions[j, i]) && interactions[j, i] !== true
                throw(ArgumentError("IISPH fluid $j must use the default interaction with EOS extrapolating wall $i"))
            end
            fluid.reference_density == eos.reference_density ||
                throw(ArgumentError("IISPH fluid $j and extrapolating wall $i must have the same EOS reference density"))
            check_iisph_extrapolating_wall_eos(eos, i, j, systems, interactions)
        end
    end
    return nothing
end

function check_iisph_extrapolating_wall_eos(eos, wall_index, fluid_index, systems,
                                            interactions)
    for (i, wall) in pairs(systems)
        wall isa IISPHExtrapolatingWall || continue
        is_enabled_interaction(interactions[fluid_index, i]) || continue
        wall.boundary_model.state_equation === eos ||
            throw(ArgumentError("IISPH extrapolating walls $wall_index and $i in the same pressure component must use the same EOS and parameters, or both omit the EOS"))
    end
    return nothing
end

@inline valid_iisph_wall_eos(eos) = false

function valid_iisph_wall_eos(eos::Union{StateEquationCole, StateEquationIdealGas})
    power = iisph_eos_power(eos)
    all(x -> isfinite(x) && x > 0, (power, eos.sound_speed, eos.reference_density)) ||
        return false
    isfinite(eos.background_pressure) || return false
    zero_pressure = zero(float(eos.reference_density))
    density = predict_density_from_pressure(eos, zero_pressure, oftype(zero_pressure, NaN))
    return isfinite(density)
end

@inline iisph_eos_power(eos::StateEquationCole) = eos.exponent
@inline iisph_eos_power(eos::StateEquationIdealGas) = eos.gamma

@inline function prepare_iisph_pressure_components!(systems, interactions)
    any(configured_iisph_eos_wall, systems) || return nothing
    labels = iisph_component_workspace(systems)
    labels === nothing && return nothing
    label_iisph_pressure_components!(labels, systems, interactions)

    # Negative labels mark components with geometric contact in this RHS evaluation.
    contacts = map(has_iisph_wall_contact, systems)
    for i in eachindex(labels)
        contacts[i] || continue
        label = labels[i]
        for j in eachindex(labels)
            if labels[j] == label
                labels[j] = -abs(label)
            end
        end
    end
    return labels
end

@inline in_iisph_rows(system, ::AllIISPHRows, semi) = true
@inline in_iisph_rows(system, rows::IISPHComponentRows,
                      semi) = rows.labels[system_indices(system, semi)] == rows.label

@inline preserve_iisph_wall_state(labels, system, semi) = false

@inline function preserve_iisph_wall_state(labels::Vector{Int},
                                           wall::IISPHExtrapolatingWall, semi)
    configured_iisph_eos_wall(wall) || return false
    return labels[system_indices(wall, semi)] < 0
end

seed_iisph_wall_densities!(::Nothing, semi, v_ode, u_ode) = nothing

function seed_iisph_wall_densities!(labels, semi, v_ode, u_ode)
    foreach_system(semi) do system
        system isa IISPHPressureUnknown || return
        if labels[system_indices(system, semi)] < 0
            set_zero!(current_pressure(nothing, system))
        end
    end
    foreach_system_wrapped(semi, v_ode, u_ode) do wall, v, u
        preserve_iisph_wall_state(labels, wall, semi) || return
        model = wall.boundary_model
        predict_pressure_extrapolated_density!(model, wall, v, u, v_ode, u_ode, semi;
                                               viscosity=model.viscosity)
    end
    return nothing
end

function try_freeze_iisph_wall_state!(rows, semi, v_ode, u_ode)
    valid = Ref(true)
    foreach_system_wrapped(semi, v_ode, u_ode) do wall, v, u
        preserve_iisph_wall_state(rows.labels, wall, semi) || return
        in_iisph_rows(wall, rows, semi) || return
        valid[] &= try_freeze_iisph_wall_state!(wall, v, u, v_ode, u_ode, semi)
    end
    return valid[]
end

function try_freeze_iisph_wall_state!(wall, v, u, v_ode, u_ode, semi)
    model = wall.boundary_model
    (; pressure, cache, state_equation, viscosity) = model
    extrapolate_boundary_pressure!(model, wall, v, u, v_ode, u_ode, semi, viscosity)
    @threaded semi for particle in eachparticle(wall)
        p = normalized_boundary_pressure(model, particle)
        @inbounds pressure[particle] = p
        @inbounds cache.density[particle] = predict_density_from_pressure(state_equation, p,
                                                                          oftype(p, NaN))
        update_extrapolated_wall_velocity!(cache, viscosity, wall, v, particle)
    end
    return all(isfinite, cache.density)
end
