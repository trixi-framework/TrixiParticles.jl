struct SPSTurbulenceModelDalrymple{ELTYPE}
    smagorinsky_constant  :: ELTYPE
    isotropic_constant    :: ELTYPE
    smallest_length_scale :: ELTYPE
end

function SPSTurbulenceModelDalrymple(; smallest_length_scale, smagorinsky_constant=0.12,
                                     isotropic_constant=6.6e-4)
    return SPSTurbulenceModelDalrymple(smagorinsky_constant, isotropic_constant,
                                       smallest_length_scale)
end

# Dispatch on system
@inline function turbulence_model(system)
    return nothing
end

function create_cache_turbulence(initial_condition, turbulence_model::Nothing)
    return (; turbulence_model=nothing)
end

function create_cache_turbulence(initial_condition, turbulence_model)
    ELTYPE = eltype(initial_condition)
    NDIMS = ndims(initial_condition)
    fluid_stress_tensor = fill(zero(SMatrix{NDIMS, NDIMS, ELTYPE}),
                               nparticles(initial_condition))

    return (; fluid_stress_tensor, turbulence_model)
end

@inline function current_fluid_stress_tensor(system, particle)
    current_fluid_stress_tensor(system, turbulence_model(system), particle,
                                Val(eltype(system)))
end

@inline function current_fluid_stress_tensor(system::AbstractSystem{NDIMS}, ::Nothing,
                                             particle, ::Val{ELTYPE}) where {NDIMS, ELTYPE}
    return zero(SMatrix{NDIMS, NDIMS, ELTYPE})
end
@inline function current_fluid_stress_tensor(system, turbulence_model, particle, ELTYPE)
    return system.cache.fluid_stress_tensor[particle]
end

function dv_stress(system, neighbor_system,
                   particle, neighbor, pos_diff, distance,
                   m_a, m_b, rho_a, rho_b, grad_kernel)
    return dv_stress(turbulence_model(system), system, neighbor_system,
                     particle, neighbor, pos_diff, distance,
                     m_a, m_b, rho_a, rho_b, grad_kernel)
end

function dv_stress(::Nothing, system, neighbor_system, particle, neighbor, pos_diff,
                   distance, m_a, m_b, rho_a, rho_b, grad_kernel)
    return zero(grad_kernel)
end

function dv_stress(turbulence_model, system, neighbor_system, particle, neighbor, pos_diff,
                   distance, m_a, m_b, rho_a, rho_b, grad_kernel)
    tau_a = current_fluid_stress_tensor(system, particle)
    tau_b = current_fluid_stress_tensor(neighbor_system, neighbor)

    return pressure_acceleration(system, neighbor_system, particle, neighbor,
                                 m_a, m_b, tau_a, tau_b, rho_a, rho_b, pos_diff,
                                 distance, grad_kernel, nothing)
end

function update_turbulence_models!(system, ::Nothing, v, u, v_ode, u_ode, semi)
    return system
end

function update_turbulence_models!(system, turbulence_model, v, u, v_ode, u_ode, semi)
    calculate_fluid_stress_tensor!(system, turbulence_model, v, u, v_ode, u_ode, semi)

    return system
end

@inline calculate_fluid_stress_tensor!(system, ::Nothing, v, u, v_ode, u_ode, semi) = system

@inline function calculate_fluid_stress_tensor!(system, turbulence_model, v, u,
                                                v_ode, u_ode, semi)
    (; fluid_stress_tensor) = system.cache
    (; smagorinsky_constant, isotropic_constant, smallest_length_scale) = turbulence_model

    fill!(fluid_stress_tensor, zero(eltype(fluid_stress_tensor)))

    # Accumulate per-particle velocity-gradient increments into the preallocated
    # `fluid_stress_tensor`. Symmetrize to strain-rate later.
    system_coords = current_coordinates(u, system)
    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        neighbor_system_coords = current_coordinates(u_neighbor, neighbor_system)
        foreach_point_neighbor(system, neighbor_system,
                               system_coords, neighbor_system_coords, semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            volume_b = hydrodynamic_mass(neighbor_system, neighbor) /
                       current_density(v_neighbor, neighbor_system, neighbor)
            v_a = viscous_velocity(v, system, particle)
            v_b = viscous_velocity(v_neighbor, neighbor_system, neighbor)

            v_diff = v_b - v_a

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            grad_v = v_diff * grad_kernel'

            # TODO: Laha et al. 2025: eq. 9 why do they multiply with m_j^2?
            fluid_stress_tensor[particle] += volume_b * grad_v
        end
    end

    @threaded semi for particle in each_integrated_particle(system)
        rho_a = current_density(v, system, particle)
        grad_v = fluid_stress_tensor[particle]

        S = (grad_v + grad_v') / 2
        invariant = sum(S .* S) # S:S
        S_mag = sqrt(2 * invariant) # |S|

        mu_T = rho_a * (smagorinsky_constant * smallest_length_scale)^2 * S_mag

        term1 = 2 .* S
        term2 = -(2 / 3) * tr(S) .* I(ndims(system))
        term3 = -(2 / 3) * isotropic_constant * smallest_length_scale^2 * (S_mag^2) .*
                I(ndims(system))

        fluid_stress_tensor[particle] = mu_T .* (term1 + term2 + term3)
    end

    return system
end
