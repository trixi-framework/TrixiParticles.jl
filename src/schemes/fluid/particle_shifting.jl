struct ParticleShiftingSun2019{ELTYPE}
    v_max::ELTYPE
end

@inline Base.eltype(::ParticleShiftingSun2019{ELTYPE}) where {ELTYPE} = ELTYPE

# No shifting for a system by default
@inline particle_shifting(system) = nothing

create_cache_shifting(particle_shifting, NDIMS, nparticles) = (;)

function create_cache_shifting(particle_shifting::ParticleShiftingSun2019, NDIMS,
                               nparticles)
    # Create a cache for the particle shifting
    delta_v = zeros(eltype(particle_shifting), NDIMS, nparticles)
    return (; delta_v)
end

@inline function dv_particle_shifting(particle_shifting,
                                      system, neighbor_system,
                                      v_system, v_neighbor_system, particle,
                                      neighbor, m_b, rho_b, grad_kernel)
    return zero(grad_kernel)
end

@propagate_inbounds function dv_particle_shifting(::ParticleShiftingSun2019,
                                                  system, neighbor_system,#::FluidSystem,
                                                  v_system, v_neighbor_system, particle,
                                                  neighbor,
                                                  m_b, rho_b, grad_kernel)
    delta_v_a = delta_v(system, particle)
    delta_v_b = delta_v(neighbor_system, neighbor)

    v_a = current_velocity(v_system, system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    tensor_product = v_a * delta_v_a' + v_b * delta_v_b'
    return m_b / rho_b * (tensor_product * grad_kernel + v_a * dot(delta_v_a - delta_v_b, grad_kernel))
end

@inline function shifting_continuity_equation!(dv, particle_shifting, v_a, v_b,
                                               m_b, rho_a, rho_b, system, neighbor_system,
                                               particle, neighbor, grad_kernel)
    return dv
end

# `BoundarySPHSystem` and `TotalLagrangianSPHSystem` don't have a `delta_v`
@inline function shifting_continuity_equation!(dv, ::ParticleShiftingSun2019, v_a, v_b,
                                               m_b, rho_a, rho_b, system, neighbor_system,
                                               particle, neighbor, grad_kernel)
    delta_v_diff = rho_a * delta_v(system, particle)
    rho_v = rho_a * delta_v(system, particle)
    dv[end, particle] += m_b / rho_b * dot(delta_v_diff + rho_v, grad_kernel)

    return dv
end

@inline function shifting_continuity_equation!(dv, ::ParticleShiftingSun2019, v_a, v_b,
                                               m_b, rho_a, rho_b, system,
                                               neighbor_system::FluidSystem,
                                               particle, neighbor, grad_kernel)
    delta_v_diff = rho_a * (delta_v(system, particle) - delta_v(neighbor_system, neighbor))
    rho_v = rho_a * delta_v(system, particle) + rho_b * delta_v(neighbor_system, neighbor)
    dv[end, particle] += m_b / rho_b * dot(delta_v_diff + rho_v, grad_kernel)

    return dv
end

@propagate_inbounds function delta_v(system, particle)
    return delta_v(system, particle_shifting(system), particle)
end

@propagate_inbounds function delta_v(system, ::ParticleShiftingSun2019, particle)
    return extract_svector(system.cache.delta_v, system, particle)
end

# Zero when no particle shifting is used
@inline function delta_v(system, particle_shifting, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

function update_shifting!(system, particle_shifting, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_shifting!(system, particle_shifting::ParticleShiftingSun2019, v, u, v_ode,
                          u_ode, semi, t)
    (; delta_v) = system.cache
    (; v_max) = particle_shifting

    set_zero!(delta_v)

    Wdx = smoothing_kernel(system, particle_spacing(system, 1), 1)
    h = smoothing_length(system, 1)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi;
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            rho_b = current_density(v_neighbor, neighbor_system, neighbor)

            kernel = smoothing_kernel(system, distance, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            # According to p. 29 below Eq. 9
            R = 2 // 10
            n = 4

            # Eq. 7 in Sun et al. (2017).
            # CFL * Ma can be rewritten as Δt * v_max / h (see p. 29, right above Eq. 9).
            delta_v_ = -v_max * 2 * h * (1 + R * (kernel / Wdx)^n) * m_b / rho_b *
                       grad_kernel

            # Write into the buffer
            for i in eachindex(delta_v_)
                @inbounds delta_v[i, particle] += delta_v_[i]
            end
        end
    end

    return system
end
