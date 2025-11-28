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

function create_cache_turbulence(initial_condition, turbulence_model; boundary=false)
    ELTYPE = eltype(initial_condition)
    NDIMS = ndims(initial_condition)
    fluid_stress_tensor = fill(zero(SMatrix{NDIMS, NDIMS, ELTYPE}),
                               nparticles(initial_condition))

    if boundary
        surface_normals = fill(zero(SVector{NDIMS, ELTYPE}), nparticles(initial_condition))
        stress_vectors = copy(surface_normals)
        return (; fluid_stress_tensor, turbulence_model, surface_normals, stress_vectors)
    end
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

function update_turbulence_model!(system, ::Nothing, v, u, v_ode, u_ode, semi)
    return system
end

function update_turbulence_model!(system, ::SPSTurbulenceModelDalrymple,
                                  v, u, v_ode, u_ode, semi)
    @trixi_timeit timer() "calculate fluid stress tensor" begin
        calculate_fluid_stress_tensor!(system, v, u, v_ode, u_ode, semi)
    end

    return system
end

function update_turbulence_model!(system::AbstractBoundarySystem,
                                  ::SPSTurbulenceModelDalrymple,
                                  v, u, v_ode, u_ode, semi)
    @trixi_timeit timer() "extrapolate fluid stress tensor" begin
        extrapolate_fluid_stress_tensor!(system, v, u, v_ode, u_ode, semi)
    end

    return system
end

function calculate_fluid_stress_tensor!(system, v, u, v_ode, u_ode, semi)
    (; fluid_stress_tensor, turbulence_model) = system.cache
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
            # TODO: current or viscous velocity
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

        # Strain rate
        S = (grad_v + grad_v') / 2
        S_mag = sqrt(2 * (S * S))

        # Eddy viscosity
        mu_T = rho_a * (smagorinsky_constant * smallest_length_scale)^2 * S_mag

        term1 = 2 .* S
        term2 = -(2 / 3) * tr(S) .* I(ndims(system))
        term3 = -(2 / 3) * isotropic_constant * smallest_length_scale^2 * (S_mag^2) .*
                I(ndims(system))

        tau_a = mu_T .* (term1 + term2 + term3)

        fluid_stress_tensor[particle] = tau_a
    end

    return system
end

function extrapolate_fluid_stress_tensor!(system, v, u, v_ode, u_ode, semi)
    (; fluid_stress_tensor, surface_normals, stress_vectors) = system.cache
    (; volume) = system.boundary_model.cache

    set_zero!(volume)
    fill!(fluid_stress_tensor, zero(eltype(fluid_stress_tensor)))
    fill!(surface_normals, zero(eltype(surface_normals)))
    fill!(stress_vectors, zero(eltype(stress_vectors)))

    foreach_system(semi) do neighbor_system
        extrapolate_fluid_stress_tensor!(system, neighbor_system, v, u, v_ode, u_ode, semi)
    end

    @threaded semi for particle in eachparticle(system)
        # The summation is only over fluid particles, thus the volume stays zero when a boundary
        # particle isn't surrounded by fluid particles.
        # Check the volume to avoid NaNs in pressure and velocity.
        if @inbounds volume[particle] > eps()
            S_interpolated = fluid_stress_tensor[particle]
            n = -normalize(surface_normals[particle])
            surface_tangent = tangent_from_normal(n)

            stress_vectors[particle] = S_interpolated ./ volume[particle] * surface_tangent
        end
    end
end

@inline function extrapolate_fluid_stress_tensor!(system, neighbor_system, v, u,
                                                  v_ode, u_ode, semi)
    return system
end

function extrapolate_fluid_stress_tensor!(system, neighbor_system::AbstractFluidSystem,
                                          v, u, v_ode, u_ode, semi)
    (; boundary_model) = system
    (; fluid_stress_tensor, surface_normals) = system.cache

    system_coords = current_coordinates(u, system)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=eachparticle(system)) do particle, neighbor,
                                                           pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = current_density(v_neighbor, neighbor_system, neighbor)
        kernel_weight = smoothing_kernel(boundary_model, distance, particle) * m_b / rho_b
        kernel_grad_weight = smoothing_kernel_grad(system, pos_diff, distance, particle) *
                             m_b / rho_b

        surface_normals[particle] += kernel_grad_weight
        fluid_stress_tensor[particle] += kernel_weight *
                                         current_fluid_stress_tensor(neighbor_system,
                                                                     neighbor)
        @inbounds boundary_model.cache.volume[particle] += kernel_weight
    end
end

function tangent_from_normal(n::SVector{2})
    nx, ny = n
    return SVector(-ny, nx)
end

function tangent_from_normal(n::SVector{3})
    e = abs(n[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    t = cross(n, e)
    return normalize(t)
end
