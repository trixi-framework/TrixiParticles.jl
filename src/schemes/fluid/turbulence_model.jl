struct SPSTurbulenceModelDalrymple{ELTYPE, FV, C}
    smagorinsky_constant   :: ELTYPE
    isotropic_constant     :: ELTYPE
    smallest_length_scale  :: ELTYPE
    mu                     :: ELTYPE
    field_variables        :: FV
    only_wall_shear_stress :: Bool
    cache                  :: C
end

function SPSTurbulenceModelDalrymple(; smallest_length_scale, smagorinsky_constant=0.12,
                                     isotropic_constant=6.6e-4, dynamic_viscosity,
                                     only_wall_shear_stress=false)
    return SPSTurbulenceModelDalrymple(smagorinsky_constant, isotropic_constant,
                                       smallest_length_scale, dynamic_viscosity,
                                       nothing, only_wall_shear_stress, nothing)
end

function SPSTurbulenceModelDalrymple(initial_condition;
                                     smallest_length_scale=first(initial_condition.particle_spacing),
                                     smagorinsky_constant=0.12, isotropic_constant=6.6e-4,
                                     dynamic_viscosity, only_wall_shear_stress=false)
    ELTYPE = eltype(initial_condition)
    NDIMS = ndims(initial_condition)
    velocity_gradient_tensor = fill(zero(SMatrix{NDIMS, NDIMS, ELTYPE}),
                                    nparticles(initial_condition))
    strain_rate_tensor = fill(zero(SMatrix{NDIMS, NDIMS, ELTYPE}),
                              nparticles(initial_condition))
    stress_tensor = fill(zero(SMatrix{NDIMS, NDIMS, ELTYPE}),
                         nparticles(initial_condition))
    surface_normals = fill(zero(SVector{NDIMS, ELTYPE}), nparticles(initial_condition))

    field_variables = (velocity_gradient_tensor=velocity_gradient_tensor,
                       strain_rate_tensor=strain_rate_tensor, stress_tensor=stress_tensor)
    if only_wall_shear_stress
        stress_vectors = copy(surface_normals)
        sample_points = copy(initial_condition.coordinates)
        volume = zeros(eltype(initial_condition), nparticles(initial_condition))

        field_variables = (; stress_tensor=stress_tensor, stress_vectors=stress_vectors)
        cache = (; sample_points=sample_points, surface_normals=surface_normals,
                 volume=volume)
    else
        cache = (; surface_normals=surface_normals)

        # TODO: This is for debugging
        stress_vectors = copy(surface_normals)
        field_variables = (; field_variables..., stress_vectors=stress_vectors)
    end

    return SPSTurbulenceModelDalrymple(smagorinsky_constant, isotropic_constant,
                                       smallest_length_scale, dynamic_viscosity,
                                       field_variables, only_wall_shear_stress, cache)
end

function calculate_fluid_stress_tensor!(system::AbstractFluidSystem, turbulence_model,
                                        v_ode, u_ode, semi)
    (; velocity_gradient_tensor, strain_rate_tensor) = turbulence_model.field_variables

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    # We cannot calculate surface normals from the wall due to robustness issues with thin walls.
    # Normal vectors could toggle.
    # Instead, we compute fluid free surface normals and extrapolate them to the wall with reversed sign.
    calculate_surface_normals!(system, turbulence_model, v, u, semi)

    calculate_velocity_gradients!(system, velocity_gradient_tensor,
                                  v, u, v_ode, u_ode, semi)

    calculate_strain_rate!(system, velocity_gradient_tensor, strain_rate_tensor, semi)

    calculate_stress_tensor!(system, turbulence_model, v, semi)

    return system
end

function calculate_velocity_gradients!(system, velocity_gradient_tensor,
                                       v, u, v_ode, u_ode, semi)
    # Set zero
    fill!(velocity_gradient_tensor, zero(eltype(velocity_gradient_tensor)))

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
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            rho_b = current_density(v_neighbor, neighbor_system, neighbor)
            volume_b = m_b / rho_b

            # We use the `viscous_velocity` to ensure a continuous velocity gradient at the boundary
            v_a = viscous_velocity(v, system, particle)
            v_b = viscous_velocity(v_neighbor, neighbor_system, neighbor)

            v_diff = v_b - v_a

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            grad_v = v_diff * grad_kernel'

            # TODO: Laha et al. 2025: eq. 9 why do they multiply with m_j^2?
            velocity_gradient_tensor[particle] += volume_b * grad_v
        end
    end

    return system
end

function calculate_strain_rate!(system, velocity_gradient_tensor, strain_rate_tensor, semi)
    @threaded semi for particle in each_integrated_particle(system)
        grad_v = velocity_gradient_tensor[particle]
        strain_rate_tensor[particle] = (grad_v + grad_v') / 2
    end

    return system
end

# The stress tensor formulation presented in Laha et al. (2024) is not fully
# consistent and requires clarification before implementation:
#
# (1) The velocity-gradient expression (Eq. 9 in the paper) contains an additional
#     factor m_j, which leads to incorrect physical dimensions. The standard SPH
#     gradient operator uses only (m_j / rho_j) and is adopted here.
#
# (2) The stress tensor reported in the paper corresponds to a sub-particle-scale
#     (SPS) stress contribution only. The laminar viscous stress
#         tau_lam = 2 * mu * S
#     is not explicitly included in the presented stress formulation. Why?
#
# (3) For wall shear stress (WSS) evaluation, however, the total deviatoric stress
#     must be reconstructed. Therefore, the present implementation explicitly
#     combines the laminar viscous stress and the SPS stress contribution:
#
#         sigma = tau_lam + tau_dev (+ optional isotropic term)
#
# (4) The isotropic SPS term reported in the paper is pressure-like and does not
#     contribute to the wall shear stress after tangential projection. Due to
#     ambiguities in its dimensional consistency and definition, it is treated
#     separately and can be disabled (isotropic_constant=0.0) without affecting WSS results.
function calculate_stress_tensor!(system, turbulence_model, v, semi)
    (; smagorinsky_constant, isotropic_constant,
    smallest_length_scale, mu, field_variables) = turbulence_model
    (; strain_rate_tensor, stress_tensor) = field_variables

    @threaded semi for particle in each_integrated_particle(system)
        rho_a = current_density(v, system, particle)

        S = strain_rate_tensor[particle]

        S_mag = sqrt(2 * sum(S .* S))  # scalar |S|

        # Eddy viscosity (Smagorinsky model)
        mu_T = rho_a * (smagorinsky_constant * smallest_length_scale)^2 * S_mag

        # Deviatoric shear stress
        tau_dev = (2 * mu_T) .* (S .- (tr(S) / 3) .* I(ndims(system)))

        # Isotropic turbulent part
        tau_iso = -(2 / 3) * rho_a * isotropic_constant *
                  smallest_length_scale^2 * S_mag^2 .* I(ndims(system))

        # TODO: is this correct?
        # laminar stress
        tau_lam = (2 * mu) .* S

        stress_tensor[particle] = tau_lam + tau_dev + tau_iso

        # TODO: This is for debugging
        # `n` must point inside the fluid domain
        n = -turbulence_model.cache.surface_normals[particle]

        turbulence_model.field_variables.stress_vectors[particle] = stress_tensor[particle] *
                                                                    tangent_from_normal(n)
    end

    return system
end

function calculate_wall_shear_stress!(turbulence_model,
                                      turbulence_model_neighbor,
                                      neighbor_system::AbstractFluidSystem,
                                      v_ode, u_ode, semi)
    (; stress_tensor, stress_vectors) = turbulence_model.field_variables
    (; sample_points, surface_normals, volume) = turbulence_model.cache

    # Set zero
    fill!(stress_tensor, zero(eltype(stress_tensor)))
    fill!(surface_normals, zero(eltype(surface_normals)))
    fill!(stress_vectors, zero(eltype(stress_vectors)))
    fill!(volume, zero(eltype(volume)))

    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    v_neighbor = wrap_v(v_ode, neighbor_system, semi)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    nhs = get_neighborhood_search(neighbor_system, semi)
    foreach_point_neighbor(sample_points, neighbor_coords, nhs;
                           parallelization_backend=semi.parallelization_backend) do point,
                                                                                    neighbor,
                                                                                    pos_diff,
                                                                                    distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = current_density(v_neighbor, neighbor_system, neighbor)
        kernel_weight = smoothing_kernel(neighbor_system, distance, neighbor) * m_b / rho_b

        stress_tensor_neigbor = turbulence_model_neighbor.field_variables.stress_tensor
        surface_normals_neighbor = turbulence_model_neighbor.cache.surface_normals

        surface_normals[point] += surface_normals_neighbor[neighbor]
        stress_tensor[point] += kernel_weight * stress_tensor_neigbor[neighbor]
        volume[point] += kernel_weight
    end

    @threaded default_backend(sample_points) for point in axes(sample_points, 2)
        # Check the volume to avoid NaNs
        if volume[point] > eps(eltype(volume))
            stress_tensor_extrapolated = stress_tensor[point] ./ volume[point]
            n = -normalize(surface_normals[point] / volume[point])

            stress_vectors[point] = stress_tensor_extrapolated * tangent_from_normal(n)
        end
    end

    return turbulence_model
end

function calculate_surface_normals!(system, turbulence_model, v, u, semi)
    (; surface_normals) = turbulence_model.cache

    fill!(surface_normals, zero(eltype(surface_normals)))

    system_coords = current_coordinates(u, system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(system, system, system_coords, system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        m_b = hydrodynamic_mass(system, neighbor)
        rho_b = current_density(v, system, neighbor)
        kernel_grad_weight = smoothing_kernel_grad(system, pos_diff, distance, particle) *
                             m_b / rho_b

        surface_normals[particle] += kernel_grad_weight
    end

    @threaded semi for particle in each_integrated_particle(system)
        surface_normals[particle] = normalize(surface_normals[particle])
    end

    return system
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
