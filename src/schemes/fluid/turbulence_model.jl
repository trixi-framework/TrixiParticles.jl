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
        cache = (; surface_normals=surface_normals,
                 neighbor_count=zeros(Int, nparticles(initial_condition)))

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
            # Use `viscous_velocity` to ensure continuous velocity gradients at boundaries,
            # unlike Laha et al. (2024) who employ mDBC (English et al., 2020) with zero boundary velocities.
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

# The stress tensor formulation presented in Laha et al. (2024) and Dalrymple et al. (2006)
# is not fully consistent and requires clarification before implementation:
#
# (1) The laminar viscous stress `tau_lam = 2 * mu * S`
#     is not explicitly included in the presented stress formulation. Why?
#
# (2) The isotropic SPS term reported in the paper is pressure-like and does not
#     contribute to the wall shear stress after tangential projection. Due to
#     ambiguities in its dimensional consistency and definition, it is treated
#     separately and can be disabled (isotropic_constant=0.0) without affecting WSS results.
#
# (3) When the stress vector is interpolated to the boundary particles, it is not
#     evaluated exactly at the physical wall location. This introduces a spatial
#     offset that may affect the accuracy of wall shear stress calculations.
#
# (4) The velocity gradient is not computed from a continuous velocity field at
#     the boundary, since the modified dynamic boundary condition (mDBC) approach
#     (English et al., 2020) enforces zero velocity on boundary particles. This
#     discontinuity can lead to inaccurate gradient estimates near walls.
#
# (5) The relationship between equation (15) in Laha et al. and equation (10) in Dalrymple et al.
#     is unclear. The formulations appear to be inconsistent.
#
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
        traction = stress_tensor[particle] * n  # traction vector
        tn = dot(n, traction)                   # normal component

        turbulence_model.field_variables.stress_vectors[particle] = traction - tn * n
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

            traction = stress_tensor_extrapolated * n  # traction vector
            tn = dot(n, traction)                      # normal component
            stress_vectors[point] = traction - tn * n
        end
    end

    return turbulence_model
end

function calculate_surface_normals!(system, turbulence_model, v, u, semi)
    (; surface_normals, neighbor_count) = turbulence_model.cache

    fill!(surface_normals, zero(eltype(surface_normals)))
    fill!(neighbor_count, zero(eltype(system)))

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
        neighbor_count[particle] += 1
    end

    compact_support_ = compact_support(system, system)
    particle_spacing = first(system.initial_condition.particle_spacing)
    max_neighbors = ideal_neighbor_count(Val(ndims(system)), particle_spacing,
                                         compact_support_) * 0.7

    @threaded semi for particle in each_integrated_particle(system)
        if neighbor_count[particle] < max_neighbors
            surface_normals[particle] = normalize(surface_normals[particle])
        end
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
