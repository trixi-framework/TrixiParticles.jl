abstract type DensityDiffusion end

struct DensityDiffusionMolteniColagrossi{ELTYPE} <: DensityDiffusion
    delta::ELTYPE
end

@inline function density_diffusion_psi(::DensityDiffusionMolteniColagrossi, rho_a, rho_b,
                                       pos_diff, distance, system, particle, neighbor)
    return 2 * (rho_a - rho_b) * pos_diff / distance^2
end

struct DensityDiffusionFerrari <: DensityDiffusion
    delta::Int

    # δ is always 1 in this formulation
    DensityDiffusionFerrari() = new(1)
end

@inline function density_diffusion_psi(::DensityDiffusionFerrari, rho_a, rho_b,
                                       pos_diff, distance, system, particle, neighbor)
    (; smoothing_length) = system

    return ((rho_a - rho_b) / 2smoothing_length) * pos_diff / distance
end

struct DensityDiffusionAntuono{NDIMS, ELTYPE} <: DensityDiffusion
    delta                       :: ELTYPE
    correction_matrix           :: Array{ELTYPE, 3} # [i, j, particle]
    normalized_density_gradient :: Array{ELTYPE, 2} # [i, particle]

    function DensityDiffusionAntuono(delta, initial_condition)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)
        correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS,
                                             nparticles(initial_condition))

        normalized_density_gradient = Array{ELTYPE, 2}(undef, NDIMS,
                                                       nparticles(initial_condition))

        new{NDIMS, ELTYPE}(delta, correction_matrix, normalized_density_gradient)
    end
end

@inline Base.ndims(::DensityDiffusionAntuono{NDIMS}) where {NDIMS} = NDIMS

@inline function density_diffusion_psi(density_diffusion::DensityDiffusionAntuono,
                                       rho_a, rho_b,
                                       pos_diff, distance, system, particle, neighbor)
    (; normalized_density_gradient) = density_diffusion

    normalized_gradient_a = extract_svector(normalized_density_gradient, system, particle)
    normalized_gradient_b = extract_svector(normalized_density_gradient, system, neighbor)

    return 2 *
           (rho_a - rho_b -
            0.5 * dot(normalized_gradient_a + normalized_gradient_b, pos_diff)) * pos_diff /
           distance^2
end

function update!(density_diffusion::DensityDiffusionAntuono, neighborhood_search,
                 v, u, system, semi)
    (; normalized_density_gradient) = density_diffusion

    # Compute correction matrix
    density_fun = @inline(particle->particle_density(v, system, particle))
    system_coords = current_coordinates(u, system)

    compute_gradient_correction_matrix!(density_diffusion.correction_matrix,
                                        neighborhood_search, system,
                                        system_coords, density_fun)

    # Compute normalized density gradient
    set_zero!(normalized_density_gradient)

    for_particle_neighbor(system, system, system_coords, system_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0
        distance < sqrt(eps()) && return

        rho_a = particle_density(v, system, particle)
        rho_b = particle_density(v, system, neighbor)

        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        L = correction_matrix(density_diffusion, particle)

        m_b = hydrodynamic_mass(system, neighbor)
        volume_b = m_b / rho_b

        normalized_gradient = (rho_b - rho_a) * L * grad_kernel * volume_b

        for i in eachindex(normalized_gradient)
            normalized_density_gradient[i, particle] += normalized_gradient[i]
        end
    end

    return density_diffusion
end

function update!(density_diffusion, neighborhood_search, v, u, system, semi)
    return density_diffusion
end
