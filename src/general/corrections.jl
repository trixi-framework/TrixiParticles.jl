# Sorted in order of computational cost

@doc raw"""
    AkinciFreeSurfaceCorrection(rho0)

Free surface correction according to Akinci et al. (2013).
At a free surface, the mean density is typically lower than the reference density,
resulting in reduced surface tension and viscosity forces.
The free surface correction adjusts the viscosity, pressure, and surface tension forces
near free surfaces to counter this effect.
It's important to note that this correlation is unphysical and serves as an approximation.
The computation time added by this method is about 2-3%.

# Arguments
- `rho0`: Reference density.

## References
- Akinci, N., Akinci, G., & Teschner, M. (2013).
  "Versatile Surface Tension and Adhesion for SPH Fluids".
  ACM Transactions on Graphics (TOG), 32(6), 182.
  [doi: 10.1145/2508363.2508405](https://doi.org/10.1145/2508363.2508405)
"""
struct AkinciFreeSurfaceCorrection{ELTYPE}
    rho0::ELTYPE

    function AkinciFreeSurfaceCorrection(rho0)
        ELTYPE = eltype(rho0)
        return new{ELTYPE}(rho0)
    end
end

# `rho_mean` is the mean density of the fluid, which is used to determine correction values near the free surface.
#  Return a tuple `(viscosity_correction, pressure_correction, surface_tension_correction)` representing the correction terms.
@inline function free_surface_correction(correction::AkinciFreeSurfaceCorrection,
                                         particle_system, rho_mean)
    # Equation 4 in ref
    k = correction.rho0 / rho_mean

    # Viscosity, pressure, surface_tension
    return k, 1.0, k
end

@inline function free_surface_correction(correction, particle_system, rho_mean)
    return 1.0, 1.0, 1.0
end

@doc raw"""
    ShepardKernelCorrection()

Kernel correction uses Shepard interpolation to obtain a 0-th order accurate result, which
was first proposed by Li et al.

The kernel correction coefficient is determined by
```math
c(x) = \sum_{b=1}^{N} V_b W_b(x)
```

This correction is applied with SummationDensity to correct the density and leads to an improvement
as especially for free surfaces.

# Notes
- Also referred to as 0th order correction (2D: +5-6% computational time)


## References:
- J. Bonet, T.-S.L. Lok.
  "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Computer Methods in Applied Mechanics and Engineering 180 (1999), pages 97-115.
  [doi: 10.1016/S0045-7825(99)00051-1](https://doi.org/10.1016/S0045-7825(99)00051-1)
- Mihai Basa, Nathan Quinlan, Martin Lastiwka.
  "Robustness and accuracy of SPH formulations for viscous flow".
  In: International Journal for Numerical Methods in Fluids 60 (2009), pages 1127-1148.
  [doi: 10.1002/fld.1927](https://doi.org/10.1002/fld.1927)
-  Shaofan Li, Wing Kam Liu.
  "Moving least-square reproducing kernel method Part II: Fourier analysis".
  In: Computer Methods in Applied Mechanics and Engineering 139 (1996), pages 159-193.
  [doi:10.1016/S0045-7825(96)01082-1](https://doi.org/10.1016/S0045-7825(96)01082-1)
"""
struct ShepardKernelCorrection end

@doc raw"""
    KernelGradientCorrection()

Kernel gradient correction uses Shepard interpolation to obtain a 0-th order accurate result, which
was first proposed by Li et al. This can be further extended to obtain a kernel corrected gradient
as shown by Basa et al.

The kernel correction coefficient is determined by
```math
c(x) = \sum_{b=1}^{N} V_b W_b(x)
```
The gradient of corrected kernel is determined by
```math
\nabla \tilde(W)_{b}(x) = \frac{\naba W_{b}(x) - \gamma(x)}{\sum_{b=1}^{N} V_b W_b(x)}
\gamma(x) = \frac{\sum_{b=1}^{N} V_b \nabla W_b(x)}{\sum_{b=1}^{N} V_b W_b(x)}
```

This correction can be applied with SummationDensity and ContinuityDensity which leads to an improvement
especially for free surfaces.

# Notes
- This only works when the boundary model uses `SummationDensity` (yet).
- It is also referred to as 0th order correction.
- In 2D, we can expect an increase of about 10-15% in computation time.


## References:
- J. Bonet, T.-S.L. Lok.
  "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Computer Methods in Applied Mechanics and Engineering 180 (1999), pages 97-115.
  [doi: 10.1016/S0045-7825(99)00051-1](https://doi.org/10.1016/S0045-7825(99)00051-1)
- Mihai Basa, Nathan Quinlan, Martin Lastiwka.
  "Robustness and accuracy of SPH formulations for viscous flow".
  In: International Journal for Numerical Methods in Fluids 60 (2009), pages 1127-1148.
  [doi: 10.1002/fld.1927](https://doi.org/10.1002/fld.1927)
- S.F. Li, W.K. Liu, "Moving least square Kernel Galerkin method (II) Fourier analysis",
  Computer Methods in Applied Mechanics and Engineering., 139 (1996) pages 159ff
  [doi:10.1016/S0045-7825(96)01082-1] (https://doi.org/10.1016/S0045-7825(96)01082-1).
"""
struct KernelGradientCorrection end

function kernel_correction_coefficient(system, particle)
    return system.cache.kernel_correction_coefficient[particle]
end

function compute_correction_values!(system, system_index, v, u, v_ode, u_ode, semi,
                                    density_calculator, correction)
    return system
end

function compute_correction_values!(system, system_index, v, u, v_ode, u_ode, semi,
                                    ::SummationDensity, ::ShepardKernelCorrection)
    return compute_shepard_coeff!(system, system_index, v, u, v_ode, u_ode, semi,
                                  system.cache.kernel_correction_coefficient)
end

function compute_shepard_coeff!(system, system_index, v, u, v_ode, u_ode, semi,
                                kernel_correction_coefficient)
    (; systems, neighborhood_searches) = semi

    set_zero!(kernel_correction_coefficient)

    # Use all other systems for the density summation
    @trixi_timeit timer() "compute correction value" foreach_enumerate(systems) do (neighbor_system_index,
                                                                                    neighbor_system)
        u_neighbor_system = wrap_u(u_ode, neighbor_system_index, neighbor_system,
                                   semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system_index, neighbor_system,
                                   semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        neighborhood_search = neighborhood_searches[system_index][neighbor_system_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff
        for_particle_neighbor(system, neighbor_system, system_coords,
                              neighbor_coords,
                              neighborhood_search) do particle, neighbor, pos_diff, distance
            rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            volume = m_b / rho_b

            kernel_correction_coefficient[particle] += volume *
                                                       smoothing_kernel(system, distance)
        end
    end

    return kernel_correction_coefficient
end

function dw_gamma(system, particle)
    return extract_svector(system.cache.dw_gamma, system, particle)
end

function compute_correction_values!(system, system_index, v, u, v_ode, u_ode, semi,
                                    ::Union{SummationDensity, ContinuityDensity},
                                    ::KernelGradientCorrection)
    (; systems, neighborhood_searches) = semi
    (; cache) = system
    (; kernel_correction_coefficient, dw_gamma) = cache

    set_zero!(kernel_correction_coefficient)
    set_zero!(dw_gamma)

    # Use all other systems for the density summation
    @trixi_timeit timer() "compute correction value" foreach_enumerate(systems) do (neighbor_system_index,
                                                                                    neighbor_system)
        u_neighbor_system = wrap_u(u_ode, neighbor_system_index, neighbor_system,
                                   semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system_index, neighbor_system,
                                   semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        neighborhood_search = neighborhood_searches[system_index][neighbor_system_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff
        for_particle_neighbor(system, neighbor_system, system_coords,
                              neighbor_coords,
                              neighborhood_search) do particle, neighbor, pos_diff, distance
            rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            volume = m_b / rho_b

            kernel_correction_coefficient[particle] += volume *
                                                       smoothing_kernel(system, distance)
            if distance > sqrt(eps())
                tmp = volume * smoothing_kernel_grad(system, pos_diff, distance)
                for i in axes(dw_gamma, 1)
                    dw_gamma[i, particle] += tmp[i]
                end
            end
        end
    end

    for particle in eachparticle(system), i in axes(dw_gamma, 1)
        dw_gamma[i, particle] /= kernel_correction_coefficient[particle]
    end
end

"""
    GradientCorrection()

Compute the corrected gradient of particle interactions based on their relative positions.
The corrected gradient is stored in the `corr_matrix`.

# Mathematical Details

Given the standard SPH representation of a gradient of a field A at particle i is given by:

```math
\\nabla A_i = \\sum_j m_j \\frac{A_j - A_i}{\\rho_j} \\nabla W_{ij}
```

Where:
- m_j is the mass of particle j
- rho_j is the density of particle j
- W_{ij} is the SPH kernel function, describing the influence of particle j on particle i

The gradient correction, as commonly proposed, involves multiplying this gradient with a correction matrix L :

```math
\\nabla A_i^{corrected} = L_i \\nabla A_i
```

The correction matrix L_i is computed based on the provided particle configuration,
aiming to make the corrected gradient more accurate, especially near domain boundaries.
This matrix is usually symmetric and dependent on the spatial dimension of the system.

# Notes:
- Stability issues as especially when particles separate into small clusters.
- Doubles the computational effort.

## References:
- J. Bonet, T.-S.L. Lok.
  "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Computer Methods in Applied Mechanics and Engineering 180 (1999), pages 97-115.
  [doi: 10.1016/S0045-7825(99)00051-1](https://doi.org/10.1016/S0045-7825(99)00051-1)
- Mihai Basa, Nathan Quinlan, Martin Lastiwka.
  "Robustness and accuracy of SPH formulations for viscous flow".
  In: International Journal for Numerical Methods in Fluids 60 (2009), pages 1127-1148.
  [doi: 10.1002/fld.1927](https://doi.org/10.1002/fld.1927)
"""
struct GradientCorrection end

function compute_gradient_correction_matrix!(corr_matrix, neighborhood_search, system,
                                            coordinates)
    (; mass, material_density) = system

    set_zero!(corr_matrix)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(system, system,
                          coordinates, coordinates,
                          neighborhood_search;
                          particles=eachparticle(system)) do particle, neighbor,
                                                             pos_diff,
                                                             distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        volume = mass[neighbor] / material_density[neighbor]

        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)
        result = volume * grad_kernel * pos_diff'

        @inbounds for j in 1:ndims(system), i in 1:ndims(system)
            corr_matrix[i, j, particle] -= result[i, j]
        end
    end

    @threaded for particle in eachparticle(system)
        L = correction_matrix(system, particle)
        result = inv(L)

        @inbounds for j in 1:ndims(system), i in 1:ndims(system)
            corr_matrix[i, j, particle] = result[i, j]
        end
    end

    return corr_matrix
end
