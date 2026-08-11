@doc raw"""
    MarroneMLSKernel{NDIMS}()

The Moving Least-Squares Kernel by Marrone et al. is used to compute the pressure of dummy particles for `MarronePressureExtrapolation`.
"""

struct MarroneMLSKernel{NDIMS, IK, MI} <: AbstractSmoothingKernel{NDIMS}
    inner_kernel::IK
    momentum_inv::MI
end

function MarroneMLSKernel(inner_kernel::AbstractSmoothingKernel{NDIMS},
                          n_boundary_particles, ELTYPE) where {NDIMS}
    momentum_inv = zeros(SMatrix{NDIMS+1, NDIMS+1, ELTYPE, (NDIMS+1)^2},
                         n_boundary_particles)

    return MarroneMLSKernel{NDIMS, typeof(inner_kernel), typeof(momentum_inv)}(inner_kernel,
                                                                               momentum_inv)
end

@inline function boundary_kernel_marrone(marrone_kernel::MarroneMLSKernel{NDIMS},
                                         smoothing_length,
                                         particle, pos_diff, distance) where {NDIMS}
    (; inner_kernel, momentum_inv) = marrone_kernel
    ELTYPE = typeof(distance)
    E = SVector(one(ELTYPE), ntuple(_ -> zero(ELTYPE), Val(NDIMS))...)
    kernel_weight = kernel(inner_kernel, distance, smoothing_length)
    basis_particle_neighbor = vcat(SVector(one(ELTYPE)), -pos_diff)
    M_inv = momentum_inv[particle]

    return dot((M_inv * E), (basis_particle_neighbor * kernel_weight))
end

function compute_momentum(marrone_kernel::MarroneMLSKernel,
                          boundary_system,
                          fluid_system,
                          boundary_coords, fluid_coords, v_fluid_system, semi,
                          smoothing_length, particle)
    (; inner_kernel) = marrone_kernel

    backend = semi.parallelization_backend
    NDIMS = ndims(boundary_system)
    ELTYPE = eltype(boundary_coords)
    neighborhood_search = get_neighborhood_search(boundary_system, fluid_system, semi)

    # Initialize the momentum with zero
    momentum_particle = Ref(zero(SMatrix{NDIMS+1, NDIMS+1, ELTYPE, (NDIMS+1)^2}))

    foreach_neighbor(boundary_coords, fluid_coords,
                     neighborhood_search, backend,
                     particle) do particle, neighbor,
                                  pos_diff, distance
        basis_neighbor = vcat(SVector(one(ELTYPE)), -pos_diff)

        kernel_weight = TrixiParticles.kernel(inner_kernel, distance, smoothing_length)
        density_neighbor = current_density(v_fluid_system, fluid_system, neighbor)
        volume_neighbor = !iszero(density_neighbor) ?
                          hydrodynamic_mass(fluid_system, neighbor) / density_neighbor :
                          zero(ELTYPE)

        momentum_particle[] += basis_neighbor * basis_neighbor' * kernel_weight *
                               volume_neighbor
    end

    return momentum_particle[]
end

@inline compact_support(kernel::MarroneMLSKernel,
                        h) = compact_support(kernel.inner_kernel, h)

@inline function boundary_pressure_extrapolation!(parallel::Val{true},
                                                  boundary_model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                                                  system::AbstractBoundarySystem,
                                                  neighbor_system::AbstractFluidSystem,
                                                  system_coords, neighbor_coords, v,
                                                  v_neighbor_system,
                                                  semi)
    (; pressure, cache, viscosity, density_calculator, smoothing_kernel,
     smoothing_length) = boundary_model
    (; interpolation_coords, _pressure) = cache
    (; momentum_inv) = smoothing_kernel

    NDIMS = ndims(system)
    ELTYPE = eltype(system_coords)

    # TODO: Check if the scaling of the tolerance here is really necessary,
    # to check whether we can invert the momentum or not. 
    tolerance = ELTYPE(1e-9) * smoothing_length^(2 * NDIMS)
    for particle in eachparticle(system)
        momentum_particle = compute_momentum(smoothing_kernel, system, neighbor_system,
                                             interpolation_coords,
                                             neighbor_coords, v_neighbor_system, semi,
                                             smoothing_length, particle)
        momentum_inv[particle] = abs(det(momentum_particle)) < tolerance ?
                                 SMatrix{NDIMS+1, NDIMS+1, ELTYPE}(I) :
                                 inv(momentum_particle)
    end

    # Loop over all pairs of interpolation points and fluid particles within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, interpolation_coords,
                           neighbor_coords,
                           semi) do particle, neighbor,
                                    pos_diff, distance
        boundary_pressure_inner!(boundary_model, density_calculator, system,
                                 neighbor_system, system_coords, neighbor_coords, v,
                                 v_neighbor_system, semi, particle,
                                 neighbor,
                                 pos_diff, distance, viscosity, cache, _pressure)
    end

    # Copy the updated pressure values from the buffer
    pressure .= _pressure
end

@inline function boundary_pressure_inner!(boundary_model,
                                          boundary_density_calculator::MarronePressureExtrapolation,
                                          system, neighbor_system::AbstractFluidSystem,
                                          system_coords, neighbor_coords, v,
                                          v_neighbor_system, semi, particle, neighbor,
                                          pos_diff,
                                          distance, viscosity, cache, pressure)
    (; smoothing_kernel, smoothing_length) = boundary_model
    (; interpolation_coords) = boundary_model.cache

    kernel_weight = boundary_kernel_marrone(smoothing_kernel, smoothing_length,
                                            particle, pos_diff, distance)
    neighbor_density = current_density(v_neighbor_system, neighbor_system, neighbor)
    neighbor_volume = neighbor_density != 0 ?
                      hydrodynamic_mass(neighbor_system, neighbor) / neighbor_density : 0

    neighbor_pressure = current_pressure(v_neighbor_system, neighbor_system, neighbor)

    # Hydrostatic pressure term 
    density_neighbor = current_density(v_neighbor_system, neighbor_system, neighbor)
    resulting_acceleration = acceleration_source(neighbor_system) -
                             current_acceleration(system, particle)
    r_boundary = extract_svector(system_coords, system, particle)
    r_interpolation = extract_svector(interpolation_coords, system, particle)
    pos_diff_boundary = r_boundary - r_interpolation
    hydrostatic_pressure = dot(resulting_acceleration, density_neighbor * pos_diff_boundary)

    pressure[particle] += (neighbor_pressure + hydrostatic_pressure) * kernel_weight *
                          neighbor_volume
    cache.volume[particle] += kernel_weight * neighbor_volume

    # Update the boundary particle velocity
    interpolate_fluid_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                                kernel_weight * neighbor_volume, particle, neighbor)
end

function compute_marrone_density!(boundary_model, system, v, particle)
    (; pressure, state_equation, cache, viscosity) = boundary_model
    (; volume, density) = cache

    # The summation is only over fluid particles, thus the volume stays zero when a boundary
    # particle isn't surrounded by fluid particles.
    # Check the volume to avoid NaNs in pressure and velocity.
    particle_volume = volume[particle]
    if @inbounds particle_volume > eps()
        # To impose no-slip condition
        compute_wall_velocity!(viscosity, system, v, particle)
    end

    # Limit pressure to be non-negative to avoid attractive forces between fluid and
    # boundary particles at free surfaces (sticking artifacts).
    @inbounds pressure[particle] = max(pressure[particle], 0)

    # Apply inverse state equation to compute density (not used with EDAC)
    inverse_state_equation!(density, state_equation, pressure, particle)
end
