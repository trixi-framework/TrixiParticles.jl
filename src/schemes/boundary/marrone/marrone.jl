@doc raw"""
    MarroneMLSKernel{NDIMS}()

The Moving Least-Squares Kernel by Marrone et al. is used to compute the pressure of dummy particles for `MarronePressureExtrapolation`.
"""

struct MarroneMLSKernel{NDIMS} <: SmoothingKernel{NDIMS}
    inner_kernel  :: SmoothingKernel
    basis         :: Array{Float64, 3}
    momentum      :: Array{Float64, 3}
end

function MarroneMLSKernel(inner_kernel::SmoothingKernel{NDIMS},
                          n_boundary_particles, n_fluid_particles) where {NDIMS}
    basis = zeros(n_boundary_particles, n_fluid_particles, NDIMS+1) # Big sparse tensor
    momentum = zeros(n_boundary_particles, NDIMS+1, NDIMS+1)

    return MarroneMLSKernel{NDIMS}(inner_kernel, basis, momentum)
end

# Compute the Marrone MLS-Kernel for the points with indices i and j
@inline function boundary_kernel_marrone(smoothing_kernel::MarroneMLSKernel, i, j, distance,
                                         smoothing_length)
    (; inner_kernel, basis, momentum) = smoothing_kernel
    ndims = size(momentum, 2)
    canonical_vector = ndims == 3 ? [1.0, 0.0, 0.0] : [1.0, 0.0, 0.0, 0.0]

    kernel_weight = kernel(inner_kernel, distance, smoothing_length)
    M_inv = momentum[i, :, :]

    return dot((M_inv * canonical_vector), (basis[i, j, :] * kernel_weight))
end

function compute_basis_marrone(kernel::MarroneMLSKernel, system, neighbor_system,
                               system_coords, neighbor_coords, semi)
    (; basis) = kernel
    basis .= 0.0
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=eachparticle(system)) do particle, neighbor,
                                                           pos_diff, distance
        basis[particle, neighbor, :] = [1; -pos_diff] # We use -pos_diff since we need to calculate coords_neighbor - coords_particle 
    end
end

function compute_momentum_marrone(kernel::MarroneMLSKernel, system, neighbor_system,
                                  system_coords, neighbor_coords, v_neighbor_system, semi,
                                  smoothing_length)
    (; inner_kernel, basis, momentum) = kernel
    momentum .= 0.0

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=eachparticle(system)) do particle, neighbor,
                                                           pos_diff, distance
        kernel_weight = TrixiParticles.kernel(inner_kernel, distance, smoothing_length)
        density_neighbor = current_density(v_neighbor_system, neighbor_system, neighbor)
        volume_neighbor = density_neighbor != 0 ?
                          hydrodynamic_mass(neighbor_system, neighbor) / density_neighbor :
                          0
        momentum[particle, :,
                 :] += basis[particle, neighbor, :] .*
                       basis[particle, neighbor, :]' .*
                       kernel_weight .* volume_neighbor
    end

    n = size(momentum, 2)
    for particle in eachparticle(system)
        if abs(det(momentum[particle, :, :])) < 1.0f-9
            momentum[particle, :,
                     :] = Matrix{eltype(momentum[particle, :, :])}(I, n, n) # Create identity matrix of fitting size and type
        else
            momentum[particle, :, :] = inv(momentum[particle, :, :])
        end
    end
end

# I dont know if this is correct...
@inline compact_support(kernel::MarroneMLSKernel,
                        h) = compact_support(kernel.inner_kernel, h)

struct DefaultKernel{NDIMS} <: SmoothingKernel{NDIMS} end

function kernel(kernel::DefaultKernel, r::Real, h)
    return 1
end

@inline function boundary_pressure_extrapolation!(parallel::Val{true},
                                                  boundary_model::BoundaryModelDummyParticles{MarronePressureExtrapolation},
                                                  system, neighbor_system::FluidSystem,
                                                  system_coords, neighbor_coords, v,
                                                  v_neighbor_system,
                                                  semi)
    (; pressure, cache, viscosity, density_calculator, smoothing_kernel,
     smoothing_length) = boundary_model
    (; interpolation_coords, _pressure) = cache

    compute_basis_marrone(smoothing_kernel, system, neighbor_system, interpolation_coords,
                          neighbor_coords, semi)
    compute_momentum_marrone(smoothing_kernel, system, neighbor_system,
                             interpolation_coords,
                             neighbor_coords, v_neighbor_system, semi,
                             smoothing_length)

    # Loop over all pairs of interpolation points and fluid particles within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, interpolation_coords, neighbor_coords,
                           semi) do particle, neighbor,
                                    pos_diff, distance
        boundary_pressure_inner!(boundary_model, density_calculator, system,
                                 neighbor_system, v, v_neighbor_system, particle, neighbor,
                                 pos_diff, distance, viscosity, cache, _pressure)
    end

    # Copy the updated pressure values from the buffer
    pressure .= _pressure
end

@inline function boundary_pressure_inner!(boundary_model,
                                          boundary_density_calculator::MarronePressureExtrapolation,
                                          system, neighbor_system::FluidSystem, v,
                                          v_neighbor_system, particle, neighbor, pos_diff,
                                          distance, viscosity, cache, pressure)
    (; smoothing_kernel, smoothing_length) = boundary_model
    kernel_weight = boundary_kernel_marrone(smoothing_kernel, particle, neighbor, distance,
                                            smoothing_length)
    # Update the pressure
    neighbor_pressure = current_pressure(v_neighbor_system, neighbor_system, neighbor)
    neighbor_density = current_density(v_neighbor_system, neighbor_system, neighbor)
    neighbor_volume = neighbor_density != 0 ?
                      hydrodynamic_mass(neighbor_system, neighbor) / neighbor_density : 0

    pressure[particle] += neighbor_pressure * kernel_weight * neighbor_volume

    # Update the boundary particle velocity
    # TODO: This method takes an additional parameter `neighbor_volume`, maybe unify all methods of this function 
    # to accept the same arguments
    compute_smoothed_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                               kernel_weight, particle, neighbor, neighbor_volume)
end

# Change the dispatched type of `viscosity` so that it also accepts Marrone
function compute_smoothed_velocity!(cache, viscosity, neighbor_system,
                                    v_neighbor_system, kernel_weight, particle, neighbor,
                                    neighbor_volume)
    neighbor_velocity = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    
    # CHECK: is the neighbor_volume term necessary?
    for dim in eachindex(neighbor_velocity)
        @inbounds cache.wall_velocity[dim, particle] += neighbor_velocity[dim] * kernel_weight * neighbor_volume
    end

    return cache
end

# For more, see `dummy_particles.jl`
function compute_boundary_density!(boundary_model::BoundaryModelDummyParticles{MarronePressureExtrapolation}, system, system_coords, particle)
    (; pressure, state_equation, cache, viscosity) = boundary_model
    (; volume, density) = cache

    # The summation is only over fluid particles, thus the volume stays zero when a boundary
    # particle isn't surrounded by fluid particles.
    # Check the volume to avoid NaNs in pressure and velocity.
    particle_volume = volume[particle]
    if @inbounds particle_volume > eps()
        # To impose no-slip condition
        compute_wall_velocity!(viscosity, system, system_coords, particle)
    end

    # Limit pressure to be non-negative to avoid attractive forces between fluid and
    # boundary particles at free surfaces (sticking artifacts).
    @inbounds pressure[particle] = max(pressure[particle], 0)

    # Apply inverse state equation to compute density (not used with EDAC)
    # CHECK: this computes the density based on the updated pressure, is this correct?
    inverse_state_equation!(density, state_equation, pressure, particle)
end

