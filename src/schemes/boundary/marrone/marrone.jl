@doc raw"""
    MarroneMLSKernel{NDIMS}()

The Moving Least-Squares Kernel by Marrone et al. is used to compute the pressure of dummy particles for `MarronePressureExtrapolation`.
"""

struct MarroneMLSKernel{NDIMS} <: SmoothingKernel{NDIMS}
    inner_kernel :: SmoothingKernel
    basis        :: Array{Float64, 3}
    momentum     :: Array{Float64, 3}
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
                                                  boundary_model::BoundaryModelDummyParticles{MarronePressureExtrapolation,
                                                                                              ELTYPE,
                                                                                              VECTOR,
                                                                                              SE,
                                                                                              K,
                                                                                              V,
                                                                                              COR,
                                                                                              C},
                                                  system, neighbor_system::FluidSystem,
                                                  system_coords, neighbor_coords, v,
                                                  v_neighbor_system,
                                                  semi) where {ELTYPE, VECTOR, SE, K, V,
                                                               COR, C}
    (; pressure, cache, viscosity, density_calculator, smoothing_kernel,
     smoothing_length) = boundary_model
    (; normals) = system.initial_condition
    interpolation_coords = system_coords + (2 * normals) # Need only be computed once -> put into cache 

    compute_basis_marrone(smoothing_kernel, system, neighbor_system, system_coords,
                          neighbor_coords, semi)
    compute_momentum_marrone(smoothing_kernel, system, neighbor_system, system_coords,
                             neighbor_coords, v_neighbor_system, semi,
                             smoothing_length)

    # Loop over all pairs of interpolation points and fluid particles within the kernel cutoff
    foreach_point_neighbor(system, neighbor_system, interpolation_coords, neighbor_coords,
                           semi) do particle, neighbor,
                                    pos_diff, distance
        boundary_pressure_inner!(boundary_model, density_calculator, system,
                                 neighbor_system, v, v_neighbor_system, particle, neighbor,
                                 pos_diff, distance, viscosity, cache, pressure)
    end

    # Loop over all boundary particle
    for particle in eachparticle(system)
        f = neighbor_system.acceleration
        particle_density = isnan(current_density(v, system, particle)) ? 0 :
                           current_density(v, system, particle) # This can return NaN 
        particle_boundary_distance = norm(normals[:, particle]) # distance from boundary particle to the boundary
        particle_normal = particle_boundary_distance != 0 ?
                          normals[:, particle] / particle_boundary_distance :
                          zeros(size(normals[:, particle])) # normal unit vector to the boundary

        # Checked everything here for NaN's except the dot()
        pressure[particle] += 2 * particle_boundary_distance * particle_density *
                              dot(f, particle_normal)
    end
end

@inline function boundary_pressure_inner!(boundary_model,
                                          boundary_density_calculator::MarronePressureExtrapolation,
                                          system, neighbor_system::FluidSystem, v,
                                          v_neighbor_system, particle, neighbor, pos_diff,
                                          distance, viscosity, cache, pressure)
    (; smoothing_kernel, smoothing_length) = boundary_model
    neighbor_pressure = current_pressure(v_neighbor_system, neighbor_system, neighbor)
    kernel_weight = boundary_kernel_marrone(smoothing_kernel, particle, neighbor, distance,
                                            smoothing_length)
    neighbor_density = current_density(v_neighbor_system, neighbor_system, neighbor)
    neighbor_volume = neighbor_density != 0 ?
                      hydrodynamic_mass(neighbor_system, neighbor) / neighbor_density : 0

    pressure[particle] += neighbor_pressure * kernel_weight * neighbor_volume
end
