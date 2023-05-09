@doc raw"""
    SolidParticleContainer(particle_coordinates, particle_velocities,
                           particle_masses, particle_material_densities,
                           hydrodynamic_density_calculator,
                           smoothing_kernel, smoothing_length,
                           young_modulus, poisson_ratio;
                           n_fixed_particles=0,
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                           penalty_force=nothing)

Container for particles of an elastic solid.

A Total Lagrangian framework is used wherein the governing equations are forumlated such that
all relevant quantities and operators are measured with respect to the
initial configuration (O’Connor & Rogers 2021, Belytschko et al. 2000).
The governing equations with respect to the initial configuration are given by:
```math
\frac{\mathrm{D}\bm{v}}{\mathrm{D}t} = \frac{1}{\rho_0} \nabla_0 \cdot \bm{P} + \bm{g},
```
where the zero subscript denotes a derivative with respect to the initial configuration
and $\bm{P}$ is the first Piola-Kirchhoff (PK1) stress tensor.

The discretized version of this equation is given by (O’Connor & Rogers 2021):
```math
\frac{\mathrm{d}\bm{v}_a}{\mathrm{d}t} = \sum_b m_{0b}
    \left( \frac{\bm{P}_a \bm{L}_{0a}}{\rho_{0a}^2} + \frac{\bm{P}_b \bm{L}_{0b}}{\rho_{0b}^2} \right)
    \nabla_{0a} W(\bm{X}_{ab}) + \frac{\bm{f}_a^{PF}}{m_{0a}} + \bm{g},
```
with
```math
\bm{L}_{0a} := \left( \sum_{b} \frac{m_{0b}}{\rho_{0b}} \nabla_{0a} W(\bm{X}_{ab}) \bm{X}_{ab}^T \right)^{-1} \in \R^{d \times d}.
```
The subscripts $a$ and $b$ denote quantities of particle $a$ and $b$, respectively.
The zero subscript on quantities denotes that the quantity is to be measured in the initial configuration.
The difference in the initial coordinates is denoted by $\bm{X}_{ab} = \bm{X}_a - \bm{X}_b$,
the difference in the current coordinates is denoted by $\bm{x}_{ab} = \bm{x}_a - \bm{x}_b$.

For the computation of the PK1 stress tensor, the deformation gradient $\bm{J}$ is computed per particle as
```math
\bm{J}_a = \sum_b \frac{m_{0b}}{\rho_{0b}} \bm{x}_{ba} (\bm{L}_{0a}\nabla_{0a} W(\bm{X}_{ab}))^T \\
    \qquad  = -\left(\sum_b \frac{m_{0b}}{\rho_{0b}} \bm{x}_{ab} (\nabla_{0a} W(\bm{X}_{ab}))^T \right) \bm{L}_{0a}^T
```
with $1 \leq i,j \leq d$.
From the deformation gradient, the Green-Lagrange strain
```math
\bm{E} = \frac{1}{2}(\bm{J}^T\bm{J} - \bm{I})
```
and the second Piola-Kirchhoff stress tensor
```math
\bm{S} = \lambda \operatorname{tr}(\bm{E}) \bm{I} + 2\mu \bm{E}
```
are computed to obtain the PK1 stress tensor as
```math
\bm{P} = \bm{J}\bm{S}.
```

Here,
```math
\mu = \frac{E}{2(1 + \nu)}
```
and
```math
\lambda = \frac{E\nu}{(1 + \nu)(1 - 2\nu)}
```
are the Lamé coefficients, where $E$ is the Young's modulus and $\nu$ is the Poisson ratio.

The term $\bm{f}_a^{PF}$ is an optional penalty force. See e.g. [`PenaltyForceGanzenmueller`](@ref).

## References:
- Joseph O’Connor, Benedict D. Rogers.
  "A fluid–structure interaction model for free-surface flows and flexible structures using
  smoothed particle hydrodynamics on a GPU".
  In: Journal of Fluids and Structures 104 (2021).
  [doi: 10.1016/J.JFLUIDSTRUCTS.2021.103312](https://doi.org/10.1016/J.JFLUIDSTRUCTS.2021.103312)
- Ted Belytschko, Yong Guo, Wing Kam Liu, Shao Ping Xiao.
  "A unified stability analysis of meshless particle methods".
  In: International Journal for Numerical Methods in Engineering 48 (2000), pages 1359–1400.
  [doi: 10.1002/1097-0207](https://doi.org/10.1002/1097-0207)
"""
struct SolidParticleContainer{NDIMS, ELTYPE <: Real, K, BM, PF} <: ParticleContainer{NDIMS}
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    current_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    correction_matrix   :: Array{ELTYPE, 3} # [i, j, particle]
    pk1_corrected       :: Array{ELTYPE, 3} # [i, j, particle]
    deformation_grad    :: Array{ELTYPE, 3} # [i, j, particle]
    material_density    :: Array{ELTYPE, 1} # [particle]
    n_moving_particles  :: Int64
    young_modulus       :: ELTYPE
    poisson_ratio       :: ELTYPE
    lame_lambda         :: ELTYPE
    lame_mu             :: ELTYPE
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    acceleration        :: SVector{NDIMS, ELTYPE}
    boundary_model      :: BM
    penalty_force       :: PF

    function SolidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses, particle_material_densities,
                                    smoothing_kernel, smoothing_length,
                                    young_modulus, poisson_ratio, boundary_model;
                                    n_fixed_particles=0,
                                    acceleration=ntuple(_ -> 0.0,
                                                        size(particle_coordinates, 1)),
                                    penalty_force=nothing)
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            error("Acceleration must be of length $NDIMS for a $(NDIMS)D problem")
        end

        if ndims(smoothing_kernel) != NDIMS
            error("Smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem")
        end

        current_coordinates = copy(particle_coordinates)
        correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
        pk1_corrected = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
        deformation_grad = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)

        n_moving_particles = nparticles - n_fixed_particles

        lame_lambda = young_modulus * poisson_ratio /
                      ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        lame_mu = 0.5 * young_modulus / (1 + poisson_ratio)

        # cache = create_cache(hydrodynamic_density_calculator, ELTYPE, nparticles)

        return new{NDIMS, ELTYPE,
                   typeof(smoothing_kernel), typeof(boundary_model),
                   typeof(penalty_force)}(particle_coordinates, current_coordinates,
                                          particle_velocities, particle_masses,
                                          correction_matrix, pk1_corrected,
                                          deformation_grad, particle_material_densities,
                                          n_moving_particles, young_modulus, poisson_ratio,
                                          lame_lambda, lame_mu,
                                          smoothing_kernel, smoothing_length,
                                          acceleration_, boundary_model, penalty_force)
    end
end

function Base.show(io::IO, container::SolidParticleContainer)
    @nospecialize container # reduce precompilation time

    print(io, "SolidParticleContainer{", ndims(container), "}(")
    print(io, container.young_modulus)
    print(io, ", ", container.poisson_ratio)
    print(io, ", ", container.smoothing_kernel)
    print(io, ", ", container.acceleration)
    print(io, ", ", container.boundary_model)
    print(io, ", ", container.penalty_force)
    print(io, ") with ", nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::SolidParticleContainer)
    @nospecialize container # reduce precompilation time

    if get(io, :compact, false)
        show(io, container)
    else
        n_fixed_particles = nparticles(container) - n_moving_particles(container)

        summary_header(io, "SolidParticleContainer{$(ndims(container))}")
        summary_line(io, "total #particles", nparticles(container))
        summary_line(io, "#fixed particles", n_fixed_particles)
        summary_line(io, "Young's modulus", container.young_modulus)
        summary_line(io, "Poisson ratio", container.poisson_ratio)
        summary_line(io, "smoothing kernel", container.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", container.acceleration)
        summary_line(io, "boundary model", container.boundary_model)
        summary_line(io, "penalty force", container.penalty_force |> typeof |> nameof)
        summary_footer(io)
    end
end

@inline function v_nvariables(container::SolidParticleContainer)
    v_nvariables(container, container.boundary_model)
end
# This is dispatched in boundary_container.jl
@inline v_nvariables(container::SolidParticleContainer, model) = ndims(container)

@inline n_moving_particles(container::SolidParticleContainer) = container.n_moving_particles

@inline function current_coordinates(u, container::SolidParticleContainer)
    return container.current_coordinates
end

@inline function current_coords(container::SolidParticleContainer, particle)
    # For this container, the current coordinates are stored in the container directly,
    # so we don't need a `u` array. This function is only to be used in this file
    # when no `u` is available.
    current_coords(nothing, container, particle)
end

@inline function current_velocity(v, container::SolidParticleContainer, particle)
    if particle > n_moving_particles(container)
        return SVector(ntuple(_ -> 0.0, Val(ndims(container))))
    end

    return extract_svector(v, container, particle)
end

@inline function correction_matrix(container, particle)
    extract_smatrix(container.correction_matrix, container, particle)
end
@inline function deformation_gradient(container, particle)
    extract_smatrix(container.deformation_grad, container, particle)
end
@inline function pk1_corrected(container, particle)
    extract_smatrix(container.pk1_corrected, container, particle)
end

function initialize!(container::SolidParticleContainer, neighborhood_search)
    @unpack correction_matrix = container

    # Calculate kernel correction matrix
    calc_correction_matrix!(correction_matrix, neighborhood_search, container)
end

function calc_correction_matrix!(corr_matrix, neighborhood_search, container)
    @unpack mass, material_density = container

    set_zero!(corr_matrix)

    # Calculate kernel correction matrix
    initial_coords = initial_coordinates(container)

    for_particle_neighbor(container, container,
                          initial_coords, initial_coords,
                          neighborhood_search;
                          particles=eachparticle(container)) do particle, neighbor,
                                                                initial_pos_diff,
                                                                initial_distance
        if initial_distance < sqrt(eps())
            return
        end

        volume = mass[neighbor] / material_density[neighbor]

        grad_kernel = smoothing_kernel_grad(container, initial_pos_diff,
                                            initial_distance)
        result = volume * grad_kernel * initial_pos_diff'

        @inbounds for j in 1:ndims(container), i in 1:ndims(container)
            corr_matrix[i, j, particle] -= result[i, j]
        end
    end

    @threaded for particle in eachparticle(container)
        L = correction_matrix(container, particle)
        result = inv(L)

        @inbounds for j in 1:ndims(container), i in 1:ndims(container)
            corr_matrix[i, j, particle] = result[i, j]
        end
    end

    return corr_matrix
end

function update!(container::SolidParticleContainer, container_index, v, u,
                 v_ode, u_ode, semi, t)
    @unpack neighborhood_searches = semi

    # Update current coordinates
    update_current_coordinates(u, container)

    # Precompute PK1 stress tensor
    neighborhood_search = neighborhood_searches[container_index][container_index]
    @trixi_timeit timer() "precompute pk1" compute_pk1_corrected(neighborhood_search,
                                                                 container)

    return container
end

@inline function update_current_coordinates(u, container)
    @unpack current_coordinates = container

    for particle in each_moving_particle(container)
        for i in 1:ndims(container)
            current_coordinates[i, particle] = u[i, particle]
        end
    end
end

@inline function compute_pk1_corrected(neighborhood_search, container)
    @unpack pk1_corrected, deformation_grad = container

    calc_deformation_grad!(deformation_grad, neighborhood_search, container)

    @threaded for particle in eachparticle(container)
        J_particle = deformation_gradient(container, particle)
        pk1_particle = pk1_stress_tensor(J_particle, container)
        pk1_particle_corrected = pk1_particle * correction_matrix(container, particle)

        @inbounds for j in 1:ndims(container), i in 1:ndims(container)
            pk1_corrected[i, j, particle] = pk1_particle_corrected[i, j]
        end
    end
end

@inline function calc_deformation_grad!(deformation_grad, neighborhood_search, container)
    @unpack mass, material_density = container

    # Reset deformation gradient
    set_zero!(deformation_grad)

    initial_coords = initial_coordinates(container)
    for_particle_neighbor(container, container,
                          initial_coords, initial_coords,
                          neighborhood_search;
                          particles=eachparticle(container)) do particle, neighbor,
                                                                initial_pos_diff,
                                                                initial_distance
        volume = mass[neighbor] / material_density[neighbor]
        pos_diff = current_coords(container, particle) - current_coords(container, neighbor)

        if initial_distance < sqrt(eps())
            return
        end

        grad_kernel = smoothing_kernel_grad(container, initial_pos_diff,
                                            initial_distance)

        result = volume * pos_diff * grad_kernel'

        # Mulitply by L_{0a}
        result *= correction_matrix(container, particle)'

        @inbounds for j in 1:ndims(container), i in 1:ndims(container)
            deformation_grad[i, j, particle] -= result[i, j]
        end
    end

    return deformation_grad
end

# First Piola-Kirchhoff stress tensor
@inline function pk1_stress_tensor(J, container)
    S = pk2_stress_tensor(J, container)

    return J * S
end

# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(J, container)
    @unpack lame_lambda, lame_mu = container

    # Compute the Green-Lagrange strain
    E = 0.5 * (transpose(J) * J - I)

    return lame_lambda * tr(E) * I + 2 * lame_mu * E
end

@inline function calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                                     initial_distance, container,
                                     penalty_force::PenaltyForceGanzenmueller)
    @unpack mass, material_density, current_coordinates, young_modulus = container

    current_pos_diff = current_coords(container, particle) -
                       current_coords(container, neighbor)
    current_distance = norm(current_pos_diff)

    volume_particle = mass[particle] / material_density[particle]
    volume_neighbor = mass[neighbor] / material_density[neighbor]

    kernel_ = smoothing_kernel(container, initial_distance)

    J_a = deformation_gradient(container, particle)
    J_b = deformation_gradient(container, neighbor)

    # Use the symmetry of epsilon to simplify computations
    eps_sum = (J_a + J_b) * initial_pos_diff - 2 * current_pos_diff
    delta_sum = dot(eps_sum, current_pos_diff) / current_distance

    f = 0.5 * penalty_force.alpha * volume_particle * volume_neighbor *
        kernel_ / initial_distance^2 * young_modulus * delta_sum *
        current_pos_diff / current_distance

    for i in 1:ndims(container)
        # Divide force by mass to obtain acceleration
        dv[i, particle] += f[i] / mass[particle]
    end

    return dv
end

@inline function calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                                     initial_distance, container, ::Nothing)
    return dv
end

function write_u0!(u0, container::SolidParticleContainer)
    @unpack initial_coordinates = container

    for particle in each_moving_particle(container)
        # Write particle coordinates
        for dim in 1:ndims(container)
            u0[dim, particle] = initial_coordinates[dim, particle]
        end
    end

    return u0
end

function write_v0!(v0, container::SolidParticleContainer)
    @unpack initial_velocity, boundary_model = container

    for particle in each_moving_particle(container)
        # Write particle velocities
        for dim in 1:ndims(container)
            v0[dim, particle] = initial_velocity[dim, particle]
        end
    end

    write_v0!(v0, boundary_model, container)

    return v0
end

# This is dispatched in boundary_container.jl
function write_v0!(v0, boundary_model, container)
    return v0
end

function write_v0!(v0, boundary_model, density_calculator, container)
    return v0
end

function write_v0!(v0, boundary_model, ::ContinuityDensity, container)
    @unpack cache = boundary_model
    @unpack initial_density = cache

    for particle in each_moving_particle(container)
        # Set particle densities
        v0[ndims(container) + 1, particle] = initial_density[particle]
    end

    return v0
end
