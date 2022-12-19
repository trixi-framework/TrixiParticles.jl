@doc raw"""
    SolidParticleContainer(particle_coordinates, particle_velocities,
                           particle_masses, particle_material_densities,
                           hydrodynamic_density_calculator,
                           smoothing_kernel, smoothing_length,
                           young_modulus, poisson_ratio;
                           n_fixed_particles=0,
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                           neighborhood_search=nothing)

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
    \nabla_{0a} W(\bm{X}_{ab}) + \bm{g},
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
(\bm{J}_a)^{i,j} = \sum_b \frac{m_{0b}}{\rho_{0b}} (\bm{x}_b^i - \bm{x}_a^i) (\bm{L}_{0a}\nabla_{0a} W(\bm{X}_{ab}))^j
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

References:
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
struct SolidParticleContainer{NDIMS, ELTYPE<:Real, DC, K, BM, C} <: ParticleContainer{NDIMS}
    initial_coordinates ::Array{ELTYPE, 2} # [dimension, particle]
    current_coordinates ::Array{ELTYPE, 2} # [dimension, particle]
    initial_velocity    ::Array{ELTYPE, 2} # [dimension, particle]
    mass                ::Array{ELTYPE, 1} # [particle]
    correction_matrix   ::Array{ELTYPE, 3} # [i, j, particle]
    pk1_corrected       ::Array{ELTYPE, 3} # [i, j, particle]
    material_density    ::Array{ELTYPE, 1} # [particle]
    n_moving_particles  ::Int64
    lame_lambda         ::ELTYPE
    lame_mu             ::ELTYPE
    hydrodynamic_density_calculator::DC # TODO
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    acceleration        ::SVector{NDIMS, ELTYPE}
    boundary_model      ::BM
    cache               ::C

    function SolidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses, particle_material_densities,
                                    hydrodynamic_density_calculator,
                                    smoothing_kernel, smoothing_length,
                                    young_modulus, poisson_ratio, boundary_model;
                                    n_fixed_particles=0,
                                    acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)))
        NDIMS = size(particle_coordinates, 1)
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)

        current_coordinates = copy(particle_coordinates)
        correction_matrix   = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
        pk1_corrected       = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)

        n_moving_particles = nparticles - n_fixed_particles

        lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2*poisson_ratio))
        lame_mu = 0.5 * young_modulus / (1 + poisson_ratio)

        # cache = create_cache(hydrodynamic_density_calculator, ELTYPE, nparticles)
        cache = (; )

        return new{NDIMS, ELTYPE, typeof(hydrodynamic_density_calculator),
                   typeof(smoothing_kernel), typeof(boundary_model), typeof(cache)}(
            particle_coordinates, current_coordinates, particle_velocities, particle_masses,
            correction_matrix, pk1_corrected, particle_material_densities,
            n_moving_particles, lame_lambda, lame_mu,
            hydrodynamic_density_calculator, smoothing_kernel, smoothing_length,
            acceleration_, boundary_model, cache)
    end
end


@inline n_moving_particles(container::SolidParticleContainer) = container.n_moving_particles

@inline function get_current_coords(particle, u, container::SolidParticleContainer)
    @unpack current_coordinates = container

    return get_particle_coords(particle, current_coordinates, container)
end


@inline get_correction_matrix(particle, container) = extract_smatrix(container.correction_matrix, particle, container)
@inline get_pk1_corrected(particle, container) = extract_smatrix(container.pk1_corrected, particle, container)

@inline function extract_smatrix(array, particle, container)
    # Extract the matrix elements for this particle as a tuple to pass to SMatrix
    return SMatrix{ndims(container), ndims(container)}(
        # Convert linear index to Cartesian index
        ntuple(@inline(i -> array[mod(i-1, ndims(container))+1, div(i-1, ndims(container))+1, particle]), Val(ndims(container)^2)))
end

# Extract the j-th column of the correction matrix for this particle as an SVector
@inline function get_correction_matrix_column(j, particle, container)
    @unpack correction_matrix = container

    return SVector(ntuple(@inline(dim -> correction_matrix[dim, j, particle]), Val(ndims(container))))
end


function initialize!(container::SolidParticleContainer, neighborhood_search)
    @unpack correction_matrix = container

    # Calculate kernel correction matrix
    calc_correction_matrix!(correction_matrix, neighborhood_search, container)
end


function calc_correction_matrix!(correction_matrix, neighborhood_search, container)
    @unpack initial_coordinates, mass, material_density,
        smoothing_kernel, smoothing_length = container

    # Calculate kernel correction matrix
    for particle in eachparticle(container)
        L = zeros(eltype(mass), ndims(container), ndims(container))

        particle_coordinates = get_particle_coords(particle, initial_coordinates, container)
        for neighbor in eachneighbor(particle_coordinates, neighborhood_search)
            volume = mass[neighbor] / material_density[neighbor]

            initial_pos_diff = particle_coordinates - get_particle_coords(neighbor, initial_coordinates, container)
            initial_distance = norm(initial_pos_diff)

            if initial_distance > eps()
                grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
                    initial_pos_diff / initial_distance

                L -= volume * grad_kernel * transpose(initial_pos_diff)
            end
        end

        correction_matrix[:, :, particle] = inv(L)
    end

    return correction_matrix
end


function update!(container::SolidParticleContainer, u, u_ode, neighborhood_search, semi, t)
    # Update current coordinates
    @pixie_timeit timer() "update current coordinates" update_current_coordinates(u, container)

    # Precompute PK1 stress tensor
    @pixie_timeit timer() "precompute pk1 stress tensor" compute_pk1_corrected(neighborhood_search, container)

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
    @unpack pk1_corrected = container

    @threaded for particle in eachparticle(container)
        pk1_particle = pk1_stress_tensor(particle, neighborhood_search, container)
        pk1_particle_corrected = pk1_particle * get_correction_matrix(particle, container)

        for j in 1:ndims(container), i in 1:ndims(container)
            pk1_corrected[i, j, particle] = pk1_particle_corrected[i, j]
        end
    end
end


# First Piola-Kirchhoff stress tensor
function pk1_stress_tensor(particle, neighborhood_search, container)
    J = deformation_gradient(particle, neighborhood_search, container)

    S = pk2_stress_tensor(J, container)

    return J * S
end


# We cannot use a variable for the number of dimensions here, it has to be hardcoded
@inline function deformation_gradient(particle, neighborhood_search, container::SolidParticleContainer{2})
    return @SMatrix [deformation_gradient(i, j, particle, neighborhood_search, container) for i in 1:2, j in 1:2]
end

@inline function deformation_gradient(particle, container::SolidParticleContainer{3})
    return @SMatrix [deformation_gradient(i, j, particle, neighborhood_search, container) for i in 1:3, j in 1:3]
end


function deformation_gradient(i, j, particle, neighborhood_search, container)
    @unpack initial_coordinates, current_coordinates, correction_matrix,
        mass, material_density, smoothing_kernel, smoothing_length = container

    result = zero(eltype(mass))

    initial_particle_coords = get_particle_coords(particle, initial_coordinates, container)
    for neighbor in eachneighbor(initial_particle_coords, neighborhood_search)
        volume = mass[neighbor] / material_density[neighbor]
        pos_diff = get_particle_coords(particle, current_coordinates, container) -
            get_particle_coords(neighbor, current_coordinates, container)

        initial_pos_diff = initial_particle_coords - get_particle_coords(neighbor, initial_coordinates, container)
        initial_distance = norm(initial_pos_diff)

        if initial_distance > sqrt(eps())
            # TODO pull L multiplication out of the neighbor loop
            grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
                dot(get_correction_matrix_column(j, particle, container), initial_pos_diff) / initial_distance

            result -= volume * pos_diff[i] * grad_kernel
        end
    end

    return result
end


# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(J, container)
    @unpack lame_lambda, lame_mu = container

    # Compute the Green-Lagrange strain
    E = 0.5 * (transpose(J) * J - I)

    return lame_lambda * tr(E) * I + 2 * lame_mu * E
end


function write_variables!(u0, container::SolidParticleContainer)
    @unpack initial_coordinates, initial_velocity = container

    for particle in each_moving_particle(container)
        # Write particle coordinates
        for dim in 1:ndims(container)
            u0[dim, particle] = initial_coordinates[dim, particle]
        end

        # Write particle velocities
        for dim in 1:ndims(container)
            u0[dim + ndims(container), particle] = initial_velocity[dim, particle]
        end
    end

    return u0
end
