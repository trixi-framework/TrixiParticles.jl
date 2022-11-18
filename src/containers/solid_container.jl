"""
    SolidParticleContainer(particle_coordinates, particle_velocities,
                           particle_masses, particle_material_densities,
                           hydrodynamic_density_calculator,
                           smoothing_kernel, smoothing_length,
                           young_modulus, poisson_ratio;
                           n_fixed_particles=0,
                           acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                           neighborhood_search=nothing)

Container for particles of an elastic solid.
"""
struct SolidParticleContainer{NDIMS, ELTYPE<:Real, DC, K, NS, C} <: ParticleContainer{NDIMS}
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
    neighborhood_search ::NS
    cache               ::C

    function SolidParticleContainer(particle_coordinates, particle_velocities,
                                    particle_masses, particle_material_densities,
                                    hydrodynamic_density_calculator,
                                    smoothing_kernel, smoothing_length,
                                    young_modulus, poisson_ratio;
                                    n_fixed_particles=0,
                                    acceleration=ntuple(_ -> 0.0, size(particle_coordinates, 1)),
                                    neighborhood_search=nothing)
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

        return new{NDIMS, ELTYPE, typeof(hydrodynamic_density_calculator), typeof(smoothing_kernel),
                   typeof(neighborhood_search), typeof(cache)}(
            particle_coordinates, current_coordinates, particle_velocities, particle_masses,
            correction_matrix, pk1_corrected, particle_material_densities,
            n_moving_particles, lame_lambda, lame_mu,
            hydrodynamic_density_calculator, smoothing_kernel, smoothing_length,
            acceleration_, neighborhood_search, cache)
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


function initialize!(container::SolidParticleContainer)
    @unpack initial_coordinates, correction_matrix, neighborhood_search = container

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, initial_coordinates, container)

    # Calculate kernel correction matrix
    calc_correction_matrix!(correction_matrix, container)
end


function calc_correction_matrix!(correction_matrix, container)
    @unpack initial_coordinates, mass, material_density,
        smoothing_kernel, smoothing_length = container

    # Calculate kernel correction matrix
    for particle in eachparticle(container)
        L = zeros(eltype(mass), ndims(container), ndims(container))

        particle_coordinates = get_particle_coords(particle, initial_coordinates, container)
        for neighbor in eachneighbor(particle_coordinates, container)
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


function update!(container::SolidParticleContainer, u, u_ode, semi)
    # Update current coordinates
    @pixie_timeit timer() "update current coordinates" update_current_coordinates(u, container)

    # Precompute PK1 stress tensor
    @pixie_timeit timer() "precompute pk1 stress tensor" compute_pk1_corrected(container)

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


@inline function compute_pk1_corrected(container)
    @unpack pk1_corrected = container

    @threaded for particle in eachparticle(container)
        pk1_particle = pk1_stress_tensor(particle, container)
        pk1_particle_corrected = pk1_particle * get_correction_matrix(particle, container)

        for j in 1:ndims(container), i in 1:ndims(container)
            pk1_corrected[i, j, particle] = pk1_particle_corrected[i, j]
        end
    end
end


# First Piola-Kirchhoff stress tensor
function pk1_stress_tensor(particle, container)
    J = deformation_gradient(particle, container)

    S = pk2_stress_tensor(J, container)

    return J * S
end


# We cannot use a variable for the number of dimensions here, it has to be hardcoded
@inline function deformation_gradient(particle, container::SolidParticleContainer{2})
    return @SMatrix [deformation_gradient(i, j, particle, container) for i in 1:2, j in 1:2]
end

@inline function deformation_gradient(particle, container::SolidParticleContainer{3})
    return @SMatrix [deformation_gradient(i, j, particle, container) for i in 1:3, j in 1:3]
end


function deformation_gradient(i, j, particle, container)
    @unpack initial_coordinates, current_coordinates, correction_matrix,
        mass, material_density, smoothing_kernel, smoothing_length = container

    result = zero(eltype(mass))

    initial_particle_coords = get_particle_coords(particle, initial_coordinates, container)
    for neighbor in eachneighbor(initial_particle_coords, container)
        volume = mass[neighbor] / material_density[neighbor]
        pos_diff = get_particle_coords(particle, current_coordinates, container) -
            get_particle_coords(neighbor, current_coordinates, container)

        initial_pos_diff = initial_particle_coords - get_particle_coords(neighbor, initial_coordinates, container)
        initial_distance = norm(initial_pos_diff)

        if initial_distance > sqrt(eps())
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
