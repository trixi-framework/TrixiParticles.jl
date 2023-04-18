@testset verbose=true "Constructor" begin
    coordinates_ = [
        [1.0 2.0
         1.0 2.0],
        [1.0 2.0
         1.0 2.0
         1.0 2.0],
    ]
    @testset "$(i+1)D" for i in 1:2
        NDIMS = i + 1
        coordinates = coordinates_[i]
        velocities = zero(coordinates)
        masses = [1.25, 1.5]
        material_densities = [990.0, 1000.0]
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = i + 1
        smoothing_length = 0.362
        # Make both Lamé constants equal to 1
        nu = 0.25
        E = 2.5
        boundary_model = Val(:boundary_model)

        container = SolidParticleContainer(coordinates, velocities, masses,
                                           material_densities, smoothing_kernel,
                                           smoothing_length, E, nu, boundary_model)

        @test container isa SolidParticleContainer{NDIMS}
        @test container.initial_coordinates == coordinates
        @test container.current_coordinates == coordinates
        @test container.initial_velocity == velocities
        @test container.mass == masses
        @test container.material_density == material_densities
        @test container.n_moving_particles == 2
        @test container.young_modulus == E
        @test container.poisson_ratio == nu
        @test container.lame_lambda == 1.0
        @test container.lame_mu == 1.0
        @test container.smoothing_kernel == smoothing_kernel
        @test container.smoothing_length == smoothing_length
        @test container.acceleration == [0.0 for _ in 1:NDIMS]
        @test container.boundary_model == boundary_model
    end
end

@testset verbose=true "show" begin
    coordinates = [1.0 2.0
                   1.0 2.0]
    velocities = zero(coordinates)
    masses = [1.25, 1.5]
    material_densities = [990.0, 1000.0]
    smoothing_kernel = Val(:smoothing_kernel)
    TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
    smoothing_length = 0.362
    # Make both Lamé constants equal to 1
    nu = 0.25
    E = 2.5
    boundary_model = Val(:boundary_model)

    container = SolidParticleContainer(coordinates, velocities, masses,
                                       material_densities, smoothing_kernel,
                                       smoothing_length, E, nu, boundary_model)

    show_compact = "SolidParticleContainer{2}(2.5, 0.25, Val{:smoothing_kernel}(), " *
                   "[0.0, 0.0], Val{:boundary_model}(), nothing) with 2 particles"
    @test repr(container) == show_compact

    show_box = """
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ SolidParticleContainer{2}                                                                        │
    │ ═════════════════════════                                                                        │
    │ total #particles: ………………………………… 2                                                                │
    │ #fixed particles: ………………………………… 0                                                                │
    │ Young's modulus: …………………………………… 2.5                                                              │
    │ Poisson ratio: ………………………………………… 0.25                                                             │
    │ smoothing kernel: ………………………………… Val                                                              │
    │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
    │ boundary model: ……………………………………… Val{:boundary_model}()                                           │
    │ penalty force: ………………………………………… Nothing                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
    @test repr("text/plain", container) == show_box
end

@testset verbose=true "Deformation Gradient" begin
    @testset verbose=true "Manual Calculation" begin
        # Compare against manually calculated values.
        #
        # This is a proof of concept showing that mocking in Julia unit tests works without
        # the need for any mocking packages or modifying source code.
        #
        # We construct a lowest-level unit test where we mock all function calls and don't use
        # a real container object.
        # We replace all objects that we don't need by objects of the type Val{:mock_something},
        # for which we define specific behaviour below.
        # This makes it easy to calculate the deformation gradient by hand, since we can
        # manually define the neighbors and kernel derivatives to be used in the calculation.

        current_coordinates = [[2 3; 0 0], # coords[neighbor] - coords[particle] = (1, 0)
            [6 8; -1 3]] # coords[neighbor] - coords[particle] = (2, 4)

        initial_coordinates = [[5 5; 3 4], # initial_coords[particle] - initial_coords[neighbor] = (0, -1)
            [0 -3; 0 -5]] # initial_coords[particle] - initial_coords[neighbor] = (3, 5)

        expected = [[0 -0.5; 0 0], # Tensor product of the two vectors above, multiplied by the volume of 0.5
            [3 5; 6 10] / sqrt(34)] # Same as above but divided by the length of the vector (3, 5)

        @testset "Tensor Product $i" for i in 1:2
            #### Setup
            particle = 1
            neighbors = [1, 2]
            mass = [2.0, 2.0]
            density = 4.0

            correction_matrix = [1 0; 0 1]

            # This will cause the computed gradient
            # to be equal to `initial_coords[particle] - initial_coords[neighbor]`
            kernel_derivative = 1.0

            #### Mocking
            # Mock the container
            container = Val(:mock_container_tensor)
            TrixiParticles.ndims(::Val{:mock_container_tensor}) = 2

            # All @unpack calls should return another mock object
            # of the type `Val{:mock_property_name}`, but we want to have some real matrices
            # as properties as opposed to only mock objects.
            function Base.getproperty(::Val{:mock_container_tensor}, f::Symbol)
                if f === :initial_coordinates
                    return initial_coordinates[i]
                elseif f === :correction_matrix
                    return correction_matrix
                elseif f === :current_coordinates
                    return current_coordinates[i]
                elseif f === :mass
                    return mass
                end

                # For all other properties, return mock objects.
                return Val(Symbol("mock_" * string(f)))
            end

            TrixiParticles.eachneighbor(_, ::Val{:mock_nhs}) = neighbors

            Base.getindex(::Val{:mock_material_density}, ::Int64) = density

            function TrixiParticles.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _)
                kernel_derivative
            end

            #### Verification
            @test TrixiParticles.deformation_gradient(particle, Val(:mock_nhs),
                                                      container) == expected[i]
        end
    end

    @testset verbose=true "Deformation Functions" begin
        # We generate a grid of particles, apply a deformation, and verify that the computed
        # deformation gradient matches the deformation matrix.
        deformations = Dict("Stretch x" => x -> [2.0 0.0; 0.0 1.0] * x,
                            "Stretch Both" => x -> [2.0 0.0; 0.0 3.0] * x,
                            "Rotation" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * x,
                            "Nonlinear Stretching" => x -> [x[1]^2, x[2]])

        deformation_gradients = Dict("Stretch x" => [2.0 0.0; 0.0 1.0],
                                     "Stretch Both" => [2.0 0.0; 0.0 3.0],
                                     "Rotation" => [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)],
                                     "Nonlinear Stretching" => [1.0 0.0; 0.0 1.0])

        @testset "$deformation_name" for deformation_name in keys(deformations)
            deformation = deformations[deformation_name]

            # 9 x 9 grid of particles with spacing 0.1
            range = 0.1:0.1:0.9
            particle_coordinates = hcat(collect.(Iterators.product(range, range))...)

            n_particles_per_dimension = (9, 9)
            particle_velocities = zero(particle_coordinates)
            particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension))
            particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

            smoothing_length = 0.12
            smoothing_kernel = SchoenbergCubicSplineKernel{2}()
            search_radius = TrixiParticles.compact_support(smoothing_kernel,
                                                           smoothing_length)

            container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                               particle_masses, particle_densities,
                                               smoothing_kernel, smoothing_length,
                                               1.0, 1.0, nothing)
            nhs = TrixiParticles.TrivialNeighborhoodSearch(container)
            TrixiParticles.initialize!(container, nhs)

            # Apply the deformation matrix
            for particle in TrixiParticles.eachparticle(container)
                new_coords = deformation(container.initial_coordinates[:, particle])
                container.current_coordinates[:, particle] = new_coords
            end

            # Compute the deformation gradient for the particle in the middle
            J = TrixiParticles.deformation_gradient(41, nhs, container)

            #### Verification
            @test J ≈ deformation_gradients[deformation_name]
        end
    end
end

@testset verbose=true "Stress tensors" begin
    deformations = Dict("rotation" => [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)],
                        "stretch both" => [2.0 0.0; 0.0 3.0],
                        "rotate and stretch" => [cos(0.3) -sin(0.3);
                                                 sin(0.3) cos(0.3)] * [2.0 0.0; 0.0 3.0])

    expected_pk2 = Dict("rotation" => zeros(2, 2), # No stress in rotations only
                        "stretch both" => [8.5 0.0; 0.0 13.5], # Calculated by hand
                        "rotate and stretch" => [8.5 0.0; 0.0 13.5])

    # Just deformation * expected_pk2
    expected_pk1 = Dict("rotation" => zeros(2, 2),
                        "stretch both" => [17.0 0.0; 0.0 40.5],
                        "rotate and stretch" => [cos(0.3) -sin(0.3);
                                                 sin(0.3) cos(0.3)] *
                                                [2.0 0.0; 0.0 3.0] * [8.5 0.0; 0.0 13.5])

    @testset "Deformation Function: $deformation" for deformation in keys(deformations)
        #### Setup
        J = deformations[deformation]
        lame_lambda = 1.0
        lame_mu = 1.0

        #### Mocking
        # It is easier to mock the container and specify the Lamé constants
        # and deformation gradient than to actually construct a container.
        container = Val(:mock_container)

        # All @unpack calls should return another mock object
        # of the type `Val{:mock_property_name}`, but we want to have the actual
        # Lamé constants as properties.
        function Base.getproperty(::Val{:mock_container}, f::Symbol)
            if f === :lame_lambda
                return lame_lambda
            elseif f === :lame_mu
                return lame_mu
            end

            # For all other properties, return mock objects
            return Val(Symbol("mock_" * string(f)))
        end

        TrixiParticles.deformation_gradient(_, ::Val{:mock_container}) = J

        #### Verification
        @test TrixiParticles.pk2_stress_tensor(J, container) ≈ expected_pk2[deformation]
        @test TrixiParticles.pk1_stress_tensor(J, container) ≈ expected_pk1[deformation]
    end
end

@testset verbose=true "write_u0!" begin
    coordinates = [1.0 2.0
                   1.0 2.0]
    velocities = zero(coordinates)
    masses = [1.25, 1.5]
    material_densities = [990.0, 1000.0]
    smoothing_kernel = Val(:smoothing_kernel)
    smoothing_length = 0.362
    # Make both Lamé constants equal to 1
    nu = 0.25
    E = 2.5
    boundary_model = Val(:boundary_model)

    container = SolidParticleContainer(coordinates, velocities, masses,
                                       material_densities, smoothing_kernel,
                                       smoothing_length, E, nu, boundary_model)

    u0 = zeros(TrixiParticles.u_nvariables(container),
               TrixiParticles.n_moving_particles(container))
    TrixiParticles.write_u0!(u0, container)

    @test u0 == coordinates
end

@testset verbose=true "write_v0!" begin
    coordinates = [1.0 2.0
                   1.0 2.0]
    velocities = zero(coordinates)
    masses = [1.25, 1.5]
    material_densities = [990.0, 1000.0]
    smoothing_kernel = Val(:smoothing_kernel)
    smoothing_length = 0.362
    # Make both Lamé constants equal to 1
    nu = 0.25
    E = 2.5
    boundary_model = Val(:boundary_model)

    container = SolidParticleContainer(coordinates, velocities, masses,
                                       material_densities, smoothing_kernel,
                                       smoothing_length, E, nu, boundary_model)

    v0 = zeros(TrixiParticles.v_nvariables(container),
               TrixiParticles.n_moving_particles(container))
    TrixiParticles.write_v0!(v0, container)

    @test v0 == velocities
end
