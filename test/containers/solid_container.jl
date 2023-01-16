@testset "Deformation Gradient" begin
    @testset "Unit Tests" begin
        # This is a proof of concept to show that mocking in Julia unit tests works without
        # the need for any mocking packages or modifying source code.

        current_coordinates = [[2 3; 0 0], # coords[neighbor] - coords[particle] = (1, 0)
            [6 8; -1 3]] # coords[neighbor] - coords[particle] = (2, 4)

        initial_coordinates = [[5 5; 3 4], # initial_coords[particle] - initial_coords[neighbor] = (0, -1)
            [0 -3; 0 -5]] # initial_coords[particle] - initial_coords[neighbor] = (3, 5)

        expected = [[0 -0.5; 0 0], # Tensor product of the two vectors above, multiplied by the volume of 0.5
            [3 5; 6 10] / sqrt(34)] # Same as above but divided by the length of the vector (3, 5)

        # This is a true unit test where we mock all function calls and don't need
        # a real container object.
        # We replace all objects that we don't need by objects of the type Val{:mock_something},
        # for which we define specific behaviour below.
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
            Pixie.ndims(::Val{:mock_container_tensor}) = 2

            # All @unpack calls should return another mock object of the type Val{:mock_property_name},
            # but we want to have some real matrices as properties as opposed to only mock objects
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

                # For all other properties, return mock objects
                return Val(Symbol("mock_" * string(f)))
            end

            Pixie.eachneighbor(_, ::Val{:mock_nhs}) = neighbors

            Base.getindex(::Val{:mock_material_density}, ::Int64) = density

            Pixie.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _) = kernel_derivative

            #### Verification
            @test Pixie.deformation_gradient(particle, Val(:mock_nhs), container) ==
                  expected[i]
        end
    end

    @testset "Integration Tests" begin
        deformations = Dict("stretch x" => x -> [2.0 0.0; 0.0 1.0] * x,
                            "stretch both" => x -> [2.0 0.0; 0.0 3.0] * x,
                            "rotation" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * x,
                            "nonlinear stretching" => x -> [x[1]^2, x[2]])

        deformation_gradients = Dict("stretch x" => [2.0 0.0; 0.0 1.0],
                                     "stretch both" => [2.0 0.0; 0.0 3.0],
                                     "rotation" => [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)],
                                     "nonlinear stretching" => [1.0 0.0; 0.0 1.0])

        # We generate a grid of particles, apply a deformation, and verify that the computed
        # deformation gradient matches the deformation matrix.
        @testset "Deformation Function: $deformation" for deformation in keys(deformations)
            # 9 x 9 grid of particles
            n_particles_per_dimension = (9, 9)
            particle_coordinates = Array{Float64, 2}(undef, 2,
                                                     prod(n_particles_per_dimension))
            # particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
            particle_velocities = zeros(2, prod(n_particles_per_dimension))
            particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension))
            particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

            for y in 1:n_particles_per_dimension[2],
                x in 1:n_particles_per_dimension[1]

                particle = (x - 1) * n_particles_per_dimension[2] + y

                # Coordinates
                particle_coordinates[1, particle] = x * 0.1
                particle_coordinates[2, particle] = y * 0.1
            end

            smoothing_length = 0.12
            smoothing_kernel = SchoenbergCubicSplineKernel{2}()
            search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

            container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                               particle_masses, particle_densities,
                                               smoothing_kernel, smoothing_length,
                                               1.0, 1.0, nothing)
            nhs = Pixie.TrivialNeighborhoodSearch(container)
            Pixie.initialize!(container, nhs)

            # Apply the deformation matrix
            for particle in Pixie.eachparticle(container)
                container.current_coordinates[:, particle] = deformations[deformation](container.initial_coordinates[:,
                                                                                                                     particle])
            end

            # Compute the deformation gradient for the particle in the middle
            J = Pixie.deformation_gradient(41, nhs, container)

            #### Verification
            @test J ≈ deformation_gradients[deformation]
        end
    end
end

@testset "Stress tensors" begin
    @testset "Unit tests" begin
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
            container = Val(:mock_container_deform)

            # All @unpack calls should return another mock object of the type Val{:mock_property_name},
            # but we want to have the actual Lamé constants as properties
            function Base.getproperty(::Val{:mock_container_deform}, f::Symbol)
                if f === :lame_lambda
                    return lame_lambda
                elseif f === :lame_mu
                    return lame_mu
                end

                # For all other properties, return mock objects
                return Val(Symbol("mock_" * string(f)))
            end

            Pixie.deformation_gradient(_, ::Val{:mock_container_deform}) = J

            #### Verification
            @test Pixie.pk2_stress_tensor(J, container) ≈ expected_pk2[deformation]
            @test Pixie.pk1_stress_tensor(J, container) ≈ expected_pk1[deformation]
        end
    end

    @testset "Integration Tests" begin @testset "Rotate and Stretch" begin
        # Make both Lamé constants equal to 1
        nu = 0.25
        E = 2.5

        # 9 x 9 grid of particles
        n_particles_per_dimension = (9, 9)
        particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
        particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension))
        particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

        for y in 1:n_particles_per_dimension[2],
            x in 1:n_particles_per_dimension[1]

            particle = (x - 1) * n_particles_per_dimension[2] + y

            # Coordinates
            particle_coordinates[1, particle] = x * 0.1
            particle_coordinates[2, particle] = y * 0.1
        end

        smoothing_length = 0.12
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)
        container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                           particle_masses, particle_densities,
                                           smoothing_kernel, smoothing_length,
                                           E, nu, nothing)
        nhs = Pixie.TrivialNeighborhoodSearch(container)
        Pixie.initialize!(container, nhs)

        # Apply the deformation matrix
        for particle in Pixie.eachparticle(container)
            # Rotate and stretch with the same deformation as in the unit test above
            container.current_coordinates[:, particle] = [cos(0.3) -sin(0.3);
                                                          sin(0.3) cos(0.3)] *
                                                         [2.0 0.0; 0.0 3.0] *
                                                         container.initial_coordinates[:,
                                                                                       particle]
        end

        #### Verification for the particle in the middle
        particle = 41
        J_expected = [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * [2.0 0.0; 0.0 3.0]

        # Deformation gradient
        @test Pixie.deformation_gradient(particle, nhs, container) ≈ J_expected

        # PK2 stress tensor (same as in the unit test above)
        @test Pixie.pk2_stress_tensor(J_expected, container) ≈ [8.5 0.0; 0.0 13.5]

        # PK1 stress tensor (same as in the unit test above)
        @test Pixie.pk1_stress_tensor(J_expected, container) ≈
              J_expected * [8.5 0.0; 0.0 13.5]
    end end
end
