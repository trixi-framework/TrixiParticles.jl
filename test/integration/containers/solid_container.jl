@testset verbose=true "Deformation Gradient" begin
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

@testset verbose=true "Stress tensors" begin
    @testset "Rotate and Stretch" begin
        # Make both Lamé constants equal to 1
        nu = 0.25
        E = 2.5

        # 9 x 9 grid of particles
        n_particles_per_dimension = (9, 9)
        particle_coordinates = Array{Float64, 2}(undef, 2,
                                                 prod(n_particles_per_dimension))
        particle_velocities = Array{Float64, 2}(undef, 2,
                                                prod(n_particles_per_dimension))
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
    end
end
