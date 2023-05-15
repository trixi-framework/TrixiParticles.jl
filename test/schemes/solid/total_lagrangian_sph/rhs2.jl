@testset verbose=true "Solid RHS" begin
    deformations = Dict("rotation" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * x,
                        "stretch both" => x -> [2.0 0.0; 0.0 3.0] * x,
                        "rotate and stretch" => x -> [cos(0.3) -sin(0.3);
                                                      sin(0.3) cos(0.3)] *
                                                     [2.0 0.0; 0.0 3.0] * x,
                        "nonlinear stretching" => x -> [x[1]^2, x[2]])

    # The acceleration in the first three should be zero (linear stretching)
    # The fourth one is calculated by hand
    dv_expected_41 = Dict("rotation" => [0.0, 0.0],
                          "stretch both" => [0.0, 0.0],
                          "rotate and stretch" => [0.0, 0.0],
                          "nonlinear stretching" => [
                              10 / 1000^2 * 1.5400218087591082 * 324.67072684047224 *
                              1.224,
                              0.0,
                          ])

    @testset "Deformation Function: $deformation" for deformation in keys(deformations)
        J = deformations[deformation]
        u = zeros(2, 81)
        v = zeros(2, 81)
        v[1, :] .= 1.0
        v[2, :] .= -2.0

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

        smoothing_length = 0.07
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        container = TotalLagrangianSPHSystem(particle_coordinates, particle_velocities,
                                           particle_masses, particle_densities,
                                           smoothing_kernel, smoothing_length,
                                           E, nu, nothing)

        semi = Semidiscretization(container)
        tspan = (0.0, 1.0)
        semidiscretize(semi, tspan)

        # Apply the deformation matrix
        for particle in axes(u, 2)
            # Apply deformation
            u[1:2, particle] = deformations[deformation](particle_coordinates[:,
                                                                              particle])
        end

        #### Verification for the particle in the middle
        particle = 41

        dv = zeros(ndims(container), 81)
        TrixiParticles.kick!(dv, v, u, semi, 0.0)

        @test isapprox(dv[:, particle], dv_expected_41[deformation],
                       rtol=sqrt(eps()), atol=sqrt(eps()))
    end
end
