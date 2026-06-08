@testset verbose=true "Structure RHS" begin
    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "interact! Mocked" begin
        # Pass specific PK1 and `pos_diff` to `interact!` and verify with
        # values calculated by hand.
        pk1_particle_corrected = [
            [2.0 0.0; 0.0 3.0],
            [0.230015 0.0526099; 0.348683 0.911251],
            [0.364281 0.763916; 0.433868 0.9755],
            [0.503965 0.125224; 0.739591 0.0323651]
        ]
        pk1_neighbor_corrected = [
            zeros(2, 2),
            [0.794769 0.113958; 0.1353 0.741521],
            [0.97936 0.302584; 0.333461 0.642575],
            [0.503965 0.125224; 0.739591 0.0323651]
        ]
        # Create initial coordinates so that
        # initial_pos_diff is [[0.1, 0.0], [0.5, 0.3], [1.0, 0.37], [0.0, 0.0]]
        initial_coordinates_particle = [
            [1.1, 2.0], [-1.5, 5.0], [-1.5, -6.43], [120.0, 32.2]
        ]
        initial_coordinates_neighbor = [
            [1.0, 2.0], [-2.0, 4.7], [-2.5, -6.8], [120.0, 32.2]
        ]
        particle = [1, 3, 10, 7]
        neighbor = [10, 4, 9, 7]
        dv_particle_expected = [
            [2.0, 0.0],
            [0.07118137645953881, 0.11640973221047637],
            [0.01870315713691948, 0.014067976038520915],
            [0.0, 0.0]
        ]

        @testset verbose=true "Test $i" for i in 1:4
            #### Setup
            each_integrated_particle = [particle[i]] # Only calculate dv for this one particle
            eachparticle = [particle[i], neighbor[i]]
            initial_coordinates = 1000 * ones(2, 10) # Just something that's not zero to catch errors
            initial_coordinates[:, particle[i]] = initial_coordinates_particle[i]
            initial_coordinates[:, neighbor[i]] = initial_coordinates_neighbor[i]
            current_coordinates = zeros(2, 10)
            v_system = zeros(2, 10)

            # Density equals the ID of the particle
            material_density = 1:10

            pk1_rho2 = 2000 * ones(2, 2, 10) # Just something that's not zero to catch errors
            pk1_rho2[:, :,
                     particle[i]] = pk1_particle_corrected[i] /
                                    material_density[particle[i]]^2
            pk1_rho2[:, :,
                     neighbor[i]] = pk1_neighbor_corrected[i] /
                                    material_density[neighbor[i]]^2

            # Use the same setup as in the unit test above for calc_dv!
            mass = ones(Float64, 10)
            kernel_deriv = 1.0

            #### Mocking
            struct MockSystem <: TrixiParticles.AbstractSystem{2}
                current_coordinates::Any
                material_density::Any
                pk1_rho2::Any
                mass::Any
                penalty_force::Any
                viscosity::Any
                buffer::Any
            end
            @inline Base.eltype(::MockSystem) = Float64
            system = MockSystem(current_coordinates, material_density, pk1_rho2, mass,
                                nothing, nothing, nothing)

            function TrixiParticles.initial_coordinates(::MockSystem)
                return initial_coordinates
            end

            TrixiParticles.eachparticle(::MockSystem) = eachparticle
            TrixiParticles.each_integrated_particle(::MockSystem) = each_integrated_particle
            TrixiParticles.smoothing_length(::MockSystem, _) = eps()

            function TrixiParticles.smoothing_kernel_grad_unsafe(::MockSystem,
                                                                 pos_diff, distance,
                                                                 particle)
                return kernel_deriv * pos_diff / distance
            end
            TrixiParticles.compact_support(::MockSystem, ::MockSystem) = 1000.0
            function TrixiParticles.current_coords(system::MockSystem, particle)
                return TrixiParticles.current_coords(initial_coordinates, system, particle)
            end
            function TrixiParticles.deformation_gradient(::MockSystem, particle)
                return nothing
            end

            #### Verification
            backends = [
                SerialBackend(), # CPU code
                TrixiParticles.KernelAbstractions.CPU() # Emulate GPU code
            ]
            names = ["CPU code", "Emulate GPU"]
            @testset "$(names[j])" for j in eachindex(names)
                dv = zeros(ndims(system), 10)
                dv_expected = copy(dv)
                dv_expected[:, particle[i]] = dv_particle_expected[i]

                semi = DummySemidiscretization(parallelization_backend=backends[j])
                TrixiParticles.interact_structure_structure!(dv, v_system, system, semi)

                @test dv ≈ dv_expected
            end
        end
    end

    @testset verbose=true "interact! with System and Deformation Function" begin
        deformations = Dict(
            "rotation" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * x,
            "stretch both" => x -> [2.0 0.0; 0.0 3.0] * x,
            "rotate and stretch" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] *
                                         [2.0 0.0; 0.0 3.0] * x,
            "nonlinear stretching" => x -> [x[1]^2, x[2]])

        # The acceleration in the first three should be zero (linear stretching)
        # The fourth one is calculated by hand
        dv_expected_41 = Dict(
            "rotation" => [0.0, 0.0],
            "stretch both" => [0.0, 0.0],
            "rotate and stretch" => [0.0, 0.0],
            "nonlinear stretching" => [
                10 / 1000^2 * 1.5400218087591082 * 324.67072684047224 * 1.224, 0.0
            ])

        @testset verbose=true "Deformation Function: $deformation" for deformation in
                                                                       keys(deformations)
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
            coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
            mass = 10.0
            density = 1000.0

            for y in 1:n_particles_per_dimension[2],
                x in 1:n_particles_per_dimension[1]
                particle = (x - 1) * n_particles_per_dimension[2] + y

                # Coordinates
                coordinates[1, particle] = x * 0.1
                coordinates[2, particle] = y * 0.1
            end

            smoothing_length = 0.07
            smoothing_kernel = SchoenbergCubicSplineKernel{2}()

            initial_condition = InitialCondition(; coordinates, mass, density)
            system = TotalLagrangianSPHSystem(initial_condition; smoothing_kernel,
                                              smoothing_length, young_modulus=E,
                                              poisson_ratio=nu)
            tspan = (0.0, 1.0)

            names = ["CPU code", "GPU code emulated on the CPU"]
            backends = [SerialBackend(), TrixiParticles.KernelAbstractions.CPU()]
            @testset "$(names[i])" for i in eachindex(names)
                semi = Semidiscretization(system, parallelization_backend=backends[i])
                system_ = semi.systems[1] # Get updated system
                ode = semidiscretize(semi, tspan)

                # Apply the deformation matrix
                for particle in axes(u, 2)
                    # Apply deformation
                    u[1:2, particle] = deformations[deformation](coordinates[:, particle])
                end

                v_ode = ode.u0.x[1]
                if backends[i] isa TrixiParticles.KernelAbstractions.GPU
                    u_ode = vec(u)
                else
                    u_ode = TrixiParticles.ThreadedBroadcastArray(vec(u);
                                                                  parallelization_backend=backends[i])
                end

                @test typeof(v_ode) == typeof(u_ode)
                @test length(v_ode) == length(u_ode)

                #### Verification for the particle in the middle
                particle = 41

                dv_ode = zero(v_ode)
                TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

                dv = TrixiParticles.wrap_v(dv_ode, system_, semi)

                @test isapprox(dv[:, particle], dv_expected_41[deformation],
                               rtol=sqrt(eps()), atol=sqrt(eps()))
            end
        end
    end

    @testset verbose=true "Bond-Associated Quadrature" begin
        coordinate_range = 0.1:0.1:0.5
        coordinates = hcat(collect.(Iterators.product(coordinate_range,
                                                      coordinate_range))...)
        n_particles = size(coordinates, 2)
        mass = fill(10.0, n_particles)
        density = fill(1000.0, n_particles)
        young_modulus = collect(range(2.0, 3.0, length=n_particles))
        poisson_ratio = fill(0.25, n_particles)
        initial_condition = InitialCondition(; coordinates, mass, density)
        system = TotalLagrangianSPHSystem(initial_condition;
                                          smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                          smoothing_length=0.12, young_modulus,
                                          poisson_ratio,
                                          model=BondAssociatedTLSPHModel())
        semi = Semidiscretization(system, parallelization_backend=SerialBackend())
        system = semi.systems[1]
        ode = semidiscretize(semi, (0.0, 1.0))
        v_ode = copy(ode.u0.x[1])
        u_ode = copy(ode.u0.x[2])
        u = TrixiParticles.wrap_u(u_ode, system, semi)

        for particle in axes(u, 2)
            x, y = coordinates[:, particle]
            u[:, particle] .= (1.05 * x + 0.1 * y + 0.03 * x * y,
                               -0.04 * x + 0.95 * y + 0.02 * x^2)
        end

        dv_ode = zero(v_ode)
        TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
        dv = TrixiParticles.wrap_v(dv_ode, system, semi)

        # Reconstruct the C1 reproducing-kernel quantities from Peridynamics.jl directly.
        volumes = system.mass ./ system.material_density
        weights = zeros(n_particles, n_particles)
        gradient_weights = zeros(2, n_particles, n_particles)
        weighted_volume_reference = zeros(n_particles)
        moment_matrices = zeros(2, 2, n_particles)
        system_coords = TrixiParticles.initial_coordinates(system)
        neighborhood_search = TrixiParticles.get_neighborhood_search(system, semi)

        for particle in TrixiParticles.eachparticle(system)
            TrixiParticles.foreach_neighbor(system_coords, system_coords,
                                            neighborhood_search, SerialBackend(),
                                            particle) do _, neighbor, initial_pos_diff,
                                                         initial_distance
                initial_distance < sqrt(eps(system.smoothing_length^2)) && return

                grad_kernel = TrixiParticles.smoothing_kernel_grad_unsafe(system,
                                                                          initial_pos_diff,
                                                                          initial_distance,
                                                                          particle)
                initial_bond = -initial_pos_diff
                weight = -dot(grad_kernel, initial_pos_diff) / initial_distance^2
                weights[neighbor, particle] = weight
                weighted_volume_reference[particle] += weight * volumes[neighbor]
                moment_matrices[:, :, particle] .+= weight * volumes[neighbor] *
                                                    initial_bond * initial_bond'
            end

            moment_matrix_inv = inv(moment_matrices[:, :, particle])
            for neighbor in TrixiParticles.eachparticle(system)
                initial_bond = coordinates[:, neighbor] - coordinates[:, particle]
                gradient_weights[:, neighbor, particle] .= weights[neighbor, particle] *
                                                           volumes[neighbor] *
                                                           moment_matrix_inv * initial_bond
            end
        end

        deformation_gradients = zeros(2, 2, n_particles)
        for particle in TrixiParticles.eachparticle(system)
            deformation_gradients[:, :, particle] .= Matrix{Float64}(I, 2, 2)
            for neighbor in TrixiParticles.eachparticle(system)
                initial_bond = coordinates[:, neighbor] - coordinates[:, particle]
                current_bond = u[:, neighbor] - u[:, particle]
                displacement_bond = current_bond - initial_bond
                deformation_gradients[:, :, particle] .+=
                    displacement_bond * gradient_weights[:, neighbor, particle]'
            end
        end

        stress_integrals = zeros(2, 2, n_particles)
        bond_stresses = zeros(2, 2, n_particles, n_particles)
        for particle in TrixiParticles.eachparticle(system)
            for neighbor in TrixiParticles.eachparticle(system)
                weight = weights[neighbor, particle]
                iszero(weight) && continue

                initial_bond = coordinates[:, neighbor] - coordinates[:, particle]
                current_bond = u[:, neighbor] - u[:, particle]
                initial_distance = norm(initial_bond)
                F_average = (deformation_gradients[:, :, particle] +
                             deformation_gradients[:, :, neighbor]) / 2
                F_bond = F_average +
                         (current_bond - F_average * initial_bond) *
                         (initial_bond' / initial_distance^2)
                P_bond = TrixiParticles.pk1_stress_tensor(F_bond, system, particle)
                bond_stresses[:, :, neighbor, particle] .= P_bond
                transverse_projection = I -
                                        initial_bond * initial_bond' / initial_distance^2
                quadrature_weight = weight *
                                    (1 / (2 * weighted_volume_reference[particle]) +
                                     1 / (2 * weighted_volume_reference[neighbor])) *
                                    volumes[neighbor]
                stress_integrals[:, :, particle] .+=
                    quadrature_weight * P_bond * transverse_projection
            end
        end

        @test system.cache.weighted_volume ≈ weighted_volume_reference
        for particle in TrixiParticles.eachparticle(system)
            @test TrixiParticles.deformation_gradient(system, particle) ≈
                  deformation_gradients[:, :, particle]
            @test TrixiParticles.bond_associated_stress_integral(system, particle) ≈
                  stress_integrals[:, :, particle]
        end

        # Evaluate the directed-bond force updates from Peridynamics.jl directly.
        force_density = zeros(size(dv))
        for particle in TrixiParticles.eachparticle(system)
            for neighbor in TrixiParticles.eachparticle(system)
                weight = weights[neighbor, particle]
                iszero(weight) && continue

                initial_bond = coordinates[:, neighbor] - coordinates[:, particle]
                initial_distance = norm(initial_bond)
                force_state = weight /
                              (weighted_volume_reference[particle] * initial_distance^2) *
                              (bond_stresses[:, :, neighbor, particle] * initial_bond) +
                              stress_integrals[:, :, particle] *
                              gradient_weights[:, neighbor, particle] / volumes[neighbor]

                force_density[:, particle] += force_state * volumes[neighbor]
                force_density[:, neighbor] -= force_state * volumes[particle]
            end
        end

        dv_reference = force_density ./ reshape(system.material_density, 1, :)
        @test dv ≈ dv_reference rtol=2e-13 atol=2e-13
        @test vec(sum(dv .* reshape(system.mass, 1, :), dims=2)) ≈ zeros(2) atol=1e-13

        gpu_system = TotalLagrangianSPHSystem(initial_condition;
                                              smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                              smoothing_length=0.12, young_modulus,
                                              poisson_ratio,
                                              model=BondAssociatedTLSPHModel())
        gpu_backend = TrixiParticles.KernelAbstractions.CPU()
        gpu_semi = Semidiscretization(gpu_system; parallelization_backend=gpu_backend)
        gpu_ode = semidiscretize(gpu_semi, (0.0, 1.0))
        gpu_v_ode = gpu_ode.u0.x[1]
        gpu_u = zeros(size(coordinates))
        for particle in axes(gpu_u, 2)
            x, y = coordinates[:, particle]
            gpu_u[:, particle] .= (1.05 * x + 0.1 * y + 0.03 * x * y,
                                   -0.04 * x + 0.95 * y + 0.02 * x^2)
        end
        gpu_u_ode = vec(gpu_u)
        gpu_dv_ode = zero(gpu_v_ode)
        TrixiParticles.kick!(gpu_dv_ode, gpu_v_ode, gpu_u_ode, gpu_ode.p, 0.0)
        gpu_dv = TrixiParticles.wrap_v(gpu_dv_ode, gpu_semi.systems[1], gpu_semi)
        @test gpu_dv ≈ dv_reference rtol=2e-13 atol=2e-13

        # A uniform deformation must be reproduced exactly in the interior.
        affine_map = [1.1 0.2; -0.1 0.9]
        for particle in axes(u, 2)
            u[:, particle] .= affine_map * coordinates[:, particle]
        end
        TrixiParticles.update_tlsph_positions!(system, u, semi)
        TrixiParticles.compute_stress!(system, system.model, semi)
        @test TrixiParticles.deformation_gradient(system, 13) ≈ affine_map
        P = TrixiParticles.pk1_stress_tensor(affine_map, system, 13)
        sigma = TrixiParticles.cauchy_stress(system)[:, :, 13]
        @test sigma ≈ P * affine_map' / det(affine_map)
    end
end;
