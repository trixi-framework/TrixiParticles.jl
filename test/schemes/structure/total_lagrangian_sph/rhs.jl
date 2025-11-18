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
            struct MockSystem <: TrixiParticles.AbstractSystem{2} end
            system = MockSystem()

            function TrixiParticles.initial_coordinates(::MockSystem)
                return initial_coordinates
            end

            # Unpack calls should return predefined values or
            # another mock object of the type Val{:mock_property_name}.
            function Base.getproperty(::MockSystem, f::Symbol)
                if f === :current_coordinates
                    return current_coordinates
                elseif f === :material_density
                    return material_density
                elseif f === :pk1_rho2
                    return pk1_rho2
                elseif f === :mass
                    return mass
                elseif f === :penalty_force
                    return nothing
                elseif f === :viscosity
                    return nothing
                elseif f === :buffer
                    return nothing
                end

                # For all other properties, return mock objects
                return Val(Symbol("mock_" * string(f)))
            end

            TrixiParticles.eachparticle(::MockSystem) = eachparticle
            TrixiParticles.each_integrated_particle(::MockSystem) = each_integrated_particle
            TrixiParticles.smoothing_length(::MockSystem, _) = eps()

            function TrixiParticles.add_acceleration!(_, _, ::MockSystem)
                return nothing
            end
            TrixiParticles.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _) = kernel_deriv
            TrixiParticles.compact_support(::MockSystem, ::MockSystem) = 1000.0
            function TrixiParticles.current_coords(system::MockSystem, particle)
                return TrixiParticles.current_coords(initial_coordinates, system, particle)
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
            system = TotalLagrangianSPHSystem(initial_condition,
                                              smoothing_kernel, smoothing_length, E, nu)
            tspan = (0.0, 1.0)

            names = ["CPU code", "GPU code emulated on the CPU"]
            backends = [SerialBackend(), TrixiParticles.KernelAbstractions.CPU()]
            @testset "$(names[i])" for i in eachindex(names)
                semi = Semidiscretization(system, parallelization_backend=backends[i])
                ode = semidiscretize(semi, tspan)

                # Apply the deformation matrix
                for particle in axes(u, 2)
                    # Apply deformation
                    u[1:2, particle] = deformations[deformation](coordinates[:, particle])
                end

                v_ode = ode.u0.x[1]
                if backends[i] isa TrixiParticles.KernelAbstractions.Backend
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

                dv = TrixiParticles.wrap_v(dv_ode, system, semi)

                @test isapprox(dv[:, particle], dv_expected_41[deformation],
                               rtol=sqrt(eps()), atol=sqrt(eps()))
            end
        end
    end
end;
