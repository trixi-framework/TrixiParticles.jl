@testset verbose=true "TotalLagrangianSPHSystem" begin
    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructor" begin
        coordinates_ = [
            [1.0 2.0
             1.0 2.0],
            [1.0 2.0
             1.0 2.0
             1.0 2.0]
        ]
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]
            mass = [1.25, 1.5]
            material_densities = [990.0, 1000.0]
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = i + 1
            smoothing_length = 0.362
            # Make both Lamé constants equal to 1
            nu = 0.25
            E = 2.5
            boundary_model = Val(:boundary_model)

            initial_condition = InitialCondition(; coordinates, mass,
                                                 density=material_densities)
            system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                              smoothing_length, E, nu,
                                              boundary_model=boundary_model)

            @test system isa TotalLagrangianSPHSystem
            @test ndims(system) == NDIMS
            @test system.initial_condition == initial_condition
            @test system.initial_coordinates == coordinates
            @test system.current_coordinates == coordinates
            @test system.mass == mass
            @test system.material_density == material_densities
            @test system.n_integrated_particles == 2
            @test system.young_modulus == E
            @test system.poisson_ratio == nu
            @test system.lame_lambda == 1.0
            @test system.lame_mu == 1.0
            @test system.smoothing_kernel == smoothing_kernel
            @test TrixiParticles.initial_smoothing_length(system) == smoothing_length
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test system.boundary_model == boundary_model
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        material_densities = [990.0, 1000.0]
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.362
        # Make both Lamé constants equal to 1
        nu = 0.25
        E = 2.5
        boundary_model = Val(:boundary_model)

        initial_condition = InitialCondition(; coordinates, mass,
                                             density=material_densities)
        system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                          smoothing_length, E, nu,
                                          boundary_model=boundary_model)

        show_compact = "TotalLagrangianSPHSystem{2}(Val{:smoothing_kernel}(), " *
                       "[0.0, 0.0], Val{:boundary_model}(), nothing, nothing) with 2 particles"
        @test repr(system) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ TotalLagrangianSPHSystem{2}                                                                      │
        │ ═══════════════════════════                                                                      │
        │ total #particles: ………………………………… 2                                                                │
        │ #clamped particles: …………………………… 0                                                                │
        │ Young's modulus: …………………………………… 2.5                                                              │
        │ Poisson ratio: ………………………………………… 0.25                                                             │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        │ boundary model: ……………………………………… Val{:boundary_model}()                                           │
        │ penalty force: ………………………………………… nothing                                                          │
        │ viscosity: …………………………………………………… nothing                                                          │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box

        E = [1.2, 3.4]
        nu = [0.2, 0.4]
        system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                          smoothing_length, E, nu,
                                          boundary_model=boundary_model)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ TotalLagrangianSPHSystem{2}                                                                      │
        │ ═══════════════════════════                                                                      │
        │ total #particles: ………………………………… 2                                                                │
        │ #clamped particles: …………………………… 0                                                                │
        │ Young's modulus: …………………………………… min = 1.2, max = 3.4                                             │
        │ Poisson ratio: ………………………………………… min = 0.2, max = 0.4                                             │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        │ boundary model: ……………………………………… Val{:boundary_model}()                                           │
        │ penalty force: ………………………………………… nothing                                                          │
        │ viscosity: …………………………………………………… nothing                                                          │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end

    @testset verbose=true "Deformation Gradient" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "Manual Calculation" begin
            # Compare against manually calculated values.
            #
            # This is a proof of concept showing that mocking in Julia unit tests works without
            # the need for any mocking packages or modifying source code.
            #
            # We construct a lowest-level unit test where we mock all function calls and don't use
            # a real system object.
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

                correction_matrix = [1 0; 0 1;;;
                                     1 0; 0 1]

                # This will cause the computed gradient
                # to be equal to `initial_coords[particle] - initial_coords[neighbor]`
                kernel_derivative = 1.0

                #### Mocking
                # Mock the system
                system = Val(:mock_system_tensor)
                TrixiParticles.ndims(::Val{:mock_system_tensor}) = 2
                Base.ntuple(f, ::Symbol) = ntuple(f, 2) # Make `extract_svector` work
                function TrixiParticles.current_coords(system::Val{:mock_system_tensor},
                                                       particle)
                    return TrixiParticles.extract_svector(current_coordinates[i], Val(2),
                                                          particle)
                end

                function TrixiParticles.initial_coordinates(::Val{:mock_system_tensor})
                    return initial_coordinates[i]
                end

                TrixiParticles.smoothing_length(::Val{:mock_system_tensor}, _) = 0.12

                # All unpack calls should return another mock object
                # of the type `Val{:mock_property_name}`, but we want to have some real matrices
                # as properties as opposed to only mock objects.
                function Base.getproperty(::Val{:mock_system_tensor}, f::Symbol)
                    if f === :correction_matrix
                        return correction_matrix
                    elseif f === :mass
                        return mass
                    end

                    # For all other properties, return mock objects.
                    return Val(Symbol("mock_" * string(f)))
                end
                function TrixiParticles.compact_support(::Val{:mock_system_tensor},
                                                        ::Val{:mock_system_tensor})
                    return Inf
                end

                Base.getindex(::Val{:mock_material_density}, ::Int64) = density

                function TrixiParticles.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _)
                    return kernel_derivative
                end
                Base.eps(::Type{Val{:mock_smoothing_length}}) = eps()
                semi = DummySemidiscretization()

                # Compute deformation gradient
                deformation_grad = ones(2, 2, 2)
                TrixiParticles.calc_deformation_grad!(deformation_grad, system, semi)

                #### Verification
                @test deformation_grad[:, :, particle] == expected[i]
            end
        end

        @testset verbose=true "Deformation Functions" begin
            # We generate a grid of particles, apply a deformation, and verify that the computed
            # deformation gradient matches the deformation matrix.
            deformations = Dict(
                "Stretch x" => x -> [2.0 0.0; 0.0 1.0] * x,
                "Stretch Both" => x -> [2.0 0.0; 0.0 3.0] * x,
                "Rotation" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * x,
                "Nonlinear Stretching" => x -> [x[1]^2, x[2]])

            deformation_gradients = Dict(
                "Stretch x" => [2.0 0.0; 0.0 1.0],
                "Stretch Both" => [2.0 0.0; 0.0 3.0],
                "Rotation" => [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)],
                "Nonlinear Stretching" => [1.0 0.0; 0.0 1.0])

            @testset "$deformation_name" for deformation_name in keys(deformations)
                deformation = deformations[deformation_name]

                # 9 x 9 grid of particles with spacing 0.1
                range = 0.1:0.1:0.9
                coordinates = hcat(collect.(Iterators.product(range, range))...)

                n_particles_per_dimension = (9, 9)
                mass = 10 * ones(Float64, prod(n_particles_per_dimension))
                density = 1000 * ones(Float64, prod(n_particles_per_dimension))

                smoothing_length = 0.12
                smoothing_kernel = SchoenbergCubicSplineKernel{2}()
                search_radius = TrixiParticles.compact_support(smoothing_kernel,
                                                               smoothing_length)

                initial_condition = InitialCondition(; coordinates, mass, density)
                system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                                  smoothing_length, 1.0, 1.0)
                semi = DummySemidiscretization()

                TrixiParticles.initialize!(system, semi)

                # Apply the deformation matrix
                for particle in TrixiParticles.eachparticle(system)
                    new_coords = deformation(system.initial_coordinates[:, particle])
                    system.current_coordinates[:, particle] = new_coords
                end

                # Compute the deformation gradient for the particle in the middle
                TrixiParticles.calc_deformation_grad!(system.deformation_grad,
                                                      system, semi)
                J = TrixiParticles.deformation_gradient(system, 41)

                #### Verification
                @test J ≈ deformation_gradients[deformation_name]
            end
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Stress tensors" begin
        deformations = Dict("rotation" => [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)],
                            "stretch both" => [2.0 0.0; 0.0 3.0],
                            "rotate and stretch" => [cos(0.3) -sin(0.3);
                                                     sin(0.3) cos(0.3)] *
                                                    [2.0 0.0; 0.0 3.0]
                            )

        expected_pk2 = Dict("rotation" => zeros(2, 2), # No stress in rotations only
                            "stretch both" => [8.5 0.0; 0.0 13.5], # Calculated by hand
                            "rotate and stretch" => [8.5 0.0; 0.0 13.5])

        # Just deformation * expected_pk2
        expected_pk1 = Dict("rotation" => zeros(2, 2),
                            "stretch both" => [17.0 0.0; 0.0 40.5],
                            "rotate and stretch" => [cos(0.3) -sin(0.3);
                                                     sin(0.3) cos(0.3)] *
                                                    [2.0 0.0; 0.0 3.0] *
                                                    [8.5 0.0; 0.0 13.5]
                            )

        @testset "Deformation Function: $deformation" for deformation in keys(deformations)
            #### Setup
            J = deformations[deformation]
            lame_lambda = 1.0
            lame_mu = 1.0

            #### Mocking
            # It is easier to mock the system and specify the Lamé constants
            # and deformation gradient than to actually construct a system.
            system = Val(:mock_system)
            TrixiParticles.smoothing_length(::Val{:mock_system}, _) = 0.12
            TrixiParticles.deformation_gradient(::Val{:mock_system}, _) = J

            # All unpack calls should return another mock object
            # of the type `Val{:mock_property_name}`, but we want to have the actual
            # Lamé constants as properties.
            function Base.getproperty(::Val{:mock_system}, f::Symbol)
                if f === :lame_lambda
                    return lame_lambda
                elseif f === :lame_mu
                    return lame_mu
                end

                # For all other properties, return mock objects
                return Val(Symbol("mock_" * string(f)))
            end

            #### Verification
            @test TrixiParticles.pk2_stress_tensor(J, lame_lambda, lame_mu, 1) ≈
                  expected_pk2[deformation]
            @test TrixiParticles.pk1_stress_tensor(system, 1) ≈ expected_pk1[deformation]
        end
    end

    @testset verbose=true "write_u0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        material_densities = [990.0, 1000.0]
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362
        # Make both Lamé constants equal to 1
        nu = 0.25
        E = 2.5
        boundary_model = Val(:boundary_model)
        TrixiParticles.smoothing_length(::Val{:boundary_model}, _) = smoothing_length

        initial_condition = InitialCondition(; coordinates, mass,
                                             density=material_densities)
        system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                          smoothing_length, E, nu,
                                          boundary_model=boundary_model)

        u0 = zeros(TrixiParticles.u_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_u0!(u0, system)

        @test u0 == coordinates
    end

    @testset verbose=true "write_v0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        velocity = zero(coordinates)
        mass = [1.25, 1.5]
        material_densities = [990.0, 1000.0]
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362
        # Make both Lamé constants equal to 1
        nu = 0.25
        E = 2.5
        boundary_model = Val(:boundary_model)
        TrixiParticles.smoothing_length(::Val{:boundary_model}, _) = smoothing_length

        initial_condition = InitialCondition(; coordinates, velocity, mass,
                                             density=material_densities)
        system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                          smoothing_length, E, nu,
                                          boundary_model=boundary_model)

        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == velocity
    end

    @testset verbose=true "compute_von_mises_stress" begin
        # System setup
        coordinates = [1.0 2.0; 1.0 2.0]
        velocity = zero(coordinates)
        mass = [1.25, 1.5]
        material_densities = [990.0, 1000.0]
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362
        nu = 0.25  # Poisson's ratio
        E = 2.5    # Young's modulus

        initial_condition = InitialCondition(; coordinates, velocity, mass,
                                             density=material_densities)
        system = TotalLagrangianSPHSystem(initial_condition, smoothing_kernel,
                                          smoothing_length, E, nu)

        # Initialize deformation_grad and pk1_rho2 with arbitrary values
        for particle in TrixiParticles.eachparticle(system)
            system.deformation_grad[:, :, particle] = [1.0 0.2; 0.2 1.0]
            system.pk1_rho2[:, :,
                            particle] = [1.0 0.5; 0.5 1.0] / material_densities[particle]^2
        end

        von_mises_stress = TrixiParticles.von_mises_stress(system)
        cauchy_stress = TrixiParticles.cauchy_stress(system)

        reference_stress_tensor = [1.145833 0.729167; 0.729167 1.145833;;;
                                   1.145833 0.729167; 0.729167 1.145833]

        # Verify against calculation by hand
        @test isapprox(von_mises_stress[1], 1.4257267477533202, atol=1e-14)
        @test isapprox(reference_stress_tensor, cauchy_stress, atol=1e-6)
    end
end;
