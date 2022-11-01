@testset "Deformation Gradient" begin
    @testset "Unit Tests" begin
        # This is a proof of concept to show that mocking in Julia unit tests works without
        # the need for any mocking packages or modifying source code.

        u = [[2 3;  0 0], # coords[neighbor] - coords[particle] = (1, 0)
            [6 8; -1 3]] # coords[neighbor] - coords[particle] = (2, 4)

        initial_coordinates = [[5  5; 3  4], # initial_coords[particle] - initial_coords[neighbor] = (0, -1)
                            [0 -3; 0 -5]] # initial_coords[particle] - initial_coords[neighbor] = (3, 5)

        expected = [[0 -0.5; 0 0], # Tensor product of the two vectors above, multiplied by the volume of 0.5
                    [3    5; 6 10] / sqrt(34)] # Same as above but divided by the length of the vector (3, 5)


        # This is a true unit test where we mock all function calls and don't need
        # a real semidiscretization object.
        # We replace all objects that we don't need by objects of the type Val{:mock_something},
        # for which we define specific behaviour below.
        @testset "Tensor Product $i" for i in 1:2
            #### Setup
            particle = 1
            neighbors = [1, 2]
            mass = 2.0
            density = 4.0

            correction_matrix = [1 0; 0 1]

            # This will cause the computed gradient
            # to be equal to `initial_coords[particle] - initial_coords[neighbor]`
            kernel_derivative = 1.0

            #### Mocking
            # Mock the semidiscretization
            semi = Val(:mock_semi)
            Pixie.ndims(::Val{:mock_semi}) = 2

            # All @unpack calls should return another mock object of the type Val{:mock_property_name}
            Base.getproperty(::Val{:mock_semi}, f::Symbol) = Val(Symbol("mock_" * string(f)))

            # For the cache, we want to have some real matrices as properties as opposed to only mock objects
            function Base.getproperty(::Val{:mock_cache}, f::Symbol)
                if f === :initial_coordinates
                    return initial_coordinates[i]
                elseif f === :correction_matrix
                    return correction_matrix
                end

                # For all other properties, return mock objects
                return Val(Symbol("mock_" * string(f)))
            end

            Pixie.eachneighbor(_, _, _, ::Val{:mock_semi}) = neighbors
            Base.eltype(::Val{:mock_mass}) = Float64
            Base.getindex(::Val{:mock_mass}, ::Int64) = mass
            Base.getindex(::Val{:mock_solid_density}, ::Int64) = density

            Pixie.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _) = kernel_derivative

            #### Verification
            # We expect the tensor product of (1, 0) and (0, -1), multiplied by the volume of 0.5
            @test Pixie.deformation_gradient(u[i], 1, 1, particle, semi) == expected[i][1, 1]
            @test Pixie.deformation_gradient(u[i], 1, 2, particle, semi) == expected[i][1, 2]
            @test Pixie.deformation_gradient(u[i], 2, 1, particle, semi) == expected[i][2, 1]
            @test Pixie.deformation_gradient(u[i], 2, 2, particle, semi) == expected[i][2, 2]
        end
    end

    @testset "Integration Tests" begin
        deformations = Dict(
            "stretch x" => [2.0 0.0; 0.0 1.0],
            "stretch both" => [2.0 0.0; 0.0 3.0],
            "rotation" => [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)]
        )

        # We generate a grid of particles, apply a deformation, and verify that the computed
        # deformation gradient matches the deformation matrix.
        @testset "Deformation Function: $(deformation[1])" for deformation in deformations
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
            semi = SPHSolidSemidiscretization{2}(particle_masses, particle_densities, SummationDensity(),
                                                 smoothing_kernel, smoothing_length)

            tspan = (0.0, 1.0)
            semidiscretize(semi, particle_coordinates, particle_velocities, tspan)

            # Apply the deformation matrix
            u = copy(particle_coordinates)
            for particle in axes(u, 2)
                u[:, particle] = deformation[2] * u[:, particle]
            end

            # Compute the deformation gradient for the particle in the middle
            J = [Pixie.deformation_gradient(u, 1, 1, 41, semi) Pixie.deformation_gradient(u, 1, 2, 41, semi);
                 Pixie.deformation_gradient(u, 2, 1, 41, semi) Pixie.deformation_gradient(u, 2, 2, 41, semi)]

            # Verification
            @test J ≈ deformation[2]
        end
    end
end
