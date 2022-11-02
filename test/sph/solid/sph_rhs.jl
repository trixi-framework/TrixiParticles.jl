@testset "Solid RHS" begin
    @testset "Unit Tests" begin
        @testset "calc_dv!" begin
            pk1_particle_corrected = [
                [2.0 0.0; 0.0 3.0],
                [0.230015 0.0526099; 0.348683 0.911251],
                [0.364281 0.763916; 0.433868 0.9755]
            ]
            pk1_neighbor_corrected = [
                zeros(2, 2),
                [0.794769 0.113958; 0.1353 0.741521],
                [0.97936 0.302584; 0.333461 0.642575]
            ]
            initial_pos_diff = [[0.1, 0.0], [0.5, 0.3], [1.0, 0.37]]
            particle = [1, 3, 10]
            neighbor = [10, 4, 9]
            dv_expected = [
                [2.0, 0.0],
                [0.07118137645953881, 0.11640973221047637],
                [0.01870315713691948, 0.014067976038520915]
            ]

            @testset "Test $i" for i in 1:3
                #### Setup
                neighbor_mass = 1.0
                kernel_deriv = 1.0
                initial_distance = norm(initial_pos_diff[i])

                # Density equals the ID of the particle
                solid_density = 1:10

                #### Mocking
                # Mock the semidiscretization
                semi = Val(:mock_semi)
                Pixie.ndims(::Val{:mock_semi}) = 2

                # All @unpack calls should return another mock object of the type Val{:mock_property_name}
                Base.getproperty(::Val{:mock_semi}, f::Symbol)  = Val(Symbol("mock_" * string(f)))

                # For the cache, we want to have some real matrices as properties as opposed to only mock objects
                function Base.getproperty(::Val{:mock_cache}, f::Symbol)
                    if f === :solid_density
                        return solid_density
                    end

                    # For all other properties, return mock objects
                    return Val(Symbol("mock_" * string(f)))
                end

                Base.getindex(::Val{:mock_mass}, ::Int64) = neighbor_mass

                Pixie.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _) = kernel_deriv

                #### Verification
                du = zeros(2 * ndims(semi), 10)
                du_expected = copy(du)
                du_expected[3:4, particle[i]] = dv_expected[i]

                Pixie.calc_dv!(du, particle[i], neighbor[i], initial_pos_diff[i], initial_distance,
                         pk1_particle_corrected[i], pk1_neighbor_corrected[i], semi)

                @test du ≈ du_expected
            end
        end

        @testset "rhs!" begin
            # Use the same setup as in the unit test above for calc_dv!
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
                [0.107677 0.623585; 0.593918 0.638598]
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
            dv_expected = [
                [2.0, 0.0],
                [0.07118137645953881, 0.11640973221047637],
                [0.01870315713691948, 0.014067976038520915],
                [0.0, 0.0]
            ]

            @testset "Test $i" for i in 1:4
                #### Setup
                eachparticle = [particle[i]] # Only calculate dv for this one particle
                eachneighbor = [neighbor[i], particle[i]]
                correction_matrix = 100 * ones(2, 2, 10) # Just something that's not zero to catch errors
                correction_matrix[:, :, particle[i]] = Matrix(I, 2, 2)
                correction_matrix[:, :, neighbor[i]] = Matrix(I, 2, 2)
                initial_coordinates = 1000 * ones(2, 10) # Just something that's not zero to catch errors
                initial_coordinates[:, particle[i]] = initial_coordinates_particle[i]
                initial_coordinates[:, neighbor[i]] = initial_coordinates_neighbor[i]
                current_coordinates = zeros(2, 10)

                # Density equals the ID of the particle
                solid_density = 1:10

                # Use the same setup as in the unit test above for calc_dv!
                neighbor_mass = 1.0
                kernel_deriv = 1.0

                #### Mocking
                # Mock the semidiscretization
                semi = Val{:mock_semi_rhs!}()
                Pixie.ndims(::Val{:mock_semi_rhs!}) = 2

                # All @unpack calls should return another mock object of the type Val{:mock_property_name}
                Base.getproperty(::Val{:mock_semi_rhs!}, f::Symbol)  = Val(Symbol("mock_" * string(f)))

                # For the cache, we want to have some real matrices as properties as opposed to only mock objects
                function Base.getproperty(::Val{:mock_cache}, f::Symbol)
                    if f === :initial_coordinates
                        return initial_coordinates
                    elseif f === :current_coordinates
                        return current_coordinates
                    elseif f === :correction_matrix
                        return correction_matrix
                    elseif f === :solid_density
                        return solid_density
                    end

                    # For all other properties, return mock objects
                    return Val(Symbol("mock_" * string(f)))
                end

                Pixie.each_moving_particle(_, ::Val{:mock_semi_rhs!}) = eachparticle
                Pixie.eachneighbor(_, _, _, ::Val{:mock_semi_rhs!}) = eachneighbor
                Pixie.compact_support(::Val{:mock_smoothing_kernel}, _) = 100.0

                function Pixie.pk1_stress_tensor(_, particle_, ::Val{:mock_semi_rhs!})
                    if particle_ == particle[i]
                        return pk1_particle_corrected[i]
                    elseif particle_ == neighbor[i]
                        return pk1_neighbor_corrected[i]
                    end
                end

                Pixie.calc_gravity!(_, _, ::Val{:mock_semi_rhs!}) = nothing

                # Use the same mocking as in the unit test above for calc_dv!
                Base.getindex(::Val{:mock_mass}, ::Int64) = neighbor_mass
                Pixie.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _) = kernel_deriv

                #### Verification
                du = zeros(2 * ndims(semi), 10)
                u = copy(du)
                du_expected = copy(du)
                du_expected[3:4, particle[i]] = dv_expected[i]

                Pixie.rhs_solid!(du, u, semi, 0.0)

                @test du ≈ du_expected
            end
        end
    end

    @testset "Integration Tests" begin
        deformations = Dict(
            "rotation" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * x,
            "stretch both" => x -> [2.0 0.0; 0.0 3.0] * x,
            "rotate and stretch" => x -> [cos(0.3) -sin(0.3); sin(0.3) cos(0.3)] * [2.0 0.0; 0.0 3.0] * x,
            "nonlinear stretching" => x -> [x[1]^2, x[2]]
        )

        # The acceleration in the first three should be zero (linear stretching)
        # The fourth one is calculated by hand
        expected_du_41 = Dict(
            "rotation" => [1.0, -2.0, 0.0, 0.0],
            "stretch both" => [1.0, -2.0, 0.0, 0.0],
            "rotate and stretch" => [1.0, -2.0, 0.0, 0.0],
            "nonlinear stretching" => [1.0, -2.0, 10/1000^2 * 1.5400218087591082 * 324.67072684047224 * 1.224, 0.0]
        )

        @testset "Deformation Function: $deformation" for deformation in keys(deformations)
            J = deformations[deformation]
            u = zeros(4, 81)
            u[3, :] .= 1.0
            u[4, :] .= -2.0

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

            smoothing_length = 0.07
            smoothing_kernel = SchoenbergCubicSplineKernel{2}()
            search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)
            semi = SPHSolidSemidiscretization{2}(particle_masses, particle_densities, SummationDensity(),
                                                smoothing_kernel, smoothing_length,
                                                E, nu,
                                                neighborhood_search=SpatialHashingSearch{2}(search_radius))

            tspan = (0.0, 1.0)
            semidiscretize(semi, particle_coordinates, particle_velocities, tspan)

            # Apply the deformation matrix
            for particle in axes(u, 2)
                # Apply deformation
                u[1:2, particle] = deformations[deformation](particle_coordinates[:, particle])
            end

            #### Verification for the particle in the middle
            particle = 41

            du = zeros(2 * ndims(semi), 81)
            Pixie.rhs!(du, u, semi, 0.0)

            @test du[:, particle] ≈ expected_du_41[deformation]
        end
    end
end
