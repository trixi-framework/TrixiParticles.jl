@testset verbose=true "Solid RHS" begin
    @testset "calc_dv!" begin
        # Pass specific pk1 and pos_diff to calc_dv! and verify with
        # values calculated by hand
        pk1_particle_corrected = [
            [2.0 0.0; 0.0 3.0],
            [0.230015 0.0526099; 0.348683 0.911251],
            [0.364281 0.763916; 0.433868 0.9755],
        ]
        pk1_neighbor_corrected = [
            zeros(2, 2),
            [0.794769 0.113958; 0.1353 0.741521],
            [0.97936 0.302584; 0.333461 0.642575],
        ]
        initial_pos_diff = [[0.1, 0.0], [0.5, 0.3], [1.0, 0.37]]
        particle = [1, 3, 10]
        neighbor = [10, 4, 9]
        dv_particle_expected = [
            [2.0, 0.0],
            [0.07118137645953881, 0.11640973221047637],
            [0.01870315713691948, 0.014067976038520915],
        ]

        @testset "Test $i" for i in 1:3
            #### Setup
            mass = ones(Float64, 10)
            kernel_deriv = 1.0
            initial_distance = norm(initial_pos_diff[i])

            # Density equals the ID of the particle
            material_density = 1:10

            #### Mocking
            # Mock the container
            container = Val(:mock_container_dv)
            TrixiParticles.ndims(::Val{:mock_container_dv}) = 2

            # All @unpack calls should return another mock object of the type Val{:mock_property_name},
            # but we want to have some predefined values as properties
            function Base.getproperty(::Val{:mock_container_dv}, f::Symbol)
                if f === :material_density
                    return material_density
                elseif f === :mass
                    return mass
                end

                # For all other properties, return mock objects
                return Val(Symbol("mock_" * string(f)))
            end

            TrixiParticles.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _) = kernel_deriv

            function TrixiParticles.get_pk1_corrected(particle_, ::Val{:mock_container_dv})
                if particle_ == particle[i]
                    return pk1_particle_corrected[i]
                end
                return pk1_neighbor_corrected[i]
            end

            #### Verification
            dv = zeros(ndims(container), 10)
            dv_expected = copy(dv)
            dv_expected[:, particle[i]] = dv_particle_expected[i]

            TrixiParticles.calc_dv!(dv, particle[i], neighbor[i], initial_pos_diff[i],
                           initial_distance,
                           container, container)

            @test dv ≈ dv_expected
        end
    end

    @testset "interact!" begin
        # Use the same setup as in the unit test above for calc_dv!
        pk1_particle_corrected = [
            [2.0 0.0; 0.0 3.0],
            [0.230015 0.0526099; 0.348683 0.911251],
            [0.364281 0.763916; 0.433868 0.9755],
            [0.503965 0.125224; 0.739591 0.0323651],
        ]
        pk1_neighbor_corrected = [
            zeros(2, 2),
            [0.794769 0.113958; 0.1353 0.741521],
            [0.97936 0.302584; 0.333461 0.642575],
            [0.503965 0.125224; 0.739591 0.0323651],
        ]
        # Create initial coordinates so that
        # initial_pos_diff is [[0.1, 0.0], [0.5, 0.3], [1.0, 0.37], [0.0, 0.0]]
        initial_coordinates_particle = [
            [1.1, 2.0], [-1.5, 5.0], [-1.5, -6.43], [120.0, 32.2],
        ]
        initial_coordinates_neighbor = [
            [1.0, 2.0], [-2.0, 4.7], [-2.5, -6.8], [120.0, 32.2],
        ]
        particle = [1, 3, 10, 7]
        neighbor = [10, 4, 9, 7]
        dv_particle_expected = [
            [2.0, 0.0],
            [0.07118137645953881, 0.11640973221047637],
            [0.01870315713691948, 0.014067976038520915],
            [0.0, 0.0],
        ]

        @testset "Test $i" for i in 1:4
            #### Setup
            each_moving_particle = [particle[i]] # Only calculate dv for this one particle
            eachparticle = [particle[i], neighbor[i]]
            eachneighbor = [neighbor[i], particle[i]]
            initial_coordinates = 1000 * ones(2, 10) # Just something that's not zero to catch errors
            initial_coordinates[:, particle[i]] = initial_coordinates_particle[i]
            initial_coordinates[:, neighbor[i]] = initial_coordinates_neighbor[i]
            current_coordinates = zeros(2, 10)
            pk1_corrected = 2000 * ones(2, 2, 10) # Just something that's not zero to catch errors
            pk1_corrected[:, :, particle[i]] = pk1_particle_corrected[i]
            pk1_corrected[:, :, neighbor[i]] = pk1_neighbor_corrected[i]

            # Density equals the ID of the particle
            material_density = 1:10

            # Use the same setup as in the unit test above for calc_dv!
            mass = ones(Float64, 10)
            kernel_deriv = 1.0

            #### Mocking
            # Mock the container
            container = Val{:mock_container_interact}()
            TrixiParticles.ndims(::Val{:mock_container_interact}) = 2

            # @unpack calls should return predefined values or
            # another mock object of the type Val{:mock_property_name}
            function Base.getproperty(::Val{:mock_container_interact}, f::Symbol)
                if f === :initial_coordinates
                    return initial_coordinates
                elseif f === :current_coordinates
                    return current_coordinates
                elseif f === :material_density
                    return material_density
                elseif f === :pk1_corrected
                    return pk1_corrected
                elseif f === :mass
                    return mass
                elseif f === :penalty_force
                    return nothing
                end

                # For all other properties, return mock objects
                return Val(Symbol("mock_" * string(f)))
            end

            function TrixiParticles.each_moving_particle(::Val{:mock_container_interact})
                each_moving_particle
            end
            TrixiParticles.eachparticle(::Val{:mock_container_interact}) = eachparticle
            TrixiParticles.eachneighbor(_, ::Val{:nhs}) = eachneighbor
            TrixiParticles.compact_support(::Val{:mock_smoothing_kernel}, _) = 100.0

            function TrixiParticles.get_pk1_corrected(particle_, ::Val{:mock_container_dv})
                if particle_ == particle[i]
                    return pk1_particle_corrected[i]
                end
                return pk1_neighbor_corrected[i]
            end

            TrixiParticles.add_acceleration!(_, _, ::Val{:mock_container_interact}) = nothing
            TrixiParticles.kernel_deriv(::Val{:mock_smoothing_kernel}, _, _) = kernel_deriv

            #### Verification
            dv = zeros(ndims(container), 10)
            dv_expected = copy(dv)
            dv_expected[:, particle[i]] = dv_particle_expected[i]

            TrixiParticles.interact_solid_solid!(dv, Val(:nhs), container, container)

            @test dv ≈ dv_expected
        end
    end
end
