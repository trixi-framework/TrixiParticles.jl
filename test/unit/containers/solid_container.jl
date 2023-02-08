@testset verbose=true "Deformation Gradient" begin
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
