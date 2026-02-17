@testset verbose=true "`vtk2trixi`" begin
    mktempdir() do tmp_dir
        coordinates = fill(1.0, 2, 12)
        velocity = fill(2.0, 2, 12)

        expected_data = InitialCondition(coordinates=coordinates, velocity=velocity,
                                         density=1000.0, pressure=900.0,
                                         particle_spacing=0.1)

        expected_scalar = 3.0
        expected_vector = fill(expected_scalar, nparticles(expected_data))
        scalar_function(system, data, t) = expected_scalar
        vector_function(system, data, t) = fill(expected_scalar, nparticles(system))

        @testset verbose=true "`InitialCondition`" begin
            @testset verbose=true "`Float64`" begin
                trixi2vtk(expected_data; filename="tmp_initial_condition_64",
                          output_directory=tmp_dir)
                file = joinpath(tmp_dir, "tmp_initial_condition_64.vtu")
                data = vtk2trixi(file)

                @test isapprox(expected_data.coordinates, data.coordinates, rtol=1e-5)
                @test isapprox(expected_data.velocity, data.velocity, rtol=1e-5)
                @test isapprox(expected_data.density, data.density, rtol=1e-5)
                @test isapprox(expected_data.pressure, data.pressure, rtol=1e-5)
                @test all(key -> eltype(data[key]) === Float64, keys(data))
                @test eltype(data.coordinates) === Float64
            end

            @testset verbose=true "`Float32`" begin
                expected_ic_32 = InitialCondition(;
                                                  coordinates=convert.(Float32,
                                                                       coordinates),
                                                  velocity=convert.(Float32, velocity),
                                                  density=1000.0f0, pressure=900.0f0,
                                                  mass=50.0f0, particle_spacing=0.1f0)
                trixi2vtk(expected_ic_32; filename="tmp_initial_condition_32",
                          output_directory=tmp_dir)
                file = joinpath(tmp_dir, "tmp_initial_condition_32.vtu")
                data = vtk2trixi(file)

                @test isapprox(expected_ic_32.coordinates, data.coordinates, rtol=1e-5)
                @test isapprox(expected_ic_32.velocity, data.velocity, rtol=1e-5)
                @test isapprox(expected_ic_32.density, data.density, rtol=1e-5)
                @test isapprox(expected_ic_32.pressure, data.pressure, rtol=1e-5)
                @test all(key -> eltype(data[key]) === Float32, keys(data))
                @test eltype(data.coordinates) === Float32
            end

            @testset verbose=true "Custom Element Type" begin
                trixi2vtk(expected_data; filename="tmp_initial_condition_64",
                          output_directory=tmp_dir)
                file = joinpath(tmp_dir, "tmp_initial_condition_64.vtu")
                data = vtk2trixi(file, element_type=Float32, coordinates_eltype=Float32)

                @test isapprox(expected_data.coordinates, data.coordinates, rtol=1e-5)
                @test isapprox(expected_data.velocity, data.velocity, rtol=1e-5)
                @test isapprox(expected_data.density, data.density, rtol=1e-5)
                @test isapprox(expected_data.pressure, data.pressure, rtol=1e-5)
                @test all(key -> eltype(data[key]) === Float32, keys(data))
                @test eltype(data.coordinates) === Float32
            end

            @testset verbose=true "Scalar Custom Quantity" begin
                trixi2vtk(expected_data; filename="tmp_initial_condition_scalar",
                          output_directory=tmp_dir, scalar=expected_scalar)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_initial_condition_scalar.vtu"))

                @test isapprox(expected_data.coordinates, test_data.coordinates, rtol=1e-5)
                @test isapprox(expected_data.velocity, test_data.velocity, rtol=1e-5)
                @test isapprox(expected_data.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_data.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(expected_scalar, test_data.scalar, rtol=1e-5)
            end

            @testset verbose=true "Vector Custom Quantity" begin
                trixi2vtk(expected_data; filename="tmp_initial_condition_vector",
                          output_directory=tmp_dir,
                          vector=expected_vector)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_initial_condition_vector.vtu"))

                @test isapprox(expected_data.coordinates, test_data.coordinates, rtol=1e-5)
                @test isapprox(expected_data.velocity, test_data.velocity, rtol=1e-5)
                @test isapprox(expected_data.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_data.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(expected_vector, test_data.vector, rtol=1e-5)
            end

            @testset verbose=true "Custom Element Type Mixed" begin
                trixi2vtk(expected_data; filename="tmp_initial_condition_64",
                          output_directory=tmp_dir)
                file = joinpath(tmp_dir, "tmp_initial_condition_64.vtu")
                data = vtk2trixi(file, element_type=Float32, coordinates_eltype=Float64)

                @test isapprox(expected_data.coordinates, data.coordinates, rtol=1e-5)
                @test isapprox(expected_data.velocity, data.velocity, rtol=1e-5)
                @test isapprox(expected_data.density, data.density, rtol=1e-5)
                @test isapprox(expected_data.pressure, data.pressure, rtol=1e-5)
                @test all(key -> eltype(data[key]) === Float32,
                          setdiff(keys(data), (:coordinates,)))
                @test eltype(data.coordinates) === Float64
            end
        end

        @testset verbose=true "`AbstractFluidSystem`" begin
            fluid_system = EntropicallyDampedSPHSystem(expected_data,
                                                       SchoenbergCubicSplineKernel{2}(),
                                                       1.5, 1.5)

            # Overwrite values because we skip the update step
            fluid_system.cache.density .= expected_data.density

            semi = Semidiscretization(fluid_system)

            # Create random ODE solutions
            v = fill(2.0, ndims(fluid_system), nparticles(fluid_system))
            pressure = fill(3.0, nparticles(fluid_system))
            v_ode = vec([v; pressure'])
            u = fill(1.0, ndims(fluid_system), nparticles(fluid_system))
            u_ode = vec(u)
            dv_ode = zero(v_ode)
            du_ode = zero(u_ode)
            x = (; v_ode, u_ode)
            vu_ode = (; x)
            dvdu_ode = (; x=(; v_ode=dv_ode, u_ode=du_ode))

            @testset verbose=true "Scalar Custom Quantity" begin
                trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_fluid_scalar",
                          output_directory=tmp_dir, iter=1, scalar=scalar_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_scalar_1.vtu"))

                @test isapprox(u, test_data.coordinates, rtol=1e-5)
                @test isapprox(v, test_data.velocity, rtol=1e-5)
                @test isapprox(pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(fluid_system.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_scalar, test_data.scalar, rtol=1e-5)
            end

            @testset verbose=true "Vector Custom Quantity" begin
                trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_fluid_vector",
                          output_directory=tmp_dir, iter=1, vector=vector_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_vector_1.vtu"))

                @test isapprox(u, test_data.coordinates, rtol=1e-5)
                @test isapprox(v, test_data.velocity, rtol=1e-5)
                @test isapprox(pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(fluid_system.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(fill(expected_scalar, nparticles(fluid_system)),
                               test_data.vector, rtol=1e-5)
            end
        end

        @testset verbose=true "`WallBoundarySystem`" begin
            boundary_model = BoundaryModelDummyParticles(expected_data.density,
                                                         expected_data.mass,
                                                         SummationDensity(),
                                                         SchoenbergCubicSplineKernel{2}(),
                                                         1.5)

            # Overwrite values because we skip the update step
            boundary_model.pressure .= expected_data.pressure
            boundary_model.cache.density .= expected_data.density

            boundary_system = WallBoundarySystem(expected_data, boundary_model)
            semi = Semidiscretization(boundary_system)

            # Create dummy ODE solutions
            v_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            u_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            dv_ode = zero(v_ode)
            du_ode = zero(u_ode)
            x = (; v_ode, u_ode)
            vu_ode = (; x)
            dvdu_ode = (; x=(; v_ode=dv_ode, u_ode=du_ode))

            @testset verbose=true "Scalar Custom Quantity" begin
                trixi2vtk(boundary_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_boundary_scalar",
                          output_directory=tmp_dir, iter=1, scalar=scalar_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_scalar_1.vtu"))

                @test isapprox(boundary_system.coordinates, test_data.coordinates,
                               rtol=1e-5)
                # The velocity is always zero for `WallBoundarySystem`
                @test isapprox(zeros(size(test_data.velocity)), test_data.velocity,
                               rtol=1e-5)
                @test isapprox(boundary_model.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(boundary_model.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_scalar, test_data.scalar, rtol=1e-5)
            end

            @testset verbose=true "Vector Custom Quantity" begin
                trixi2vtk(boundary_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_boundary_vector",
                          output_directory=tmp_dir, iter=1, vector=vector_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_vector_1.vtu"))

                @test isapprox(boundary_system.coordinates, test_data.coordinates,
                               rtol=1e-5)
                # The velocity is always zero for `WallBoundarySystem`
                @test isapprox(zeros(size(test_data.velocity)), test_data.velocity,
                               rtol=1e-5)
                @test isapprox(boundary_model.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(boundary_model.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(fill(expected_scalar, nparticles(boundary_system)),
                               test_data.vector, rtol=1e-5)
            end
        end
    end
end
