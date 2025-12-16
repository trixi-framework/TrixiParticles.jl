@testset verbose=true "`vtk2trixi`" begin
    mktempdir() do tmp_dir
        coordinates = fill(1.0, 2, 12)
        velocity = fill(2.0, 2, 12)

        expected_ic = InitialCondition(coordinates=coordinates, velocity=velocity,
                                       density=1000.0, pressure=900.0, mass=50.0)

        expected_cq_scalar = 3.0
        expected_cq_vector = fill(expected_cq_scalar,
                                  size(expected_ic.coordinates, 2))
        scalar_cq_function(system, data, t) = expected_cq_scalar
        vector_cq_function(system, data,
                           t) = fill(expected_cq_scalar, nparticles(system))

        @testset verbose=true "`InitialCondition`" begin
            @testset verbose=true "Scalar custom quantity" begin
                trixi2vtk(expected_ic; filename="tmp_initial_condition_scalar",
                          output_directory=tmp_dir,
                          cq_scalar=expected_cq_scalar)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_initial_condition_scalar.vtu");
                                      cq_scalar="cq_scalar")

                @test isapprox(expected_ic.coordinates, test_data.coordinates, rtol=1e-5)
                @test isapprox(expected_ic.velocity, test_data.velocity, rtol=1e-5)
                @test isapprox(expected_ic.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_ic.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(expected_cq_scalar, test_data.cq_scalar,
                               rtol=1e-5)
            end

            @testset verbose=true "Vector custom quantity" begin
                trixi2vtk(expected_ic; filename="tmp_initial_condition_vector",
                          output_directory=tmp_dir,
                          cq_vector=expected_cq_vector)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_initial_condition_vector.vtu");
                                      cq_vector="cq_vector")

                @test isapprox(expected_ic.coordinates, test_data.coordinates, rtol=1e-5)
                @test isapprox(expected_ic.velocity, test_data.velocity, rtol=1e-5)
                @test isapprox(expected_ic.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_ic.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(expected_cq_vector, test_data.cq_vector,
                               rtol=1e-5)
            end
        end

        @testset verbose=true "`AbstractFluidSystem`" begin
            fluid_system = EntropicallyDampedSPHSystem(expected_ic,
                                                       SchoenbergCubicSplineKernel{2}(),
                                                       1.5, 1.5)

            # Overwrite values because we skip the update step
            fluid_system.cache.density .= expected_ic.density

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

            @testset verbose=true "Scalar custom quantity" begin
                trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_fluid_scalar",
                          output_directory=tmp_dir,
                          iter=1, cq_scalar=scalar_cq_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_scalar_1.vtu");
                                      cq_scalar="cq_scalar")

                @test isapprox(u, test_data.coordinates, rtol=1e-5)
                @test isapprox(v, test_data.velocity, rtol=1e-5)
                @test isapprox(pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(fluid_system.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_cq_scalar, test_data.cq_scalar,
                               rtol=1e-5)
            end

            @testset verbose=true "Vector custom quantity" begin
                trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_fluid_vector",
                          output_directory=tmp_dir,
                          iter=1, cq_vector=vector_cq_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_vector_1.vtu");
                                      cq_vector="cq_vector")

                @test isapprox(u, test_data.coordinates, rtol=1e-5)
                @test isapprox(v, test_data.velocity, rtol=1e-5)
                @test isapprox(pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(fluid_system.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(fill(expected_cq_scalar, nparticles(fluid_system)),
                               test_data.cq_vector, rtol=1e-5)
            end
        end

        @testset verbose=true "`WallBoundarySystem`" begin
            boundary_model = BoundaryModelDummyParticles(expected_ic.density,
                                                         expected_ic.mass,
                                                         SummationDensity(),
                                                         SchoenbergCubicSplineKernel{2}(),
                                                         1.5)

            # Overwrite values because we skip the update step
            boundary_model.pressure .= expected_ic.pressure
            boundary_model.cache.density .= expected_ic.density

            boundary_system = WallBoundarySystem(expected_ic, boundary_model)
            semi = Semidiscretization(boundary_system)

            # Create dummy ODE solutions
            v_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            u_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            dv_ode = zero(v_ode)
            du_ode = zero(u_ode)
            x = (; v_ode, u_ode)
            vu_ode = (; x)
            dvdu_ode = (; x=(; v_ode=dv_ode, u_ode=du_ode))

            @testset verbose=true "Scalar custom quantity" begin
                trixi2vtk(boundary_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_boundary_scalar",
                          output_directory=tmp_dir,
                          iter=1, cq_scalar=scalar_cq_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_scalar_1.vtu");
                                      cq_scalar="cq_scalar")

                @test isapprox(boundary_system.coordinates, test_data.coordinates,
                               rtol=1e-5)
                # The velocity is always zero for `WallBoundarySystem`
                @test isapprox(zeros(size(test_data.velocity)), test_data.velocity,
                               rtol=1e-5)
                @test isapprox(boundary_model.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(boundary_model.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(expected_cq_scalar, test_data.cq_scalar,
                               rtol=1e-5)
            end

            @testset verbose=true "Vector custom quantity" begin
                trixi2vtk(boundary_system, dvdu_ode, vu_ode, semi, 0.0,
                          nothing; system_name="tmp_file_boundary_vector",
                          output_directory=tmp_dir,
                          iter=1, cq_vector=vector_cq_function)

                # Load file containing test data
                test_data = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_vector_1.vtu");
                                      cq_vector="cq_vector")

                @test isapprox(boundary_system.coordinates, test_data.coordinates,
                               rtol=1e-5)
                # The velocity is always zero for `WallBoundarySystem`
                @test isapprox(zeros(size(test_data.velocity)), test_data.velocity,
                               rtol=1e-5)
                @test isapprox(boundary_model.pressure, test_data.pressure, rtol=1e-5)
                @test isapprox(boundary_model.cache.density, test_data.density, rtol=1e-5)
                @test isapprox(fill(expected_cq_scalar, nparticles(boundary_system)),
                               test_data.cq_vector, rtol=1e-5)
            end
        end
    end
end
