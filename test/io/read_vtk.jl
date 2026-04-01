@testset verbose=true "`vtk2trixi`" begin
    mktempdir() do tmp_dir
        coordinates = fill(1.0, 2, 12)
        velocity = fill(2.0, 2, 12)

        expected_ic = InitialCondition(; coordinates=coordinates, velocity=velocity,
                                       density=1000.0, pressure=900.0, mass=50.0)

        @testset verbose=true "`InitialCondition`" begin
            @testset verbose=true "`Float64`" begin
                trixi2vtk(expected_ic; filename="tmp_initial_condition_64",
                          output_directory=tmp_dir)
                file = joinpath(tmp_dir, "tmp_initial_condition_64.vtu")
                test_ic = vtk2trixi(file)

                @test isapprox(expected_ic.coordinates, test_ic.coordinates, rtol=1e-5)
                @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
                @test isapprox(expected_ic.density, test_ic.density, rtol=1e-5)
                @test isapprox(expected_ic.pressure, test_ic.pressure, rtol=1e-5)
                @test eltype(test_ic) === Float64
                @test eltype(test_ic.coordinates) === Float64
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
                test_ic = vtk2trixi(file)

                @test isapprox(expected_ic_32.coordinates, test_ic.coordinates, rtol=1e-5)
                @test isapprox(expected_ic_32.velocity, test_ic.velocity, rtol=1e-5)
                @test isapprox(expected_ic_32.density, test_ic.density, rtol=1e-5)
                @test isapprox(expected_ic_32.pressure, test_ic.pressure, rtol=1e-5)
                @test eltype(test_ic) === Float32
                @test eltype(test_ic.coordinates) === Float32
            end

            @testset verbose=true "Custom Element Type" begin
                trixi2vtk(expected_ic; filename="tmp_initial_condition_custom",
                          output_directory=tmp_dir)
                file = joinpath(tmp_dir, "tmp_initial_condition_custom.vtu")
                test_ic = vtk2trixi(file, element_type=Float32, coordinates_eltype=Float32)

                @test isapprox(expected_ic.coordinates, test_ic.coordinates, rtol=1e-5)
                @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
                @test isapprox(expected_ic.density, test_ic.density, rtol=1e-5)
                @test isapprox(expected_ic.pressure, test_ic.pressure, rtol=1e-5)
                @test eltype(test_ic) === Float32
                @test eltype(test_ic.coordinates) === Float32
            end

            @testset verbose=true "Mixed Types" begin
                expected_ic_mixed = InitialCondition(;
                                                     coordinates=coordinates,
                                                     velocity=convert.(Float32, velocity),
                                                     density=1000.0f0, pressure=900.0f0,
                                                     mass=50.0f0, particle_spacing=0.1f0)
                trixi2vtk(expected_ic_mixed; filename="tmp_initial_condition_mixed",
                          output_directory=tmp_dir)
                file = joinpath(tmp_dir, "tmp_initial_condition_mixed.vtu")
                test_ic = vtk2trixi(file)

                @test isapprox(expected_ic.coordinates, test_ic.coordinates, rtol=1e-5)
                @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
                @test isapprox(expected_ic.density, test_ic.density, rtol=1e-5)
                @test isapprox(expected_ic.pressure, test_ic.pressure, rtol=1e-5)
                @test eltype(test_ic) === Float32
                @test eltype(test_ic.coordinates) === Float64
            end

            @testset verbose=true "Exact Field Matches" begin
                trixi2vtk(expected_ic; filename="tmp_initial_condition_exact_field_match",
                          output_directory=tmp_dir,
                          center_of_mass_velocity=fill(42.0, size(expected_ic.velocity)))
                file = joinpath(tmp_dir, "tmp_initial_condition_exact_field_match.vtu")
                test_ic = vtk2trixi(file)

                @test isapprox(expected_ic.velocity, test_ic.velocity, rtol=1e-5)
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
            dvdu_ode = nothing
            v = fill(2.0, ndims(fluid_system), nparticles(fluid_system))
            pressure = fill(3.0, nparticles(fluid_system))
            v_ode = vec([v; pressure'])
            u = fill(1.0, ndims(fluid_system), nparticles(fluid_system))
            u_ode = vec(u)
            x = (; v_ode, u_ode)
            vu_ode = (; x)

            # Write out `AbstractFluidSystem` Simulation-File
            trixi2vtk(fluid_system, dvdu_ode, vu_ode, semi, 0.0,
                      nothing; system_name="tmp_file_fluid", output_directory=tmp_dir,
                      iter=1)

            # Load `AbstractFluidSystem` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_fluid_1.vtu"))

            @test isapprox(u, test.coordinates, rtol=1e-5)
            @test isapprox(v, test.velocity, rtol=1e-5)
            @test isapprox(pressure, test.pressure, rtol=1e-5)
            @test isapprox(fluid_system.cache.density, test.density, rtol=1e-5)
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
            dvdu_ode = nothing
            v_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            u_ode = zeros(ndims(boundary_system) * nparticles(boundary_system))
            x = (; v_ode, u_ode)
            vu_ode = (; x)

            # Write out `WallBoundarySystem` Simulation-File
            trixi2vtk(boundary_system, dvdu_ode, vu_ode, semi, 0.0,
                      nothing; system_name="tmp_file_boundary", output_directory=tmp_dir,
                      iter=1)

            # Load `WallBoundarySystem` Simulation-File
            test = vtk2trixi(joinpath(tmp_dir, "tmp_file_boundary_1.vtu"))

            @test isapprox(boundary_system.coordinates, test.coordinates, rtol=1e-5)
            # The velocity is always zero for `WallBoundarySystem`
            @test isapprox(zeros(size(test.velocity)), test.velocity, rtol=1e-5)
            @test isapprox(boundary_model.pressure, test.pressure, rtol=1e-5)
            @test isapprox(boundary_model.cache.density, test.density, rtol=1e-5)
        end

        @testset verbose=true "`RigidBodySystem`" begin
            coordinates_rigid = [-1.0 1.0
                                 0.0 0.0]
            velocity_rigid = [0.0 0.0
                              -1.0 1.0]
            rigid_ic = InitialCondition(; coordinates=coordinates_rigid,
                                        velocity=velocity_rigid,
                                        density=[1000.0, 1000.0],
                                        mass=[1.0, 1.0])

            rigid_system = RigidBodySystem(rigid_ic; acceleration=(0.0, 0.0))
            semi = Semidiscretization(rigid_system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            v = TrixiParticles.wrap_v(v_ode, rigid_system, semi)
            u = TrixiParticles.wrap_u(u_ode, rigid_system, semi)
            TrixiParticles.update_final!(rigid_system, v, u, v_ode, u_ode, semi, 0.0)

            vu_ode = (; x=(v_ode, u_ode))
            trixi2vtk(rigid_system, nothing, vu_ode, semi, 0.0,
                      nothing; system_name="tmp_file_rigid", output_directory=tmp_dir,
                      iter=1)

            vtk_file = TrixiParticles.ReadVTK.VTKFile(joinpath(tmp_dir,
                                                               "tmp_file_rigid_1.vtu"))
            point_data = TrixiParticles.ReadVTK.get_point_data(vtk_file)
            field_data = TrixiParticles.ReadVTK.get_field_data(vtk_file)

            @test Array(TrixiParticles.ReadVTK.get_data(point_data["relative_coordinates"])) ==
                  rigid_system.relative_coordinates
            @test vec(Array(TrixiParticles.ReadVTK.get_data(field_data["center_of_mass"]))) ==
                  collect(rigid_system.center_of_mass[])
            @test vec(Array(TrixiParticles.ReadVTK.get_data(field_data["center_of_mass_velocity"]))) ==
                  collect(rigid_system.center_of_mass_velocity[])
            @test vec(Array(TrixiParticles.ReadVTK.get_data(field_data["resultant_force"]))) ==
                  collect(rigid_system.resultant_force[])
            @test only(Array(TrixiParticles.ReadVTK.get_data(field_data["angular_velocity"]))) ==
                  rigid_system.angular_velocity[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(field_data["resultant_torque"]))) ==
                  rigid_system.resultant_torque[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(field_data["angular_acceleration_force"]))) ==
                  rigid_system.angular_acceleration_force[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(field_data["gyroscopic_acceleration"]))) ==
                  rigid_system.gyroscopic_acceleration[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(field_data["contact_count"]))) ==
                  rigid_system.cache.contact_count[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(field_data["max_contact_penetration"]))) ==
                  rigid_system.cache.max_contact_penetration[]

            rigid_coordinates_1 = reshape([0.0, 0.0], 2, 1)
            rigid_velocity_1 = reshape([0.0, 0.0], 2, 1)
            rigid_mass_1 = [2.0]
            rigid_density_1 = [1000.0]
            rigid_ic_1 = InitialCondition(; coordinates=rigid_coordinates_1,
                                          velocity=rigid_velocity_1,
                                          mass=rigid_mass_1,
                                          density=rigid_density_1,
                                          particle_spacing=0.1)

            rigid_coordinates_2 = reshape([0.08, 0.0], 2, 1)
            rigid_velocity_2 = reshape([-1.5, 0.0], 2, 1)
            rigid_mass_2 = [3.0]
            rigid_density_2 = [1000.0]
            rigid_ic_2 = InitialCondition(; coordinates=rigid_coordinates_2,
                                          velocity=rigid_velocity_2,
                                          mass=rigid_mass_2,
                                          density=rigid_density_2,
                                          particle_spacing=0.1)

            contact_model_1 = RigidContactModel(; normal_stiffness=20.0,
                                                normal_damping=4.0,
                                                contact_distance=0.1)
            contact_model_2 = RigidContactModel(; normal_stiffness=30.0,
                                                normal_damping=8.0,
                                                contact_distance=0.12)

            rigid_system_1 = RigidBodySystem(rigid_ic_1;
                                             acceleration=(0.0, 0.0),
                                             contact_model=contact_model_1)
            rigid_system_2 = RigidBodySystem(rigid_ic_2;
                                             acceleration=(0.0, 0.0),
                                             contact_model=contact_model_2)
            semi_contact = Semidiscretization(rigid_system_1, rigid_system_2)
            ode_contact = semidiscretize(semi_contact, (0.0, 0.01))
            v_ode_contact, u_ode_contact = ode_contact.u0.x
            dv_ode_contact = zero(v_ode_contact)
            du_ode_contact = zero(u_ode_contact)

            v_rigid_1 = TrixiParticles.wrap_v(v_ode_contact, rigid_system_1, semi_contact)
            u_rigid_1 = TrixiParticles.wrap_u(u_ode_contact, rigid_system_1, semi_contact)
            v_rigid_2 = TrixiParticles.wrap_v(v_ode_contact, rigid_system_2, semi_contact)
            u_rigid_2 = TrixiParticles.wrap_u(u_ode_contact, rigid_system_2, semi_contact)
            TrixiParticles.update_final!(rigid_system_1, v_rigid_1, u_rigid_1,
                                         v_ode_contact, u_ode_contact, semi_contact, 0.0)
            TrixiParticles.update_final!(rigid_system_2, v_rigid_2, u_rigid_2,
                                         v_ode_contact, u_ode_contact, semi_contact, 0.0)

            TrixiParticles.interact!(dv_ode_contact, v_ode_contact, u_ode_contact,
                                     rigid_system_1, rigid_system_2, semi_contact)
            TrixiParticles.interact!(dv_ode_contact, v_ode_contact, u_ode_contact,
                                     rigid_system_2, rigid_system_1, semi_contact)

            dv_rigid_1 = TrixiParticles.wrap_v(dv_ode_contact, rigid_system_1, semi_contact)
            dv_rigid_2 = TrixiParticles.wrap_v(dv_ode_contact, rigid_system_2, semi_contact)
            TrixiParticles.finalize_interaction!(rigid_system_1, dv_rigid_1, v_rigid_1,
                                                 u_rigid_1, dv_ode_contact, v_ode_contact,
                                                 u_ode_contact, semi_contact)
            TrixiParticles.finalize_interaction!(rigid_system_2, dv_rigid_2, v_rigid_2,
                                                 u_rigid_2, dv_ode_contact, v_ode_contact,
                                                 u_ode_contact, semi_contact)

            @test rigid_system_1.cache.contact_count[] > 0
            @test rigid_system_1.cache.max_contact_penetration[] > 0

            dvdu_ode_contact = (; x=(dv_ode_contact, du_ode_contact))
            vu_ode_contact = (; x=(v_ode_contact, u_ode_contact))
            trixi2vtk(dvdu_ode_contact, vu_ode_contact, semi_contact, 0.0;
                      output_directory=tmp_dir, iter=1)

            contact_filename = TrixiParticles.system_names(semi_contact.systems)[1]
            vtk_contact = TrixiParticles.ReadVTK.VTKFile(joinpath(tmp_dir,
                                                                  "$(contact_filename)_1.vtu"))
            point_data_contact = TrixiParticles.ReadVTK.get_point_data(vtk_contact)

            @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["contact_count"]))) ==
                  rigid_system_1.cache.contact_count[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["contact_count"]))) > 0
            @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["max_contact_penetration"]))) ≈
                  rigid_system_1.cache.max_contact_penetration[]
            @test only(Array(TrixiParticles.ReadVTK.get_data(point_data_contact["max_contact_penetration"]))) > 0
        end
    end
end
