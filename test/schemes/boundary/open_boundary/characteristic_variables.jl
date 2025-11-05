@testset verbose=true "Characteristic Variables" begin
    particle_spacing = 0.1

    # Number of boundary particles in the influence of fluid particles
    influenced_particles = [20, 50, 26]

    open_boundary_layers = 8
    sound_speed = 20.0
    density = 1000.0
    pressure = 5.0

    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    smoothing_length = 1.2particle_spacing

    # Prescribed quantities
    reference_velocity = (pos, t) -> SVector(t, 0.0)
    reference_pressure = (pos, t) -> 50_000.0 * t
    # Add small offset to avoid "ArgumentError: density must be positive and larger than `eps()`"
    reference_density = (pos, t) -> 1000.0 * (t + sqrt(eps()))

    # Face vertices of open boundary
    face_vertices_1 = [[0.0, 0.0], [0.5, -0.5], [1.0, 0.5]]
    face_vertices_2 = [[0.0, 1.0], [0.2, 2.0], [2.3, 0.5]]

    @testset "Points $i" for i in eachindex(face_vertices_1)
        n_influenced = influenced_particles[i]

        face_vertices = [face_vertices_1[i], face_vertices_2[i]]

        face_size = face_vertices[2] - face_vertices[1]
        flow_directions = [
            normalize([-face_size[2], face_size[1]]),
            -normalize([-face_size[2], face_size[1]])
        ]

        @testset "Flow Direction $j" for j in eachindex(flow_directions)
            flow_direction = flow_directions[j]
            inflow = BoundaryZone(; boundary_face=face_vertices, particle_spacing, density,
                                  face_normal=flow_direction, open_boundary_layers,
                                  boundary_type=InFlow(), reference_velocity,
                                  reference_pressure, reference_density)
            outflow = BoundaryZone(; boundary_face=face_vertices, particle_spacing, density,
                                   face_normal=(-flow_direction), open_boundary_layers,
                                   boundary_type=OutFlow(), reference_velocity,
                                   reference_pressure, reference_density)

            boundary_zones = [
                inflow,
                outflow
            ]

            @testset "`$(TrixiParticles.boundary_type_name(boundary_zone))`" for boundary_zone in
                                                                                 boundary_zones

                sign_ = (TrixiParticles.boundary_type_name(boundary_zone) == "inflow") ?
                        1 : -1
                fluid = extrude_geometry(face_vertices; particle_spacing, n_extrude=4,
                                         density, pressure,
                                         direction=(sign_ * flow_direction))

                fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel,
                                                           buffer_size=0,
                                                           density_calculator=ContinuityDensity(),
                                                           smoothing_length, sound_speed)

                boundary_system = OpenBoundarySystem(boundary_zone;
                                                     fluid_system, buffer_size=0,
                                                     boundary_model=BoundaryModelCharacteristicsLastiwka())

                semi = Semidiscretization(fluid_system, boundary_system)

                ode = semidiscretize(semi, (0.0, 5.0))

                v0_ode, u0_ode = ode.u0.x
                v = TrixiParticles.wrap_v(v0_ode, boundary_system, semi)
                u = TrixiParticles.wrap_u(u0_ode, boundary_system, semi)

                # ==== Characteristic Variables
                # `J1 = -sound_speed^2 * (rho - rho_ref) + (p - p_ref)`
                # `J2 = rho * sound_speed * (v - v_ref) + (p - p_ref)`
                # `J3 = - rho * sound_speed * (v - v_ref) + (p - p_ref)`
                function J1(t)
                    return -sound_speed^2 * (density - reference_density(0, t)) +
                           (pressure - reference_pressure(0, t))
                end
                function J2(t)
                    return density * sound_speed *
                           dot(-reference_velocity(0, t), flow_direction) +
                           (pressure - reference_pressure(0, t))
                end
                function J3(t)
                    return -density * sound_speed *
                           dot(-reference_velocity(0, t), flow_direction) +
                           (pressure - reference_pressure(0, t))
                end

                # First evaluation.
                # Particles not influenced by the fluid have zero values.
                t1 = 2.0
                TrixiParticles.evaluate_characteristics!(boundary_system,
                                                         v, u, v0_ode, u0_ode, semi, t1)
                evaluated_vars1 = boundary_system.cache.characteristics

                if TrixiParticles.boundary_type_name(boundary_zone) == "inflow"
                    @test all(isapprox.(evaluated_vars1[1, :], 0.0))
                    @test all(isapprox.(evaluated_vars1[2, :], 0.0))
                    @test all(isapprox.(evaluated_vars1[3, 1:n_influenced], J3(t1)))
                    @test all(isapprox.(evaluated_vars1[3, (n_influenced + 1):end], 0.0))

                elseif TrixiParticles.boundary_type_name(boundary_zone) == "outflow"
                    @test all(isapprox.(evaluated_vars1[1, 1:n_influenced], J1(t1)))
                    @test all(isapprox.(evaluated_vars1[2, 1:n_influenced], J2(t1)))
                    @test all(isapprox.(evaluated_vars1[1:2, (n_influenced + 1):end], 0.0))
                    @test all(isapprox.(evaluated_vars1[3, :], 0.0))
                end

                # Second evaluation.
                # Particles not influenced by the fluid have previous values.
                t2 = 3.0
                TrixiParticles.evaluate_characteristics!(boundary_system,
                                                         v, u, v0_ode, u0_ode, semi, t2)
                evaluated_vars2 = boundary_system.cache.characteristics

                if TrixiParticles.boundary_type_name(boundary_zone) == "inflow"
                    @test all(isapprox.(evaluated_vars2[1, :], 0.0))
                    @test all(isapprox.(evaluated_vars2[2, :], 0.0))
                    @test all(isapprox.(evaluated_vars2[3, 1:n_influenced], J3(t2)))
                    @test all(isapprox.(evaluated_vars2[3, (n_influenced + 1):end], J3(t1)))

                elseif TrixiParticles.boundary_type_name(boundary_zone) == "outflow"
                    @test all(isapprox.(evaluated_vars2[1, 1:n_influenced], J1(t2)))
                    @test all(isapprox.(evaluated_vars2[2, 1:n_influenced], J2(t2)))
                    @test all(isapprox.(evaluated_vars2[1, (n_influenced + 1):end], J1(t1)))
                    @test all(isapprox.(evaluated_vars2[2, (n_influenced + 1):end], J2(t1)))
                    @test all(isapprox.(evaluated_vars2[3, :], 0.0))
                end
            end
        end
    end

    @testset verbose=true "integrated variables $(n_dims)D" for n_dims in (2, 3)
        particle_spacing = 0.1
        initial_condition = rectangular_patch(particle_spacing, ntuple(_ -> 2, n_dims))

        boundary_face = n_dims == 2 ? ([0.0, 0.0], [0.0, 1.0]) :
                        ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0])
        face_normal = n_dims == 2 ? [1.0, 0.0] : [1.0, 0.0, 0.0]
        inflow = BoundaryZone(; boundary_face, boundary_type=InFlow(), face_normal,
                              open_boundary_layers=10, density=1.0, particle_spacing)

        system_wcsph = WeaklyCompressibleSPHSystem(initial_condition, ContinuityDensity(),
                                                   nothing,
                                                   SchoenbergCubicSplineKernel{n_dims}(), 1)

        open_boundary_wcsph = OpenBoundarySystem(inflow; fluid_system=system_wcsph,
                                                 buffer_size=0,
                                                 boundary_model=BoundaryModelCharacteristicsLastiwka())

        @test TrixiParticles.v_nvariables(open_boundary_wcsph) == n_dims

        system_edac = EntropicallyDampedSPHSystem(initial_condition,
                                                  SchoenbergCubicSplineKernel{n_dims}(),
                                                  1.0, 1.0)

        open_boundary_edac = OpenBoundarySystem(inflow; fluid_system=system_edac,
                                                buffer_size=0,
                                                boundary_model=BoundaryModelCharacteristicsLastiwka())

        @test TrixiParticles.v_nvariables(open_boundary_edac) == n_dims
    end
end
