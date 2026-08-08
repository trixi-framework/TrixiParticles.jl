@testset verbose=true "Surface Tension" begin
    @testset "constructors and capabilities" begin
        constructors = (CohesionForceAkinci, SurfaceTensionAkinci,
                        SurfaceTensionMorris, SurfaceTensionMomentumMorris)

        for constructor in constructors
            model = constructor(surface_tension_coefficient=0.5f0)
            @test model.surface_tension_coefficient === 0.5f0
            @test iszero(constructor(surface_tension_coefficient=0).surface_tension_coefficient)

            for coefficient in (-1.0, NaN, Inf, -Inf, 1.0im, "invalid")
                @test_throws ArgumentError constructor(surface_tension_coefficient=coefficient)
            end
        end

        normalized_akinci = SurfaceTensionAkinci(surface_tension_coefficient=0.5f0,
                                                 reference_smoothing_length=0.25f0)
        @test normalized_akinci.reference_smoothing_length === 0.25f0
        @test isnothing(SurfaceTensionAkinci().reference_smoothing_length)
        system_data = Dict{String, Any}()
        TrixiParticles.add_system_data!(system_data, normalized_akinci)
        @test system_data["surface_tension"]["model"] == "SurfaceTensionAkinci"
        @test system_data["surface_tension"]["surface_tension_coefficient"] === 0.5f0
        @test system_data["surface_tension"]["reference_smoothing_length"] === 0.25f0
        for reference_smoothing_length in (0.0, -1.0, NaN, Inf, -Inf, 1.0im, "invalid")
            @test_throws ArgumentError SurfaceTensionAkinci(;
                                                            reference_smoothing_length)
        end

        @test !TrixiParticles.requires_surface_normal(nothing)
        @test !TrixiParticles.requires_surface_normal(CohesionForceAkinci())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionAkinci())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionMorris())
        @test TrixiParticles.requires_surface_normal(SurfaceTensionMomentumMorris())

        normal_method = ColorfieldSurfaceNormal(boundary_contact_threshold=1,
                                                interface_threshold=0.1f0,
                                                ideal_density_threshold=0.25)
        @test normal_method isa ColorfieldSurfaceNormal{Float64}
        @test ColorfieldSurfaceNormal(boundary_contact_threshold=0.1f0,
                                      interface_threshold=0.01f0,
                                      ideal_density_threshold=0.0f0) isa
              ColorfieldSurfaceNormal{Float32}
    end

    @testset "cohesion-only systems do not require normals" begin
        coordinates = [0.0 1.0;
                       0.0 0.0]
        initial_condition = InitialCondition(; coordinates, density=ones(2),
                                             particle_spacing=1.0)
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 1.0
        surface_tension = CohesionForceAkinci(surface_tension_coefficient=0.1)

        wcsph = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                            smoothing_length,
                                            density_calculator=SummationDensity(),
                                            state_equation=StateEquationCole(sound_speed=10.0,
                                                                             reference_density=1.0,
                                                                             exponent=1),
                                            surface_tension)
        edac = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                           smoothing_length, sound_speed=10.0,
                                           density_calculator=SummationDensity(),
                                           surface_tension)

        for system in (wcsph, edac)
            @test isnothing(system.surface_normal_method)
            @test !haskey(system.cache, :surface_normal)
            @test !haskey(system.cache, :neighbor_count)
            @test !haskey(system.cache, :reference_particle_spacing)
        end

        @test_throws ArgumentError WeaklyCompressibleSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               density_calculator=SummationDensity(),
                                                               state_equation=StateEquationCole(sound_speed=10.0,
                                                                                                reference_density=1.0,
                                                                                                exponent=1),
                                                               surface_tension=SurfaceTensionAkinci())
        @test_throws ArgumentError EntropicallyDampedSPHSystem(initial_condition;
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               sound_speed=10.0,
                                                               density_calculator=SummationDensity(),
                                                               surface_tension=SurfaceTensionAkinci())

        full_akinci = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                  smoothing_length,
                                                  density_calculator=SummationDensity(),
                                                  state_equation=StateEquationCole(sound_speed=10.0,
                                                                                   reference_density=1.0,
                                                                                   exponent=1),
                                                  surface_tension=SurfaceTensionAkinci(),
                                                  reference_particle_spacing=1.0)
        @test full_akinci.surface_normal_method isa ColorfieldSurfaceNormal
        @test haskey(full_akinci.cache, :surface_normal)
    end

    @testset "zero Morris coefficient does not restrict the time step" begin
        function calculate_initial_dt(surface_tension)
            initial_condition = InitialCondition(; coordinates=[0.0 1.0; 0.0 0.0],
                                                 density=ones(2), particle_spacing=1.0)
            reference_particle_spacing = isnothing(surface_tension) ? 0 : 1.0
            system = WeaklyCompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel=WendlandC2Kernel{2}(),
                                                 smoothing_length=1.0,
                                                 density_calculator=SummationDensity(),
                                                 state_equation=StateEquationCole(sound_speed=10.0,
                                                                                  reference_density=1.0,
                                                                                  exponent=1),
                                                 surface_tension,
                                                 reference_particle_spacing)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.1))
            v_ode, u_ode = ode.u0.x
            return TrixiParticles.calculate_dt(v_ode, u_ode, 0.25, semi.systems[1], semi)
        end

        dt_without_surface_tension = calculate_initial_dt(nothing)
        dt_with_zero_csf = calculate_initial_dt(SurfaceTensionMorris(;
                                                                     surface_tension_coefficient=0.0))
        dt_with_zero_css = calculate_initial_dt(SurfaceTensionMomentumMorris(;
                                                                             surface_tension_coefficient=0.0))

        @test dt_with_zero_csf == dt_without_surface_tension
        @test dt_with_zero_css == dt_without_surface_tension
    end

    @testset verbose=true "`cohesion_force_akinci`" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        m_b = 1.0
        pos_diff = [1.0, 1.0]

        # These values can be extracted from the graphs in the paper by Akinci et al. or by manual calculation.
        # Additional digits have been accepted from the actual calculation.
        test_distance = 0.1
        val = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance, Val(3)) *
              test_distance
        @test isapprox(val[1], 0.1443038770421044, atol=6e-15)
        @test isapprox(val[2], 0.1443038770421044, atol=6e-15)

        # Maximum repulsion force
        test_distance = 0.01
        max = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance, Val(3)) *
              test_distance
        @test isapprox(max[1], 0.15913517632298307, atol=6e-15)
        @test isapprox(max[2], 0.15913517632298307, atol=6e-15)

        # Near 0
        test_distance = 0.2725
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0004360543645195717, atol=6e-15)
        @test isapprox(zero[2], 0.0004360543645195717, atol=6e-15)

        # Maximum attraction force
        test_distance = 0.5
        maxa = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, Val(3)) *
               test_distance
        @test isapprox(maxa[1], -0.15915494309189535, atol=6e-15)
        @test isapprox(maxa[2], -0.15915494309189535, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)
    end

    @testset verbose=true "adhesion_force_akinci" begin
        surface_tension = TrixiParticles.SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        m_b = 1.0
        pos_diff = [1.0, 1.0]

        # These values can be extracted from the graphs in the paper by Akinci et al. or by manual calculation.
        # Additional digits have been accepted from the actual calculation.
        test_distance = 0.1
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        test_distance = 0.5
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        # Near 0
        test_distance = 0.51
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], -0.002619160170741761, atol=6e-15)
        @test isapprox(zero[2], -0.002619160170741761, atol=6e-15)

        # Maximum adhesion force
        test_distance = 0.75
        max = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                   pos_diff, test_distance, 1.0, Val(3)) *
              test_distance
        @test isapprox(max[1], -0.004949747468305833, atol=6e-15)
        @test isapprox(max[2], -0.004949747468305833, atol=6e-15)

        # Should be 0
        test_distance = 1.0
        zero = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, m_b,
                                                    pos_diff, test_distance, 1.0, Val(3)) *
               test_distance
        @test isapprox(zero[1], 0.0, atol=6e-15)
        @test isapprox(zero[2], 0.0, atol=6e-15)

        support_radius_f32 = 15.594092f0
        distance_f32 = prevfloat(support_radius_f32)
        near_support = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                            support_radius_f32, 1.0f0,
                                                            Float32[1, 0], distance_f32,
                                                            1.0f0, Val(3))
        @test eltype(near_support) == Float32
        @test all(isfinite, near_support)
        @test 0 < norm(near_support) < eps(Float32)
    end

    @testset "two-dimensional Akinci kernels" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.0
        cohesion_normalization = 25280 / (627 * pi)

        for distance in (0.25, 0.75)
            pos_diff = SVector(distance, 0.0)
            shape = if distance > 0.5 * support_radius
                (support_radius - distance)^3 * distance^3
            else
                2 * (support_radius - distance)^3 * distance^3 - support_radius^6 / 64
            end
            expected = -cohesion_normalization * shape * pos_diff / distance
            force = TrixiParticles.cohesion_force_akinci(surface_tension, support_radius,
                                                         1.0, pos_diff, distance, Val(2))
            @test isapprox(force, expected; rtol=5eps(), atol=5eps())
        end

        distance = 0.75
        pos_diff = SVector(distance, 0.0)
        radicand = -4 * distance^2 / support_radius + 6 * distance -
                   2 * support_radius
        expected = -(13 / 1200) * radicand^(1 / 4) * pos_diff / distance
        force = TrixiParticles.adhesion_force_akinci(surface_tension, support_radius, 1.0,
                                                     pos_diff, distance, 1.0, Val(2))
        @test isapprox(force, expected; rtol=5eps(), atol=5eps())

        surface_tension_f32 = SurfaceTensionAkinci(surface_tension_coefficient=1.0f0)
        distance_f32 = 0.75f0
        pos_diff_f32 = SVector(distance_f32, 0.0f0)
        cohesion_f32 = TrixiParticles.cohesion_force_akinci(surface_tension_f32, 1.0f0,
                                                            1.0f0, pos_diff_f32,
                                                            distance_f32, Val(2))
        adhesion_f32 = TrixiParticles.adhesion_force_akinci(surface_tension_f32, 1.0f0,
                                                            1.0f0, pos_diff_f32,
                                                            distance_f32, 1.0f0, Val(2))
        @test eltype(cohesion_f32) == Float32
        @test eltype(adhesion_f32) == Float32
        @test all(isfinite, cohesion_f32)
        @test all(isfinite, adhesion_f32)

        for dimensions in (1, 4)
            @test_throws ArgumentError TrixiParticles.create_cache_surface_tension(surface_tension,
                                                                                   Float64,
                                                                                   dimensions,
                                                                                   1)
        end
    end

    @testset "Akinci kernel resolution scaling" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.8)
        adhesion_coefficient = 0.6

        function forces(scale, dimensions::Val{NDIMS}) where {NDIMS}
            support_radius = scale
            distance = 0.75 * support_radius
            pos_diff = SVector{NDIMS}(ntuple(i -> i == 1 ? distance : zero(distance),
                                             NDIMS))
            mass = scale^NDIMS
            cohesion = TrixiParticles.cohesion_force_akinci(surface_tension,
                                                            support_radius, mass,
                                                            pos_diff, distance, dimensions)
            adhesion = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                            support_radius, mass,
                                                            pos_diff, distance,
                                                            adhesion_coefficient,
                                                            dimensions)
            return cohesion, adhesion
        end

        for dimensions in (Val(2), Val(3))
            reference_cohesion, reference_adhesion = forces(1.0, dimensions)
            for scale in (0.25, 0.5, 2.0, 4.0)
                cohesion, adhesion = forces(scale, dimensions)
                @test isapprox(cohesion, reference_cohesion; rtol=5eps(), atol=5eps())
                @test isapprox(adhesion, reference_adhesion; rtol=5eps(), atol=5eps())
            end
        end
    end

    @testset "volume-normalized Akinci normal force" begin
        reference_density = 2.0
        particle_spacing = 0.25
        smoothing_length = 0.5
        coordinates = [0.0 0.375; 0.0 0.0]
        density = [reference_density, 2reference_density]
        mass = density * particle_spacing^2
        initial_condition = InitialCondition(; coordinates, velocity=zeros(2, 2), mass,
                                             density,
                                             particle_spacing)
        smoothing_kernel = WendlandC2Kernel{2}()
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density,
                                           exponent=1)

        function pair_acceleration(surface_tension, particle, neighbor, pos_diff)
            system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                 smoothing_length,
                                                 density_calculator=SummationDensity(),
                                                 state_equation, surface_tension,
                                                 reference_particle_spacing=particle_spacing)
            system.cache.surface_normal[:, 1] .= (1.0, 0.0)
            system.cache.surface_normal[:, 2] .= (0.0, 0.0)
            distance = norm(pos_diff)
            acceleration = Ref(zero(pos_diff))
            TrixiParticles.surface_tension_force!(acceleration, surface_tension,
                                                  surface_tension, system, system,
                                                  particle, neighbor, pos_diff, distance,
                                                  density[particle], density[neighbor],
                                                  zero(pos_diff), 1)
            return acceleration[], system
        end

        coefficient = 0.8
        legacy = SurfaceTensionAkinci(surface_tension_coefficient=coefficient)
        normalized = SurfaceTensionAkinci(surface_tension_coefficient=coefficient,
                                          reference_smoothing_length=1.0)
        pos_diff = SVector(-0.375, 0.0)
        legacy_acceleration, legacy_system = pair_acceleration(legacy, 1, 2, pos_diff)
        normalized_acceleration,
        normalized_system = pair_acceleration(normalized, 1, 2,
                                              pos_diff)
        support_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
        cohesion = TrixiParticles.cohesion_force_akinci(legacy, support_radius, mass[2],
                                                        pos_diff, norm(pos_diff), Val(2))
        legacy_normal_acceleration = legacy_acceleration - cohesion
        normalized_normal_acceleration = normalized_acceleration - cohesion
        pair_density = (density[1] + density[2]) / 2
        volume_factor = mass[2] / (pair_density * smoothing_length^2)
        expected_ratio = volume_factor * normalized.reference_smoothing_length /
                         smoothing_length

        @test legacy_normal_acceleration ≈ SVector(-coefficient * smoothing_length, 0.0)
        @test normalized_normal_acceleration ≈
              expected_ratio * legacy_normal_acceleration
        reverse_acceleration, _ = pair_acceleration(normalized, 2, 1, -pos_diff)
        @test mass[1] * normalized_acceleration ≈ -mass[2] * reverse_acceleration
        @test legacy_system.surface_tension === legacy
        @test normalized_system.surface_tension === normalized
    end

    @testset "Akinci kernel integral matching" begin
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
        support_radius = 1.3

        function pos_diff_at_radius(radius, ::Val{NDIMS}) where {NDIMS}
            return SVector{NDIMS}(ntuple(i -> i == 1 ? radius : zero(radius), NDIMS))
        end

        function integrate_cohesion(dimensions::Val{NDIMS}) where {NDIMS}
            radial_integral,
            _ = quadgk(0.0, support_radius / 2, support_radius;
                       rtol=1e-13) do radius
                pos_diff = pos_diff_at_radius(radius, dimensions)
                force = TrixiParticles.cohesion_force_akinci(surface_tension,
                                                             support_radius, 1.0,
                                                             pos_diff, radius, dimensions)
                return radius^(NDIMS - 1) * -force[1]
            end
            surface_measure = NDIMS == 2 ? 2pi : 4pi
            return surface_measure * radial_integral
        end

        function integrate_adhesion(dimensions::Val{NDIMS}) where {NDIMS}
            radial_integral,
            _ = quadgk(support_radius / 2, support_radius;
                       rtol=1e-13) do radius
                pos_diff = pos_diff_at_radius(radius, dimensions)
                force = TrixiParticles.adhesion_force_akinci(surface_tension,
                                                             support_radius, 1.0,
                                                             pos_diff, radius, 1.0,
                                                             dimensions)
                return radius^(NDIMS - 1) * -force[1]
            end
            surface_measure = NDIMS == 2 ? 2pi : 4pi
            return surface_measure * radial_integral
        end

        cohesion_2d = integrate_cohesion(Val(2))
        cohesion_3d = integrate_cohesion(Val(3))
        @test isapprox(cohesion_2d, 79 / 336; rtol=1e-12)
        @test isapprox(cohesion_3d, 79 / 336; rtol=1e-12)
        @test isapprox(integrate_adhesion(Val(2)), integrate_adhesion(Val(3));
                       rtol=1e-12)
    end

    @testset "Akinci free-surface correction" begin
        correction = AkinciFreeSurfaceCorrection(1000.0)
        @test TrixiParticles.free_surface_correction(correction, nothing, 1000.0,
                                                     1000.0) == (1.0, 1, 1.0)
        expected = 1000.0 / ((500.0 + 1000.0) / 2)
        viscosity, pressure,
        surface_tension = TrixiParticles.free_surface_correction(correction, nothing,
                                                                 500.0, 1000.0)
        @test viscosity == expected
        @test pressure == 1
        @test surface_tension == expected
        @test TrixiParticles.free_surface_correction(correction, nothing, 1000.0,
                                                     500.0) == (expected, 1, expected)
    end

    @testset "Akinci ContinuityDensity reconstruction" begin
        particle_spacing = 1.0
        rho0 = 1000.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=rho0,
                                           exponent=1)
        correction = AkinciFreeSurfaceCorrection(rho0)
        surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.2)
        fluid = RectangularShape(particle_spacing, (7, 7), (0.0, 0.0); density=rho0)

        function correction_density_values(density_calculator)
            system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                 smoothing_length=particle_spacing,
                                                 density_calculator, state_equation,
                                                 correction, surface_tension,
                                                 reference_particle_spacing=particle_spacing)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            density = GC.@preserve v_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                collect(TrixiParticles.current_density(v, system))
            end
            correction_density = [TrixiParticles.correction_density(correction, system,
                                                                    particle,
                                                                    density[particle])
                                  for particle in TrixiParticles.eachparticle(system)]
            return system, density, correction_density
        end

        continuity_system, continuity_density,
        continuity_correction_density = correction_density_values(ContinuityDensity())
        summation_system, summation_density,
        summation_correction_density = correction_density_values(SummationDensity())

        @test all(==(rho0), continuity_density)
        @test isapprox(continuity_system.cache.kernel_summation_density,
                       summation_density; rtol=2eps())
        @test isapprox(continuity_correction_density,
                       summation_correction_density; rtol=2eps())
        @test isapprox(continuity_system.cache.surface_normal,
                       summation_system.cache.surface_normal; rtol=2eps())

        function edac_correction_density_values(density_calculator)
            system = EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                                 smoothing_length=particle_spacing,
                                                 sound_speed=10.0, density_calculator,
                                                 correction, surface_tension,
                                                 reference_particle_spacing=particle_spacing)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            density = GC.@preserve v_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                collect(TrixiParticles.current_density(v, system))
            end
            correction_density = [TrixiParticles.correction_density(correction, system,
                                                                    particle,
                                                                    density[particle])
                                  for particle in TrixiParticles.eachparticle(system)]
            return system, density, correction_density
        end

        edac_continuity_system, edac_continuity_density,
        edac_continuity_correction_density = edac_correction_density_values(ContinuityDensity())
        edac_summation_system, edac_summation_density,
        edac_summation_correction_density = edac_correction_density_values(SummationDensity())

        @test all(==(rho0), edac_continuity_density)
        @test isapprox(edac_continuity_system.cache.kernel_summation_density,
                       edac_summation_density; rtol=2eps())
        @test isapprox(edac_continuity_correction_density,
                       edac_summation_correction_density; rtol=2eps())
        @test isapprox(edac_continuity_system.cache.surface_normal,
                       edac_summation_system.cache.surface_normal; rtol=2eps())

        coordinates = fluid.coordinates
        particle_at(position) = findfirst(particle -> coordinates[:, particle] == position,
                                          axes(coordinates, 2))
        center = particle_at([3.5, 3.5])
        face = particle_at([3.5, 0.5])
        corner = particle_at([0.5, 0.5])
        k = rho0 ./ continuity_correction_density

        @test isapprox(k[center], 1; atol=0.002)
        @test k[face] > 1.15
        @test k[corner] > k[face]

        # Dummy boundary masses complete the kernel sum at a wall, so wall particles are
        # not mistaken for a free surface by the reconstructed density.
        tank = RectangularTank(particle_spacing, (7.0, 5.0), (7.0, 8.0), rho0;
                               n_layers=2, faces=(false, false, true, false))
        wall_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                  smoothing_length=particle_spacing,
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation, correction)
        boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                     tank.boundary.mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, particle_spacing;
                                                     state_equation, correction)
        boundary_system = WallBoundarySystem(tank.boundary, boundary_model)
        wall_semi = Semidiscretization(wall_system, boundary_system)
        wall_ode = semidiscretize(wall_semi, (0.0, 0.01))
        TrixiParticles.update_systems_and_nhs(wall_ode.u0.x..., wall_semi, 0.0)

        wall_coordinates = tank.fluid.coordinates
        wall_particle_at(position) = findfirst(particle -> wall_coordinates[:, particle] ==
                                                           position,
                                               axes(wall_coordinates, 2))
        bottom = wall_particle_at([3.5, 0.5])
        interior = wall_particle_at([3.5, 2.5])
        top = wall_particle_at([3.5, 4.5])
        reconstructed_density = wall_system.cache.kernel_summation_density
        wall_k = rho0 ./ reconstructed_density

        @test isapprox(wall_k[bottom], wall_k[interior]; rtol=2eps())
        @test isapprox(wall_k[interior], 1; atol=0.002)
        @test wall_k[top] > 1.15
    end

    @testset "Akinci correction force assembly" begin
        rho0 = 1000.0
        particle_spacing = 0.5
        coordinates = [0.0 0.75; 0.0 0.0]
        initial_condition = InitialCondition(; coordinates, velocity=zeros(2, 2),
                                             mass=fill(rho0 * particle_spacing^2, 2),
                                             density=fill(rho0, 2), particle_spacing)
        smoothing_kernel = WendlandC2Kernel{2}()
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=rho0,
                                           exponent=1)
        surface_tension = CohesionForceAkinci(surface_tension_coefficient=0.2)

        function initial_acceleration(correction)
            system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                 smoothing_length=0.5,
                                                 density_calculator=ContinuityDensity(),
                                                 state_equation, surface_tension,
                                                 correction,
                                                 reference_particle_spacing=particle_spacing)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
            dv = GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                dv_inner = zeros(eltype(v), size(v))
                TrixiParticles.interact!(dv_inner, v, u, v, u, system, system, semi)
                dv_inner
            end
            return system, dv[1:2, :]
        end

        corrected_system,
        corrected_acceleration = initial_acceleration(AkinciFreeSurfaceCorrection(rho0))
        _, uncorrected_acceleration = initial_acceleration(nothing)
        correction_factor = rho0 / corrected_system.cache.kernel_summation_density[1]

        @test correction_factor > 1
        @test maximum(abs, uncorrected_acceleration) > 0
        @test isapprox(corrected_acceleration,
                       correction_factor * uncorrected_acceleration; rtol=2eps())
    end

    @testset "EDAC Akinci correction force assembly" begin
        rho0 = 1000.0
        particle_spacing = 0.5
        coordinates = [0.0 0.75; 0.0 0.0]
        initial_condition = InitialCondition(; coordinates, velocity=zeros(2, 2),
                                             mass=fill(rho0 * particle_spacing^2, 2),
                                             density=fill(rho0, 2), particle_spacing)
        smoothing_kernel = WendlandC2Kernel{2}()
        surface_tension = CohesionForceAkinci(surface_tension_coefficient=0.2)

        function initial_acceleration(density_calculator, correction)
            system = EntropicallyDampedSPHSystem(initial_condition; smoothing_kernel,
                                                 smoothing_length=particle_spacing,
                                                 sound_speed=10.0, density_calculator,
                                                 correction, surface_tension)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

            acceleration,
            correction_density = GC.@preserve v_ode u_ode begin
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)
                dv = zeros(eltype(v), size(v))
                TrixiParticles.interact!(dv, v, u, v, u, system, system, semi)
                density = TrixiParticles.current_density(v, system, 1)
                dv[1:2, :],
                TrixiParticles.correction_density(correction, system, 1, density)
            end
            return acceleration, correction_density
        end

        for density_calculator in (ContinuityDensity(), SummationDensity())
            corrected_acceleration,
            correction_density = initial_acceleration(density_calculator,
                                                      AkinciFreeSurfaceCorrection(rho0))
            uncorrected_acceleration, _ = initial_acceleration(density_calculator, nothing)
            correction_factor = rho0 / correction_density

            @test correction_factor > 1
            @test maximum(abs, uncorrected_acceleration) > 0
            @test isapprox(corrected_acceleration,
                           correction_factor * uncorrected_acceleration; rtol=2eps())
        end
    end

    @testset "compute_stress_tensors! (MomentumMorris)" begin
        # 1. Define Minimal Initial Condition with 2 Particles in 2D
        coords = [0.0 1.0;
                  0.0 0.0]
        velocity = zeros(2, 2)
        mass = ones(2)
        density = ones(2)

        ic = InitialCondition(; coordinates=coords, velocity, mass, density,
                              particle_spacing=1.0)

        # 2. Define Density Calculator, State Equation, and Kernel
        density_calc = SummationDensity()
        eq_state = StateEquationCole(sound_speed=10.0,
                                     reference_density=1.0,
                                     exponent=1)
        kernel = WendlandC2Kernel{2}()
        smoothing_length = 0.5

        # 3. Create the WeaklyCompressibleSPHSystem with Surface Tension
        system = WeaklyCompressibleSPHSystem(ic; smoothing_kernel=kernel,
                                             smoothing_length,
                                             density_calculator=density_calc,
                                             state_equation=eq_state,
                                             surface_tension=SurfaceTensionMomentumMorris(surface_tension_coefficient=1.0),
                                             surface_normal_method=ColorfieldSurfaceNormal(interface_threshold=0.1,
                                                                                           ideal_density_threshold=0.9),
                                             reference_particle_spacing=1.0,)

        # 4. Verify Cache Contains Necessary Fields
        @test haskey(system.cache, :delta_s)
        @test haskey(system.cache, :surface_normal)
        @test haskey(system.cache, :stress_tensor)

        # 5. Manually Populate `delta_s` and `surface_normal`
        system.cache.delta_s .= [1.0, 2.0]
        system.cache.surface_normal .= hcat([1.0, 0.0], [1 / sqrt(2), 1 / sqrt(2)])
        system.cache.stress_tensor .= zeros(2, 2, 2)  # Reset to zero before computation

        # 6. Call `compute_stress_tensors!` with `SurfaceTensionMomentumMorris`
        TrixiParticles.compute_stress_tensors!(system,
                                               SurfaceTensionMomentumMorris(),
                                               nothing, nothing,  # v, u (not needed for stress computation)
                                               nothing, nothing,  # v_ode, u_ode (not needed)
                                               SerialBackend(),   # semi (only passed to `@threaded`)
                                               0.0)

        # 7. Define Reference Stress Tensors by Hand
        #
        # Reference calculations based on the formula:
        # σ_ij(a) = δs_a (δ_ij - n_i n_j) - δ_ij max(δs)
        #
        # For Particle 1:
        # δs = 1.0
        # n = (1.0, 0.0)
        # max(δs) = 2.0
        # σ_11 = 1*(1 - 1^2) - 1*2 = -2
        # σ_12 = 1*(0 - 1*0) - 0*2 = 0
        # σ_21 = 1*(0 - 1*0) - 0*2 = 0
        # σ_22 = 1*(1 - 0^2) - 1*2 = 1 - 2 = -1
        #
        # Resulting Stress Tensor for Particle 1:
        # [-2.0  0.0
        #   0.0 -1.0]
        #
        # For Particle 2:
        # δs = 2.0
        # n = (1/√2, 1/√2)
        # max(δs) = 2.0
        # σ_11 = 2*(1 - (1/√2)^2) - 1*2 = 2*(1 - 0.5) - 2 = 1 - 2 = -1
        # σ_12 = 2*(0 - (1/√2)^2) - 0*2 = 2*(0 - 0.5) = -1
        # σ_21 = 2*(0 - (1/√2)^2) - 0*2 = -1
        # σ_22 = 2*(1 - (1/√2)^2) - 1*2 = 2*(1 - 0.5) - 2 = 1 - 2 = -1
        #
        # Resulting Stress Tensor for Particle 2:
        # [-1.0 -1.0
        #  -1.0 -1.0]

        ref_particle_1 = [-2.0 0.0;
                          0.0 -1.0]
        ref_particle_2 = [-1.0 -1.0;
                          -1.0 -1.0]

        # 8. Retrieve Computed Stress Tensor
        computed = system.cache.stress_tensor

        # 9. Perform Assertions
        @test all(isfinite, computed)

        @test isapprox(computed[:, :, 1], ref_particle_1; atol=1e-14)
        @test isapprox(computed[:, :, 2], ref_particle_2; atol=1e-14)
    end
end
