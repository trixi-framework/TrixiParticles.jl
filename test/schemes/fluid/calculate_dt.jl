@trixi_testset "Fluid calculate_dt" begin
    using LinearAlgebra: norm

    struct TestViscosity
        nu::Float64
    end

    struct TestFluidSystem <: TrixiParticles.AbstractFluidSystem{2}
        smoothing_length::Float64
        sound_speed::Float64
        viscosity::TestViscosity
        acceleration::NTuple{2, Float64}
        surface_tension::Any
    end

    TrixiParticles.initial_smoothing_length(system::TestFluidSystem) = system.smoothing_length
    TrixiParticles.system_sound_speed(system::TestFluidSystem) = system.sound_speed
    TrixiParticles.v_nvariables(::TestFluidSystem) = 1
    TrixiParticles.n_integrated_particles(::TestFluidSystem) = 1
    TrixiParticles.current_density(v, ::TestFluidSystem) = view(v, 1, :)

    function TrixiParticles.kinematic_viscosity(::TestFluidSystem, viscosity::TestViscosity,
                                                smoothing_length, sound_speed)
        return viscosity.nu
    end

    function interface_dt(system_a, system_b, rho_a, rho_b)
        semi = (; systems=(system_a, system_b), ranges_v=(1:1, 2:2))
        return TrixiParticles.calculate_interface_dt([rho_a, rho_b], nothing, 0.2,
                                                     system_a, system_b, semi)
    end

    @testset "single-system dt" begin
        system = TestFluidSystem(0.1, 10.0, TestViscosity(0.5), (0.0, 9.81), nothing)
        cfl = 0.25
        dt = TrixiParticles.calculate_dt(nothing, nothing, cfl, system, nothing)

        h = system.smoothing_length
        nu = system.viscosity.nu
        dt_viscosity = 0.125 * h^2 / nu
        dt_acceleration = 0.25 * sqrt(h / norm(system.acceleration))
        dt_sound = cfl * h / system.sound_speed

        @test dt == minimum((dt_viscosity, dt_acceleration, dt_sound))
    end

    @testset "water example" begin
        # dt_sound = 0.2 * 0.01 / 1482 ≈ 1.35e-6 (acoustic limit dominates)
        system = TestFluidSystem(0.01, 1482.0, TestViscosity(1.0e-6), (0.0, 9.81), nothing)
        dt = TrixiParticles.calculate_dt(nothing, nothing, 0.2, system, nothing)
        @test isapprox(dt, 1.35e-6; atol=1e-9)
    end

    @testset "air example" begin
        # dt_sound = 0.2 * 0.01 / 343 ≈ 5.83e-6
        system = TestFluidSystem(0.01, 343.0, TestViscosity(1.5e-5), (0.0, 9.81), nothing)
        dt = TrixiParticles.calculate_dt(nothing, nothing, 0.2, system, nothing)
        @test isapprox(dt, 5.83e-6; atol=1e-8)
    end

    @testset "interface surface tension dt" begin
        viscosity = TestViscosity(1.0e-6)
        acceleration = (0.0, 9.81)
        no_surface_tension = TestFluidSystem(0.01, 343.0, viscosity, acceleration, nothing)
        morris = SurfaceTensionMorris(surface_tension_coefficient=0.072)
        morris_momentum = SurfaceTensionMomentumMorris(surface_tension_coefficient=0.072)
        with_morris = TestFluidSystem(0.012, 1482.0, viscosity, acceleration, morris)
        with_morris_momentum = TestFluidSystem(0.014, 1482.0, viscosity, acceleration,
                                               morris_momentum)

        # No inter-system surface tension force is evaluated unless both systems use
        # the same Morris model.
        @test interface_dt(no_surface_tension, with_morris, 1.2, 1000.0) == Inf
        @test interface_dt(with_morris, with_morris_momentum, 1000.0, 900.0) == Inf

        stronger_morris = SurfaceTensionMorris(surface_tension_coefficient=0.08)
        system_a = TestFluidSystem(0.008, 343.0, viscosity, acceleration, morris)
        system_b = TestFluidSystem(0.012, 1482.0, viscosity, acceleration,
                                   stronger_morris)
        rho_a = 1.2
        rho_b = 1000.0
        expected = sqrt((rho_a + rho_b) *
                        min(system_a.smoothing_length,
                            system_b.smoothing_length)^3 /
                        (4 * pi * stronger_morris.surface_tension_coefficient))
        @test interface_dt(system_a, system_b, rho_a, rho_b) == expected

        system_c = TestFluidSystem(0.01, 100.0, viscosity, acceleration,
                                   morris_momentum)
        system_d = TestFluidSystem(0.02, 100.0, viscosity, acceleration,
                                   SurfaceTensionMomentumMorris(surface_tension_coefficient=0.05))
        expected_momentum = sqrt((800.0 + 1000.0) * system_c.smoothing_length^3 /
                                 (4 * pi * morris_momentum.surface_tension_coefficient))
        @test interface_dt(system_c, system_d, 800.0, 1000.0) == expected_momentum

        physical_a = SurfaceTensionAkinciCohesionPhysical(;
                                                          surface_tension_coefficient=0.072,
                                                          reference_density=1000.0)
        physical_b = SurfaceTensionAkinciCohesionPhysical(;
                                                          surface_tension_coefficient=0.08,
                                                          reference_density=800.0)
        system_e = TestFluidSystem(0.008, 100.0, viscosity, acceleration, physical_a)
        system_f = TestFluidSystem(0.012, 100.0, viscosity, acceleration, physical_b)
        expected_physical = sqrt((physical_a.reference_density +
                                  physical_b.reference_density) *
                                 system_e.smoothing_length^3 /
                                 (4 * pi * physical_b.surface_tension_coefficient))
        @test interface_dt(system_e, system_f, 1.0, 1.0) == expected_physical
        @test interface_dt(system_e, with_morris, 1000.0, 1000.0) == Inf
    end
end
