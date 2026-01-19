@trixi_testset "Fluid calculate_dt" begin
    using LinearAlgebra: norm
    using StatsBase: harmmean

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

    function TrixiParticles.kinematic_viscosity(::TestFluidSystem, viscosity::TestViscosity,
                                                smoothing_length, sound_speed)
        return viscosity.nu
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

    @testset "interface dt" begin
        system_a = TestFluidSystem(0.1, 8.0, TestViscosity(0.4), (0.0, 9.81), nothing)
        system_b = TestFluidSystem(0.2, 12.0, TestViscosity(0.6), (0.0, 9.81), nothing)
        cfl = 0.3

        dt = TrixiParticles.calculate_interface_dt(nothing, nothing, cfl,
                                                   system_a, system_b, nothing)

        h_interface = harmmean((system_a.smoothing_length, system_b.smoothing_length))
        signal_speed = harmmean((system_a.sound_speed, system_b.sound_speed))
        dt_acoustic = cfl * h_interface / signal_speed

        nu_interface = system_a.viscosity.nu + system_b.viscosity.nu
        dt_viscosity = 0.125 * h_interface^2 / nu_interface

        @test dt == minimum((dt_acoustic, dt_viscosity))
    end

    @testset "air-water interface example" begin
        air = TestFluidSystem(0.008, 343.0, TestViscosity(1.5e-5), (0.0, 9.81), nothing)
        water = TestFluidSystem(0.012, 1482.0, TestViscosity(1.0e-6), (0.0, 9.81), nothing)
        cfl = 0.2

        # dt_acoustic = 0.2 * harmonicmean(0.008,0.012)/harmonicmean(343,1482) ≈ 3.45e-6
        # dt_viscosity = 0.125 * h^2 / (1.5e-5 + 1e-6) ≈ 5.7e-4 → acoustic limit wins
        dt = TrixiParticles.calculate_interface_dt(nothing, nothing, cfl, air, water,
                                                   nothing)
        @test isapprox(dt, 3.45e-6; atol=1e-8)
    end

    @testset "high-viscosity interface example" begin
        # Example: water (ν≈1e-6 m^2/s) interacting with bitumen/pitch (ν≈1e3 m^2/s, c≈200 m/s)
        water = TestFluidSystem(0.012, 1482.0, TestViscosity(1.0e-6), (0.0, 9.81), nothing)
        bitumen = TestFluidSystem(0.012, 200.0, TestViscosity(1.0e3), (0.0, 9.81), nothing)
        cfl = 0.2

        # dt_acoustic ≈ 0.2 * 0.012 / harmonicmean(1482,200) ≈ 6.8e-6
        # dt_viscosity ≈ 0.125 * 0.012^2 / (1e-6 + 1e3) ≈ 1.8e-8 → viscous limit dominates
        dt = TrixiParticles.calculate_interface_dt(nothing, nothing, cfl, water, bitumen,
                                                   nothing)
        @test isapprox(dt, 1.8e-8; atol=1e-10)
    end

    @testset "interface dt zero sound speed" begin
        silent = TestFluidSystem(0.1, 0.0, TestViscosity(0.4), (0.0, 9.81), nothing)
        dt = TrixiParticles.calculate_interface_dt(nothing, nothing, 0.2,
                                                   silent, silent, nothing)
        @test dt == Inf
    end
end
