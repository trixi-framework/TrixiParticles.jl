include("../test_util.jl")

# Dummy particle system (not used in the correction computation)
const dummy_particle_system = nothing

@testset "AkinciFreeSurfaceCorrection Numerical Tests" begin
    # Set up the rest density and correction object.
    rho0 = 1000.0
    correction = AkinciFreeSurfaceCorrection(rho0)

    # Test 1: Interior particle where mean density equals rest density.
    # Expected correction factor k = rho0 / rho_mean = 1.
    rho_mean_interior = 1000.0
    viscosity_corr, pressure_corr, surface_tension_corr =
        TrixiParticles.free_surface_correction(correction, dummy_particle_system, rho_mean_interior)
    @test isapprox(viscosity_corr, 1.0; atol=1e-8)
    @test pressure_corr == 1
    @test isapprox(surface_tension_corr, 1.0; atol=1e-8)

    # Test 2: Free surface particle where mean density is lower than the rest density.
    # Expected correction factor k = rho0 / rho_mean > 1.
    rho_mean_free_surface = 800.0
    expected_factor = rho0 / rho_mean_free_surface  # 1000/800 = 1.25
    viscosity_corr, pressure_corr, surface_tension_corr =
    TrixiParticles.free_surface_correction(correction, dummy_particle_system, rho_mean_free_surface)
    @test isapprox(viscosity_corr, expected_factor; atol=1e-8)
    @test pressure_corr == 1
    @test isapprox(surface_tension_corr, expected_factor; atol=1e-8)

    # Test 3: Fallback behavior.
    # For any correction that is not an AkinciFreeSurfaceCorrection the function returns (1, 1, 1).
    non_akinci_correction = "dummy"  # any type that is not AkinciFreeSurfaceCorrection
    viscosity_corr, pressure_corr, surface_tension_corr =
    TrixiParticles.free_surface_correction(non_akinci_correction, dummy_particle_system, rho_mean_free_surface)
    @test viscosity_corr == 1
    @test pressure_corr == 1
    @test surface_tension_corr == 1

    # Test 4: Check that passing a zero mean density throws a DivideError.
    @test_throws DivideError TrixiParticles.free_surface_correction(correction, dummy_particle_system, 0.0)
end
