using Test
using TrixiParticles

@testset "Carreau-Yasuda viscosity" begin
    visc = ViscosityCarreauYasuda(
        nu0    = 3.5e-6,
        nu_inf = 1.0e-6,
        lambda = 3.313e-2,
        a      = 2.0,
        n      = 0.3,
        epsilon = 0.01,
    )

    @test TrixiParticles.carreau_yasuda_nu(visc, 0.0) ≈ visc.nu0

    @test TrixiParticles.carreau_yasuda_nu(visc, 1e6) ≈ visc.nu_inf atol=1e-8
# or, using relative tolerance:
# @test TrixiParticles.carreau_yasuda_nu(visc, 1e6) ≈ visc.nu_inf rtol=1e-3

    viscN = ViscosityCarreauYasuda(
        nu0    = 1.0e-6,
        nu_inf = 1.0e-6,
        lambda = 1.0,
        a      = 2.0,
        n      = 1.0,
        epsilon = 0.01,
    )

    @test TrixiParticles.carreau_yasuda_nu(viscN, 0.0)   ≈ 1.0e-6
    @test TrixiParticles.carreau_yasuda_nu(viscN, 10.0)  ≈ 1.0e-6
    @test TrixiParticles.carreau_yasuda_nu(viscN, 1e5)   ≈ 1.0e-6
end
