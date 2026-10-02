@testset verbose=true "Shared near-zero interaction tolerance" begin
    for T in (Float32, Float64)
        # IEEE spacing at scale 1 gives these exact references. Scaling the length
        # by two multiplies the squared-coordinate tolerance's square root by two.
        unit_tolerance = T == Float64 ? ldexp(one(T), -26) : ldexp(sqrt(T(2)), -12)
        @test TrixiParticles.interaction_zero_distance(one(T)) === unit_tolerance
        @test TrixiParticles.interaction_zero_distance(T(2)) === T(2) * unit_tolerance

        # A generic system uses h, independent of the neighbor's smoothing length.
        system = (; smoothing_length=one(T))
        neighbor = (; smoothing_length=T(4))
        @test TrixiParticles.interaction_zero_distance(system, neighbor) === unit_tolerance

        # Cubic-spline WCSPH support is 2h. Its override must therefore differ from
        # the generic h policy, including for single-precision simulations.
        ic = InitialCondition(; coordinates=zeros(T, 2, 1), density=T[1000],
                              mass=T[1], particle_spacing=one(T))
        eos = StateEquationCole(; sound_speed=T(10), reference_density=T(1000), exponent=1)
        fluid = WeaklyCompressibleSPHSystem(ic;
                                            smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                            smoothing_length=one(T), state_equation=eos,
                                            density_calculator=ContinuityDensity())
        @test TrixiParticles.interaction_zero_distance(fluid, system) ===
              T(2) * unit_tolerance
        distance = T(1.5) * unit_tolerance
        @test distance > TrixiParticles.interaction_zero_distance(system, neighbor)
        @test distance < TrixiParticles.interaction_zero_distance(fluid, system)
    end
end
