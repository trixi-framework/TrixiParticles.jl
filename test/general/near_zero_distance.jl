@testset verbose=true "Uniform relative near-zero distance" begin
    for T in (Float32, Float64), h in T.((0.3, 1.0, 3.5))
        kernel = SchoenbergCubicSplineKernel{2}()
        q_tolerance = sqrt(eps(T))
        # These relative distances cover the skipped region and the newly retained
        # region between the old support-scaled WCSPH tolerance and the h-based rule.
        for factor in T.((0.0, 0.5, 1.5))
            r = factor * q_tolerance * h
            pos_diff = SVector(r, zero(T))
            grad = TrixiParticles.kernel_grad(kernel, pos_diff, r, h)
            derivative = TrixiParticles.kernel_deriv(kernel, r, h)
            if factor < 1
                @test iszero(grad)
                @test iszero(derivative)
            else
                q = r / h
                expected = T(10) / (T(7) * T(pi) * h^3) * (-T(3) * q + T(2.25) * q^2)
                @test grad[1] ≈ expected
                @test derivative ≈ expected
            end
        end
    end

    # The strict '<' rule retains the exact boundary. At h=1 this threshold is
    # representable in Float64, exposing inconsistent '<' versus '<=' guards.
    kernel = SchoenbergCubicSplineKernel{2}()
    r = ldexp(1.0, -26)
    @test !iszero(TrixiParticles.kernel_grad(kernel, SVector(r, 0.0), r, 1.0))
    @test !iszero(TrixiParticles.kernel_deriv(kernel, r, 1.0))

    @testset "Fluid-structure force balance" begin
        h = 0.3
        kernel = Poly6Kernel{2}()
        ic = InitialCondition(; coordinates=zeros(2, 1), mass=[1.0], density=1.0,
                              particle_spacing=1.0)
        state_equation = StateEquationCole(sound_speed=10.0, reference_density=1.0,
                                           exponent=1)
        fluid = WeaklyCompressibleSPHSystem(ic; smoothing_kernel=kernel, smoothing_length=h,
                                            density_calculator=SummationDensity(),
                                            state_equation)
        boundary_model = BoundaryModelDummyParticles([1.0], [1.0], SummationDensity(),
                                                     kernel, h)
        structure = RigidBodySystem(ic; boundary_model)
        fluid.pressure .= 1.0
        fluid.cache.density .= 1.0
        boundary_model.pressure .= 1.0
        v = zeros(2, 1)
        u_fluid, u_structure = ic.coordinates, copy(ic.coordinates)
        semi = DummySemidiscretization()

        # At h=0.3, factor 0.9 lies between the old and new cutoffs; 1.5 is retained.
        for factor in (0.9, 1.5)
            u_structure[1, 1] = factor * sqrt(eps()) * h
            dv_fluid = zero(v)
            structure.force_per_particle .= 0.0
            TrixiParticles.interact!(dv_fluid, v, u_fluid, v, u_structure,
                                     fluid, structure, semi)
            TrixiParticles.interact_structure_fluid!(zero(v), v, u_structure, v, u_fluid,
                                                     structure, fluid, semi)
            @test dv_fluid ≈ -structure.force_per_particle
            @test iszero(dv_fluid) == (factor < 1)
        end
    end
end
