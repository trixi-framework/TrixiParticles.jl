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
end
