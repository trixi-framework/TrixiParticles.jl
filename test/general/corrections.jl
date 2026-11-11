@trixi_testset "Correction Consistency" begin
    include("corrections/common.jl")
    include("corrections/lifecycle.jl")
    include("corrections/shepard.jl")
    include("corrections/kernel.jl")
    include("corrections/gradient.jl")
    include("corrections/mixed.jl")
    include("corrections/coupling.jl")
    include("corrections/rigid_density.jl")
end

@testset verbose=true "Corrections" begin
    @testset "Gradient correction matrix of a fluid system" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=1.0)

        # Independent formulas of the cubic spline kernel for ``h = 1`` (compact support 2)
        W(r) = 10 / (7pi) * (r < 1 ? 1 - 1.5r^2 + 0.75r^3 : r < 2 ? 0.25(2 - r)^3 : 0.0)
        function grad_W(pos_diff)
            r = norm(pos_diff)
            (iszero(r) || r >= 2) && return zero(pos_diff)
            derivative = 10 / (7pi) * (r < 1 ? -3r + 2.25r^2 : -0.75(2 - r)^2)
            return derivative / r * pos_diff
        end

        # Irregular neighborhoods, so that the correction matrices are far from the identity.
        # The first and the last particle are outside of the support of each other.
        coordinates = [0.0 0.8 1.7 0.3 1.1 2.1; 0.0 0.1 0.2 0.9 1.0 1.2]
        mass = [1.0, 1.1, 0.9, 1.2, 1.0, 0.8]
        density = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1000.0]
        volumes = mass ./ density
        positions = [SVector{2}(coordinates[:, i]) for i in axes(coordinates, 2)]

        # Gradient of the kernel correction of particle `a` at displacement `pos_diff`
        # to particle `b` (`k` is the kernel correction coefficient and `gamma` the vector
        # ``\sum_b V_b \nabla W_b / k``).
        function reference(a)
            pos_diffs = [positions[a] - x for x in positions]
            kernel_values = W.(norm.(pos_diffs))
            gradients = grad_W.(pos_diffs)
            k = dot(volumes, kernel_values)
            gamma = sum(volumes .* gradients) / k
            kernel_corrected = [(g - w * gamma) / k
                                for (g, w) in zip(gradients, kernel_values)]
            return (; pos_diffs, kernel_values, gradients, k, gamma, kernel_corrected)
        end

        # ``L = (-\sum_b V_b \nabla W_b \otimes r_{ab})^{-1}``
        function inverse_moment(volumes, gradients, pos_diffs)
            return inv(-sum(v * g * transpose(d)
                            for (v, g, d) in zip(volumes, gradients, pos_diffs)))
        end

        @testset "$(nameof(typeof(correction)))" for correction in
                                                     (GradientCorrection(),
                                                      BlendedGradientCorrection(0.3),
                                                      MixedKernelGradientCorrection())
            ic = InitialCondition(; coordinates, mass, density, particle_spacing=1.0)
            system = WeaklyCompressibleSPHSystem(ic; smoothing_kernel=kernel,
                                                 smoothing_length=1.0, state_equation,
                                                 density_calculator=ContinuityDensity(),
                                                 correction)
            semi = Semidiscretization(system; parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

            for a in axes(coordinates, 2)
                (; pos_diffs, gradients, kernel_corrected) = reference(a)

                # The matrix is assembled from the gradient without gradient correction,
                # but with kernel correction for `MixedKernelGradientCorrection`.
                matrix_gradients = correction isa MixedKernelGradientCorrection ?
                                   kernel_corrected : gradients
                L = inverse_moment(volumes, matrix_gradients, pos_diffs)

                # The matrix must not be the identity matrix
                @test !isapprox(L, I(2), atol=0.1)
                @test system.cache.correction_matrix[:, :, a] ≈ L
            end

            # Corrected kernel gradient of particle 1 with respect to particle 2
            (; gradients, kernel_corrected) = reference(1)
            L = system.cache.correction_matrix[:, :, 1]
            grad_kernel = TrixiParticles.smoothing_kernel_grad(system,
                                                               positions[1] -
                                                               positions[2],
                                                               norm(positions[1] -
                                                                    positions[2]), 1)
            expected = if correction isa GradientCorrection
                L * gradients[2]
            elseif correction isa BlendedGradientCorrection
                0.7 * gradients[2] + 0.3 * L * gradients[2]
            else
                L * kernel_corrected[2]
            end
            @test grad_kernel ≈ expected
        end
    end
end
