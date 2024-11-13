@testset verbose=true "State Equations" begin
    @testset verbose=true "StateEquationCole" begin
        # The equation of state was designed by Cole to accurately describe the
        # physical properties of water under pressures up to 25 kbar.
        # We verify that it roughly coincides with online calculators (because I couldn't
        # find the original data from 1942 that Cole used to design his equation).
        @testset "Physical Properties of Water" begin
            # Standard speed of sound for pure water at 20°C
            sound_speed = 1484.0

            # Density of pure water at 20°C
            rest_density = 998.34

            # Work with pressures in ATM
            ATM = 101_325.0

            # 7.15 is the value used by Cole (see p. 39 of Cole 1948)
            state_equation = StateEquationCole(; sound_speed, exponent=7.15,
                                               reference_density=rest_density,
                                               background_pressure=1ATM)

            # These densities differ from an online calculator by less than 0.05%.
            # See https://www.omnicalculator.com/physics/water-density
            @test TrixiParticles.inverse_state_equation(state_equation, 1ATM) == 998.34
            @test TrixiParticles.inverse_state_equation(state_equation, 100ATM) ==
                  1002.8323123356663
            @test TrixiParticles.inverse_state_equation(state_equation, 500ATM) ==
                  1019.8235062499685

            # The online calculator says it's only accurate up to pressures of 1000 bar,
            # while Cole designed his equation to be accurate up to 25 kbar.
            # Therefore, these densities differ by up to 0.6%.
            @test TrixiParticles.inverse_state_equation(state_equation, 1000ATM) ==
                  1038.8747989986027
            @test TrixiParticles.inverse_state_equation(state_equation, 3000ATM) ==
                  1099.0607413267035
            @test TrixiParticles.inverse_state_equation(state_equation, 9000ATM) ==
                  1210.4689472510186
        end

        @testset "Background Pressure and Clipping" begin
            # Test background pressure
            background_pressures = [0.0, 10_000.0, 100_000.0, 200_000.0]

            for background_pressure in background_pressures
                state_equation = StateEquationCole(sound_speed=10.0, exponent=7,
                                                   reference_density=1000.0,
                                                   background_pressure=background_pressure)
                @test state_equation(1000.0) == background_pressure
                @test state_equation(1001.0) > background_pressure + 10
                # No pressure clipping
                @test state_equation(999.0) < background_pressure - 10
            end

            # Test pressure clipping
            state_equation = StateEquationCole(sound_speed=10.0, exponent=7,
                                               reference_density=1000.0,
                                               clip_negative_pressure=true)
            @test state_equation(999.0) == 0.0
            @test state_equation(900.0) == 0.0
        end
    end

    @testset verbose=true "Linear StateEquationCole" begin
        # This equation of state does not accurately describe the physical properties of
        # water under high pressures. However, it gives good results for low pressures,
        # where the relation of pressure and density is almost linear.
        # We verify again that it roughly coincides with online calculators.
        @testset "Physical Properties of Water" begin
            # Standard speed of sound for pure water at 20°C
            sound_speed = 1484.0

            # Density of pure water at 20°C
            rest_density = 998.34

            # Work with pressures in ATM
            ATM = 101_325.0

            state_equation = StateEquationCole(; sound_speed, exponent=1,
                                               reference_density=rest_density,
                                               background_pressure=1ATM)

            # These densities differ from an online calculator by less than 0.1%.
            # See https://www.omnicalculator.com/physics/water-density
            @test TrixiParticles.inverse_state_equation(state_equation, 1ATM) == 998.34
            @test TrixiParticles.inverse_state_equation(state_equation, 100ATM) ==
                  1002.8949541016121
            @test TrixiParticles.inverse_state_equation(state_equation, 500ATM) ==
                  1021.298809057621

            # For higher pressures, this state equation fails. These densities differ
            # by up to 16%.
            @test TrixiParticles.inverse_state_equation(state_equation, 1000ATM) ==
                  1044.3036277526319
            @test TrixiParticles.inverse_state_equation(state_equation, 3000ATM) ==
                  1136.3229025326757
            @test TrixiParticles.inverse_state_equation(state_equation, 9000ATM) ==
                  1412.380726872807
        end
    end

    # Don't show all state equations in the final overview
    @testset verbose=false "inverse_state_equation" begin
        # Verify that the `inverse_state_equation` actually is the inverse
        state_equations = [
            StateEquationCole(sound_speed=1484.0, exponent=7.15, reference_density=998.34,
                              background_pressure=101_325.0),
            StateEquationCole(sound_speed=10.0, exponent=7, reference_density=1000.0,
                              background_pressure=10_000.0),
            StateEquationCole(sound_speed=10.0, exponent=7, reference_density=1000.0,
                              background_pressure=0.0),
            StateEquationCole(sound_speed=10.0, exponent=7, reference_density=1000.0,
                              background_pressure=-100_000.0),
            StateEquationCole(sound_speed=1484.0, exponent=1, reference_density=998.34,
                              background_pressure=101_325.0),
            StateEquationCole(sound_speed=10.0, exponent=1, reference_density=1000.0,
                              background_pressure=100_000.0),
            StateEquationCole(sound_speed=10.0, exponent=1, reference_density=1000.0,
                              background_pressure=90_000.0),
            StateEquationCole(sound_speed=10.0, exponent=1, reference_density=1000.0,
                              background_pressure=0.0),
            StateEquationCole(sound_speed=10.0, exponent=1, reference_density=1000.0,
                              background_pressure=-100_000.0)
        ]

        densities = [100.0, 500.0, 900.0, 990.0, 1000.0, 1005.0, 1100.0, 1600.0]
        pressures = [-100.0, 0.0, 100.0, 10_000.0, 100_000.0, 100_000_000.0]

        @testset "$state_equation" for state_equation in state_equations
            for density in densities
                pressure = state_equation(density)
                @test TrixiParticles.inverse_state_equation(state_equation,
                                                            pressure) ≈ density
            end

            for pressure in pressures
                density = TrixiParticles.inverse_state_equation(state_equation, pressure)
                @test isapprox(state_equation(density), pressure, atol=2e-7, rtol=1e-10)
            end
        end
    end
end
