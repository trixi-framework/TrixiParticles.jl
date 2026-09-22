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
                state_equation = StateEquationCole(; sound_speed=10.0, exponent=7,
                                                   reference_density=1000.0,
                                                   background_pressure)
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

        @testset "Minimum Pressure" begin
            minimum_pressure = -100_000.0
            transition_width = 10_000.0
            sound_speed = 100.0
            exponent = 7.0
            reference_density = 1000.0

            state_equations = (StateEquationCole(; sound_speed, exponent, reference_density,
                                                 minimum_pressure,
                                                 minimum_pressure_transition_width=transition_width),
                               StateEquationAdaptiveCole(; min_sound_speed=sound_speed,
                                                         max_sound_speed=sound_speed,
                                                         mach_number_target=0.1,
                                                         exponent, reference_density,
                                                         minimum_pressure,
                                                         minimum_pressure_transition_width=transition_width))

            B = reference_density * sound_speed^2 / exponent
            density_from_raw_pressure(pressure) = reference_density *
                                                  (pressure / B + 1)^(1 / exponent)

            for state_equation in state_equations
                limited_pressure(raw_pressure) = state_equation(density_from_raw_pressure(raw_pressure))

                @test TrixiParticles.has_minimum_pressure(state_equation)
                @test state_equation(reference_density) == 0.0
                @test limited_pressure(-120_000.0) ≈ minimum_pressure
                @test limited_pressure(-110_000.0) ≈ minimum_pressure
                @test limited_pressure(-100_000.0) ≈ -97_500.0
                @test limited_pressure(-90_000.0) ≈ -90_000.0
                @test limited_pressure(10_000.0) ≈ 10_000.0

                epsilon = 0.01
                lower_transition_pressure = minimum_pressure - transition_width
                upper_transition_pressure = minimum_pressure + transition_width
                derivative(pressure) = (limited_pressure(pressure + epsilon) -
                                        limited_pressure(pressure - epsilon)) /
                                       (2 * epsilon)
                @test derivative(lower_transition_pressure) ≈ 0.0 atol=1e-6
                @test derivative(upper_transition_pressure) ≈ 1.0 atol=1e-6

                raw_pressures = range(-150_000.0, 10_000.0; length=101)
                pressures = limited_pressure.(raw_pressures)
                @test minimum(pressures) >= minimum_pressure
                @test issorted(pressures)

                for pressure in (-100_000.0, -99_000.0, -95_000.0, -90_000.0,
                                 0.0, 10_000.0)
                    density = TrixiParticles.inverse_state_equation(state_equation,
                                                                    pressure)
                    @test state_equation(density) ≈ pressure atol=1e-8
                end

                metadata = Dict{String, Any}()
                TrixiParticles.add_system_data!(metadata, state_equation)
                @test metadata["state_equation"]["minimum_pressure"] == minimum_pressure
                @test metadata["state_equation"]["minimum_pressure_transition_width"] ==
                      transition_width
            end

            unbounded_state_equation = StateEquationCole(; sound_speed, exponent,
                                                         reference_density)
            unbounded_metadata = Dict{String, Any}()
            TrixiParticles.add_system_data!(unbounded_metadata, unbounded_state_equation)
            @test !TrixiParticles.has_minimum_pressure(unbounded_state_equation)
            @test !haskey(unbounded_metadata["state_equation"], "minimum_pressure")

            state_equation32 = StateEquationCole(; sound_speed=100.0f0, exponent=7.0f0,
                                                 reference_density=1000.0f0,
                                                 minimum_pressure=-100_000.0f0,
                                                 minimum_pressure_transition_width=10_000.0f0)
            @test state_equation32(900.0f0) isa Float32

            adapted_state_equation = TrixiParticles.Adapt.adapt(Array, state_equations[2])
            @test TrixiParticles.has_minimum_pressure(adapted_state_equation)
            @test adapted_state_equation.minimum_pressure == minimum_pressure
            @test adapted_state_equation.minimum_pressure_transition_width ==
                  transition_width

            @test_throws ArgumentError StateEquationCole(; sound_speed, exponent,
                                                         reference_density,
                                                         minimum_pressure)
            @test_throws ArgumentError StateEquationCole(;
                                                         sound_speed, exponent,
                                                         reference_density,
                                                         minimum_pressure_transition_width=transition_width)
            @test_throws ArgumentError StateEquationCole(; sound_speed, exponent,
                                                         reference_density,
                                                         minimum_pressure,
                                                         minimum_pressure_transition_width=transition_width,
                                                         clip_negative_pressure=true)
            @test_throws ArgumentError StateEquationCole(; sound_speed, exponent,
                                                         reference_density,
                                                         minimum_pressure,
                                                         minimum_pressure_transition_width=110_000.0)
            @test_throws ArgumentError StateEquationCole(; sound_speed, exponent,
                                                         reference_density,
                                                         minimum_pressure=0.0,
                                                         minimum_pressure_transition_width=1.0)
            @test_throws ArgumentError StateEquationCole(; sound_speed, exponent,
                                                         reference_density,
                                                         minimum_pressure,
                                                         minimum_pressure_transition_width=0.0)
            @test_throws ArgumentError StateEquationCole(; sound_speed, exponent,
                                                         reference_density,
                                                         minimum_pressure=NaN,
                                                         minimum_pressure_transition_width=transition_width)
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
    @testset verbose=true "StateEquationIdealGas" begin
        # This is the classical ideal gas equation giving a linear relationship
        # between density and pressure.
        @testset "Physical Properties of Water" begin
            # Standard speed of sound for air at 20°C and 1 atm
            sound_speed = 343.3

            # Density of air at 20°C and 1 atm
            rest_density = 1.205

            # Heat capacity ratio of air
            gamma = 1.4

            # Work with pressures in ATM
            ATM = 101_325.0

            state_equation = StateEquationIdealGas(; sound_speed, gamma,
                                                   reference_density=rest_density,
                                                   background_pressure=1ATM)

            # Results by manual calculation
            @test TrixiParticles.inverse_state_equation(state_equation, 1ATM) ==
                  rest_density
            @test TrixiParticles.inverse_state_equation(state_equation, 100ATM) ==
                  120.36547777058719
            @test TrixiParticles.inverse_state_equation(state_equation, 500ATM) ==
                  601.8219536113436

            @test state_equation(rest_density) == 1ATM
            @test state_equation(120.36547777058719) == 100ATM
            @test state_equation(601.8219536113436) == 500ATM
        end
    end
end
