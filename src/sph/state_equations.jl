struct StateEquationIdealGas{ELTYPE}
    sound_speed         ::ELTYPE
    reference_density   ::ELTYPE
    reference_pressure  ::ELTYPE
    background_pressure ::ELTYPE

    function StateEquationIdealGas(sound_speed, reference_density, reference_pressure;
                                   background_pressure=0.0)
        new{typeof(sound_speed)}(sound_speed, reference_density, reference_pressure,
                                 background_pressure)
    end
end

function (state_equation::StateEquationIdealGas)(density)
    @unpack sound_speed, reference_density, reference_pressure, background_pressure = state_equation

    return sound_speed^2 * (density - reference_density) + reference_pressure - background_pressure
end


struct StateEquationTait{ELTYPE}
    sound_speed         ::ELTYPE
    gamma               ::ELTYPE
    reference_density   ::ELTYPE
    reference_pressure  ::ELTYPE
    background_pressure ::ELTYPE

    function StateEquationTait(sound_speed, gamma, reference_density, reference_pressure;
                                   background_pressure=0.0)
        new{typeof(sound_speed)}(sound_speed, gamma, reference_density, reference_pressure,
                                 background_pressure)
    end
end

function (state_equation::StateEquationTait)(density)
    @unpack sound_speed, gamma, reference_density, reference_pressure, background_pressure = state_equation

    return reference_density * sound_speed^2 / gamma * ((density / reference_density)^gamma - 1) +
        reference_pressure - background_pressure
end
