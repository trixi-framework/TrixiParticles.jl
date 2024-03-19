@doc raw"""
    StateEquationCole(; sound_speed, reference_density, exponent,
                      background_pressure=0.0, clip_negative_pressure=false)

Equation of state to describe the relationship between pressure and density
of water up to high pressures.

# Keywords
- `sound_speed`: Artificial speed of sound.
- `reference_density`: Reference density of the fluid.
- `exponent`: A value of `7` is usually used for most simulations.
- `background_pressure=0.0`: Background pressure.
"""
struct StateEquationCole{ELTYPE, CLIP} # Boolean to clip negative pressure
    sound_speed         :: ELTYPE
    exponent            :: ELTYPE
    reference_density   :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationCole(; sound_speed, reference_density, exponent,
                               background_pressure=0.0, clip_negative_pressure=false)
        new{typeof(sound_speed),
            clip_negative_pressure}(sound_speed, exponent, reference_density,
                                    background_pressure)
    end
end

clip_negative_pressure(::StateEquationCole{<:Any, CLIP}) where {CLIP} = CLIP

function (state_equation::StateEquationCole)(density)
    (; sound_speed, exponent, reference_density, background_pressure) = state_equation

    B = reference_density * sound_speed^2 / exponent
    pressure = B * ((density / reference_density)^exponent - 1) + background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0.0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::StateEquationCole, pressure)
    (; sound_speed, exponent, reference_density, background_pressure) = state_equation

    B = reference_density * sound_speed^2 / exponent
    tmp = (pressure - background_pressure) / B + 1

    return reference_density * tmp^(1 / exponent)
end
