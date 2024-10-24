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

@inline sound_speed(eos) = eos.sound_speed

@doc raw"""
    StateEquationIdealGas(; gas_constant, temperature, gamma)
Equation of state to describe the relationship between pressure and density
of a gas using the Ideal Gas Law.
# Keywords
- `gas_constant`: Specific gas constant (R) for the gas, typically in J/(kg*K).
- `temperature` : Absolute temperature of the gas in Kelvin.
- `gamma`       : Heat-capacity ratio
This struct calculates the pressure of a gas from its density using the formula:
\[ P = \rho \cdot R \cdot T \]
where \( P \) is pressure, \( \rho \) is density, \( R \) is the gas constant, and \( T \) is temperature.
Note:
For basic WCSPH this boils down to the assumption of a linear pressure-density relationship.
"""
struct StateEquationIdealGas{ELTYPE}
    gas_constant :: ELTYPE
    temperature  :: ELTYPE
    gamma        :: ELTYPE

    function StateEquationIdealGas(; gas_constant, temperature, gamma)
        new{typeof(gas_constant)}(gas_constant, temperature, gamma)
    end
end

function (state_equation::StateEquationIdealGas)(density)
    (; gas_constant, temperature) = state_equation
    pressure = density * gas_constant * temperature
    return pressure
end

# This version is for simulations that include a temperature.
function (state_equation::StateEquationIdealGas)(density, internal_energy)
    (; gamma) = state_equation
    pressure = (gamma - 1.0) * density * internal_energy
    return pressure
end

function inverse_state_equation(state_equation::StateEquationIdealGas, pressure)
    (; gas_constant, temperature) = state_equation
    density = pressure / (gas_constant * temperature)
    return density
end

# This version is for simulations that include a temperature.
function inverse_state_equation(state_equation::StateEquationIdealGas, pressure,
                                internal_energy)
    gamma = state_equation.gamma

    density = pressure / ((gamma - 1.0) * internal_energy)
    return density
end

@inline sound_speed(eos::StateEquationIdealGas) = sqrt(eos.gamma * eos.gas_constant *
                                                       eos.temperature)
# This version is for simulations that include a temperature.
@inline sound_speed(eos::StateEquationIdealGas, pressure, density) = sqrt(eos.gamma *
                                                                          pressure /
                                                                          (density *
                                                                           eos.gas_constant))
