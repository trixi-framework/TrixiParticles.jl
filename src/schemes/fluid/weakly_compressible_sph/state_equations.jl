@doc raw"""
StateEquationAdaptiveCole(; machnumber=0.1, average_velocity=1.0,
reference_density, max_sound_speed=100.0, exponent, background_pressure=0.0,
clip_negative_pressure=false)

This variant is adaptive, allowing the speed of sound to be updated during a simulation.
The speed of sound is initialized as average_velocity / machnumber.

# Keywords
- `machnumber=0.1`: The Mach number of the fluid flow, used to initialize the speed of sound.
- `average_velocity=1.0`: The estimated average velocity of the fluid.
- `reference_density`: Reference density of the fluid.
- `max_sound_speed=100.0`: The maximum permissible speed of sound.
- `exponent`: An exponent, typically 7 for water simulations.
- `background_pressure=0.0`: A constant background pressure.
- `clip_negative_pressure=false`: When true, negative pressure values are clipped to 0.0. This can prevent spurious surface tension effects but might allow for unphysical fluid rarefaction.
"""
struct StateEquationAdaptiveCole{ELTYPE, CLIP} # Boolean to clip negative pressure
    sound_speed_ref     :: Base.RefValue{ELTYPE}
    machnumber          :: ELTYPE
    average_velocity    :: ELTYPE
    max_sound_speed     :: ELTYPE
    exponent            :: ELTYPE
    reference_density   :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationAdaptiveCole(; machnumber=0.1, average_velocity=1, reference_density, max_sound_speed=100, exponent,
                               background_pressure=0.0, clip_negative_pressure=false)
        sound_speed = average_velocity / machnumber
        new{typeof(machnumber),
            clip_negative_pressure}(Ref(sound_speed), machnumber, average_velocity, max_sound_speed, exponent, reference_density,
                                    background_pressure)
    end
end

# unwrap sound_speed on read
function Base.getproperty(se::StateEquationAdaptiveCole, name::Symbol)
    if name === :sound_speed
        return se.sound_speed_ref[]  # expose as plain value
    else
        return getfield(se, name)
    end
end

# allow assignment to .sound_speed to mutate the underlying Ref
function Base.setproperty!(se::StateEquationAdaptiveCole, name::Symbol, val)
    if name === :sound_speed
        se.sound_speed_ref[] = val
    else
        # fall back to default, will error if struct is immutable for other fields
        Base.setfield!(se, name, val)
    end
end

@doc raw"""
    StateEquationCole(; sound_speed, reference_density, exponent,
                      background_pressure=0.0, clip_negative_pressure=false)

Equation of state to describe the relationship between pressure and density
of water up to high pressures.

# Keywords
- `sound_speed`:             Artificial speed of sound.
- `reference_density`:       Reference density of the fluid.
- `exponent`:                A value of `7` is usually used for most simulations.
- `background_pressure=0.0`: Background pressure.
- `clip_negative_pressure=false`: Negative pressure values are clipped to 0, which prevents spurious surface tension with `SummationDensity` but allows unphysical rarefaction of the fluid.
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
clip_negative_pressure(::StateEquationAdaptiveCole{<:Any, CLIP}) where {CLIP} = CLIP

function (state_equation::Union{StateEquationCole, StateEquationAdaptiveCole})(density)
    (; sound_speed, exponent, reference_density, background_pressure) = state_equation

    B = reference_density * sound_speed^2 / exponent
    pressure = B * ((density / reference_density)^exponent - 1) + background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::Union{StateEquationCole, StateEquationAdaptiveCole}, pressure)
    (; sound_speed, exponent, reference_density, background_pressure) = state_equation

    B = reference_density * sound_speed^2 / exponent
    tmp = (pressure - background_pressure) / B + 1

    return reference_density * tmp^(1 / exponent)
end

@doc raw"""
    StateEquationIdealGas( ;sound_speed, reference_density, gamma, background_pressure=0.0,
                           clip_negative_pressure=false)

Equation of state to describe the relationship between pressure and density
of a gas using the Ideal Gas Law.

# Keywords
- `sound_speed`                 : Artificial speed of sound.
- `reference_density`           : Reference density of the fluid.
- `gamma`                       : Heat-capacity ratio
- `background_pressure=0.0`     : Background pressure.
- `clip_negative_pressure=false`: Negative pressure values are clipped to 0, which prevents spurious surface tension with `SummationDensity` but allows unphysical rarefaction of the fluid.
"""
struct StateEquationIdealGas{ELTYPE, CLIP}
    sound_speed         :: ELTYPE
    reference_density   :: ELTYPE
    gamma               :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationIdealGas(; sound_speed, reference_density, gamma,
                                   background_pressure=0.0, clip_negative_pressure=false)
        new{typeof(sound_speed), clip_negative_pressure}(sound_speed, reference_density,
                                                         gamma, background_pressure)
    end
end

clip_negative_pressure(::StateEquationIdealGas{<:Any, CLIP}) where {CLIP} = CLIP

function (state_equation::StateEquationIdealGas)(density)
    (; reference_density, sound_speed, gamma, background_pressure) = state_equation
    pressure = (density - reference_density) * sound_speed^2 / gamma + background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::StateEquationIdealGas, pressure)
    (; reference_density, sound_speed, gamma, background_pressure) = state_equation
    density = (pressure - background_pressure) * gamma / sound_speed^2 + reference_density

    return density
end
