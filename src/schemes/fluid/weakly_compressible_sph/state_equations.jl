@doc raw"""
    StateEquationAdaptiveCole(; mach_number_target=0.1f0, min_sound_speed=10.0f0,
                              reference_density, max_sound_speed=100.0f0, exponent,
                              background_pressure=0.0f0, clip_negative_pressure=false)

This variant of [`StateEquationCole`](@ref) adapts the speed of sound during simulation to
balance accuracy and performance.
While a constant high speed of sound effectively reduces compressibility,
it increases computational cost proportionally.  This state equation reduces
computational cost compared to constant high values, while maintaining their advantages.
The speed of sound is initialized as `min_sound_speed`.

# Keywords
- `mach_number_target=0.1`: Target Mach number ratio for the simulation.
  The adaptive scheme obtains the current maximum particle velocity and adjusts the
  reference sound speed ``c`` to the ratio
  ```math
  c = \frac{U_\text{max}}{\mathrm{Ma_\text{limit}}}.
  ```
  A smaller `mach_number_target` enforces a higher sound speed (reducing compressibility
  effects but increasing computational cost), while a larger value allows stronger
  compressibility at lower runtime cost.
- `reference_density`: Reference density of the fluid.
- `min_sound_speed=10.0f0`: The minimum permissible speed of sound.
- `max_sound_speed=100.0f0`: The maximum permissible speed of sound.
- `exponent`: An exponent, typically 7 for water simulations.
- `background_pressure=0.0f0`: A constant background pressure.
- `clip_negative_pressure=false`: When true, negative pressure values are clipped to 0.0. This can prevent spurious surface tension effects but might allow for unphysical fluid rarefaction.
"""
struct StateEquationAdaptiveCole{ELTYPE, CLIP, SR} # Boolean to clip negative pressure
    sound_speed_ref     :: SR
    mach_number_target  :: ELTYPE
    min_sound_speed     :: ELTYPE
    max_sound_speed     :: ELTYPE
    exponent            :: ELTYPE
    reference_density   :: ELTYPE
    background_pressure :: ELTYPE
end

function StateEquationAdaptiveCole(; mach_number_target=0.1f0, min_sound_speed=10.0f0,
                                   reference_density, max_sound_speed=100.0f0, exponent,
                                   background_pressure=0.0f0,
                                   clip_negative_pressure=false)
    sound_speed = min_sound_speed
    return StateEquationAdaptiveCole{typeof(mach_number_target), clip_negative_pressure,
                                     typeof(Ref(sound_speed))}(Ref(sound_speed),
                                                               mach_number_target,
                                                               min_sound_speed,
                                                               max_sound_speed,
                                                               exponent,
                                                               reference_density,
                                                               background_pressure)
end

function Adapt.adapt_structure(to,
                               se::StateEquationAdaptiveCole{ELTYPE, CLIP}) where {ELTYPE,
                                                                                   CLIP}
    sound_speed_ref = Adapt.adapt_structure(to, se.sound_speed_ref)
    mach_number_target = Adapt.adapt_structure(to, se.mach_number_target)
    min_sound_speed = Adapt.adapt_structure(to, se.min_sound_speed)
    max_sound_speed = Adapt.adapt_structure(to, se.max_sound_speed)
    exponent = Adapt.adapt_structure(to, se.exponent)
    reference_density = Adapt.adapt_structure(to, se.reference_density)
    background_pressure = Adapt.adapt_structure(to, se.background_pressure)

    return StateEquationAdaptiveCole{ELTYPE, CLIP, typeof(sound_speed_ref)}(sound_speed_ref,
                                                                            mach_number_target,
                                                                            min_sound_speed,
                                                                            max_sound_speed,
                                                                            exponent,
                                                                            reference_density,
                                                                            background_pressure)
end

@inline function sound_speed(state_equation::StateEquationAdaptiveCole)
    return state_equation.sound_speed_ref[]
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
    (; exponent, reference_density, background_pressure) = state_equation

    B = reference_density * sound_speed(state_equation)^2 / exponent
    pressure = B * ((density / reference_density)^exponent - 1) + background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::Union{StateEquationCole,
                                                      StateEquationAdaptiveCole}, pressure)
    (; exponent, reference_density, background_pressure) = state_equation

    B = reference_density * sound_speed(state_equation)^2 / exponent
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
    (; reference_density, gamma, background_pressure) = state_equation
    pressure = (density - reference_density) * sound_speed(state_equation)^2 / gamma +
               background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::StateEquationIdealGas, pressure)
    (; reference_density, gamma, background_pressure) = state_equation
    density = (pressure - background_pressure) * gamma / sound_speed(state_equation)^2 +
              reference_density

    return density
end

@inline function sound_speed(state_equation)
    return state_equation.sound_speed
end
