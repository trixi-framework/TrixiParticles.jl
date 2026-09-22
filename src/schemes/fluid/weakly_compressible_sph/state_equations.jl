@doc raw"""
    StateEquationAdaptiveCole(; mach_number_target=0.1f0, min_sound_speed=10.0f0,
                              reference_density, max_sound_speed=100.0f0, exponent,
                              background_pressure=0.0f0, clip_negative_pressure=false,
                              minimum_pressure=nothing,
                              minimum_pressure_transition_width=nothing)

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
- `clip_negative_pressure=false`: When true, negative pressure values are clipped to 0.0.
  This can prevent spurious surface tension effects but allows for unphysical fluid
  rarefaction and causes incorrect fluid behavior in regions of low pressure.
  Note that negative pressure values of boundary particles in the
  [`BoundaryModelDummyParticles`](@ref) can lead to sticking artifacts at the boundary.
  The `BoundaryModelDummyParticles` itself also provides an option to clip negative
  pressure of boundary particles, which is a more targeted way to prevent sticking artifacts
  without affecting the fluid behavior in the rest of the domain.
- `minimum_pressure=nothing`: Optional lower pressure bound, for example the gauge vapor
  pressure when modeling cavitation. This cannot be combined with
  `clip_negative_pressure=true`.
- `minimum_pressure_transition_width=nothing`: Half-width of the continuously
  differentiable transition around `minimum_pressure`. Both pressure-floor keywords must
  be specified together. The unmodified Cole pressure is used above
  `minimum_pressure + minimum_pressure_transition_width`.
"""
struct StateEquationAdaptiveCole{ELTYPE, CLIP, SR, MIN_PRESSURE}
    sound_speed_ref::SR
    mach_number_target::ELTYPE
    min_sound_speed::ELTYPE
    max_sound_speed::ELTYPE
    exponent::ELTYPE
    reference_density::ELTYPE
    background_pressure::ELTYPE
    minimum_pressure::ELTYPE
    minimum_pressure_transition_width::ELTYPE
end

function StateEquationAdaptiveCole(; mach_number_target=0.1f0, min_sound_speed=10.0f0,
                                   reference_density, max_sound_speed=100.0f0, exponent,
                                   background_pressure=0.0f0,
                                   clip_negative_pressure=false, minimum_pressure=nothing,
                                   minimum_pressure_transition_width=nothing)
    sound_speed = min_sound_speed
    ELTYPE = typeof(mach_number_target)
    minimum_pressure_, transition_width_,
    has_minimum_pressure = minimum_pressure_parameters(ELTYPE, background_pressure,
                                                       clip_negative_pressure,
                                                       minimum_pressure,
                                                       minimum_pressure_transition_width)

    return StateEquationAdaptiveCole{ELTYPE, clip_negative_pressure,
                                     typeof(Ref(sound_speed)),
                                     has_minimum_pressure}(Ref(sound_speed),
                                                           mach_number_target,
                                                           min_sound_speed,
                                                           max_sound_speed,
                                                           exponent,
                                                           reference_density,
                                                           background_pressure,
                                                           minimum_pressure_,
                                                           transition_width_)
end

function Adapt.adapt_structure(to,
                               se::StateEquationAdaptiveCole{ELTYPE, CLIP, SR,
                                                             MIN_PRESSURE}) where {ELTYPE,
                                                                                   CLIP, SR,
                                                                                   MIN_PRESSURE}
    sound_speed_ref = Adapt.adapt_structure(to, se.sound_speed_ref)
    mach_number_target = Adapt.adapt_structure(to, se.mach_number_target)
    min_sound_speed = Adapt.adapt_structure(to, se.min_sound_speed)
    max_sound_speed = Adapt.adapt_structure(to, se.max_sound_speed)
    exponent = Adapt.adapt_structure(to, se.exponent)
    reference_density = Adapt.adapt_structure(to, se.reference_density)
    background_pressure = Adapt.adapt_structure(to, se.background_pressure)
    minimum_pressure = Adapt.adapt_structure(to, se.minimum_pressure)
    transition_width = Adapt.adapt_structure(to, se.minimum_pressure_transition_width)

    return StateEquationAdaptiveCole{ELTYPE, CLIP, typeof(sound_speed_ref),
                                     MIN_PRESSURE}(sound_speed_ref,
                                                   mach_number_target,
                                                   min_sound_speed,
                                                   max_sound_speed,
                                                   exponent,
                                                   reference_density,
                                                   background_pressure,
                                                   minimum_pressure,
                                                   transition_width)
end

@inline function sound_speed(state_equation::StateEquationAdaptiveCole)
    return state_equation.sound_speed_ref[]
end

@doc raw"""
    StateEquationCole(; sound_speed, reference_density, exponent,
                      background_pressure=0.0, clip_negative_pressure=false,
                      minimum_pressure=nothing,
                      minimum_pressure_transition_width=nothing)

Equation of state to describe the relationship between pressure and density
of water up to high pressures.

# Keywords
- `sound_speed`:             Artificial speed of sound.
- `reference_density`:       Reference density of the fluid.
- `exponent`:                A value of `7` is usually used for most simulations.
- `background_pressure=0.0`: Background pressure.
- `clip_negative_pressure=false`: Negative pressure values are clipped to 0, which prevents spurious surface tension with `SummationDensity` but allows unphysical rarefaction of the fluid.
- `minimum_pressure=nothing`: Optional lower pressure bound, for example the gauge vapor
  pressure when modeling cavitation. This cannot be combined with
  `clip_negative_pressure=true`.
- `minimum_pressure_transition_width=nothing`: Half-width of the continuously
  differentiable transition around `minimum_pressure`. Both pressure-floor keywords must
  be specified together. The unmodified Cole pressure is used above
  `minimum_pressure + minimum_pressure_transition_width`.
"""
struct StateEquationCole{ELTYPE, CLIP, MIN_PRESSURE}
    sound_speed::ELTYPE
    exponent::ELTYPE
    reference_density::ELTYPE
    background_pressure::ELTYPE
    minimum_pressure::ELTYPE
    minimum_pressure_transition_width::ELTYPE

    function StateEquationCole(; sound_speed, reference_density, exponent,
                               background_pressure=0.0, clip_negative_pressure=false,
                               minimum_pressure=nothing,
                               minimum_pressure_transition_width=nothing)
        ELTYPE = typeof(sound_speed)
        minimum_pressure_, transition_width_,
        has_minimum_pressure = minimum_pressure_parameters(ELTYPE, background_pressure,
                                                           clip_negative_pressure,
                                                           minimum_pressure,
                                                           minimum_pressure_transition_width)

        new{typeof(sound_speed),
            clip_negative_pressure, has_minimum_pressure}(sound_speed, exponent,
                                                          reference_density,
                                                          background_pressure,
                                                          minimum_pressure_,
                                                          transition_width_)
    end
end

clip_negative_pressure(::StateEquationCole{<:Any, CLIP}) where {CLIP} = CLIP
clip_negative_pressure(::StateEquationAdaptiveCole{<:Any, CLIP}) where {CLIP} = CLIP

@inline function has_minimum_pressure(::StateEquationCole{<:Any, CLIP,
                                                          MIN_PRESSURE}) where {CLIP,
                                                                                MIN_PRESSURE}
    return MIN_PRESSURE
end

@inline function has_minimum_pressure(::StateEquationAdaptiveCole{<:Any, CLIP, SR,
                                                                  MIN_PRESSURE}) where {CLIP,
                                                                                        SR,
                                                                                        MIN_PRESSURE}
    return MIN_PRESSURE
end

function minimum_pressure_parameters(::Type{ELTYPE}, background_pressure,
                                     clip_negative_pressure, minimum_pressure,
                                     transition_width) where {ELTYPE}
    if isnothing(minimum_pressure) != isnothing(transition_width)
        throw(ArgumentError("`minimum_pressure` and " *
                            "`minimum_pressure_transition_width` must be specified together"))
    end

    if isnothing(minimum_pressure)
        return zero(ELTYPE), zero(ELTYPE), false
    end

    clip_negative_pressure &&
        throw(ArgumentError("`minimum_pressure` cannot be combined with " *
                            "`clip_negative_pressure=true`"))

    background_pressure_ = convert(ELTYPE, background_pressure)
    minimum_pressure_ = convert(ELTYPE, minimum_pressure)
    transition_width_ = convert(ELTYPE, transition_width)

    isfinite(minimum_pressure_) ||
        throw(ArgumentError("`minimum_pressure` must be finite"))
    isfinite(transition_width_) ||
        throw(ArgumentError("`minimum_pressure_transition_width` must be finite"))
    transition_width_ > zero(ELTYPE) ||
        throw(ArgumentError("`minimum_pressure_transition_width` must be positive"))
    minimum_pressure_ < background_pressure_ ||
        throw(ArgumentError("`minimum_pressure` must be below `background_pressure`"))
    transition_width_ <= background_pressure_ - minimum_pressure_ ||
        throw(ArgumentError("the pressure-floor transition must end at or below " *
                            "`background_pressure`"))

    return minimum_pressure_, transition_width_, true
end

@inline function apply_minimum_pressure(state_equation, pressure)
    # This branch is compile-time constant through the state-equation type parameter.
    has_minimum_pressure(state_equation) || return pressure

    minimum_pressure = state_equation.minimum_pressure
    transition_width = state_equation.minimum_pressure_transition_width
    lower_transition_pressure = minimum_pressure - transition_width
    upper_transition_pressure = minimum_pressure + transition_width

    pressure <= lower_transition_pressure && return minimum_pressure
    pressure >= upper_transition_pressure && return pressure

    normalized_pressure = (pressure - lower_transition_pressure) / (2 * transition_width)
    return minimum_pressure + transition_width * normalized_pressure^2
end

@inline function remove_minimum_pressure(state_equation, pressure)
    has_minimum_pressure(state_equation) || return pressure

    minimum_pressure = state_equation.minimum_pressure
    transition_width = state_equation.minimum_pressure_transition_width
    lower_transition_pressure = minimum_pressure - transition_width
    upper_transition_pressure = minimum_pressure + transition_width

    # The constant part of the floor is not uniquely invertible. Use its highest raw
    # pressure, which gives the least rarefied density compatible with the floor.
    pressure <= minimum_pressure && return lower_transition_pressure
    pressure >= upper_transition_pressure && return pressure

    normalized_pressure = sqrt((pressure - minimum_pressure) / transition_width)
    return lower_transition_pressure + 2 * transition_width * normalized_pressure
end

function (state_equation::Union{StateEquationCole, StateEquationAdaptiveCole})(density)
    (; exponent, reference_density, background_pressure) = state_equation

    B = reference_density * sound_speed(state_equation)^2 / exponent
    pressure = B * ((density / reference_density)^exponent - 1) + background_pressure
    pressure = apply_minimum_pressure(state_equation, pressure)

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
    pressure = remove_minimum_pressure(state_equation, pressure)
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
