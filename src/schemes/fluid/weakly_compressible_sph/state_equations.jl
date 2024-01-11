@doc raw"""
    StateEquationCole(; sound_speed, reference_density, exponent,
                      background_pressure=0.0, clip_negative_pressure=false)

Equation of state to describe the relationship between pressure and density
of water up to high pressures by Cole (Cole 1948, pp. 39 and 43).
The pressure ``p`` is calculated as
```math
    p = B \left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right) + p_{\text{background}},
```
where ``\rho`` denotes the density, ``\rho_0`` the reference density,
and ``p_{\text{background}}`` the background pressure, which is set to zero when applied to
free-surface flows (Adami et al., 2012).

The bulk modulus, ``B =  \frac{\rho_0 c^2}{\gamma}``, is calculated from the artificial
speed of sound ``c`` and the isentropic exponent ``\gamma``.

An ideal gas equation of state with a linear relationship between pressure and density can
be obtained by choosing `exponent=1`, i.e.
```math
    p = B \left( \frac{\rho}{\rho_0} -1 \right) = c^2(\rho - \rho_0).
```

For higher Reynolds numbers, `exponent=7` is recommended, whereas at lower Reynolds
numbers `exponent=1` yields more accurate pressure estimates since pressure and
density are proportional.

When using [`SummationDensity`](@ref) (or [`DensityReinitializationCallback`](@ref))
and free surfaces, initializing particles with equal spacing will cause underestimated
density and therefore strong attractive forces between particles at the free surface.
Setting `clip_negative_pressure=true` can avoid this.

# Keywords
- `sound_speed`: Artificial speed of sound.
- `reference_density`: Reference density of the fluid.
- `exponent`: To limit density variation to 1%, an average value of `7` is usually used.
- `background_pressure=0.0`: Background pressure.

## References:
- Robert H. Cole. "Underwater Explosions". Princeton University Press, 1948.
- J. P. Morris, P. J. Fox, Y. Zhu
  "Modeling Low Reynolds Number Incompressible Flows Using SPH ".
  In: Journal of Computational Physics , Vol. 136, No. 1, pages 214-226.
  [doi: 10.1006/jcph.1997.5776](https://doi.org/10.1006/jcph.1997.5776)
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)
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
