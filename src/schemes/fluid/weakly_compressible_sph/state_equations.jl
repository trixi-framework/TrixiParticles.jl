@doc raw"""
    StateEquationCole(; sound_speed, reference_density, gamma=7,
                      background_pressure=0.0, clip_negative_pressure=false)

Equation of state to describe the relationship between pressure and density
of water up to high pressures by Cole (Cole 1948, pp. 39 and 43).
The pressure ``p`` is calculated as
```math
    p = p_0 \left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right) + p_{\text{background}},
```
where ``c`` denotes the speed of sound, ``\rho`` the density,
``\rho_0`` an arbitrary reference density, ``p_0`` the pressure
at the reference density, and ``p_{\text{background}}`` the atmospheric or
background pressure which is set to zero when applied to free-surfaces flows
(Adami et al., 2012).

The reference pressure ``p_0 =  \frac{\rho_0 c^2}{\gamma}`` is calculated with an artificial
speed of sound ``c`` and ``\gamma`` as a parameter of density variation.

Morris et al. 1997 find that the square of the sound speed should be comparable with the
largest of
```math
c^2 \sim \frac{v_0^2}{\delta}, \frac{\nu v_0^2}{L_0 \delta}, \frac{g L_0}{\delta},
```
where ``\delta=\frac{\Delta \rho}{\rho_0} ``,  ``\nu`` is the kinematic viscosity, ``v_0`` is
a velocity scale, ``L_0`` is a length scale and ``g`` is a body force.

For simulations with particles filling all space, it is suggested to use ``\gamma=1`` and
``p_{\text{background}} = 0``, i.e.
```math
    p = c^2(\rho - \rho_0) = p_0 \left( \frac{\rho}{\rho_0} -1 \right),
```
which reduces significantly the magnitude of spurious pressure (Adami et al. 2013).

When using [`SummationDensity`](@ref) (or [`DensityReinitializationCallback`](@ref))
and free surfaces, initializing particles with equal spacing will cause underestimated
density and therefore strong attractive forces between particles at the free surface.
Setting `clip_negative_pressure=true` can avoid this.

# Keywords
- `gamma=7`: To limit density variation to 1%, an average value of ``\gamma=7`` is usually used.
- `background_pressure=0.0`: Background pressure
- `reference_density`: Reference density of fluid
- `sound_speed`: Artificial speed of sound.

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
- S. Adami, X. Y. Hu, N. A. Adams.
  "A transport-velocity formulation for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 241 (2013), pages 292-307.
  [doi: 10.1016/j.jcp.2013.01.043](http://dx.doi.org/10.1016/j.jcp.2013.01.043)
"""
struct StateEquationCole{ELTYPE, CLIP} # Boolean to clip negative pressure
    sound_speed         :: ELTYPE
    gamma               :: ELTYPE
    reference_density   :: ELTYPE
    background_pressure :: ELTYPE

    function StateEquationCole(; sound_speed, reference_density, gamma=7,
                               background_pressure=0.0, clip_negative_pressure=false)
        new{typeof(sound_speed),
            clip_negative_pressure}(sound_speed, gamma, reference_density,
                                    background_pressure)
    end
end

clip_negative_pressure(::StateEquationCole{<:Any, CLIP}) where {CLIP} = CLIP

function (state_equation::StateEquationCole)(density)
    (; sound_speed, gamma, reference_density, background_pressure) = state_equation

    p_0 = reference_density * sound_speed^2 / gamma
    pressure = p_0 * ((density / reference_density)^gamma - 1) + background_pressure

    # This is determined statically and has therefore no overhead
    if clip_negative_pressure(state_equation)
        return max(0.0, pressure)
    end

    return pressure
end

function inverse_state_equation(state_equation::StateEquationCole, pressure)
    (; sound_speed, gamma, reference_density, background_pressure) = state_equation

    p_0 = reference_density * sound_speed^2 / gamma
    tmp = (pressure - background_pressure) / p_0 + 1

    return reference_density * tmp^(1 / gamma)
end
