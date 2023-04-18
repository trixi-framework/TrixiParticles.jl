"""
    WCSPH(state_equation)

Weakly-compressible SPH introduced by (Monaghan, 1994). This formulation relies on a stiff
equation of state (see  [`StateEquationCole`](@ref)) that generates large pressure changes
for small density variations.

## References:
- Joseph J. Monaghan. "Simulating Free Surface Flows in SPH".
  In: Journal of Computational Physics 110 (1994), pages 399-406.
  [doi: 10.1006/jcph.1994.1034](https://doi.org/10.1006/jcph.1994.1034)
"""
struct WCSPH{SE}
    state_equation::SE
    function WCSPH(state_equation)
        new{typeof(state_equation)}(state_equation)
    end
end
