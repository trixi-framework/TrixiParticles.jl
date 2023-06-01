struct NoCorrection end

@doc raw"""
KernelCorrection()
KernelGradientCorrection()

Kernel correction uses Shepard interpolation to obtain a 0-th order accurate result.

## References:
- J. Bonet, T.-S.L. Lok.
  "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Computer Methods in Applied Mechanics and Engineering 180 (1999), pages 97-115.
  [doi: 10.1016/S0045-7825(99)00051-1](https://doi.org/10.1016/S0045-7825(99)00051-1)
- Mihai Basa, Nathan Quinlan, Martin Lastiwka.
  "Robustness and accuracy of SPH formulations for viscous flow".
  In: International Journal for Numerical Methods in Fluids 60 (2009), pages 1127-1148.
  [doi: 10.1002/fld.1927](https://doi.org/10.1002/fld.1927)
"""

# sorted in order of computational cost

# also referred to as 0th order correction (cheapest)
struct KernelCorrection end
