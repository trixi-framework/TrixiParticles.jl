struct NoCorrection end

@doc raw"""
KernelCorrection()
KernelGradientCorrection()

Kernel correction uses Shepard interpolation to obtain a 0-th order accurate result.

## References:
- Bonet and Lok. "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Comput. Methods Appl. Mech. Eng. (1999), pages 97-115.
- Basa et al. "Robustness and accuracy of SPH formulations for viscous flow".
  In: Int. J. Numer. Meth. Fluids (2009), pages 1127-1148.
"""

# sorted in order of computational cost

# also refered to as 0th order correction (cheapest)
struct KernelCorrection end
