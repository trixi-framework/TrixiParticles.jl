struct NoCorrection end

@doc raw"""
KernelCorrection()
KernelGradientCorrection()

Cubic spline kernel by Schoenberg (Schoenberg, 1946), given by
```math
W(r, h) = \frac{1}{h^d} w(r/h)
```
with
```math
w(q) = \sigma \begin{cases}
    \frac{1}{4} (2 - q)^3 - (1 - q)^3   & \text{if } 0 \leq q < 1, \\
    \frac{1}{4} (2 - q)^3               & \text{if } 1 \leq q < 2, \\
    0                                   & \text{if } q \geq 2, \\
\end{cases}
```
where ``d`` is the number of dimensions and ``\sigma = 17/(7\pi)``
in two dimensions or ``\sigma = 1/\pi`` in three dimensions is a normalization factor.

## References:
- Bonet and Lok. "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Comput. Methods Appl. Mech. Eng. (1999), pages 97-115.
- Basa et al. "Robustness and accuracy of SPH formulations for viscous flow".
  In: Int. J. Numer. Meth. Fluids (2009), pages 1127-1148.
"""

# sorted in order of computational cost

# also refered to as 0th order correction (cheapest)
struct KernelCorrection end
