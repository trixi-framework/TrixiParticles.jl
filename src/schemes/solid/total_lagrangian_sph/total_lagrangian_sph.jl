@doc raw"""
    TotalLagrangianSPH(material_densities, smoothing_kernel,
                                smoothing_length, young_modulus, poisson_ratio,
                                boundary_scheme;
                                n_fixed_particles=0, penalty_force=nothing)

Scheme to model an elastic solid.

A Total Lagrangian framework is used wherein the governing equations are forumlated such that
all relevant quantities and operators are measured with respect to the
initial configuration (O’Connor & Rogers 2021, Belytschko et al. 2000).
The governing equations with respect to the initial configuration are given by:
```math
\frac{\mathrm{D}\bm{v}}{\mathrm{D}t} = \frac{1}{\rho_0} \nabla_0 \cdot \bm{P} + \bm{g},
```
where the zero subscript denotes a derivative with respect to the initial configuration
and $\bm{P}$ is the first Piola-Kirchhoff (PK1) stress tensor.

The discretized version of this equation is given by (O’Connor & Rogers 2021):
```math
\frac{\mathrm{d}\bm{v}_a}{\mathrm{d}t} = \sum_b m_{0b}
    \left( \frac{\bm{P}_a \bm{L}_{0a}}{\rho_{0a}^2} + \frac{\bm{P}_b \bm{L}_{0b}}{\rho_{0b}^2} \right)
    \nabla_{0a} W(\bm{X}_{ab}) + \frac{\bm{f}_a^{PF}}{m_{0a}} + \bm{g},
```
with
```math
\bm{L}_{0a} := \left( \sum_{b} \frac{m_{0b}}{\rho_{0b}} \nabla_{0a} W(\bm{X}_{ab}) \bm{X}_{ab}^T \right)^{-1} \in \R^{d \times d}.
```
The subscripts $a$ and $b$ denote quantities of particle $a$ and $b$, respectively.
The zero subscript on quantities denotes that the quantity is to be measured in the initial configuration.
The difference in the initial coordinates is denoted by $\bm{X}_{ab} = \bm{X}_a - \bm{X}_b$,
the difference in the current coordinates is denoted by $\bm{x}_{ab} = \bm{x}_a - \bm{x}_b$.

For the computation of the PK1 stress tensor, the deformation gradient $\bm{J}$ is computed per particle as
```math
\bm{J}_a = \sum_b \frac{m_{0b}}{\rho_{0b}} \bm{x}_{ba} (\bm{L}_{0a}\nabla_{0a} W(\bm{X}_{ab}))^T \\
    \qquad  = -\left(\sum_b \frac{m_{0b}}{\rho_{0b}} \bm{x}_{ab} (\nabla_{0a} W(\bm{X}_{ab}))^T \right) \bm{L}_{0a}^T
```
with $1 \leq i,j \leq d$.
From the deformation gradient, the Green-Lagrange strain
```math
\bm{E} = \frac{1}{2}(\bm{J}^T\bm{J} - \bm{I})
```
and the second Piola-Kirchhoff stress tensor
```math
\bm{S} = \lambda \operatorname{tr}(\bm{E}) \bm{I} + 2\mu \bm{E}
```
are computed to obtain the PK1 stress tensor as
```math
\bm{P} = \bm{J}\bm{S}.
```

Here,
```math
\mu = \frac{E}{2(1 + \nu)}
```
and
```math
\lambda = \frac{E\nu}{(1 + \nu)(1 - 2\nu)}
```
are the Lamé coefficients, where $E$ is the Young's modulus and $\nu$ is the Poisson ratio.

The term $\bm{f}_a^{PF}$ is an optional penalty force. See e.g. [`PenaltyForceGanzenmueller`](@ref).

## References:
- Joseph O’Connor, Benedict D. Rogers.
  "A fluid–structure interaction model for free-surface flows and flexible structures using
  smoothed particle hydrodynamics on a GPU".
  In: Journal of Fluids and Structures 104 (2021).
  [doi: 10.1016/J.JFLUIDSTRUCTS.2021.103312](https://doi.org/10.1016/J.JFLUIDSTRUCTS.2021.103312)
- Ted Belytschko, Yong Guo, Wing Kam Liu, Shao Ping Xiao.
  "A unified stability analysis of meshless particle methods".
  In: International Journal for Numerical Methods in Engineering 48 (2000), pages 1359–1400.
  [doi: 10.1002/1097-0207](https://doi.org/10.1002/1097-0207)
"""
struct TotalLagrangianSPH{BM, ELTYPE <: Real, K, PF}
    current_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    correction_matrix   :: Array{ELTYPE, 3} # [i, j, particle]
    pk1_corrected       :: Array{ELTYPE, 3} # [i, j, particle]
    deformation_grad    :: Array{ELTYPE, 3} # [i, j, particle]
    material_density    :: Array{ELTYPE, 1} # [particle]
    n_moving_particles  :: Int
    young_modulus       :: ELTYPE
    poisson_ratio       :: ELTYPE
    lame_lambda         :: ELTYPE
    lame_mu             :: ELTYPE
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    boundary_scheme     :: BM
    penalty_force       :: PF

    function TotalLagrangianSPH(material_densities, smoothing_kernel,
                                smoothing_length, young_modulus, poisson_ratio,
                                boundary_scheme;
                                n_fixed_particles=0, penalty_force=nothing)
        ELTYPE = eltype(smoothing_length)
        nparticles = length(material_densities)
        NDIMS = ndims(smoothing_kernel)

        current_coordinates = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
        correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
        pk1_corrected = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
        deformation_grad = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)

        n_moving_particles = nparticles - n_fixed_particles

        lame_lambda = young_modulus * poisson_ratio /
                      ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        lame_mu = 0.5 * young_modulus / (1 + poisson_ratio)

        return new{typeof(boundary_scheme), ELTYPE, typeof(smoothing_kernel),
                   typeof(penalty_force)}(current_coordinates, correction_matrix,
                                          pk1_corrected,
                                          deformation_grad, material_densities,
                                          n_moving_particles, young_modulus, poisson_ratio,
                                          lame_lambda, lame_mu, smoothing_kernel,
                                          smoothing_length, boundary_scheme, penalty_force)
    end
end

function initialize!(scheme::TotalLagrangianSPH, container, neighborhood_search)
    @unpack correction_matrix = scheme
    NDIMS = ndims(container)

    if ndims(scheme.smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    # Calculate kernel correction matrix
    calc_correction_matrix!(correction_matrix, neighborhood_search, container)
end

@inline function v_nvariables(container, scheme::TotalLagrangianSPH)
    return v_nvariables(container, scheme, scheme.boundary_scheme)
end

@inline function v_nvariables(container, ::TotalLagrangianSPH,
                              ::BoundarySchemeDummyParticles{ContinuityDensity})
    # Density is integrated with `ContinuityDensity`
    return ndims(container) + 1
end

@inline function v_nvariables(container, ::TotalLagrangianSPH, boundary_scheme)
    return ndims(container)
end

@inline function current_coordinates(u, scheme::TotalLagrangianSPH)
    return scheme.current_coordinates
end

@inline function current_coords(container::ParticleContainer{TotalLagrangianSPH}, particle)
    # For this scheme, the current coordinates are stored in the scheme directly,
    # so we don't need a `u` array. This function is only to be used in this file
    # when no `u` is available.
    current_coords(nothing, container, particle)
end

@inline function n_moving_particles(container, scheme::TotalLagrangianSPH)
    return scheme.n_moving_particles
end

@inline function current_velocity(v, container, scheme::TotalLagrangianSPH, particle)
    if particle > n_moving_particles(container)
        return SVector(ntuple(_ -> 0.0, Val(ndims(container))))
    end

    return extract_svector(v, container, particle)
end

# include("penalty_force.jl") TODO comment back in
include("rhs.jl")
