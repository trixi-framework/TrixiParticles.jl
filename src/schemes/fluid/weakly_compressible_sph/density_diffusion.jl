@doc raw"""
    DensityDiffusion

An abstract supertype of all density diffusion formulations.

Currently, the following formulations are available:

| Formulation                                 | Suitable for Steady-State Simulations | Low Computational Cost |
| :------------------------------------------ | :------------------------------------ | :--------------------- |
| [`DensityDiffusionMolteniColagrossi`](@ref) | ❌                                    | ✅                     |
| [`DensityDiffusionFerrari`](@ref)           | ❌                                    | ✅                     |
| [`DensityDiffusionAntuono`](@ref)           | ✅                                    | ❌                     |

## SPH Formulation

All formulations extend the continuity equation (see [`ContinuityDensity`](@ref))
by an additional term
```math
\frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \sum_{b} m_b v_{ab} \cdot \nabla_{r_a} W(\Vert r_{ab} \Vert, h)
    + \delta h c \sum_{b} V_b \psi_{ab} \cdot \nabla_{r_a} W(\Vert r_{ab} \Vert, h),
```
where ``V_b = m_b / rho_b`` is the volume of particle ``b`` and ``psi_{ab}`` depends on
the density diffusion formulation.
Also, ``\rho_a`` denotes the density of particle ``a`` and ``r_{ab} = r_a - r_b`` is the
difference of the coordinates, ``v_{ab} = v_a - v_b`` of the velocities of particles
``a`` and ``b``.

## Numerical Results

All formulations remove numerical noise in the pressure field and produce more
accurate results than weakly commpressible SPH without density diffusion.
This can be demonstrated with with dam break examples in 2D and 3D. Here, ``δ = 0.1`` has
been used for all formulations.

![density_diffusion_2d](https://lh3.googleusercontent.com/drive-viewer/AK7aPaBL-tqW6p9ry3NHvNnHVNufRfh_NSz0Le4vJ4n2rS-10Vr3Dkm2Cjb4T861vk6yhnvqMgS_PLXeZsNoVepIfYgpw-hlgQ=s1600)

![density_diffusion_3d](https://lh3.googleusercontent.com/drive-viewer/AK7aPaDKc0DCJfFH606zWFkjutMYzs70Y4Ot_33avjcIRxV3xNbrX1gqx6EpeSmysai338aRsOoqJ8B1idUs5U30SA_o12OQ=s1600)

The simpler formulations [`DensityDiffusionMolteniColagrossi`](@ref) and
[`DensityDiffusionFerrari`](@ref) do not solve the hydrostatic problem and lead to incorrect
solutions in long-running steady-state hydrostatic simulations with free surfaces
(Antuono et al., 2012). This can be seen when running the simple rectangular tank example
until ``t = 40`` (again using ``δ = 0.1``):

![density_diffusion_tank](https://lh3.googleusercontent.com/drive-viewer/AK7aPaCf1gDlbxkQjxpyffPJ-ijx-DdVxlwUVb_DLYIW4X5E0hkDeJcuAqCae6y4eDydgTKe752zWa08tKVL5yhB-ad8Uh8J=s1600)

[`DensityDiffusionAntuono`](@ref) adds a correction term to solve this problem, but this
term is very expensive and adds about 30--40% of computational cost.

## References
- M. Antuono, A. Colagrossi, S. Marrone.
  "Numerical Diffusive Terms in Weakly-Compressible SPH Schemes."
  In: Computer Physics Communications 183.12 (2012), pages 2570--2580.
  [doi: 10.1016/j.cpc.2012.07.006](https://doi.org/10.1016/j.cpc.2012.07.006)
"""
abstract type DensityDiffusion end

# Most density diffusion formulations don't need updating
function update!(density_diffusion, neighborhood_search, v, u, system, semi)
    return density_diffusion
end

@doc raw"""
    DensityDiffusionMolteniColagrossi(delta)

The commonly used density diffusion term by Molteni & Colagrossi (2009).

The term ``\psi_{ab}`` in the continuity equation in [`DensityDiffusion`](@ref) is defined
by
```math
\psi_{ab} = 2(\rho_a - \rho_b) \frac{r_{ab}}{\Vert r_{ab} \Vert^2},
```
where ``\rho_a`` and ``\rho_b`` denote the densities of particles ``a`` and ``b`` respectively
and ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``.

See [`DensityDiffusion`](@ref) for an overview and comparison of implemented density
diffusion terms.

## References:
- Diego Molteni, Andrea Colagrossi.
  "A Simple Procedure to Improve the Pressure Evaluation in Hydrodynamic Context Using the SPH."
  In: Computer Physics Communications 180.6 (2009), pages 861--872.
  [doi: 10.1016/j.cpc.2008.12.004](https://doi.org/10.1016/j.cpc.2008.12.004)
"""
struct DensityDiffusionMolteniColagrossi{ELTYPE} <: DensityDiffusion
    delta::ELTYPE
end

@inline function density_diffusion_psi(::DensityDiffusionMolteniColagrossi, rho_a, rho_b,
                                       pos_diff, distance, system, particle, neighbor)
    return 2 * (rho_a - rho_b) * pos_diff / distance^2
end

@doc raw"""
    DensityDiffusionFerrari()

A density diffusion term by Ferrari et al. (2009).

The term ``\psi_{ab}`` in the continuity equation in [`DensityDiffusion`](@ref) is defined
by
```math
\psi_{ab} = \frac{\rho_a - \rho_b}{2h} \frac{r_{ab}}{\Vert r_{ab} \Vert},
```
where ``\rho_a`` and ``\rho_b`` denote the densities of particles ``a`` and ``b`` respectively,
``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b`` and
``h`` is the smoothing length.

See [`DensityDiffusion`](@ref) for an overview and comparison of implemented density
diffusion terms.

## References:
- Angela Ferrari, Michael Dumbser, Eleuterio F. Toro, Aronne Armanini.
  "A New 3D Parallel SPH Scheme for Free Surface Flows."
  In: Computers & Fluids 38.6 (2009), pages 1203--1217.
  [doi: 10.1016/j.compfluid.2008.11.012](https://doi.org/10.1016/j.compfluid.2008.11.012).
"""
struct DensityDiffusionFerrari <: DensityDiffusion
    delta::Int

    # δ is always 1 in this formulation
    DensityDiffusionFerrari() = new(1)
end

@inline function density_diffusion_psi(::DensityDiffusionFerrari, rho_a, rho_b,
                                       pos_diff, distance, system, particle, neighbor)
    (; smoothing_length) = system

    return ((rho_a - rho_b) / 2smoothing_length) * pos_diff / distance
end

@doc raw"""
    DensityDiffusionAntuono(delta, initial_condition)

The commonly used density diffusion terms by Antuono et al. (2010), also referred to as
δ-SPH. The density diffusion term by Molteni & Colagrossi (2009) is extended by a second
term, which is nicely written down by Antuono et al. (2012).

The term ``\psi_{ab}`` in the continuity equation in [`DensityDiffusion`](@ref) is defined
by
```math
\psi_{ab} = 2\left(\rho_a - \rho_b - \frac{1}{2}\big(\nabla\rho^L_a + \nabla\rho^L_b\big) \cdot r_{ab}\right)
    \frac{r_{ab}}{\Vert r_{ab} \Vert^2},
```
where ``\rho_a`` and ``\rho_b`` denote the densities of particles ``a`` and ``b`` respectively
and ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles ``a`` and ``b``.
The symbol ``\nabla\rho^L_a`` denotes the renormalized density gradient defined as
```math
\nabla\rho^L_a = -\sum_b (\rho_a - \rho_a) V_b L_a \nabla_{r_a} W(\Vert r_{ab} \Vert, h)
```
with
```math
L_a := \left( -\sum_{b} V_b r_{ab} \otimes \nabla_{r_a} W(\Vert r_{ab} \Vert, h) \right)^{-1} \in \R^{d \times d},
```
where ``d`` is the number of dimensions.

See [`DensityDiffusion`](@ref) for an overview and comparison of implemented density
diffusion terms.

## References:
- M. Antuono, A. Colagrossi, S. Marrone, D. Molteni.
  "Free-Surface Flows Solved by Means of SPH Schemes with Numerical Diffusive Terms."
  In: Computer Physics Communications 181.3 (2010), pages 532--549.
  [doi: 10.1016/j.cpc.2009.11.002](https://doi.org/10.1016/j.cpc.2009.11.002)
- M. Antuono, A. Colagrossi, S. Marrone.
  "Numerical Diffusive Terms in Weakly-Compressible SPH Schemes."
  In: Computer Physics Communications 183.12 (2012), pages 2570--2580.
  [doi: 10.1016/j.cpc.2012.07.006](https://doi.org/10.1016/j.cpc.2012.07.006)
- Diego Molteni, Andrea Colagrossi.
  "A Simple Procedure to Improve the Pressure Evaluation in Hydrodynamic Context Using the SPH."
  In: Computer Physics Communications 180.6 (2009), pages 861--872.
  [doi: 10.1016/j.cpc.2008.12.004](https://doi.org/10.1016/j.cpc.2008.12.004)
"""
struct DensityDiffusionAntuono{NDIMS, ELTYPE} <: DensityDiffusion
    delta                       :: ELTYPE
    correction_matrix           :: Array{ELTYPE, 3} # [i, j, particle]
    normalized_density_gradient :: Array{ELTYPE, 2} # [i, particle]

    function DensityDiffusionAntuono(delta, initial_condition)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)
        correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS,
                                             nparticles(initial_condition))

        normalized_density_gradient = Array{ELTYPE, 2}(undef, NDIMS,
                                                       nparticles(initial_condition))

        new{NDIMS, ELTYPE}(delta, correction_matrix, normalized_density_gradient)
    end
end

@inline Base.ndims(::DensityDiffusionAntuono{NDIMS}) where {NDIMS} = NDIMS

@inline function density_diffusion_psi(density_diffusion::DensityDiffusionAntuono,
                                       rho_a, rho_b,
                                       pos_diff, distance, system, particle, neighbor)
    (; normalized_density_gradient) = density_diffusion

    normalized_gradient_a = extract_svector(normalized_density_gradient, system, particle)
    normalized_gradient_b = extract_svector(normalized_density_gradient, system, neighbor)

    # Fist term by Molteni & Colagrossi
    result = 2 * (rho_a - rho_b)

    # Second correction term
    result -= dot(normalized_gradient_a + normalized_gradient_b, pos_diff)

    return result * pos_diff / distance^2
end

function update!(density_diffusion::DensityDiffusionAntuono, neighborhood_search,
                 v, u, system, semi)
    (; normalized_density_gradient) = density_diffusion

    # Compute correction matrix
    density_fun = @inline(particle->particle_density(v, system, particle))
    system_coords = current_coordinates(u, system)

    compute_gradient_correction_matrix!(density_diffusion.correction_matrix,
                                        neighborhood_search, system,
                                        system_coords, density_fun)

    # Compute normalized density gradient
    set_zero!(normalized_density_gradient)

    for_particle_neighbor(system, system, system_coords, system_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0
        distance < sqrt(eps()) && return

        rho_a = particle_density(v, system, particle)
        rho_b = particle_density(v, system, neighbor)

        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        L = correction_matrix(density_diffusion, particle)

        m_b = hydrodynamic_mass(system, neighbor)
        volume_b = m_b / rho_b

        normalized_gradient = -(rho_a - rho_b) * L * grad_kernel * volume_b

        for i in eachindex(normalized_gradient)
            normalized_density_gradient[i, particle] += normalized_gradient[i]
        end
    end

    return density_diffusion
end
