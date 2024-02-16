@doc raw"""
    BoundaryModelDummyParticles(initial_density, hydrodynamic_mass,
                                density_calculator, smoothing_kernel,
                                smoothing_length; viscosity=NoViscosity(),
                                state_equation=nothing, correction=nothing)

Boundaries modeled as dummy particles, which are treated like fluid particles,
but their positions and velocities are not evolved in time. Since the force towards the fluid
should not change with the material density when used with a [`TotalLagrangianSPHSystem`](@ref), the
dummy particles need to have a mass corresponding to the fluid's rest density, which we call
"hydrodynamic mass", as opposed to mass corresponding to the material density of a
[`TotalLagrangianSPHSystem`](@ref).

Here, `initial_density` and `hydrodynamic_mass` are vectors that contains the initial density
and the hydrodynamic mass respectively for each boundary particle.
Note that when used with [`SummationDensity`](@ref) (see below), this is only used to determine
the element type and the number of boundary particles.

To establish a relationship between density and pressure, a `state_equation` has to be passed,
which should be the same as for the adjacent fluid systems.
To sum over neighboring particles, a `smoothing_kernel` and `smoothing_length` needs to be passed.
This should be the same as for the adjacent fluid system with the largest smoothing length.

In the literature, this kind of boundary particles is referred to as
"dummy particles" (Adami et al., 2012 and Valizadeh & Monaghan, 2015),
"frozen fluid particles" (Akinci et al., 2012) or "dynamic boundaries (Crespo et al., 2007).
The key detail of this boundary condition and the only difference between the boundary models
in these references is the way the density and pressure of boundary particles is computed.

Since boundary particles are treated like fluid particles, the force
on fluid particle ``a`` due to boundary particle ``b`` is given by
```math
f_{ab} = m_a m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_{r_a} W(\Vert r_a - r_b \Vert, h).
```
The quantities to be defined here are the density ``\rho_b`` and pressure ``p_b``
of the boundary particle ``b``.

## Hydrodynamic density of dummy particles

We provide five options to compute the boundary density and pressure, determined by the `density_calculator`:
1. (Recommended) With [`AdamiPressureExtrapolation`](@ref), the pressure is extrapolated from the pressure of the
   fluid according to (Adami et al., 2012), and the density is obtained by applying the inverse of the state equation.
   This option usually yields the best results of the options listed here.
2. With [`SummationDensity`](@ref), the density is calculated by summation over the neighboring particles,
   and the pressure is computed from the density with the state equation.
3. With [`ContinuityDensity`](@ref), the density is integrated from the continuity equation,
   and the pressure is computed from the density with the state equation.
   Note that this causes a gap between fluid and boundary where the boundary is initialized
   without any contact to the fluid. This is due to overestimation of the boundary density
   as soon as the fluid comes in contact with boundary particles that initially did not have
   contact to the fluid.
   Therefore, in dam break simulations, there is a visible "step", even though the boundary is supposed to be flat.
   See also [dual.sphysics.org/faq/#Q_13](https://dual.sphysics.org/faq/#Q_13).
4. With [`PressureZeroing`](@ref), the density is set to the reference density and the pressure
   is computed from the density with the state equation.
   This option is not recommended. The other options yield significantly better results.
5. With [`PressureMirroring`](@ref), the density is set to the reference density. The pressure
   is not used. Instead, the fluid pressure is mirrored as boundary pressure in the
   momentum equation.
   This option is not recommended due to stability issues. See [`PressureMirroring`](@ref)
   for more details.

## No-slip conditions

For the interaction of dummy particles and fluid particles, Adami et al. (2012)
impose a no-slip boundary condition by assigning a wall velocity ``v_w`` to the dummy particle.

The wall velocity of particle ``a`` is calculated from the prescribed boundary particle
velocity ``v_a`` and the smoothed velocity field
```math
v_w = 2 v_a - \frac{\sum_b v_b W_{ab}}{\sum_b W_{ab}},
```
where the sum is over all fluid particles.

By choosing the viscosity model [`ViscosityAdami`](@ref) for `viscosity`, a no-slip
condition is imposed. It is recommended to choose `nu` in the order of either the kinematic
viscosity parameter of the adjacent fluid or the equivalent from the artificial parameter
`alpha` of the adjacent fluid (``\nu = \frac{\alpha h c }{2d + 4}``). When omitting the
viscous interaction (default `viscosity=NoViscosity()`), a free-slip wall boundary
condition is applied.

# Arguments
- `initial_density`: Vector holding the initial density of each boundary particle.
- `hydrodynamic_mass`: Vector holding the "hydrodynamic mass" of each boundary particle.
                       See description above for more information.
- `density_calculator`: Strategy to compute the hydrodynamic density of the boundary particles.
                        See description above for more information.
- `smoothing_kernel`: Smoothing kernel should be the same as for the adjacent fluid system.
- `smoothing_length`: Smoothing length should be the same as for the adjacent fluid system.

# Keywords
- `state_equation`: This should be the same as for the adjacent fluid system
                    (see e.g. [`StateEquationCole`](@ref)).
- `correction`:     Correction method of the adjacent fluid system (see [Corrections](@ref corrections)).
- `viscosity`:      Slip (default) or no-slip condition. See description above for further
                    information.

# Examples

```julia
# Free-slip condition
boundary_model = BoundaryModelDummyParticles(densities, masses, AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length))

# No-slip condition
boundary_model = BoundaryModelDummyParticles(densities, masses, AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length),
                                             viscosity=ViscosityAdami(nu))

```
## References:
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)
- Alireza Valizadeh, Joseph J. Monaghan.
  "A study of solid wall models for weakly compressible SPH".
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)
- Nadir Akinci, Markus Ihmsen, Gizem Akinci, Barbara Solenthaler, Matthias Teschner.
  "Versatile rigid-fluid coupling for incompressible SPH".
  ACM Transactions on Graphics 31, 4 (2012), pages 1–8.
  [doi: 10.1145/2185520.2185558](https://doi.org/10.1145/2185520.2185558)
- A. J. C. Crespo, M. Gómez-Gesteira, R. A. Dalrymple.
  "Boundary conditions generated by dynamic particles in SPH methods"
  In: Computers, Materials and Continua 5 (2007), pages 173-184.
  [doi: 10.3970/cmc.2007.005.173](https://doi.org/10.3970/cmc.2007.005.173)
"""
struct BoundaryModelDummyParticles{DC, ELTYPE <: Real, SE, K, V, COR, C}
    pressure           :: Vector{ELTYPE}
    hydrodynamic_mass  :: Vector{ELTYPE}
    state_equation     :: SE
    density_calculator :: DC
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    viscosity          :: V
    correction         :: COR
    cache              :: C

    function BoundaryModelDummyParticles(initial_density, hydrodynamic_mass,
                                         density_calculator, smoothing_kernel,
                                         smoothing_length; viscosity=NoViscosity(),
                                         state_equation=nothing, correction=nothing)
        pressure = initial_boundary_pressure(initial_density, density_calculator,
                                             state_equation)
        NDIMS = ndims(smoothing_kernel)

        n_particles = length(initial_density)

        cache = (; create_cache_model(viscosity, n_particles, NDIMS)...,
                 create_cache_model(initial_density, density_calculator)...)
        cache = (;
                 create_cache_model(correction, initial_density, NDIMS,
                                    n_particles)..., cache...)

        new{typeof(density_calculator), eltype(initial_density),
            typeof(state_equation), typeof(smoothing_kernel), typeof(viscosity),
            typeof(correction), typeof(cache)}(pressure, hydrodynamic_mass, state_equation,
                                               density_calculator,
                                               smoothing_kernel, smoothing_length,
                                               viscosity, correction, cache)
    end
end

@doc raw"""
    AdamiPressureExtrapolation()

The pressure of the boundary particles is obtained by extrapolating the pressure of the fluid
according to (Adami et al., 2012).
The pressure of a boundary particle ``b`` is given by
```math
p_b = \frac{\sum_f (p_f + \rho_f (\bm{g} - \bm{a}_b) \cdot \bm{r}_{bf}) W(\Vert r_{bf} \Vert, h)}{\sum_f W(\Vert r_{bf} \Vert, h)},
```
where the sum is over all fluid particles, ``\rho_f`` and ``p_f`` denote the density and pressure of fluid particle ``f``, respectively,
``r_{bf} = r_b - r_f`` denotes the difference of the coordinates of particles ``b`` and ``f``,
``\bm{g}`` denotes the gravitational acceleration acting on the fluid, and ``\bm{a}_b`` denotes the acceleration of the boundary particle ``b``.

## References:
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)
"""
struct AdamiPressureExtrapolation end

@doc raw"""
    PressureMirroring()

Instead of calculating density and pressure for each boundary particle, we modify the
momentum equation,
```math
\frac{\mathrm{d}v_a}{\mathrm{d}t} = -\sum_b m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_a W_{ab}
```
to replace the unknown density $\rho_b$ if $b$ is a boundary particle by the reference density
and the unknown pressure $p_b$ if $b$ is a boundary particle by the pressure $p_a$ of the
interacting fluid particle.
The momentum equation therefore becomes
```math
\frac{\mathrm{d}v_a}{\mathrm{d}t} = -\sum_f m_f \left( \frac{p_a}{\rho_a^2} + \frac{p_f}{\rho_f^2} \right) \nabla_a W_{af}
-\sum_b m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_a}{\rho_0^2} \right) \nabla_a W_{ab},
```
where the first sum is over all fluid particles and the second over all boundary particles.

This approach was first mentioned by Akinci et al. (2012) and written down in this form
by Band et al. (2018).

!!! note
    This boundary model requires high viscosity for stability with WCSPH.
    It also produces significantly worse results than [`AdamiPressureExtrapolation`](@ref)
    and is not more efficient because smaller time steps are required due to more noise
    in the pressure.
    We added this model only for research purposes and for comparison with
    [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH).

## References:
- Nadir Akinci, Markus Ihmsen, Gizem Akinci, Barbara Solenthaler, and Matthias Teschner.
  "Versatile Rigid-Fluid Coupling for Incompressible SPH."
  In: ACM Transactions on Graphics 31, 4 (2012), pages 1–8.
  [doi: 10.1145/2185520.2185558](https://doi.org/10.1145/2185520.2185558)
- Stefan Band, Christoph Gissler, Andreas Peer, and Matthias Teschner.
  "MLS Pressure Boundaries for Divergence-Free and Viscous SPH Fluids."
  In: Computers & Graphics 76 (2018), pages 37–46.
  [doi: 10.1016/j.cag.2018.08.001](https://doi.org/10.1016/j.cag.2018.08.001)
"""
struct PressureMirroring end

@doc raw"""
    PressureZeroing()

This is the simplest way to implement dummy boundary particles.
The density of each particle is set to the reference density and the pressure to the
reference pressure (the corresponding pressure to the reference density by the state equation).

!!! note
    This boundary model produces significantly worse results than all other models and
    is only included for research purposes.
"""
struct PressureZeroing end

create_cache_model(correction, density, NDIMS, nparticles) = (;)

function create_cache_model(::ShepardKernelCorrection, density, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density))
end

function create_cache_model(::KernelCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{Float64}(undef, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density), dw_gamma)
end

function create_cache_model(::Union{GradientCorrection, BlendedGradientCorrection}, density,
                            NDIMS, n_particles)
    correction_matrix = Array{Float64, 3}(undef, NDIMS, NDIMS, n_particles)
    return (; correction_matrix)
end

function create_cache_model(::MixedKernelGradientCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{Float64}(undef, NDIMS, n_particles)
    correction_matrix = Array{Float64, 3}(undef, NDIMS, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density), dw_gamma, correction_matrix)
end

function create_cache_model(initial_density,
                            ::Union{SummationDensity, PressureMirroring, PressureZeroing})
    density = copy(initial_density)

    return (; density)
end

function create_cache_model(initial_density, ::ContinuityDensity)
    return (; initial_density)
end

function create_cache_model(initial_density, ::AdamiPressureExtrapolation)
    density = copy(initial_density)
    volume = similar(initial_density)

    return (; density, volume)
end

function create_cache_model(viscosity::NoViscosity, n_particles, n_dims)
    return (;)
end

function create_cache_model(viscosity::ViscosityAdami, n_particles, n_dims)
    ELTYPE = eltype(viscosity.nu)

    wall_velocity = zeros(ELTYPE, n_dims, n_particles)

    return (; wall_velocity)
end

@inline reset_cache!(cache, viscosity) = set_zero!(cache.volume)

function reset_cache!(cache, viscosity::ViscosityAdami)
    (; volume, wall_velocity) = cache

    set_zero!(volume)
    set_zero!(wall_velocity)

    return cache
end

function Base.show(io::IO, model::BoundaryModelDummyParticles)
    @nospecialize model # reduce precompilation time

    print(io, "BoundaryModelDummyParticles(")
    print(io, model.density_calculator |> typeof |> nameof)
    print(io, ", ")
    print(io, model.viscosity |> typeof |> nameof)
    print(io, ")")
end

# For most density calculators, the pressure is updated in every step
initial_boundary_pressure(initial_density, density_calculator, _) = similar(initial_density)
# Pressure mirroring does not use the pressure, so we set it to zero for the visualization
initial_boundary_pressure(initial_density, ::PressureMirroring, _) = zero(initial_density)

# For pressure zeroing, set the pressure to the reference pressure (zero with free surfaces)
function initial_boundary_pressure(initial_density, ::PressureZeroing, state_equation)
    return state_equation.(initial_density)
end

# With EDAC, just use zero pressure
function initial_boundary_pressure(initial_density, ::PressureZeroing, ::Nothing)
    return zero(initial_density)
end

@inline function particle_density(v, model::BoundaryModelDummyParticles, system, particle)
    return particle_density(v, model.density_calculator, model, particle)
end

# Note that the other density calculators are dispatched in `density_calculators.jl`
@inline function particle_density(v,
                                  ::Union{AdamiPressureExtrapolation, PressureMirroring,
                                          PressureZeroing},
                                  boundary_model, particle)
    (; cache) = boundary_model

    return cache.density[particle]
end

@inline function particle_pressure(v, model::BoundaryModelDummyParticles, system, particle)
    return model.pressure[particle]
end

@inline function update_density!(boundary_model::BoundaryModelDummyParticles,
                                 system, v, u, v_ode, u_ode, semi)
    (; density_calculator) = boundary_model

    compute_density!(boundary_model, density_calculator, system, v, u, v_ode, u_ode, semi)

    return boundary_model
end

function compute_density!(boundary_model,
                          ::Union{ContinuityDensity, AdamiPressureExtrapolation,
                                  PressureMirroring, PressureZeroing},
                          system, v, u, v_ode, u_ode, semi)
    # No density update for `ContinuityDensity`, `PressureMirroring` and `PressureZeroing`.
    # For `AdamiPressureExtrapolation`, the density is updated in `compute_pressure!`.
    return boundary_model
end

@inline function update_pressure!(boundary_model::BoundaryModelDummyParticles,
                                  system, v, u, v_ode, u_ode, semi)
    (; density_calculator, correction) = boundary_model

    compute_correction_values!(system,
                               correction, v, u, v_ode, u_ode, semi, density_calculator)

    compute_gradient_correction_matrix!(correction, boundary_model, system, u, v_ode, u_ode,
                                        semi)

    # `kernel_correct_density!` only performed for `SummationDensity`
    kernel_correct_density!(boundary_model, v, u, v_ode, u_ode, semi, correction,
                            density_calculator)

    compute_pressure!(boundary_model, density_calculator, system, v, u, v_ode, u_ode, semi)

    return boundary_model
end

function kernel_correct_density!(boundary_model, v, u, v_ode, u_ode, semi,
                                 correction, density_calculator)
    return boundary_model
end

function kernel_correct_density!(boundary_model, v, u, v_ode, u_ode, semi,
                                 corr::ShepardKernelCorrection, ::SummationDensity)
    boundary_model.cache.density ./= boundary_model.cache.kernel_correction_coefficient
end

function compute_gradient_correction_matrix!(correction, boundary_model, system, u,
                                             v_ode, u_ode, semi)
    return system
end

function compute_gradient_correction_matrix!(corr::Union{GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection},
                                             boundary_model,
                                             system, u, v_ode, u_ode, semi)
    (; cache, correction, smoothing_kernel, smoothing_length) = boundary_model
    (; correction_matrix) = cache

    system_coords = current_coordinates(u, system)

    compute_gradient_correction_matrix!(correction_matrix, system, system_coords,
                                        v_ode, u_ode, semi, correction, smoothing_length,
                                        smoothing_kernel)
end

function compute_density!(boundary_model, ::SummationDensity, system, v, u, v_ode, u_ode,
                          semi)
    (; cache) = boundary_model
    (; density) = cache # Density is in the cache for SummationDensity

    summation_density!(system, semi, u, u_ode, density, particles=eachparticle(system))
end

function compute_pressure!(boundary_model, ::Union{SummationDensity, ContinuityDensity},
                           system, v, u, v_ode, u_ode, semi)

    # Limit pressure to be non-negative to avoid attractive forces between fluid and
    # boundary particles at free surfaces (sticking artifacts).
    @threaded for particle in eachparticle(system)
        apply_state_equation!(boundary_model, particle_density(v, boundary_model,
                                                               particle), particle)
    end

    return boundary_model
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function apply_state_equation!(boundary_model, density, particle)
    boundary_model.pressure[particle] = max(boundary_model.state_equation(density), 0.0)
end

function compute_pressure!(boundary_model, ::AdamiPressureExtrapolation,
                           system, v, u, v_ode, u_ode, semi)
    (; pressure, state_equation, cache, viscosity) = boundary_model
    (; volume, density) = cache

    set_zero!(pressure)

    # Set `volume` to zero. For `ViscosityAdami` the `wall_velocity` is also set to zero.
    reset_cache!(cache, viscosity)

    system_coords = current_coordinates(u, system)

    # Use all other systems for the pressure extrapolation
    @trixi_timeit timer() "compute boundary pressure" foreach_system(semi) do neighbor_system
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

        nhs = get_neighborhood_search(system, neighbor_system, semi)

        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        adami_pressure_extrapolation!(boundary_model, system, neighbor_system,
                                      system_coords, neighbor_coords,
                                      v_neighbor_system, nhs)
    end

    @trixi_timeit timer() "inverse state equation" @threaded for particle in eachparticle(system)
        compute_adami_density!(boundary_model, system, system_coords, particle)
    end
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
function compute_adami_density!(boundary_model, system, system_coords, particle)
    (; pressure, state_equation, cache, viscosity) = boundary_model
    (; volume, density) = cache

    # The summation is only over fluid particles, thus the volume stays zero when a boundary
    # particle isn't surrounded by fluid particles.
    # Check the volume to avoid NaNs in pressure and velocity.
    if volume[particle] > eps()
        pressure[particle] /= volume[particle]

        # To impose no-slip condition
        compute_wall_velocity!(viscosity, system, system_coords, particle)
    end

    # Apply inverse state equation to compute density (not used with EDAC)
    inverse_state_equation!(density, state_equation, pressure, particle)
end

function compute_pressure!(boundary_model, ::Union{PressureMirroring, PressureZeroing},
                           system, v, u, v_ode, u_ode, semi)
    # No pressure update needed with `PressureMirroring` and `PressureZeroing`.
    return boundary_model
end

@inline function adami_pressure_extrapolation!(boundary_model, system,
                                               neighbor_system::FluidSystem,
                                               system_coords, neighbor_coords,
                                               v_neighbor_system, neighborhood_search)
    (; pressure, cache, viscosity) = boundary_model

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search;
                          particles=eachparticle(system)) do particle, neighbor,
                                                             pos_diff, distance
        density_neighbor = particle_density(v_neighbor_system, neighbor_system, neighbor)

        resulting_acc = neighbor_system.acceleration -
                        current_acceleration(system, particle)

        kernel_weight = smoothing_kernel(boundary_model, distance)

        pressure[particle] += (particle_pressure(v_neighbor_system, neighbor_system,
                                                 neighbor) +
                               dot(resulting_acc, density_neighbor * pos_diff)) *
                              kernel_weight

        cache.volume[particle] += kernel_weight

        compute_smoothed_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                                   kernel_weight, particle, neighbor)
    end

    for particle in eachparticle(system)
        # Limit pressure to be non-negative to avoid attractive forces between fluid and
        # boundary particles at free surfaces (sticking artifacts).
        pressure[particle] = max(pressure[particle], 0.0)
    end
end

@inline function adami_pressure_extrapolation!(boundary_model, system, neighbor_system,
                                               system_coords, neighbor_coords,
                                               v_neighbor_system, neighborhood_search)
    return boundary_model
end

function compute_smoothed_velocity!(cache, viscosity, neighbor_system, v_neighbor_system,
                                    kernel_weight, particle, neighbor)
    return cache
end

function compute_smoothed_velocity!(cache, viscosity::ViscosityAdami,
                                    neighbor_system, v_neighbor_system, kernel_weight,
                                    particle, neighbor)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

    for dim in 1:ndims(neighbor_system)
        cache.wall_velocity[dim, particle] += kernel_weight * v_b[dim]
    end

    return cache
end

@inline function compute_wall_velocity!(viscosity, system, system_coords, particle)
    return viscosity
end

@inline function compute_wall_velocity!(viscosity::ViscosityAdami, system,
                                        system_coords, particle)
    (; boundary_model) = system
    (; cache) = boundary_model
    (; volume, wall_velocity) = cache

    # Prescribed velocity of the boundary particle.
    # This velocity is zero when not using moving boundaries.
    v_boundary = current_velocity(system_coords, system, particle)

    for dim in 1:ndims(system)
        # The second term is the precalculated smoothed velocity field of the fluid.
        wall_velocity[dim, particle] = 2 * v_boundary[dim] -
                                       wall_velocity[dim, particle] / volume[particle]
    end
    return viscosity
end

@inline function inverse_state_equation!(density, state_equation, pressure, particle)
    density[particle] = inverse_state_equation(state_equation, pressure[particle])
    return density
end

@inline function inverse_state_equation!(density, state_equation::Nothing, pressure,
                                         particle)
    # The density is constant when using EDAC
    return density
end

@inline function smoothing_kernel_grad(system::BoundarySystem, pos_diff,
                                       distance, particle)
    (; smoothing_kernel, smoothing_length, correction) = system.boundary_model

    return corrected_kernel_grad(smoothing_kernel, pos_diff, distance,
                                 smoothing_length, correction, system, particle)
end

@inline function correction_matrix(system::BoundarySystem, particle)
    extract_smatrix(system.boundary_model.cache.correction_matrix, system, particle)
end
