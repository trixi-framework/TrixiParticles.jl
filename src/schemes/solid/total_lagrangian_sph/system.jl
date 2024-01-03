@doc raw"""
    TotalLagrangianSPHSystem(initial_condition,
                             smoothing_kernel, smoothing_length,
                             young_modulus, poisson_ratio, boundary_model;
                             n_fixed_particles=0,
                             acceleration=ntuple(_ -> 0.0, NDIMS),
                             penalty_force=nothing)

System for particles of an elastic solid.

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
with the correction matrix (see also [`GradientCorrection`](@ref))
```math
\bm{L}_{0a} := \left( -\sum_{b} \frac{m_{0b}}{\rho_{0b}} \nabla_{0a} W(\bm{X}_{ab}) \bm{X}_{ab}^T \right)^{-1} \in \R^{d \times d}.
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
struct TotalLagrangianSPHSystem{BM, NDIMS, ELTYPE <: Real, K, PF} <: SolidSystem{NDIMS}
    initial_condition   :: InitialCondition{ELTYPE}
    initial_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    current_coordinates :: Array{ELTYPE, 2} # [dimension, particle]
    mass                :: Array{ELTYPE, 1} # [particle]
    correction_matrix   :: Array{ELTYPE, 3} # [i, j, particle]
    pk1_corrected       :: Array{ELTYPE, 3} # [i, j, particle]
    deformation_grad    :: Array{ELTYPE, 3} # [i, j, particle]
    material_density    :: Array{ELTYPE, 1} # [particle]
    n_moving_particles  :: Int64
    young_modulus       :: ELTYPE
    poisson_ratio       :: ELTYPE
    lame_lambda         :: ELTYPE
    lame_mu             :: ELTYPE
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    acceleration        :: SVector{NDIMS, ELTYPE}
    boundary_model      :: BM
    penalty_force       :: PF
    function TotalLagrangianSPHSystem(initial_condition,
                                      smoothing_kernel, smoothing_length,
                                      young_modulus, poisson_ratio, boundary_model;
                                      n_fixed_particles=0,
                                      acceleration=ntuple(_ -> 0.0,
                                                          ndims(smoothing_kernel)),
                                      penalty_force=nothing)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)
        n_particles = nparticles(initial_condition)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        # Make acceleration an SVector
        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        initial_coordinates = copy(initial_condition.coordinates)
        current_coordinates = copy(initial_condition.coordinates)
        mass = copy(initial_condition.mass)
        material_density = copy(initial_condition.density)
        correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)
        pk1_corrected = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)
        deformation_grad = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)

        n_moving_particles = n_particles - n_fixed_particles

        lame_lambda = young_modulus * poisson_ratio /
                      ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        lame_mu = 0.5 * young_modulus / (1 + poisson_ratio)

        return new{typeof(boundary_model),
                   NDIMS, ELTYPE,
                   typeof(smoothing_kernel),
                   typeof(penalty_force)}(initial_condition,
                                                              initial_coordinates,
                                                              current_coordinates, mass,
                                                              correction_matrix,
                                                              pk1_corrected,
                                                              deformation_grad,
                                                              material_density,
                                                              n_moving_particles,
                                                              young_modulus, poisson_ratio,
                                                              lame_lambda, lame_mu,
                                                              smoothing_kernel,
                                                              smoothing_length,
                                                              acceleration_, boundary_model,
                                                              penalty_force)
    end
end

function Base.show(io::IO, system::TotalLagrangianSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "TotalLagrangianSPHSystem{", ndims(system), "}(")
    print(io, system.young_modulus)
    print(io, ", ", system.poisson_ratio)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.boundary_model)
    print(io, ", ", system.penalty_force)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::TotalLagrangianSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        n_fixed_particles = nparticles(system) - n_moving_particles(system)

        summary_header(io, "TotalLagrangianSPHSystem{$(ndims(system))}")
        summary_line(io, "total #particles", nparticles(system))
        summary_line(io, "#fixed particles", n_fixed_particles)
        summary_line(io, "Young's modulus", system.young_modulus)
        summary_line(io, "Poisson ratio", system.poisson_ratio)
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "penalty force", system.penalty_force |> typeof |> nameof)
        summary_footer(io)
    end
end

timer_name(::TotalLagrangianSPHSystem) = "solid"

@inline function v_nvariables(system::TotalLagrangianSPHSystem)
    return ndims(system)
end

@inline function v_nvariables(system::TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}})
    return ndims(system) + 1
end

@inline function n_moving_particles(system::TotalLagrangianSPHSystem)
    system.n_moving_particles
end

@inline initial_coordinates(system::TotalLagrangianSPHSystem) = system.initial_coordinates

@inline function current_coordinates(u, system::TotalLagrangianSPHSystem)
    return system.current_coordinates
end

@inline function current_coords(system::TotalLagrangianSPHSystem, particle)
    # For this system, the current coordinates are stored in the system directly,
    # so we don't need a `u` array. This function is only to be used in this file
    # when no `u` is available.
    current_coords(nothing, system, particle)
end

@inline function current_velocity(v, system::TotalLagrangianSPHSystem, particle)
    if particle > n_moving_particles(system)
        return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
    end

    return extract_svector(v, system, particle)
end

@inline function viscous_velocity(v, system::TotalLagrangianSPHSystem, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@inline function particle_density(v, system::TotalLagrangianSPHSystem, particle)
    return particle_density(v, system.boundary_model, system, particle)
end

# In fluid-solid interaction, use the "hydrodynamic pressure" of the solid particles
# corresponding to the chosen boundary model.
@inline function particle_pressure(v, system::TotalLagrangianSPHSystem, particle)
    return particle_pressure(v, system.boundary_model, system, particle)
end

@inline function hydrodynamic_mass(system::TotalLagrangianSPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@inline function correction_matrix(system, particle)
    extract_smatrix(system.correction_matrix, system, particle)
end

@inline function deformation_gradient(system, particle)
    extract_smatrix(system.deformation_grad, system, particle)
end
@inline function pk1_corrected(system, particle)
    extract_smatrix(system.pk1_corrected, system, particle)
end

function initialize!(system::TotalLagrangianSPHSystem, neighborhood_search)
    (; correction_matrix) = system

    initial_coords = initial_coordinates(system)

    density_fun(particle) = system.material_density[particle]

    # Calculate correction matrix
    compute_gradient_correction_matrix!(correction_matrix, neighborhood_search, system,
                                        initial_coords, density_fun)
end

function update_positions!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; current_coordinates) = system

    for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            current_coordinates[i, particle] = u[i, particle]
        end
    end
end

function update_quantities!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t)
    # Precompute PK1 stress tensor
    nhs = neighborhood_searches(system, system, semi)
    @trixi_timeit timer() "stress tensor" compute_pk1_corrected(nhs, system)

    return system
end

function update_final!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; boundary_model) = system

    # Only update boundary model
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)
end

@inline function compute_pk1_corrected(neighborhood_search, system)
    (; deformation_grad) = system

    calc_deformation_grad!(deformation_grad, neighborhood_search, system)

    @threaded for particle in eachparticle(system)
        J_particle = deformation_gradient(system, particle)
        pk1_particle = pk1_stress_tensor(J_particle, system)
        pk1_particle_corrected = pk1_particle * correction_matrix(system, particle)

        @inbounds for j in 1:ndims(system), i in 1:ndims(system)
            system.pk1_corrected[i, j, particle] = pk1_particle_corrected[i, j]
        end
    end
end

@inline function calc_deformation_grad!(deformation_grad, neighborhood_search, system)
    (; mass, material_density) = system

    # Reset deformation gradient
    set_zero!(deformation_grad)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    initial_coords = initial_coordinates(system)
    for_particle_neighbor(system, system,
                          initial_coords, initial_coords,
                          neighborhood_search;
                          particles=eachparticle(system)) do particle, neighbor,
                                                             initial_pos_diff,
                                                             initial_distance
        # Only consider particles with a distance > 0.
        initial_distance < sqrt(eps()) && return

        volume = mass[neighbor] / material_density[neighbor]
        pos_diff = current_coords(system, particle) - current_coords(system, neighbor)

        grad_kernel = smoothing_kernel_grad(system, initial_pos_diff,
                                            initial_distance)

        result = volume * pos_diff * grad_kernel'

        # Multiply by L_{0a}
        result *= correction_matrix(system, particle)'

        @inbounds for j in 1:ndims(system), i in 1:ndims(system)
            deformation_grad[i, j, particle] -= result[i, j]
        end
    end

    return deformation_grad
end

# First Piola-Kirchhoff stress tensor
@inline function pk1_stress_tensor(J, system)
    S = pk2_stress_tensor(J, system)

    return J * S
end

# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(J, system)
    (; lame_lambda, lame_mu) = system

    # Compute the Green-Lagrange strain
    E = 0.5 * (transpose(J) * J - I)

    return lame_lambda * tr(E) * I + 2 * lame_mu * E
end

@inline function calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                                     initial_distance, system, ::Nothing)
    return dv
end

function write_u0!(u0, system::TotalLagrangianSPHSystem)
    (; initial_condition) = system

    for particle in each_moving_particle(system)
        # Write particle coordinates
        for dim in 1:ndims(system)
            u0[dim, particle] = initial_condition.coordinates[dim, particle]
        end
    end

    return u0
end

function write_v0!(v0, system::TotalLagrangianSPHSystem)
    (; initial_condition, boundary_model) = system

    for particle in each_moving_particle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    write_v0!(v0, boundary_model, system)

    return v0
end

function write_v0!(v0, model, system::TotalLagrangianSPHSystem)
    return v0
end

function write_v0!(v0, ::BoundaryModelDummyParticles{ContinuityDensity},
                   system::TotalLagrangianSPHSystem)
    (; cache) = system.boundary_model
    (; initial_density) = cache

    for particle in each_moving_particle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::TotalLagrangianSPHSystem, v, u)
    for particle in each_moving_particle(system)
        system.current_coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
    end

    # This is dispatched in the boundary system.jl file
    restart_with!(system, system.boundary_model, v, u)
end

function viscosity_model(system::TotalLagrangianSPHSystem)
    return system.boundary_model.viscosity
end

@inline function pressure_acceleration(pressure_correction, m_b, p_a, p_b,
                                       rho_a, rho_b, pos_diff, distance, grad_kernel,
                                       particle_system, neighbor,
                                       neighbor_system::TotalLagrangianSPHSystem,
                                       density_calculator, correction)
    (; boundary_model) = neighbor_system
    (; smoothing_length) = particle_system

    # Pressure acceleration for fluid-solid interaction. This is identical to
    # `pressure_acceleration` for the `BoundarySPHSystem`.
    return pressure_acceleration_bnd(pressure_correction, m_b, p_a, p_b,
                                     rho_a, rho_b, pos_diff, distance,
                                     smoothing_length, grad_kernel,
                                     particle_system, neighbor, neighbor_system,                                      boundary_model,

                                     density_calculator, correction)
end
