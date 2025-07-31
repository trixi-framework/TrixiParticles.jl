@doc raw"""
    TotalLagrangianSPHSystem(initial_condition,
                             smoothing_kernel, smoothing_length,
                             young_modulus, poisson_ratio;
                             n_fixed_particles=0, boundary_model=nothing,
                             acceleration=ntuple(_ -> 0.0, NDIMS),
                             penalty_force=nothing, source_terms=nothing)

System for particles of an elastic structure.

A Total Lagrangian framework is used wherein the governing equations are formulated such that
all relevant quantities and operators are measured with respect to the
initial configuration (O’Connor & Rogers 2021, Belytschko et al. 2000).
See [Total Lagrangian SPH](@ref tlsph) for more details on the method.

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.
- `young_modulus`:      Young's modulus.
- `poisson_ratio`:      Poisson ratio.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).

# Keyword Arguments
- `n_fixed_particles`:  Number of fixed particles which are used to clamp the structure
                        particles. Note that the fixed particles must be the **last**
                        particles in the `InitialCondition`. See the info box below.
- `boundary_model`: Boundary model to compute the hydrodynamic density and pressure for
                    fluid-structure interaction (see [Boundary Models](@ref boundary_models)).
- `penalty_force`:  Penalty force to ensure regular particle position under large deformations
                    (see [`PenaltyForceGanzenmueller`](@ref)).
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).

!!! note
    The fixed particles must be the **last** particles in the `InitialCondition`.
    To do so, e.g. use the `union` function:
    ```jldoctest; output = false, setup = :(fixed_particles = RectangularShape(0.1, (1, 4), (0.0, 0.0), density=1.0); beam = RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0))
    solid = union(beam, fixed_particles)

    # output
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ InitialCondition{Float64}                                                                        │
    │ ═════════════════════════                                                                        │
    │ #dimensions: ……………………………………………… 2                                                                │
    │ #particles: ………………………………………………… 16                                                               │
    │ particle spacing: ………………………………… 0.1                                                              │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    where `beam` and `fixed_particles` are of type `InitialCondition`.
"""
struct TotalLagrangianSPHSystem{BM, NDIMS, ELTYPE <: Real, IC, ARRAY1D, ARRAY2D, ARRAY3D,
                                YM, PR, LL, LM, K, PF, ST} <: SolidSystem{NDIMS}
    initial_condition   :: IC
    initial_coordinates :: ARRAY2D # Array{ELTYPE, 2}: [dimension, particle]
    current_coordinates :: ARRAY2D # Array{ELTYPE, 2}: [dimension, particle]
    mass                :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    correction_matrix   :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    pk1_corrected       :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    deformation_grad    :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    material_density    :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    n_moving_particles  :: Int64
    young_modulus       :: YM
    poisson_ratio       :: PR
    lame_lambda         :: LL
    lame_mu             :: LM
    smoothing_kernel    :: K
    smoothing_length    :: ELTYPE
    acceleration        :: SVector{NDIMS, ELTYPE}
    boundary_model      :: BM
    penalty_force       :: PF
    source_terms        :: ST
    buffer              :: Nothing
end

function TotalLagrangianSPHSystem(initial_condition,
                                  smoothing_kernel, smoothing_length,
                                  young_modulus, poisson_ratio;
                                  n_fixed_particles=0, boundary_model=nothing,
                                  acceleration=ntuple(_ -> 0.0,
                                                      ndims(smoothing_kernel)),
                                  penalty_force=nothing, source_terms=nothing)
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

    lame_lambda = @. young_modulus * poisson_ratio /
                     ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    lame_mu = @. (young_modulus / 2) / (1 + poisson_ratio)

    return TotalLagrangianSPHSystem(initial_condition, initial_coordinates,
                                    current_coordinates, mass, correction_matrix,
                                    pk1_corrected, deformation_grad, material_density,
                                    n_moving_particles, young_modulus, poisson_ratio,
                                    lame_lambda, lame_mu, smoothing_kernel,
                                    smoothing_length, acceleration_, boundary_model,
                                    penalty_force, source_terms, nothing)
end

function Base.show(io::IO, system::TotalLagrangianSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "TotalLagrangianSPHSystem{", ndims(system), "}(")
    print(io, "", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.boundary_model)
    print(io, ", ", system.penalty_force)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::TotalLagrangianSPHSystem)
    @nospecialize system # reduce precompilation time

    function display_param(param)
        if param isa AbstractVector
            min_val = round(minimum(param), digits=3)
            max_val = round(maximum(param), digits=3)
            return "min = $(min_val), max = $(max_val)"
        else
            return string(param)
        end
    end

    if get(io, :compact, false)
        show(io, system)
    else
        n_fixed_particles = nparticles(system) - n_moving_particles(system)

        summary_header(io, "TotalLagrangianSPHSystem{$(ndims(system))}")
        summary_line(io, "total #particles", nparticles(system))
        summary_line(io, "#fixed particles", n_fixed_particles)
        summary_line(io, "Young's modulus", display_param(system.young_modulus))
        summary_line(io, "Poisson ratio", display_param(system.poisson_ratio))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "penalty force", system.penalty_force |> typeof |> nameof)
        summary_footer(io)
    end
end

@inline function Base.eltype(::TotalLagrangianSPHSystem{<:Any, <:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

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
        return zero(SVector{ndims(system), eltype(system)})
    end

    return extract_svector(v, system, particle)
end

@inline function viscous_velocity(v, system::TotalLagrangianSPHSystem, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@inline function current_density(v, system::TotalLagrangianSPHSystem)
    return current_density(v, system.boundary_model, system)
end

# In fluid-solid interaction, use the "hydrodynamic pressure" of the solid particles
# corresponding to the chosen boundary model.
@inline function current_pressure(v, system::TotalLagrangianSPHSystem)
    return current_pressure(v, system.boundary_model, system)
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

function initialize!(system::TotalLagrangianSPHSystem, semi)
    (; correction_matrix) = system

    initial_coords = initial_coordinates(system)

    density_fun(particle) = system.material_density[particle]

    # Calculate correction matrix
    compute_gradient_correction_matrix!(correction_matrix, system, initial_coords,
                                        density_fun, semi)
end

function update_positions!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; current_coordinates) = system

    # `current_coordinates` stores the coordinates of both integrated and fixed particles.
    # Copy the coordinates of the integrated particles from `u`.
    @threaded semi for particle in each_moving_particle(system)
        for i in 1:ndims(system)
            current_coordinates[i, particle] = u[i, particle]
        end
    end
end

function update_quantities!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t)
    # Precompute PK1 stress tensor
    @trixi_timeit timer() "stress tensor" compute_pk1_corrected!(system, semi)

    return system
end

function update_boundary_interpolation!(system::TotalLagrangianSPHSystem, v, u,
                                        v_ode, u_ode, semi, t; update_from_callback=false)
    (; boundary_model) = system

    # Only update boundary model
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)
end

@inline function compute_pk1_corrected!(system, semi)
    (; deformation_grad) = system

    calc_deformation_grad!(deformation_grad, system, semi)

    @threaded semi for particle in eachparticle(system)
        pk1_particle = pk1_stress_tensor(system, particle)
        pk1_particle_corrected = pk1_particle * correction_matrix(system, particle)

        @inbounds for j in 1:ndims(system), i in 1:ndims(system)
            system.pk1_corrected[i, j, particle] = pk1_particle_corrected[i, j]
        end
    end
end

@inline function calc_deformation_grad!(deformation_grad, system, semi)
    (; mass, material_density) = system

    # Reset deformation gradient
    set_zero!(deformation_grad)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    initial_coords = initial_coordinates(system)
    foreach_point_neighbor(system, system, initial_coords, initial_coords,
                           semi) do particle, neighbor, initial_pos_diff, initial_distance
        # Only consider particles with a distance > 0.
        initial_distance < sqrt(eps()) && return

        volume = mass[neighbor] / material_density[neighbor]
        pos_diff = current_coords(system, particle) - current_coords(system, neighbor)

        grad_kernel = smoothing_kernel_grad(system, initial_pos_diff,
                                            initial_distance, particle)

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
@inline function pk1_stress_tensor(system, particle)
    (; lame_lambda, lame_mu) = system

    F = deformation_gradient(system, particle)
    S = pk2_stress_tensor(F, lame_lambda, lame_mu, particle)

    return F * S
end

# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(F, lame_lambda::AbstractVector, lame_mu::AbstractVector,
                                   particle)

    # Compute the Green-Lagrange strain
    E = (transpose(F) * F - I) / 2

    return lame_lambda[particle] * tr(E) * I + 2 * lame_mu[particle] * E
end

# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(F, lame_lambda, lame_mu, particle)

    # Compute the Green-Lagrange strain
    E = (transpose(F) * F - I) / 2

    return lame_lambda * tr(E) * I + 2 * lame_mu * E
end

@inline function calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                                     initial_distance, system, m_a, m_b, rho_a, rho_b,
                                     ::Nothing)
    return dv
end

function write_u0!(u0, system::TotalLagrangianSPHSystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices((ndims(system), each_moving_particle(system)))
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

function write_v0!(v0, system::TotalLagrangianSPHSystem)
    (; initial_condition, boundary_model) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices((ndims(system), each_moving_particle(system)))
    copyto!(v0, indices, initial_condition.velocity, indices)

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

# An explanation of these equation can be found in
# J. Lubliner, 2008. Plasticity theory.
# See here below Equation 5.3.21 for the equation for the equivalent stress.
# The von-Mises stress is one form of equivalent stress, where sigma is the deviatoric stress.
# See pages 32 and 123.
function von_mises_stress(system)
    von_mises_stress_vector = zeros(eltype(system.pk1_corrected), nparticles(system))

    @threaded default_backend(von_mises_stress_vector) for particle in
                                                           each_moving_particle(system)
        von_mises_stress_vector[particle] = von_mises_stress(system, particle)
    end

    return von_mises_stress_vector
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function von_mises_stress(system, particle::Integer)
    F = deformation_gradient(system, particle)
    J = det(F)
    P = pk1_corrected(system, particle)
    sigma = (1.0 / J) * P * F'

    # Calculate deviatoric stress tensor
    s = sigma - (1.0 / 3.0) * tr(sigma) * I

    return sqrt(3.0 / 2.0 * sum(s .^ 2))
end

# An explanation of these equation can be found in
# J. Lubliner, 2008. Plasticity theory.
# See here page 473 for the relation between the `pk1`, the first Piola-Kirchhoff tensor,
# and the Cauchy stress.
function cauchy_stress(system::TotalLagrangianSPHSystem)
    NDIMS = ndims(system)

    cauchy_stress_tensors = zeros(eltype(system.pk1_corrected), NDIMS, NDIMS,
                                  nparticles(system))

    @threaded default_backend(cauchy_stress_tensors) for particle in
                                                         each_moving_particle(system)
        F = deformation_gradient(system, particle)
        J = det(F)
        P = pk1_corrected(system, particle)
        sigma = (1.0 / J) * P * F'
        cauchy_stress_tensors[:, :, particle] = sigma
    end

    return cauchy_stress_tensors
end

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline function viscosity_model(system::TotalLagrangianSPHSystem, neighbor_system)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::FluidSystem,
                                 neighbor_system::TotalLagrangianSPHSystem)
    return neighbor_system.boundary_model.viscosity
end

function system_data(system::TotalLagrangianSPHSystem, v_ode, u_ode, semi)
    (; mass, material_density, deformation_grad, pk1_corrected, young_modulus,
     poisson_ratio, lame_lambda, lame_mu) = system

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    initial_coordinates_ = initial_coordinates(system)
    velocity = current_velocity(v, system)

    return (; coordinates, initial_coordinates=initial_coordinates_, velocity, mass,
            material_density, deformation_grad, pk1_corrected, young_modulus, poisson_ratio,
            lame_lambda, lame_mu)
end

function available_data(::TotalLagrangianSPHSystem)
    return (:coordinates, :initial_coordinates, :velocity, :mass, :material_density,
            :deformation_grad, :pk1_corrected, :young_modulus, :poisson_ratio,
            :lame_lambda, :lame_mu)
end
