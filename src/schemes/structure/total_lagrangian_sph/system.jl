@doc raw"""
    TotalLagrangianSPHSystem(initial_condition, smoothing_kernel, smoothing_length,
                             young_modulus, poisson_ratio;
                             n_clamped_particles=0,
                             clamped_particles=Int[],
                             clamped_particles_motion=nothing,
                             acceleration=ntuple(_ -> 0.0, NDIMS),
                             penalty_force=nothing, viscosity=nothing,
                             source_terms=nothing, boundary_model=nothing)

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

# Keywords
- `n_clamped_particles` (deprecated): Number of clamped particles that are fixed and not integrated
                         to clamp the structure. Note that the clamped particles must be the **last**
                         particles in the `InitialCondition`. See the info box below.
                         This keyword is deprecated and will be removed in a future release.
                         Instead pass `clamped_particles` with the explicit particle indices to be clamped.
- `clamped_particles`: Indices specifying the clamped particles that are fixed
                       and not integrated to clamp the structure.
- `clamped_particles_motion`: Prescribed motion of the clamped particles.
                    If `nothing` (default), the clamped particles are fixed.
                    See [`PrescribedMotion`](@ref) for details.
- `boundary_model`: Boundary model to compute the hydrodynamic density and pressure for
                    fluid-structure interaction (see [Boundary Models](@ref boundary_models)).
- `penalty_force`:  Penalty force to ensure regular particle position under large deformations
                    (see [`PenaltyForceGanzenmueller`](@ref)).
- `viscosity`:      Artificial viscosity model to stabilize both the TLSPH and the FSI.
                    Currently, only [`ArtificialViscosityMonaghan`](@ref) is supported.
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).

!!! note
    If specifying the clamped particles manually (via `n_clamped_particles`),
    the clamped particles must be the **last** particles in the `InitialCondition`.
    To do so, e.g. use the `union` function:
    ```jldoctest; output = false, setup = :(clamped_particles = RectangularShape(0.1, (1, 4), (0.0, 0.0), density=1.0); beam = RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0))
    structure = union(beam, clamped_particles)

    # output
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ InitialCondition                                                                                 │
    │ ════════════════                                                                                 │
    │ #dimensions: ……………………………………………… 2                                                                │
    │ #particles: ………………………………………………… 16                                                               │
    │ particle spacing: ………………………………… 0.1                                                              │
    │ eltype: …………………………………………………………… Float64                                                          │
    │ coordinate eltype: ……………………………… Float64                                                          │
    └──────────────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    where `beam` and `clamped_particles` are of type [`InitialCondition`](@ref).
"""
struct TotalLagrangianSPHSystem{BM, NDIMS, ELTYPE <: Real, IC, ARRAY1D, ARRAY2D, ARRAY3D,
                                YM, PR, LL, LM, K, PF, V, ST, M, IM,
                                C} <: AbstractStructureSystem{NDIMS}
    initial_condition   :: IC
    initial_coordinates :: ARRAY2D # Array{ELTYPE, 2}: [dimension, particle]
    # `current_coordinates` contains `u` plus coordinates of the fixed particles
    current_coordinates      :: ARRAY2D # Array{ELTYPE, 2}: [dimension, particle]
    mass                     :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    correction_matrix        :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    pk1_rho2                 :: ARRAY3D # PK1 corrected divided by rho^2: [i, j, particle]
    deformation_grad         :: ARRAY3D # Array{ELTYPE, 3}: [i, j, particle]
    material_density         :: ARRAY1D # Array{ELTYPE, 1}: [particle]
    n_integrated_particles   :: Int64
    young_modulus            :: YM
    poisson_ratio            :: PR
    lame_lambda              :: LL
    lame_mu                  :: LM
    smoothing_kernel         :: K
    smoothing_length         :: ELTYPE
    acceleration             :: SVector{NDIMS, ELTYPE}
    boundary_model           :: BM
    penalty_force            :: PF
    viscosity                :: V
    source_terms             :: ST
    clamped_particles_motion :: M
    clamped_particles_moving :: IM
    cache                    :: C
end

function TotalLagrangianSPHSystem(initial_condition, smoothing_kernel, smoothing_length,
                                  young_modulus, poisson_ratio;
                                  n_clamped_particles=0,
                                  clamped_particles=Int[],
                                  clamped_particles_motion=nothing,
                                  acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                                      ndims(smoothing_kernel)),
                                  penalty_force=nothing, viscosity=nothing,
                                  source_terms=nothing, boundary_model=nothing)
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

    # Backwards compatibility: `n_clamped_particles` is deprecated.
    # Emit a deprecation warning and (if the user didn't supply explicit indices)
    # convert the old `n_clamped_particles` convention to `clamped_particles`.
    if n_clamped_particles != 0
        Base.depwarn("keyword `n_clamped_particles` is deprecated and will be removed in a future release; " *
                     "pass `clamped_particles` (Vector{Int} of indices) instead.",
                     :n_clamped_particles)
        if isempty(clamped_particles)
            clamped_particles = collect((n_particles - n_clamped_particles + 1):n_particles)
        else
            throw(ArgumentError("Either `n_clamped_particles` or `clamped_particles` can be specified, not both."))
        end
    end

    # Handle clamped particles
    if !isempty(clamped_particles)
        @assert allunique(clamped_particles) "`clamped_particles` contains duplicate particle indices"

        n_clamped_particles = length(clamped_particles)
        initial_condition_sorted = deepcopy(initial_condition)
        young_modulus_sorted = copy(young_modulus)
        poisson_ratio_sorted = copy(poisson_ratio)
        move_particles_to_end!(initial_condition_sorted, clamped_particles)
        move_particles_to_end!(young_modulus_sorted, clamped_particles)
        move_particles_to_end!(poisson_ratio_sorted, clamped_particles)
    else
        initial_condition_sorted = initial_condition
        young_modulus_sorted = young_modulus
        poisson_ratio_sorted = poisson_ratio
    end

    initial_coordinates = copy(initial_condition_sorted.coordinates)
    current_coordinates = copy(initial_condition_sorted.coordinates)
    mass = copy(initial_condition_sorted.mass)
    material_density = copy(initial_condition_sorted.density)
    correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)
    pk1_rho2 = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)
    deformation_grad = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, n_particles)

    n_integrated_particles = n_particles - n_clamped_particles

    lame_lambda = @. young_modulus_sorted * poisson_ratio_sorted /
                     ((1 + poisson_ratio_sorted) *
                      (1 - 2 * poisson_ratio_sorted))
    lame_mu = @. (young_modulus_sorted / 2) / (1 + poisson_ratio_sorted)

    ismoving = Ref(!isnothing(clamped_particles_motion))
    initialize_prescribed_motion!(clamped_particles_motion, initial_condition_sorted,
                                  n_clamped_particles)

    cache = create_cache_tlsph(clamped_particles_motion, initial_condition_sorted)

    return TotalLagrangianSPHSystem(initial_condition_sorted, initial_coordinates,
                                    current_coordinates, mass, correction_matrix,
                                    pk1_rho2, deformation_grad, material_density,
                                    n_integrated_particles, young_modulus_sorted,
                                    poisson_ratio_sorted,
                                    lame_lambda, lame_mu, smoothing_kernel,
                                    smoothing_length, acceleration_, boundary_model,
                                    penalty_force, viscosity, source_terms,
                                    clamped_particles_motion, ismoving, cache)
end

create_cache_tlsph(::Nothing, initial_condition) = (;)

function create_cache_tlsph(::PrescribedMotion, initial_condition)
    velocity = zero(initial_condition.velocity)
    acceleration = zero(initial_condition.velocity)

    return (; velocity, acceleration)
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

@inline function n_integrated_particles(system::TotalLagrangianSPHSystem)
    system.n_integrated_particles
end

@inline initial_coordinates(system::TotalLagrangianSPHSystem) = system.initial_coordinates

@inline function current_coordinates(u, system::TotalLagrangianSPHSystem)
    return system.current_coordinates
end

@propagate_inbounds function current_coords(system::TotalLagrangianSPHSystem, particle)
    # For this system, the current coordinates are stored in the system directly,
    # so we don't need a `u` array. This function is only to be used in this file
    # when no `u` is available.
    current_coords(nothing, system, particle)
end

@propagate_inbounds function current_velocity(v, system::TotalLagrangianSPHSystem, particle)
    if particle <= system.n_integrated_particles
        return extract_svector(v, system, particle)
    end

    return current_clamped_velocity(v, system, system.clamped_particles_motion, particle)
end

@inline function current_clamped_velocity(v, system, prescribed_motion, particle)
    (; cache, clamped_particles_moving) = system

    if clamped_particles_moving[]
        return extract_svector(cache.velocity, system, particle)
    end

    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_clamped_velocity(v, system, prescribed_motion::Nothing, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@inline function current_velocity(v, system::TotalLagrangianSPHSystem)
    error("`current_velocity(v, system)` is not implemented for `TotalLagrangianSPHSystem`")
end

@propagate_inbounds function viscous_velocity(v, system::TotalLagrangianSPHSystem, particle)
    return extract_svector(system.boundary_model.cache.wall_velocity, system, particle)
end

@propagate_inbounds function current_density(v, system::TotalLagrangianSPHSystem)
    return current_density(v, system.boundary_model, system)
end

# In fluid-structure interaction, use the "hydrodynamic pressure" of the structure particles
# corresponding to the chosen boundary model.
@propagate_inbounds function current_pressure(v, system::TotalLagrangianSPHSystem)
    return current_pressure(v, system.boundary_model, system)
end

@propagate_inbounds function hydrodynamic_mass(system::TotalLagrangianSPHSystem, particle)
    return system.boundary_model.hydrodynamic_mass[particle]
end

@propagate_inbounds function correction_matrix(system, particle)
    extract_smatrix(system.correction_matrix, system, particle)
end

@propagate_inbounds function deformation_gradient(system, particle)
    extract_smatrix(system.deformation_grad, system, particle)
end
@propagate_inbounds function pk1_rho2(system, particle)
    extract_smatrix(system.pk1_rho2, system, particle)
end

@propagate_inbounds function young_modulus(system::TotalLagrangianSPHSystem, particle)
    return young_modulus(system, system.young_modulus, particle)
end

@inline function young_modulus(::TotalLagrangianSPHSystem, young_modulus, particle)
    return young_modulus
end

@propagate_inbounds function young_modulus(::TotalLagrangianSPHSystem,
                                           young_modulus::AbstractVector, particle)
    return young_modulus[particle]
end

@propagate_inbounds function poisson_ratio(system::TotalLagrangianSPHSystem, particle)
    return poisson_ratio(system, system.poisson_ratio, particle)
end

@inline function poisson_ratio(::TotalLagrangianSPHSystem, poisson_ratio, particle)
    return poisson_ratio
end

@inline function poisson_ratio(::TotalLagrangianSPHSystem,
                               poisson_ratio::AbstractVector, particle)
    return poisson_ratio[particle]
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
    (; current_coordinates, clamped_particles_motion) = system

    # `current_coordinates` stores the coordinates of both integrated and clamped particles.
    # Copy the coordinates of the integrated particles from `u`.
    @threaded semi for particle in each_integrated_particle(system)
        for i in 1:ndims(system)
            current_coordinates[i, particle] = u[i, particle]
        end
    end

    apply_prescribed_motion!(system, clamped_particles_motion, semi, t)
end

function apply_prescribed_motion!(system::TotalLagrangianSPHSystem,
                                  prescribed_motion::PrescribedMotion, semi, t)
    (; clamped_particles_moving, current_coordinates, cache) = system
    (; acceleration, velocity) = cache

    prescribed_motion(current_coordinates, velocity, acceleration, clamped_particles_moving,
                      system, semi, t)

    return system
end

function apply_prescribed_motion!(system::TotalLagrangianSPHSystem, ::Nothing, semi, t)
    return system
end

function update_quantities!(system::TotalLagrangianSPHSystem, v, u, v_ode, u_ode, semi, t)
    # Precompute PK1 stress tensor
    @trixi_timeit timer() "stress tensor" compute_pk1_corrected!(system, semi)

    return system
end

function update_boundary_interpolation!(system::TotalLagrangianSPHSystem, v, u,
                                        v_ode, u_ode, semi, t)
    (; boundary_model) = system

    # Only update boundary model
    update_pressure!(boundary_model, system, v, u, v_ode, u_ode, semi)
end

@inline function compute_pk1_corrected!(system, semi)
    (; deformation_grad, pk1_rho2, material_density) = system

    calc_deformation_grad!(deformation_grad, system, semi)

    @threaded semi for particle in eachparticle(system)
        pk1_particle = pk1_stress_tensor(system, particle)
        pk1_particle_corrected = pk1_particle * correction_matrix(system, particle)
        rho2_inv = 1 / material_density[particle]^2

        for j in 1:ndims(system), i in 1:ndims(system)
            # Precompute PK1 / rho^2 to avoid repeated divisions in the interaction loop
            @inbounds pk1_rho2[i, j, particle] = pk1_particle_corrected[i, j] * rho2_inv
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
        # Only consider particles with a distance > 0. See `src/general/smoothing_kernels.jl` for more details.
        initial_distance^2 < eps(initial_smoothing_length(system)^2) && return

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

function write_u0!(u0, system::TotalLagrangianSPHSystem)
    (; initial_condition) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices((ndims(system), each_integrated_particle(system)))
    copyto!(u0, indices, initial_condition.coordinates, indices)

    return u0
end

function write_v0!(v0, system::TotalLagrangianSPHSystem)
    (; initial_condition, boundary_model) = system

    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices((ndims(system), each_integrated_particle(system)))
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

    for particle in each_integrated_particle(system)
        # Set particle densities
        v0[ndims(system) + 1, particle] = initial_density[particle]
    end

    return v0
end

function restart_with!(system::TotalLagrangianSPHSystem, v, u)
    for particle in each_integrated_particle(system)
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
    von_mises_stress_vector = zeros(eltype(system.pk1_rho2), nparticles(system))

    @threaded default_backend(von_mises_stress_vector) for particle in
                                                           each_integrated_particle(system)
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
    P = pk1_rho2(system, particle) * system.material_density[particle]^2
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

    cauchy_stress_tensors = zeros(eltype(system.pk1_rho2), NDIMS, NDIMS,
                                  nparticles(system))

    @threaded default_backend(cauchy_stress_tensors) for particle in
                                                         each_integrated_particle(system)
        F = deformation_gradient(system, particle)
        J = det(F)
        P = pk1_rho2(system, particle) * system.material_density[particle]^2
        sigma = (1.0 / J) * P * F'
        cauchy_stress_tensors[:, :, particle] = sigma
    end

    return cauchy_stress_tensors
end

# To account for boundary effects in the viscosity term of the RHS, use the viscosity model
# of the neighboring particle systems.
@inline function viscosity_model(system::TotalLagrangianSPHSystem,
                                 neighbor_system::AbstractFluidSystem)
    return neighbor_system.viscosity
end

@inline function viscosity_model(system::Union{AbstractFluidSystem, OpenBoundarySystem},
                                 neighbor_system::TotalLagrangianSPHSystem)
    return neighbor_system.boundary_model.viscosity
end

function system_data(system::TotalLagrangianSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi)
    (; mass, material_density, deformation_grad, young_modulus,
     poisson_ratio, lame_lambda, lame_mu) = system

    dv = wrap_v(dv_ode, system, semi)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    coordinates = current_coordinates(u, system)
    initial_coordinates_ = initial_coordinates(system)
    velocity = [current_velocity(v, system, particle) for particle in eachparticle(system)]
    acceleration = system_data_acceleration(dv, system, system.clamped_particles_motion)
    pk1_corrected = [pk1_rho2(system, particle) * system.material_density[particle]^2
                     for particle in eachparticle(system)]

    return (; coordinates, initial_coordinates=initial_coordinates_, velocity, mass,
            material_density, deformation_grad, pk1_corrected, young_modulus, poisson_ratio,
            lame_lambda, lame_mu, acceleration)
end

function system_data_acceleration(dv, system::TotalLagrangianSPHSystem, ::Nothing)
    return dv
end

function system_data_acceleration(dv, system::TotalLagrangianSPHSystem, ::PrescribedMotion)
    clamped_particles = (n_integrated_particles(system) + 1):nparticles(system)
    return hcat(dv, view(system.cache.acceleration, :, clamped_particles))
end

function available_data(::TotalLagrangianSPHSystem)
    return (:coordinates, :initial_coordinates, :velocity, :mass, :material_density,
            :deformation_grad, :pk1_corrected, :young_modulus, :poisson_ratio,
            :lame_lambda, :lame_mu, :acceleration)
end

function Base.show(io::IO, system::TotalLagrangianSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "TotalLagrangianSPHSystem{", ndims(system), "}(")
    print(io, "", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.boundary_model)
    print(io, ", ", system.penalty_force)
    print(io, ", ", system.viscosity)
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
        n_clamped_particles = nparticles(system) - n_integrated_particles(system)

        summary_header(io, "TotalLagrangianSPHSystem{$(ndims(system))}")
        summary_line(io, "total #particles", nparticles(system))
        summary_line(io, "#clamped particles", n_clamped_particles)
        summary_line(io, "Young's modulus", display_param(system.young_modulus))
        summary_line(io, "Poisson ratio", display_param(system.poisson_ratio))
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "boundary model", system.boundary_model)
        summary_line(io, "penalty force", system.penalty_force)
        summary_line(io, "viscosity", system.viscosity)
        summary_footer(io)
    end
end
