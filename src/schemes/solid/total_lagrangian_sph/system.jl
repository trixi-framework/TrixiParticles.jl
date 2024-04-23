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
    ```jldoctest; output = false, filter = r"InitialCondition{Float64}.*", setup = :(fixed_particles = RectangularShape(0.1, (1, 4), (0.0, 0.0), density=1.0); beam = RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0))
    solid = union(beam, fixed_particles)

    # output
    InitialCondition{Float64}(...) *the rest of this line is ignored by filter*
    ```
    where `beam` and `fixed_particles` are of type `InitialCondition`.
"""
struct TotalLagrangianSPHSystem{BM, NDIMS, ELTYPE <: Real, IC, ARRAY1, ARRAY2, ARRAY3,
                                K, PF, ST} <: SolidSystem{NDIMS}
    initial_condition   :: IC
    initial_coordinates :: ARRAY2 # Array{ELTYPE, 2}: [dimension, particle]
    current_coordinates :: ARRAY2 # Array{ELTYPE, 2}: [dimension, particle]
    mass                :: ARRAY1 # Array{ELTYPE, 1}: [particle]
    correction_matrix   :: ARRAY3 # Array{ELTYPE, 3}: [i, j, particle]
    pk1_corrected       :: ARRAY3 # Array{ELTYPE, 3}: [i, j, particle]
    deformation_grad    :: ARRAY3 # Array{ELTYPE, 3}: [i, j, particle]
    material_density    :: ARRAY1 # Array{ELTYPE, 1}: [particle]
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
    source_terms        :: ST

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

        lame_lambda = young_modulus * poisson_ratio /
                      ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        lame_mu = 0.5 * young_modulus / (1 + poisson_ratio)

        return new{typeof(boundary_model), NDIMS, ELTYPE,
                   typeof(initial_condition),
                   typeof(mass), typeof(initial_coordinates),
                   typeof(deformation_grad), typeof(smoothing_kernel),
                   typeof(penalty_force),
                   typeof(source_terms)}(initial_condition, initial_coordinates,
                                         current_coordinates, mass, correction_matrix,
                                         pk1_corrected, deformation_grad, material_density,
                                         n_moving_particles, young_modulus, poisson_ratio,
                                         lame_lambda, lame_mu, smoothing_kernel,
                                         smoothing_length, acceleration_, boundary_model,
                                         penalty_force, source_terms)
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
    nhs = get_neighborhood_search(system, semi)
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
        F_particle = deformation_gradient(system, particle)
        pk1_particle = pk1_stress_tensor(F_particle, system)
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
@inline function pk1_stress_tensor(F, system)
    S = pk2_stress_tensor(F, system)

    return F * S
end

# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(F, system)
    (; lame_lambda, lame_mu) = system

    # Compute the Green-Lagrange strain
    E = 0.5 * (transpose(F) * F - I)

    return lame_lambda * tr(E) * I + 2 * lame_mu * E
end

@inline function calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                                     initial_distance, system, m_a, m_b, rho_a, rho_b,
                                     ::Nothing)
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

# An explanation of these equation can be found in
# J. Lubliner, 2008. Plasticity theory.
# See here below Equation 5.3.21 for the equation for the equivalent stress.
# The von-Mises stress is one form of equivalent stress, where sigma is the deviatoric stress.
# See pages 32 and 123.
function von_mises_stress(system::TotalLagrangianSPHSystem)
    von_mises_stress_vector = zeros(eltype(system.pk1_corrected), nparticles(system))

    @threaded for particle in each_moving_particle(system)
        von_mises_stress_vector[particle] = von_mises_stress(system, particle)
    end

    return von_mises_stress_vector
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl
# with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function von_mises_stress(system, particle)
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

    @threaded for particle in each_moving_particle(system)
        F = deformation_gradient(system, particle)
        J = det(F)
        P = pk1_corrected(system, particle)
        sigma = (1.0 / J) * P * F'
        cauchy_stress_tensors[:, :, particle] = sigma
    end

    return cauchy_stress_tensors
end
