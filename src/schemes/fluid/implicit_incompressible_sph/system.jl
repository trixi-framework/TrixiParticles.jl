"""
    ImplicitIncompressibleSPHSystem(initial_condition,
                                    smoothing_kernel, smoothing_length,
                                    reference_density;
                                    viscosity=nothing,
                                    acceleration=ntuple(_ -> 0.0, ndims(smoothing_kernel)),
                                    omega=0.5, max_error=0.1, min_iterations=2,
                                    max_iterations=20, time_step)

System for particles of a fluid.
The system employs implicit incompressible SPH (IISPH), iteratively solving a linear system
for the pressure so that density remains within a specified tolerance of the rest value.
See [Implicit Incompressible SPH](@ref iisph) for more details on the method.

# Arguments
- `initial_condition`:  [`InitialCondition`](@ref) representing the system's particles.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `reference_density`:  Reference density used for the fluid particles

# Keyword Arguments
- `viscosity`:                  Currently, only [`ViscosityMorris`](@ref)
                                and [`ViscosityAdami`](@ref) are supported.
- `acceleration`:               Acceleration vector for the system. (default: zero vector)
- `omega = 0.5`:                Relaxation parameter for the relaxed jacobi scheme
- `max_error = 0.1`:            Maximum error (in %) for the termination condition in the relaxed jacobi scheme
- `min_iterations = 2`:         Minimum number of iterations in the relaxed jacobi scheme, independent from the termination condition
- `max_iterations = 20`:        Maximum number of iterations in the relaxed jacobi scheme, independent from the termination condition
- `time_step`:                  Time step size used for the simulation
"""
struct ImplicitIncompressibleSPHSystem{NDIMS, ELTYPE <: Real, ARRAY1D, ARRAY2D,
                                       IC, K, V, PF, C} <: AbstractFluidSystem{NDIMS}
    initial_condition                 :: IC
    mass                              :: ARRAY1D # Array{ELTYPE, 1}
    pressure                          :: ARRAY1D
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
    reference_density                 :: ELTYPE
    acceleration                      :: SVector{NDIMS, ELTYPE}
    viscosity                         :: V
    pressure_acceleration_formulation :: PF
    surface_normal_method             :: Nothing # TODO
    surface_tension                   :: Nothing # TODO
    particle_refinement               :: Nothing # TODO
    density                           :: ARRAY1D
    predicted_density                 :: ARRAY1D
    advection_velocity                :: ARRAY2D # Array{ELTYPE, 2}
    d_ii                              :: ARRAY2D # Eq. 9
    a_ii                              :: ARRAY1D # Diagonal elements of the implicit pressure equation (Eq. 6)
    sum_d_ij_pj                       :: ARRAY2D # \sum_j d_{ij} p_j (Eq. 10)
    sum_term                          :: ARRAY1D # Sum term of Eq. 13
    density_error                     :: ARRAY1D # Temporary storage for parallel reduction
    omega                             :: ELTYPE  # Relaxed Jacobi parameter
    max_error                         :: ELTYPE  # maximal error of the average density deviation (in %)
    min_iterations                    :: Int     # minimum number of iterations in the pressure solver
    max_iterations                    :: Int     # maximum number of iterations in the pressure solver
    time_step                         :: ELTYPE
    artificial_sound_speed            :: ELTYPE  # TODO
    cache                             :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function ImplicitIncompressibleSPHSystem(initial_condition,
                                         smoothing_kernel, smoothing_length,
                                         reference_density;
                                         viscosity=nothing,
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         omega=0.5, max_error=0.1, min_iterations=2,
                                         max_iterations=20, time_step,
                                         artificial_sound_speed=1000.0)
    particle_refinement = nothing # TODO
    surface_tension = nothing # TODO

    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)
    n_particles = nparticles(initial_condition)

    mass = copy(initial_condition.mass)
    pressure = copy(initial_condition.pressure)

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    # Make acceleration an SVector
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    if reference_density <= 0
        throw(ArgumentError("`reference_density` must be a positive number"))
    end

    if !(0 < max_error <= 100)
        throw(ArgumentError("`max_error` is given in percentage, so it must be a number between 0 and 100"))
    end

    if min_iterations < 1
        throw(ArgumentError("`min_iterations` must be a positive number"))
    end

    if max_iterations < min_iterations
        throw(ArgumentError("`min_iterations` must be smaller or equal to `max_iterations`"))
    end

    if time_step <= 0
        throw(ArgumentError("`time_step` must be a positive number"))
    end

    pressure_acceleration = pressure_acceleration_summation_density

    density = copy(initial_condition.density)
    predicted_density = zeros(ELTYPE, n_particles)
    a_ii = zeros(ELTYPE, n_particles)
    d_ii = zeros(ELTYPE, NDIMS, n_particles)
    advection_velocity = zeros(ELTYPE, NDIMS, n_particles)
    sum_d_ij_pj = zeros(ELTYPE, NDIMS, n_particles)
    sum_term = zeros(ELTYPE, n_particles)
    density_error = zeros(ELTYPE, n_particles)

    cache = (;
             create_cache_refinement(initial_condition, particle_refinement,
                                     smoothing_length)...,)

    return ImplicitIncompressibleSPHSystem(initial_condition, mass, pressure,
                                           smoothing_kernel, smoothing_length,
                                           reference_density, acceleration_, viscosity,
                                           pressure_acceleration, nothing, surface_tension,
                                           particle_refinement, density, predicted_density,
                                           advection_velocity, d_ii, a_ii, sum_d_ij_pj,
                                           sum_term, density_error, omega, max_error,
                                           min_iterations, max_iterations, time_step,
                                           artificial_sound_speed, cache)
end

function write_v0!(v0, system::ImplicitIncompressibleSPHSystem)
    # This is as fast as a loop with `@inbounds`, but it's GPU-compatible
    indices = CartesianIndices(system.initial_condition.velocity)
    copyto!(v0, indices, system.initial_condition.velocity, indices)
    return v0
end

function Base.show(io::IO, system::ImplicitIncompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "ImplicitIncompressibleSPHSystem{", ndims(system), "}(")
    print(io, system.reference_density)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.omega)
    print(io, ", ", system.max_error)
    print(io, ", ", system.min_iterations)
    print(io, ", ", system.max_iterations)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::ImplicitIncompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "ImplicitIncompressibleSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "reference density", system.reference_density)
        summary_line(io, "density calculator", "SummationDensity")
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "omega", system.omega)
        summary_line(io, "max_error", system.max_error)
        summary_line(io, "min_iterations", system.min_iterations)
        summary_line(io, "max_iterations", system.max_iterations)
        summary_footer(io)
    end
end

@inline source_terms(system::ImplicitIncompressibleSPHSystem) = nothing

@inline function Base.eltype(::ImplicitIncompressibleSPHSystem{<:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

@inline function surface_tension_model(system::ImplicitIncompressibleSPHSystem)
    return nothing
end

@propagate_inbounds function current_pressure(v, system::ImplicitIncompressibleSPHSystem)
    return system.pressure
end

@propagate_inbounds function current_density(v, system::ImplicitIncompressibleSPHSystem)
    return system.density
end

# TODO: What do we do with the sound speed? This is needed for the viscosity.
@inline system_sound_speed(system::ImplicitIncompressibleSPHSystem) = system.artificial_sound_speed

# Calculates the pressure values by solving a linear system with a relaxed Jacobi scheme
function update_quantities!(system::ImplicitIncompressibleSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    @trixi_timeit timer() "predict advection" predict_advection(system, v, u, v_ode, u_ode,
                                                                semi, t)

    @trixi_timeit timer() "pressure solver" pressure_solve(system, v, u, v_ode, u_ode, semi,
                                                           t)

    return system
end

function predict_advection(system, v, u, v_ode, u_ode, semi, t)
    (; density, predicted_density, advection_velocity, pressure, time_step) = system
    d_ii_array = system.d_ii

    sound_speed = system_sound_speed(system) # TODO

    # Compute density by kernel summation
    summation_density!(system, semi, u, u_ode, density)

    # Initialize arrays
    v_particle_system = wrap_v(v_ode, system, semi)
    predicted_density .= density
    set_zero!(d_ii_array)

    @threaded semi for particle in each_integrated_particle(system)
        # Initialize the advection velocity with the current velocity plus the system acceleration
        v_particle = current_velocity(v_particle_system, system, particle)
        for i in 1:ndims(system)
            advection_velocity[i,
                               particle] = v_particle[i] +
                                           time_step * system.acceleration[i]
        end
    end

    # Compute predicted velocity
    foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

        foreach_point_neighbor(system, neighbor_system,
                               system_coords, neighbor_system_coords, semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            m_a = @inbounds hydrodynamic_mass(system, particle)
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

            rho_a = @inbounds current_density(v_particle_system, system, particle)
            rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            dv_viscosity_ = @inbounds dv_viscosity(system, neighbor_system,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   sound_speed, m_a, m_b, rho_a, rho_b,
                                                   grad_kernel)
            # Add all other non-pressure forces
            for i in 1:ndims(system)
                @inbounds advection_velocity[i, particle] += time_step * dv_viscosity_[i]
            end
            # Calculate d_ii with eq. 9 in Ihmsen et al. (2013)
            for i in 1:ndims(system)
                d_ii_array[i,
                           particle] += calculate_d_ii(neighbor_system, m_b, rho_a,
                                                       grad_kernel[i], time_step)
            end
        end
    end

    # Set initial pressure (p_0) to a half of the current pressure value
    @threaded semi for particle in each_integrated_particle(system)
        pressure[particle] = pressure[particle] / 2
    end

    # Calculation the diagonal elements (a_ii-values)
    calculate_diagonal_elements!(system, v, u, v_ode, u_ode, semi)

    # Calculate the predicted density (with the continuity equation and predicted velocities)
    foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords,
                               neighbor_system_coords, semi,
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            # Calculate the predicted velocity differences
            advection_velocity_diff = predicted_velocity(system, particle) -
                                      predicted_velocity(neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            # Compute \rho_adv in eq. 4 in Ihmsen et al. (2013)
            predicted_density[particle] += time_step * m_b *
                                           dot(advection_velocity_diff, grad_kernel)
        end
    end
end

function calculate_diagonal_elements!(system, v, u, v_ode, u_ode, semi)
    (; a_ii, time_step) = system

    set_zero!(a_ii)

    foreach_system(semi) do neighbor_system
        calculate_diagonal_elements!(a_ii, system, neighbor_system, v, u, v_ode, u_ode,
                                     semi, time_step)
    end
end

# Calculation of the contribution of the fluid particles to the diagonal elements (a_ii-values)
# according to eq. 12 in Ihmsen et al. (2013).
function calculate_diagonal_elements!(a_ii, system, neighbor_system, v, u, v_ode, u_ode,
                                      semi, time_step)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

        # Compute d_ji according to eq. 9 in Ihmsen et al. (2013).
        # Note that we compute d_ji and not d_ij. We can use the antisymmetry
        # of the kernel gradient and just flip the sign of W_ij to obtain W_ji.
        d_ji_ = -time_step^2 * hydrodynamic_mass(system, particle) /
                system.density[particle]^2 * (-grad_kernel)

        d_ii_ = d_ii(system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        # According to eq. 12 in Ihmsen et al. (2013)
        a_ii[particle] += m_b * dot((d_ii_ - d_ji_), grad_kernel)
    end
end

# Calculation of the contribution of the Abstractboundary particles the diagonal elements (a_ii-values)
# according to Ihmsen et al. (2013)
function calculate_diagonal_elements!(a_ii, system, neighbor_system::AbstractBoundarySystem,
                                      v, u,
                                      v_ode, u_ode, semi, time_step)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

        d_ii_ = d_ii(system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        # Contribution to the diagonal elements without d_ji value (see eq. 16 in Ihmsen et al. (2013))
        a_ii[particle] += m_b * dot(d_ii_, grad_kernel)
    end
end

# Calculate pressure values with iterative pressure solver (relaxed Jacobi scheme)
function pressure_solve(system, v, u, v_ode, u_ode, semi, t)
    (; reference_density, max_error, min_iterations, max_iterations, time_step) = system

    l = 1
    terminate = false
    # Convert relative error in percent to absolute error
    eta = max_error * reference_density / 100
    while (!terminate)
        @trixi_timeit timer() "pressure solver iteration" begin
            avg_density_error = pressure_solve_iteration(system, u,
                                                         u_ode, semi, time_step)
            # Update termination condition
            terminate = (avg_density_error <= eta && l >= min_iterations) ||
                        l >= max_iterations
            l += 1
        end
    end
end

function pressure_solve_iteration(system, u, u_ode, semi, time_step)
    (; reference_density, sum_d_ij_pj, sum_term, pressure, predicted_density, a_ii,
     omega, density_error) = system

    set_zero!(sum_d_ij_pj)

    system_coords = current_coordinates(u, system)

    foreach_point_neighbor(system, system, system_coords, system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        # Calculate the sum d_ij * p_j over all neighbors j for each particle i
        # (Ihmsen et al. 2013, eq. 13)
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        p_b = pressure[neighbor]
        d_ab = calculate_d_ij(system, neighbor, grad_kernel, time_step)
        sum_dij_pj_ = d_ab * p_b

        for i in 1:ndims(system)
            sum_d_ij_pj[i, particle] += sum_dij_pj_[i]
        end
    end

    # Calculate the large sum in eq. 13 of Ihmsen et al. (2013) for each particle (as `sum_term`)
    set_zero!(sum_term)

    foreach_system(semi) do neighbor_system
        calculate_sum_term!(sum_term, system, neighbor_system,
                            pressure, u, u_ode, semi, time_step)
    end

    # Update the pressure values
    avg_density_error = zero(eltype(system))
    @threaded semi for particle in eachparticle(system)
        # Removing instabilities by avoiding to divide by very low values of `a_ii`.
        # This is not mentioned in the paper but done in SPlisHSPlasH as well.
        if abs(a_ii[particle]) > 1.0e-9
            pressure[particle] = max((1-omega) * pressure[particle] +
                                     omega / a_ii[particle] *
                                     (reference_density - predicted_density[particle] -
                                      sum_term[particle]), 0)
        else
            pressure[particle] = zero(pressure[particle])
        end
        # Calculate the average density error for the termination condition
        if (pressure[particle] != 0.0)
            new_density = a_ii[particle]*pressure[particle] + sum_term[particle] -
                          (reference_density - predicted_density[particle]) +
                          reference_density
            density_error[particle] = (new_density - reference_density)
        end
    end
    avg_density_error = sum(density_error) / nparticles(system)

    return avg_density_error
end

# Function barrier for type stability
function calculate_sum_term!(sum_term, system, neighbor_system,
                             pressure, u, u_ode, semi, time_step)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords,
                           neighbor_system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        sum_term[particle] += calculate_sum_term(system, neighbor_system, particle,
                                                 neighbor, pressure, grad_kernel, time_step)
    end

    return sum_term
end

@propagate_inbounds function predicted_velocity(system::ImplicitIncompressibleSPHSystem,
                                                particle)
    return extract_svector(system.advection_velocity, system, particle)
end

@propagate_inbounds function predicted_velocity(system::AbstractBoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function d_ii(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.d_ii, system, particle)
end

@propagate_inbounds function d_ii(system::AbstractBoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function sum_dij_pj(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.sum_d_ij_pj, system, particle)
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(system::ImplicitIncompressibleSPHSystem, m_b, rho_a, grad_kernel,
                        time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(system::AbstractBoundarySystem, m_b, rho_a, grad_kernel, time_step)
    return calculate_d_ii(system, system.boundary_model, m_b, rho_a,
                          grad_kernel, time_step)
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(system, boundary_model::BoundaryModelDummyParticles, m_b, rho_a,
                        grad_kernel, time_step)
    return calculate_d_ii(system, boundary_model, boundary_model.density_calculator, m_b,
                          rho_a, grad_kernel, time_step)
end

# Calculates a summand for the calculation of the d_ii values (pressure mirroring)
function calculate_d_ii(system, boundary_model, density_calculator::PressureMirroring, m_b,
                        rho_a, grad_kernel, time_step)
    # The linear system to solve originates from the pressure acceleration:
    #     ∑ m_j (p_i/ρ_i + p_b/ρ_b) ∇W_ij.
    # With pressure mirroring, this becomes
    #     ∑ m_j (p_i/ρ_i + p_i/ρ_i) ∇W_ij,
    # which simplifies to
    #     ∑ m_j (2 * p_i / ρ_i) ∇W_ij.
    # Therefore, the diagonal element in the system now appears with a factor 2,
    # whereas the entry `ij` disappears from the system.
    return -time_step^2 * 2 * m_b / rho_a^2 * grad_kernel
end

# Calculates a summand for the calculation of the d_ii values (pressure zeroing)
function calculate_d_ii(system, boundary_model, density_calculator::PressureZeroing, m_b,
                        rho_a, grad_kernel, time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel
end

# Calculates the d_ij value for a particle `i` and its neighbor `j` from eq. 9 in Ihmsen et al. (2013).
# Note that `i` is only implicitly included in the kernel gradient.
function calculate_d_ij(system, particle_j, grad_kernel, time_step)
    # (Δt)^2 * m_j / ρ_j ^2 * ∇W_ij
    return -time_step^2 * hydrodynamic_mass(system, particle_j) /
           system.density[particle_j]^2 * grad_kernel
end

# Calculate the large sum in eq. 13 of Ihmsen et al. (2013) for each particle (as `sum_term`)
function calculate_sum_term(system, neighbor_system::ImplicitIncompressibleSPHSystem,
                            particle, neighbor, pressure, grad_kernel, time_step)
    m_j = hydrodynamic_mass(neighbor_system, neighbor)
    sum_dik_pk = sum_dij_pj(system, particle)
    d_jj = d_ii(neighbor_system, neighbor)
    p_i = pressure[particle]
    p_j = pressure[neighbor]
    sum_djk_pk = sum_dij_pj(neighbor_system, neighbor)
    d_ji = calculate_d_ij(system, particle, -grad_kernel, time_step)

    # Equation 13 of Ihmsen et al. (2013):
    # m_j * (\sum_k d_ik * p_k - d_jj * p_j - \sum_{k != i} d_jk * p_k) * ∇W_ij
    return m_j * dot(sum_dik_pk - d_jj * p_j - (sum_djk_pk - d_ji * p_i), grad_kernel)
end

function calculate_sum_term(system, neighbor_system::AbstractBoundarySystem, particle,
                            neighbor,
                            pressure, grad_kernel, time_step)
    sum_dik_pk = sum_dij_pj(system, particle)
    m_j = hydrodynamic_mass(neighbor_system, neighbor)

    # Equation 16 of Ihmsen et al. (2013):
    # m_j * sum_k d_ik * p_k * ∇W_ij
    return m_j * dot(sum_dik_pk, grad_kernel)
end
