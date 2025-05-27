"""
ImplicitIncompressibleSPHSystem(initial_condition,
                                smoothing_kernel, smoothing_length;
                                viscosity=nothing,
                                acceleration=ntuple(_ -> 0.0,ndims(smoothing_kernel)))

System for particles of a fluid.
The implicit incompressible SPH (IISPH) scheme is used, wherein a linear systems gets solved
iteratively in order to compute the correct pressure values such that the density deviation
from the rest density is (almost) zero.
See [Implicit Incompressible SPH](@ref iisph) for more details on the method.

# Arguments
- `initial_condition`:  [`InitialCondition`](@ref) representing the system's particles.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `reference_density`:  Reference density used for the fluid particles

# Keyword Arguments
- `viscosity`:                  Viscosity model for this system (default: no viscosity).
                                See [`ArtificialViscosityMonaghan`](@ref) or [`ViscosityAdami`](@ref).
- `acceleration`:               Acceleration vector for the system. (default: zero vector)
- `omega`:                      Relaxiaion parameter for the relaxed jacobi scheme(default: 0.5)
- `max_error`:                  Maximal error for the termination condition in the relaxed jacobi scheme (default: 0.1)
- `min_iterations`:             Minimal number of iterations in the relaxed jacobi scheme, independent from the termination condition. (default: 2)
- `max_iterations`:             Maximal number of iterations in the relaxed jacobi scheme, independent from the termination condition. (default: 20)
- `time_step`:                  Time step size used for the simulation (default: 0.001)
"""

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
struct ImplicitIncompressibleSPHSystem{NDIMS, ELTYPE <: Real, IC, MA, P, K,
                                       V, PF, SRFN, SRFT, PR, Dens, preDens, PV, D, A, SD,
                                       S, O, ME,
                                       MINI, MAXI, TS} <: FluidSystem{NDIMS}
    initial_condition                 :: IC
    mass                              :: MA     # Array{ELTYPE, 1}
    pressure                          :: P      # Array{ELTYPE, 1}
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
    reference_density                 :: ELTYPE
    acceleration                      :: SVector{NDIMS, ELTYPE}
    viscosity                         :: V
    pressure_acceleration_formulation :: PF
    transport_velocity                :: Nothing # TODO
    surface_normal_method             :: SRFN
    surface_tension                   :: SRFT
    particle_refinement               :: PR  #TODO
    density                           :: Dens    # Array{ELTYPE, 1}
    predicted_density                 :: preDens # Array{ELTYPE, 1}
    v_adv                             :: PV      # Array{ELTYPE, NDIMS}
    d_ii                              :: D       # Array{ELTYPE, NDIMS}
    a_ii                              :: A       # Array{ELTYPE, 1}
    sum_d_ij                          :: SD      # Array{ELTYPE, NDIMS}
    sum_term                          :: S       # Array{ELTYPE, 1}
    omega                             :: O
    max_error                         :: ME
    min_iterations                    :: MINI
    max_iterations                    :: MAXI
    time_step                         :: TS
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function ImplicitIncompressibleSPHSystem(initial_condition,
                                         smoothing_kernel, smoothing_length,
                                         reference_density;
                                         viscosity=nothing,
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         surface_tension=nothing,
                                         reference_particle_spacing=0.0, omega=0.5,
                                         max_error=0.1, min_iterations=2, max_iterations=20,
                                         time_step=0.001)
    particle_refinement = nothing # TODO

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

    if reference_density <= 0.0
        throw(ArgumentError("`reference_density` must be a positive number"))
    end

    pressure_acceleration = pressure_acceleration_summation_density

    density = copy(initial_condition.density)
    predicted_density = zeros(ELTYPE, n_particles)
    a_ii = zeros(ELTYPE, n_particles)
    d_ii = zeros(ELTYPE, NDIMS, n_particles)
    v_adv = zeros(ELTYPE, NDIMS, n_particles)
    sum_d_ij = zeros(ELTYPE, NDIMS, n_particles)
    sum_term = zeros(ELTYPE, n_particles)
    return ImplicitIncompressibleSPHSystem(initial_condition, mass, pressure,
                                           smoothing_kernel, smoothing_length,
                                           reference_density,
                                           acceleration_, viscosity,
                                           pressure_acceleration, nothing,
                                           nothing, surface_tension, particle_refinement,
                                           density, predicted_density,
                                           v_adv, d_ii, a_ii, sum_d_ij, sum_term, omega,
                                           max_error,
                                           min_iterations, max_iterations, time_step)
end

function reset_callback_flag!(system::ImplicitIncompressibleSPHSystem)
    return system
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
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::ImplicitIncompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "ImplicitIncompressibleSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
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
@inline function Base.eltype(::ImplicitIncompressibleSPHSystem{<:Any, ELTYPE}) where {ELTYPE}
    return ELTYPE
end

function initial_smoothing_length(system::ImplicitIncompressibleSPHSystem, ::Nothing)
    system.smoothing_length
end

function smoothing_length(system::ImplicitIncompressibleSPHSystem, particle)
    return system.smoothing_length
end

@inline each_moving_particle(system::ImplicitIncompressibleSPHSystem) = Base.OneTo(n_moving_particles(system))
@inline active_coordinates(u,
                           system::ImplicitIncompressibleSPHSystem) = current_coordinates(u,
                                                                                          system)
@inline active_particles(system::ImplicitIncompressibleSPHSystem) = eachparticle(system)

@inline function surface_tension_model(system::ImplicitIncompressibleSPHSystem)
    return nothing
end

@propagate_inbounds function current_pressure(v, system::ImplicitIncompressibleSPHSystem)
    return system.pressure
end

@propagate_inbounds function current_density(v, system::ImplicitIncompressibleSPHSystem)
    return system.density
end

#TODO: Was machen mit dem Soundspeed? Wird für viscosity benötigt
@inline system_sound_speed(system::ImplicitIncompressibleSPHSystem) = 1000.0

# Calculates the pressure values by solving a linear system with a relaxed jacobi scheme
function update_quantities!(system::ImplicitIncompressibleSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    (; time_step) = system

    @trixi_timeit timer() "predict advections" predict_advection(system, v, u, v_ode, u_ode,
                                                                 semi, t, time_step)

    @trixi_timeit timer() "pressure solver" pressure_solve(system, v, u, v_ode, u_ode, semi,
                                                           t, time_step)

    return system
end

function predict_advection(system, v, u, v_ode, u_ode, semi, t, time_step)
    # Get necessary fields
    (; density, predicted_density, d_ii, a_ii, v_adv, pressure) = system
    # Update density (with summation density)
    summation_density!(system, semi, u, u_ode, density)
    # Initialize predicted velocity and density
    v_particle_system = wrap_v(v_ode, system, semi)
    v_adv .= v_particle_system
    predicted_density .= density

    sound_speed = system_sound_speed(system) #TODO
    # Set d_ii and a_ii to zero
    set_zero!(d_ii)
    set_zero!(a_ii)
    # Calculate predicted velocity
    @threaded semi for particle in each_moving_particle(system)
        # Add acceleration
        for i in 1:ndims(system)
            v_adv[i, particle] += time_step * system.acceleration[i]
        end
    end
    foreach_system(semi) do neighbor_system
        # Get neighbor system u and v values
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        # Get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

        foreach_point_neighbor(system, neighbor_system,
                               system_coords, neighbor_system_coords,
                               semi;
                               points=each_moving_particle(system)) do particle,
                                                                       neighbor,
                                                                       pos_diff,
                                                                       distance
            m_a = hydrodynamic_mass(system, particle)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)

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
                v_adv[i,
                      particle] += time_step * dv_viscosity_[i]
            end
            # Calculate d_ii with the formula in eq. 9 from 'IHMSEN et al'
            for i in 1:ndims(system)
                d_ii[i,
                     particle] += calculate_d_ii(neighbor_system, m_b, rho_a,
                                                 grad_kernel[i],
                                                 time_step)
            end
        end
    end
    # Set initial pressure (p_0) to a half of the current pressure value
    @threaded semi for particle in each_moving_particle(system)
        pressure[particle] = 0.5 * pressure[particle]
    end
    # Calculation the daigonal elements (a_ii-values) according to equation 12 from 'IHMSEN et al'
    foreach_system(semi) do neighbor_system
        # Get neighbor system u and v values
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        # Get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

        foreach_point_neighbor(system, neighbor_system,
                               system_coords, neighbor_system_coords,
                               semi;
                               points=each_moving_particle(system)) do particle,
                                                                       neighbor,
                                                                       pos_diff,
                                                                       distance
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            # Compute d_ji
            dji = calculate_d_ij(system, particle, -grad_kernel, time_step)

            # Get d_ii
            dii = @inbounds get_d_ii(system, particle)

            m_b = hydrodynamic_mass(neighbor_system, neighbor)

            # Calculate a_ii values
            a_ii[particle] += m_b * dot((dii - dji), grad_kernel)
        end
    end
    # Calculate the predicted density (with the continuity equation and predicted velocities)
    foreach_system(semi) do neighbor_system
        # Get neighbor system u and v values
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        # Get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords,
                               neighbor_system_coords, semi,
                               points=each_moving_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
            # Calculate the predicted velocity differences
            v_adv_diff = predicted_velocity(system, particle) -
                         predicted_velocity(neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            # Compute rho_adv (eq. 4 from 'IHMSEN et al')
            predicted_density[particle] += time_step * m_b * dot(v_adv_diff, grad_kernel)
        end
    end
end

# Calculate pressure values with iterative pressure solver (relaxed jacobi scheme)
function pressure_solve(system, v, u, v_ode, u_ode, semi, t, time_step)
    # Get neccesary fields
    (; reference_density, max_error, min_iterations, max_iterations) = system
    avg_density_error = 0.0
    l = 1
    check = false
    while (!check)
        @trixi_timeit timer() "pressure solver iteration" pressure_solve_iteration(system,
                                                                                   avg_density_error,
                                                                                   u, u_ode,
                                                                                   semi,
                                                                                   time_step)
        eta = max_error * 0.01 * reference_density
        # Update termination condition
        check = (avg_density_error <= eta && l >= min_iterations) || l >= max_iterations
        l += 1
    end
end

function pressure_solve_iteration(system, avg_density_error, u, u_ode, semi, time_step)
    # Get neccesary fields
    (; reference_density, sum_d_ij, sum_term, pressure, predicted_density, d_ii, a_ii,
     omega) = system
    set_zero!(sum_d_ij)
    neighbor_system = system
    # Get neighbor system u and v values
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    # Get coordinates
    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords,
                           neighbor_system_coords, semi;
                           points=each_moving_particle(system)) do particle,
                                                                   neighbor,
                                                                   pos_diff,
                                                                   distance
        # Calculate the 'sum_d_ij' value for each particle
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        sum_d_ij_ = calculate_sum_dij(neighbor_system, neighbor, pressure,
                                      grad_kernel, time_step)
        for i in 1:ndims(system)
            sum_d_ij[i, particle] += sum_d_ij_[i]
        end
    end
    # Reset the sum_term
    set_zero!(sum_term)
    foreach_system(semi) do neighbor_system
        # Get neighbor system u and v values
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        # Get coordinates
        system_coords = current_coordinates(u, system)
        neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords,
                               neighbor_system_coords, semi;
                               points=each_moving_particle(system)) do particle,
                                                                       neighbor,
                                                                       pos_diff,
                                                                       distance
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            # Calculate sum_term ('sum_term') for each fluid particle
            sum_term[particle] += calculate_sum_term(system, neighbor_system, particle,
                                                     neighbor, pressure, grad_kernel,
                                                     time_step)
        end
    end
    # Update the pressure values
    for particle in eachparticle(system)
        # Removing instability by avoiding to divide through very low numbers for a_ii
        if abs(a_ii[particle]) > 1.0e-9
            pressure[particle] = max((1-omega) * pressure[particle] +
                                     omega / a_ii[particle] *
                                     (reference_density - predicted_density[particle] -
                                      sum_term[particle]), 0.0)
        else
            pressure[particle] = 0.0
        end
        # Calculate the average density error for the termination condition
        if (pressure[particle] != 0.0)
            new_density = a_ii[particle]*pressure[particle] + sum_term[particle] -
                          (reference_density - predicted_density[particle]) +
                          reference_density
            avg_density_error += (new_density - reference_density)
        end
    end
    avg_density_error /= nparticles(system)
end

@propagate_inbounds function predicted_velocity(system::ImplicitIncompressibleSPHSystem,
                                                particle)
    return extract_svector(system.v_adv, system, particle)
end

@propagate_inbounds function predicted_velocity(system::BoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function get_d_ii(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.d_ii, system, particle)
end

@propagate_inbounds function get_d_ii(system::BoundarySystem, particle)
    return zero(SVector{ndims(system), eltype(system)})
end

@propagate_inbounds function get_sum_d_ij(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.sum_d_ij, system, particle)
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(system::ImplicitIncompressibleSPHSystem, m_b, rho_a, grad_kernel,
                        time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(system::BoundarySystem, m_b, rho_a, grad_kernel, time_step)
    return calculate_d_ii(system::BoundarySystem, system.boundary_model, m_b, rho_a,
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
    return -time_step^2 * 2 * m_b / rho_a^2 * grad_kernel #for pressure mirroring
end

# Calculates a summand for the calculation of the d_ii values (pressure zeroing)
function calculate_d_ii(system, boundary_model, density_calculator::PressureZeroing, m_b,
                        rho_a, grad_kernel, time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel
end

# Calculates the d_ij value for a particle i and his neighbor j from the equation 9 in 'IHMSEN et al'
function calculate_d_ij(system::ImplicitIncompressibleSPHSystem, particle_j, grad_kernel,
                        time_step)
    # (delta t)^2 * m_i / rho_i ^2 * gradW_ij
    return -time_step^2 * hydrodynamic_mass(system, particle_j) /
           system.density[particle_j]^2 * grad_kernel
end

# Calculates the d_ij value for a particle i and his neighbor j from the equation 9 in 'IHMSEN et al'
function calculate_d_ij(system::BoundarySystem, particle_b, grad_kernel, time_step)
    # (delta t)^2 * m_b / rho_b ^2 * gradW_ib
    return -time_step^2 * hydrodynamic_mass(system, particle_b) /
           system.density[particle_b]^2 * grad_kernel
end

# Calculates the sum d_ij*p_j over all j for a given particle i ('IHMSEN et al' section 3.1.1)
function calculate_sum_dij(system::ImplicitIncompressibleSPHSystem, particle, pressure,
                           grad_kernel, time_step)
    p_b = pressure[particle]
    d_ab = calculate_d_ij(system, particle, grad_kernel, time_step)
    # d_ij * p_j
    return d_ab * p_b
end

# Calculates the sum term in equation (13) from 'IHMSEN et al'
function calculate_sum_term(system, neighbor_system::ImplicitIncompressibleSPHSystem,
                            particle, neighbor, pressure, grad_kernel,
                            time_step)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    sum_da = get_sum_d_ij(system, particle)
    d_bb = get_d_ii(neighbor_system, neighbor)
    p_a = pressure[particle]
    p_b = pressure[neighbor]
    sum_db = get_sum_d_ij(neighbor_system, neighbor)
    dba = calculate_d_ij(system, particle, -grad_kernel, time_step)
    # m_j * sum_j d_ij*p_j - d_jj*p_j - sum_k\neq i d_jk*p_k * grad_W_ij
    return m_b * dot(sum_da - d_bb * p_b - (sum_db - dba * p_a), grad_kernel)
end

# Calculates the sum term in equation (13) from 'IHMSEN et al'
function calculate_sum_term(system, neighbor_system::BoundarySystem, particle, neighbor,
                            pressure, grad_kernel, time_step)
    sum_da = get_sum_d_ij(system, particle)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    # m_b * sum_j d_ij*p_j * grad_W_ib
    return m_b * dot(sum_da, grad_kernel)
end
