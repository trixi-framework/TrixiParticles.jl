"""
    ImplicitIncompressibleSPHSystem(initial_condition; smoothing_kernel,
                                    smoothing_length, reference_density,
                                    viscosity=nothing,
                                    acceleration=ntuple(_ -> 0.0, ndims(smoothing_kernel)),
                                    omega=0.5, max_error=0.1, min_iterations=2,
                                    max_iterations=20, time_step)

System for particles of a fluid.
The system employs implicit incompressible SPH (IISPH), iteratively solving a linear system
for the pressure so that density remains within a specified tolerance of the rest value.
See [Implicit Incompressible SPH](@ref iisph) for more details on the method.
!!! note "Time Integration"
    IISPH supports fixed-step time integration. When using fixed-step Runge-Kutta
    methods from OrdinaryDiffEq.jl, add [`IISPHTimeStepCallback`](@ref) so the
    pressure projection uses the current integrator step size. Algorithms exposing
    `stage_limiter!` and `step_limiter!` can also use [`IISPHTimeStepLimiter`](@ref)
    for stage-local synchronization. Adaptive time integration is currently
    experimental because rejected steps require restoring IISPH pressure caches.

# Arguments
- `initial_condition`:  [`InitialCondition`](@ref) representing the system's particles.

# Keywords
- `smoothing_kernel`:           Smoothing kernel to be used for this system.
                                See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:           Smoothing length to be used for this system.
                                See [Smoothing Kernels](@ref smoothing_kernel).
- `reference_density`:          Reference density used for the fluid particles.
- `viscosity`:                  Currently, only [`ViscosityMorris`](@ref)
                                and [`ViscosityAdami`](@ref) are supported.
- `acceleration`:               Acceleration vector for the system. (default: zero vector)
- `omega = 0.5`:                Relaxation parameter for the relaxed Jacobi scheme
- `max_error = 0.1`:            Maximum error (in %) for the termination condition in the relaxed Jacobi scheme
- `min_iterations = 2`:         Minimum number of iterations in the relaxed Jacobi scheme, independent from the termination condition
- `max_iterations = 20`:        Maximum number of iterations in the relaxed Jacobi scheme, independent from the termination condition
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
    inv_a_ii                          :: ARRAY1D # Cached inverse diagonal elements
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
function ImplicitIncompressibleSPHSystem(initial_condition; smoothing_kernel,
                                         smoothing_length, reference_density,
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
    inv_a_ii = zeros(ELTYPE, n_particles)
    d_ii = zeros(ELTYPE, NDIMS, n_particles)
    advection_velocity = zeros(ELTYPE, NDIMS, n_particles)
    sum_d_ij_pj = zeros(ELTYPE, NDIMS, n_particles)
    sum_term = zeros(ELTYPE, n_particles)
    density_error = zeros(ELTYPE, n_particles)

    cache = (;
             create_cache_refinement(initial_condition, particle_refinement,
                                     smoothing_length)...,
             create_cache_iisph_pressure(initial_condition, NDIMS, ELTYPE)...,)

    return ImplicitIncompressibleSPHSystem(initial_condition, mass, pressure,
                                           smoothing_kernel, smoothing_length,
                                           reference_density, acceleration_, viscosity,
                                           pressure_acceleration, nothing, surface_tension,
                                           particle_refinement, density, predicted_density,
                                           advection_velocity, d_ii, a_ii, inv_a_ii,
                                           sum_d_ij_pj,
                                           sum_term, density_error, omega, max_error,
                                           min_iterations, max_iterations, time_step,
                                           artificial_sound_speed, cache)
end

function create_cache_iisph_pressure(initial_condition, NDIMS, ELTYPE)
    n_particles = nparticles(initial_condition)

    return (pressure_neighbor_count=zeros(Int, n_particles),
            pressure_neighbor_offsets=zeros(Int, n_particles + 1),
            pressure_neighbor_cursor=zeros(Int, n_particles),
            pressure_neighbor=zeros(Int, 0),
            pressure_d_ij=zeros(ELTYPE, 0),
            pressure_grad_mass=zeros(ELTYPE, 0),
            pressure_d_ji_dot_grad=zeros(ELTYPE, 0),
            pressure_boundary_grad_mass_sum=zeros(ELTYPE, NDIMS, n_particles))
end

"""
    IISPHPressureOperator

Matrix-free IISPH pressure operator assembled from cached per-step neighbor coefficients.

The operator applies ``A p`` for the current IISPH pressure equation, where ``A`` contains
the diagonal entries `system.a_ii` and the cached off-diagonal pressure couplings. Build
the operator after the IISPH prediction step has updated the pressure cache, for example
inside the pressure solver or after `update_systems_and_nhs_before_pressure!`.

Use [`iisph_pressure_operator`](@ref) to construct an operator with compatibility checks.
"""
struct IISPHPressureOperator{S, SEMI}
    system :: S
    semi   :: SEMI
end

"""
    iisph_pressure_operator(system, semi)

Construct a matrix-free [`IISPHPressureOperator`](@ref) for `system`.

The operator currently uses the optimized cached IISPH pressure loop and therefore
requires exactly one [`ImplicitIncompressibleSPHSystem`](@ref) and no
[`PressureBoundaries`](@ref). Construct it after the per-step pressure cache has been
built.
"""
function iisph_pressure_operator(system::ImplicitIncompressibleSPHSystem, semi)
    supports_cached_iisph_pressure_loop(semi) ||
        throw(ArgumentError("the matrix-free IISPH pressure operator currently requires " *
                            "one `ImplicitIncompressibleSPHSystem` and no " *
                            "`PressureBoundaries`"))

    return IISPHPressureOperator(system, semi)
end

Base.eltype(operator::IISPHPressureOperator) = eltype(operator.system)
Base.size(operator::IISPHPressureOperator) = (nparticles(operator.system),
                                             nparticles(operator.system))
Base.size(operator::IISPHPressureOperator, dim::Integer) = dim in (1, 2) ?
                                                           nparticles(operator.system) : 1

function mul!(Ap, operator::IISPHPressureOperator, pressure)
    (; system, semi) = operator
    length(Ap) == nparticles(system) ||
        throw(DimensionMismatch("output length $(length(Ap)) does not match " *
                                "number of IISPH particles $(nparticles(system))"))
    length(pressure) == nparticles(system) ||
        throw(DimensionMismatch("pressure length $(length(pressure)) does not match " *
                                "number of IISPH particles $(nparticles(system))"))

    calculate_iisph_pressure_sum_d_ij_pj!(system.sum_d_ij_pj, pressure, system, semi)
    calculate_iisph_pressure_sum_term!(Ap, pressure, system, semi)

    @threaded semi for particle in eachparticle(system)
        @inbounds Ap[particle] += system.a_ii[particle] * pressure[particle]
    end

    return Ap
end

"""
    iisph_pressure_rhs!(rhs, operator)

Write the IISPH pressure right-hand side ``b`` into `rhs`.
"""
function iisph_pressure_rhs!(rhs, operator::IISPHPressureOperator)
    (; system, semi) = operator
    length(rhs) == nparticles(system) ||
        throw(DimensionMismatch("rhs length $(length(rhs)) does not match " *
                                "number of IISPH particles $(nparticles(system))"))

    @threaded semi for particle in eachparticle(system)
        @inbounds rhs[particle] = iisph_source_term(system, particle)
    end

    return rhs
end

"""
    iisph_pressure_residual!(residual, pressure, rhs, operator)

Write the matrix-free pressure residual ``rhs - A * pressure`` into `residual`.
"""
function iisph_pressure_residual!(residual, pressure, rhs,
                                  operator::IISPHPressureOperator)
    mul!(residual, operator, pressure)

    (; system, semi) = operator
    @threaded semi for particle in eachparticle(system)
        @inbounds residual[particle] = rhs[particle] - residual[particle]
    end

    return residual
end

"""
    iisph_pressure_apply_preconditioner!(z, residual, operator)

Apply the IISPH diagonal preconditioner ``D^{-1}`` to `residual` and write the result to
`z`.
"""
function iisph_pressure_apply_preconditioner!(z, residual,
                                              operator::IISPHPressureOperator)
    (; system, semi) = operator
    length(z) == nparticles(system) ||
        throw(DimensionMismatch("output length $(length(z)) does not match " *
                                "number of IISPH particles $(nparticles(system))"))
    length(residual) == nparticles(system) ||
        throw(DimensionMismatch("residual length $(length(residual)) does not match " *
                                "number of IISPH particles $(nparticles(system))"))

    @threaded semi for particle in eachparticle(system)
        @inbounds z[particle] = system.inv_a_ii[particle] * residual[particle]
    end

    return z
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
    (; density) = system

    # Compute density by kernel summation
    summation_density!(system, semi, u, u_ode, density)

    @trixi_timeit timer() "predict advection" predict_advection!(system, v, u, v_ode, u_ode,
                                                                 semi)
end

function update_implicit_sph!(semi, v_ode, u_ode, t)
    # This check is performed statically by the compiler and has no overhead
    if !any(system -> system isa ImplicitIncompressibleSPHSystem, semi.systems)
        return semi
    end
    iisph_step_end_projection_enabled(semi) && return semi

    @trixi_timeit timer() "pressure solver" pressure_solve!(semi, v_ode, u_ode)

    return semi
end

function predict_advection!(system::Union{ImplicitIncompressibleSPHSystem,
                                          WallBoundarySystem{<:BoundaryModelDummyParticles{<:PressureBoundaries}}},
                            v, u, v_ode, u_ode, semi)
    calculate_predicted_velocity_and_d_ii_values!(system, v, u, v_ode, u_ode, semi)

    calculate_diagonal_elements_and_predicted_density!(system, v, u, v_ode, u_ode, semi)

    build_iisph_pressure_cache!(system, u, u_ode, semi)

    return system
end

function build_iisph_pressure_cache!(system, u, u_ode, semi)
    return system
end

function build_iisph_pressure_cache!(system::ImplicitIncompressibleSPHSystem, u, u_ode,
                                     semi)
    supports_cached_iisph_pressure_loop(semi) || return system

    (; cache) = system
    (; pressure_neighbor_count, pressure_neighbor_offsets, pressure_neighbor_cursor,
       pressure_neighbor, pressure_d_ij, pressure_grad_mass,
       pressure_d_ji_dot_grad, pressure_boundary_grad_mass_sum) = cache

    fill!(pressure_neighbor_count, 0)
    set_zero!(pressure_boundary_grad_mass_sum)

    foreach_system(semi) do neighbor_system
        count_iisph_pressure_cache_entries!(pressure_neighbor_count, system,
                                            neighbor_system, u, u_ode, semi)
        accumulate_iisph_boundary_pressure_cache!(pressure_boundary_grad_mass_sum,
                                                  system, neighbor_system, u, u_ode,
                                                  semi)
    end

    pressure_neighbor_offsets[1] = 1
    for particle in eachindex(pressure_neighbor_count)
        @inbounds pressure_neighbor_offsets[particle + 1] = pressure_neighbor_offsets[particle] +
                                                            pressure_neighbor_count[particle]
    end

    n_entries = last(pressure_neighbor_offsets) - 1
    resize!(pressure_neighbor, n_entries)
    resize!(pressure_d_ij, ndims(system) * n_entries)
    resize!(pressure_grad_mass, ndims(system) * n_entries)
    resize!(pressure_d_ji_dot_grad, n_entries)
    copyto!(pressure_neighbor_cursor, 1, pressure_neighbor_offsets, 1,
            length(pressure_neighbor_cursor))

    foreach_system(semi) do neighbor_system
        fill_iisph_pressure_cache_entries!(system, neighbor_system, u, u_ode, semi)
    end

    return system
end

function supports_cached_iisph_pressure_loop(semi)
    iisph_system_count = 0
    has_pressure_boundaries = false

    foreach_system(semi) do system
        iisph_system_count += system isa ImplicitIncompressibleSPHSystem ? 1 : 0
        has_pressure_boundaries |= is_iisph_pressure_boundary_system(system)
    end

    return iisph_system_count == 1 && !has_pressure_boundaries
end

is_iisph_pressure_boundary_system(system) = false
function is_iisph_pressure_boundary_system(::WallBoundarySystem{<:BoundaryModelDummyParticles{<:PressureBoundaries}})
    return true
end

function count_iisph_pressure_cache_entries!(pressure_neighbor_count, system,
                                             neighbor_system, u, u_ode, semi)
    return pressure_neighbor_count
end

function count_iisph_pressure_cache_entries!(pressure_neighbor_count,
                                             system::ImplicitIncompressibleSPHSystem,
                                             neighbor_system::ImplicitIncompressibleSPHSystem,
                                             u, u_ode, semi)
    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(wrap_u(u_ode, neighbor_system, semi),
                                          neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        @inbounds pressure_neighbor_count[particle] += 1
    end

    return pressure_neighbor_count
end

function accumulate_iisph_boundary_pressure_cache!(pressure_boundary_grad_mass_sum,
                                                   system, neighbor_system, u, u_ode,
                                                   semi)
    return pressure_boundary_grad_mass_sum
end

function accumulate_iisph_boundary_pressure_cache!(pressure_boundary_grad_mass_sum,
                                                   system::ImplicitIncompressibleSPHSystem,
                                                   neighbor_system::AbstractBoundarySystem,
                                                   u, u_ode, semi)
    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(wrap_u(u_ode, neighbor_system, semi),
                                          neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        for i in 1:ndims(system)
            @inbounds pressure_boundary_grad_mass_sum[i, particle] += m_b *
                                                                      grad_kernel[i]
        end
    end

    return pressure_boundary_grad_mass_sum
end

function fill_iisph_pressure_cache_entries!(system, neighbor_system, u, u_ode, semi)
    return system
end

function fill_iisph_pressure_cache_entries!(system::ImplicitIncompressibleSPHSystem,
                                            neighbor_system::ImplicitIncompressibleSPHSystem,
                                            u, u_ode, semi)
    (; pressure_neighbor_cursor, pressure_neighbor, pressure_d_ij, pressure_grad_mass,
       pressure_d_ji_dot_grad) = system.cache

    time_step = iisph_projection_dt(semi)
    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(wrap_u(u_ode, neighbor_system, semi),
                                          neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        d_ij_ = calculate_d_ij(system, neighbor_system, neighbor, grad_kernel, time_step)
        d_ji_ = calculate_d_ji(system, neighbor_system, particle, -grad_kernel, time_step)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        @inbounds begin
            entry = pressure_neighbor_cursor[particle]
            pressure_neighbor_cursor[particle] = entry + 1
            pressure_neighbor[entry] = neighbor
            pressure_d_ji_dot_grad[entry] = m_b * dot(d_ji_, grad_kernel)

            base = (entry - 1) * ndims(system)
            for i in 1:ndims(system)
                pressure_d_ij[base + i] = d_ij_[i]
                pressure_grad_mass[base + i] = m_b * grad_kernel[i]
            end
        end
    end

    return system
end

function calculate_predicted_velocity_and_d_ii_values!(system, v, u, v_ode, u_ode, semi)
    return system
end

function calculate_predicted_velocity_and_d_ii_values!(system::ImplicitIncompressibleSPHSystem,
                                                       v, u, v_ode, u_ode, semi)
    (; advection_velocity) = system
    time_step = iisph_projection_dt(semi)
    projection_only = iisph_pressure_projection_only_enabled(semi)
    d_ii_array = system.d_ii

    v_particle_system = wrap_v(v_ode, system, semi)

    set_zero!(d_ii_array)

    sound_speed = system_sound_speed(system) # TODO

    @threaded semi for particle in each_integrated_particle(system)
        # Initialize the advection velocity with the current velocity plus the system acceleration
        v_particle = current_velocity(v_particle_system, system, particle)
        for i in 1:ndims(system)
            advection_velocity[i, particle] = v_particle[i]
            if !projection_only
                advection_velocity[i, particle] += time_step * system.acceleration[i]
            end
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
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

            rho_a = @inbounds current_density(v_particle_system, system, particle)

            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            if !projection_only
                m_a = @inbounds hydrodynamic_mass(system, particle)
                rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
                v_a = @inbounds current_velocity(v_particle_system, system, particle)
                v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

                dv_viscosity_ = Ref(zero(pos_diff))
                @inbounds dv_viscosity!(dv_viscosity_, system, neighbor_system,
                                        v_particle_system, v_neighbor_system,
                                        particle, neighbor, pos_diff, distance,
                                        sound_speed, m_a, m_b, rho_a, rho_b,
                                        v_a, v_b, grad_kernel)
                # Add all other non-pressure forces
                for i in 1:ndims(system)
                    @inbounds advection_velocity[i,
                                                 particle] += time_step * dv_viscosity_[][i]
                end
            end
            # Calculate d_ii with eq. 9 in Ihmsen et al. (2013)
            for i in 1:ndims(system)
                d_ii_array[i,
                           particle] += calculate_d_ii(neighbor_system, m_b, rho_a,
                                                       grad_kernel[i], time_step)
            end
        end
    end

    return system
end

function calculate_diagonal_elements_and_predicted_density!(system, v, u, v_ode, u_ode,
                                                            semi)
    return system
end

function calculate_diagonal_elements_and_predicted_density!(system::ImplicitIncompressibleSPHSystem,
                                                            v, u, v_ode,
                                                            u_ode, semi)
    (; a_ii, density, predicted_density, inv_a_ii) = system
    time_step = iisph_projection_dt(semi)

    set_zero!(a_ii)
    predicted_density .= density

    foreach_system(semi) do neighbor_system
        calculate_diagonal_elements_and_predicted_density(a_ii, predicted_density, system,
                                                          neighbor_system, v, u, v_ode,
                                                          u_ode, semi, time_step)
    end

    @threaded semi for particle in eachparticle(system)
        a_ii_particle = a_ii[particle]
        inv_a_ii[particle] = abs(a_ii_particle) > 1.0e-9 ? inv(a_ii_particle) :
                             zero(a_ii_particle)
    end
end

# Calculation of the contribution of the fluid particles to the diagonal elements
# (a_ii-values) and the predcited density (\rho_adv) according to eq. 12 and 4 in
# Ihmsen et al. (2013).
function calculate_diagonal_elements_and_predicted_density(a_ii, predicted_density, system,
                                                           neighbor_system::ImplicitIncompressibleSPHSystem,
                                                           v, u, v_ode, u_ode,
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
        d_ji_ = calculate_d_ji(system, neighbor_system, particle, -grad_kernel, time_step)
        d_ii_ = d_ii(system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        # According to eq. 12 in Ihmsen et al. (2013)
        a_ii[particle] += m_b * dot((d_ii_ - d_ji_), grad_kernel)

        # Calculate the predicted velocity differences
        advection_velocity_diff = predicted_velocity(system, particle) -
                                  predicted_velocity(neighbor_system, neighbor)

        # Compute \rho_adv in eq. 4 in Ihmsen et al. (2013)
        predicted_density[particle] += time_step * m_b *
                                       dot(advection_velocity_diff, grad_kernel)
    end
end

# Calculation of the contribution of the boundary particles to the diagonal elements
# (a_ii-values) and the predcited density (\rho_adv) according to eq. 12 and 4 in
# Ihmsen et al. (2013).
function calculate_diagonal_elements_and_predicted_density(a_ii, predicted_density, system,
                                                           neighbor_system::AbstractBoundarySystem,
                                                           v, u, v_ode, u_ode, semi,
                                                           time_step)
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

        # Calculate the predicted velocity differences
        advection_velocity_diff = predicted_velocity(system, particle) -
                                  predicted_velocity(neighbor_system, neighbor)

        # Compute \rho_adv in eq. 4 in Ihmsen et al. (2013)
        predicted_density[particle] += time_step * m_b *
                                       dot(advection_velocity_diff, grad_kernel)
    end
end

# Calculate pressure values with iterative pressure solver (relaxed Jacobi scheme)
function pressure_solve!(semi, v_ode, u_ode)
    initialize_iisph_pressure!(semi)
    start_time = time_ns()

    # Determine global number of particles included in the PPE solver
    n_iisph_particles_ = sum(n_iisph_particles, semi.systems)

    # Determine global iteration and error constraints across all IISPH systems
    min_iters = maximum(minimum_iisph_iterations, semi.systems)
    max_iters = minimum(maximum_iisph_iterations, semi.systems)
    max_err_percent = minimum(maximum_iisph_error, semi.systems)

    max_error = max_err_percent / 100
    terminate = false
    l = 1
    while (!terminate)
        @trixi_timeit timer() "pressure solver iteration" begin
            avg_density_error = pressure_solve_iteration(semi, u_ode, n_iisph_particles_)
            # Update termination condition
            terminate = (avg_density_error <= max_error && l >= min_iters) ||
                        l >= max_iters
            l += 1
        end
    end
    solve_time = (time_ns() - start_time) * 1.0e-9
    record_iisph_pressure_iterations!(semi, l - 1, solve_time)

    return semi
end

function initialize_iisph_pressure!(semi)
    damp_pressure = should_damp_iisph_pressure(semi)

    foreach_system(semi) do system
        initialize_pressure!(system, semi, damp_pressure)
    end
    mark_iisph_pressure_initialized!(semi)

    return semi
end

function initialize_pressure!(system, semi, damp_pressure)
    return system
end

function initialize_pressure!(system::Union{ImplicitIncompressibleSPHSystem,
                                            WallBoundarySystem{<:BoundaryModelDummyParticles{<:PressureBoundaries}}},
                              semi, damp_pressure)
    damp_pressure || return system

    # Set initial pressure (p_0) to a half of the current pressure value
    @threaded semi for particle in eachparticle(system)
        current_pressure(nothing, system)[particle] = current_pressure(nothing, system)[particle] /
                                                      2
    end
end

function pressure_solve_iteration(semi, u_ode, n_particles)
    if supports_cached_iisph_pressure_loop(semi)
        return cached_iisph_pressure_solve_iteration(semi, n_particles)
    end

    foreach_system(semi) do system
        u = wrap_u(u_ode, system, semi)
        calculate_sum_d_ij_pj!(system, u, u_ode, semi)
    end
    foreach_system(semi) do system
        u = wrap_u(u_ode, system, semi)
        calculate_sum_term_values!(system, u, u_ode, semi)
    end

    # Wrap with `Ref` to allow modification inside the anonymous function below (without implicit boxing)
    total_density_error = Ref(0.0)
    foreach_system(semi) do system
        u = wrap_u(u_ode, system, semi)
        total_density_error[] += pressure_update(system, semi)
    end
    avg_density_error = total_density_error[] / n_particles

    return avg_density_error
end

function cached_iisph_pressure_solve_iteration(semi, n_particles)
    foreach_system(semi) do system
        calculate_cached_sum_d_ij_pj!(system, semi)
    end
    foreach_system(semi) do system
        calculate_cached_sum_term_values!(system, semi)
    end

    total_density_error = Ref(0.0)
    foreach_system(semi) do system
        total_density_error[] += pressure_update(system, semi)
    end

    return total_density_error[] / n_particles
end

function calculate_cached_sum_d_ij_pj!(system, semi)
    return system
end

function calculate_cached_sum_d_ij_pj!(system::ImplicitIncompressibleSPHSystem, semi)
    calculate_iisph_pressure_sum_d_ij_pj!(system.sum_d_ij_pj, system.pressure, system,
                                          semi)

    return system
end

function calculate_iisph_pressure_sum_d_ij_pj!(sum_d_ij_pj, pressure,
                                               system::ImplicitIncompressibleSPHSystem,
                                               semi)
    (; pressure_neighbor_offsets, pressure_neighbor, pressure_d_ij) = system.cache

    set_zero!(sum_d_ij_pj)

    @threaded semi for particle in each_integrated_particle(system)
        @inbounds for entry in pressure_neighbor_offsets[particle]:(pressure_neighbor_offsets[particle + 1] - 1)
            neighbor = pressure_neighbor[entry]
            p_neighbor = pressure[neighbor]
            base = (entry - 1) * ndims(system)
            for i in 1:ndims(system)
                sum_d_ij_pj[i, particle] += pressure_d_ij[base + i] * p_neighbor
            end
        end
    end

    return sum_d_ij_pj
end

function calculate_cached_sum_term_values!(system, semi)
    return system
end

function calculate_cached_sum_term_values!(system::ImplicitIncompressibleSPHSystem, semi)
    calculate_iisph_pressure_sum_term!(system.sum_term, system.pressure, system, semi)

    return system
end

function calculate_iisph_pressure_sum_term!(sum_term, pressure,
                                            system::ImplicitIncompressibleSPHSystem,
                                            semi)
    (; d_ii, sum_d_ij_pj) = system
    (; pressure_neighbor_offsets, pressure_neighbor, pressure_grad_mass,
       pressure_d_ji_dot_grad, pressure_boundary_grad_mass_sum) = system.cache

    set_zero!(sum_term)

    @threaded semi for particle in each_integrated_particle(system)
        sum_term_particle = zero(eltype(system))
        pressure_particle = @inbounds pressure[particle]

        @inbounds for i in 1:ndims(system)
            sum_term_particle += sum_d_ij_pj[i, particle] *
                                 pressure_boundary_grad_mass_sum[i, particle]
        end

        @inbounds for entry in pressure_neighbor_offsets[particle]:(pressure_neighbor_offsets[particle + 1] - 1)
            neighbor = pressure_neighbor[entry]
            pressure_neighbor_ = pressure[neighbor]
            base = (entry - 1) * ndims(system)

            for i in 1:ndims(system)
                sum_term_particle += (sum_d_ij_pj[i, particle] -
                                      d_ii[i, neighbor] * pressure_neighbor_ -
                                      sum_d_ij_pj[i, neighbor]) *
                                     pressure_grad_mass[base + i]
            end
            sum_term_particle += pressure_particle * pressure_d_ji_dot_grad[entry]
        end

        @inbounds sum_term[particle] = sum_term_particle
    end

    return sum_term
end

function calculate_sum_d_ij_pj!(system, u, u_ode, semi)
    return system
end

function calculate_sum_d_ij_pj!(system::ImplicitIncompressibleSPHSystem, u, u_ode, semi)
    (; sum_d_ij_pj) = system

    set_zero!(sum_d_ij_pj)

    foreach_system(semi) do neighbor_system
        calculate_sum_d_ij_pj!(sum_d_ij_pj, system, neighbor_system, u, u_ode, semi)
    end
end

# With `PressureBoundaries`, the boundary particles have their own pressure values that contribute to the sum_j d_ij*p_j
function calculate_sum_d_ij_pj!(sum_d_ij_pj, system,
                                neighbor_system::Union{ImplicitIncompressibleSPHSystem,
                                                       WallBoundarySystem{<:BoundaryModelDummyParticles{<:PressureBoundaries}}},
                                u, u_ode, semi)
    time_step = iisph_projection_dt(semi)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        # Calculate the sum d_ij * p_j over all neighbors j for each particle i
        # (Ihmsen et al. 2013, eq. 13)
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        p_b = current_pressure(nothing, neighbor_system, neighbor)
        d_ab = calculate_d_ij(system, neighbor_system, neighbor, grad_kernel, time_step)
        sum_dij_pj_ = d_ab * p_b

        for i in 1:ndims(system)
            sum_d_ij_pj[i, particle] += sum_dij_pj_[i]
        end
    end

    return sum_d_ij_pj
end

# For everything but IISPH system and `PressureBoundaries`, there is no contribution to sum_j d_ij*p_j
function calculate_sum_d_ij_pj!(sum_d_ij_pj, system,
                                neighbor_system, u, u_ode, semi)
    return sum_d_ij_pj
end

function calculate_sum_term_values!(system, u, u_ode, semi)
    return system
end

# Calculate the large sum in eq. 13 of Ihmsen et al. (2013) for each particle (as `sum_term`)
function calculate_sum_term_values!(system::ImplicitIncompressibleSPHSystem, u, u_ode, semi)
    (; sum_term, pressure) = system
    time_step = iisph_projection_dt(semi)

    set_zero!(sum_term)

    foreach_system(semi) do neighbor_system
        calculate_sum_term!(sum_term, system, neighbor_system, u, u_ode, semi, time_step)
    end
end

# Function barrier for type stability
function calculate_sum_term!(sum_term, system, neighbor_system, u, u_ode, semi, time_step)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system, system_coords,
                           neighbor_system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        sum_term[particle] += calculate_sum_term(system, neighbor_system, particle,
                                                 neighbor, grad_kernel, time_step)
    end
end

function pressure_update(system, semi)
    return 0.0
end

function pressure_update(system::ImplicitIncompressibleSPHSystem, semi)
    (; pressure, sum_term, reference_density, a_ii, inv_a_ii, omega, density_error) = system

    # Update the pressure values
    relative_density_error = pressure_update(system, pressure, reference_density, a_ii,
                                             inv_a_ii, sum_term, omega, density_error, semi)

    return relative_density_error
end

function pressure_update(system, pressure, reference_density, a_ii, inv_a_ii, sum_term,
                         omega, density_error, semi)
    relative_density_error = zero(eltype(system))

    @threaded semi for particle in eachparticle(system)
        inv_a_ii_particle = inv_a_ii[particle]
        source_term = iisph_source_term(system, particle)
        if inv_a_ii_particle != 0
            pressure[particle] = max((1 - omega) * pressure[particle] +
                                     omega * inv_a_ii_particle *
                                     (source_term - sum_term[particle]), 0)
        else
            pressure[particle] = zero(pressure[particle])
        end
        # Calculate the average density error for the termination condition
        if (pressure[particle] != 0.0)
            new_density = a_ii[particle] * pressure[particle] + sum_term[particle] -
                          source_term + reference_density
            density_error[particle] = (new_density - reference_density)
        else
            density_error[particle] = zero(density_error[particle])
        end
    end
    relative_density_error = sum(density_error) / reference_density

    return relative_density_error
end

function pressure_update(system, pressure, reference_density, a_ii, sum_term, omega,
                         density_error, semi)
    relative_density_error = zero(eltype(system))

    @threaded semi for particle in eachparticle(system)
        source_term = iisph_source_term(system, particle)
        # Removing instabilities by avoiding to divide by very low values of `a_ii`.
        # This is not mentioned in the paper but done in SPlisHSPlasH as well.
        if abs(a_ii[particle]) > 1.0e-9
            pressure[particle] = max((1 - omega) * pressure[particle] +
                                     omega / a_ii[particle] *
                                     (source_term - sum_term[particle]), 0)
        else
            pressure[particle] = zero(pressure[particle])
        end
        # Calculate the average density error for the termination condition
        if (pressure[particle] != 0.0)
            new_density = a_ii[particle] * pressure[particle] + sum_term[particle] -
                          source_term + reference_density
            density_error[particle] = (new_density - reference_density)
        else
            density_error[particle] = zero(density_error[particle])
        end
    end
    relative_density_error = sum(density_error) / reference_density

    return relative_density_error
end
@propagate_inbounds function predicted_velocity(system::ImplicitIncompressibleSPHSystem,
                                                particle)
    return extract_svector(system.advection_velocity, system, particle)
end

@propagate_inbounds function d_ii(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.d_ii, system, particle)
end

@propagate_inbounds function sum_dij_pj(system::ImplicitIncompressibleSPHSystem, particle)
    return extract_svector(system.sum_d_ij_pj, system, particle)
end

@inline n_iisph_particles(system) = 0

@inline function n_iisph_particles(system::Union{ImplicitIncompressibleSPHSystem,
                                                 WallBoundarySystem{<:BoundaryModelDummyParticles{<:PressureBoundaries}}})
    return nparticles(system)
end

@inline maximum_iisph_error(system) = convert(eltype(system), Inf)

@inline function maximum_iisph_error(system::ImplicitIncompressibleSPHSystem)
    return system.max_error
end

@inline minimum_iisph_iterations(system) = 0

@inline function minimum_iisph_iterations(system::ImplicitIncompressibleSPHSystem)
    return system.min_iterations
end

@inline maximum_iisph_iterations(system) = typemax(Int)

@inline function maximum_iisph_iterations(system::ImplicitIncompressibleSPHSystem)
    return system.max_iterations
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(neighbor_system::ImplicitIncompressibleSPHSystem, m_b, rho_a,
                        grad_kernel, time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(neighbor_system::AbstractBoundarySystem, m_b, rho_a, grad_kernel,
                        time_step)
    return calculate_d_ii(neighbor_system, neighbor_system.boundary_model, m_b, rho_a,
                          grad_kernel, time_step)
end

# Calculates a summand for the calculation of the d_ii values
function calculate_d_ii(neighbor_system, boundary_model::BoundaryModelDummyParticles, m_b,
                        rho_a,
                        grad_kernel, time_step)
    return calculate_d_ii(neighbor_system, boundary_model,
                          boundary_model.density_calculator, m_b,
                          rho_a, grad_kernel, time_step)
end

# Calculates a summand for the calculation of the d_ii values (pressure zeroing, pressure
# extrapolation and pressure boundaries)
function calculate_d_ii(neighbor_system, boundary_model, density_calculator, m_b,
                        rho_a, grad_kernel, time_step)
    return -time_step^2 * m_b / rho_a^2 * grad_kernel
end

# Calculates a summand for the calculation of the d_ii values (pressure mirroring)
function calculate_d_ii(neighbor_system, boundary_model,
                        density_calculator::PressureMirroring, m_b,
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

# Calculates the d_ij value for a particle `i` and its neighbor `j` from eq. 9 in Ihmsen et al. (2013).
# Note that `i` is only implicitly included in the kernel gradient.
function calculate_d_ij(system::ImplicitIncompressibleSPHSystem,
                        neighbor_system::ImplicitIncompressibleSPHSystem, particle_j,
                        grad_kernel, time_step)
    # (delta t)^2 * m_i / rho_i ^2 * gradW_ij
    return -time_step^2 * hydrodynamic_mass(neighbor_system, particle_j) /
           neighbor_system.density[particle_j]^2 * grad_kernel
end

# Calculates the d_ij value for a particle `i` and its neighbor `j` from eq. 9 in Ihmsen et al. (2013).
# Note that `i` is only implicitly included in the kernel gradient.
function calculate_d_ij(system::ImplicitIncompressibleSPHSystem,
                        neighbor_system::AbstractBoundarySystem,
                        particle_j, grad_kernel, time_step)
    # (delta t)^2 * m_i / rho_i ^2 * gradW_ij
    # TODO This doesn't work for the boundary density calculator `ContinuityDensity`
    return -time_step^2 * hydrodynamic_mass(neighbor_system, particle_j) /
           current_density(nothing, neighbor_system, particle_j)^2 * grad_kernel
end

# Calculates the d_ji value for a particle `i` and its neighbor `j` from eq. 9 in Ihmsen et al. (2013).
# Note that `i` is only implicitly included in the kernel gradient.
function calculate_d_ji(system, neighbor_system,
                        particle_i, grad_kernel, time_step)
    # TODO This doesn't work for the boundary density calculator `ContinuityDensity`
    return -time_step^2 * hydrodynamic_mass(system, particle_i) /
           current_density(nothing, system)[particle_i]^2 * grad_kernel
end

# Calculate the large sum in eq. 13 of Ihmsen et al. (2013) for each particle (as `sum_term`)
function calculate_sum_term(system::ImplicitIncompressibleSPHSystem,
                            neighbor_system::ImplicitIncompressibleSPHSystem,
                            particle, neighbor, grad_kernel, time_step)
    pressure_system = system.pressure
    pressure_neighbor = neighbor_system.pressure
    m_j = hydrodynamic_mass(neighbor_system, neighbor)
    sum_dik_pk = sum_dij_pj(system, particle)
    d_jj = d_ii(neighbor_system, neighbor)
    p_i = pressure_system[particle]
    p_j = pressure_neighbor[neighbor]
    sum_djk_pk = sum_dij_pj(neighbor_system, neighbor)
    d_ji = calculate_d_ji(system, neighbor_system, particle, -grad_kernel, time_step)

    # Equation 13 of Ihmsen et al. (2013):
    # m_j * (\sum_k d_ik * p_k - d_jj * p_j - \sum_{k != i} d_jk * p_k) * ∇W_ij
    return m_j * dot(sum_dik_pk - d_jj * p_j - (sum_djk_pk - d_ji * p_i), grad_kernel)
end

function calculate_sum_term(system::ImplicitIncompressibleSPHSystem,
                            neighbor_system::AbstractBoundarySystem,
                            particle, neighbor, grad_kernel, time_step)
    sum_dik_pk = sum_dij_pj(system, particle)
    m_j = hydrodynamic_mass(neighbor_system, neighbor)

    # Equation 16 of Ihmsen et al. (2013):
    # m_j * sum_k d_ik * p_k * ∇W_ij
    return m_j * dot(sum_dik_pk, grad_kernel)
end

function iisph_source_term(system::ImplicitIncompressibleSPHSystem, particle)
    (; reference_density, predicted_density) = system

    return reference_density - predicted_density[particle]
end

function check_configuration(system::ImplicitIncompressibleSPHSystem, systems, nhs)
    (; time_step, omega) = system
    foreach_system(systems) do neighbor
        if neighbor isa WeaklyCompressibleSPHSystem
            throw(ArgumentError("`ImplicitIncompressibleSPHSystem` cannot be used together with
            `WeaklyCompressibleSPHSystem`"))
        end
        if neighbor isa WallBoundarySystem
            if (neighbor.boundary_model isa BoundaryModelDummyParticles &&
                neighbor.boundary_model.density_calculator isa PressureBoundaries)
                time_step_boundary = neighbor.boundary_model.density_calculator.time_step
                omega_boundary = neighbor.boundary_model.density_calculator.omega
                if !(time_step==time_step_boundary && omega==omega_boundary)
                    throw(ArgumentError("`PressureBoundaries` parameters have to be the same as the
                    `ImplicitIncompressibleSPHSystem` parameters"))
                end
            end
        end
    end
end
