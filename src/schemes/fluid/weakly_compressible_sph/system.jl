"""
    WeaklyCompressibleSPHSystem(initial_condition,
                                density_calculator, state_equation,
                                smoothing_kernel, smoothing_length;
                                viscosity=nothing, density_diffusion=nothing,
                                acceleration=ntuple(_ -> 0.0, NDIMS),
                                buffer_size=nothing,
                                correction=nothing, source_terms=nothing)

System for particles of a fluid.
The weakly compressible SPH (WCSPH) scheme is used, wherein a stiff equation of state
generates large pressure changes for small density variations.
See [Weakly Compressible SPH](@ref wcsph) for more details on the method.

# Arguments
- `initial_condition`:  [`InitialCondition`](@ref) representing the system's particles.
- `density_calculator`: Density calculator for the system.
                        See [`ContinuityDensity`](@ref) and [`SummationDensity`](@ref).
- `state_equation`:     Equation of state for the system. See [`StateEquationCole`](@ref).
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [Smoothing Kernels](@ref smoothing_kernel).

# Keyword Arguments
- `viscosity`:      Viscosity model for this system (default: no viscosity).
                    See [`ArtificialViscosityMonaghan`](@ref) or [`ViscosityAdami`](@ref).
- `density_diffusion`: Density diffusion terms for this system. See [`DensityDiffusion`](@ref).
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `buffer_size`:    Number of buffer particles.
                    This is needed when simulating with [`OpenBoundarySPHSystem`](@ref).
- `correction`:     Correction method used for this system. (default: no correction, see [Corrections](@ref corrections))
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure, t)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).
                    Note that these source terms will not be used in the calculation of the
                    boundary pressure when using a boundary with
                    [`BoundaryModelDummyParticles`](@ref) and [`AdamiPressureExtrapolation`](@ref).
                    The keyword argument `acceleration` should be used instead for
                    gravity-like source terms.
- `surface_tension`:   Surface tension model used for this SPH system. (default: no surface tension)


"""
struct WeaklyCompressibleSPHSystem{NDIMS, ELTYPE <: Real, IC, MA, P, DC, SE, K,
                                   V, DD, COR, PF, ST, B, SRFT, C} <: FluidSystem{NDIMS, IC}
    initial_condition                 :: IC
    mass                              :: MA     # Array{ELTYPE, 1}
    pressure                          :: P      # Array{ELTYPE, 1}
    density_calculator                :: DC
    state_equation                    :: SE
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
    acceleration                      :: SVector{NDIMS, ELTYPE}
    viscosity                         :: V
    density_diffusion                 :: DD
    correction                        :: COR
    pressure_acceleration_formulation :: PF
    transport_velocity                :: Nothing # TODO
    source_terms                      :: ST
    surface_tension                   :: SRFT
    buffer                            :: B
    cache                             :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function WeaklyCompressibleSPHSystem(initial_condition,
                                     density_calculator, state_equation,
                                     smoothing_kernel, smoothing_length;
                                     pressure_acceleration=nothing,
                                     buffer_size=nothing,
                                     viscosity=nothing, density_diffusion=nothing,
                                     acceleration=ntuple(_ -> 0.0,
                                                         ndims(smoothing_kernel)),
                                     correction=nothing, source_terms=nothing,
                                     surface_tension=nothing)
    buffer = isnothing(buffer_size) ? nothing :
             SystemBuffer(nparticles(initial_condition), buffer_size)

    initial_condition = allocate_buffer(initial_condition, buffer)

    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)
    n_particles = nparticles(initial_condition)

    mass = copy(initial_condition.mass)
    pressure = similar(initial_condition.pressure)

    if ndims(smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    # Make acceleration an SVector
    acceleration_ = SVector(acceleration...)
    if length(acceleration_) != NDIMS
        throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    if correction isa ShepardKernelCorrection &&
       density_calculator isa ContinuityDensity
        throw(ArgumentError("`ShepardKernelCorrection` cannot be used with `ContinuityDensity`"))
    end

    pressure_acceleration = choose_pressure_acceleration_formulation(pressure_acceleration,
                                                                     density_calculator,
                                                                     NDIMS, ELTYPE,
                                                                     correction)

    cache = create_cache_density(initial_condition, density_calculator)
    cache = (;
             create_cache_wcsph(correction, initial_condition.density, NDIMS,
                                n_particles)..., cache...)
    cache = (;
             create_cache_wcsph(surface_tension, ELTYPE, NDIMS, n_particles)...,
             cache...)

    return WeaklyCompressibleSPHSystem(initial_condition, mass, pressure,
                                       density_calculator, state_equation,
                                       smoothing_kernel, smoothing_length,
                                       acceleration_, viscosity,
                                       density_diffusion, correction,
                                       pressure_acceleration, nothing,
                                       source_terms, surface_tension, buffer, cache)
end

create_cache_wcsph(correction, density, NDIMS, nparticles) = (;)

function create_cache_wcsph(::ShepardKernelCorrection, density, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density))
end

function create_cache_wcsph(::KernelCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{Float64}(undef, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density), dw_gamma)
end

function create_cache_wcsph(::Union{GradientCorrection, BlendedGradientCorrection}, density,
                            NDIMS, n_particles)
    correction_matrix = Array{Float64, 3}(undef, NDIMS, NDIMS, n_particles)
    return (; correction_matrix)
end

function create_cache_wcsph(::MixedKernelGradientCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{Float64}(undef, NDIMS, n_particles)
    correction_matrix = Array{Float64, 3}(undef, NDIMS, NDIMS, n_particles)

    return (; kernel_correction_coefficient=similar(density), dw_gamma, correction_matrix)
end

function create_cache_wcsph(::SurfaceTensionAkinci, ELTYPE, NDIMS, nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    return (; surface_normal)
end

function Base.show(io::IO, system::WeaklyCompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "WeaklyCompressibleSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
    print(io, ", ", system.correction)
    print(io, ", ", system.state_equation)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.viscosity)
    print(io, ", ", system.density_diffusion)
    print(io, ", ", system.surface_tension)
    print(io, ", ", system.acceleration)
    print(io, ", ", system.source_terms)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::WeaklyCompressibleSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "WeaklyCompressibleSPHSystem{$(ndims(system))}")
        if system.buffer isa SystemBuffer
            summary_line(io, "#particles", nparticles(system))
            summary_line(io, "#buffer_particles", system.buffer.buffer_size)
        else
            summary_line(io, "#particles", nparticles(system))
        end
        summary_line(io, "density calculator",
                     system.density_calculator |> typeof |> nameof)
        summary_line(io, "correction method",
                     system.correction |> typeof |> nameof)
        summary_line(io, "state equation", system.state_equation |> typeof |> nameof)
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity)
        summary_line(io, "density diffusion", system.density_diffusion)
        summary_line(io, "surface tension", system.surface_tension)
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "source terms", system.source_terms |> typeof |> nameof)
        summary_footer(io)
    end
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, density_calculator)
    return ndims(system)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, ::ContinuityDensity)
    return ndims(system) + 1
end

@propagate_inbounds function particle_pressure(v, system::WeaklyCompressibleSPHSystem,
                                               particle)
    return system.pressure[particle]
end

@inline system_sound_speed(system::WeaklyCompressibleSPHSystem) = system.state_equation.sound_speed

function update_quantities!(system::WeaklyCompressibleSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    (; density_calculator, density_diffusion, correction) = system

    compute_density!(system, u, u_ode, semi, density_calculator)

    nhs = get_neighborhood_search(system, semi)
    @trixi_timeit timer() "update density diffusion" update!(density_diffusion, nhs, v, u,
                                                             system, semi)

    return system
end

function update_pressure!(system::WeaklyCompressibleSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; density_calculator, correction, surface_tension) = system

    compute_correction_values!(system, correction, u, v_ode, u_ode, semi)

    compute_gradient_correction_matrix!(correction, system, u, v_ode, u_ode, semi)

    # `kernel_correct_density!` only performed for `SummationDensity`
    kernel_correct_density!(system, v, u, v_ode, u_ode, semi, correction,
                            density_calculator)
    compute_pressure!(system, v)
    compute_surface_normal!(system, surface_tension, v, u, v_ode, u_ode, semi, t)

    return system
end

function kernel_correct_density!(system::WeaklyCompressibleSPHSystem, v, u, v_ode, u_ode,
                                 semi, correction, density_calculator)
    return system
end

function kernel_correct_density!(system::WeaklyCompressibleSPHSystem, v, u, v_ode, u_ode,
                                 semi, corr::ShepardKernelCorrection, ::SummationDensity)
    system.cache.density ./= system.cache.kernel_correction_coefficient
end

function compute_gradient_correction_matrix!(correction,
                                             system::WeaklyCompressibleSPHSystem, u,
                                             v_ode, u_ode, semi)
    return system
end

function compute_gradient_correction_matrix!(corr::Union{GradientCorrection,
                                                         BlendedGradientCorrection,
                                                         MixedKernelGradientCorrection},
                                             system::WeaklyCompressibleSPHSystem, u,
                                             v_ode, u_ode, semi)
    (; cache, correction, smoothing_kernel, smoothing_length) = system
    (; correction_matrix) = cache

    system_coords = current_coordinates(u, system)

    compute_gradient_correction_matrix!(correction_matrix, system, system_coords,
                                        v_ode, u_ode, semi, correction, smoothing_length,
                                        smoothing_kernel)
end

function reinit_density!(vu_ode, semi)
    v_ode, u_ode = vu_ode.x

    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        reinit_density!(system, v, u, v_ode, u_ode, semi)
    end

    return vu_ode
end

function reinit_density!(system::WeaklyCompressibleSPHSystem, v, u,
                         v_ode, u_ode, semi)
    # Compute density with `SummationDensity` and store the result in `v`,
    # overwriting the previous integrated density.
    summation_density!(system, semi, u, u_ode, v[end, :])

    # Apply `ShepardKernelCorrection`
    kernel_correction_coefficient = zeros(size(v[end, :]))
    compute_shepard_coeff!(system, current_coordinates(u, system), v_ode, u_ode, semi,
                           kernel_correction_coefficient)
    v[end, :] ./= kernel_correction_coefficient

    compute_pressure!(system, v)

    return system
end

function reinit_density!(system, v, u, v_ode, u_ode, semi)
    return system
end

function compute_pressure!(system, v)
    @threaded system for particle in eachparticle(system)
        apply_state_equation!(system, particle_density(v, system, particle), particle)
    end
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function apply_state_equation!(system::WeaklyCompressibleSPHSystem, density,
                                       particle)
    system.pressure[particle] = system.state_equation(density)
end

function write_v0!(v0, system::WeaklyCompressibleSPHSystem, ::ContinuityDensity)
    # Note that `.=` is very slightly faster, but not GPU-compatible
    v0[end, :] = system.initial_condition.density

    return v0
end

function restart_with!(system::WeaklyCompressibleSPHSystem, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
    end

    restart_with!(system, system.density_calculator, v, u)
end

function restart_with!(system, ::SummationDensity, v, u)
    return system
end

function restart_with!(system, ::ContinuityDensity, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.density[particle] = v[end, particle]
    end

    return system
end

@inline function correction_matrix(system::WeaklyCompressibleSPHSystem, particle)
    extract_smatrix(system.cache.correction_matrix, system, particle)
end

function compute_surface_normal!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_surface_normal!(system, surface_tension::SurfaceTensionAkinci, v, u, v_ode,
                                 u_ode, semi, t)
    (; cache) = system

    # Reset surface normal
    set_zero!(cache.surface_normal)

    @trixi_timeit timer() "compute surface normal" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
        nhs = get_neighborhood_search(system, semi)

        calc_normal_akinci!(system, neighbor_system, surface_tension, u, v_neighbor_system,
                            u_neighbor_system, nhs)
    end
    return system
end

@inline function surface_normal(::SurfaceTensionAkinci, particle_system::FluidSystem,
                                particle)
    (; cache) = particle_system
    return extract_svector(cache.surface_normal, particle_system, particle)
end

@inline function surface_tension_model(system::WeaklyCompressibleSPHSystem)
    return system.surface_tension
end

@inline function surface_tension_model(system)
    return nothing
end
