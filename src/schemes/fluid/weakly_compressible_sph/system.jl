"""
    WeaklyCompressibleSPHSystem(initial_condition,
                                density_calculator, state_equation,
                                smoothing_kernel, smoothing_length;
                                acceleration=ntuple(_ -> 0.0, NDIMS),
                                viscosity=nothing, density_diffusion=nothing,
                                pressure_acceleration=nothing,
                                shifting_technique=nothing,
                                buffer_size=nothing,
                                correction=nothing, source_terms=nothing,
                                surface_tension=nothing, surface_normal_method=nothing,
                                reference_particle_spacing=0.0))

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

# Keywords
- `acceleration`:               Acceleration vector for the system. (default: zero vector)
- `viscosity`:                  Viscosity model for this system (default: no viscosity).
                                See [`ArtificialViscosityMonaghan`](@ref) or [`ViscosityAdami`](@ref).
- `density_diffusion`:          Density diffusion terms for this system.
                                See [`AbstractDensityDiffusion`](@ref TrixiParticles.AbstractDensityDiffusion).
- `pressure_acceleration`:      Pressure acceleration formulation for this system.
                                By default, the correct formulation is chosen based on the
                                density calculator and the correction method.
                                To use [Tensile Instability Control](@ref tic), pass
                                [`tensile_instability_control`](@ref) here.
- `shifting_technique`:         [Shifting technique](@ref shifting) or [transport velocity
                                formulation](@ref transport_velocity_formulation) to use
                                with this system. Default is no shifting.
- `buffer_size`:                Number of buffer particles.
                                This is needed when simulating with [`OpenBoundarySystem`](@ref).
- `correction`:                 Correction method used for this system. (default: no correction, see [Corrections](@ref corrections))
- `source_terms`:               Additional source terms for this system. Has to be either `nothing`
                                (by default), or a function of `(coords, velocity, density, pressure, t)`
                                (which are the quantities of a single particle), returning a `Tuple`
                                or `SVector` that is to be added to the acceleration of that particle.
                                See, for example, [`SourceTermDamping`](@ref).
                                Note that these source terms will not be used in the calculation of the
                                boundary pressure when using a boundary with
                                [`BoundaryModelDummyParticles`](@ref) and [`AdamiPressureExtrapolation`](@ref).
                                The keyword argument `acceleration` should be used instead for
                                gravity-like source terms.
- `surface_tension`:            Surface tension model used for this SPH system. (default: no surface tension)
- `surface_normal_method`:      The surface normal method to be used for this SPH system.
                                (default: no surface normal method or `ColorfieldSurfaceNormal()` if a surface_tension model is used)
- `reference_particle_spacing`: The reference particle spacing used for weighting values at the boundary,
                                which currently is only needed when using surface tension.
- `color_value`:                The value used to initialize the color of particles in the system.
"""
struct WeaklyCompressibleSPHSystem{NDIMS, ELTYPE <: Real, IC, MA, P, DC, SE, K, V, DD, COR,
                                   PF, SC, ST, B, SRFT, SRFN, PR,
                                   C} <: AbstractFluidSystem{NDIMS}
    initial_condition                 :: IC
    mass                              :: MA     # Array{ELTYPE, 1}
    pressure                          :: P      # Array{ELTYPE, 1}
    density_calculator                :: DC
    state_equation                    :: SE
    smoothing_kernel                  :: K
    acceleration                      :: SVector{NDIMS, ELTYPE}
    viscosity                         :: V
    density_diffusion                 :: DD
    correction                        :: COR
    pressure_acceleration_formulation :: PF
    shifting_technique                :: SC
    source_terms                      :: ST
    surface_tension                   :: SRFT
    surface_normal_method             :: SRFN
    buffer                            :: B
    particle_refinement               :: PR # TODO
    cache                             :: C
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function WeaklyCompressibleSPHSystem(initial_condition, density_calculator, state_equation,
                                     smoothing_kernel, smoothing_length;
                                     acceleration=ntuple(_ -> zero(eltype(initial_condition)),
                                                         ndims(smoothing_kernel)),
                                     viscosity=nothing, density_diffusion=nothing,
                                     pressure_acceleration=nothing,
                                     shifting_technique=nothing,
                                     buffer_size=nothing,
                                     correction=nothing, source_terms=nothing,
                                     surface_tension=nothing, surface_normal_method=nothing,
                                     reference_particle_spacing=0, color_value=1)
    buffer = isnothing(buffer_size) ? nothing :
             SystemBuffer(nparticles(initial_condition), buffer_size)

    particle_refinement = nothing # TODO

    initial_condition,
    density_diffusion = allocate_buffer(initial_condition,
                                        density_diffusion, buffer)

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

    if surface_tension !== nothing && surface_normal_method === nothing
        surface_normal_method = ColorfieldSurfaceNormal()
    end

    if surface_normal_method !== nothing && reference_particle_spacing < eps()
        throw(ArgumentError("`reference_particle_spacing` must be set to a positive value when using `ColorfieldSurfaceNormal` or a surface tension model"))
    end

    pressure_acceleration = choose_pressure_acceleration_formulation(pressure_acceleration,
                                                                     density_calculator,
                                                                     NDIMS, ELTYPE,
                                                                     correction)

    cache = (; create_cache_density(initial_condition, density_calculator)...,
             create_cache_correction(correction, initial_condition.density, NDIMS,
                                     n_particles)...,
             create_cache_surface_normal(surface_normal_method, ELTYPE, NDIMS,
                                         n_particles)...,
             create_cache_surface_tension(surface_tension, ELTYPE, NDIMS,
                                          n_particles)...,
             create_cache_refinement(initial_condition, particle_refinement,
                                     smoothing_length)...,
             create_cache_shifting(initial_condition, shifting_technique)...,
             color=Int(color_value))

    # If the `reference_density_spacing` is set calculate the `ideal_neighbor_count`
    if reference_particle_spacing > 0
        # `reference_particle_spacing` has to be set for surface normals to be determined
        cache = (;
                 cache...,  # Existing cache fields
                 reference_particle_spacing=reference_particle_spacing)
    end

    return WeaklyCompressibleSPHSystem(initial_condition, mass, pressure,
                                       density_calculator, state_equation,
                                       smoothing_kernel, acceleration_, viscosity,
                                       density_diffusion, correction, pressure_acceleration,
                                       shifting_technique, source_terms, surface_tension,
                                       surface_normal_method, buffer, particle_refinement,
                                       cache)
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
    print(io, ", ", system.shifting_technique)
    print(io, ", ", system.surface_tension)
    print(io, ", ", system.surface_normal_method)
    if system.surface_normal_method isa ColorfieldSurfaceNormal
        print(io, ", ", system.color)
    end
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
        summary_line(io, "shifting technique", system.shifting_technique)
        summary_line(io, "surface tension", system.surface_tension)
        summary_line(io, "surface normal method", system.surface_normal_method)
        if system.surface_normal_method isa ColorfieldSurfaceNormal
            summary_line(io, "color", system.cache.color)
        end
        summary_line(io, "acceleration", system.acceleration)
        summary_line(io, "source terms", system.source_terms |> typeof |> nameof)
        summary_footer(io)
    end
end

@inline Base.eltype(::WeaklyCompressibleSPHSystem{<:Any, ELTYPE}) where {ELTYPE} = ELTYPE

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, ::SummationDensity)
    return ndims(system)
end

@inline function v_nvariables(system::WeaklyCompressibleSPHSystem, ::ContinuityDensity)
    return ndims(system) + 1
end

@inline buffer(system::WeaklyCompressibleSPHSystem) = system.buffer

system_correction(system::WeaklyCompressibleSPHSystem) = system.correction

@propagate_inbounds function current_velocity(v, system::WeaklyCompressibleSPHSystem)
    return current_velocity(v, system.density_calculator, system)
end

@inline function current_velocity(v, ::SummationDensity,
                                  system::WeaklyCompressibleSPHSystem)
    # When using `SummationDensity`, `v` contains only the velocity
    return v
end

@propagate_inbounds function current_velocity(v, ::ContinuityDensity,
                                              system::WeaklyCompressibleSPHSystem)
    # When using `ContinuityDensity`, the velocity is stored
    # in the first `ndims(system)` rows of `v`.
    return view(v, 1:ndims(system), :)
end

@propagate_inbounds function current_density(v, system::WeaklyCompressibleSPHSystem)
    return current_density(v, system.density_calculator, system)
end

@inline function current_density(v, ::SummationDensity,
                                 system::WeaklyCompressibleSPHSystem)
    # When using `SummationDensity`, the density is stored in the cache
    return system.cache.density
end

@propagate_inbounds function current_density(v, ::ContinuityDensity,
                                             system::WeaklyCompressibleSPHSystem)
    # When using `ContinuityDensity`, the density is stored in the last row of `v`
    return view(v, size(v, 1), :)
end

@inline function current_pressure(v, system::WeaklyCompressibleSPHSystem)
    return system.pressure
end

@inline system_sound_speed(system::WeaklyCompressibleSPHSystem) = sound_speed(system.state_equation)

@inline shifting_technique(system::WeaklyCompressibleSPHSystem) = system.shifting_technique

@inline density_diffusion(system::WeaklyCompressibleSPHSystem) = system.density_diffusion

function update_quantities!(system::WeaklyCompressibleSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    (; density_calculator, density_diffusion, correction) = system

    # Update speed of sound when an adaptive state equation is used
    update_speed_of_sound!(system, v, system.state_equation)
    compute_density!(system, u, u_ode, semi, density_calculator)

    @trixi_timeit timer() "update density diffusion" update!(density_diffusion, v, u,
                                                             system, semi)

    return system
end

@inline update_speed_of_sound!(system, v, state_equation) = system

@inline function update_speed_of_sound!(system::WeaklyCompressibleSPHSystem, v,
                                        state_equation::StateEquationAdaptiveCole)
    # This has similar performance to `maximum(..., eachparticle(system))`,
    # but is GPU-compatible.
    v_max2 = maximum(x -> dot(x, x),
                     reinterpret(reshape, SVector{ndims(system), eltype(v)},
                                 current_velocity(v, system)))
    v_max = sqrt(v_max2)

    state_equation.sound_speed_ref[] = min(state_equation.max_sound_speed,
                                           max(state_equation.min_sound_speed,
                                               v_max /
                                               state_equation.mach_number_target))
    return system
end

function update_pressure!(system::WeaklyCompressibleSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; density_calculator, correction, surface_normal_method, surface_tension) = system

    compute_pressure!(system, v, semi)

    # These are only computed when using corrections
    compute_correction_values!(system, correction, u, v_ode, u_ode, semi)
    compute_gradient_correction_matrix!(correction, system, u, v_ode, u_ode, semi)
    # `kernel_correct_density!` only performed for `SummationDensity`
    kernel_correct_density!(system, v, u, v_ode, u_ode, semi, correction,
                            density_calculator)

    # These are only computed when using surface tension
    compute_surface_normal!(system, surface_normal_method, v, u, v_ode, u_ode, semi, t)
    compute_surface_delta_function!(system, surface_tension, semi)
    return system
end

function update_final!(system::WeaklyCompressibleSPHSystem, v, u, v_ode, u_ode, semi, t)
    (; surface_tension) = system

    # Surface normal of neighbor and boundary needs to have been calculated already
    compute_curvature!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    compute_stress_tensors!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    update_shifting!(system, shifting_technique(system), v, u, v_ode, u_ode, semi)
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
    (; cache, correction, smoothing_kernel) = system
    (; correction_matrix) = cache

    system_coords = current_coordinates(u, system)

    compute_gradient_correction_matrix!(correction_matrix, system, system_coords,
                                        v_ode, u_ode, semi, correction, smoothing_kernel)
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

    compute_pressure!(system, v, semi)

    return system
end

function reinit_density!(system, v, u, v_ode, u_ode, semi)
    return system
end

function compute_pressure!(system, v, semi)
    @threaded semi for particle in eachparticle(system)
        apply_state_equation!(system, current_density(v, system, particle), particle)
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
    for particle in each_integrated_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
    end

    restart_with!(system, system.density_calculator, v, u)
end

function restart_with!(system, ::SummationDensity, v, u)
    return system
end

function restart_with!(system, ::ContinuityDensity, v, u)
    for particle in each_integrated_particle(system)
        system.initial_condition.density[particle] = v[end, particle]
    end

    return system
end

@inline function correction_matrix(system::WeaklyCompressibleSPHSystem, particle)
    extract_smatrix(system.cache.correction_matrix, system, particle)
end

@inline function curvature(particle_system::AbstractFluidSystem, particle)
    (; cache) = particle_system
    return cache.curvature[particle]
end
