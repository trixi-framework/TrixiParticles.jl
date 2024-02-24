@doc raw"""
    EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                smoothing_length, sound_speed;
                                pressure_acceleration=inter_particle_averaged_pressure,
                                density_calculator=SummationDensity(),
                                alpha=0.5, viscosity=NoViscosity(),
                                acceleration=ntuple(_ -> 0.0, NDIMS),
                                source_terms=nothing)

Entropically damped artiﬁcial compressibility (EDAC) for SPH introduced by (Ramachandran 2019).
As opposed to the weakly compressible SPH scheme, which uses an equation of state
(see [`WeaklyCompressibleSPHSystem`](@ref)), this scheme uses a pressure evolution equation
(PEE) to calculate the pressure. This equation is similar to the continuity equation (see
[`ContinuityDensity`](@ref)), but also contains a pressure damping term, which reduces
oscillations and is discretized as
```math
\frac{\mathrm{d} p_a}{\mathrm{d}t} = \sum_{b} m_b \frac{\rho_a}{\rho_b} c_s^2 v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h) +
\frac{V_a^2 + V_b^2}{m_a} \tilde{\eta}_{ab} \frac{p_{ab}}{\Vert r_{ab}^2 \Vert + \eta h_{ab}^2} \nabla_{r_a}
W(\Vert r_a - r_b \Vert, h) \cdot r_{ab},
```
where ``\rho_a``, ``\rho_b``,  ``r_a``, ``r_b``, ``V_a`` and
``V_b`` denote the density, coordinates and volume of particles ``a`` and ``b`` respectively, ``c_s``
is the speed of sound and ``v_{ab} = v_a - v_b`` and ``p_{ab}= p_a -p_b`` is the difference
in the velocity and pressure between particles ``a`` and ``b`` respectively.

The viscosity parameter ``\eta_a`` for a particle ``a`` is given as
```math
\eta_a = \rho_a \frac{\alpha h c_s}{8}
```
where it is found in the numerical experiments of (Ramachandran 2019) that ``\alpha = 0.5``
is a good choice for a wide range of Reynolds numbers (0.0125 to 10000).

# Arguments
- `initial_condition`:  Initial condition representing the system's particles.
- `sound_speed`:        Speed of sound.
- `smoothing_kernel`:   Smoothing kernel to be used for this system.
                        See [`SmoothingKernel`](@ref).
- `smoothing_length`:   Smoothing length to be used for this system.
                        See [`SmoothingKernel`](@ref).

# Keyword Arguments
- `viscosity`:      Viscosity model for this system (default: no viscosity).
                    Recommended: [`ViscosityAdami`](@ref).
- `acceleration`:   Acceleration vector for the system. (default: zero vector)
- `pressure_acceleration`: Pressure acceleration formulation (default: inter-particle averaged pressure).
                        When set to `nothing` the pressure acceleration formulation for the
                        corresponding [density calculator](@ref density_calculator) is chosen.
- `density_calculator`: [Density calculator](@ref density_calculator) (default: [`SummationDensity`](@ref))
- `source_terms`:   Additional source terms for this system. Has to be either `nothing`
                    (by default), or a function of `(coords, velocity, density, pressure)`
                    (which are the quantities of a single particle), returning a `Tuple`
                    or `SVector` that is to be added to the acceleration of that particle.
                    See, for example, [`SourceTermDamping`](@ref).
                    Note that these source terms will not be used in the calculation of the
                    boundary pressure when using a boundary with
                    [`BoundaryModelDummyParticles`](@ref) and [`AdamiPressureExtrapolation`](@ref).
                    The keyword argument `acceleration` should be used instead for
                    gravity-like source terms.

# References:
- Prabhu Ramachandran. "Entropically damped artiﬁcial compressibility for SPH".
  In: Computers and Fluids 179 (2019), pages 579-594.
  [doi: 10.1016/j.compfluid.2018.11.023](https://doi.org/10.1016/j.compfluid.2018.11.023)
"""
struct EntropicallyDampedSPHSystem{NDIMS, ELTYPE <: Real, DC, K, V,
                                   PF, ST, C} <: FluidSystem{NDIMS}
    initial_condition                 :: InitialCondition{ELTYPE}
    mass                              :: Array{ELTYPE, 1} # [particle]
    density_calculator                :: DC
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
    sound_speed                       :: ELTYPE
    viscosity                         :: V
    nu_edac                           :: ELTYPE
    acceleration                      :: SVector{NDIMS, ELTYPE}
    correction                        :: Nothing
    pressure_acceleration_formulation :: PF
    source_terms                      :: ST
    cache                             :: C

    function EntropicallyDampedSPHSystem(initial_condition, smoothing_kernel,
                                         smoothing_length, sound_speed;
                                         pressure_acceleration=inter_particle_averaged_pressure,
                                         alpha=0.5, viscosity=NoViscosity(),
                                         density_calculator=SummationDensity(),
                                         acceleration=ntuple(_ -> 0.0,
                                                             ndims(smoothing_kernel)),
                                         source_terms=nothing)
        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        mass = copy(initial_condition.mass)

        if ndims(smoothing_kernel) != NDIMS
            throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
        end

        acceleration_ = SVector(acceleration...)
        if length(acceleration_) != NDIMS
            throw(ArgumentError("`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"))
        end

        pressure_acceleration = choose_pressure_acceleration_formulation(pressure_acceleration,
                                                                         density_calculator,
                                                                         NDIMS, ELTYPE,
                                                                         nothing)

        nu_edac = (alpha * smoothing_length * sound_speed) / 8

        cache = create_cache_density(initial_condition, density_calculator)

        new{NDIMS, ELTYPE, typeof(density_calculator), typeof(smoothing_kernel),
            typeof(viscosity), typeof(pressure_acceleration), typeof(source_terms),
            typeof(cache)}(initial_condition, mass, density_calculator, smoothing_kernel,
                           smoothing_length, sound_speed, viscosity, nu_edac, acceleration_,
                           nothing, pressure_acceleration, source_terms, cache)
    end
end

function Base.show(io::IO, system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "EntropicallyDampedSPHSystem{", ndims(system), "}(")
    print(io, system.density_calculator)
    print(io,  ", ", system.viscosity)
    print(io, ", ", system.smoothing_kernel)
    print(io, ", ", system.acceleration)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::EntropicallyDampedSPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "EntropicallyDampedSPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "density calculator",
                     system.density_calculator |> typeof |> nameof)
        summary_line(io, "viscosity", system.viscosity |> typeof |> nameof)
        summary_line(io, "ν₍EDAC₎", "≈ $(round(system.nu_edac; digits=3))")
        summary_line(io, "smoothing kernel", system.smoothing_kernel |> typeof |> nameof)
        summary_line(io, "acceleration", system.acceleration)
        summary_footer(io)
    end
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem)
    return v_nvariables(system, system.density_calculator)
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, density_calculator)
    return ndims(system) + 1
end

@inline function v_nvariables(system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    return ndims(system) + 2
end

@inline function particle_density(v, ::ContinuityDensity,
                                  system::EntropicallyDampedSPHSystem, particle)
    return v[end - 1, particle]
end

@inline function particle_pressure(v, system::EntropicallyDampedSPHSystem, particle)
    return v[end, particle]
end

function update_quantities!(system::EntropicallyDampedSPHSystem, v, u,
                            v_ode, u_ode, semi, t)
    compute_density!(system, u, u_ode, semi, system.density_calculator)
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, density_calculator)
    for particle in eachparticle(system)
        v0[end, particle] = system.initial_condition.pressure[particle]
    end

    return v0
end

function write_v0!(v0, system::EntropicallyDampedSPHSystem, ::ContinuityDensity)
    for particle in eachparticle(system)
        v0[end - 1, particle] = system.initial_condition.density[particle]
        v0[end, particle] = system.initial_condition.pressure[particle]
    end

    return v0
end

function restart_with!(system::EntropicallyDampedSPHSystem, v, u)
    for particle in each_moving_particle(system)
        system.initial_condition.coordinates[:, particle] .= u[:, particle]
        system.initial_condition.velocity[:, particle] .= v[1:ndims(system), particle]
        system.initial_condition.pressure[particle] = v[end, particle]
    end
end
