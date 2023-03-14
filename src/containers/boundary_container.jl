"""
    BoundaryParticleContainer(coordinates, mass, model;
                              movement_function=nothing)

Container for boundaries modeled by boundary particles.
The container is initialized with the coordinates of the particles and their masses.
The interaction between fluid and boundary particles is specified by the boundary model.

The `movement_function` is to define in which way the boundary particles move over time. Its
boolean return value is mandatory to determine in each timestep if the particles are moving or not.
This determines if the neighborhood search will be updated.
In the example below the `movement_function` only returns `true` (container is moving)
if the simulation time is lower than `0.1`.


# Examples
```julia
function movement_function(coordinates, t)

    if t < 0.1
        f(t) = 0.5*t^2 + t
        pos_1 = coordinates[2,1]
        pos_2 = f(t)
        diff_pos = pos_2 - pos_1
        coordinates[2,:] .+= diff_pos

        return true
    end

    return false
end
```
"""
struct BoundaryParticleContainer{NDIMS, ELTYPE <: Real, BM, MF} <: ParticleContainer{NDIMS}
    initial_coordinates :: Array{ELTYPE, 2}
    boundary_model      :: BM
    movement_function   :: MF
    ismoving            :: Vector{Bool}

    function BoundaryParticleContainer(coordinates, model; movement_function=nothing)
        NDIMS = size(coordinates, 1)
        ismoving = zeros(Bool, 1)

        return new{NDIMS, eltype(coordinates), typeof(model), typeof(movement_function)}(coordinates,
                                                                                         model,
                                                                                         movement_function,
                                                                                         ismoving)
    end
end

function Base.show(io::IO, container::BoundaryParticleContainer)
    @nospecialize container # reduce precompilation time

    print(io, "BoundaryParticleContainer{", ndims(container), "}(")
    print(io, container.boundary_model)
    print(io, ", ", container.movement_function)
    print(io, ") with ", nparticles(container), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", container::BoundaryParticleContainer)
    @nospecialize container # reduce precompilation time

    if get(io, :compact, false)
        show(io, container)
    else
        summary_header(io, "BoundaryParticleContainer{$(ndims(container))}")
        summary_line(io, "#particles", nparticles(container))
        summary_line(io, "boundary model", container.boundary_model)
        summary_line(io, "movement function", container.movement_function)
        summary_footer(io)
    end
end

@inline function boundary_particle_impact(particle, boundary_particle,
                                          v_particle_container, v_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b)
    @unpack boundary_model = boundary_container

    boundary_particle_impact(particle, boundary_particle,
                             v_particle_container, v_boundary_container,
                             particle_container, boundary_container,
                             pos_diff, distance, m_b, boundary_model)
end

@doc raw"""
    BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing)

Boundaries modeled as boundary particles which exert forces on the fluid particles (Monaghan, Kajtar, 2009).
The force on fluid particle ``a`` due to boundary particle ``b`` is given by
```math
f_{ab} = m_a \left(\tilde{f}_{ab} - m_b \Pi_{ab} \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)\right)
```
with
```math
\tilde{f}_{ab} = \frac{K}{\beta^{n-1}} \frac{r_{ab}}{\Vert r_{ab} \Vert (\Vert r_{ab} \Vert - d)} \Phi(\Vert r_{ab} \Vert, h)
\frac{2 m_b}{m_a + m_b},
```
where ``m_a`` and ``m_b`` are the masses of fluid particle ``a`` and boundary particle ``b``
respectively, ``r_{ab} = r_a - r_b`` is the difference of the coordinates of particles
``a`` and ``b``, ``d`` denotes the boundary particle spacing and ``n`` denotes the number of
dimensions (see (Monaghan, Kajtar, 2009, Equation (3.1)) and (Valizadeh, Monaghan, 2015)).
Note that the repulsive acceleration $\tilde{f}_{ab}$ does not depend on the masses of
the boundary particles.
Here, ``\Phi`` denotes the 1D Wendland C4 kernel, normalized to ``1.77`` for ``q=0``
(Monaghan, Kajtar, 2009, Section 4), with ``\Phi(r, h) = w(r/h)`` and
```math
w(q) =
\begin{cases}
  (1.77/32) (1 + (5/2)q + 2q^2)(2 - q)^5  & \text{if } 0 \leq q < 2 \\
  0                                       & \text{if } q \geq 2.
\end{cases}
```

The boundary particles are assumed to have uniform spacing by the factor ``\beta`` smaller
than the expected fluid particle spacing.
For example, if the fluid particles have an expected spacing of ``0.3`` and the boundary particles
have a uniform spacing of ``0.1``, then this parameter should be set to ``\beta = 3``.
According to (Monaghan, Kajtar, 2009), a value of ``\beta = 3`` for the Wendland C4 that
we use here is reasonable for most computing purposes.

The parameter ``K`` is used to scale the force exerted by the boundary particles.
In (Monaghan, Kajtar, 2009), a value of ``gD`` is used for static tank simulations,
where ``g`` is the gravitational acceleration and ``D`` is the depth of the fluid.

The viscosity ``\Pi_{ab}`` is calculated according to the viscosity used in the
simulation, where the density of the boundary particle if needed is assumed to be
identical to the density of the fluid particle.

## References:
- Joseph J. Monaghan, Jules B. Kajtar. "SPH particle boundary forces for arbitrary boundaries".
  In: Computer Physics Communications 180.10 (2009), pages 1811–1820.
  [doi: 10.1016/j.cpc.2009.05.008](https://doi.org/10.1016/j.cpc.2009.05.008)
- Alireza Valizadeh, Joseph J. Monaghan. "A study of solid wall models for weakly compressible SPH."
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)
"""
struct BoundaryModelMonaghanKajtar{ELTYPE <: Real, DC}
    K                         :: ELTYPE
    beta                      :: ELTYPE
    boundary_particle_spacing :: ELTYPE
    hydrodynamic_mass         :: Vector{ELTYPE}
    density_calculator        :: DC

    function BoundaryModelMonaghanKajtar(K, beta, boundary_particle_spacing, mass)
        # No density calculator for this model.
        # However, this field is mandatory for proper dispatching.
        density_calculator = nothing

        new{typeof(K), typeof(density_calculator)}(K, beta, boundary_particle_spacing, mass,
                                                   density_calculator)
    end
end

function Base.show(io::IO, model::BoundaryModelMonaghanKajtar)
    @nospecialize model # reduce precompilation time

    print(io, "BoundaryModelMonaghanKajtar")
end

@inline function boundary_particle_impact(particle, boundary_particle,
                                          v_particle_container, v_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b,
                                          boundary_model::BoundaryModelMonaghanKajtar)
    @unpack smoothing_length = particle_container
    @unpack K, beta, boundary_particle_spacing = boundary_model

    NDIMS = ndims(particle_container)
    return K / beta^(NDIMS - 1) * pos_diff /
           (distance * (distance - boundary_particle_spacing)) *
           boundary_kernel(distance, smoothing_length)
end

@inline function boundary_kernel(r, h)
    q = r / h

    # TODO The neighborhood search fluid->boundary should use this search distance
    if q >= 2
        return 0.0
    end

    # (Monaghan, Kajtar, 2009, Section 4): The kernel should be normalized to 1.77 for q=0
    return 1.77 / 32 * (1 + 5 / 2 * q + 2 * q^2) * (2 - q)^5
end

@doc raw"""
    BoundaryModelDummyParticles(initial_density, hydrodynamic_mass, state_equation,
                                density_calculator, smoothing_kernel, smoothing_length)

Boundaries modeled as dummy particles, which are treated like fluid particles,
but their positions and velocities are not evolved in time. Since the force towards the fluid
should not change with the material density when used with a `SolidParticleContainer`, the
dummy particles need to have a mass corresponding to the fluid's rest density, which we call
"hydrodynamic mass", as opposed to mass corresponding to the material density of a
`SolidParticleContainer`.

Here, `initial_density` and `hydrodynamic_mass` are vectors that contains the initial density
and the hydrodynamic mass respectively for each boundary particle.
Note that when used with [`SummationDensity`](@ref) (see below), this is only used to determine
the element type and the number of boundary particles.

To establish a relationship between density and pressure, a `state_equation` has to be passed,
which should be the same as for the adjacent fluid containers.
To sum over neighboring particles, a `smoothing_kernel` and `smoothing_length` needs to be passed.
This should be the same as for the adjacent fluid container with the largest smoothing length.

In the literature, this kind of boundary particles is referred to as
"dummy particles" (Adami et al., 2012 and Valizadeh & Monaghan, 2015),
"frozen fluid particles" (Akinci et al., 2012) or "dynamic boundaries (Crespo et al., 2007).
The key detail of this boundary condition and the only difference between the boundary models
in these references is the way the density and pressure of boundary particles is computed.

Since boundary particles are treated like fluid particles, the force
on fluid particle ``a`` due to boundary particle ``b`` is given by
```math
f_{ab} = m_a m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_{r_a} W(\Vert r_a - r_b \Vert, h).
```
The quantities to be defined here are the density ``\rho_b`` and pressure ``p_b``
of the boundary particle ``b``.

We provide three options to compute the boundary density and pressure, determined by the `density_calculator`:
1. With [`SummationDensity`](@ref), the density is calculated by summation over the neighboring particles,
   and the pressure is computed from the density with the state equation.
2. With [`ContinuityDensity`](@ref), the density is integrated from the continuity equation,
   and the pressure is computed from the density with the state equation.
   Note that this causes a gap between fluid and boundary where the boundary is initialized
   without any contact to the fluid. This is due to overestimation of the boundary density
   as soon as the fluid comes in contact with boundary particles that initially did not have
   contact to the fluid.
   Therefore, in dam break simulations, there is a visible "step", even though the boundary is supposed to be flat.
   See also [dual.sphysics.org/faq/#Q_13](https://dual.sphysics.org/faq/#Q_13).
3. With [`AdamiPressureExtrapolation`](@ref), the pressure is extrapolated from the pressure of the
   fluid according to (Adami et al., 2012), and the density is obtained by applying the inverse of the state equation.

## References:
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)
- Alireza Valizadeh, Joseph J. Monaghan.
  "A study of solid wall models for weakly compressible SPH".
  In: Journal of Computational Physics 300 (2015), pages 5–19.
  [doi: 10.1016/J.JCP.2015.07.033](https://doi.org/10.1016/J.JCP.2015.07.033)
- Nadir Akinci, Markus Ihmsen, Gizem Akinci, Barbara Solenthaler, Matthias Teschner.
  "Versatile rigid-fluid coupling for incompressible SPH".
  ACM Transactions on Graphics 31, 4 (2012), pages 1–8.
  [doi: 10.1145/2185520.2185558](https://doi.org/10.1145/2185520.2185558)
- A. J. C. Crespo, M. Gómez-Gesteira, R. A. Dalrymple.
  "Boundary conditions generated by dynamic particles in SPH methods"
  In: Computers, Materials and Continua 5 (2007), pages 173-184.
  [doi: 10.3970/cmc.2007.005.173](https://doi.org/10.3970/cmc.2007.005.173)
"""
struct BoundaryModelDummyParticles{ELTYPE <: Real, SE, DC, K, C}
    pressure           :: Vector{ELTYPE}
    hydrodynamic_mass  :: Vector{ELTYPE}
    state_equation     :: SE
    density_calculator :: DC
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    cache              :: C

    function BoundaryModelDummyParticles(initial_density, hydrodynamic_mass, state_equation,
                                         density_calculator, smoothing_kernel,
                                         smoothing_length)
        pressure = similar(initial_density)

        cache = create_cache(initial_density, density_calculator)

        new{eltype(initial_density), typeof(state_equation),
            typeof(density_calculator), typeof(smoothing_kernel),
            typeof(cache)}(pressure, hydrodynamic_mass, state_equation, density_calculator,
                           smoothing_kernel, smoothing_length, cache)
    end
end

function Base.show(io::IO, model::BoundaryModelDummyParticles)
    @nospecialize model # reduce precompilation time

    print(io, "BoundaryModelDummyParticles(")
    print(io, model.density_calculator |> typeof |> nameof)
    print(io, ")")
end

@inline function boundary_particle_impact(particle, boundary_particle,
                                          v_particle_container, v_boundary_container,
                                          particle_container, boundary_container,
                                          pos_diff, distance, m_b,
                                          boundary_model::BoundaryModelDummyParticles)
    @unpack smoothing_kernel, smoothing_length = particle_container

    density_particle = get_particle_density(particle, v_particle_container,
                                            particle_container)
    density_boundary_particle = get_particle_density(boundary_particle,
                                                     v_boundary_container,
                                                     boundary_container)

    grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff /
                  distance

    return -m_b *
           (particle_container.pressure[particle] / density_particle^2 +
            boundary_model.pressure[boundary_particle] / density_boundary_particle^2) *
           grad_kernel
end

@doc raw"""
    AdamiPressureExtrapolation()

The pressure of the boundary particles is obtained by extrapolating the pressure of the fluid
according to (Adami et al., 2012).
The pressure of a boundary particle ``b`` is given by
```math
p_b = \frac{\sum_f (p_f + \rho_f (\bm{g} - \bm{a}_b) \cdot \bm{r}_{bf}) W(\Vert r_{bf} \Vert, h)}{\sum_f W(\Vert r_{bf} \Vert, h)},
```
where the sum is over all fluid particles, ``\rho_f`` and ``p_f`` denote the density and pressure of fluid particle ``f``, respectively,
``r_{bf} = r_b - r_f`` denotes the difference of the coordinates of particles ``b`` and ``f``,
``\bm{g}`` denotes the gravitational acceleration acting on the fluid, and ``\bm{a}_b`` denotes the acceleration of the boundary particle ``b``.

## References:
- S. Adami, X. Y. Hu, N. A. Adams.
  "A generalized wall boundary condition for smoothed particle hydrodynamics".
  In: Journal of Computational Physics 231, 21 (2012), pages 7057–7075.
  [doi: 10.1016/J.JCP.2012.05.005](https://doi.org/10.1016/J.JCP.2012.05.005)
"""
struct AdamiPressureExtrapolation end

function create_cache(initial_density, ::SummationDensity)
    density = similar(initial_density)

    return (; density)
end

function create_cache(initial_density, ::ContinuityDensity)
    return (; initial_density)
end

function create_cache(initial_density, ::AdamiPressureExtrapolation)
    density = similar(initial_density)
    volume = similar(initial_density)

    return (; density, volume)
end

@inline function nparticles(container::BoundaryParticleContainer)
    length(container.boundary_model.hydrodynamic_mass)
end

# No particle positions are advanced for boundary containers,
# except when using BoundaryModelDummyParticles with ContinuityDensity.
@inline function n_moving_particles(container::BoundaryParticleContainer)
    n_moving_particles(container, container.boundary_model)
end
@inline n_moving_particles(container::BoundaryParticleContainer, model) = 0
@inline function n_moving_particles(container::BoundaryParticleContainer,
                                    model::BoundaryModelDummyParticles)
    n_moving_particles(container, model.density_calculator)
end
@inline function n_moving_particles(container::BoundaryParticleContainer,
                                    ::ContinuityDensity)
    nparticles(container)
end

@inline u_nvariables(container::BoundaryParticleContainer) = 0

# For BoundaryModelDummyParticles with ContinuityDensity, this needs to be 1.
# For all other models and density calculators, it's irrelevant.
@inline v_nvariables(container::BoundaryParticleContainer) = 1

@inline function v_nvariables(container::SolidParticleContainer,
                              model::BoundaryModelDummyParticles)
    v_nvariables(container, model.density_calculator)
end
@inline function v_nvariables(container::SolidParticleContainer, ::ContinuityDensity)
    2 * ndims(container) + 1
end

@inline function current_coordinates(u, container::BoundaryParticleContainer)
    return container.initial_coordinates
end

@inline function get_particle_vel(particle, v, container::BoundaryParticleContainer)
    # TODO moving boundaries
    return SVector(ntuple(_ -> 0.0, Val(ndims(container))))
end

# This will only be called for BoundaryModelDummyParticles
@inline function get_particle_density(particle, v,
                                      container::Union{BoundaryParticleContainer,
                                                       SolidParticleContainer})
    @unpack boundary_model = container
    @unpack density_calculator = boundary_model

    get_particle_density(particle, v, density_calculator, boundary_model, container)
end

@inline function get_particle_density(particle, v,
                                      ::Union{AdamiPressureExtrapolation, SummationDensity},
                                      boundary_model, container)
    @unpack cache = boundary_model

    return cache.density[particle]
end

@inline function get_particle_density(particle, v, ::ContinuityDensity, boundary_model,
                                      container)
    return v[end, particle]
end

@inline function get_particle_density(particle, v, density_calculator,
                                      boundary_model::BoundaryModelMonaghanKajtar,
                                      container)
    @unpack hydrodynamic_mass, boundary_particle_spacing = boundary_model

    # This model does not use any particle density. However, a mean density is used for
    # `ArtificialViscosityMonaghan` in the fluid interaction
    return hydrodynamic_mass[particle] / boundary_particle_spacing^ndims(container)
end

@inline function get_hydrodynamic_mass(particle, boundary_model, container)
    return boundary_model.hydrodynamic_mass[particle]
end

function update!(container::BoundaryParticleContainer, container_index,
                 v, u, v_ode, u_ode, semi, t)
    @unpack initial_coordinates, movement_function, boundary_model = container

    container.ismoving[1] = move_boundary_particles!(movement_function, initial_coordinates,
                                                     t)

    update!(boundary_model, container, container_index, v, u, v_ode, u_ode, semi)

    return container
end

function move_boundary_particles!(movement_function, coordinates, t)
    movement_function(coordinates, t)
end
move_boundary_particles!(movement_function::Nothing, coordinates, t) = false

@inline function update!(boundary_model::BoundaryModelMonaghanKajtar, container,
                         container_index, v, u, v_ode, u_ode, semi)
    # Nothing to do in the update step
    return boundary_model
end

@inline function update!(boundary_model::BoundaryModelDummyParticles,
                         container, container_index, v, u, v_ode, u_ode, semi)
    @unpack pressure, density_calculator = boundary_model
    @unpack particle_containers, neighborhood_searches = semi

    pressure .= zero(eltype(pressure))

    compute_quantities!(boundary_model, density_calculator,
                        container, container_index, v, u, v_ode, u_ode, semi)

    return boundary_model
end

function compute_quantities!(boundary_model, ::SummationDensity,
                             container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack state_equation, pressure, cache = boundary_model
    @unpack density = cache # Density is in the cache for SummationDensity

    density .= zero(eltype(density))

    # Use all other containers for the density summation
    @pixie_timeit timer() "compute density" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                       neighbor_container)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index,
                                      neighbor_container, semi)

        @threaded for particle in eachparticle(container)
            compute_density_per_particle(particle, u, u_neighbor_container,
                                         container, neighbor_container,
                                         neighborhood_searches[container_index][neighbor_container_index])
        end
    end

    for particle in eachparticle(container)
        pressure[particle] = state_equation(get_particle_density(particle, v,
                                                                 boundary_model))
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_density_per_particle(particle, u_particle_container,
                                              u_neighbor_container,
                                              particle_container::Union{
                                                                        BoundaryParticleContainer,
                                                                        SolidParticleContainer
                                                                        },
                                              neighbor_container, neighborhood_search)
    @unpack boundary_model = particle_container
    @unpack smoothing_kernel, smoothing_length, cache = boundary_model
    @unpack density = cache # Density is in the cache for SummationDensity

    particle_coords = get_current_coords(particle, u_particle_container, particle_container)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        mass = get_hydrodynamic_mass(neighbor, neighbor_container)
        distance = norm(particle_coords -
                        get_current_coords(neighbor, u_neighbor_container,
                                           neighbor_container))

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density[particle] += mass * kernel(smoothing_kernel, distance, smoothing_length)
        end
    end
end

function compute_quantities!(boundary_model, ::ContinuityDensity,
                             container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack pressure, state_equation = boundary_model

    for particle in eachparticle(container)
        pressure[particle] = state_equation(get_particle_density(particle, v,
                                                                 boundary_model))
    end
end

function compute_quantities!(boundary_model, ::AdamiPressureExtrapolation,
                             container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack pressure, state_equation, cache = boundary_model
    @unpack density, volume = cache

    density .= zero(eltype(density))
    volume .= zero(eltype(volume))

    # Use all other containers for the pressure summation
    @pixie_timeit timer() "compute boundary pressure" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                                 neighbor_container)
        v_neighbor_container = wrap_v(v_ode, neighbor_container_index,
                                      neighbor_container, semi)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index,
                                      neighbor_container, semi)

        @threaded for particle in eachparticle(container)
            compute_pressure_per_particle(particle, u,
                                          v_neighbor_container, u_neighbor_container,
                                          container, neighbor_container,
                                          neighborhood_searches[container_index][neighbor_container_index],
                                          boundary_model)
        end
    end

    pressure ./= volume

    for particle in eachparticle(container)
        density[particle] = inverse_state_equation(state_equation, pressure[particle])
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_pressure_per_particle(particle, u_particle_container,
                                               v_neighbor_container, u_neighbor_container,
                                               particle_container,
                                               neighbor_container::FluidParticleContainer,
                                               neighborhood_search, boundary_model)
    @unpack pressure, smoothing_kernel, smoothing_length, cache = boundary_model
    @unpack volume = cache

    particle_coords = get_current_coords(particle, u_particle_container, particle_container)
    for neighbor in eachneighbor(particle_coords, neighborhood_search)
        pos_diff = particle_coords -
                   get_current_coords(neighbor, u_neighbor_container, neighbor_container)
        distance = norm(pos_diff)

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density_neighbor = get_particle_density(neighbor, v_neighbor_container,
                                                    neighbor_container)

            # TODO moving boundaries
            pressure[particle] += (neighbor_container.pressure[neighbor] +
                                   dot(neighbor_container.acceleration,
                                       density_neighbor * pos_diff)) *
                                  kernel(smoothing_kernel, distance, smoothing_length)
            volume[particle] += kernel(smoothing_kernel, distance, smoothing_length)
        end
    end

    # Limit pressure to be non-negative to avoid negative pressures at free surfaces
    pressure[particle] = max(pressure[particle], 0.0)
end

@inline function compute_pressure_per_particle(particle, u_particle_container,
                                               v_neighbor_container, u_neighbor_container,
                                               particle_container, neighbor_container,
                                               neighborhood_search,
                                               boundary_model)
    return nothing
end

function write_u0!(u0, container::BoundaryParticleContainer)
    return u0
end

function write_v0!(v0, container::BoundaryParticleContainer)
    @unpack boundary_model = container

    write_v0!(v0, boundary_model, container)
end

function write_v0!(v0, model, container::BoundaryParticleContainer)
    return v0
end

function write_v0!(v0, boundary_model::BoundaryModelDummyParticles,
                   container::SolidParticleContainer)
    @unpack density_calculator = boundary_model

    write_v0!(v0, boundary_model, density_calculator, container)
end

function write_v0!(v0, model::BoundaryModelDummyParticles,
                   container::BoundaryParticleContainer)
    @unpack density_calculator = model

    write_v0!(v0, density_calculator, container)
end

function write_v0!(v0, ::ContinuityDensity, container::BoundaryParticleContainer)
    @unpack cache = container.boundary_model
    @unpack initial_density = cache

    for particle in eachparticle(container)
        # Set particle densities
        v0[1, particle] = initial_density[particle]
    end

    return v0
end
