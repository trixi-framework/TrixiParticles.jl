"""
    MechanicalWorkCalculator(system::TotalLagrangianSPHSystem, semi;
                             eachparticle=(n_integrated_particles(system) + 1):nparticles(system),
                             only_compute_force_on_fluid=false)

Functor that accumulates the work done by a set of particles in a
[`TotalLagrangianSPHSystem`](@ref) by integrating the instantaneous power over time.
It can be passed as a custom quantity to [`PostprocessCallback`](@ref), which controls
when the work is sampled and whether it is written to a file.

With the default arguments it tracks the work done by the clamped particles
that follow a [`PrescribedMotion`](@ref). By selecting a different particle set, it can also
be used to measure the work done by the structure on the surrounding fluid.

- **Prescribed/clamped motion work** (default) -- monitor only the clamped particles by
    leaving `eachparticle` at its default range
    `(n_integrated_particles(system) + 1):nparticles(system)`.
- **Fluid energy transfer** -- set `eachparticle=eachparticle(system)` together with
    `only_compute_force_on_fluid=true` to accumulate the work that the entire structure
    exerts on the surrounding fluid.

Internally the calculator integrates the instantaneous power, i.e. the dot product between
the force exerted by the particle and its prescribed velocity, using an explicit Euler
time integration scheme.

The accumulated value can be retrieved via [`calculated_mechanical_work`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Arguments
- `system`: The [`TotalLagrangianSPHSystem`](@ref) whose particles should be monitored.
- `semi`: The [`Semidiscretization`](@ref) that contains `system`.

# Keywords
- `eachparticle=(n_integrated_particles(system) + 1):nparticles(system)`: Iterator
                selecting which particles contribute. The default includes all clamped
                particles in the system; pass `eachparticle(system)` to include every particle.
- `only_compute_force_on_fluid=false`: When `true`, only interactions with
                fluid systems are accounted for. Combined with
                `eachparticle=eachparticle(system)`, this accumulates the work that the
                entire structure exerts on the fluid.

# Examples
```jldoctest; output = false, setup = :(system = TotalLagrangianSPHSystem(RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); smoothing_kernel=WendlandC2Kernel{2}(), smoothing_length=1.0, young_modulus=1.0, poisson_ratio=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend()))
semi = Semidiscretization(system)
ode = semidiscretize(semi, (0.0, 1.0))

# Note that `Semidiscretization` might create a deep copy of the system,
# which means we have to extract the new system from `semi`.
# When working with GPUs, `semidiscretize` also creates a deep copy of `semi` and another
# copy of the system, so the clean way to get the correct new system is this:
semi_new = ode.p.semi
system_new = semi_new.systems[1]

# Create a mechanical work calculator that is called every 2 time steps.
mechanical_work_calculator = MechanicalWorkCalculator(system_new, semi_new)
postprocess_cb = PostprocessCallback(; interval=2, mechanical_work_calculator)

# After the simulation, retrieve the accumulated mechanical work.
mechanical_work = calculated_mechanical_work(mechanical_work_calculator)

# output
[ Info: To create the self-interaction neighborhood search of a `TotalLagrangianSPHSystem`, a deep copy of the system is created inside the `Semidiscretization`. Use `system = semi.systems[i]` to access simulation data.
0.0
```
"""
mutable struct MechanicalWorkCalculator{ELTYPE, DV, EP}
    t                           :: ELTYPE # Time of last call
    work                        :: ELTYPE
    initialized                 :: Bool
    system_index                :: Int
    dv                          :: DV
    eachparticle                :: EP
    only_compute_force_on_fluid :: Bool
end

# This should dispatch on `TotalLagrangianSPHSystem`, but this name is not yet defined
# due to the include order.
function MechanicalWorkCalculator(system::AbstractStructureSystem, semi;
                                  eachparticle=(n_integrated_particles(system) + 1):nparticles(system),
                                  only_compute_force_on_fluid=false)
    ELTYPE = eltype(system)
    system_index = system_indices(system, semi)

    # Allocate buffer to write accelerations for all particles (including clamped ones).
    # `PostprocessCallback` transfers data to the CPU before calling custom quantities,
    # so this can be a regular `Array` even when the simulation is running on a GPU.
    dv = Array{ELTYPE, 2}(undef, (v_nvariables(system), nparticles(system)))

    return MechanicalWorkCalculator(zero(ELTYPE), zero(ELTYPE), false,
                                    system_index, dv, eachparticle,
                                    only_compute_force_on_fluid)
end

function reset!(calculator::MechanicalWorkCalculator)
    calculator.t = zero(calculator.t)
    calculator.work = zero(calculator.work)
    calculator.initialized = false

    return calculator
end

"""
    calculated_mechanical_work(calculator::MechanicalWorkCalculator)

Get the accumulated mechanical work from a [`MechanicalWorkCalculator`](@ref).

# Arguments
- `calculator`: The [`MechanicalWorkCalculator`](@ref) functor.

# Examples
```jldoctest; output = false, setup = :(system = TotalLagrangianSPHSystem(RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); smoothing_kernel=WendlandC2Kernel{2}(), smoothing_length=1.0, young_modulus=1.0, poisson_ratio=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend()))
# Create a mechanical work calculator
mechanical_work_calculator = MechanicalWorkCalculator(system, semi)

# After the simulation, retrieve the accumulated mechanical work
mechanical_work = calculated_mechanical_work(mechanical_work_calculator)

# output
0.0
```
"""
function calculated_mechanical_work(calculator::MechanicalWorkCalculator)
    return calculator.work
end

function (calculator::MechanicalWorkCalculator)(system, dv_ode, du_ode, v_ode, u_ode,
                                                semi, t)
    if system_indices(system, semi) != calculator.system_index
        return nothing
    end

    if !calculator.initialized
        calculator.t = t
        calculator.work = zero(calculator.work)
        calculator.initialized = true
        return calculator.work
    end

    dt = t - calculator.t
    calculator.t = t

    @trixi_timeit timer() "calculate mechanical work" begin
        calculator.work = update_mechanical_work(calculator.work, system,
                                                 calculator.eachparticle,
                                                 calculator.only_compute_force_on_fluid,
                                                 calculator.dv,
                                                 v_ode, u_ode, semi, t, dt)
    end

    return calculator.work
end

function update_mechanical_work(work, system, eachparticle,
                                only_compute_force_on_fluid, dv,
                                v_ode, u_ode, semi, t, dt)
    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    compute_structure_acceleration!(dv, v, u, system, eachparticle,
                                    only_compute_force_on_fluid,
                                    v_ode, u_ode, semi, t)

    # Note that this is a reduction, so we cannot use `@threaded` here.
    for particle in eachparticle
        velocity = current_velocity(v, system, particle)
        dv_particle = extract_svector(dv, system, particle)

        # The force on the clamped particle is mass times acceleration
        F_particle = system.mass[particle] * dv_particle

        # To obtain mechanical work, we need to integrate the instantaneous power.
        # Instantaneous power is force applied BY the particle times its velocity.
        # The force applied BY the particle is the negative of the force applied ON it.
        work += dot(-F_particle, velocity) * dt
    end

    return work
end

function compute_structure_acceleration!(dv, v, u, system, eachparticle,
                                         only_compute_force_on_fluid,
                                         v_ode, u_ode, semi, t)
    set_zero!(dv)

    foreach_system(semi) do neighbor_system
        if only_compute_force_on_fluid && !(neighbor_system isa AbstractFluidSystem)
            # Not a fluid system, ignore this system.
            return nothing
        end

        v_neighbor = wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)

        interact!(dv, v, u, v_neighbor, u_neighbor,
                  system, neighbor_system, semi,
                  integrate_tlsph=true, # Required when using split integration
                  eachparticle=eachparticle)

        return nothing
    end

    if !only_compute_force_on_fluid
        @threaded semi for particle in eachparticle
            add_acceleration!(dv, system, particle)
            add_source_terms_inner!(dv, v, u, particle, system, source_terms(system), t)
        end
    end

    return dv
end

"""
    ThrustCalculator(system::TotalLagrangianSPHSystem, semi;
                     direction,
                     eachparticle=eachparticle(system))

Functor that computes the instantaneous hydrodynamic force exerted by the fluid on a
[`TotalLagrangianSPHSystem`](@ref), projected onto `direction`.
It can be passed as a custom quantity to [`PostprocessCallback`](@ref).

For a fixed fluid-interacting structure in a channel flow, choose `direction` as the
direction of useful force and multiply the recorded thrust by the corresponding reference
speed to obtain useful mechanical power.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Arguments
- `system`: The [`TotalLagrangianSPHSystem`](@ref) whose hydrodynamic force should be
            monitored.
- `semi`: The [`Semidiscretization`](@ref) that contains `system`.

# Keywords
- `direction`: Direction onto which the hydrodynamic force is projected. The vector is
               normalized internally.
- `eachparticle=eachparticle(system)`: Iterator selecting which particles contribute.

# Examples
```jldoctest; output = false, setup = :(system = TotalLagrangianSPHSystem(RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); smoothing_kernel=WendlandC2Kernel{2}(), smoothing_length=1.0, young_modulus=1.0, poisson_ratio=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend()))
# Create a thrust calculator in x-direction
thrust_calculator = ThrustCalculator(system, semi; direction=SVector(1.0, 0.0))

# After postprocessing, retrieve the latest thrust value
thrust = calculated_thrust(thrust_calculator)

# output
0.0
```
"""
mutable struct ThrustCalculator{ELTYPE, DV, EP, D}
    thrust       :: ELTYPE
    system_index :: Int
    dv           :: DV
    eachparticle :: EP
    direction    :: D
end

# This should dispatch on `TotalLagrangianSPHSystem`, but this name is not yet defined
# due to the include order.
function ThrustCalculator(system::AbstractStructureSystem, semi;
                          direction, eachparticle=eachparticle(system))
    ELTYPE = eltype(system)
    system_index = system_indices(system, semi)

    # Check vector length before converting to `SVector` to avoid extremely long
    # compile times when accidentally passing large vectors.
    if length(direction) != ndims(system)
        throw(ArgumentError("length of `direction` must match the number of dimensions"))
    end
    direction_ = SVector(Tuple(direction))
    if iszero(direction_)
        throw(ArgumentError("`direction` must not be zero"))
    end
    direction_ = normalize(direction_)

    # Allocate buffer to write hydrodynamic accelerations for all particles.
    # `PostprocessCallback` transfers data to the CPU before calling custom quantities,
    # so this can be a regular `Array` even when the simulation is running on a GPU.
    dv = Array{ELTYPE, 2}(undef, (v_nvariables(system), nparticles(system)))

    return ThrustCalculator(zero(ELTYPE), system_index, dv, eachparticle, direction_)
end

function reset!(calculator::ThrustCalculator)
    calculator.thrust = zero(calculator.thrust)

    return calculator
end

"""
    calculated_thrust(calculator::ThrustCalculator)

Get the latest projected hydrodynamic force from a [`ThrustCalculator`](@ref).
"""
function calculated_thrust(calculator::ThrustCalculator)
    return calculator.thrust
end

function (calculator::ThrustCalculator)(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    if system_indices(system, semi) != calculator.system_index
        return nothing
    end

    thrust = update_thrust!(system, calculator.eachparticle, calculator.direction,
                            calculator.dv, v_ode, u_ode, semi, t)

    calculator.thrust = thrust

    return calculator.thrust
end

function update_thrust!(system, eachparticle, direction, dv, v_ode, u_ode, semi, t)
    # Note that the systems and NHS have already been updated by the
    # `PostprocessCallback` before calling this function.
    @trixi_timeit timer() "calculate thrust" begin
        compute_structure_acceleration!(dv, system, eachparticle, true,
                                        v_ode, u_ode, semi, t)

        return projected_force(dv, system, eachparticle, direction)
    end
end

function projected_force(dv, system, eachparticle, direction)
    force = zero(eltype(system))

    # Note that this is a reduction, so we cannot use `@threaded` here.
    for particle in eachparticle
        dv_particle = extract_svector(dv, system, particle)
        force_particle = system.mass[particle] * dv_particle
        force += dot(force_particle, direction)
    end

    return force
end
