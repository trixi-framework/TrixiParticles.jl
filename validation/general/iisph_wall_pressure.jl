# Run from the package root: julia --project validation/general/iisph_wall_pressure.jl
#
# Checks the coupling of IISPH with Adami walls, whose pressure is extrapolated from the
# fluid pressure in every iteration of the pressure solve:
# - with one fluid neighbor, the wall pressure equals the analytic solution of the pair,
# - the wall pressure is the extrapolation of the solved fluid pressure,
# - the actual RHS satisfies the linearized density balance,
# - a resting water column and a water column in a tank that accelerates upward with g/2
#   reach the hydrostatic wall pressure and load.
using LinearAlgebra: dot, norm
using Statistics: mean
using TrixiParticles

const TP = TrixiParticles
const TO = TP.TrixiBase.TimerOutputs

function wall_case(kind, n, dt; viscosity=nothing, max_error=0.01, max_iterations=400)
    rho, height, g = 1000.0, 0.4, 9.81
    # The tank of the accelerating case moves upward with constant acceleration g/2
    wall_acceleration = kind == :accelerating ? g / 2 : 0.0
    h = height / n
    kernel = SchoenbergCubicSplineKernel{2}()
    tank = RectangularTank(h, (height, height), (height, 2height), rho; n_layers=2,
                           acceleration=(0.0, -g), coordinates_eltype=Float64)
    # Normalize the mass so a fully sampled square lattice starts at the target density.
    mass = rho / sum(TP.kernel(kernel, hypot(i, j) * h, h) for i in -2:2, j in -2:2)
    fill!(tank.fluid.mass, mass)
    fill!(tank.boundary.mass, mass)
    fluid = ImplicitIncompressibleSPHSystem(tank.fluid; smoothing_kernel=kernel,
                                            smoothing_length=h, acceleration=(0.0, -g),
                                            time_step=dt, reference_density=rho,
                                            max_error, max_iterations, viscosity)
    model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                        AdamiPressureExtrapolation(), kernel, h;
                                        clip_negative_pressure=true)
    motion = iszero(wall_acceleration) ? nothing :
             PrescribedMotion((x, t) -> x + SVector(0.0, wall_acceleration * t^2 / 2),
                              t -> true)
    wall = WallBoundarySystem(tank.boundary, model; prescribed_motion=motion)
    semi = Semidiscretization(fluid, wall; parallelization_backend=SerialBackend())
    bottom = tank.boundary.coordinates[2, :] .< 0
    # First wall layer below the fluid, without the corners
    bottom_layer = abs.(tank.boundary.coordinates[2, :] .+ h / 2) .< h / 4 .&&
                   0 .< tank.boundary.coordinates[1, :] .< height
    return (; fluid, wall, semi, h, rho, height, mass, bottom, bottom_layer, dt,
            gravity=SVector(0.0, -g), wall_acceleration=SVector(0.0, wall_acceleration))
end

# Advance with Symplectic Euler, calling `measure(step)` after the kick of each step
function advance!(ode, dt, nsteps; measure=Returns(nothing))
    v, u = ode.u0.x
    dv = similar(v)
    for step in 0:nsteps
        TP.kick!(dv, v, u, ode.p, step * dt)
        measure(step)
        step == nsteps && break
        v .+= dt .* dv
        u .+= dt .* v
    end
    return dv
end

# Adami extrapolation of the current fluid pressure to wall particle `b`
function extrapolated_pressure(case, coords, b)
    (; fluid, wall, gravity, wall_acceleration) = case
    model = wall.boundary_model
    x_b = SVector(wall.coordinates[1, b], wall.coordinates[2, b])
    pressure = 0.0
    volume = 0.0
    for f in axes(coords, 2)
        pos_diff = x_b - SVector(coords[1, f], coords[2, f])
        weight = TP.kernel(model.smoothing_kernel, norm(pos_diff), model.smoothing_length)
        pressure += (fluid.pressure[f] +
                     fluid.density[f] * dot(gravity - wall_acceleration, pos_diff)) *
                    weight
        volume += weight
    end
    volume > eps() && (pressure /= volume)
    return max(pressure, 0.0) # The wall model clips negative pressure
end

# Maximum deviation of the wall pressure from the extrapolated fluid pressure,
# relative to the maximum wall pressure
function wall_pressure_mismatch(case, coords)
    pressure = case.wall.boundary_model.pressure
    mismatch = maximum(b -> abs(pressure[b] - extrapolated_pressure(case, coords, b)),
                       eachindex(pressure))
    return mismatch / maximum(abs, pressure)
end

# Normal pressure reaction of the bottom per unit depth
function wall_load(case, coords)
    (; fluid, wall, h, mass, bottom) = case
    model = wall.boundary_model
    load = 0.0
    for i in axes(coords, 2), j in findall(bottom)
        pos_diff = SVector(coords[1, i], coords[2, i]) -
                   SVector(wall.coordinates[1, j], wall.coordinates[2, j])
        distance = norm(pos_diff)
        distance < 2h || continue
        grad = TP.smoothing_kernel_grad(fluid, pos_diff, distance, i)
        load -= mass^2 *
                (fluid.pressure[i] / fluid.density[i]^2 +
                 model.pressure[j] / model.cache.density[j]^2) * grad[2]
    end
    return abs(load)
end

# Diagnostics of the pressure solve in the first time steps of a resting water column
function measure_transient(n, dt; t_end=0.02)
    case = wall_case(:hydrostatic, n, dt)
    (; fluid, wall, semi, h, mass) = case
    ode = semidiscretize(semi, (0.0, t_end); reset_threads=false)
    v, u = ode.u0.x
    nsteps = round(Int, t_end / dt)
    t_end ≈ nsteps * dt || throw(ArgumentError("t_end must be a multiple of dt"))
    # Warm compilation, then advance with Symplectic Euler and measure the kick work.
    TP.kick!(similar(v), copy(v), copy(u), ode.p, 0.0)
    TO.reset_timer!(TP.timer())
    dv = advance!(ode, dt, nsteps)
    timings = TP.timer()["kick!"]
    solver = timings["update systems and nhs"]["pressure solver"]
    calls = TO.ncalls(timings)
    kick_ms = TO.time(timings) / (1e6 * calls)
    iterations = TO.ncalls(solver["pressure solver iteration"]) / calls

    # Independently reconstruct the linearized density from the actual RHS
    updated_velocity = reshape(v + dt * dv, 2, :)
    coords = reshape(u, 2, :)
    wall_coords = wall.coordinates
    density_error = similar(fluid.density)
    for i in axes(coords, 2)
        x_i = SVector(coords[1, i], coords[2, i])
        v_i = SVector(updated_velocity[1, i], updated_velocity[2, i])
        density = fluid.density[i]
        for j in axes(coords, 2)
            d = x_i - SVector(coords[1, j], coords[2, j])
            distance = norm(d)
            distance < 2h || continue
            v_j = SVector(updated_velocity[1, j], updated_velocity[2, j])
            grad = TP.smoothing_kernel_grad(fluid, d, distance, i)
            density += dt * mass * dot(v_i - v_j, grad)
        end
        for j in axes(wall_coords, 2)
            d = x_i - SVector(wall_coords[1, j], wall_coords[2, j])
            distance = norm(d)
            distance < 2h || continue
            grad = TP.smoothing_kernel_grad(fluid, d, distance, i)
            density += dt * mass * dot(v_i, grad)
        end
        residual = density - fluid.reference_density
        density_error[i] = fluid.pressure[i] > 0 ? abs(residual) : max(residual, 0.0)
    end

    return (; kind=:transient, h, dt, t_end, projected_density_error=mean(density_error),
            wall_pressure_mismatch=wall_pressure_mismatch(case, coords), iterations,
            kick_ms)
end

# Integrate a water column until it rests relative to the tank and compare the
# time-averaged pressure of the first wall layer below the fluid and the load on the
# bottom with the hydrostatic values. A fluid viscosity damps the initial transient.
function measure_equilibrium(kind, n, dt; t_end=5.0, averaging_time=1.25,
                             viscosity=ViscosityAdami(nu=1e-3))
    case = wall_case(kind, n, dt; viscosity)
    (; fluid, wall, semi, h, rho, height, mass, bottom_layer, gravity,
     wall_acceleration) = case
    ode = semidiscretize(semi, (0.0, t_end); reset_threads=false)
    v, u = ode.u0.x
    nsteps = round(Int, t_end / dt)
    first_averaged_step = nsteps - round(Int, averaging_time / dt)
    wall_pressure = Float64[]
    extrapolation = Float64[]
    load = Float64[]
    kinetic_energy = Float64[]
    function measure(step)
        step >= first_averaged_step || return
        coords = reshape(u, 2, :)
        push!(wall_pressure, mean(wall.boundary_model.pressure[bottom_layer]))
        push!(extrapolation,
              mean(b -> extrapolated_pressure(case, coords, b), findall(bottom_layer)))
        push!(load, wall_load(case, coords))
        # Kinetic energy relative to the tank per unit depth
        tank_velocity = wall_acceleration * step * dt
        velocity = reshape(v, 2, :)
        push!(kinetic_energy,
              sum(i -> mass / 2 *
                       sum(abs2, SVector(velocity[1, i], velocity[2, i]) - tank_velocity),
                  axes(velocity, 2)))
    end
    advance!(ode, dt, nsteps; measure)

    effective_gravity = norm(gravity - wall_acceleration)
    hydrostatic_wall_pressure = rho * effective_gravity * (height + h / 2)
    weight = length(fluid.density) * mass * effective_gravity
    return (; kind, h, dt, t_end, wall_pressure=mean(wall_pressure),
            hydrostatic_wall_pressure,
            wall_pressure_error=mean(wall_pressure) / hydrostatic_wall_pressure - 1,
            extrapolated_wall_pressure=mean(extrapolation), load=mean(load), weight,
            load_error=mean(load) / weight - 1, kinetic_energy=mean(kinetic_energy))
end

function measure_pair_wall_pressure(dt)
    rho, h, speed = 1000.0, 1.0, -1.0
    kernel = SchoenbergCubicSplineKernel{2}()
    mass = rho / (TP.kernel(kernel, 0.0, h) + TP.kernel(kernel, h, h))
    function point(x)
        return InitialCondition(; coordinates=reshape([x, 0.0], 2, 1), density=rho,
                                mass, particle_spacing=h)
    end
    fluid = ImplicitIncompressibleSPHSystem(point(0.0); smoothing_kernel=kernel,
                                            smoothing_length=h, time_step=dt,
                                            reference_density=rho, max_error=1e-9,
                                            max_iterations=100)
    ic = point(h)
    model = BoundaryModelDummyParticles(ic.density, ic.mass, AdamiPressureExtrapolation(),
                                        kernel, h)
    motion = PrescribedMotion((x, t) -> x + SVector(speed * t, 0.0), t -> true)
    wall = WallBoundarySystem(ic, model; prescribed_motion=motion)
    semi = Semidiscretization(fluid, wall; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, dt); reset_threads=false)
    v, u = map(Array, ode.u0.x)
    dv = zero(v)
    TP.kick!(dv, v, u, ode.p, 0.0)

    # With one fluid neighbor and no gravity or pressure offset, the extrapolated wall
    # pressure is the fluid pressure. With equal densities, the linearized density balance
    # of the pair then gives this pressure.
    density_weight = mass * TP.smoothing_kernel_grad(fluid, SVector(-h, 0.0), h, 1)[1]
    exact_pressure = -speed * rho^2 / (2 * dt * density_weight)
    wall_pressure = only(model.pressure)
    density_error = abs(only(fluid.density) + dt * density_weight * (dt * dv[1] - speed) -
                        rho)
    return (; kind=:pair, dt, wall_pressure, fluid_pressure=only(fluid.pressure),
            exact_pressure, pressure_ratio=wall_pressure / exact_pressure, density_error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for dt in (0.005, 0.0025)
        println(measure_pair_wall_pressure(dt))
    end
    for n in (10, 20), dt in (0.001, 0.0005)
        println(measure_transient(n, dt))
    end
    for kind in (:hydrostatic, :accelerating), n in (10, 20)
        println(measure_equilibrium(kind, n, 0.001))
    end
end
