# Run from the package root: julia --project validation/general/iisph_wall_pressure.jl
using LinearAlgebra: dot, norm
using Statistics: mean
using TrixiParticles

const TP = TrixiParticles
const TO = TP.TrixiBase.TimerOutputs

function wall_case(kind, n, dt)
    rho, height, g = 1000.0, 0.4, 9.81
    moving = kind == :piston
    gravity = moving ? 0.0 : g
    speed = moving ? -0.1 : 0.0
    h = height / n
    kernel = SchoenbergCubicSplineKernel{2}()
    eos = StateEquationCole(; sound_speed=40.0, reference_density=rho, exponent=7)
    tank = RectangularTank(h, (height, height), (height, moving ? height : 2height),
                           rho; n_layers=2, acceleration=(0.0, -gravity),
                           faces=(true, true, true, moving), coordinates_eltype=Float64)
    # Normalize the mass so a fully sampled square lattice starts at the target density.
    mass = rho / sum(TP.kernel(kernel, hypot(i, j) * h, h) for i in -2:2, j in -2:2)
    fill!(tank.fluid.mass, mass)
    fill!(tank.boundary.mass, mass)
    fluid = ImplicitIncompressibleSPHSystem(tank.fluid; smoothing_kernel=kernel,
                                            smoothing_length=h,
                                            acceleration=(0.0, -gravity),
                                            time_step=dt, reference_density=rho,
                                            max_error=0.01, max_iterations=400)
    model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                        AdamiPressureExtrapolation(), kernel, h;
                                        state_equation=eos, clip_negative_pressure=true)
    motion = moving ?
             PrescribedMotion((x, t) -> x[2] > height ? x + SVector(0.0, t * speed) : x,
                              t -> true) : nothing
    wall = WallBoundarySystem(tank.boundary, model; prescribed_motion=motion)
    semi = Semidiscretization(fluid, wall; parallelization_backend=SerialBackend())
    loaded_wall = moving ? tank.boundary.coordinates[2, :] .> height :
                  tank.boundary.coordinates[2, :] .< 0
    return (; fluid, wall, semi, h, mass, loaded_wall, dt, speed, g)
end

function measure_wall_case(kind, n, dt; t_end=0.02)
    (; fluid, wall, semi, h, mass, loaded_wall, speed, g) = wall_case(kind, n, dt)
    ode = semidiscretize(semi, (0.0, t_end); reset_threads=false)
    v, u = map(Array, ode.u0.x)
    dv = similar(v)
    nsteps = round(Int, t_end / dt)
    t_end ≈ nsteps * dt || throw(ArgumentError("t_end must be a multiple of dt"))
    # Warm compilation, then advance with Symplectic Euler and measure the kick work.
    TP.kick!(dv, v, u, ode.p, 0.0)
    TO.reset_timer!(TP.timer())
    for step in 0:(nsteps - 1)
        TP.kick!(dv, v, u, ode.p, step * dt)
        v .+= dt .* dv
        u .+= dt .* v
    end
    TP.kick!(dv, v, u, ode.p, t_end)
    timings = TP.timer()["kick!"]
    updates = timings["update systems and nhs"]
    provisional = updates["provisional IISPH pressure"]["pressure solver"]
    final = updates["pressure solver"]
    calls = TO.ncalls(timings)
    kick_ms = TO.time(timings) / (1e6 * calls)
    provisional_iterations = TO.ncalls(provisional["pressure solver iteration"]) / calls
    final_iterations = TO.ncalls(final["pressure solver iteration"]) / calls

    # Independently reconstruct the linearized density from the actual RHS and
    # measure the normal pressure reaction of the bottom or the piston (unit depth).
    updated_velocity = reshape(v + dt * dv, 2, :)
    coords = reshape(u, 2, :)
    wall_coords = wall.coordinates
    model = wall.boundary_model
    density_error = similar(fluid.density)
    load = 0.0
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
            v_j = SVector(0.0, loaded_wall[j] ? speed : 0.0)
            grad = TP.smoothing_kernel_grad(fluid, d, distance, i)
            density += dt * mass * dot(v_i - v_j, grad)
            if loaded_wall[j]
                load -= mass^2 *
                        (fluid.pressure[i] / fluid.density[i]^2 +
                         model.pressure[j] / model.cache.density[j]^2) * grad[2]
            end
        end
        residual = density - fluid.reference_density
        density_error[i] = fluid.pressure[i] > 0 ? abs(residual) : max(residual, 0.0)
    end

    equilibrium_weight = kind == :hydrostatic ? length(fluid.density) * mass * g : nothing
    return (; kind, h, dt, t_end, projected_density_error=mean(density_error),
            peak_wall_pressure=maximum(model.pressure[loaded_wall]), normal_load=abs(load),
            equilibrium_weight, provisional_iterations, final_iterations, kick_ms)
end

function measure_pair_pressure_split(dt)
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
    eos = StateEquationCole(; sound_speed=40.0, reference_density=rho, exponent=7)
    ic = point(h)
    model = BoundaryModelDummyParticles(ic.density, ic.mass, AdamiPressureExtrapolation(),
                                        kernel, h; state_equation=eos)
    motion = PrescribedMotion((x, t) -> x + SVector(speed * t, 0.0), t -> true)
    wall = WallBoundarySystem(ic, model; prescribed_motion=motion)
    semi = Semidiscretization(fluid, wall; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, dt); reset_threads=false)
    v, u = map(Array, ode.u0.x)
    dv = zero(v)
    TP.kick!(dv, v, u, ode.p, 0.0)

    # A self-consistent Adami wall with one fluid neighbor and no pressure offset
    # has p_b = p_f. Solve the resulting scalar discrete density equation by bisection.
    density_weight = mass * TP.smoothing_kernel_grad(fluid, SVector(-h, 0.0), h, 1)[1]
    balance(p) = dt * density_weight *
                 (p / rho^2 + p / TP.inverse_state_equation(eos, p)^2) + speed
    lower, upper = 0.0, -speed * rho^2 / (dt * density_weight)
    for _ in 1:60
        p = (lower + upper) / 2
        if balance(p) > 0
            upper = p
        else
            lower = p
        end
    end
    coupled_pressure = (lower + upper) / 2
    wall_pressure = only(model.pressure)
    density_error = abs(only(fluid.density) + dt * density_weight * (dt * dv[1] - speed) -
                        rho)
    return (; kind=:pair, dt, wall_pressure, coupled_pressure,
            pressure_ratio=wall_pressure / coupled_pressure, density_error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for dt in (0.005, 0.0025)
        println(measure_pair_pressure_split(dt))
    end
    for kind in (:hydrostatic, :piston), n in (10, 20), dt in (0.001, 0.0005)
        println(measure_wall_case(kind, n, dt))
    end
end
