include("signed_distance.jl")
include("system.jl")
include("rhs.jl")

function ParticlePacking(initial_condition, shape;
                         smoothing_kernel=SchoenbergCubicSplineKernel{ndims(initial_condition)}(),
                         smoothing_length=1.2initial_condition.particle_spacing,
                         neighborhood_search=true,
                         precalculate_sdf=false,
                         background_pressure=100.0,
                         tlsph=true,
                         time_integrator=RK4(),
                         dtmax=1e-2,
                         info_callback=nothing,
                         solution_saving_callback=nothing,
                         maxiters=100, tspan=(0.0, 10.0))

    # start packing with custom arguments
    return ParticlePacking(initial_condition, shape, smoothing_kernel, smoothing_length,
                           time_integrator, dtmax, tspan, info_callback,
                           solution_saving_callback, maxiters, background_pressure, tlsph,
                           neighborhood_search, precalculate_sdf)
end

# TODO: call this with `trixi_include` so we can get rid of `OrdinaryDiffEq` dependency
function ParticlePacking(ic, shape, smoothing_kernel, smoothing_length,
                         time_integrator, dtmax, tspan, info_callback,
                         solution_saving_callback, maxiters, background_pressure, tlsph,
                         neighborhood_search, precalculate_sdf)
    packing_system = ParticlePackingSystem(ic, smoothing_kernel, smoothing_length;
                                           precalculate_sdf,
                                           neighborhood_search, boundary=shape,
                                           background_pressure, tlsph)

    semi = Semidiscretization(packing_system)

    ode = semidiscretize(semi, tspan)

    callbacks = CallbackSet(UpdateCallback(), solution_saving_callback, info_callback)
    sol = solve(ode, time_integrator;
                save_everystep=false, maxiters, callback=callbacks, dtmax)

    v_ode, u_ode = sol.u[end].x

    u = wrap_u(u_ode, packing_system, semi)

    # TODO: Remove particles that are closer than `max_distance` to any particle
    max_distance = 0.25ic.particle_spacing
    too_close = find_too_close_particles(u, max_distance)

    if isempty(too_close)
        ic.coordinates .= u

        return ic
    end

    not_too_close = setdiff(eachparticle(packing_system), too_close)

    coordinates = u[:, not_too_close]
    velocity = ic.velocity[:, not_too_close]
    mass = ic.mass[not_too_close]
    density = ic.density[not_too_close]
    pressure = ic.pressure[not_too_close]

    @info "$(length(too_close)) removed particles that are too close together"

    return InitialCondition{ndims(ic)}(coordinates, velocity, mass, density, pressure,
                                       ic.particle_spacing)
end
