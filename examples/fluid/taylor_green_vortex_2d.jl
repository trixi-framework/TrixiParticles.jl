# Taylor Green vortex
#
# P. Ramachandran, K. Puri
# "Entropically damped artiﬁcial compressibility for SPH".
# In: Computers and Fluids, Volume 179 (2019), pages 579-594.
# https://doi.org/10.1016/j.compﬂuid.2018.11.023

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Fluid

particle_spacing = 0.02

reynolds_number = 100.0

box_length = 1.0

n_particles_xy = round(Int, box_length / particle_spacing)

U = 1.0 # m/s

b = -8pi^2 / reynolds_number

fluid_density = 1.0
sound_speed = 10U

nu = U * box_length / reynolds_number

background_pressure = sound_speed^2 * fluid_density

p(pos) = -U^2 * exp(2 * b * 0) * (cos(4pi * pos[1]) + cos(4pi * pos[2])) / 4
p(pos, t) = -U^2 * exp(2 * b * t) * (cos(4pi * pos[1]) + cos(4pi * pos[2])) / 4

v_x(pos) = -U * exp(b * 0) * cos(2pi * pos[1]) * sin(2pi * pos[2])
v_y(pos) = U * exp(b * 0) * sin(2pi * pos[1]) * cos(2pi * pos[2])
v_x(pos, t) = -U * exp(b * t) * cos(2pi * pos[1]) * sin(2pi * pos[2])
v_y(pos, t) = U * exp(b * t) * sin(2pi * pos[1]) * cos(2pi * pos[2])

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

fluid = RectangularShape(particle_spacing, (n_particles_xy, n_particles_xy), (0.0, 0.0),
                         fluid_density, pressure=background_pressure,
                         init_velocity=(0.0, 0.0))

# Add small random displacement to the particles to avoid stagnant streamlines.
seed!(42);
fluid.coordinates .+= rand((-particle_spacing / 5):1e-5:(particle_spacing / 5),
                           size(fluid.coordinates))

fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, initial_pressure_function=p,
                                           initial_velocity_function=(v_x, v_y),
                                           transport_velocity=TransportVelocityAdami(background_pressure),
                                           viscosity=ViscosityAdami(nu))
semi = Semidiscretization(fluid_system,
                          neighborhood_search=GridNeighborhoodSearch,
                          periodic_box_min_corner=[0.0, 0.0],
                          periodic_box_max_corner=[box_length, box_length])

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

dt_max = min(smoothing_length / 4 * (sound_speed + U), smoothing_length^2 / (8 * nu))

function compute_L1v_error(v, u, t, system)
    (; initial_velocity_function) = system

    v_analytical_avg = 0.0
    L1v = 0.0

    for particle in TrixiParticles.eachparticle(system)
        position = TrixiParticles.current_coords(u, system, particle)

        v_mag = norm(TrixiParticles.current_velocity(v, system, particle))
        v_analytical = norm(SVector(ntuple(i -> initial_velocity_function[i](position, t),
                                           Val(ndims(system)))))

        v_analytical_avg += abs(v_analytical)
        L1v += abs(v_mag - v_analytical)
    end
    v_analytical_avg /= TrixiParticles.nparticles(system)

    L1v /= TrixiParticles.nparticles(system)

    return L1v /= v_analytical_avg
end

function compute_L1p_error(v, u, t, system)
    p_max_exact = 0.0

    L1p = 0.0

    for particle in TrixiParticles.eachparticle(system)
        position = TrixiParticles.current_coords(u, system, particle)

        # compute pressure error
        p_analytical = system.initial_pressure_function(position, t)
        p_max_exact = max(p_max_exact, abs(p_analytical))

        # p_computed - p_average
        p_computed = TrixiParticles.particle_pressure(v, system, particle) -
                     TrixiParticles.average_pressure(system, particle)
        L1p += abs(p_computed - p_analytical)
    end

    L1p /= TrixiParticles.nparticles(system)

    return L1p /= p_max_exact
end

function average_pressure(v, u, t, system)
    return [TrixiParticles.average_pressure(system, i) for i in axes(v, 2)]
end

function time_vector(v, u, t, system)
    return t
end

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02,
                                         L1v=compute_L1v_error,
                                         L1p=compute_L1p_error,
                                         p_avg=average_pressure,
                                         t=time_vector)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode,
            RDPK3SpFSAL49((step_limiter!)=TrixiParticles.update_transport_velocity!),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=dt_max,#1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
