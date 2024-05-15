using TrixiParticles
using OrdinaryDiffEq

particle_spacing = 0.5
tspan = (0, 10.0)

filename = joinpath("out_preprocessing", "hexagon.asc")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)

packing_system = ParticlePackingSystem(shape_sampled; tlsph=true,
                                       precalculate_sdf=true, neighborhood_search=true,
                                       boundary=shape, background_pressure=100.0)

semi = Semidiscretization(packing_system)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback)

sol = solve(ode, RK4();
            save_everystep=false, maxiters=200, callback=callbacks, dtmax=1e-2)

packed_ic = InitialCondition(sol, packing_system, semi);
