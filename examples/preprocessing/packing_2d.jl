using TrixiParticles
using OrdinaryDiffEq

particle_spacing = 0.3
tspan = (0, 10.0)

filename = joinpath("out_preprocessing", "hexagon.asc")

# Returns `Shape`
shape = load_shape(filename)

# Returns `InitialCondition`.
shape_sampled = ComplexShape(shape; particle_spacing, density=1.0)

signed_distance_field = SignedDistanceField(shape, particle_spacing;
                                            max_signed_distance=5particle_spacing,
                                            neighborhood_search=true)

packing_system = ParticlePackingSystem(shape_sampled; tlsph=true,
                                       signed_distance_field,
                                       neighborhood_search=true,
                                       boundary=shape, background_pressure=100.0)

boundary_system = ParticlePackingSystem(shape_sampled; tlsph=true,
                                        is_boundary=true,
                                        signed_distance_field,
                                        neighborhood_search=true,
                                        boundary=shape, background_pressure=100.0)

semi = Semidiscretization(packing_system, boundary_system)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback)

sol = solve(ode, RK4();
            save_everystep=false, maxiters=200, callback=callbacks, dtmax=1e-2)

packed_ic = InitialCondition(sol, packing_system, semi);
#packed_boundary_ic = InitialCondition(sol, boundary_system, semi);
