using TrixiParticles
using OrdinaryDiffEq

particle_spacing = 0.05
density = 1000.0
tspan = (0, 10.0)

file = "hexagon"

trixi_include(joinpath(examples_dir(), "preprocessing", "complex_shape_2d.jl"), file=file,
              particle_spacing=particle_spacing, density=density)

background_pressure = 1e6 * particle_spacing^2

signed_distance_field = SignedDistanceField(shape, particle_spacing;
                                            max_signed_distance=5particle_spacing,
                                            use_for_boundary_packing=true,
                                            neighborhood_search=true)

packing_system = ParticlePackingSystem(shape_sampled; tlsph=true,
                                       signed_distance_field,
                                       neighborhood_search=true,
                                       boundary=shape, background_pressure)

boundary_system = ParticlePackingSystem(shape_sampled; tlsph=true,
                                        is_boundary=true,
                                        signed_distance_field,
                                        neighborhood_search=true,
                                        boundary=shape, background_pressure)

semi = Semidiscretization(packing_system, boundary_system)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(interval=10, prefix="")

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback)

sol = solve(ode, RK4();
            save_everystep=false, maxiters=100, callback=callbacks, dtmax=1e-2)

packed_ic = InitialCondition(sol, packing_system, semi);
#packed_boundary_ic = InitialCondition(sol, boundary_system, semi);
