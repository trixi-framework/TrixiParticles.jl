using TrixiParticles
using OrdinaryDiffEq

filename = "sphere"

# ==========================================================================================
# ==== Packing parameters
tlsph = true
maxiters = 100
save_intervals = false
pack_boundary = true

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.1

# The following depends on the sampling of the particles. In this case `boundary_thickness`
# means literally the thickness of the boundary packed with boundary particles and *not*
# how many rows of boundary particles will be sampled.
boundary_thickness = 8particle_spacing

# ==========================================================================================
# ==== Load complex shape
density = 1000.0

trixi_include(joinpath(examples_dir(), "preprocessing", "complex_shape_3d.jl"),
              density=density, particle_spacing=particle_spacing, filename=filename,
              boundary_thickness=boundary_thickness, sample_boundary=pack_boundary)

# ==========================================================================================
# ==== Packing

# Large `background_pressure` can cause high accelerations. That is, the adaptive
# time-stepsize will be adjusted properly. We found that the following order of
# `background_pressure` result in appropriate time-stepsizes.
background_pressure = 1e8 * particle_spacing^3

if pack_boundary
    boundary_system = ParticlePackingSystem(shape_sampled; tlsph=tlsph,
                                            boundary_compress_factor=0.8,
                                            is_boundary=true, background_pressure)
else
    boundary_system = nothing
end

packing_system = ParticlePackingSystem(shape_sampled; tlsph=tlsph, background_pressure)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(packing_system, boundary_system)

# Use a high `tspan` to guarantee that the simulation runs at least for `maxiters`
tspan = (0, 10.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = save_intervals ? SolutionSavingCallback(interval=10, prefix="") : nothing

callbacks = CallbackSet(UpdateCallback(), saving_callback, info_callback)

sol = solve(ode, RK4();
            save_everystep=false, maxiters=maxiters, callback=callbacks, dtmax=1e-2)

packed_ic = InitialCondition(sol, packing_system, semi)
trixi2vtk(packed_ic, filename="initial_condition_packed")

if pack_boundary
    packed_boundary_ic = InitialCondition(sol, boundary_system, semi)
    trixi2vtk(packed_boundary_ic, filename="initial_condition_packed_boundary")
end
