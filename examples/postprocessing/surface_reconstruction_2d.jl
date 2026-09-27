# Reconstruct a closed, area-corrected contour from a 2D dam-break simulation.
# The callback writes VTK line cells in the z=0 plane, directly viewable in ParaView.
using TrixiParticles
using OrdinaryDiffEqLowStorageRK

fluid_particle_spacing = 0.05
gravity = 9.81
tspan = (0.0, 0.2)
tank_size = (1.4, 0.8)
tank = RectangularTank(fluid_particle_spacing, (0.8, 0.4), tank_size, 1000.0;
                       n_layers=3, acceleration=(0.0, -gravity))
smoothing_kernel = WendlandC2Kernel{2}()
smoothing_length = 1.5 * fluid_particle_spacing
state_equation = StateEquationCole(; sound_speed=25.0, reference_density=1000.0, exponent=7)
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel, smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           state_equation, acceleration=(0.0, -gravity))
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             smoothing_length; state_equation,
                                             clip_negative_pressure=true)
boundary_system = WallBoundarySystem(tank.boundary, boundary_model)
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

# A two-component tank size selects 2D; for an adaptive grid use `ndims=2` instead.
# In 2D, m/rho is particle area and `volume_tolerance_percent` controls area error.
reconstruction = SurfaceReconstruction(; particle_spacing=fluid_particle_spacing, tank_size,
                                       sparse_component_fallback=true)
surface_callback = SurfaceReconstructionCallback(reconstruction, semi; interval=25,
                                                 output_directory="out", prefix="contour")
sol = solve(ode, RDPK3SpFSAL35(); abstol=1.0e-5, reltol=1.0e-4, dtmax=0.005,
            save_everystep=false, callback=CallbackSet(surface_callback))

if !isnothing(sol)
    contour,
    stats = reconstruct_surface(semi, sol; tank_size,
                                sparse_component_fallback=true)
    println("Reconstructed contour: ", length(contour.faces), " segments, area ",
            round(stats["area"]; digits=6), ", perimeter ",
            round(stats["perimeter"]; digits=6))
end
