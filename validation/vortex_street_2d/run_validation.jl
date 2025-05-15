using TrixiParticles

# results in [90k particles, 220k particles, 1.2M particles, 5M particles]
for resolution_factor in [0.08, 0.05, 0.02, 0.01]
    trixi_include(joinpath(validation_dir(), "vortex_street_2d", "vortex_street_2d.jl"),
                  factor_D=resolution_factor, write_to_vtk=false, tspan=(0.0, 0.04))
end
