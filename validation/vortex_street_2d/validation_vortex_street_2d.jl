using TrixiParticles

# Results in [90k particles, 220k particles, 1.2M particles, 5M particles]
resolution_factors = [0.08, 0.05, 0.02, 0.01]

for resolution_factor in resolution_factors
    trixi_include(joinpath(validation_dir(), "vortex_street_2d", "vortex_street_2d.jl"),
                  factor_d=resolution_factor, saving_callback=nothing, tspan=(0.0, 50.0))
end
