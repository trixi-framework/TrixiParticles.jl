using TrixiParticles
using BenchmarkTools

# Suppress the output of the simulation
function suppress_simulation_output(f)
    open("/dev/null", "w") do devnull
        redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                f()
            end
        end
    end
end

# Benchmark using @btime and suppress output
result = @belapsed suppress_simulation_output(() -> trixi_include(@__MODULE__,
                                                                  joinpath(examples_dir(),
                                                                           "fluid",
                                                                           "dam_break_2d.jl"),
                                                                  saving_callback=nothing,
                                                                  saving_paper=nothing,
                                                                  fluid_particle_spacing=0.01))

# Print the timing result in nanoseconds
println(result)
