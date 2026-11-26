@trixi_testset "Taylor–Green correction decay" begin
    # A half-life makes the energy check independent of how the runner computes its reference.
    half_life = log(2) * 100 / (16pi^2)
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(), "corrections",
                                              "validation_tgv_decay.jl");
                                     setups=(16 => 0.01,), t_end=half_life, n_intervals=2,
                                     timing_repeats=1,
                                     output_directory=mktempdir()) [
        r"WARNING: Method definition pressure_function.*\n",
        r"WARNING: Method definition initial_pressure_function.*\n",
        r"WARNING: Method definition velocity_function.*\n",
        r"WARNING: Method definition initial_velocity_function.*\n"
    ]
    history = CSV.read(decay_csv, DataFrame)
    timings = CSV.read(timing_csv, DataFrame)
    @test Set(timings.method) == Set(history.method) && all(>(0), timings.seconds)
    initial = filter(row -> row.time == 0, history)
    final = filter(row -> row.time == half_life, history)
    @test Set(final.method) == Set(["none", "shepard", "gradient", "shepard_gradient",
                  "mixed", "shepard_mixed"])
    @test all(e -> e ≈ 0.25, initial.kinetic_energy)
    @test maximum(initial.velocity_l2_error) < 1e-12
    @test all(e -> e ≈ 0.5, final.energy_ratio_exact)
    @test maximum(abs, final.energy_ratio .- 0.5) < 0.04
    @test all(e -> 0 < e < 0.12, final.velocity_l2_error)
    @test only(final.velocity_l2_error[final.method .== "shepard_gradient"]) <
          only(final.velocity_l2_error[final.method .== "none"])
end
