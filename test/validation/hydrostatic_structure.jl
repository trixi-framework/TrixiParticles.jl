@trixi_testset "Isolated hydrostatic plate refinement" begin
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "validation_structure.jl");
                                     resolutions=(3, 5, 21), output_directory=mktempdir()) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    @test length(results) == 3
    @test all(row -> row.rhs_linearization_residual < 1e-7, results)
    @test all(row -> row.equilibrium_residual < 1e-7, results)
    coarse = only(filter(row -> row.resolution == 5, results))
    fine = only(filter(row -> row.resolution == 21, results))
    @test abs(fine.relative_bias) < 0.08
    @test abs(fine.relative_bias) < abs(coarse.relative_bias) / 2
    @test all(row -> row.beam_reference ≈ row.reference / (384 * 0.0026), results)
    @test CSV.read(csv_file, DataFrame).relative_bias ==
          getproperty.(results, :relative_bias)
    @test JSON.parsefile(joinpath(output_directory, "parameters.json"))["resolutions"] ==
          [3, 5, 21]

    # The diagnostic must exercise the actual example's spatial discretization.
    for resolution in (3, 5)
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fsi",
                                                  "hydrostatic_water_column_2d.jl");
                                         n_particles_plate_y=resolution,
                                         parallelization_backend=SerialBackend(),
                                         sol=nothing) [
            r"\[ Info: To create the self-interaction neighborhood search.*\n"
        ]
        isolated = HydrostaticPlateValidation.plate_system(resolution)
        @test isolated.initial_coordinates == structure_system.initial_coordinates
        @test isolated.mass == structure_system.mass
        @test isolated.material_density == structure_system.material_density
        @test isolated.n_integrated_particles == structure_system.n_integrated_particles
        @test isolated.smoothing_length == structure_system.smoothing_length
        @test isolated.young_modulus == structure_system.young_modulus
        @test isolated.poisson_ratio == structure_system.poisson_ratio
    end
end
