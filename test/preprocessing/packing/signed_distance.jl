
@testset verbose=true "Signed Distance Field" begin
    @testset verbose=true "`show`" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")

        geometry = load_geometry(joinpath(data_dir, "hexagon.asc"))

        signed_distance_field = SignedDistanceField(geometry, 0.1)

        show_compact = "SignedDistanceField()"
        @test repr(signed_distance_field) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SignedDistanceField                                                                              │
        │ ═══════════════════                                                                              │
        │ #particles: ………………………………………………… 474                                                              │
        │ max signed distance: ………………………… 0.4                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", signed_distance_field) == show_box

        signed_distance_field = SignedDistanceField(geometry, 0.1; max_signed_distance=0.89)
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SignedDistanceField                                                                              │
        │ ═══════════════════                                                                              │
        │ #particles: ………………………………………………… 1037                                                             │
        │ max signed distance: ………………………… 0.89                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", signed_distance_field) == show_box

        signed_distance_field = SignedDistanceField(geometry, 0.1; max_signed_distance=0.45,
                                                    use_for_boundary_packing=true)
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SignedDistanceField                                                                              │
        │ ═══════════════════                                                                              │
        │ #particles: ………………………………………………… 998                                                              │
        │ max signed distance: ………………………… 0.45                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", signed_distance_field) == show_box
    end

    @testset verbose=true "Real World Data" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
        validation_dir = pkgdir(TrixiParticles, "test", "preprocessing", "data")

        files = ["hexagon.asc", "circle.asc", "sphere.stl"]

        @testset verbose=true "$(files[i])" for i in eachindex(files)
            geometry = load_geometry(joinpath(data_dir, files[i]))

            signed_distance_field = SignedDistanceField(geometry, 0.1)

            nhs = TrixiParticles.FaceNeighborhoodSearch{ndims(geometry)}(search_radius=0.1)

            TrixiParticles.initialize!(nhs, geometry)

            positions_test = [
                [SVector(ntuple(dim -> 0.0, ndims(geometry)))],
                [SVector(ntuple(dim -> 5.0, ndims(geometry)))],
                [first(geometry.vertices)]
            ]

            positions_expected = [
                empty(first(positions_test)),
                empty(first(positions_test)),
                [first(geometry.vertices)]
            ]

            normals_expected = [
                empty(first(positions_test)),
                empty(first(positions_test)),
                [first(geometry.vertex_normals) isa Tuple ?
                 first(geometry.vertex_normals[1]) :
                 first(geometry.vertex_normals)]
            ]

            distances_expected = [
                empty(first(positions_test)),
                empty(first(positions_test)),
                [zero(eltype(geometry))]
            ]

            @testset verbose=true "Positions $j" for j in eachindex(positions_test)
                positions = positions_test[j]
                normals = fill(SVector(ntuple(dim -> Inf, ndims(geometry))),
                               length(positions))
                distances = fill(Inf, length(positions))

                TrixiParticles.calculate_signed_distances!(positions, distances, normals,
                                                           geometry, 1, 0.4, nhs)

                @test positions_expected[j] == positions
                @test normals_expected[j] == normals
                @test distances_expected[j] == distances
            end

            @testset verbose=true "Pointcloud" begin
                # Checked in ParaView with `trixi2vtk(signed_distance_field)`
                data = TrixiParticles.CSV.read(joinpath(validation_dir,
                                                        "signed_distances_" *
                                                        files[i][1:(end - 4)] *
                                                        ".csv"),
                                               TrixiParticles.DataFrame)

                if ndims(geometry) == 3
                    vertex_normals = reinterpret(reshape, SVector{3, Float64},
                                                 vcat((data.var"normals:0")',
                                                      (data.var"normals:1")',
                                                      (data.var"normals:2")'))

                    points = reinterpret(reshape, SVector{3, Float64},
                                         vcat((data.var"Points:0")',
                                              (data.var"Points:1")',
                                              (data.var"Points:2")'))
                else
                    vertex_normals = reinterpret(reshape, SVector{2, Float64},
                                                 vcat((data.var"normals:0")',
                                                      (data.var"normals:1")'))

                    points = reinterpret(reshape, SVector{2, Float64},
                                         vcat((data.var"Points:0")',
                                              (data.var"Points:1")'))
                end

                distances = data.signed_distances

                @test isapprox(signed_distance_field.positions, points; rtol=1e-4)
                @test isapprox(signed_distance_field.normals, vertex_normals; rtol=1e-4)
                @test isapprox(signed_distance_field.distances, distances; rtol=1e-4)
            end
        end
    end
end
