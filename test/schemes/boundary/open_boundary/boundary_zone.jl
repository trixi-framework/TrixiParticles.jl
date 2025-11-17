@testset verbose=true "Boundary Zone" begin
    @testset "`show`" begin
        inflow = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                              particle_spacing=0.05,
                              face_normal=(1.0, 0.0), density=1.0,
                              reference_density=0.0,
                              reference_pressure=0.0,
                              reference_velocity=[0.0, 0.0],
                              open_boundary_layers=4, boundary_type=InFlow())

        show_compact = "BoundaryZone() with 80 particles"
        @test repr(inflow) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ BoundaryZone                                                                                     │
        │ ════════════                                                                                     │
        │ boundary type: ………………………………………… inflow                                                           │
        │ #particles: ………………………………………………… 80                                                               │
        │ width: ……………………………………………………………… 0.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", inflow) == show_box

        outflow = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                               particle_spacing=0.05,
                               reference_density=0.0,
                               reference_pressure=0.0,
                               reference_velocity=[0.0, 0.0],
                               face_normal=(1.0, 0.0), density=1.0, open_boundary_layers=4,
                               boundary_type=OutFlow())

        show_compact = "BoundaryZone() with 80 particles"
        @test repr(outflow) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ BoundaryZone                                                                                     │
        │ ════════════                                                                                     │
        │ boundary type: ………………………………………… outflow                                                          │
        │ #particles: ………………………………………………… 80                                                               │
        │ width: ……………………………………………………………… 0.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", outflow) == show_box
    end

    @testset verbose=true "Illegal Inputs" begin
        boundary_face = ([0.0, 0.0], [0.0, 1.0])
        flow_direction = (1.0, 0.0)

        error_str = "`reference_velocity` must be either a function mapping " *
                    "each particle's coordinates and time to its velocity, " *
                    "or, for a constant fluid velocity, a vector of length 2 for a 2D problem holding this velocity"

        reference_velocity = 1.0
        @test_throws ArgumentError(error_str) BoundaryZone(; boundary_face,
                                                           particle_spacing=0.1,
                                                           face_normal=flow_direction,
                                                           density=1.0,
                                                           reference_density=0,
                                                           reference_pressure=0,
                                                           reference_velocity,
                                                           open_boundary_layers=2,
                                                           boundary_type=InFlow())

        error_str = "`reference_pressure` must be either a function mapping " *
                    "each particle's coordinates and time to its pressure, " *
                    "a scalar, or a pressure model"

        reference_pressure = [1.0, 1.0]

        @test_throws ArgumentError(error_str) BoundaryZone(; boundary_face,
                                                           particle_spacing=0.1,
                                                           face_normal=flow_direction,
                                                           density=1.0,
                                                           reference_density=0,
                                                           reference_velocity=[1.0,
                                                               1.0], reference_pressure,
                                                           open_boundary_layers=2,
                                                           boundary_type=InFlow())

        error_str = "`reference_density` must be either a function mapping " *
                    "each particle's coordinates and time to its density, " *
                    "or a scalar"

        reference_density = [1.0, 1.0]
        @test_throws ArgumentError(error_str) BoundaryZone(; boundary_face,
                                                           particle_spacing=0.1,
                                                           face_normal=flow_direction,
                                                           density=1.0,
                                                           reference_density,
                                                           reference_velocity=[1.0,
                                                               1.0],
                                                           reference_pressure=0,
                                                           open_boundary_layers=2,
                                                           boundary_type=InFlow())
    end
    @testset verbose=true "Boundary Zone 2D" begin
        particle_spacing = 0.2
        open_boundary_layers = 4

        face_vertices_1 = [[0.0, 0.0], [0.5, -0.5], [1.0, 0.5]]
        face_vertices_2 = [[0.0, 1.0], [0.2, 2.0], [2.3, 0.5]]

        @testset verbose=true "Points $i" for i in eachindex(face_vertices_1)
            vertex_1 = face_vertices_1[i]
            vertex_2 = face_vertices_2[i]

            face_size = vertex_2 - vertex_1

            flow_directions = [
                normalize([-face_size[2], face_size[1]]),
                -normalize([-face_size[2], face_size[1]])
            ]

            @testset verbose=true "Flow Direction $j" for j in eachindex(flow_directions)
                inflow = BoundaryZone(; boundary_face=(vertex_1, vertex_2),
                                      particle_spacing,
                                      face_normal=flow_directions[j], density=1.0,
                                      open_boundary_layers, boundary_type=InFlow())
                outflow = BoundaryZone(; boundary_face=(vertex_1, vertex_2),
                                       particle_spacing,
                                       face_normal=(-flow_directions[j]), density=1.0,
                                       open_boundary_layers, boundary_type=OutFlow())

                boundary_zones = [
                    inflow,
                    outflow
                ]

                @testset verbose=true "$(TrixiParticles.boundary_type_name(boundary_zone))" for boundary_zone in
                                                                                                boundary_zones

                    zone_width = open_boundary_layers *
                                 boundary_zone.initial_condition.particle_spacing
                    sign_ = (TrixiParticles.boundary_type_name(boundary_zone) == "inflow") ?
                            -1 : 1

                    @test face_vertices_1[i] == boundary_zone.zone_origin
                    @test face_vertices_2[i] - boundary_zone.zone_origin ==
                          boundary_zone.spanning_set[2]
                    @test isapprox(sign_ * flow_directions[j],
                                   normalize(boundary_zone.spanning_set[1]), atol=1e-14)
                    @test isapprox(zone_width, norm(boundary_zone.spanning_set[1]),
                                   atol=1e-14)
                end
            end
        end
    end

    @testset verbose=true "Boundary Zone 3D" begin
        particle_spacing = 0.05
        open_boundary_layers = 4

        face_vertices_1 = [
            [0.0, 0.0, 0.0],
            [0.3113730847835541, 0.19079485535621643, -0.440864622592926]
        ]
        face_vertices_2 = [
            [1.0, 0.0, 0.0],
            [-0.10468611121177673, 0.252103328704834, -0.44965094327926636]
        ]
        face_vertices_3 = [
            [0.0, 1.0, 0.0],
            [0.3113730847835541, 0.25057315826416016, -0.02374829351902008]
        ]

        @testset verbose=true "Points $i" for i in eachindex(face_vertices_1)
            vertex_1 = face_vertices_1[i]
            vertex_2 = face_vertices_2[i]
            vertex_3 = face_vertices_3[i]

            edge1 = vertex_2 - vertex_1
            edge2 = vertex_3 - vertex_1

            flow_directions = [
                normalize(cross(edge1, edge2)),
                -normalize(cross(edge1, edge2))
            ]

            @testset verbose=true "Flow Direction $j" for j in eachindex(flow_directions)
                inflow = BoundaryZone(; boundary_face=(vertex_1, vertex_2, vertex_3),
                                      particle_spacing,
                                      face_normal=flow_directions[j], density=1.0,
                                      open_boundary_layers, boundary_type=InFlow())
                outflow = BoundaryZone(; boundary_face=(vertex_1, vertex_2, vertex_3),
                                       particle_spacing,
                                       face_normal=(-flow_directions[j]), density=1.0,
                                       open_boundary_layers, boundary_type=OutFlow())

                boundary_zones = [
                    inflow,
                    outflow
                ]

                @testset verbose=true "$(TrixiParticles.boundary_type_name(boundary_zone))" for boundary_zone in
                                                                                                boundary_zones

                    zone_width = open_boundary_layers *
                                 boundary_zone.initial_condition.particle_spacing
                    sign_ = (TrixiParticles.boundary_type_name(boundary_zone) == "inflow") ?
                            -1 : 1

                    @test face_vertices_1[i] == boundary_zone.zone_origin
                    @test face_vertices_2[i] - boundary_zone.zone_origin ==
                          boundary_zone.spanning_set[2]
                    @test face_vertices_3[i] - boundary_zone.zone_origin ==
                          boundary_zone.spanning_set[3]
                    @test isapprox(sign_ * flow_directions[j],
                                   normalize(boundary_zone.spanning_set[1]), atol=1e-14)
                    @test isapprox(zone_width, norm(boundary_zone.spanning_set[1]),
                                   atol=1e-14)
                end
            end
        end
    end

    @testset verbose=true "Particle In Boundary Zone 2D" begin
        face_vertices = [[-0.2, -0.5], [0.3, 0.6]]
        face_size = face_vertices[2] - face_vertices[1]

        flow_direction = normalize([-face_size[2], face_size[1]])

        inflow = BoundaryZone(; boundary_face=face_vertices, particle_spacing=0.1,
                              face_normal=flow_direction, density=1.0,
                              open_boundary_layers=4, boundary_type=InFlow())
        outflow = BoundaryZone(; boundary_face=face_vertices, particle_spacing=0.1,
                               face_normal=(-flow_direction), density=1.0,
                               open_boundary_layers=4, boundary_type=OutFlow())

        boundary_zones = [
            inflow,
            outflow
        ]

        @testset verbose=true "$(TrixiParticles.boundary_type_name(boundary_zone))" for boundary_zone in
                                                                                        boundary_zones

            perturb_ = TrixiParticles.boundary_type_name(boundary_zone) == "inflow" ?
                       sqrt(eps()) :
                       -sqrt(eps())

            point1 = face_vertices[1]
            point2 = face_vertices[2]
            point3 = boundary_zone.spanning_set[1] + boundary_zone.zone_origin

            query_points = Dict(
                "Behind" => ([-1.0, -1.0], false),
                "Before" => ([2.0, 2.0], false),
                "Closely On Point 1" => (point1 + perturb_ * flow_direction, false),
                "Closely On Point 2" => (point2 + perturb_ * flow_direction, false),
                "Closely On Point 3" => (point3 - perturb_ * flow_direction, false))

            @testset verbose=true "$k" for k in keys(query_points)
                (particle_position, evaluation) = query_points[k]

                @test evaluation ==
                      TrixiParticles.is_in_boundary_zone(boundary_zone, particle_position)
            end
        end
    end

    @testset verbose=true "Particle In Boundary Zone 3D" begin
        point1 = [-0.2, -0.5, 0.0]
        point2 = [0.3, 0.5, 0.0]
        point3 = [0.13111173850909402, -0.665555869254547, 0.0]

        flow_direction = normalize(cross(point2 - point1, point3 - point1))

        inflow = BoundaryZone(; boundary_face=[point1, point2, point3],
                              particle_spacing=0.1,
                              face_normal=flow_direction, density=1.0,
                              open_boundary_layers=4, boundary_type=InFlow())
        outflow = BoundaryZone(; boundary_face=[point1, point2, point3],
                               particle_spacing=0.1,
                               face_normal=(-flow_direction), density=1.0,
                               open_boundary_layers=4, boundary_type=OutFlow())

        boundary_zones = [
            inflow,
            outflow
        ]

        @testset verbose=true "$(TrixiParticles.boundary_type_name(boundary_zone))" for boundary_zone in
                                                                                        boundary_zones

            perturb_ = TrixiParticles.boundary_type_name(boundary_zone) == "inflow" ?
                       eps() : -eps()
            point4 = boundary_zone.spanning_set[1] + boundary_zone.zone_origin

            query_points = Dict(
                "Behind" => ([-1.0, -1.0, 1.2], false),
                "Before" => ([2.0, 2.0, -1.2], false),
                "Closely On Point 1" => (point1 + perturb_ * flow_direction, false),
                "Closely On Point 2" => (point2 + perturb_ * flow_direction, false),
                "Closely On Point 3" => (point3 + perturb_ * flow_direction, false),
                "Closely On Point 4" => (point4 - perturb_ * flow_direction, false))

            @testset verbose=true "$k" for k in keys(query_points)
                (particle_position, evaluation) = query_points[k]

                @test evaluation ==
                      TrixiParticles.is_in_boundary_zone(boundary_zone, particle_position)
            end
        end
    end

    @testset verbose=true "Illegal Inputs" begin
        no_rectangular_face = [[0.2, 0.3, -0.5], [-1.0, 1.5, 0.2], [-0.4, 0.9, -0.15]]
        flow_direction = [0.0, 0.0, 1.0]

        error_str = "the vectors `AB` and `AC` must not be collinear"

        @test_throws ArgumentError(error_str) BoundaryZone(;
                                                           boundary_face=no_rectangular_face,
                                                           particle_spacing=0.1,
                                                           face_normal=flow_direction,
                                                           density=1.0,
                                                           open_boundary_layers=2,
                                                           boundary_type=InFlow())
        @test_throws ArgumentError(error_str) BoundaryZone(;
                                                           boundary_face=no_rectangular_face,
                                                           particle_spacing=0.1,
                                                           face_normal=(-flow_direction),
                                                           density=1.0,
                                                           open_boundary_layers=2,
                                                           boundary_type=OutFlow())

        rectangular_face = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        flow_direction = [0.0, 1.0, 0.0]

        error_str = "`face_normal` is not normal to the boundary face"

        @test_throws ArgumentError(error_str) BoundaryZone(;
                                                           boundary_face=rectangular_face,
                                                           particle_spacing=0.1,
                                                           face_normal=flow_direction,
                                                           density=1.0,
                                                           open_boundary_layers=2,
                                                           boundary_type=InFlow())

        error_str = "`face_normal` is not normal to the boundary face"

        @test_throws ArgumentError(error_str) BoundaryZone(;
                                                           boundary_face=rectangular_face,
                                                           particle_spacing=0.1,
                                                           face_normal=(-flow_direction),
                                                           density=1.0,
                                                           open_boundary_layers=2,
                                                           boundary_type=OutFlow())
    end
end
