@testset verbose=true "Boundary Zone" begin
    @testset verbose=true "Boundary Zone 2D" begin
        particle_spacing = 0.2
        open_boundary_layers = 4

        plane_points_1 = [[0.0, 0.0], [0.5, -0.5], [1.0, 0.5]]
        plane_points_2 = [[0.0, 1.0], [0.2, 2.0], [2.3, 0.5]]

        @testset verbose=true "Points $i" for i in eachindex(plane_points_1)
            point_1 = plane_points_1[i]
            point_2 = plane_points_2[i]

            plane_size = point_2 - point_1

            flow_directions = [
                normalize([-plane_size[2], plane_size[1]]),
                -normalize([-plane_size[2], plane_size[1]])
            ]

            @testset verbose=true "Flow Direction $j" for j in eachindex(flow_directions)
                inflow = InFlow(; plane=(point_1, point_2), particle_spacing,
                                flow_direction=flow_directions[j], density=1.0,
                                open_boundary_layers)
                outflow = OutFlow(; plane=(point_1, point_2), particle_spacing,
                                  flow_direction=flow_directions[j], density=1.0,
                                  open_boundary_layers)

                boundary_zones = [
                    inflow,
                    outflow
                ]

                @testset verbose=true "$(nameof(typeof(boundary_zone)))" for boundary_zone in boundary_zones
                    zone_width = open_boundary_layers *
                                 boundary_zone.initial_condition.particle_spacing
                    sign_ = (boundary_zone isa InFlow) ? -1 : 1

                    @test plane_points_1[i] == boundary_zone.zone_origin
                    @test plane_points_2[i] - boundary_zone.zone_origin ==
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

        plane_points_1 = [
            [0.0, 0.0, 0.0],
            [0.3113730847835541, 0.19079485535621643, -0.440864622592926]
        ]
        plane_points_2 = [
            [1.0, 0.0, 0.0],
            [-0.10468611121177673, 0.252103328704834, -0.44965094327926636]
        ]
        plane_points_3 = [
            [0.0, 1.0, 0.0],
            [0.3113730847835541, 0.25057315826416016, -0.02374829351902008]
        ]

        @testset verbose=true "Points $i" for i in eachindex(plane_points_1)
            point_1 = plane_points_1[i]
            point_2 = plane_points_2[i]
            point_3 = plane_points_3[i]

            edge1 = point_2 - point_1
            edge2 = point_3 - point_1

            flow_directions = [
                normalize(cross(edge1, edge2)),
                -normalize(cross(edge1, edge2))
            ]

            @testset verbose=true "Flow Direction $j" for j in eachindex(flow_directions)
                inflow = InFlow(; plane=(point_1, point_2, point_3), particle_spacing,
                                flow_direction=flow_directions[j], density=1.0,
                                open_boundary_layers)
                outflow = OutFlow(; plane=(point_1, point_2, point_3), particle_spacing,
                                  flow_direction=flow_directions[j], density=1.0,
                                  open_boundary_layers)

                boundary_zones = [
                    inflow,
                    outflow
                ]

                @testset verbose=true "$(nameof(typeof(boundary_zone)))" for boundary_zone in boundary_zones
                    zone_width = open_boundary_layers *
                                 boundary_zone.initial_condition.particle_spacing
                    sign_ = (boundary_zone isa InFlow) ? -1 : 1

                    @test plane_points_1[i] == boundary_zone.zone_origin
                    @test plane_points_2[i] - boundary_zone.zone_origin ==
                          boundary_zone.spanning_set[2]
                    @test plane_points_3[i] - boundary_zone.zone_origin ==
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
        plane_points = [[-0.2, -0.5], [0.3, 0.6]]
        plane_size = plane_points[2] - plane_points[1]

        flow_direction = normalize([-plane_size[2], plane_size[1]])

        inflow = InFlow(; plane=plane_points, particle_spacing=0.1,
                        flow_direction, density=1.0, open_boundary_layers=4)
        outflow = OutFlow(; plane=plane_points, particle_spacing=0.1,
                          flow_direction, density=1.0, open_boundary_layers=4)

        boundary_zones = [
            inflow,
            outflow
        ]

        @testset verbose=true "$(nameof(typeof(boundary_zone)))" for boundary_zone in boundary_zones
            perturb_ = boundary_zone isa InFlow ? sqrt(eps()) : -sqrt(eps())

            point1 = plane_points[1]
            point2 = plane_points[2]
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

        inflow = InFlow(; plane=[point1, point2, point3], particle_spacing=0.1,
                        flow_direction, density=1.0, open_boundary_layers=4)
        outflow = OutFlow(; plane=[point1, point2, point3], particle_spacing=0.1,
                          flow_direction, density=1.0, open_boundary_layers=4)

        boundary_zones = [
            inflow,
            outflow
        ]

        @testset verbose=true "$(nameof(typeof(boundary_zone)))" for boundary_zone in boundary_zones
            perturb_ = boundary_zone isa InFlow ? eps() : -eps()
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
        no_rectangular_plane = [[0.2, 0.3, -0.5], [-1.0, 1.5, 0.2], [-0.4, 0.9, -0.15]]
        flow_direction = [0.0, 0.0, 1.0]

        error_str = "the vectors `AB` and `AC` must not be collinear"

        @test_throws ArgumentError(error_str) InFlow(; plane=no_rectangular_plane,
                                                     particle_spacing=0.1,
                                                     flow_direction, density=1.0,
                                                     open_boundary_layers=2)
        @test_throws ArgumentError(error_str) OutFlow(; plane=no_rectangular_plane,
                                                      particle_spacing=0.1,
                                                      flow_direction, density=1.0,
                                                      open_boundary_layers=2)

        rectangular_plane = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        flow_direction = [0.0, 1.0, 0.0]

        error_str = "`flow_direction` is not normal to inflow plane"

        @test_throws ArgumentError(error_str) InFlow(; plane=rectangular_plane,
                                                     particle_spacing=0.1,
                                                     flow_direction, density=1.0,
                                                     open_boundary_layers=2)

        error_str = "`flow_direction` is not normal to outflow plane"

        @test_throws ArgumentError(error_str) OutFlow(; plane=rectangular_plane,
                                                      particle_spacing=0.1,
                                                      flow_direction, density=1.0,
                                                      open_boundary_layers=2)
    end
end
