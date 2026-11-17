#include "surface_ibm_geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace culbm::surface_ibm {
namespace {

struct Vec3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3d operator+(const Vec3d& a, const Vec3d& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3d operator-(const Vec3d& a, const Vec3d& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3d operator*(const Vec3d& value, double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

Vec3d operator/(const Vec3d& value, double scale) {
    return {value.x / scale, value.y / scale, value.z / scale};
}

double dot(const Vec3d& a, const Vec3d& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3d cross(const Vec3d& a, const Vec3d& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double norm_squared(const Vec3d& value) {
    return dot(value, value);
}

double norm(const Vec3d& value) {
    return std::sqrt(norm_squared(value));
}

Vec3d normalized(const Vec3d& value) {
    const double length = norm(value);
    return length > 0.0 ? value / length : Vec3d{};
}

bool finite(const Vec3d& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

Vec3d to_vec3d(const float3& value) {
    return {static_cast<double>(value.x), static_cast<double>(value.y), static_cast<double>(value.z)};
}

float3 to_float3(const Vec3d& value) {
    float3 result{};
    result.x = static_cast<float>(value.x);
    result.y = static_cast<float>(value.y);
    result.z = static_cast<float>(value.z);
    return result;
}

double component(const Vec3d& value, int axis) {
    if (axis == 0) return value.x;
    if (axis == 1) return value.y;
    return value.z;
}

struct VertexKey {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;

    bool operator==(const VertexKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator<(const VertexKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

std::uint32_t float_bits(float value) {
    if (value == 0.0f) return 0;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

VertexKey vertex_key(const float3& value) {
    return {float_bits(value.x), float_bits(value.y), float_bits(value.z)};
}

std::size_t hash_combine(std::size_t seed, std::uint32_t value) {
    return seed ^ (static_cast<std::size_t>(value) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U));
}

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& key) const {
        std::size_t seed = 0;
        seed = hash_combine(seed, key.x);
        seed = hash_combine(seed, key.y);
        return hash_combine(seed, key.z);
    }
};

struct EdgeKey {
    VertexKey a;
    VertexKey b;

    bool operator==(const EdgeKey& other) const {
        return a == other.a && b == other.b;
    }
};

EdgeKey edge_key(VertexKey a, VertexKey b) {
    if (b < a) std::swap(a, b);
    return {a, b};
}

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const {
        std::size_t seed = VertexKeyHash{}(key.a);
        seed = hash_combine(seed, key.b.x);
        seed = hash_combine(seed, key.b.y);
        return hash_combine(seed, key.b.z);
    }
};

struct TriangleKey {
    std::array<VertexKey, 3> vertices;

    bool operator==(const TriangleKey& other) const {
        return vertices == other.vertices;
    }
};

TriangleKey triangle_key(const float3& a, const float3& b, const float3& c) {
    TriangleKey key{{vertex_key(a), vertex_key(b), vertex_key(c)}};
    std::sort(key.vertices.begin(), key.vertices.end());
    return key;
}

struct TriangleKeyHash {
    std::size_t operator()(const TriangleKey& key) const {
        std::size_t seed = 0;
        for (const VertexKey& vertex : key.vertices) {
            seed = hash_combine(seed, vertex.x);
            seed = hash_combine(seed, vertex.y);
            seed = hash_combine(seed, vertex.z);
        }
        return seed;
    }
};

struct Triangle {
    Vec3d a;
    Vec3d b;
    Vec3d c;
    Vec3d normal;
    double area = 0.0;
    double max_edge = 0.0;
    std::uint32_t source = 0;
    std::uint32_t component = 0;
    std::array<VertexKey, 3> vertex_keys;
};

struct DisjointSet {
    explicit DisjointSet(std::size_t size) : parent(size), rank(size, 0) {
        std::iota(parent.begin(), parent.end(), std::size_t{0});
    }

    std::size_t find(std::size_t value) {
        while (parent[value] != value) {
            parent[value] = parent[parent[value]];
            value = parent[value];
        }
        return value;
    }

    void unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) ++rank[a];
    }

    std::vector<std::size_t> parent;
    std::vector<unsigned char> rank;
};

struct EdgeRecord {
    std::size_t first_triangle = 0;
    std::size_t count = 0;
};

struct Aabb {
    Vec3d lo{std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
    Vec3d hi{-std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity(),
             -std::numeric_limits<double>::infinity()};
};

void grow(Aabb& box, const Vec3d& value) {
    box.lo.x = std::min(box.lo.x, value.x);
    box.lo.y = std::min(box.lo.y, value.y);
    box.lo.z = std::min(box.lo.z, value.z);
    box.hi.x = std::max(box.hi.x, value.x);
    box.hi.y = std::max(box.hi.y, value.y);
    box.hi.z = std::max(box.hi.z, value.z);
}

void grow(Aabb& box, const Aabb& other) {
    grow(box, other.lo);
    grow(box, other.hi);
}

Aabb triangle_bounds(const Triangle& triangle) {
    Aabb box;
    grow(box, triangle.a);
    grow(box, triangle.b);
    grow(box, triangle.c);
    return box;
}

Vec3d triangle_centroid(const Triangle& triangle) {
    return (triangle.a + triangle.b + triangle.c) / 3.0;
}

bool ray_intersects_box(const Vec3d& origin, const Vec3d& direction, const Aabb& box) {
    double t_min = 0.0;
    double t_max = std::numeric_limits<double>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        const double o = component(origin, axis);
        const double d = component(direction, axis);
        const double lo = component(box.lo, axis);
        const double hi = component(box.hi, axis);
        if (std::abs(d) < 1.0e-18) {
            if (o < lo || o > hi) return false;
            continue;
        }
        double a = (lo - o) / d;
        double b = (hi - o) / d;
        if (a > b) std::swap(a, b);
        t_min = std::max(t_min, a);
        t_max = std::min(t_max, b);
        if (t_max < t_min) return false;
    }
    return t_max > 0.0;
}

bool ray_intersects_triangle(const Vec3d& origin,
                             const Vec3d& direction,
                             const Triangle& triangle,
                             double& distance) {
    const Vec3d edge1 = triangle.b - triangle.a;
    const Vec3d edge2 = triangle.c - triangle.a;
    const Vec3d p = cross(direction, edge2);
    const double determinant = dot(edge1, p);
    const double scale = std::max(1.0, triangle.max_edge * triangle.max_edge);
    if (std::abs(determinant) <= 1.0e-14 * scale) return false;
    const double inverse = 1.0 / determinant;
    const Vec3d t = origin - triangle.a;
    const double u = dot(t, p) * inverse;
    const double barycentric_epsilon = 2.0e-12;
    if (u < -barycentric_epsilon || u > 1.0 + barycentric_epsilon) return false;
    const Vec3d q = cross(t, edge1);
    const double v = dot(direction, q) * inverse;
    if (v < -barycentric_epsilon || u + v > 1.0 + barycentric_epsilon) return false;
    const double hit = dot(edge2, q) * inverse;
    if (hit <= 1.0e-12) return false;
    distance = hit;
    return true;
}

struct BvhNode {
    Aabb bounds;
    std::uint32_t left = 0;
    std::uint32_t right = 0;
    std::size_t begin = 0;
    std::size_t count = 0;
};

class TriangleBvh {
public:
    TriangleBvh(const std::vector<Triangle>& triangles,
                const std::vector<bool>& watertight_components)
        : triangles_(triangles) {
        order_.reserve(triangles.size());
        for (std::size_t i = 0; i < triangles.size(); ++i) {
            if (watertight_components[triangles[i].component]) order_.push_back(i);
        }
        if (!order_.empty()) {
            nodes_.reserve(order_.size() * 2);
            build(0, order_.size());
        }
    }

    bool empty() const {
        return nodes_.empty();
    }

    bool inside_union(const Vec3d& point) const {
        if (empty()) return false;
        static const std::array<Vec3d, 3> directions = {
            normalized(Vec3d{1.0, 0.347134, 0.159731}),
            normalized(Vec3d{0.173271, 1.0, 0.419117}),
            normalized(Vec3d{0.293113, 0.211731, 1.0}),
        };
        unsigned inside_votes = 0;
        for (const Vec3d& direction : directions) {
            if (inside_union_single_ray(point, direction)) ++inside_votes;
        }
        return inside_votes >= 2;
    }

private:
    struct Hit {
        std::uint32_t component = 0;
        double distance = 0.0;
    };

    std::uint32_t build(std::size_t begin, std::size_t end) {
        const std::uint32_t node_index = static_cast<std::uint32_t>(nodes_.size());
        nodes_.push_back({});
        BvhNode node;
        Aabb centroid_bounds;
        for (std::size_t i = begin; i < end; ++i) {
            const Triangle& triangle = triangles_[order_[i]];
            grow(node.bounds, triangle_bounds(triangle));
            grow(centroid_bounds, triangle_centroid(triangle));
        }
        const std::size_t count = end - begin;
        if (count <= 8) {
            node.begin = begin;
            node.count = count;
            nodes_[node_index] = node;
            return node_index;
        }
        const Vec3d extent = centroid_bounds.hi - centroid_bounds.lo;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (component(extent, 2) > component(extent, axis)) axis = 2;
        const std::size_t middle = begin + count / 2;
        std::nth_element(order_.begin() + static_cast<std::ptrdiff_t>(begin),
                         order_.begin() + static_cast<std::ptrdiff_t>(middle),
                         order_.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::size_t lhs, std::size_t rhs) {
                             const double a = component(triangle_centroid(triangles_[lhs]), axis);
                             const double b = component(triangle_centroid(triangles_[rhs]), axis);
                             return a == b ? lhs < rhs : a < b;
                         });
        node.left = build(begin, middle);
        node.right = build(middle, end);
        nodes_[node_index] = node;
        return node_index;
    }

    bool inside_union_single_ray(const Vec3d& point,
                                  const Vec3d& direction) const {
        // Geometry construction is intentionally deterministic and serial.
        // Reuse BVH traversal scratch across the millions of point queries
        // instead of allocating two vectors for every ray.
        hits_.clear();
        stack_.clear();
        stack_.push_back(0);
        while (!stack_.empty()) {
            const std::uint32_t index = stack_.back();
            stack_.pop_back();
            const BvhNode& node = nodes_[index];
            if (!ray_intersects_box(point, direction, node.bounds)) continue;
            if (node.count != 0) {
                for (std::size_t i = node.begin; i < node.begin + node.count; ++i) {
                    const Triangle& triangle = triangles_[order_[i]];
                    double distance = 0.0;
                    if (ray_intersects_triangle(point, direction, triangle, distance)) {
                        hits_.push_back({triangle.component, distance});
                    }
                }
            } else {
                stack_.push_back(node.left);
                stack_.push_back(node.right);
            }
        }
        std::sort(hits_.begin(), hits_.end(), [](const Hit& a, const Hit& b) {
            if (a.component != b.component) return a.component < b.component;
            return a.distance < b.distance;
        });
        std::size_t begin = 0;
        while (begin < hits_.size()) {
            const std::uint32_t current_component = hits_[begin].component;
            unsigned char parity = 0;
            double previous_distance = -std::numeric_limits<double>::infinity();
            std::size_t end = begin;
            while (end < hits_.size() && hits_[end].component == current_component) {
                const double tolerance =
                    1.0e-9 * std::max(1.0, std::abs(hits_[end].distance));
                if (std::abs(hits_[end].distance - previous_distance) > tolerance) {
                    parity ^= 1U;
                    previous_distance = hits_[end].distance;
                }
                ++end;
            }
            if (parity != 0) return true;
            begin = end;
        }
        return false;
    }

    const std::vector<Triangle>& triangles_;
    std::vector<std::size_t> order_;
    std::vector<BvhNode> nodes_;
    mutable std::vector<Hit> hits_;
    mutable std::vector<std::uint32_t> stack_;
};

struct ProjectedVertex {
    Vec3d position;
    double u = 0.0;
    double v = 0.0;
};

// A triangle clipped by an axis-aligned rectangle in its own plane has at
// most seven vertices. Keep one spare slot so clipping cannot silently grow
// beyond the proven bound if round-off leaves a redundant boundary vertex.
struct PolygonPatch {
    std::array<ProjectedVertex, 8> vertices{};
    std::size_t size = 0;
};

double projected_component(const ProjectedVertex& vertex, std::size_t axis) {
    return axis == 0 ? vertex.u : vertex.v;
}

bool same_projected_vertex(const ProjectedVertex& a, const ProjectedVertex& b) {
    return a.u == b.u && a.v == b.v &&
           a.position.x == b.position.x &&
           a.position.y == b.position.y &&
           a.position.z == b.position.z;
}

bool append_polygon_vertex(PolygonPatch& polygon,
                           const ProjectedVertex& vertex,
                           std::string& error) {
    if (polygon.size != 0 &&
        same_projected_vertex(polygon.vertices[polygon.size - 1], vertex)) {
        return true;
    }
    if (polygon.size >= polygon.vertices.size()) {
        error = "surface-IBM adaptive subdivision polygon overflow";
        return false;
    }
    polygon.vertices[polygon.size++] = vertex;
    return true;
}

void remove_closing_duplicate(PolygonPatch& polygon) {
    if (polygon.size > 1 &&
        same_projected_vertex(polygon.vertices[0], polygon.vertices[polygon.size - 1])) {
        --polygon.size;
    }
}

void remove_near_duplicate_vertices(PolygonPatch& polygon) {
    if (polygon.size < 2) return;
    double minimum_u = polygon.vertices[0].u;
    double maximum_u = minimum_u;
    double minimum_v = polygon.vertices[0].v;
    double maximum_v = minimum_v;
    for (std::size_t i = 1; i < polygon.size; ++i) {
        minimum_u = std::min(minimum_u, polygon.vertices[i].u);
        maximum_u = std::max(maximum_u, polygon.vertices[i].u);
        minimum_v = std::min(minimum_v, polygon.vertices[i].v);
        maximum_v = std::max(maximum_v, polygon.vertices[i].v);
    }
    const double scale = std::max(maximum_u - minimum_u, maximum_v - minimum_v);
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * scale;
    if (!(tolerance > 0.0)) return;

    bool changed = true;
    while (changed && polygon.size >= 2) {
        changed = false;
        for (std::size_t i = 0; i < polygon.size; ++i) {
            const std::size_t next = (i + 1) % polygon.size;
            if (std::abs(polygon.vertices[i].u - polygon.vertices[next].u) > tolerance ||
                std::abs(polygon.vertices[i].v - polygon.vertices[next].v) > tolerance) {
                continue;
            }
            const std::size_t remove = next == 0 ? i : next;
            for (std::size_t j = remove + 1; j < polygon.size; ++j) {
                polygon.vertices[j - 1] = polygon.vertices[j];
            }
            --polygon.size;
            changed = true;
            break;
        }
    }
}

void remove_redundant_vertices(PolygonPatch& polygon) {
    bool changed = true;
    while (changed && polygon.size >= 3) {
        changed = false;
        for (std::size_t i = 0; i < polygon.size; ++i) {
            const ProjectedVertex& previous =
                polygon.vertices[(i + polygon.size - 1) % polygon.size];
            const ProjectedVertex& current = polygon.vertices[i];
            const ProjectedVertex& next = polygon.vertices[(i + 1) % polygon.size];
            const double first = (current.u - previous.u) * (next.v - current.v);
            const double second = (current.v - previous.v) * (next.u - current.u);
            const double scale = std::abs(first) + std::abs(second);
            const double turn = first - second;
            if (scale != 0.0 &&
                std::abs(turn) > 64.0 * std::numeric_limits<double>::epsilon() * scale) {
                continue;
            }
            for (std::size_t j = i + 1; j < polygon.size; ++j) {
                polygon.vertices[j - 1] = polygon.vertices[j];
            }
            --polygon.size;
            changed = true;
            break;
        }
    }
}

ProjectedVertex interpolate_to_cut(const ProjectedVertex& a,
                                   const ProjectedVertex& b,
                                   std::size_t axis,
                                   double cut) {
    const double a_coordinate = projected_component(a, axis);
    const double b_coordinate = projected_component(b, axis);
    const double fraction = (cut - a_coordinate) / (b_coordinate - a_coordinate);
    ProjectedVertex result;
    result.position = a.position + (b.position - a.position) * fraction;
    result.u = a.u + (b.u - a.u) * fraction;
    result.v = a.v + (b.v - a.v) * fraction;
    if (axis == 0) result.u = cut;
    else result.v = cut;
    return result;
}

bool clip_polygon(const PolygonPatch& input,
                  std::size_t axis,
                  double cut,
                  bool keep_lower,
                  PolygonPatch& output,
                  std::string& error) {
    output.size = 0;
    if (input.size == 0) return true;
    ProjectedVertex previous = input.vertices[input.size - 1];
    bool previous_inside = keep_lower
        ? projected_component(previous, axis) <= cut
        : projected_component(previous, axis) >= cut;
    for (std::size_t i = 0; i < input.size; ++i) {
        const ProjectedVertex current = input.vertices[i];
        const bool current_inside = keep_lower
            ? projected_component(current, axis) <= cut
            : projected_component(current, axis) >= cut;
        if (current_inside != previous_inside) {
            if (!append_polygon_vertex(
                    output, interpolate_to_cut(previous, current, axis, cut), error)) {
                return false;
            }
        }
        if (current_inside && !append_polygon_vertex(output, current, error)) return false;
        previous = current;
        previous_inside = current_inside;
    }
    remove_closing_duplicate(output);
    remove_near_duplicate_vertices(output);
    remove_redundant_vertices(output);
    return true;
}

double polygon_diameter_squared(const PolygonPatch& polygon) {
    double diameter_squared = 0.0;
    for (std::size_t i = 0; i < polygon.size; ++i) {
        for (std::size_t j = i + 1; j < polygon.size; ++j) {
            diameter_squared = std::max(
                diameter_squared,
                norm_squared(polygon.vertices[j].position - polygon.vertices[i].position));
        }
    }
    return diameter_squared;
}

double classification_offset(const Vec3d& a,
                             const Vec3d& b,
                             const Vec3d& c,
                             double spacing) {
    const double ab = norm(b - a);
    const double bc = norm(c - b);
    const double ca = norm(a - c);
    const double longest = std::max({ab, bc, ca});
    const double twice_area = norm(cross(b - a, c - a));
    const double altitude = longest > 0.0 ? twice_area / longest : 0.0;
    const double desired = spacing * 1.0e-3;
    const double cap = altitude * 2.5e-2;
    return std::max(longest * 1.0e-10, std::min(desired, cap));
}

struct TriangleStatus {
    bool retained = false;
    bool internal = false;
    bool ambiguous = false;
};

bool append_patch(const Triangle& source,
                  const Vec3d& a,
                  const Vec3d& b,
                  const Vec3d& c,
                  bool component_watertight,
                  const TriangleBvh* bvh,
                  const Config& config,
                  std::vector<Marker>& markers,
                  GeometryReport& report,
                  TriangleStatus& status,
                  std::string& error) {
    const Vec3d cross_product = cross(b - a, c - a);
    const double twice_area = norm(cross_product);
    if (!(twice_area > 0.0) || !std::isfinite(twice_area)) {
        status.ambiguous = true;
        ++report.ambiguous_patches;
        if (config.strict_geometry) {
            error = "surface-IBM subdivision produced a degenerate patch";
            return false;
        }
        return true;
    }

    Vec3d outward = cross_product / twice_area;
    const Vec3d centroid = (a + b + c) / 3.0;
    bool keep = true;
    if (config.geometry_policy == GeometryPolicy::ExteriorUnion) {
        if (!component_watertight || bvh == nullptr || bvh->empty()) {
            status.ambiguous = true;
            ++report.ambiguous_patches;
            if (config.strict_geometry) {
                error = "surface-IBM exterior classification requires watertight geometry";
                return false;
            }
        } else {
            const double offset = classification_offset(a, b, c, config.marker_spacing_lu);
            const bool minus_inside = bvh->inside_union(centroid - outward * offset);
            const bool plus_inside = bvh->inside_union(centroid + outward * offset);
            if (minus_inside != plus_inside) {
                if (plus_inside) outward = outward * -1.0;
            } else if (minus_inside) {
                keep = false;
                status.internal = true;
                ++report.internal_patches;
            } else {
                status.ambiguous = true;
                ++report.ambiguous_patches;
                if (config.strict_geometry) {
                    std::ostringstream message;
                    message.precision(17);
                    message << "surface-IBM exterior classification was ambiguous at source triangle "
                            << source.source << " centroid (" << centroid.x << ", "
                            << centroid.y << ", " << centroid.z << ") vertices [("
                            << a.x << ", " << a.y << ", " << a.z << "), ("
                            << b.x << ", " << b.y << ", " << b.z << "), ("
                            << c.x << ", " << c.y << ", " << c.z << ")]";
                    error = message.str();
                    return false;
                }
            }
        }
    }

    if (!keep) return true;
    if (markers.size() >= config.max_markers) {
        error = "surface-IBM marker count exceeds culbm_surface_ibm_max_markers";
        return false;
    }
    const Vec3d marker_position = centroid - outward * static_cast<double>(config.retraction_lu);
    if (!finite(marker_position)) {
        error = "surface-IBM marker position overflow";
        return false;
    }
    Marker marker;
    marker.position_lu = to_float3(marker_position);
    marker.wall_velocity_lu = to_float3({0.0, 0.0, 0.0});
    marker.normal = to_float3(outward);
    marker.area_lu2 = static_cast<float>(0.5 * twice_area);
    // Mesh carries no STL solid/attribute identity after loading. A connected
    // triangle component is a topology aid, not a trustworthy physical body.
    marker.body_id = 0;
    marker.source_triangle = source.source;
    marker.global_id = static_cast<std::uint64_t>(markers.size());
    markers.push_back(marker);
    status.retained = true;
    report.marker_area_lu2 += 0.5 * twice_area;
    report.retained_area_lu2 += 0.5 * twice_area;
    return true;
}

bool append_subdivided_triangle(const Triangle& source,
                                bool component_watertight,
                                const TriangleBvh* bvh,
                                const Config& config,
                                std::vector<Marker>& markers,
                                GeometryReport& report,
                                TriangleStatus& status,
                                std::uint64_t& candidate_patch_count,
                                std::vector<PolygonPatch>& stack,
                                std::string& error) {
    const double spacing = static_cast<double>(config.marker_spacing_lu);
    const double spacing_squared = spacing * spacing;

    const std::array<Vec3d, 3> edge_vectors = {
        source.b - source.a,
        source.c - source.b,
        source.a - source.c,
    };
    const std::array<double, 3> edge_squared = {
        norm_squared(edge_vectors[0]),
        norm_squared(edge_vectors[1]),
        norm_squared(edge_vectors[2]),
    };
    std::size_t reference_edge = 0;
    if (edge_squared[1] > edge_squared[reference_edge]) reference_edge = 1;
    if (edge_squared[2] > edge_squared[reference_edge]) reference_edge = 2;
    const double altitude = 2.0 * source.area / source.max_edge;
    const bool anisotropic = altitude < 0.25 * source.max_edge;
    if (!anisotropic) {
        // Preserve the compact, alignment-friendly barycentric grid for
        // well-shaped facets. The adaptive path below is reserved for the
        // slivers where this n^2 construction is pathological.
        const double ratio = source.max_edge / spacing;
        if (!std::isfinite(ratio) ||
            ratio > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
            error = "surface-IBM triangle subdivision count overflow";
            return false;
        }
        const std::uint64_t divisions = std::max<std::uint64_t>(
            1, static_cast<std::uint64_t>(std::ceil(ratio)));
        if (divisions > config.max_markers / divisions) {
            error = "one surface triangle exceeds culbm_surface_ibm_max_markers after subdivision";
            return false;
        }
        const std::uint64_t emitted = divisions * divisions;
        if (emitted > config.max_markers - candidate_patch_count) {
            error = "surface-IBM candidate patch count exceeds culbm_surface_ibm_max_markers";
            return false;
        }
        candidate_patch_count += emitted;
        const double inverse = 1.0 / static_cast<double>(divisions);
        const auto point_at = [&](const std::uint64_t i, const std::uint64_t j) {
            return source.a + (source.b - source.a) * (static_cast<double>(i) * inverse) +
                   (source.c - source.a) * (static_cast<double>(j) * inverse);
        };
        for (std::uint64_t i = 0; i < divisions; ++i) {
            const std::uint64_t row_count = divisions - i;
            for (std::uint64_t j = 0; j < row_count; ++j) {
                const Vec3d a = point_at(i, j);
                const Vec3d b = point_at(i + 1, j);
                const Vec3d c = point_at(i, j + 1);
                if (!append_patch(source, a, b, c,
                                  component_watertight, bvh, config,
                                  markers, report, status, error)) {
                    return false;
                }
                if (j + 1 < row_count) {
                    const Vec3d d = point_at(i + 1, j + 1);
                    if (!append_patch(source, b, d, c,
                                      component_watertight, bvh, config,
                                      markers, report, status, error)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    PolygonPatch root;
    double coordinate_scales[2] = {0.0, 0.0};
    {
        // Sliver facets can have two long, nearly anti-parallel incident
        // edges. Barycentric u/v refinement would then recreate the n^2
        // explosion. Align physical coordinates with the longest edge so the
        // thin altitude remains an independent, short refinement direction.
        const Vec3d origin = reference_edge == 0 ? source.a
                             : (reference_edge == 1 ? source.b : source.c);
        const Vec3d u_axis = normalized(edge_vectors[reference_edge]);
        const Vec3d v_axis = cross(source.normal, u_axis);
        const std::array<Vec3d, 3> vertices = {source.a, source.b, source.c};
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            const Vec3d relative = vertices[i] - origin;
            root.vertices[i] = {
                vertices[i], dot(relative, u_axis), dot(relative, v_axis),
            };
        }
        coordinate_scales[0] = 1.0;
        coordinate_scales[1] = 1.0;
    }
    root.size = 3;
    stack.clear();
    stack.push_back(root);
    while (!stack.empty()) {
        const PolygonPatch polygon = stack.back();
        stack.pop_back();
        if (polygon.size < 3) {
            error = "surface-IBM adaptive subdivision produced a degenerate polygon";
            return false;
        }
        if (polygon_diameter_squared(polygon) <= spacing_squared) {
            const std::uint64_t emitted = static_cast<std::uint64_t>(polygon.size - 2);
            // Count candidate patches, including patches later rejected as
            // internal by ExteriorUnion. This keeps max_markers a hard bound
            // on construction work as well as on retained marker storage.
            if (emitted > config.max_markers - candidate_patch_count) {
                error = "surface-IBM candidate patch count exceeds culbm_surface_ibm_max_markers";
                return false;
            }
            candidate_patch_count += emitted;
            for (std::size_t i = 1; i + 1 < polygon.size; ++i) {
                const Vec3d& a = polygon.vertices[0].position;
                const Vec3d& b = polygon.vertices[i].position;
                const Vec3d& c = polygon.vertices[i + 1].position;
                const double maximum_edge_squared = std::max({
                    norm_squared(b - a),
                    norm_squared(c - b),
                    norm_squared(a - c),
                });
                if (maximum_edge_squared > spacing_squared) {
                    error = "surface-IBM adaptive subdivision violated marker spacing";
                    return false;
                }
                if (!append_patch(source,
                                  a, b, c,
                                  component_watertight, bvh, config,
                                  markers, report, status, error)) {
                    return false;
                }
            }
            continue;
        }

        double minimum[2] = {
            projected_component(polygon.vertices[0], 0),
            projected_component(polygon.vertices[0], 1),
        };
        double maximum[2] = {minimum[0], minimum[1]};
        for (std::size_t i = 1; i < polygon.size; ++i) {
            for (std::size_t axis = 0; axis < 2; ++axis) {
                const double coordinate = projected_component(polygon.vertices[i], axis);
                minimum[axis] = std::min(minimum[axis], coordinate);
                maximum[axis] = std::max(maximum[axis], coordinate);
            }
        }
        const double extent_u = (maximum[0] - minimum[0]) * coordinate_scales[0];
        const double extent_v = (maximum[1] - minimum[1]) * coordinate_scales[1];
        const std::size_t split_axis = extent_v > extent_u ? 1 : 0;
        const double cut = minimum[split_axis] +
                           (maximum[split_axis] - minimum[split_axis]) * 0.5;
        if (!std::isfinite(cut) || cut == minimum[split_axis] || cut == maximum[split_axis]) {
            error = "surface-IBM adaptive subdivision exhausted coordinate precision";
            return false;
        }
        PolygonPatch lower;
        PolygonPatch upper;
        if (!clip_polygon(polygon, split_axis, cut, true, lower, error) ||
            !clip_polygon(polygon, split_axis, cut, false, upper, error)) {
            return false;
        }
        if (lower.size < 3 || upper.size < 3) {
            error = "surface-IBM adaptive subdivision produced an empty child polygon";
            return false;
        }
        // LIFO insertion preserves stable depth-first ordering.
        stack.push_back(upper);
        stack.push_back(lower);
    }
    return true;
}

void add_warning_once(GeometryReport& report, const std::string& warning) {
    if (std::find(report.warnings.begin(), report.warnings.end(), warning) == report.warnings.end()) {
        report.warnings.push_back(warning);
    }
}

}  // namespace

bool build_surface_ibm_geometry_impl(const Mesh& mesh,
                                     const Config& config,
                                     std::vector<Marker>& markers,
                                     GeometryReport& report,
                                     std::string& error,
                                     const GeometrySlab* slab) {
    markers.clear();
    report = GeometryReport{};
    error.clear();
    if (!validate_config(config, error)) return false;
    if (slab && !(std::isfinite(slab->lower_z_lu) &&
                  std::isfinite(slab->upper_z_lu) &&
                  slab->lower_z_lu < slab->upper_z_lu)) {
        error = "surface-IBM quadrature slab is invalid";
        return false;
    }

    report.input_triangles = static_cast<std::size_t>(mesh.triangle_number);
    if (mesh.triangle_number == 0) {
        error = "surface-IBM geometry contains no triangles";
        return false;
    }
    if (mesh.p0 == nullptr || mesh.p1 == nullptr || mesh.p2 == nullptr) {
        error = "surface-IBM geometry has null triangle storage";
        return false;
    }

    std::vector<Triangle> triangles;
    triangles.reserve(mesh.triangle_number);
    std::vector<bool> cancelled_triangles;
    cancelled_triangles.reserve(mesh.triangle_number);
    std::unordered_map<TriangleKey, std::size_t, TriangleKeyHash> unique_triangles;
    unique_triangles.reserve(static_cast<std::size_t>(mesh.triangle_number) * 2U);

    for (std::uint32_t source = 0; source < mesh.triangle_number; ++source) {
        const float3& af = mesh.p0[source];
        const float3& bf = mesh.p1[source];
        const float3& cf = mesh.p2[source];
        const Vec3d a = to_vec3d(af);
        const Vec3d b = to_vec3d(bf);
        const Vec3d c = to_vec3d(cf);
        if (!finite(a) || !finite(b) || !finite(c)) {
            ++report.degenerate_triangles;
            if (config.strict_geometry) {
                error = "surface-IBM geometry contains non-finite vertices";
                return false;
            }
            continue;
        }
        const double ab = norm(b - a);
        const double bc = norm(c - b);
        const double ca = norm(a - c);
        const double longest = std::max({ab, bc, ca});
        const Vec3d cross_product = cross(b - a, c - a);
        const double twice_area = norm(cross_product);
        const double degeneracy_threshold = std::max(1.0e-24, longest * longest * 1.0e-12);
        if (!(longest > 0.0) || !std::isfinite(twice_area) || twice_area <= degeneracy_threshold) {
            ++report.degenerate_triangles;
            if (config.strict_geometry) {
                error = "surface-IBM geometry contains degenerate triangles";
                return false;
            }
            continue;
        }
        report.input_area_lu2 += 0.5 * twice_area;
        const TriangleKey key = triangle_key(af, bf, cf);
        const auto duplicate = unique_triangles.find(key);
        if (duplicate != unique_triangles.end()) {
            ++report.duplicate_triangles;
            // Coincident, opposite-winding facets are the common representation
            // of the shared internal face between two touching closed solids.
            // Under exterior-union semantics they cancel exactly. Same-winding
            // duplicates remain a single deterministic facet.
            const Vec3d normal = cross_product / twice_area;
            if (config.geometry_policy == GeometryPolicy::ExteriorUnion &&
                dot(triangles[duplicate->second].normal, normal) < -0.999999) {
                cancelled_triangles[duplicate->second] = true;
                unique_triangles.erase(duplicate);
                ++report.coincident_union_pairs;
            }
            continue;
        }
        Triangle triangle;
        triangle.a = a;
        triangle.b = b;
        triangle.c = c;
        triangle.normal = cross_product / twice_area;
        triangle.area = 0.5 * twice_area;
        triangle.max_edge = longest;
        triangle.source = source;
        triangle.vertex_keys = {vertex_key(af), vertex_key(bf), vertex_key(cf)};
        unique_triangles.emplace(key, triangles.size());
        triangles.push_back(triangle);
        cancelled_triangles.push_back(false);
    }
    if (std::find(cancelled_triangles.begin(), cancelled_triangles.end(), true) !=
        cancelled_triangles.end()) {
        std::vector<Triangle> exterior_candidates;
        exterior_candidates.reserve(triangles.size());
        for (std::size_t i = 0; i < triangles.size(); ++i) {
            if (!cancelled_triangles[i]) {
                exterior_candidates.push_back(std::move(triangles[i]));
            }
        }
        triangles = std::move(exterior_candidates);
    }
    report.valid_triangles = triangles.size();
    for (const Triangle& triangle : triangles) {
        report.valid_area_lu2 += triangle.area;
    }

    if (report.degenerate_triangles != 0) {
        add_warning_once(report, "degenerate or non-finite triangles were skipped");
    }
    if (report.duplicate_triangles != 0) {
        add_warning_once(report,
            config.geometry_policy == GeometryPolicy::ExteriorUnion
                ? "exact duplicates were collapsed and opposite-winding coincident union faces were cancelled"
                : "exact duplicate triangles were collapsed");
        if (config.strict_geometry &&
            report.duplicate_triangles != report.coincident_union_pairs) {
            error = "surface-IBM strict geometry rejects exact duplicate triangles";
            return false;
        }
    }
    if (triangles.empty()) {
        error = "surface-IBM geometry contains no usable triangles";
        return false;
    }

    DisjointSet sets(triangles.size());
    std::unordered_map<EdgeKey, EdgeRecord, EdgeKeyHash> edges;
    edges.reserve(triangles.size() * 2U);
    for (std::size_t triangle_index = 0; triangle_index < triangles.size(); ++triangle_index) {
        const auto& keys = triangles[triangle_index].vertex_keys;
        const std::array<EdgeKey, 3> triangle_edges = {
            edge_key(keys[0], keys[1]),
            edge_key(keys[1], keys[2]),
            edge_key(keys[2], keys[0]),
        };
        for (const EdgeKey& key : triangle_edges) {
            auto inserted = edges.emplace(key, EdgeRecord{triangle_index, 1});
            if (!inserted.second) {
                sets.unite(inserted.first->second.first_triangle, triangle_index);
                ++inserted.first->second.count;
            }
        }
    }

    std::unordered_map<std::size_t, std::uint32_t> component_ids;
    component_ids.reserve(triangles.size());
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        const std::size_t root = sets.find(i);
        auto inserted = component_ids.emplace(root, static_cast<std::uint32_t>(component_ids.size()));
        triangles[i].component = inserted.first->second;
    }
    report.connected_components = component_ids.size();
    if (component_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        error = "surface-IBM connected-component count overflow";
        return false;
    }

    std::vector<bool> component_open(component_ids.size(), false);
    std::vector<bool> component_nonmanifold(component_ids.size(), false);
    for (const auto& entry : edges) {
        const EdgeRecord& edge = entry.second;
        const std::uint32_t component_id = triangles[edge.first_triangle].component;
        if (edge.count == 1) {
            ++report.open_edges;
            component_open[component_id] = true;
        } else if (edge.count > 2) {
            ++report.nonmanifold_edges;
            component_nonmanifold[component_id] = true;
        }
    }
    std::vector<bool> watertight_components(component_ids.size(), true);
    for (std::size_t i = 0; i < component_ids.size(); ++i) {
        watertight_components[i] = !component_open[i] && !component_nonmanifold[i];
        if (watertight_components[i]) ++report.watertight_components;
    }
    if (report.open_edges != 0) {
        add_warning_once(report,
            config.geometry_policy == GeometryPolicy::ExteriorUnion
                ? "open surface components are incompatible with exterior_union"
                : "open surface components are retained by all_triangles");
    }
    if (report.nonmanifold_edges != 0) {
        add_warning_once(report,
            config.geometry_policy == GeometryPolicy::ExteriorUnion
                ? "non-manifold components are incompatible with exterior_union"
                : "non-manifold components are retained by all_triangles");
    }
    if (config.strict_geometry && (report.open_edges != 0 || report.nonmanifold_edges != 0)) {
        std::ostringstream message;
        message << "surface-IBM strict geometry rejected " << report.open_edges
                << " open edges and " << report.nonmanifold_edges << " non-manifold edges";
        error = message.str();
        return false;
    }
    if (config.geometry_policy == GeometryPolicy::ExteriorUnion &&
        (report.open_edges != 0 || report.nonmanifold_edges != 0)) {
        std::ostringstream message;
        message << "surface-IBM exterior_union requires watertight manifold "
                << "components; found " << report.open_edges
                << " open edges and " << report.nonmanifold_edges
                << " non-manifold edges. Repair/boolean-union the STL or use "
                   "all_triangles only for an intentionally open immersed sheet";
        error = message.str();
        return false;
    }

    if (triangles.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) / 2U) {
        error = "surface-IBM triangle BVH index range overflow";
        return false;
    }
    TriangleBvh bvh(triangles, watertight_components);
    const TriangleBvh* classifier =
        config.geometry_policy == GeometryPolicy::ExteriorUnion ? &bvh : nullptr;

    std::vector<TriangleStatus> statuses(triangles.size());
    std::vector<PolygonPatch> subdivision_stack;
    subdivision_stack.reserve(64);
    std::uint64_t candidate_patch_count = 0;
    for (std::size_t triangle_index = 0; triangle_index < triangles.size(); ++triangle_index) {
        const Triangle& triangle = triangles[triangle_index];
        TriangleStatus& status = statuses[triangle_index];
        if (!slab) {
            if (!append_subdivided_triangle(triangle, watertight_components[triangle.component],
                    classifier, config, markers, report, status, candidate_patch_count,
                    subdivision_stack, error)) {
                markers.clear();
                report.markers = 0;
                return false;
            }
            continue;
        }
        // Classify against the intact watertight solid, but subdivide only
        // the part of its surface owned by this slab. No artificial cap is
        // introduced at an interface, and large foundation faces are never
        // tessellated when the outer ground uses bounce-back.
        std::vector<Vec3d> polygon{triangle.a, triangle.b, triangle.c};
        if (slab) {
            const double low = std::min({triangle.a.z, triangle.b.z, triangle.c.z});
            const double high = std::max({triangle.a.z, triangle.b.z, triangle.c.z});
            if (low >= slab->upper_z_lu || high < slab->lower_z_lu ||
                (!slab->include_lower_plane && high <= slab->lower_z_lu)) continue;
            for (int side = 0; side < 2; ++side) {
                if (polygon.empty()) break;
                const double cut = side == 0 ? slab->lower_z_lu : slab->upper_z_lu;
                std::vector<Vec3d> clipped;
                Vec3d previous = polygon.back();
                bool previous_inside = side == 0 ? previous.z >= cut : previous.z <= cut;
                for (const Vec3d& current : polygon) {
                    const bool inside = side == 0 ? current.z >= cut : current.z <= cut;
                    if (inside != previous_inside) {
                        Vec3d point = previous + (current - previous) *
                            ((cut - previous.z) / (current.z - previous.z));
                        point.z = cut;
                        clipped.push_back(point);
                    }
                    if (inside) clipped.push_back(current);
                    previous = current;
                    previous_inside = inside;
                }
                polygon = std::move(clipped);
            }
        }
        for (std::size_t vertex = 1; vertex + 1 < polygon.size(); ++vertex) {
            Triangle patch = triangle;
            patch.a = polygon[0]; patch.b = polygon[vertex]; patch.c = polygon[vertex+1];
            patch.area = 0.5 * norm(cross(patch.b-patch.a, patch.c-patch.a));
            patch.max_edge = std::max({norm(patch.b-patch.a), norm(patch.c-patch.b),
                                       norm(patch.a-patch.c)});
            if (!(patch.area > 1.0e-20)) continue;
            if (!append_subdivided_triangle(
                    patch, watertight_components[triangle.component], classifier,
                    config, markers, report, status, candidate_patch_count,
                    subdivision_stack, error)) {
                markers.clear();
                report.markers = 0;
                return false;
            }
        }
    }

    for (const TriangleStatus& status : statuses) {
        if (status.retained) ++report.retained_triangles;
        if (status.internal) ++report.internal_triangles;
        if (status.ambiguous) ++report.ambiguous_triangles;
    }
    report.markers = markers.size();
    if (report.ambiguous_patches != 0) {
        add_warning_once(report, "ambiguous patches were retained conservatively");
    }
    if (markers.empty() && !slab) {
        error = "surface-IBM geometry produced no exterior markers";
        return false;
    }
    return true;
}

bool build_surface_ibm_geometry(const Mesh& mesh,
                                const Config& config,
                                std::vector<Marker>& markers,
                                GeometryReport& report,
                                std::string& error,
                                const GeometrySlab* slab) {
    try {
        return build_surface_ibm_geometry_impl(mesh, config, markers, report, error, slab);
    } catch (const std::bad_alloc&) {
        markers.clear();
        report.markers = 0;
        error = "surface-IBM geometry construction exhausted host memory";
        return false;
    } catch (const std::length_error& exception) {
        markers.clear();
        report.markers = 0;
        error = std::string("surface-IBM geometry allocation overflow: ") + exception.what();
        return false;
    }
}

}  // namespace culbm::surface_ibm
