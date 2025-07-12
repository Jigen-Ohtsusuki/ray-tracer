#include <SDL.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <numeric>

// ImGui includes
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_sdlrenderer2.h>

// For file dialogs
#ifdef _WIN32
#include <windows.h>
#include <commdlg.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// FPS tracking
Uint32 last_frame_time = 0;
float fps = 0.0f;
float frame_times[60] = {0};
int frame_time_index = 0;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

template <typename T>
__host__ __device__ T clamp(T x, T a, T b) {
    return fmaxf(a, fminf(x, b));
}

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }
    __host__ __device__ Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator*(Vec3 o) const { return {x * o.x, y * o.y, z * o.z}; }
    __host__ __device__ Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ float dot(Vec3 o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(Vec3 o) const { return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x}; }

    __host__ __device__ Vec3 normalized() const {
        float len = sqrtf(x * x + y * y + z * z);
        if (len == 0) return {0, 0, 0};
        return {x / len, y / len, z / len};
    }
    
    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Triangle {
    Vec3 v0, v1, v2;
    Vec3 normal;
    Vec3 color;
    int mesh_index;
    
    __host__ __device__ Triangle() : mesh_index(-1) {}
    __host__ __device__ Triangle(Vec3 a, Vec3 b, Vec3 c, Vec3 col, int mi = -1) : v0(a), v1(b), v2(c), color(col), mesh_index(mi) {
        normal = (v1 - v0).cross(v2 - v0).normalized();
    }
};

struct Mesh {
    std::vector<Triangle> triangles;
    Vec3 position;
    Vec3 rotation;
    Vec3 scale;
    Vec3 color;
    std::string name;
    bool selected;
    bool dragging;
    
    Mesh() : position(0, 0, 0), rotation(0, 0, 0), scale(1, 1, 1), color(0.7f, 0.7f, 0.7f), selected(false), dragging(false) {}
};

struct Light {
    Vec3 position;
    Vec3 color;
    float intensity;
    bool dragging;
    
    __host__ __device__ Light() : position(5, 10, 5), color(1.0f, 0.95f, 0.9f), intensity(10.0f), dragging(false) {} // Sun-like defaults
};

struct FloorPlane {
    float y;
    Vec3 color1, color2;
    float tile_size;
    
    __host__ __device__ FloorPlane() : y(-2.0f), color1(0.8f, 0.8f, 0.8f), color2(0.6f, 0.6f, 0.6f), tile_size(2.0f) {}
};

// Global variables
int WIDTH = 800;
int HEIGHT = 600;
int RENDER_WIDTH = 400;
int RENDER_HEIGHT = 300;
const float FOV = 60.0f;

std::vector<Mesh> meshes;
Light light;
FloorPlane floor_plane;
Vec3 cam_pos = {0, 0, 5};
float cam_yaw = 0.0f, cam_pitch = 0.0f;
bool show_gui = true;
int selected_mesh = -1;

// GPU data
Triangle* d_triangles = nullptr;
int num_triangles = 0;
Light* d_light = nullptr;
FloorPlane* d_floor_plane = nullptr;
uint32_t* d_pixel_buffer = nullptr;
int* d_selected_mesh = nullptr;

// Resolution presets
std::vector<std::pair<int, int>> resolutions = {
    {320, 240}, {640, 480}, {800, 600}, {1024, 768}, {1280, 720}, {1920, 1080}
};
int current_resolution = 2;

std::string open_file_dialog() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char szFile[260] = {0};
    
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.lpstrFile = szFile;
    ofn.nMaxFile = sizeof(szFile);
    ofn.lpstrFilter = "OBJ Files\0*.obj\0All Files\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.lpstrFileTitle = NULL;
    ofn.nMaxFileTitle = 0;
    ofn.lpstrInitialDir = NULL;
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    
    if (GetOpenFileNameA(&ofn)) {
        return std::string(szFile);
    }
#endif
    return "";
}

bool load_obj(const std::string& filename, Mesh& mesh, int mesh_index) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    std::vector<Vec3> vertices;
    std::vector<std::vector<int>> faces;
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        
        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back({x, y, z});
        }
        else if (prefix == "f") {
            std::vector<int> face;
            std::string vertex;
            while (iss >> vertex) {
                size_t slash_pos = vertex.find('/');
                std::string v_str = vertex.substr(0, slash_pos);
                int v_idx = std::stoi(v_str) - 1;
                face.push_back(v_idx);
            }
            faces.push_back(face);
        }
    }
    
    mesh.triangles.clear();
    for (const auto& face : faces) {
        if (face.size() >= 3) {
            for (int i = 1; i < face.size() - 1; ++i) {
                if (face[0] < vertices.size() && face[i] < vertices.size() && face[i + 1] < vertices.size()) {
                    mesh.triangles.emplace_back(
                        vertices[face[0]], vertices[face[i]], vertices[face[i + 1]], mesh.color, mesh_index
                    );
                }
            }
        }
    }
    
    std::cout << "Loaded " << vertices.size() << " vertices, " << mesh.triangles.size() << " triangles" << std::endl;
    return true;
}

void create_cube(Mesh& mesh, Vec3 size, int mesh_index) {
    mesh.triangles.clear();
    Vec3 s = size * 0.5f;
    
    Vec3 verts[8] = {
        {-s.x, -s.y, -s.z}, {s.x, -s.y, -s.z}, {s.x, s.y, -s.z}, {-s.x, s.y, -s.z},
        {-s.x, -s.y, s.z}, {s.x, -s.y, s.z}, {s.x, s.y, s.z}, {-s.x, s.y, s.z}
    };
    
    int faces[12][3] = {
        {0,1,2}, {0,2,3}, {4,7,6}, {4,6,5}, {0,4,5}, {0,5,1},
        {2,6,7}, {2,7,3}, {0,3,7}, {0,7,4}, {1,5,6}, {1,6,2}
    };
    
    for (int i = 0; i < 12; ++i) {
        mesh.triangles.emplace_back(verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]], mesh.color, mesh_index);
    }
}

void create_sphere(Mesh& mesh, float radius, int mesh_index, int segments = 16) {
    mesh.triangles.clear();
    
    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < segments; ++j) {
            float theta1 = (float)i * M_PI / segments;
            float theta2 = (float)(i + 1) * M_PI / segments;
            float phi1 = (float)j * 2 * M_PI / segments;
            float phi2 = (float)(j + 1) * 2 * M_PI / segments;
            
            Vec3 v1 = {radius * sinf(theta1) * cosf(phi1), radius * cosf(theta1), radius * sinf(theta1) * sinf(phi1)};
            Vec3 v2 = {radius * sinf(theta1) * cosf(phi2), radius * cosf(theta1), radius * sinf(theta1) * sinf(phi2)};
            Vec3 v3 = {radius * sinf(theta2) * cosf(phi1), radius * cosf(theta2), radius * sinf(theta2) * sinf(phi1)};
            Vec3 v4 = {radius * sinf(theta2) * cosf(phi2), radius * cosf(theta2), radius * sinf(theta2) * sinf(phi2)};
            
            if (i != 0) {
                mesh.triangles.emplace_back(v1, v2, v3, mesh.color, mesh_index);
            }
            if (i != segments - 1) {
                mesh.triangles.emplace_back(v2, v4, v3, mesh.color, mesh_index);
            }
        }
    }
}

__host__ __device__ Vec3 rotate_y_vector(Vec3 v, float cos_y, float sin_y) {
    return Vec3(v.x * cos_y + v.z * sin_y, v.y, -v.x * sin_y + v.z * cos_y);
}

__host__ __device__ Triangle transform_triangle(const Triangle& tri, const Vec3& position, const Vec3& rotation, const Vec3& scale) {
    Vec3 v0 = tri.v0 * scale;
    Vec3 v1 = tri.v1 * scale;
    Vec3 v2 = tri.v2 * scale;
    
    float cos_y = cosf(rotation.y);
    float sin_y = sinf(rotation.y);
    
    v0 = rotate_y_vector(v0, cos_y, sin_y);
    v1 = rotate_y_vector(v1, cos_y, sin_y);
    v2 = rotate_y_vector(v2, cos_y, sin_y);
    
    v0 = v0 + position;
    v1 = v1 + position;
    v2 = v2 + position;
    
    return Triangle(v0, v1, v2, tri.color, tri.mesh_index);
}

__device__ bool intersect_triangle(const Ray& ray, const Triangle& tri, float& t, Vec3& normal) {
    const float EPSILON = 1e-8f;
    Vec3 edge1 = tri.v1 - tri.v0;
    Vec3 edge2 = tri.v2 - tri.v0;
    Vec3 h = ray.direction.cross(edge2);
    float a = edge1.dot(h);
    
    if (a > -EPSILON && a < EPSILON) return false;
    
    float f = 1.0f / a;
    Vec3 s = ray.origin - tri.v0;
    float u = f * s.dot(h);
    
    if (u < 0.0f || u > 1.0f) return false;
    
    Vec3 q = s.cross(edge1);
    float v = f * ray.direction.dot(q);
    
    if (v < 0.0f || u + v > 1.0f) return false;
    
    t = f * edge2.dot(q);
    
    if (t > EPSILON) {
        normal = tri.normal;
        return true;
    }
    
    return false;
}

__device__ bool intersect_floor(const Ray& ray, float& t, Vec3& normal, Vec3& color, const FloorPlane& floor_plane) {
    if (fabsf(ray.direction.y) < 1e-6f) return false;
    
    t = (floor_plane.y - ray.origin.y) / ray.direction.y;
    if (t < 0) return false;
    
    Vec3 hit_point = ray.origin + ray.direction * t;
    
    int x_tile = (int)floorf(hit_point.x / floor_plane.tile_size);
    int z_tile = (int)floorf(hit_point.z / floor_plane.tile_size);
    bool checker = (x_tile + z_tile) % 2 == 0;
    
    color = checker ? floor_plane.color1 : floor_plane.color2;
    normal = {0, 1, 0};
    return true;
}

__device__ Vec3 get_sky_color(const Vec3& ray_dir, const Light& light) {
    float t = 0.5f * (ray_dir.y + 1.0f);
    Vec3 horizon_color = {1.0f, 1.0f, 1.0f};
    Vec3 zenith_color = {0.4f, 0.6f, 1.0f};
    
    Vec3 light_dir = light.position.normalized();
    float sun_dot = fmaxf(0.0f, ray_dir.dot(light_dir));
    float sun_glow = powf(sun_dot, 32.0f) * 0.5f; // Increased glow for brighter sun
    Vec3 sun_color = light.color; // Use light color for sun
    
    Vec3 base_color = horizon_color * (1.0f - t) + zenith_color * t;
    return base_color + sun_color * sun_glow;
}

__device__ bool is_in_shadow(Vec3 point, Vec3 light_pos, const Triangle* triangles, int num_triangles) {
    Vec3 light_dir = (light_pos - point).normalized();
    float light_distance = (light_pos - point).length();
    Ray shadow_ray = {point + light_dir * 0.001f, light_dir};
    
    for (int i = 0; i < num_triangles; ++i) {
        float t;
        Vec3 normal;
        if (intersect_triangle(shadow_ray, triangles[i], t, normal) && t < light_distance) {
            return true;
        }
    }
    return false;
}

__device__ Vec3 trace(const Ray& ray, const Triangle* triangles, int num_triangles, const Light& light, const FloorPlane& floor_plane, int selected_mesh, int depth = 0) {
    float closest_t = 1e9;
    Vec3 hit_normal, hit_color;
    Vec3 hit_point;
    bool hit = false;
    int hit_mesh_index = -1;

    // Check floor intersection
    float floor_t;
    Vec3 floor_normal;
    Vec3 floor_color;
    if (intersect_floor(ray, floor_t, floor_normal, floor_color, floor_plane) && floor_t < closest_t) {
        closest_t = floor_t;
        hit = true;
        hit_normal = floor_normal;
        hit_color = floor_color;
        hit_point = ray.origin + ray.direction * floor_t;
    }

    // Check triangle intersections
    for (int i = 0; i < num_triangles; ++i) {
        float t;
        Vec3 normal;
        if (intersect_triangle(ray, triangles[i], t, normal) && t < closest_t) {
            closest_t = t;
            hit = true;
            hit_normal = normal;
            hit_color = triangles[i].color;
            hit_point = ray.origin + ray.direction * t;
            hit_mesh_index = triangles[i].mesh_index;
        }
    }

    if (!hit) {
        return get_sky_color(ray.direction, light);
    }

    // Highlight selected mesh
    if (hit_mesh_index == selected_mesh && selected_mesh != -1) {
        hit_color = hit_color * 1.5f;
        hit_color.x = clamp(hit_color.x, 0.0f, 1.0f);
        hit_color.y = clamp(hit_color.y, 0.0f, 1.0f);
        hit_color.z = clamp(hit_color.z, 0.0f, 1.0f);
    }

    // Light direction and distance
    Vec3 light_dir = (light.position - hit_point).normalized();
    float light_distance = (light.position - hit_point).length();
    
    // Sun-like attenuation: softer falloff for distant light
    float attenuation = 1.0f / (1.0f + 0.01f * light_distance + 0.001f * light_distance * light_distance);

    // Shadow test
    bool in_shadow = is_in_shadow(hit_point, light.position, triangles, num_triangles);
    
    // Diffuse lighting calculation
    float diffuse_strength = fmaxf(0.0f, hit_normal.dot(light_dir));
    
    // Lighting model: enhanced for sunlight
    Vec3 ambient = Vec3(0.4f, 0.6f, 1.0f) * 0.15f * fmaxf(0.0f, hit_normal.y); // Reduced ambient for contrast
    Vec3 diffuse = light.color * light.intensity * diffuse_strength * attenuation;
    
    if (in_shadow) {
        diffuse = diffuse * 0.05f; // Sharper shadows for sunlight
    }
    
    // Combine lighting
    Vec3 total_lighting = ambient + diffuse;
    
    // Apply material color
    Vec3 final_color = hit_color * total_lighting;
    
    // Clamp colors
    final_color.x = clamp(final_color.x, 0.0f, 1.0f);
    final_color.y = clamp(final_color.y, 0.0f, 1.0f);
    final_color.z = clamp(final_color.z, 0.0f, 1.0f);
    
    return final_color;
}

__device__ Vec3 get_ray_dir(int x, int y, float fov, float aspect, float yaw, float pitch, int render_width, int render_height) {
    float px = (2 * ((x + 0.5f) / render_width) - 1) * aspect * tanf(fov * 0.5f * M_PI / 180);
    float py = (1 - 2 * ((y + 0.5f) / render_height)) * tanf(fov * 0.5f * M_PI / 180);
    Vec3 dir = {px, py, -1};

    float cp = cosf(pitch), sp = sinf(pitch);
    float cy = cosf(yaw), sy = sinf(yaw);

    float dy = dir.y * cp - dir.z * sp;
    float dz = dir.y * sp + dir.z * cp;
    dir.y = dy;
    dir.z = dz;

    float dx = dir.x * cy + dir.z * sy;
    dz = -dir.x * sy + dir.z * cy;
    dir.x = dx;
    dir.z = dz;

    return dir.normalized();
}

__global__ void render_kernel(uint32_t* pixel_buffer, Triangle* triangles, int num_triangles, Light light, FloorPlane floor_plane, 
                             Vec3 cam_pos, float cam_yaw, float cam_pitch, float fov, int render_width, int render_height, int selected_mesh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= render_width || y >= render_height) return;
    
    float aspect = (float)render_width / render_height;
    Vec3 ray_dir = get_ray_dir(x, y, fov, aspect, cam_yaw, cam_pitch, render_width, render_height);
    Ray ray = {cam_pos, ray_dir};
    Vec3 color = trace(ray, triangles, num_triangles, light, floor_plane, selected_mesh);
    
    uint8_t r = (uint8_t)(clamp(color.x, 0.0f, 1.0f) * 255.0f);
    uint8_t g = (uint8_t)(clamp(color.y, 0.0f, 1.0f) * 255.0f);
    uint8_t b = (uint8_t)(clamp(color.z, 0.0f, 1.0f) * 255.0f);
    
    int index = y * render_width + x;
    pixel_buffer[index] = (r << 16) | (g << 8) | b;
}

void update_gpu_scene_data() {
    if (d_triangles) CUDA_CHECK(cudaFree(d_triangles));
    
    std::vector<Triangle> all_triangles;
    for (size_t i = 0; i < meshes.size(); ++i) {
        for (auto& tri : meshes[i].triangles) {
            tri.mesh_index = i;
            Triangle transformed = transform_triangle(tri, meshes[i].position, meshes[i].rotation, meshes[i].scale);
            all_triangles.push_back(transformed);
        }
    }
    
    num_triangles = all_triangles.size();
    
    if (num_triangles > 0) {
        CUDA_CHECK(cudaMalloc(&d_triangles, num_triangles * sizeof(Triangle)));
        CUDA_CHECK(cudaMemcpy(d_triangles, all_triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice));
    }
    
    CUDA_CHECK(cudaMemcpy(d_light, &light, sizeof(Light), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_floor_plane, &floor_plane, sizeof(FloorPlane), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_selected_mesh, &selected_mesh, sizeof(int), cudaMemcpyHostToDevice));
}

void render_gui() {
    if (!show_gui) return;
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(200, 80), ImGuiCond_Always);
    ImGui::Begin("Performance", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame Time: %.2f ms", 1000.0f / fps);
    ImGui::Text("Resolution: %dx%d", RENDER_WIDTH, RENDER_HEIGHT);
    ImGui::End();
    
    ImGui::Begin("Ray Tracer Controls", &show_gui);
    
    if (ImGui::CollapsingHeader("Resolution")) {
        ImGui::PushID("resolution_section");
        const char* res_items[] = {"320x240", "640x480", "800x600", "1024x768", "1280x720", "1920x1080"};
        if (ImGui::Combo("Preset", &current_resolution, res_items, 6)) {
            RENDER_WIDTH = resolutions[current_resolution].first;
            RENDER_HEIGHT = resolutions[current_resolution].second;
        }
        
        ImGui::SliderInt("Width", &RENDER_WIDTH, 160, 1920);
        ImGui::SliderInt("Height", &RENDER_HEIGHT, 120, 1080);
        ImGui::PopID();
    }
    
    if (ImGui::CollapsingHeader("Camera")) {
        ImGui::PushID("camera_section");
        ImGui::SliderFloat3("Position", &cam_pos.x, -10, 10);
        ImGui::SliderFloat("Yaw", &cam_yaw, -M_PI, M_PI);
        ImGui::SliderFloat("Pitch", &cam_pitch, -M_PI/2, M_PI/2);
        if (ImGui::Button("Reset Camera")) {
            cam_pos = {0, 0, 5};
            cam_yaw = cam_pitch = 0;
        }
        ImGui::PopID();
    }
    
    if (ImGui::CollapsingHeader("Light")) {
        ImGui::PushID("light_section");
        ImGui::SliderFloat3("Position", &light.position.x, -20, 20); // Expanded range for sunlight
        ImGui::ColorEdit3("Color", &light.color.x);
        ImGui::SliderFloat("Intensity", &light.intensity, 0.1f, 20.0f); // Increased max for sunlight
        ImGui::Checkbox("Dragging", &light.dragging);
        if (ImGui::Button("Reset to Sunlight")) {
            light.position = {5, 10, 5}; // High position for sun-like angle
            light.color = {1.0f, 0.95f, 0.9f}; // Sunlight color
            light.intensity = 10.0f; // High intensity
        }
        ImGui::PopID();
    }
    
    if (ImGui::CollapsingHeader("Floor")) {
        ImGui::PushID("floor_section");
        ImGui::SliderFloat("Y Position", &floor_plane.y, -5.0f, 0.0f);
        ImGui::ColorEdit3("Color 1", &floor_plane.color1.x);
        ImGui::ColorEdit3("Color 2", &floor_plane.color2.x);
        ImGui::SliderFloat("Tile Size", &floor_plane.tile_size, 0.5f, 5.0f);
        ImGui::PopID();
    }
    
    if (ImGui::CollapsingHeader("3D Objects")) {
        ImGui::PushID("objects_section");
        
        if (ImGui::Button("Add Cube")) {
            Mesh new_mesh;
            new_mesh.name = "Cube_" + std::to_string(meshes.size());
            new_mesh.color = {0.7f, 0.3f, 0.3f};
            new_mesh.position = {0, 0, -3};
            create_cube(new_mesh, {1, 1, 1}, meshes.size());
            meshes.push_back(new_mesh);
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Add Sphere")) {
            Mesh new_mesh;
            new_mesh.name = "Sphere_" + std::to_string(meshes.size());
            new_mesh.color = {0.3f, 0.3f, 0.7f};
            new_mesh.position = {0, 0, -3};
            create_sphere(new_mesh, 1.0f, meshes.size());
            meshes.push_back(new_mesh);
        }
        
        if (ImGui::Button("Load OBJ File")) {
            std::string filename = open_file_dialog();
            if (!filename.empty()) {
                Mesh new_mesh;
                new_mesh.name = "Loaded_" + std::to_string(meshes.size());
                new_mesh.color = {0.3f, 0.7f, 0.3f};
                new_mesh.position = {0, 0, -3};
                if (load_obj(filename, new_mesh, meshes.size())) {
                    meshes.push_back(new_mesh);
                    std::cout << "Successfully loaded: " << filename << std::endl;
                } else {
                    std::cout << "Failed to load: " << filename << std::endl;
                }
            }
        }
        
        ImGui::Separator();
        
        ImGui::Text("Objects (%d):", (int)meshes.size());
        for (int i = 0; i < meshes.size(); ++i) {
            ImGui::PushID(i);
            bool is_selected = (selected_mesh == i);
            if (ImGui::Selectable(meshes[i].name.c_str(), is_selected)) {
                selected_mesh = i;
            }
            ImGui::PopID();
        }

        if (selected_mesh >= 0 && selected_mesh < meshes.size()) {
            ImGui::Separator();
            ImGui::Text("Selected: %s", meshes[selected_mesh].name.c_str());
            if (ImGui::Button("Delete Selected Object")) {
                meshes.erase(meshes.begin() + selected_mesh);
                selected_mesh = -1;
            }
        }
        
        if (selected_mesh >= 0 && selected_mesh < meshes.size()) {
            ImGui::Separator();
            ImGui::Text("Edit: %s", meshes[selected_mesh].name.c_str());
            
            ImGui::PushID("selected_mesh_edit");
            
            ImGui::Text("Position:");
            ImGui::SliderFloat3("##pos", &meshes[selected_mesh].position.x, -10, 10);
            
            ImGui::Text("Quick Move:");
            if (ImGui::Button("Left")) meshes[selected_mesh].position.x -= 0.5f;
            ImGui::SameLine();
            if (ImGui::Button("Right")) meshes[selected_mesh].position.x += 0.5f;
            ImGui::SameLine();
            if (ImGui::Button("Up")) meshes[selected_mesh].position.y += 0.5f;
            ImGui::SameLine();
            if (ImGui::Button("Down")) meshes[selected_mesh].position.y -= 0.5f;
            
            if (ImGui::Button("Forward")) meshes[selected_mesh].position.z -= 0.5f;
            ImGui::SameLine();
            if (ImGui::Button("Back")) meshes[selected_mesh].position.z += 0.5f;
            ImGui::Text("Rotation:");
            ImGui::SliderFloat3("##rot", &meshes[selected_mesh].rotation.x, -M_PI, M_PI);
            
            ImGui::Text("Scale:");
            ImGui::SliderFloat3("##scale", &meshes[selected_mesh].scale.x, 0.1f, 3.0f);
            
            ImGui::Text("Color:");
            ImGui::ColorEdit3("##color", &meshes[selected_mesh].color.x);
            
            ImGui::PopID();
        }
        
        ImGui::PopID();
    }
    
    if (ImGui::CollapsingHeader("Performance")) {
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Frame Time: %.2f ms", 1000.0f / fps);
        ImGui::Text("Render Resolution: %dx%d", RENDER_WIDTH, RENDER_HEIGHT);
        ImGui::Text("Window Resolution: %dx%d", WIDTH, HEIGHT);
        ImGui::Text("Total Triangles: %d", num_triangles);
        ImGui::Text("Meshes: %d", (int)meshes.size());
    
        ImGui::PlotLines("Frame Times", frame_times, 60, frame_time_index, nullptr, 0.0f, 0.05f, ImVec2(0, 80));
    }
    
    ImGui::End();
}

void initialize_gpu() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(1);
    }
    
    CUDA_CHECK(cudaMalloc(&d_light, sizeof(Light)));
    CUDA_CHECK(cudaMalloc(&d_floor_plane, sizeof(FloorPlane)));
    CUDA_CHECK(cudaMalloc(&d_selected_mesh, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pixel_buffer, RENDER_WIDTH * RENDER_HEIGHT * sizeof(uint32_t)));
}

void cleanup_gpu() {
    if (d_triangles) {
        cudaFree(d_triangles);
        d_triangles = nullptr;
    }
    if (d_light) {
        cudaFree(d_light);
        d_light = nullptr;
    }
    if (d_floor_plane) {
        cudaFree(d_floor_plane);
        d_floor_plane = nullptr;
    }
    if (d_pixel_buffer) {
        cudaFree(d_pixel_buffer);
        d_pixel_buffer = nullptr;
    }
    if (d_selected_mesh) {
        cudaFree(d_selected_mesh);
        d_selected_mesh = nullptr;
    }
}

void render_scene(SDL_Renderer* renderer, SDL_Texture*& texture) {
    static int last_width = 0, last_height = 0;
    if (RENDER_WIDTH != last_width || RENDER_HEIGHT != last_height) {
        if (d_pixel_buffer) CUDA_CHECK(cudaFree(d_pixel_buffer));
        CUDA_CHECK(cudaMalloc(&d_pixel_buffer, RENDER_WIDTH * RENDER_HEIGHT * sizeof(uint32_t)));
        last_width = RENDER_WIDTH;
        last_height = RENDER_HEIGHT;
        
        if (texture) SDL_DestroyTexture(texture);
        texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, RENDER_WIDTH, RENDER_HEIGHT);
        if (!texture) {
            std::cerr << "Failed to create texture: " << SDL_GetError() << std::endl;
            return;
        }
    }
    
    update_gpu_scene_data();
    
    dim3 blockSize(16, 16);
    dim3 gridSize((RENDER_WIDTH + blockSize.x - 1) / blockSize.x, (RENDER_HEIGHT + blockSize.y - 1) / blockSize.y);
    
    render_kernel<<<gridSize, blockSize>>>(d_pixel_buffer, d_triangles, num_triangles, light, floor_plane, 
                                     cam_pos, cam_yaw, cam_pitch, FOV, RENDER_WIDTH, RENDER_HEIGHT, selected_mesh);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<uint32_t> pixel_data(RENDER_WIDTH * RENDER_HEIGHT);
    CUDA_CHECK(cudaMemcpy(pixel_data.data(), d_pixel_buffer, RENDER_WIDTH * RENDER_HEIGHT * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    
    void* pixels;
    int pitch;
    SDL_LockTexture(texture, NULL, &pixels, &pitch);
    memcpy(pixels, pixel_data.data(), RENDER_WIDTH * RENDER_HEIGHT * sizeof(uint32_t));
    SDL_UnlockTexture(texture);
    
    SDL_RenderClear(renderer);
    
    float aspect_ratio = (float)RENDER_WIDTH / RENDER_HEIGHT;
    float window_aspect = (float)WIDTH / HEIGHT;
    
    SDL_Rect dst_rect;
    if (aspect_ratio > window_aspect) {
        dst_rect.w = WIDTH;
        dst_rect.h = (int)(WIDTH / aspect_ratio);
        dst_rect.x = 0;
        dst_rect.y = (HEIGHT - dst_rect.h) / 2;
    } else {
        dst_rect.w = (int)(HEIGHT * aspect_ratio);
        dst_rect.h = HEIGHT;
        dst_rect.x = (WIDTH - dst_rect.w) / 2;
        dst_rect.y = 0;
    }
    
    SDL_RenderCopy(renderer, texture, NULL, &dst_rect);
}

void handle_input(SDL_Event& e, bool& running) {
    ImGui_ImplSDL2_ProcessEvent(&e);
    
    if (e.type == SDL_QUIT) {
        running = false;
    } else if (e.type == SDL_KEYDOWN) {
        switch (e.key.keysym.sym) {
            case SDLK_ESCAPE:
                running = false;
                break;
            case SDLK_TAB:
                show_gui = !show_gui;
                break;
            case SDLK_r:
                cam_pos = {0, 0, 5};
                cam_yaw = cam_pitch = 0;
                break;
        }
    } else if (e.type == SDL_MOUSEMOTION && (e.motion.state & SDL_BUTTON_LMASK)) {
        if (!ImGui::GetIO().WantCaptureMouse) {
            cam_yaw += e.motion.xrel * 0.01f;
            cam_pitch -= e.motion.yrel * 0.01f;
            cam_pitch = clamp(cam_pitch, (float)(-M_PI/2.0 + 0.1), (float)(M_PI/2.0 - 0.1));
        }
    } else if (e.type == SDL_MOUSEWHEEL) {
        if (!ImGui::GetIO().WantCaptureMouse) {
            Vec3 forward = {sinf(cam_yaw) * cosf(cam_pitch), sinf(cam_pitch), -cosf(cam_yaw) * cosf(cam_pitch)};
            cam_pos = cam_pos + forward * (e.wheel.y * 0.5f);
        }
    }
}

void update_camera_movement(float delta_time) {
    const Uint8* keys = SDL_GetKeyboardState(NULL);
    if (!ImGui::GetIO().WantCaptureKeyboard) {
        float speed = 5.0f * delta_time;
        Vec3 forward = {-sinf(cam_yaw), 0, -cosf(cam_yaw)};
        Vec3 right = {cosf(cam_yaw), 0, -sinf(cam_yaw)};
        Vec3 up = {0, 1, 0};
        
        if (keys[SDL_SCANCODE_W]) cam_pos = cam_pos + forward * speed;
        if (keys[SDL_SCANCODE_S]) cam_pos = cam_pos - forward * speed;
        if (keys[SDL_SCANCODE_A]) cam_pos = cam_pos - right * speed;
        if (keys[SDL_SCANCODE_D]) cam_pos = cam_pos + right * speed;
        if (keys[SDL_SCANCODE_SPACE]) cam_pos = cam_pos + up * speed;
        if (keys[SDL_SCANCODE_LCTRL] || keys[SDL_SCANCODE_RCTRL]) cam_pos = cam_pos - up * speed;
        if (keys[SDL_SCANCODE_UP]) cam_pitch += speed;
        if (keys[SDL_SCANCODE_DOWN]) cam_pitch -= speed;
        if (keys[SDL_SCANCODE_LEFT]) cam_yaw -= speed;
        if (keys[SDL_SCANCODE_RIGHT]) cam_yaw += speed;

        cam_pitch = clamp(cam_pitch, (float)(-M_PI/2.0 + 0.1), (float)(M_PI/2.0 - 0.1));
    }
}

void update_fps() {
    Uint32 current_time = SDL_GetTicks();
    float frame_time = (current_time - last_frame_time) / 1000.0f;
    last_frame_time = current_time;
    
    frame_times[frame_time_index] = frame_time;
    frame_time_index = (frame_time_index + 1) % 60;
    
    float total_time = 0.0f;
    for (int i = 0; i < 60; i++) {
        total_time += frame_times[i];
    }
    fps = total_time > 0 ? 60.0f / total_time : 0.0f;
}

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }
    
    SDL_Window* window = SDL_CreateWindow("CUDA Ray Tracer", 
                                        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                        WIDTH, HEIGHT, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    last_frame_time = SDL_GetTicks();
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    ImGui::StyleColorsDark();
    
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);
    
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, 
                                   SDL_TEXTUREACCESS_STREAMING, RENDER_WIDTH, RENDER_HEIGHT);
    
    initialize_gpu();
    
    {
        Mesh cube;
        cube.name = "Default Cube";
        cube.color = {0.7f, 0.7f, 0.7f};
        cube.position = {-2, 0, -5};
        create_cube(cube, {1, 1, 1}, 0);
        meshes.push_back(cube);
    }

    {
        Mesh sphere;
        sphere.name = "Default Sphere";
        sphere.color = {0.7f, 0.7f, 0.7f};
        sphere.position = {2, 0, -5};
        create_sphere(sphere, 1.0f, 1);
        meshes.push_back(sphere);
    }
    
    bool running = true;
    SDL_Event e;
    
    std::cout << "Controls:" << std::endl;
    std::cout << "  WASD: Move camera forward/backward/left/right" << std::endl;
    std::cout << "  Arrow Keys: Look around" << std::endl;
    std::cout << "  Space: Move camera up" << std::endl;
    std::cout << "  Ctrl: Move camera down" << std::endl;
    std::cout << "  Mouse: Look around" << std::endl;
    std::cout << "  Mouse wheel: Move forward/backward" << std::endl;
    std::cout << "  Tab: Toggle GUI" << std::endl;
    std::cout << "  R: Reset camera" << std::endl;
    std::cout << "  Escape: Exit" << std::endl;
    
    while (running) {
        update_fps();
    
        static Uint32 last_time = SDL_GetTicks();
        Uint32 current_time = SDL_GetTicks();
        float delta_time = (current_time - last_time) / 1000.0f;
        last_time = current_time;

        while (SDL_PollEvent(&e)) {
            handle_input(e, running);
        }
    
        update_camera_movement(delta_time);
        
        int window_width, window_height;
        SDL_GetWindowSize(window, &window_width, &window_height);
        if (window_width != WIDTH || window_height != HEIGHT) {
            WIDTH = window_width;
            HEIGHT = window_height;
        }
        
        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        
        render_gui();
        
        render_scene(renderer, texture);
        
        ImGui::Render();
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        
        SDL_RenderPresent(renderer);
    }
    
    cleanup_gpu();
    
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}