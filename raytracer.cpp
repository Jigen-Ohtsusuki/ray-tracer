#include <SDL2/SDL.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <thread>

#define M_PI 3.14159265358979323846

// Clamp helper
template <typename T>
T clamp(T x, T a, T b) {
    return std::max(a, std::min(x, b));
}

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {} // default constructor
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {} // 👈 this fixes your error

    Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator*(Vec3 o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    float dot(Vec3 o) const { return x * o.x + y * o.y + z * o.z; }

    Vec3 normalized() const {
        float len = std::sqrt(x * x + y * y + z * z);
        return {x / len, y / len, z / len};
    }
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
};

struct Light {
    Vec3 direction;
    Vec3 color;
    float intensity;
};

const int WIDTH = 640;
const int HEIGHT = 360;
const float FOV = 60.0f;

const Vec3 AMBIENT_LIGHT = {0.1f, 0.1f, 0.1f};

// Spheres
std::vector<Sphere> spheres = {
    {{-1.5f, 0, -5}, 1.0f, {1.0f, 0.0f, 0.0f}},
    {{ 0.0f, 0, -5}, 1.0f, {0.0f, 1.0f, 0.0f}},
    {{ 1.5f, 0, -5}, 1.0f, {0.0f, 0.0f, 1.0f}},
};

// Scaled-down Sun parameters
Light sun = {
    .direction = Vec3{0.5f, -1.0f, -1.0f}.normalized(),
    .color = {1.0f, 1.0f, 1.0f},
    .intensity = 1.0f // Already scaled from 6.4e7 W/m²
};

bool intersect_sphere(const Ray& ray, const Sphere& sphere, float& t, Vec3& normal, Vec3& hit_color) {
    Vec3 oc = ray.origin - sphere.center;
    float a = ray.direction.dot(ray.direction);
    float b = 2.0f * oc.dot(ray.direction);
    float c = oc.dot(oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;
    t = (-b - std::sqrt(discriminant)) / (2.0f * a);
    if (t < 0) return false;
    Vec3 hit_point = ray.origin + ray.direction * t;
    normal = (hit_point - sphere.center).normalized();
    hit_color = sphere.color;
    return true;
}

Vec3 trace(const Ray& ray, int depth = 0) {
    float closest_t = 1e9;
    Vec3 hit_normal, hit_color;
    bool hit = false;

    for (const auto& sphere : spheres) {
        float t;
        Vec3 n, c;
        if (intersect_sphere(ray, sphere, t, n, c) && t < closest_t) {
            closest_t = t;
            hit = true;
            hit_normal = n;
            hit_color = c;
        }
    }

    if (!hit) {
        // Sky gradient background (optional, looks better than solid black)
        float t = 0.5f * (ray.direction.y + 1.0f);
        return Vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + Vec3(0.4f, 0.6f, 1.0f) * t;
    }

    // Lambertian diffuse lighting
    float diff = std::max(0.0f, hit_normal.dot(-sun.direction));
    Vec3 diffuse = sun.color * sun.intensity * diff;

    // Hemispherical ambient skylight (up-facing surfaces get more light)
    Vec3 sky_color = {0.4f, 0.6f, 1.0f}; // sky blue
    float sky_factor = std::max(0.0f, hit_normal.y); // how much surface faces up
    Vec3 ambient = sky_color * 0.2f * sky_factor;

    Vec3 lighting = ambient + diffuse;

    lighting = {
        clamp(lighting.x, 0.0f, 1.0f),
        clamp(lighting.y, 0.0f, 1.0f),
        clamp(lighting.z, 0.0f, 1.0f)
    };

    Vec3 local_color = hit_color * lighting;

    return {
        clamp(local_color.x, 0.0f, 1.0f),
        clamp(local_color.y, 0.0f, 1.0f),
        clamp(local_color.z, 0.0f, 1.0f)
    };
}


Vec3 get_ray_dir(int x, int y, float fov, float aspect, float yaw, float pitch) {
    float px = (2 * ((x + 0.5f) / WIDTH) - 1) * aspect * std::tan(fov * 0.5f * M_PI / 180);
    float py = (1 - 2 * ((y + 0.5f) / HEIGHT)) * std::tan(fov * 0.5f * M_PI / 180);
    Vec3 dir = {px, py, -1};

    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw), sy = std::sin(yaw);

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

int main() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Ray Tracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, WIDTH, HEIGHT, 32, SDL_PIXELFORMAT_RGB888);

    Vec3 cam_pos = {0, 0, 0};
    float cam_yaw = 0.0f, cam_pitch = 0.0f;

    bool running = true;
    Uint32 last_time = SDL_GetTicks();

    while (running) {
        Uint32 start = SDL_GetTicks();
        SDL_Event e;
        const Uint8* keys = SDL_GetKeyboardState(NULL);

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
        }

        float dt = (SDL_GetTicks() - last_time) / 1000.0f;
        last_time = SDL_GetTicks();

        Vec3 forward = {-std::sin(cam_yaw), 0, -std::cos(cam_yaw)};
        Vec3 right   = { std::cos(cam_yaw), 0, -std::sin(cam_yaw)};

        if (keys[SDL_SCANCODE_W]) cam_pos = cam_pos + forward * dt * 3;
        if (keys[SDL_SCANCODE_S]) cam_pos = cam_pos - forward * dt * 3;
        if (keys[SDL_SCANCODE_A]) cam_pos = cam_pos - right * dt * 3;
        if (keys[SDL_SCANCODE_D]) cam_pos = cam_pos + right * dt * 3;
        if (keys[SDL_SCANCODE_Q]) cam_pos.y += dt * 3;
        if (keys[SDL_SCANCODE_E]) cam_pos.y -= dt * 3;

        if (keys[SDL_SCANCODE_LEFT])  cam_yaw -= dt * 1.5f;
        if (keys[SDL_SCANCODE_RIGHT]) cam_yaw += dt * 1.5f;
        if (keys[SDL_SCANCODE_UP])    cam_pitch += dt * 1.5f;
        if (keys[SDL_SCANCODE_DOWN])  cam_pitch -= dt * 1.5f;

        cam_pitch = clamp(cam_pitch, (float)(-M_PI / 2.0 + 0.01), (float)(M_PI / 2.0 - 0.01));

        SDL_LockSurface(surface);
        Uint32* pixels = (Uint32*)surface->pixels;
        float aspect = (float)WIDTH / HEIGHT;

        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;

        std::vector<std::thread> threads;
        int rows_per_thread = HEIGHT / num_threads;

        auto render_chunk = [&](int start_y, int end_y) {
            for (int y = start_y; y < end_y; ++y) {
                for (int x = 0; x < WIDTH; ++x) {
                    Vec3 dir = get_ray_dir(x, y, FOV, aspect, cam_yaw, cam_pitch);
                    Vec3 color = trace({cam_pos, dir});

                    Uint8 r = (Uint8)(clamp(color.x, 0.0f, 1.0f) * 255);
                    Uint8 g = (Uint8)(clamp(color.y, 0.0f, 1.0f) * 255);
                    Uint8 b = (Uint8)(clamp(color.z, 0.0f, 1.0f) * 255);
                    pixels[y * WIDTH + x] = SDL_MapRGB(surface->format, r, g, b);
                }
            }
        };

        for (int i = 0; i < num_threads; ++i) {
            int start_y = i * rows_per_thread;
            int end_y = (i == num_threads - 1) ? HEIGHT : (i + 1) * rows_per_thread;
            threads.emplace_back(render_chunk, start_y, end_y);
        }

        for (auto& t : threads) t.join();

        SDL_UnlockSurface(surface);
        SDL_UpdateTexture(texture, NULL, surface->pixels, surface->pitch);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        Uint32 frame_time = SDL_GetTicks() - start;
        float fps = 1000.0f / std::max(1.0f, (float)frame_time);
        std::string title = "Ray Tracer - FPS: " + std::to_string((int)fps);
        SDL_SetWindowTitle(window, title.c_str());

        SDL_RenderPresent(renderer);
    }

    SDL_FreeSurface(surface);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}