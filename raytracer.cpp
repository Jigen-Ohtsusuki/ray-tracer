#include <SDL2/SDL.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

#define M_PI 3.14159265358979323846

const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 360;
const int RENDER_WIDTH = 640;
const int RENDER_HEIGHT = 360;
const float FOV = 60.0f;

struct Vec3 {
    float x, y, z;
    Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    float dot(Vec3 o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 normalized() const {
        float len = std::sqrt(x * x + y * y + z * z);
        return {x / len, y / len, z / len};
    }
};

struct Sphere {
    Vec3 center;
    float radius;
    SDL_Color color;
};

std::vector<Sphere> spheres = {
    {{-1.5f, 0, -5}, 1.0f, {255, 0, 0, 255}},
    {{ 0.0f, 0, -5}, 1.0f, {0, 255, 0, 255}},
    {{ 1.5f, 0, -5}, 1.0f, {0, 0, 255, 255}},
};

bool hit_sphere(Vec3 origin, Vec3 dir, const Sphere& s, float& t) {
    Vec3 oc = origin - s.center;
    float a = dir.dot(dir);
    float b = 2.0f * oc.dot(dir);
    float c = oc.dot(oc) - s.radius * s.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return false;
    t = (-b - std::sqrt(discriminant)) / (2.0f * a);
    return t > 0;
}

Vec3 get_ray_dir(int x, int y, float fov_deg, float aspect, float yaw, float pitch) {
    float px = (2 * ((x + 0.5f) / RENDER_WIDTH) - 1) * aspect * std::tan(fov_deg * 0.5f * M_PI / 180);
    float py = (1 - 2 * ((y + 0.5f) / RENDER_HEIGHT)) * std::tan(fov_deg * 0.5f * M_PI / 180);

    Vec3 dir = {px, py, -1};  // initial direction (camera faces -Z)

    // Apply pitch (X-axis rotation)
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw), sy = std::sin(yaw);

    float dy = dir.y * cp - dir.z * sp;
    float dz = dir.y * sp + dir.z * cp;
    dir.y = dy;
    dir.z = dz;

    // Apply yaw (Y-axis rotation)
    float dx = dir.x * cy + dir.z * sy;
    dz = -dir.x * sy + dir.z * cy;
    dir.x = dx;
    dir.z = dz;

    return dir.normalized();
}

void render(SDL_Surface* surface, Vec3 cam_pos, float cam_yaw, float cam_pitch) {
    SDL_LockSurface(surface);
    Uint32* pixels = (Uint32*)surface->pixels;
    float aspect = (float)RENDER_WIDTH / RENDER_HEIGHT;

    for (int y = 0; y < RENDER_HEIGHT; y++) {
        for (int x = 0; x < RENDER_WIDTH; x++) {
            Vec3 dir = get_ray_dir(x, y, FOV, aspect, cam_yaw, cam_pitch);
            SDL_Color color = {0, 0, 0, 255};
            float min_t = 1e9;

            for (const auto& s : spheres) {
                float t;
                if (hit_sphere(cam_pos, dir, s, t) && t < min_t) {
                    min_t = t;
                    color = s.color;
                }
            }

            pixels[y * RENDER_WIDTH + x] =
                SDL_MapRGB(surface->format, color.r, color.g, color.b);
        }
    }

    SDL_UnlockSurface(surface);
}

int main() {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("C++ Ray Tracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, RENDER_WIDTH, RENDER_HEIGHT);
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, RENDER_WIDTH, RENDER_HEIGHT, 32, SDL_PIXELFORMAT_RGB888);

    Vec3 cam_pos = {0, 0, 0};
    float cam_yaw = 0.0f;
    float cam_pitch = 0.0f;

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

        // Movement
        if (keys[SDL_SCANCODE_W]) cam_pos = cam_pos + forward * dt * 3;
        if (keys[SDL_SCANCODE_S]) cam_pos = cam_pos - forward * dt * 3;
        if (keys[SDL_SCANCODE_A]) cam_pos = cam_pos - right * dt * 3;
        if (keys[SDL_SCANCODE_D]) cam_pos = cam_pos + right * dt * 3;
        if (keys[SDL_SCANCODE_Q]) cam_pos.y += dt * 3;
        if (keys[SDL_SCANCODE_E]) cam_pos.y -= dt * 3;

        // Camera rotation
        if (keys[SDL_SCANCODE_LEFT])  cam_yaw -= dt * 1.5f;
        if (keys[SDL_SCANCODE_RIGHT]) cam_yaw += dt * 1.5f;
        if (keys[SDL_SCANCODE_UP])    cam_pitch += dt * 1.5f;
        if (keys[SDL_SCANCODE_DOWN])  cam_pitch -= dt * 1.5f;

        // Clamp pitch
        if (cam_pitch > M_PI / 2.0f - 0.01f) cam_pitch = M_PI / 2.0f - 0.01f;
        if (cam_pitch < -M_PI / 2.0f + 0.01f) cam_pitch = -M_PI / 2.0f + 0.01f;

        render(surface, cam_pos, cam_yaw, cam_pitch);
        SDL_UpdateTexture(texture, NULL, surface->pixels, surface->pitch);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        // FPS counter
        Uint32 frame_time = SDL_GetTicks() - start;
        float fps = 1000.0f / std::max(1.0f, (float)frame_time);
        std::string title = "C++ Ray Tracer - FPS: " + std::to_string((int)fps);
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
