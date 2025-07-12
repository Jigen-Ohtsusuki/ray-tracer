# Ray Tracer

A high-performance, GPU-accelerated real-time ray tracer implemented in C++ using SDL2, CUDA, and ImGui. This project significantly enhances the previous CPU-based ray tracer by leveraging CUDA for parallel rendering, adding triangle mesh support, OBJ file import, a dynamic scene editor, and an interactive GUI for scene manipulation. It demonstrates advanced ray tracing techniques with realistic lighting, shadows, and user-friendly controls.

## Features

- **GPU-Accelerated Rendering**: Utilizes CUDA for massively parallel ray tracing, achieving high FPS even at high resolutions.
- **Triangle Mesh Support**: Renders complex geometry using triangles, supporting custom meshes and OBJ file imports.
- **Interactive Scene Editor**: Add, edit, and manipulate 3D objects (cubes, spheres, custom meshes) via ImGui interface.
- **OBJ File Import**: Load 3D models from OBJ files with automatic triangle generation.
- **Realistic Lighting System**: Includes diffuse lighting, ambient occlusion, and hard shadows with sun-like light source.
- **Checkerboard Floor**: Dynamic tiled floor with customizable colors and size for scene grounding.
- **Sky Gradient with Sun Glow**: Enhanced sky rendering with a glowing sun effect based on light direction.
- **Full 6DOF Camera Controls**: Mouse and keyboard controls for free camera movement and rotation.
- **Dynamic Resolution Scaling**: Adjustable rendering resolution with preset options (320x240 to 1920x1080).
- **ImGui Interface**: Intuitive GUI for controlling camera, light, floor, and object properties.
- **Performance Monitoring**: Real-time FPS, frame time graphs, and triangle count display.
- **Mesh Transformation**: Support for position, rotation, and scale adjustments for each mesh.
- **Shadow Casting**: Hard shadows computed on the GPU for realistic lighting.
- **Optimized Memory Management**: Efficient CUDA memory handling for large scenes.

## Demo

<div align="center">
  <img src="raytracer-demo.gif" alt="Ray Tracer Demo" width="800">
</div>

The scene includes:
- A default cube and sphere with customizable properties.
- A checkerboard floor with adjustable colors and tile size.
- A dynamic light source simulating sunlight with shadow casting.
- A sky gradient with a glowing sun effect for immersive backgrounds.
- Interactive GUI for real-time scene editing.

## Requirements

- **MSYS2** (for Windows development environment, optional if using Visual Studio)
- **MinGW-w64** (C++ compiler with C++11 support, optional)
- **SDL2** (graphics library, Visual Studio compatible version)
- **CUDA Toolkit** (v12.9 or compatible, for GPU acceleration)
- **ImGui** (source files included in `imgui/` directory)
- **NVIDIA GPU** (compute capability 8.9 or compatible, e.g., RTX 4090)

## Installation

### Setting up the Environment

1. **Install CUDA Toolkit**: Download and install CUDA Toolkit v12.9 from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).
2. **Install SDL2**: Download the Visual Studio development libraries (`SDL2-VC`) from [libsdl.org](https://www.libsdl.org/) and extract to a directory (e.g., `SDL2-VC/`).
3. **ImGui**: Ensure the ImGui source files (`imgui.h`, `imgui.cpp`, `imgui_demo.cpp`, `imgui_draw.cpp`, `imgui_tables.cpp`, `imgui_widgets.cpp`, `imgui_impl_sdl2.h`, `imgui_impl_sdl2.cpp`, `imgui_impl_sdlrenderer2.h`, `imgui_impl_sdlrenderer2.cpp`) are in the `imgui/` directory.
4. **Optional MSYS2 Setup** (for MinGW-based builds):
   - Install MSYS2 from [https://www.msys2.org/](https://www.msys2.org/).
   - Update the package database:
     ```bash
     pacman -Syu
     ```
   - Install MinGW-w64 and SDL2:
     ```bash
     pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-SDL2
     ```

### Building the Ray Tracer

1. Clone the repository:
   ```bash
   git clone https://github.com/Jigen-Ohtsusuki/ray-tracer.git
   cd ray-tracer
   ```

2. Ensure the `SDL2-VC` directory is in the project root or adjust the include/library paths in the build command accordingly.
3. Ensure the CUDA Toolkit is installed and `nvcc` is accessible in your PATH.

4. Compile the source using `nvcc`:
   ```bash
   nvcc raytracer_cuda.cu imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/imgui_impl_sdl2.cpp imgui/imgui_impl_sdlrenderer2.cpp -o raytracer_cuda.exe -std=c++17 -DSDL_MAIN_HANDLED -I./imgui -I"./SDL2-VC/include/" -L"./SDL2-VC/lib/x64" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -lSDL2 -lSDL2main -lcudart -lcomdlg32 -arch=compute_89 -code=sm_89
   ```

5. Run the ray tracer:
   ```bash
   ./raytracer_cuda.exe
   ```

## Controls

| Key | Action |
|-----|--------|
| W | Move camera forward |
| S | Move camera backward |
| A | Move camera left |
| D | Move camera right |
| Space | Move camera up |
| Ctrl | Move camera down |
| Arrow Keys | Look around |
| Mouse Drag | Look around |
| Mouse Wheel | Move forward/backward |
| Tab | Toggle GUI |
| R | Reset camera |
| Esc | Exit |

## Technical Details

### GPU-Accelerated Rendering

The ray tracer leverages CUDA for parallel processing:
- **Kernel Design**: The `render_kernel` distributes ray tracing across CUDA threads, handling pixel calculations in parallel.
- **Memory Management**: Dynamic allocation and transfer of triangle, light, and floor data to GPU memory.
- **Performance**: Achieves real-time performance with complex scenes, optimized for NVIDIA GPUs with compute capability 8.9.

```cpp
__global__ void render_kernel(uint32_t* pixel_buffer, Triangle* triangles, int num_triangles, Light light, FloorPlane floor_plane, 
                             Vec3 cam_pos, float cam_yaw, float cam_pitch, float fov, int render_width, int render_height, int selected_mesh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= render_width || y >= render_height) return;
    
    float aspect = (float)render_width / render_height;
    Vec3 ray_dir = get_ray_dir(x, y, fov, aspect, cam_yaw, cam_pitch, render_width, render_height);
    Ray ray = {cam_pos, ray_dir};
    Vec3 color = trace(ray, triangles, num_triangles, light, floor_plane, selected_mesh);
    // Write to pixel buffer
}
```

### Scene Management

- **Triangle Meshes**: Supports complex geometry via triangle-based meshes with transformation (position, rotation, scale).
- **OBJ Loader**: Parses OBJ files to import custom 3D models, converting faces to triangles.
- **Dynamic Scene**: Add cubes, spheres, or OBJ models, with real-time editing of properties via ImGui.

```cpp
bool load_obj(const std::string& filename, Mesh& mesh, int mesh_index) {
    // Parse vertices and faces from OBJ file
    // Convert to triangles with associated mesh index
}
```

### Lighting System

The lighting model includes:
- **Diffuse Lighting**: Calculated with `max(0, normal · light_dir)` for realistic surface illumination.
- **Hard Shadows**: GPU-based shadow rays test for occlusions, enhancing realism.
- **Ambient Lighting**: Hemispherical ambient with sky color for natural indirect light.
- **Sun-like Light**: High-intensity, directional light with customizable position, color, and intensity.

```cpp
__device__ Vec3 trace(const Ray& ray, const Triangle* triangles, int num_triangles, const Light& light, const FloorPlane& floor_plane, int selected_mesh, int depth) {
    // Check intersections with triangles and floor
    // Compute diffuse, ambient, and shadow effects
    // Apply sun-like attenuation and sky glow
}
```

### ImGui Interface

The ImGui-based GUI provides:
- **Scene Control**: Add/delete objects, adjust position, rotation, scale, and color.
- **Light Editing**: Modify light position, color, and intensity, with a sunlight preset.
- **Floor Customization**: Adjust floor height, tile size, and checkerboard colors.
- **Performance Metrics**: Displays FPS, frame time, triangle count, and resolution.
- **Resolution Presets**: Quick selection of common resolutions for performance tuning.

```cpp
void render_gui() {
    ImGui::Begin("Ray Tracer Controls");
    if (ImGui::CollapsingHeader("3D Objects")) {
        // Add cube, sphere, or OBJ
        // Edit selected mesh properties
    }
    // Camera, light, floor, and performance controls
}
```

### Performance Optimizations

- **CUDA Parallelism**: Distributes rendering across thousands of GPU threads.
- **Efficient Memory Transfers**: Minimizes host-to-device data transfers by updating only changed scene data.
- **Dynamic Resolution**: Balances performance and quality with adjustable render resolution.
- **Triangle Transformations**: Applied on GPU to reduce CPU workload.
- **Error Checking**: Robust CUDA error handling with `CUDA_CHECK` macro.

## Configuration

Modify these constants in the code:

```cpp
int WIDTH = 800; // Window width
int HEIGHT = 600; // Window height
int RENDER_WIDTH = 400; // Render resolution
int RENDER_HEIGHT = 300;
const float FOV = 60.0f;
Light light = {{5, 10, 5}, {1.0f, 0.95f, 0.9f}, 10.0f}; // Sun-like light
FloorPlane floor_plane = {-2.0f, {0.8f, 0.8f, 0.8f}, {0.6f, 0.6f, 0.6f}, 2.0f};
```

### High Resolution Rendering

The CUDA implementation supports high resolutions efficiently:
- **800x600**: Real-time performance with complex scenes.
- **1920x1080**: Usable with optimized settings for high-quality output.

## Mathematical Foundation

### Ray-Triangle Intersection

Uses the Möller-Trumbore algorithm for efficient ray-triangle intersection:
- **Equation**: Solves `ray.origin + t * ray.direction = (1-u-v) * v0 + u * v1 + v * v2`.
- **Barycentric Coordinates**: Ensures hit point lies within triangle bounds.

```cpp
__device__ bool intersect_triangle(const Ray& ray, const Triangle& tri, float& t, Vec3& normal) {
    // Möller-Trumbore intersection
}
```

### Lighting Model

- **Diffuse**: `L_diffuse = I * max(0, N · L) * attenuation`.
- **Ambient**: `L_ambient = sky_color * max(0, N.y) * 0.15`.
- **Shadow**: Tests occlusion with shadow rays.
- **Sky Gradient**: `color = (1-t) * horizon + t * zenith + sun_glow`.

## Performance Comparison

**Test System**: NVIDIA RTX 4050, Intel i7-13700HX

| Version | Resolution | FPS | Visual Quality | Notes |
|---------|------------|-----|----------------|-------|
| Previous C++ Multi-thread | 640x360 | 120-140 | Realistic lighting | CPU-based |
| **CUDA Ray Tracer** | **800x600** | **200-300** | **Enhanced lighting, shadows** | **Current version** |
| CUDA Ray Tracer | 1920x1080 | 30-40 | Enhanced lighting, shadows | High quality |

### Performance Benefits

- **GPU Acceleration**: CUDA provides 2-3x performance over CPU multi-threading.
- **Shadows and Lighting**: Adds hard shadows and sun-like lighting with minimal performance cost.
- **Scalability**: Handles thousands of triangles efficiently due to GPU parallelism.

## Build Script

Create a `build.sh` for easier compilation:

```bash
#!/bin/bash
nvcc raytracer_cuda.cu imgui/imgui.cpp imgui/imgui_demo.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/imgui_impl_sdl2.cpp imgui/imgui_impl_sdlrenderer2.cpp -o raytracer_cuda.exe -std=c++17 -DSDL_MAIN_HANDLED -I./imgui -I"./SDL2-VC/include/" -L"./SDL2-VC/lib/x64" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64" -lSDL2 -lSDL2main -lcudart -lcomdlg32 -arch=compute_89 -code=sm_89
```

## Thread Safety and GPU Considerations

- **CUDA Memory**: Manages GPU memory with proper allocation and cleanup.
- **Thread Safety**: ImGui and SDL interactions are single-threaded; CUDA handles parallel rendering.
- **Error Handling**: Robust CUDA error checking ensures stability.
- **No CPU Bottlenecks**: Scene transformations and rendering are offloaded to GPU.

## Completed Features

- [x] **GPU Acceleration**: CUDA-based rendering for massive performance gains.
- [x] **Triangle Meshes**: Support for complex geometry and OBJ import.
- [x] **ImGui Interface**: Interactive GUI for scene editing.
- [x] **Hard Shadows**: GPU-accelerated shadow casting.
- [x] **Checkerboard Floor**: Dynamic tiled floor for scene context.
- [x] **Sun-like Lighting**: Enhanced light model with glowing sun effect.
- [x] **Dynamic Scene Management**: Add, edit, and delete objects in real-time.

## Future Enhancements

- [ ] **Reflections and Refractions**: Add recursive ray tracing for reflective/refractive materials.
- [ ] **Soft Shadows**: Implement area lights for softer shadows.
- [ ] **Material System**: Support metallic, dielectric, and emissive materials.
- [ ] **Anti-Aliasing**: Add MSAA or supersampling for smoother edges.
- [ ] **BVH Acceleration**: Implement bounding volume hierarchy for faster intersections.
- [ ] **Post-Processing**: Add bloom, tone mapping, and gamma correction.
- [ ] **Volumetric Effects**: Implement fog and atmospheric scattering.
- [ ] **Texture Support**: Add texture mapping for meshes.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make changes and test on CUDA-capable hardware (compute capability 8.9 recommended).
4. Submit a pull request.

## Acknowledgments

- CUDA for GPU acceleration.
- ImGui for intuitive GUI implementation.
- SDL2 for cross-platform rendering and input handling.
- Möller-Trumbore algorithm for ray-triangle intersection.
- Physically-based rendering techniques for lighting.