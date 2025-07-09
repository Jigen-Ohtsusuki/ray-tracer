# Ray Tracer

A high-performance real-time ray tracer implemented in C++ using SDL2 and multi-threading. This project demonstrates fundamental ray tracing concepts including sphere intersection, camera movement, and optimized rendering techniques with dramatically improved performance over the original Python implementation.

## Features

- **Multi-threaded Ray Tracer**: Utilizes all CPU cores for maximum performance
- **Real-time Rendering**: Interactive 3D rendering achieving 200+ FPS
- **Full 6DOF Camera Controls**: Complete movement and rotation with mouse-look style controls
- **Automatic Thread Detection**: Dynamically uses `std::thread::hardware_concurrency()` for optimal threading
- **Pitch/Yaw Camera System**: Look up/down and turn left/right with arrow keys
- **Multiple Objects**: Renders multiple colored spheres in 3D space
- **Real-time FPS Counter**: Performance monitoring displayed in window title
- **Perspective Camera**: Configurable field of view with proper aspect ratio handling

## Demo

<div align="center">
  <img src="raytracer-demo.gif" alt="Ray Tracer Demo" width="640">
</div>

The scene contains three spheres:
- Red sphere at position (-1.5, 0, -5)
- Green sphere at position (0.0, 0, -5)  
- Blue sphere at position (1.5, 0, -5)

## Requirements

- **MSYS2** (for Windows development environment)
- **MinGW-w64** (C++ compiler with C++11 support for std::thread)
- **SDL2** (graphics library)

## Installation

### Setting up MSYS2 Environment

1. Install MSYS2 from https://www.msys2.org/

2. Open MSYS2 terminal and update the package database:
```bash
pacman -Syu
```

3. Install the required development tools and SDL2:
```bash
pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-SDL2
```

### Building the Ray Tracer

1. Clone the repository:
```bash
git clone https://github.com/Jigen-Ohtsusuki/ray-tracer.git
cd ray-tracer
```

2. Compile the C++ source with threading support:
```bash
g++ raytracer.cpp -o raytracer.exe -std=c++17 -DSDL_MAIN_HANDLED -IC:/msys64/ucrt64/include/SDL2 -LC:/msys64/ucrt64/lib -lSDL2 -lmingw32 -mconsole -pthread -O3
```

3. Run the ray tracer:
```bash
./raytracer.exe
```

## Controls

| Key | Action |
|-----|--------|
| W | Move forward |
| S | Move backward |
| A | Move left |
| D | Move right |
| Q | Move up |
| E | Move down |
| ← | Turn left |
| → | Turn right |
| ↑ | Look up |
| ↓ | Look down |

## Technical Details

### Multi-threading Architecture

The ray tracer implements an efficient multi-threading strategy:

1. **Thread Detection**: Automatically detects available CPU cores using `std::thread::hardware_concurrency()`
2. **Row-based Partitioning**: Divides screen rows evenly among threads
3. **Lambda-based Work Distribution**: Clean thread function implementation using lambdas
4. **Synchronization**: Uses `std::thread::join()` to ensure all threads complete before frame presentation

```cpp
int num_threads = std::thread::hardware_concurrency();
if (num_threads == 0) num_threads = 4; // fallback

std::vector<std::thread> threads;
int rows_per_thread = RENDER_HEIGHT / num_threads;

auto render_chunk = [&](int start_y, int end_y) {
    // Render pixels from start_y to end_y
};

for (int i = 0; i < num_threads; ++i) {
    int start_y = i * rows_per_thread;
    int end_y = (i == num_threads - 1) ? RENDER_HEIGHT : (i + 1) * rows_per_thread;
    threads.emplace_back(render_chunk, start_y, end_y);
}

for (auto& t : threads) t.join();
```

### Ray Tracing Algorithm

The ray tracer implements a classic ray casting algorithm with multi-threading optimizations:

1. **Parallel Ray Generation**: Each thread generates rays for its assigned pixel rows
2. **Concurrent Intersection Testing**: Multiple threads test ray-sphere intersections simultaneously
3. **Independent Color Calculation**: Each thread calculates colors for its pixel region
4. **Thread-safe Pixel Writing**: Direct pixel buffer access with proper memory layout

### Performance Improvements

- **Multi-threading**: Utilizes all CPU cores for parallel rendering
- **C++ Implementation**: Native compiled code for maximum performance
- **Efficient Memory Management**: Direct pixel buffer access with SDL2
- **Optimized Vector Operations**: Custom Vec3 struct with inlined operations
- **Minimal Thread Overhead**: Efficient thread creation and synchronization

### Camera System

The camera supports full 6DOF movement with separate pitch and yaw controls:

```cpp
// Movement vectors based on yaw
Vec3 forward = {-sin(cam_yaw), 0, -cos(cam_yaw)};
Vec3 right   = {cos(cam_yaw), 0, -sin(cam_yaw)};

// Ray direction calculation with pitch/yaw rotation
Vec3 get_ray_dir(int x, int y, float fov_deg, float aspect, float yaw, float pitch);
```

### Multi-threaded Rendering Pipeline

```cpp
// For each thread processing rows [start_y, end_y):
for (int y = start_y; y < end_y; y++) {
    for (int x = 0; x < RENDER_WIDTH; x++) {
        Vec3 ray_dir = get_ray_dir(x, y, FOV, aspect, cam_yaw, cam_pitch);
        
        // Test intersection with all spheres
        for (const auto& sphere : spheres) {
            float t;
            if (hit_sphere(cam_pos, ray_dir, sphere, t) && t < min_t) {
                min_t = t;
                color = sphere.color;
            }
        }
        
        // Write pixel (thread-safe due to non-overlapping regions)
        pixels[y * RENDER_WIDTH + x] = SDL_MapRGB(surface->format, color.r, color.g, color.b);
    }
}
```

## Configuration

You can modify these constants in the code:

```cpp
const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 360;
const int RENDER_WIDTH = 640;      // Internal render resolution
const int RENDER_HEIGHT = 360;     // Internal render resolution
const float FOV = 60.0f;           // Field of view in degrees
```

### High Resolution Rendering

With multi-threading, the implementation can handle much higher resolutions efficiently:

```cpp
// For high resolution (excellent performance expected)
const int SCREEN_WIDTH = 1920;
const int SCREEN_HEIGHT = 1080;
const int RENDER_WIDTH = 1920;      // Full resolution rendering
const int RENDER_HEIGHT = 1080;

// For ultra-high resolution (still very usable)
const int SCREEN_WIDTH = 2560;
const int SCREEN_HEIGHT = 1440;
const int RENDER_WIDTH = 2560;
const int RENDER_HEIGHT = 1440;
```

## Mathematical Foundation

### Ray-Sphere Intersection

The ray tracer uses the geometric ray-sphere intersection formula:

Given a ray `P(t) = O + t*D` and sphere center `C` with radius `r`:
- `(O + t*D - C) · (O + t*D - C) = r²`
- Expanding gives quadratic: `at² + bt + c = 0`
- Where: `a = D·D`, `b = 2(O-C)·D`, `c = (O-C)·(O-C) - r²`

### Camera Transformation

The camera system implements:
- **Perspective projection** with configurable FOV
- **Pitch rotation** (look up/down) around X-axis
- **Yaw rotation** (turn left/right) around Y-axis
- **Proper aspect ratio** handling to prevent distortion

## Performance Comparison

### Evolution of Performance

**Test System**: 13th Gen Intel i7 HX Series Processor

| Version | Resolution | FPS | CPU Usage | Notes |
|---------|------------|-----|-----------|-------|
| Python  | 160x90     | ~5 fps | Single-core | Original implementation |
| Python  | 640x360    | ~0.5 fps | Single-core | Unusably slow |
| C++ Single-thread | 640x360 | 60-80 fps | Single-core | 120x+ improvement |
| C++ Multi-thread | 640x360 | 200+ fps | All cores | 3x+ improvement over single-thread |
| C++ Multi-thread | 1920x1080 | 30-40 fps | All cores | Usable at full HD! |

### Multi-threading Performance Characteristics

**Test System**: 13th Gen Intel i7 HX Series Processor

- **Scalability**: Near-linear performance scaling with CPU core count
- **Efficiency**: Minimal thread overhead due to row-based partitioning
- **Load Balancing**: Even distribution of work across threads
- **Memory Locality**: Good cache performance with row-wise processing

### Actual Performance Results

**Test System**: 13th Gen Intel i7 HX Series Processor

| Implementation | Resolution | FPS | Notes |
|----------------|------------|-----|-------|
| Multi-threaded C++ | 640x360 | 200+ fps | All cores utilized |
| Multi-threaded C++ | 1920x1080 | 30-40 fps | Full HD performance |

## Build Script (Recommended)

Create a `build.sh` file for easier compilation with all optimizations:

```bash
#!/bin/bash
g++ raytracer.cpp -o raytracer.exe \
    -std=c++17 \
    -DSDL_MAIN_HANDLED \
    -IC:/msys64/ucrt64/include/SDL2 \
    -LC:/msys64/ucrt64/lib \
    -lSDL2 \
    -lmingw32 \
    -mconsole \
    -pthread \
    -O3 \
    -march=native \
    -flto
```

## Thread Safety and Considerations

- **Memory Access**: Each thread writes to non-overlapping pixel regions
- **Shared Data**: Spheres vector and camera parameters are read-only during rendering
- **SDL Surface**: Surface is locked/unlocked around the entire multi-threaded operation
- **No Mutexes Required**: Clean parallelization without synchronization overhead

## Completed Features

- [x] **Multi-threading for improved performance**: Implemented efficient row-based parallel rendering
- [x] **Real-time performance**: Achieved 200+ FPS at 640x360 resolution
- [x] **High-resolution rendering**: Usable performance at 1920x1080 (30-40 FPS)

## Future Enhancements

- [ ] **SIMD Optimizations**: Vectorized ray-sphere intersection calculations
- [ ] **GPU Computing**: CUDA or OpenCL implementation for even higher performance
- [ ] **Advanced Threading**: Work-stealing queue for better load balancing
- [ ] **Lighting and Shadows**: Multi-threaded shadow ray calculations
- [ ] **Reflections and Refractions**: Recursive ray tracing with thread pools
- [ ] **Spatial Acceleration**: BVH or octree with thread-safe traversal
- [ ] **Anti-aliasing**: Multi-sample anti-aliasing with parallel sample processing
- [ ] **Material System**: Complex shading with parallel material evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly on multi-core systems
5. Submit a pull request

## Acknowledgments

- Multi-threading implementation using modern C++ std::thread
- Ray tracing algorithms based on classic computer graphics techniques
- Built with C++ and SDL2 for maximum performance
- Significant performance improvements through parallel processing