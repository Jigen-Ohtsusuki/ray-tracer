# Ray Tracer

A real-time ray tracer implemented in Python using Pygame and NumPy. This project demonstrates fundamental ray tracing concepts including sphere intersection, camera movement, and optimized rendering techniques.

## Features

- **Real-time Ray Tracing**: Interactive 3D rendering using ray-sphere intersection
- **Camera Controls**: Full 6DOF movement with WASD + QE for vertical movement
- **Optimized Rendering**: Uses lower internal resolution (160x90) upscaled to display resolution for better performance
- **Multiple Objects**: Renders multiple colored spheres in 3D space
- **FPS Counter**: Real-time performance monitoring
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

```
pygame
numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jigen-Ohtsusuki/ray-tracer.git
cd ray-tracer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the ray tracer:
```bash
python raytracer.py
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
| ← | Rotate left |
| → | Rotate right |
| ESC | Exit |

## Technical Details

### Architecture

The ray tracer implements a classic ray casting algorithm:

1. **Ray Generation**: For each pixel, generates a ray from camera position through the pixel
2. **Intersection Testing**: Tests ray-sphere intersections using quadratic formula
3. **Closest Hit**: Finds the nearest intersection point
4. **Color Assignment**: Assigns the sphere's color to the pixel

### Performance Optimizations

- **Low Resolution Rendering**: Renders at 160x90 internal resolution, then upscales to 640x360
- **Efficient Ray Direction Calculation**: Pre-calculates ray directions with proper FOV scaling
- **Direct Pixel Access**: Uses `pygame.surfarray.pixels3d()` for fast pixel manipulation

### Rendering Pipeline

```python
# For each pixel (x, y):
ray_dir = get_ray_dir(x, y, FOV, width, height, camera_rot)
ray_dir = normalize(ray_dir)

# Test intersection with all spheres
for sphere in spheres:
    hit, distance = hit_sphere(sphere, ray_origin, ray_dir)
    # Keep closest hit
    
# Assign color based on closest sphere
```

## Configuration

You can modify these constants in the code:

```python
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 360    # Display resolution
RENDER_WIDTH, RENDER_HEIGHT = 160, 90     # Internal render resolution
FOV = 60                                  # Field of view in degrees
```

### High Resolution Rendering

To enable high-resolution rendering, modify the resolution constants:

```python
# For high resolution (warning: will be slower)
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080  # 1080p display
RENDER_WIDTH, RENDER_HEIGHT = 1920, 1080  # Match display resolution

# For balanced performance/quality
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720   # 720p display  
RENDER_WIDTH, RENDER_HEIGHT = 640, 360    # Half-res internal rendering
```

**Note**: Higher render resolution will significantly impact performance. The current low-resolution approach maintains smooth framerates while providing acceptable visual quality.

## Project Structure

```
ray-tracer/
├── raytracer.py      # Main ray tracer implementation
├── requirements.txt  # Python dependencies
└── README.md        # This file
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
- **Y-axis rotation** for horizontal camera movement
- **Proper aspect ratio** handling to prevent distortion

## Future Enhancements

- [ ] Add lighting and shadows
- [ ] Implement reflections and refractions
- [ ] Support for different primitive types (planes, triangles)
- [ ] Multi-threading for improved performance
- [ ] Anti-aliasing support
- [ ] Material system with different surface properties

## Performance

On a typical modern system (tested on i7 13th gen):
- **Low-res mode (160x90)**: ~5 FPS
- **High-res mode (640x360)**: ~0 FPS (slideshow mode, may cause system lag)
- **Full HD (1920x1080)**: Effectively frozen

Performance scales roughly with `O(width × height × sphere_count)`.

**Warning**: Higher resolutions will make your system unresponsive. Stick to the default 160x90 for interactive use.

**Note**: This is a pure Python implementation prioritizing code clarity over performance. The abysmal performance demonstrates why production ray tracers use C++ with SIMD optimizations, GPU acceleration, or specialized ray tracing hardware. This educational implementation shows the algorithms clearly but sacrifices all performance.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Ray tracing algorithms based on classic computer graphics techniques
- Built with Python, Pygame, and NumPy