import pygame
import numpy as np
import sys
from math import sqrt, tan, radians, sin, cos

SCREEN_WIDTH, SCREEN_HEIGHT = 640, 360
RENDER_WIDTH, RENDER_HEIGHT = 640, 360  # Internal resolution (faster)
FOV = 60

# Spheres: (position, radius, color)
spheres = [
    (np.array([-1.5, 0, -5]), 1, (255, 0, 0)),
    (np.array([0.0, 0, -5]), 1, (0, 255, 0)),
    (np.array([1.5, 0, -5]), 1, (0, 0, 255)),
]

BLACK = (0, 0, 0)

def hit_sphere(center, radius, ray_origin, ray_dir):
    oc = ray_origin - center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False, float('inf')
    else:
        t = (-b - sqrt(discriminant)) / (2.0 * a)
        return True, t

def get_ray_dir(x, y, fov, width, height, camera_rot):
    aspect_ratio = width / height
    px = (2 * ((x + 0.5) / width) - 1) * aspect_ratio * tan(radians(fov / 2))
    py = (1 - 2 * ((y + 0.5) / height)) * tan(radians(fov / 2))
    dir = np.array([px, py, -1])

    # Rotate around Y (horizontal camera rotation)
    cos_a = cos(camera_rot)
    sin_a = sin(camera_rot)
    rot_x = dir[0] * cos_a - dir[2] * sin_a
    rot_z = dir[0] * sin_a + dir[2] * cos_a
    return np.array([rot_x, dir[1], rot_z])

def render(render_surface, camera_pos, camera_rot):
    surface = pygame.surfarray.pixels3d(render_surface)
    for y in range(RENDER_HEIGHT):
        for x in range(RENDER_WIDTH):
            ray_dir = get_ray_dir(x, y, FOV, RENDER_WIDTH, RENDER_HEIGHT, camera_rot)
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            color = BLACK
            min_dist = float('inf')
            for center, radius, sphere_color in spheres:
                hit, dist = hit_sphere(center, radius, camera_pos, ray_dir)
                if hit and dist < min_dist:
                    min_dist = dist
                    color = sphere_color

            surface[x, y] = color
    del surface

def move_camera(pos, rot, keys, dt):
    speed = 3 * dt
    dir_forward = np.array([-sin(rot), 0, -cos(rot)])
    dir_right = np.array([cos(rot), 0, -sin(rot)])

    if keys[pygame.K_w]: pos += dir_forward * speed
    if keys[pygame.K_s]: pos -= dir_forward * speed
    if keys[pygame.K_a]: pos -= dir_right * speed
    if keys[pygame.K_d]: pos += dir_right * speed
    if keys[pygame.K_q]: pos[1] += speed
    if keys[pygame.K_e]: pos[1] -= speed
    if keys[pygame.K_LEFT]: rot -= 1.5 * dt
    if keys[pygame.K_RIGHT]: rot += 1.5 * dt
    return pos, rot

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ray Tracer - FOV Scaling + FPS")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    render_surface = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT))
    camera_pos = np.array([0.0, 0.0, 0.0])
    camera_rot = 0.0

    running = True
    while running:
        dt = clock.tick(30) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        camera_pos, camera_rot = move_camera(camera_pos, camera_rot, keys, dt)
        render(render_surface, camera_pos, camera_rot)

        scaled = pygame.transform.scale(render_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.blit(scaled, (0, 0))

        # FPS Counter
        fps = int(clock.get_fps())
        text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
