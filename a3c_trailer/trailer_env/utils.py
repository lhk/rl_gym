import pygame

def rotate(surface, rect, angle):
    rotated_surface = pygame.transform.rotate(surface, angle)
    rotated_rect = rotated_surface.get_rect(center=rect.center)
    return rotated_surface, rotated_rect