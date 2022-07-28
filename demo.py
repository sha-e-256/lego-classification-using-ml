import sys
import pygame

WHITE = [255, 255, 255]
pygame.init()
screen_size = width, height = 1920, 1080 # Full screen
screen = pygame.display.set_mode(screen_size)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # Close window
            pygame.display.quit()
            sys.exit()

    screen.fill(WHITE)
    pygame.display.flip()
