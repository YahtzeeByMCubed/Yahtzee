# Example file showing a basic pygame "game loop"
import pygame
from misc import generate_random_yahtzee_game

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1080, 992))
clock = pygame.time.Clock()
running = True

yahtzee_card = pygame.image.load("assets/yahtzee_card.png").convert_alpha()
font = pygame.font.Font("assets/KOMIKAHB.ttf", 24)

x_offset = 0



games = [generate_random_yahtzee_game() for _ in range(5)]

def get_centers(start, end, count):
    step = (end - start) / count
    return [start + i * step + step / 2 for i in range(count)]

y_centers = (
    get_centers(140, 405, 6) +
    get_centers(410, 518, 3) +
    get_centers(552, 811, 7) +
    get_centers(840, 960, 4)
)

x_centers = get_centers(x_offset + 270, x_offset + 578, 5)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    # RENDER YOUR GAME HERE
    screen.blit(yahtzee_card, (0, 0))

    # Render games
    for col_idx, game in enumerate(games):
        cell_center_x = x_centers[col_idx]
        
        attributes = [
            game.ones, game.twos, game.threes, game.fours, game.fives, game.sixes,
            game.upper_section_total, game.upper_section_bonus, game.upper_section_total_with_bonus,
            game.three_of_a_kind, game.four_of_a_kind, game.full_house,
            game.small_straight, game.large_straight, game.yahtzee, game.chance,
            game.yahtzee_bonus, game.lower_section_total, game.upper_section_total_with_bonus, game.grand_total
        ]
        
        for row_idx, val in enumerate(attributes):
            if val is not None:
                val_text = font.render(str(val), True, "black")
                val_rect = val_text.get_rect()
                val_rect.center = (cell_center_x, y_centers[row_idx])
                screen.blit(val_text, val_rect)

    # Render mouse coordinates
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_text = font.render(f"X: {mouse_x}, Y: {mouse_y}", True, "black")
    text_rect = mouse_text.get_rect()
    text_rect.bottomright = (1470, 982) # Slight padding from the absolute edge
    screen.blit(mouse_text, text_rect)

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60

pygame.quit()