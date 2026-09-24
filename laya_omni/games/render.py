"""Draw game states as 256x256 frames for the pixels-only task.

Frames are plain top-down drawings with fixed colours. A single frame hides
velocity, so moving things leave a fading trail of their last few positions,
the way a player reads motion from a screen.
"""

from PIL import Image, ImageDraw

SIZE = 256
BG = (24, 26, 33)
GRID = (36, 39, 48)


def _canvas(width, height):
    """Fit a width x height cell board into the square frame; returns (image, draw, cell, x0, y0)."""
    cell = SIZE // max(width, height)
    x0, y0 = (SIZE - cell * width) // 2, (SIZE - cell * height) // 2
    img = Image.new("RGB", (SIZE, SIZE), (10, 11, 14))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x0 + cell * width - 1, y0 + cell * height - 1], fill=BG)
    return img, draw, cell, x0, y0


def _fade(color, t):
    return tuple(int(BG[i] + (color[i] - BG[i]) * t) for i in range(3))


def snake(game):
    img, draw, c, x0, y0 = _canvas(game.width, game.height)
    for x in range(game.width + 1):
        draw.line([x0 + x * c, y0, x0 + x * c, y0 + game.height * c], fill=GRID)
    for y in range(game.height + 1):
        draw.line([x0, y0 + y * c, x0 + game.width * c, y0 + y * c], fill=GRID)
    if game.food:
        fx, fy = game.food
        draw.ellipse([x0 + fx * c + 1, y0 + fy * c + 1, x0 + (fx + 1) * c - 2, y0 + (fy + 1) * c - 2], fill=(230, 70, 60))
    n = len(game.body)
    for i, (x, y) in enumerate(reversed(game.body)):
        t = 0.45 + 0.4 * i / max(1, n - 1)  # tail darker than the neck
        draw.rectangle([x0 + x * c + 1, y0 + y * c + 1, x0 + (x + 1) * c - 2, y0 + (y + 1) * c - 2], fill=_fade((90, 210, 110), t))
    hx, hy = game.head
    draw.rectangle([x0 + hx * c, y0 + hy * c, x0 + (hx + 1) * c - 1, y0 + (hy + 1) * c - 1], fill=(250, 240, 120))
    return img


def bird(game, history=()):
    """history: earlier bird y values, oldest first, one per decision step."""
    from .bird import BIRD_R, BIRD_X, PIPE_W

    img, draw, c, x0, y0 = _canvas(game.width, game.height)
    for p in game.pipes:
        left, right = x0 + (p["x"] - PIPE_W / 2) * c, x0 + (p["x"] + PIPE_W / 2) * c
        top, bottom = y0 + (p["gap_y"] - p["gap"] / 2) * c, y0 + (p["gap_y"] + p["gap"] / 2) * c
        if right < x0 or left > x0 + game.width * c:
            continue
        draw.rectangle([left, y0, right, top], fill=(150, 140, 120))
        draw.rectangle([left, bottom, right, y0 + game.height * c], fill=(150, 140, 120))
    r = BIRD_R * c * 1.3
    step_dx = 10 * 0.044  # frames per step x scroll: past positions drift left
    for k, y in enumerate(reversed(history[-4:]), start=1):
        bx, by = x0 + (BIRD_X - k * step_dx) * c, y0 + y * c
        draw.ellipse([bx - r * 0.7, by - r * 0.7, bx + r * 0.7, by + r * 0.7], fill=_fade((240, 200, 60), 0.55 / k))
    bx, by = x0 + BIRD_X * c, y0 + game.y * c
    draw.ellipse([bx - r, by - r, bx + r, by + r], fill=(245, 205, 60))
    return img


def bricks(game, history=()):
    """history: earlier (ball_x, ball_y) positions, oldest first."""
    from .bricks import BALL_R, BRICK_W, HEIGHT, PADDLE_W, PADDLE_Y, WIDTH

    img, draw, c, x0, y0 = _canvas(WIDTH, HEIGHT)
    palette = [(220, 90, 80), (230, 150, 70), (225, 205, 80), (110, 200, 110), (80, 170, 220), (150, 120, 220)]
    for r, col in game.bricks:
        draw.rectangle(
            [x0 + col * BRICK_W * c + 1, y0 + r * c + 1, x0 + (col + 1) * BRICK_W * c - 2, y0 + (r + 1) * c - 2],
            fill=palette[(r - 1) % len(palette)],
        )
    rad = BALL_R * c * 1.4
    for k, (bx, by) in enumerate(reversed(history[-5:]), start=1):
        px, py = x0 + bx * c, y0 + by * c
        draw.ellipse([px - rad * 0.8, py - rad * 0.8, px + rad * 0.8, py + rad * 0.8], fill=_fade((240, 240, 240), 0.6 / k))
    px, py = x0 + game.ball_x * c, y0 + game.ball_y * c
    draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=(250, 250, 250))
    half = PADDLE_W / 2
    draw.rectangle(
        [x0 + (game.paddle_x - half) * c, y0 + (PADDLE_Y - 0.2) * c, x0 + (game.paddle_x + half) * c, y0 + (PADDLE_Y + 0.4) * c],
        fill=(120, 190, 250),
    )
    return img
