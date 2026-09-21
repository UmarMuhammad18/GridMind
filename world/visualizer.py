"""
Optional Pygame visualizer for GridMind.

Falls back gracefully if pygame is not installed.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


CELL_SIZE = 48
MARGIN = 20
COLORS = {
    "#": (40, 40, 50),       # wall
    ".": (30, 32, 40),       # floor
    "A": (80, 220, 120),     # agent
    "K": (240, 200, 40),     # key
    "D": (220, 70, 70),      # door
    "G": (80, 200, 230),     # goal
    "?": (25, 25, 30),       # fog
    "bg": (18, 18, 24),
    "text": (220, 220, 230),
}


class Visualizer:
    def __init__(self, title: str = "GridMind"):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is not installed. Run: pip install pygame")

        pygame.init()
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("Consolas", 16)
        self.small_font = pygame.font.SysFont("Consolas", 13)
        self.screen: Optional[pygame.Surface] = None
        self.clock = pygame.time.Clock()

    def _ensure_screen(self, cols: int, rows: int):
        w = cols * CELL_SIZE + MARGIN * 2
        h = rows * CELL_SIZE + MARGIN * 2 + 90
        if self.screen is None or self.screen.get_size() != (w, h):
            self.screen = pygame.display.set_mode((w, h))

    def draw(
        self,
        grid: List[List[str]],
        agent_pos: Tuple[int, int],
        inventory: dict,
        door_open: bool,
        notes: List[str],
        step: int,
        max_steps: int,
        status: str = "",
    ):
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        self._ensure_screen(cols, rows)
        assert self.screen is not None

        self.screen.fill(COLORS["bg"])

        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                color = COLORS.get(cell, COLORS["."])
                rect = pygame.Rect(
                    MARGIN + x * CELL_SIZE,
                    MARGIN + y * CELL_SIZE,
                    CELL_SIZE - 2,
                    CELL_SIZE - 2,
                )
                pygame.draw.rect(self.screen, color, rect, border_radius=4)

                # Draw letter for key objects
                if cell in ("K", "D", "G", "A", "?"):
                    label = self.font.render(cell, True, (0, 0, 0) if cell != "?" else (100, 100, 110))
                    lr = label.get_rect(center=rect.center)
                    self.screen.blit(label, lr)

        # Agent on top
        ax, ay = agent_pos
        agent_rect = pygame.Rect(
            MARGIN + ax * CELL_SIZE,
            MARGIN + ay * CELL_SIZE,
            CELL_SIZE - 2,
            CELL_SIZE - 2,
        )
        pygame.draw.rect(self.screen, COLORS["A"], agent_rect, border_radius=4)
        label = self.font.render("A", True, (0, 0, 0))
        self.screen.blit(label, label.get_rect(center=agent_rect.center))

        # Status panel
        y0 = MARGIN + rows * CELL_SIZE + 8
        inv = "key" if inventory.get("has_key") else "empty"
        lines = [
            f"Step {step}/{max_steps}   Inv: {inv}   Door: {'open' if door_open else 'locked'}",
            f"Notes: {notes[-2:] if notes else '[]'}",
        ]
        if status:
            lines.append(status)

        for i, line in enumerate(lines):
            surf = self.small_font.render(line, True, COLORS["text"])
            self.screen.blit(surf, (MARGIN, y0 + i * 18))

        pygame.display.flip()
        self.clock.tick(30)

        # Process events so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True

    def close(self):
        if PYGAME_AVAILABLE:
            pygame.quit()
