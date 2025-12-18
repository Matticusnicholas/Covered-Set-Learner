"""
Streaming-Friendly Visualization Interface
==========================================
Beautiful, animated visualization for YouTube/TikTok streaming.

Features:
- Real-time neural network training visualization
- Animated number grid with coverage highlighting
- Live statistics and progress bars
- Particle effects and smooth animations
- Dark theme optimized for streaming
"""

import pygame
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
import colorsys


# Color Palette (Cyberpunk/Neon theme)
class Colors:
    # Backgrounds
    BG_DARK = (10, 10, 20)
    BG_PANEL = (20, 20, 35)
    BG_CARD = (30, 30, 50)

    # Neon accents
    NEON_CYAN = (0, 255, 255)
    NEON_PINK = (255, 0, 128)
    NEON_GREEN = (0, 255, 128)
    NEON_ORANGE = (255, 128, 0)
    NEON_PURPLE = (180, 0, 255)
    NEON_YELLOW = (255, 255, 0)

    # Status colors
    COVERED = (0, 200, 100)
    UNCOVERED = (200, 50, 50)
    PARTIAL = (255, 180, 0)
    SELECTED = (100, 200, 255)

    # Text
    TEXT_PRIMARY = (255, 255, 255)
    TEXT_SECONDARY = (150, 150, 170)
    TEXT_MUTED = (100, 100, 120)

    # Grid
    GRID_LINE = (40, 40, 60)
    NUMBER_BG = (25, 25, 40)
    NUMBER_HOT = (60, 20, 60)


@dataclass
class Particle:
    """Animated particle for visual effects."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    size: float
    life: float
    max_life: float

    def update(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 50 * dt  # Gravity
        self.life -= dt
        self.size *= 0.98

    @property
    def alpha(self) -> float:
        return max(0, self.life / self.max_life)

    @property
    def is_alive(self) -> bool:
        return self.life > 0 and self.size > 0.5


class ParticleSystem:
    """Manages particle effects."""

    def __init__(self):
        self.particles: List[Particle] = []

    def emit(self, x: float, y: float, count: int = 10,
             color: Tuple[int, int, int] = Colors.NEON_CYAN,
             speed: float = 100):
        """Emit particles at position."""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            velocity = random.uniform(speed * 0.5, speed)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(angle) * velocity,
                vy=math.sin(angle) * velocity - 50,
                color=color,
                size=random.uniform(2, 6),
                life=random.uniform(0.5, 1.5),
                max_life=1.5
            ))

    def update(self, dt: float):
        """Update all particles."""
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.is_alive]

    def draw(self, surface: pygame.Surface):
        """Draw all particles."""
        for p in self.particles:
            alpha = int(p.alpha * 255)
            color = (*p.color, alpha)
            # Create surface with alpha
            s = pygame.Surface((int(p.size * 2), int(p.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (int(p.size), int(p.size)), int(p.size))
            surface.blit(s, (int(p.x - p.size), int(p.y - p.size)))


class AnimatedValue:
    """Smoothly animated value for UI elements."""

    def __init__(self, initial: float = 0, speed: float = 5):
        self.value = initial
        self.target = initial
        self.speed = speed

    def set(self, target: float):
        self.target = target

    def update(self, dt: float):
        diff = self.target - self.value
        self.value += diff * min(1, self.speed * dt)

    def get(self) -> float:
        return self.value


class NumberGrid:
    """Animated number grid showing lottery numbers."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 pool_size: int = 36, cols: int = 6):
        self.rect = pygame.Rect(x, y, width, height)
        self.pool_size = pool_size
        self.cols = cols
        self.rows = math.ceil(pool_size / cols)

        # Calculate cell size
        padding = 10
        self.cell_width = (width - padding * 2) // cols
        self.cell_height = (height - padding * 2) // self.rows
        self.cell_size = min(self.cell_width, self.cell_height) - 4

        # Animation states
        self.number_heat = {i: AnimatedValue(0, 3) for i in range(pool_size)}
        self.selected_numbers = set()
        self.pulse_phase = 0

    def update_heat(self, heat_map: Dict[int, float]):
        """Update heat values for numbers."""
        for num, heat in heat_map.items():
            if num in self.number_heat:
                self.number_heat[num].set(heat)

    def set_selected(self, numbers: set):
        """Set currently selected numbers."""
        self.selected_numbers = numbers

    def update(self, dt: float):
        """Update animations."""
        self.pulse_phase += dt * 3
        for anim in self.number_heat.values():
            anim.update(dt)

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Draw the number grid."""
        # Background
        pygame.draw.rect(surface, Colors.BG_PANEL, self.rect, border_radius=15)
        pygame.draw.rect(surface, Colors.GRID_LINE, self.rect, 2, border_radius=15)

        padding = 10
        start_x = self.rect.x + padding + (self.rect.width - padding * 2 - self.cols * self.cell_width) // 2
        start_y = self.rect.y + padding

        for num in range(self.pool_size):
            row = num // self.cols
            col = num % self.cols

            cx = start_x + col * self.cell_width + self.cell_width // 2
            cy = start_y + row * self.cell_height + self.cell_height // 2

            # Heat color
            heat = self.number_heat[num].get()
            is_selected = num in self.selected_numbers

            # Background color based on state
            if is_selected:
                # Pulsing selection effect
                pulse = 0.5 + 0.5 * math.sin(self.pulse_phase + num * 0.5)
                bg_color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in Colors.NEON_CYAN)
                text_color = Colors.BG_DARK
            elif heat > 0.5:
                # Hot numbers (frequently in uncovered draws)
                intensity = min(1, heat)
                bg_color = (
                    int(60 + 140 * intensity),
                    int(20 + 30 * intensity),
                    int(60 + 100 * intensity)
                )
                text_color = Colors.NEON_PINK
            else:
                bg_color = Colors.NUMBER_BG
                text_color = Colors.TEXT_PRIMARY

            # Draw cell
            cell_rect = pygame.Rect(
                cx - self.cell_size // 2,
                cy - self.cell_size // 2,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(surface, bg_color, cell_rect, border_radius=8)

            if is_selected:
                # Glow effect for selected
                glow_rect = cell_rect.inflate(4, 4)
                pygame.draw.rect(surface, Colors.NEON_CYAN, glow_rect, 2, border_radius=10)

            # Draw number
            num_text = font.render(str(num + 1), True, text_color)
            text_rect = num_text.get_rect(center=(cx, cy))
            surface.blit(num_text, text_rect)


class ProgressBar:
    """Animated progress bar."""

    def __init__(self, x: int, y: int, width: int, height: int,
                 color: Tuple[int, int, int] = Colors.NEON_CYAN):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.progress = AnimatedValue(0, 4)
        self.glow_phase = 0

    def set_progress(self, value: float):
        """Set progress (0-1)."""
        self.progress.set(max(0, min(1, value)))

    def update(self, dt: float):
        self.progress.update(dt)
        self.glow_phase += dt * 5

    def draw(self, surface: pygame.Surface, label: str = "",
             font: pygame.font.Font = None):
        """Draw the progress bar."""
        # Background
        pygame.draw.rect(surface, Colors.BG_CARD, self.rect, border_radius=self.rect.height // 2)

        # Progress fill
        fill_width = int(self.rect.width * self.progress.get())
        if fill_width > 0:
            fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
            pygame.draw.rect(surface, self.color, fill_rect, border_radius=self.rect.height // 2)

            # Animated shine effect
            shine_x = (self.glow_phase * 50) % (fill_width + 100) - 50
            if 0 < shine_x < fill_width:
                shine_surface = pygame.Surface((30, self.rect.height), pygame.SRCALPHA)
                for i in range(30):
                    alpha = int(100 * (1 - abs(i - 15) / 15))
                    pygame.draw.line(shine_surface, (*Colors.TEXT_PRIMARY, alpha),
                                   (i, 0), (i, self.rect.height))
                surface.blit(shine_surface, (self.rect.x + shine_x, self.rect.y))

        # Border
        pygame.draw.rect(surface, Colors.GRID_LINE, self.rect, 2, border_radius=self.rect.height // 2)

        # Label
        if label and font:
            label_text = font.render(label, True, Colors.TEXT_PRIMARY)
            label_rect = label_text.get_rect(midleft=(self.rect.right + 10, self.rect.centery))
            surface.blit(label_text, label_rect)


class StatsPanel:
    """Panel showing training statistics."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.stats = {}
        self.history = {
            'coverage': deque(maxlen=100),
            'tickets': deque(maxlen=100),
            'efficiency': deque(maxlen=100)
        }

    def update_stats(self, stats: Dict):
        """Update statistics."""
        self.stats = stats
        if 'coverage' in stats:
            self.history['coverage'].append(stats['coverage'])
        if 'num_tickets' in stats:
            self.history['tickets'].append(stats['num_tickets'])
        if 'efficiency' in stats:
            self.history['efficiency'].append(stats['efficiency'])

    def draw(self, surface: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        """Draw the stats panel."""
        # Background
        pygame.draw.rect(surface, Colors.BG_PANEL, self.rect, border_radius=15)
        pygame.draw.rect(surface, Colors.GRID_LINE, self.rect, 2, border_radius=15)

        padding = 15
        y_offset = self.rect.y + padding

        # Title
        title = fonts['medium'].render("📊 TRAINING STATS", True, Colors.NEON_CYAN)
        surface.blit(title, (self.rect.x + padding, y_offset))
        y_offset += 35

        # Stats
        stat_items = [
            ("Generation", self.stats.get('generation', 0), Colors.NEON_PURPLE),
            ("Coverage", f"{self.stats.get('coverage', 0):.1f}%", Colors.NEON_GREEN),
            ("Tickets", self.stats.get('num_tickets', 0), Colors.NEON_ORANGE),
            ("Best Coverage", f"{self.stats.get('best_coverage', 0):.1f}%", Colors.NEON_PINK),
            ("Efficiency", f"{self.stats.get('efficiency', 0):.2f}", Colors.NEON_YELLOW),
        ]

        for label, value, color in stat_items:
            # Label
            label_text = fonts['small'].render(label, True, Colors.TEXT_SECONDARY)
            surface.blit(label_text, (self.rect.x + padding, y_offset))

            # Value
            value_text = fonts['medium'].render(str(value), True, color)
            surface.blit(value_text, (self.rect.x + padding, y_offset + 18))

            y_offset += 50

        # Mini graph
        if len(self.history['coverage']) > 1:
            graph_rect = pygame.Rect(
                self.rect.x + padding,
                y_offset + 10,
                self.rect.width - padding * 2,
                60
            )
            self._draw_mini_graph(surface, graph_rect, fonts['tiny'])

    def _draw_mini_graph(self, surface: pygame.Surface, rect: pygame.Rect,
                        font: pygame.font.Font):
        """Draw mini coverage graph."""
        pygame.draw.rect(surface, Colors.BG_CARD, rect, border_radius=5)

        data = list(self.history['coverage'])
        if len(data) < 2:
            return

        # Scale data
        max_val = max(data) if data else 100
        min_val = min(data) if data else 0
        range_val = max(max_val - min_val, 1)

        points = []
        for i, val in enumerate(data):
            x = rect.x + (i / (len(data) - 1)) * rect.width
            y = rect.bottom - ((val - min_val) / range_val) * rect.height
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(surface, Colors.NEON_GREEN, False, points, 2)

        # Label
        label = font.render("Coverage History", True, Colors.TEXT_MUTED)
        surface.blit(label, (rect.x + 5, rect.y + 5))


class TicketDisplay:
    """Display generated tickets with animation."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.tickets: List[Tuple[int, ...]] = []
        self.new_ticket_animation = 0
        self.scroll_offset = 0

    def add_ticket(self, ticket: Tuple[int, ...]):
        """Add a new ticket with animation."""
        self.tickets.append(ticket)
        self.new_ticket_animation = 1.0
        # Auto-scroll to show new ticket
        max_visible = (self.rect.height - 50) // 30
        if len(self.tickets) > max_visible:
            self.scroll_offset = len(self.tickets) - max_visible

    def clear(self):
        """Clear all tickets."""
        self.tickets = []
        self.scroll_offset = 0

    def update(self, dt: float):
        """Update animations."""
        if self.new_ticket_animation > 0:
            self.new_ticket_animation -= dt * 3

    def draw(self, surface: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        """Draw ticket display."""
        # Background
        pygame.draw.rect(surface, Colors.BG_PANEL, self.rect, border_radius=15)
        pygame.draw.rect(surface, Colors.GRID_LINE, self.rect, 2, border_radius=15)

        padding = 15
        y_offset = self.rect.y + padding

        # Title
        title = fonts['medium'].render(f"🎫 TICKETS ({len(self.tickets)})", True, Colors.NEON_ORANGE)
        surface.blit(title, (self.rect.x + padding, y_offset))
        y_offset += 35

        # Tickets list
        max_visible = (self.rect.height - 60) // 28
        visible_tickets = self.tickets[self.scroll_offset:self.scroll_offset + max_visible]

        for i, ticket in enumerate(visible_tickets):
            actual_idx = self.scroll_offset + i
            is_new = actual_idx == len(self.tickets) - 1 and self.new_ticket_animation > 0

            # Animation for new ticket
            if is_new:
                alpha = int(255 * (1 - self.new_ticket_animation))
                scale = 1 + 0.2 * self.new_ticket_animation
            else:
                alpha = 255
                scale = 1

            # Ticket number
            num_text = fonts['tiny'].render(f"#{actual_idx + 1:03d}", True,
                                           (*Colors.TEXT_MUTED[:3], alpha))
            surface.blit(num_text, (self.rect.x + padding, y_offset))

            # Numbers
            numbers_str = " ".join(f"{n+1:2d}" for n in ticket)
            color = Colors.NEON_CYAN if is_new else Colors.TEXT_PRIMARY
            numbers_text = fonts['small'].render(numbers_str, True, (*color[:3], alpha))
            surface.blit(numbers_text, (self.rect.x + padding + 50, y_offset))

            y_offset += 28

        # Scroll indicator
        if len(self.tickets) > max_visible:
            indicator_text = fonts['tiny'].render(
                f"↕ {self.scroll_offset + 1}-{min(self.scroll_offset + max_visible, len(self.tickets))} of {len(self.tickets)}",
                True, Colors.TEXT_MUTED
            )
            surface.blit(indicator_text, (self.rect.x + padding, self.rect.bottom - 25))


class NeuralNetworkViz:
    """Visual representation of the neural network."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.layers = [36, 64, 32, 36]  # Simplified representation
        self.activations = [[0.0] * n for n in self.layers]
        self.pulse_phase = 0

    def set_activations(self, layer: int, values: List[float]):
        """Set activation values for a layer."""
        if 0 <= layer < len(self.layers):
            self.activations[layer] = values[:self.layers[layer]]

    def update(self, dt: float):
        """Update animations."""
        self.pulse_phase += dt * 2

        # Random activations for visual effect
        for i, layer in enumerate(self.activations):
            for j in range(len(layer)):
                # Add some noise for visual interest
                self.activations[i][j] = 0.3 + 0.7 * abs(
                    math.sin(self.pulse_phase + i * 0.5 + j * 0.3)
                )

    def draw(self, surface: pygame.Surface, fonts: Dict[str, pygame.font.Font]):
        """Draw neural network visualization."""
        # Background
        pygame.draw.rect(surface, Colors.BG_PANEL, self.rect, border_radius=15)
        pygame.draw.rect(surface, Colors.GRID_LINE, self.rect, 2, border_radius=15)

        padding = 15

        # Title
        title = fonts['medium'].render("🧠 NEURAL NETWORK", True, Colors.NEON_PURPLE)
        surface.blit(title, (self.rect.x + padding, self.rect.y + padding))

        # Draw layers
        layer_spacing = (self.rect.width - padding * 2) / (len(self.layers) + 1)
        max_nodes_display = 8

        for layer_idx, (num_nodes, activations) in enumerate(zip(self.layers, self.activations)):
            x = self.rect.x + padding + layer_spacing * (layer_idx + 1)
            nodes_to_draw = min(num_nodes, max_nodes_display)

            node_spacing = (self.rect.height - 80) / (nodes_to_draw + 1)

            for node_idx in range(nodes_to_draw):
                y = self.rect.y + 50 + node_spacing * (node_idx + 1)

                # Activation color
                if node_idx < len(activations):
                    activation = activations[node_idx]
                else:
                    activation = 0.5

                # Color based on activation
                hue = 0.5 + 0.3 * activation  # Cyan to green
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = tuple(int(c * 255) for c in rgb)

                # Draw node
                radius = int(4 + 4 * activation)
                pygame.draw.circle(surface, color, (int(x), int(y)), radius)

                # Draw connections to next layer
                if layer_idx < len(self.layers) - 1:
                    next_x = x + layer_spacing
                    next_nodes = min(self.layers[layer_idx + 1], max_nodes_display)
                    next_spacing = (self.rect.height - 80) / (next_nodes + 1)

                    for next_node_idx in range(min(3, next_nodes)):  # Limit connections drawn
                        next_y = self.rect.y + 50 + next_spacing * (next_node_idx + 1)
                        alpha = int(50 * activation)
                        line_color = (*Colors.NEON_PURPLE[:3], alpha)
                        # Can't easily do alpha lines in pygame, so use faded color
                        faded_color = tuple(int(c * 0.3) for c in Colors.NEON_PURPLE)
                        pygame.draw.line(surface, faded_color,
                                       (int(x), int(y)), (int(next_x), int(next_y)), 1)

            # Layer label
            if num_nodes > max_nodes_display:
                more_text = fonts['tiny'].render(f"+{num_nodes - max_nodes_display} more",
                                                True, Colors.TEXT_MUTED)
                surface.blit(more_text, (x - 20, self.rect.bottom - 25))


class StreamingVisualizer:
    """Main visualization class for streaming."""

    def __init__(self, width: int = 1920, height: int = 1080,
                 pool_size: int = 36, draw_size: int = 5, match_required: int = 3):
        pygame.init()
        pygame.display.set_caption("🎰 Neural Network Abbreviated Covered Set Learner")

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        # Abbreviated wheel configuration
        self.pool_size = pool_size
        self.draw_size = draw_size
        self.match_required = match_required
        self.wheel_name = f"{match_required} of {draw_size} from {pool_size}"

        # Fonts
        self.fonts = self._load_fonts()

        # Calculate grid columns based on pool size
        grid_cols = 6 if pool_size <= 42 else 7 if pool_size <= 56 else 8

        # UI Components
        self.number_grid = NumberGrid(50, 100, 400, 320, pool_size=pool_size, cols=grid_cols)
        self.progress_bar = ProgressBar(50, 440, 400, 25, Colors.NEON_GREEN)
        self.stats_panel = StatsPanel(50, 490, 400, 330)
        self.ticket_display = TicketDisplay(480, 100, 350, 720)
        self.neural_viz = NeuralNetworkViz(860, 100, 500, 320)

        # Effects
        self.particles = ParticleSystem()

        # State
        self.running = True
        self.generation = 0
        self.best_coverage = 0

    def _load_fonts(self) -> Dict[str, pygame.font.Font]:
        """Load fonts for UI."""
        try:
            return {
                'large': pygame.font.Font(None, 48),
                'medium': pygame.font.Font(None, 32),
                'small': pygame.font.Font(None, 24),
                'tiny': pygame.font.Font(None, 18),
            }
        except:
            return {
                'large': pygame.font.SysFont('arial', 40),
                'medium': pygame.font.SysFont('arial', 28),
                'small': pygame.font.SysFont('arial', 20),
                'tiny': pygame.font.SysFont('arial', 14),
            }

    def update(self, dt: float, stats: Dict = None, current_ticket: Tuple[int, ...] = None,
               heat_map: Dict[int, float] = None):
        """Update visualization state."""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

        # Update components
        self.number_grid.update(dt)
        self.progress_bar.update(dt)
        self.particles.update(dt)
        self.neural_viz.update(dt)
        self.ticket_display.update(dt)

        # Update with new data
        if stats:
            self.stats_panel.update_stats(stats)
            coverage = stats.get('coverage', 0)
            self.progress_bar.set_progress(coverage / 100)

            if coverage > self.best_coverage:
                self.best_coverage = coverage
                # Celebration particles
                self.particles.emit(
                    self.progress_bar.rect.centerx,
                    self.progress_bar.rect.centery,
                    count=20,
                    color=Colors.NEON_GREEN,
                    speed=150
                )

        if current_ticket:
            self.number_grid.set_selected(set(current_ticket))

        if heat_map:
            self.number_grid.update_heat(heat_map)

    def add_ticket(self, ticket: Tuple[int, ...]):
        """Add a newly generated ticket."""
        self.ticket_display.add_ticket(ticket)
        # Particle effect
        self.particles.emit(
            self.ticket_display.rect.centerx,
            self.ticket_display.rect.y + 50,
            count=10,
            color=Colors.NEON_ORANGE
        )

    def new_generation(self):
        """Start a new generation."""
        self.generation += 1
        self.ticket_display.clear()

    def draw(self):
        """Render the visualization."""
        # Clear screen
        self.screen.fill(Colors.BG_DARK)

        # Draw main title with wheel configuration
        title = self.fonts['large'].render(
            f"🎰 ABBREVIATED WHEEL: {self.wheel_name}", True, Colors.NEON_CYAN
        )
        title_rect = title.get_rect(center=(self.width // 2, 25))
        self.screen.blit(title, title_rect)

        # Draw subtitle explaining what this means
        subtitle = self.fonts['small'].render(
            f"Finding minimum tickets to guarantee {self.match_required}+ matches | CUDA Accelerated",
            True, Colors.TEXT_SECONDARY
        )
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 55))
        self.screen.blit(subtitle, subtitle_rect)

        # Draw configuration box at top right
        config_box = pygame.Rect(self.width - 320, 10, 310, 75)
        pygame.draw.rect(self.screen, Colors.BG_PANEL, config_box, border_radius=10)
        pygame.draw.rect(self.screen, Colors.NEON_PURPLE, config_box, 2, border_radius=10)

        # Configuration details
        config_lines = [
            (f"Pool: {self.pool_size} numbers", Colors.TEXT_PRIMARY),
            (f"Pick: {self.draw_size} per ticket", Colors.TEXT_PRIMARY),
            (f"Match: {self.match_required}+ to win", Colors.NEON_GREEN),
        ]
        for i, (text, color) in enumerate(config_lines):
            conf_text = self.fonts['small'].render(text, True, color)
            self.screen.blit(conf_text, (self.width - 310, 18 + i * 22))

        # Draw components
        self.number_grid.draw(self.screen, self.fonts['medium'])
        self.progress_bar.draw(self.screen, f"{self.progress_bar.progress.get() * 100:.1f}%",
                              self.fonts['small'])
        self.stats_panel.draw(self.screen, self.fonts)
        self.ticket_display.draw(self.screen, self.fonts)
        self.neural_viz.draw(self.screen, self.fonts)

        # Draw particles (on top)
        self.particles.draw(self.screen)

        # Draw FPS
        fps = self.clock.get_fps()
        fps_text = self.fonts['tiny'].render(f"FPS: {fps:.0f}", True, Colors.TEXT_MUTED)
        self.screen.blit(fps_text, (self.width - 70, self.height - 25))

        # Update display
        pygame.display.flip()

    def run_frame(self) -> float:
        """Run one frame and return delta time."""
        dt = self.clock.tick(60) / 1000.0  # 60 FPS cap
        return dt

    def cleanup(self):
        """Clean up pygame."""
        pygame.quit()


if __name__ == "__main__":
    # Test visualization with configurable abbreviated wheel
    # Example: 3 of 5 from 36 (default)
    pool_size = 36
    draw_size = 5
    match_required = 3

    viz = StreamingVisualizer(1280, 720, pool_size, draw_size, match_required)

    # Demo loop
    import time
    demo_tickets = [
        tuple(sorted(random.sample(range(pool_size), draw_size)))
        for _ in range(3)
    ]

    demo_coverage = 0
    demo_gen = 0

    while viz.running:
        dt = viz.run_frame()

        # Simulate training progress
        demo_coverage += dt * 5
        if demo_coverage > 100:
            demo_coverage = 0
            demo_gen += 1
            viz.new_generation()

        # Add demo ticket periodically
        if random.random() < dt * 2:
            ticket = tuple(sorted(random.sample(range(pool_size), draw_size)))
            viz.add_ticket(ticket)

        # Update with demo stats
        stats = {
            'generation': demo_gen,
            'coverage': demo_coverage,
            'num_tickets': len(viz.ticket_display.tickets),
            'best_coverage': max(demo_coverage, 50),
            'efficiency': demo_coverage / max(1, len(viz.ticket_display.tickets))
        }

        # Demo heat map
        heat_map = {i: random.random() for i in range(pool_size)}

        viz.update(dt, stats, demo_tickets[demo_gen % 3] if demo_tickets else None, heat_map)
        viz.draw()

    viz.cleanup()
