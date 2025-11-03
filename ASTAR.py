import pygame
import math
from queue import PriorityQueue
import enum
import datetime

# --- WINDOW SETTINGS ---
WIDTH = 800
ALPHA = 30
THINNER = 1
MIN_PENALTY = 0.1
emitters = set()
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Pathfinding and modified A*")
pygame.init()
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 16)

# Global variables to persist costs
last_g_score = None
last_lowest_cost = float("inf")

IS_MODIFIED = True

# --- COLORS ---
RED = (255, 0, 0)
GREEN = (30, 255, 30)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (0xB0, 0x0B, 0x69)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (99, 214, 187)
DARK_BLUE = (102, 139, 242)
DARK_GREEN = (73, 147, 166)
BROWN = (86, 43, 0)
VIOLET = (240, 32, 204)

t = datetime.datetime.now()
# zero
g_diff_time = t - t
g_iterations = 0

g_path_length = None
g_nearest_emitter_distance = None


class SpotKind(enum.Enum):
    Empty = enum.auto()

    Barrier = enum.auto()

    Blocked = enum.auto()
    """Fire"""

    Weighted = enum.auto()
    """Around the fire"""

    Start = enum.auto()
    End = enum.auto()

    def get_color(self, penalty) -> tuple[int, int, int]:
        match self:
            case SpotKind.Empty:
                return WHITE
            case SpotKind.Barrier:
                return BLACK
            case SpotKind.Blocked:
                return RED
            case SpotKind.Weighted:
                penalty /= THINNER
                penalty = min(1, penalty)
                penalty = max(0, penalty)
                new_color = mix_colors(YELLOW, WHITE, penalty)
                return new_color
            case SpotKind.Start:
                return BROWN
            case SpotKind.End:
                return PURPLE

def mix_colors(color2, color1, penalty):
    #color1+penalty*(color2-color1)
    r1,g1,b1 = color1
    r2,g2,b2 = color2
    r = int(r1 + penalty * (r2 - r1))
    g = int(g1 + penalty * (g2 - g1))
    b = int(b1 + penalty * (b2 - b1))
    return ((r,g,b))


class SpotPathState(enum.Enum):
    Empty = enum.auto()

    Closed = enum.auto()
    Open = enum.auto()

    Path = enum.auto()

    def get_color(self) -> tuple[int, int, int]:
        match self:
            case SpotPathState.Empty:
                return WHITE
            case SpotPathState.Closed:
                return DARK_BLUE
            case SpotPathState.Open:
                return GREEN
            case SpotPathState.Path:
                return VIOLET


# --- SPOT CLASS ---
class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.kind = SpotKind.Empty
        self.path_state = SpotPathState.Empty
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.penalty = 0

    def get_pos(self):
        return self.row, self.col

    def is_empty(self):
        if IS_MODIFIED:
            return self.kind == SpotKind.Empty
        return self.kind in (SpotKind.Blocked, SpotKind.Weighted)

    def is_closed(self):
        return self.path_state == SpotPathState.Closed

    def is_open(self):
        return self.path_state == SpotPathState.Open

    def is_barrier(self):
        return self.kind == SpotKind.Barrier

    def is_blocked(self):
        if IS_MODIFIED:
            return self.kind == SpotKind.Blocked
        return False

    def is_weighted(self):
        if IS_MODIFIED:
            return self.kind == SpotKind.Weighted
        return False

    def is_start(self):
        return self.kind == SpotKind.Start

    def is_end(self):
        return self.kind == SpotKind.End

    def is_path(self):
        return self.path_state == SpotPathState.Path

    def reset_all(self):
        self.kind = SpotKind.Empty
        self.path_state = SpotPathState.Empty

    def reset_path_state(self):
        self.path_state = SpotPathState.Empty

    def make_start(self):
        self.kind = SpotKind.Start

    def make_closed(self):
        self.path_state = SpotPathState.Closed

    def make_open(self):
        self.path_state = SpotPathState.Open

    def make_barrier(self):
        self.kind = SpotKind.Barrier

    def make_blocked(self):
        self.kind = SpotKind.Blocked

    def make_weighted(self):
        self.kind = SpotKind.Weighted

    def make_end(self):
        self.kind = SpotKind.End

    def make_path(self):
        self.path_state = SpotPathState.Path

    def get_color(self) -> tuple[int, int, int]:
        if self.kind in (SpotKind.Empty, SpotKind.Weighted) and self.path_state != SpotPathState.Empty:
            return self.path_state.get_color()
        # Calculate penalty from emitters
        penalty = self.get_penalty_from_emitters() // 4
        return self.kind.get_color(penalty)

    def draw(self, win):
        color = self.get_color()
        pygame.draw.rect(win, color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        """Include 8 directions (diagonal movement)"""
        self.neighbors = []
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for dr, dc in directions:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < self.total_rows and 0 <= c < self.total_rows:
                spot = grid[r][c]
                if not spot.is_barrier() and not spot.is_blocked():
                    self.neighbors.append(grid[r][c])

    def __lt__(self, other):
        return False

    def get_penalty_from_emitters(self):
        """Calculate accumulated penalty from all emitters"""
        accumulated_penalty = 0
        for emitter in emitters:
            penalty = calculate_penalty(self.get_pos(), emitter.get_pos())
            accumulated_penalty += penalty
        
        return accumulated_penalty if accumulated_penalty >= MIN_PENALTY else 0

    def __repr__(self):
        return f"Spot<({self.row}, {self.col}), {self.kind}>"

def clear_grid(start, end, grid):
    for row in grid:
        for spot in row:
            if spot != start and spot != end:
                spot.reset_path_state()

# --- HEURISTIC FUNCTION (OCTILE DISTANCE) ---
def distance(p1, p2):
    """Heuristic that accounts for diagonal movement"""
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    D = 1
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def euclidean_distance(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def calculate_penalty(p1,p2):
    distance = euclidean_distance(p1, p2)
    return ALPHA * (1 / ((distance + 1)**3))

def update_grid_weights(grid):
    """Update all spots in grid based on emitter penalties"""
    for row in grid:
        for spot in row:
            if spot.kind == SpotKind.Empty:
                spot.penalty = spot.get_penalty_from_emitters()
                if spot.penalty >= MIN_PENALTY:
                    spot.make_weighted()
                else:
                    spot.kind = SpotKind.Empty

# --- RECONSTRUCT PATH ---
def reconstruct_path(came_from, current, draw):
    global g_path_length, g_nearest_emitter_distance
    g_path_length = 1
    g_nearest_emitter_distance = float("inf")
    while current in came_from:
        current = came_from[current]
        current.make_path()
        g_path_length += 1

        for emitter in emitters:
            current_dist = euclidean_distance(current.get_pos(), emitter.get_pos())
            if current_dist < g_nearest_emitter_distance:
                g_nearest_emitter_distance = current_dist
        draw()


# --- A* ALGORITHM ---
def algorithm(draw_func, grid, start, end):
    global last_g_score, last_lowest_cost
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = distance(start.get_pos(), end.get_pos())
    open_set_hash = {start}
    current_lowest = 0

    global g_iterations, g_diff_time
    g_iterations = 0
    initial_time = datetime.datetime.now()
    g_diff_time = initial_time - initial_time

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)
        current_lowest = g_score[current]

        if current == end:
            last_g_score = g_score.copy()
            last_lowest_cost = g_score[end]
            reconstruct_path(came_from, end, lambda: draw_func(g_score, current_lowest))
            end.make_end()
            return True

        for neighbor in current.neighbors:
            base_move_cost = math.sqrt(2) if abs(neighbor.row - current.row) == 1 and abs(neighbor.col - current.col) == 1 else 1
            
            # Calculate penalty from emitters
            emitter_penalty = neighbor.get_penalty_from_emitters()
            total_move_cost = base_move_cost * (1 + emitter_penalty)

            temp_g_score = g_score[current] + total_move_cost

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + distance(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        g_diff_time = datetime.datetime.now() - initial_time
        draw_func(g_score, current_lowest)
        if current != start:
            current.make_closed()

        g_iterations += 1

    last_g_score = g_score.copy()
    last_lowest_cost = current_lowest
    return False


# --- CREATE RING OF WEIGHTED SPOTS AROUND BLOCKED ONES ---
def surround_blocked_with_weighted(grid, radius=2):
    rows = len(grid)
    for row in range(rows):
        for col in range(rows):
            if grid[row][col].is_blocked():
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < rows:
                            if grid[nr][nc].is_empty():  # Only change empty tiles
                                grid[nr][nc].make_weighted()


# --- GRID CREATION AND DRAWING ---
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            grid[i].append(Spot(i, j, gap, rows))
    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw_costs(win, grid, g_score):
    """Draw g_score costs on explored cells"""
    for row in grid:
        for spot in row:
            if spot in g_score and g_score[spot] != float("inf") and (spot.is_closed() or spot.is_open() or spot.is_path()):
                cost_text = small_font.render(f"{g_score[spot]:.1f}", True, BLACK)
                text_rect = cost_text.get_rect(center=(spot.x + spot.width//2, spot.y + spot.width//2))
                win.blit(cost_text, text_rect)

def draw_lowest_cost(win, g_score, current_lowest):
    """Draw lowest cost in top right corner"""
    cost_text = font.render(f"Lowest Cost: {current_lowest:.1f}", True, BLACK)
    win.blit(cost_text, (WIDTH - 200, 10))


def draw_algorithm_mod(win):
    if IS_MODIFIED:
        text = font.render("A* modified", True, BLACK)
    else:
        text = font.render("A* original", True, BLACK)
    win.blit(text, (10, 10))

def draw_stats(win):
    y = 10 + 30
    text = font.render(f"Iterations: {g_iterations}", True, BLACK)
    win.blit(text, (10, y))
    y += 30

    text = font.render(f"Time: {g_diff_time}", True, BLACK)
    win.blit(text, (10, y))
    y += 30

    if g_path_length is not None:
        text = font.render(f"Path length: {g_path_length}", True, BLACK)
        win.blit(text, (10, y))
        y += 30

    if g_nearest_emitter_distance is not None:
        text = font.render(f"Distance to nearest: {g_nearest_emitter_distance}", True, BLACK)
        win.blit(text, (10, y))
        y += 30


def draw(win, grid, rows, width, g_score=None, current_lowest=float("inf")):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    if g_score:
        # draw_costs(win, grid, g_score)
        draw_lowest_cost(win, g_score, current_lowest)
    draw_algorithm_mod(win)
    draw_stats(win)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


# --- MAIN LOOP ---
def main(win, width):
    global last_g_score, last_lowest_cost
    ROWS = 50
    grid = make_grid(ROWS, width)
    start = None
    end = None
    run = True
    started = False

    while run:
        # Draw with persistent costs if available
        if last_g_score and not started:
            draw(win, grid, ROWS, width, last_g_score, last_lowest_cost)
        else:
            draw(win, grid, ROWS, width)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[row]):
                    continue
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[1] or (event.type == pygame.KEYDOWN and event.key == pygame.K_f):  # MIDDLE CLICK - BLOCKED ZONE
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[row]):
                    continue
                spot = grid[row][col]
                if spot != start and spot != end:
                    spot.make_blocked()
                    emitters.add(spot)
                    update_grid_weights(grid)

            elif pygame.mouse.get_pressed()[2]:  # RIGHT CLICK - ERASE
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[row]):
                    continue
                spot = grid[row][col]
                spot.reset_all()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
                if spot in emitters:
                    emitters.remove(spot)
                    update_grid_weights(grid)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    clear_grid(start, end, grid)
                    surround_blocked_with_weighted(grid, radius=2)
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    started = True
                    algorithm(lambda g_score, current_lowest: draw(win, grid, ROWS, width, g_score, current_lowest), grid, start, end)
                    started = False

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    last_g_score = None
                    last_lowest_cost = float("inf")
                    grid = make_grid(ROWS, width)
                    emitters.clear()
                    global g_path_length, g_nearest_emitter_distance
                    g_path_length = None
                    g_nearest_emitter_distance = None

                if event.key == pygame.K_m:
                    global IS_MODIFIED
                    IS_MODIFIED = not IS_MODIFIED

    pygame.quit()


if __name__ == "__main__":
    main(WIN, WIDTH)
