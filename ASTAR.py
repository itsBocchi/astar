import pygame
import math
from queue import PriorityQueue

# --- WINDOW SETTINGS ---
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Pathfinding with Diagonal Movement & Weighted Zones")

# --- COLORS ---
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
DARK_BLUE = (0, 0, 128)      # Blocked zone
DARK_GREEN = (0, 128, 0)     # Weighted ring zone

# --- SPOT CLASS ---
class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == DARK_BLUE

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK or self.color == RED

    def is_weighted(self):
        return self.color == DARK_GREEN

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = DARK_BLUE

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_blocked(self):
        self.color = RED

    def make_weighted(self):
        self.color = DARK_GREEN

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

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
                if not grid[r][c].is_barrier():
                    self.neighbors.append(grid[r][c])

    def __lt__(self, other):
        return False

def clear_grid(start, end, grid):
    for row in grid:
        for spot in row:
            if spot != start and spot != end and not spot.is_weighted() and not spot.is_barrier():
                spot.reset()

# --- HEURISTIC FUNCTION (OCTILE DISTANCE) ---
def h(p1, p2):
    """Heuristic that accounts for diagonal movement"""
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    D = 1
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


# --- RECONSTRUCT PATH ---
def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()


# --- A* ALGORITHM ---
def algorithm(draw, grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            return True

        for neighbor in current.neighbors:
            move_cost = math.sqrt(2) if abs(neighbor.row - current.row) == 1 and abs(neighbor.col - current.col) == 1 else 1

            # Add penalty for weighted tiles
            if neighbor.is_weighted():
                move_cost *= 3

            temp_g_score = g_score[current] + move_cost

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        if current != start:
            current.make_closed()

    return False


# --- CREATE RING OF WEIGHTED SPOTS AROUND BLOCKED ONES ---
def surround_blocked_with_weighted(grid, radius=2):
    rows = len(grid)
    for row in range(rows):
        for col in range(rows):
            if grid[row][col].color == RED:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < rows:
                            if grid[nr][nc].color == WHITE:  # Only change empty tiles
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


def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


# --- MAIN LOOP ---
def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)
    start = None
    end = None
    run = True
    started = False

    while run:
        draw(win, grid, ROWS, width)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if row >= len(grid) or col >= len(grid[row]):
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
                if row >= len(grid) or col >= len(grid[row]):
                    continue
                spot = grid[row][col]
                if spot != start and spot != end:
                    spot.make_blocked()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT CLICK - ERASE
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if row >= len(grid) or col >= len(grid[row]):
                    continue
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    clear_grid(start, end, grid)
                    surround_blocked_with_weighted(grid, radius=2)
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    started = True
                    algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    started = False

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


if __name__ == "__main__":
    main(WIN, WIDTH)
