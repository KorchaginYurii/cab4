import random
from core.config import ACTIONS, GRID_SIZE


class DynamicObstacle:
    def __init__(self, pos):
        self.pos = pos


class DynamicObstacleManager:
    def __init__(self, count=2, move_prob=0.3):
        self.count = count
        self.move_prob = move_prob
        self.obstacles = []

    def reset(self, env):
        self.obstacles = []

        free = [
            (i, j)
            for i in range(GRID_SIZE)
            for j in range(GRID_SIZE)
            if env.grid[i][j] == 0
            and (i, j) != env.pos
            and (i, j) != env.start_pos
        ]

        random.shuffle(free)

        for pos in free[:self.count]:
            self.obstacles.append(DynamicObstacle(pos))

    def positions(self):
        return {o.pos for o in self.obstacles}

    def step(self, env):
        occupied = self.positions()

        for obj in self.obstacles:
            if random.random() > self.move_prob:
                continue

            x, y = obj.pos
            random.shuffle(ACTIONS)

            for dx, dy in ACTIONS:
                nx = max(0, min(GRID_SIZE - 1, x + dx))
                ny = max(0, min(GRID_SIZE - 1, y + dy))
                np = (nx, ny)

                if np in env.obstacles:
                    continue

                if np in occupied:
                    continue

                if np == env.pos or np == env.start_pos:
                    continue

                if env.grid[nx][ny] == 1:
                    continue

                occupied.remove(obj.pos)
                obj.pos = np
                occupied.add(np)
                break