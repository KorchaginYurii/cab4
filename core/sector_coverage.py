from core.world_memory import CABBAGE
import numpy as np
from core.config import DIRECTION_BIAS_WEIGHT, BACKTRACK_PENALTY

class SectorCoveragePlanner:
    def __init__(self):
        self.cached_sector = None
        self.cached_targets = []
        self.target_index = 0

    def reset(self):
        self.cached_sector = None
        self.cached_targets = []
        self.target_index = 0

    def build_targets(self, memory, sector_manager, sector_id):
        x1, x2, y1, y2 = sector_manager.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        targets = []

        for x in range(x1, x2):
            cols = range(y1, y2)

            if (x - x1) % 2 == 1:
                cols = reversed(list(cols))

            for y in cols:
                if memory.map[x][y] == CABBAGE:
                    targets.append((x, y))

        return targets

    def get_next_target(self, memory, env, sector_manager, sector_id):
        if sector_id is None:
            return None

        if sector_id != self.cached_sector:
            self.cached_sector = sector_id
            self.cached_targets = self.build_targets(
                memory,
                sector_manager,
                sector_id
            )
            self.target_index = 0

        while self.target_index < len(self.cached_targets):
            x, y = self.cached_targets[self.target_index]

            if memory.map[x][y] == CABBAGE:
                return (x, y)

            self.target_index += 1

        self.cached_targets = self.build_targets(
            memory,
            sector_manager,
            sector_id
        )
        self.target_index = 0

        if len(self.cached_targets) == 0:
            return None

        return self.cached_targets[0]

    def get_next_target_directional(
            self,
            memory,
            env,
            sector_manager,
            sector_id,
            prev_pos=None
    ):
        if sector_id is None:
            return None

        # если сектор сменился — перестроить targets
        if sector_id != self.cached_sector:
            self.cached_sector = sector_id
            self.cached_targets = self.build_targets(
                memory,
                sector_manager,
                sector_id
            )
            self.target_index = 0

        # только ещё существующая известная капуста
        candidates = [
            p for p in self.cached_targets
            if memory.map[p[0], p[1]] == 1
        ]

        if len(candidates) == 0:
            self.cached_targets = self.build_targets(
                memory,
                sector_manager,
                sector_id
            )
            candidates = [
                p for p in self.cached_targets
                if memory.map[p[0], p[1]] == 1
            ]

        if len(candidates) == 0:
            return None

        x, y = env.pos

        # направление текущего движения
        if prev_pos is not None:
            px, py = prev_pos
            move_vec = np.array([x - px, y - py], dtype=np.float32)
        else:
            move_vec = np.array([0.0, 0.0], dtype=np.float32)

        best = None
        best_score = 1e18

        for tx, ty in candidates:
            dist = abs(tx - x) + abs(ty - y)

            target_vec = np.array([tx - x, ty - y], dtype=np.float32)

            direction_penalty = 0.0

            if np.linalg.norm(move_vec) > 0 and np.linalg.norm(target_vec) > 0:
                dot = np.dot(move_vec, target_vec)

                # если цель "назад" относительно текущего движения
                if dot < 0:
                    direction_penalty += BACKTRACK_PENALTY

                # если цель не по текущей оси движения
                cos_sim = dot / (
                        np.linalg.norm(move_vec) * np.linalg.norm(target_vec) + 1e-6
                )

                direction_penalty += DIRECTION_BIAS_WEIGHT * (1.0 - cos_sim)

            score = dist + direction_penalty

            if score < best_score:
                best_score = score
                best = (tx, ty)

        return best

