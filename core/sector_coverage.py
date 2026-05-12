from core.world_memory import CABBAGE

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

