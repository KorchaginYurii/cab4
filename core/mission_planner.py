import numpy as np


class MissionPlanner:

    def __init__(self):
        self.route = []
        self.route_index = 0
        self.cached_signature = None

    def reset(self):
        self.route = []
        self.route_index = 0
        self.cached_signature = None

    def sector_center(self, sector_manager, sector_id, memory):
        x1, x2, y1, y2 = sector_manager.get_sector_bounds(
            sector_id,
            memory.map.shape
        )

        return (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

    def build_signature(self, memory):
        return int(np.sum(memory.map == 1))

    def build_route(
        self,
        env,
        memory,
        sector_manager
    ):

        self.route = []
        self.route_index = 0

        current = env.pos

        sectors = []

        for sector_id in sector_manager.all_sector_ids(memory):

            cabbages = sector_manager.sector_cabbages(
                memory,
                sector_id
            )

            if cabbages <= 0:
                continue

            center = self.sector_center(
                sector_manager,
                sector_id,
                memory
            )

            sectors.append((sector_id, center))

        # greedy nearest-sector ordering
        while len(sectors) > 0:

            best_i = None
            best_d = 1e18

            for i, (sid, center) in enumerate(sectors):

                d = (
                    abs(center[0] - current[0])
                    + abs(center[1] - current[1])
                )

                if d < best_d:
                    best_d = d
                    best_i = i

            sid, center = sectors.pop(best_i)

            self.route.append(sid)
            current = center

        self.cached_signature = self.build_signature(memory)

    def current_sector(
        self,
        env,
        memory,
        sector_manager
    ):

        signature = self.build_signature(memory)

        if (
            len(self.route) == 0
            or signature != self.cached_signature
        ):
            self.build_route(
                env,
                memory,
                sector_manager
            )

        while self.route_index < len(self.route):

            sector_id = self.route[self.route_index]

            remaining = sector_manager.sector_cabbages(
                memory,
                sector_id
            )

            if remaining > 0:
                return sector_id

            self.route_index += 1

        return None