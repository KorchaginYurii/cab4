import numpy as np

from core.global_planner import AStarPlanner
from core.energy import EnergySystem
from core.config import ACTIONS, DIRECTIONS, MOVE_COST, TURN_COST, CUT_COST
from core.sector_manager import SectorManager
from core.sector_coverage import SectorCoveragePlanner
from core.energy_predictor import EnergyPredictor
from core.world_memory import WorldMemory
from core.frontier_manager import FrontierManager
from core.config import ACTIONS
from core.team_blackboard import TeamBlackboard

class HybridAgent:
    def __init__(self, local_agent=None, robot_id="robot_1", blackboard=None):
        self.local_agent = local_agent
        self.robot_id = robot_id
        self.blackboard = blackboard or TeamBlackboard()

        if self.local_agent is None:
            print("⚠️ HybridAgent without local RL agent")
        else:
            print("✅ HybridAgent using RL local_agent")
        self.sectors = SectorManager(sector_h=5, sector_w=5)
        self.planner = AStarPlanner()
        self.coverage = SectorCoveragePlanner()
        self.mode = "COLLECT"
        self.goal = None
        self.path = []
        self.energy_predictor = EnergyPredictor(reserve=5.0)
        self.last_sector = None
        self.sector_switches = 0
        self.memory = WorldMemory()
        self.frontiers = FrontierManager()
        self.replan_interval = 8
        self.replan_cooldown = 0
        self.prev_pos = None



    def reset(self):

        self.mode = "COLLECT"
        self.goal = None
        self.path = []
        self.coverage.reset()
        self.last_sector = None
        self.sector_switches = 0
        self.memory = WorldMemory()
        self.replan_interval = 8
        self.replan_cooldown = 0
        self.prev_pos = None

    def nearest_cabbage(self, env):
        cabbages = np.argwhere(env.grid == 1)

        if len(cabbages) == 0:
            return None

        x, y = env.pos

        dists = np.abs(cabbages - np.array([x, y])).sum(axis=1)
        nearest = cabbages[np.argmin(dists)]

        return tuple(nearest)

    def choose_goal(self, env):
        remaining = np.sum(env.grid == 1)

        # =====================================================
        # 1. ВСЁ СОБРАНО → ФИНИШ НА БАЗУ
        # =====================================================
        if remaining == 0:
            self.mode = "RETURN_FINISH"
            return env.start_pos

        # =====================================================
        # 2. ЕСЛИ АГЕНТ НА БАЗЕ — ЗАРЯДИЛСЯ И ПРОДОЛЖАЕТ
        # =====================================================
        if env.pos == env.start_pos:
            env.energy_system.recharge()

        # =====================================================
        # 3. ПРОВЕРЯЕМ, МОЖЕМ ЛИ ВООБЩЕ ВЕРНУТЬСЯ ДОМОЙ
        # =====================================================
        path_home = self.planner.find_path_oriented(
            env,
            env.pos,
            env.start_pos,
            memory=self.memory,
            robot_id=self.robot_id,
            blackboard=self.blackboard
        )

        if path_home is None:
            self.mode = "STUCK"
            return None

        home_cost = self.estimate_path_cost(env, path_home)

        if not env.energy_system.can_reach(home_cost, reserve=5.0):
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        # =====================================================
        # 4. ВЫБИРАЕМ СЕКТОР
        # =====================================================
        sector = self.sectors.choose_sector_energy_aware(
            env,
            self.memory,
            self.planner,
            self.energy_predictor,
            robot_id=self.robot_id,
            blackboard=self.blackboard
        )

        if sector is not None:
            claimed = self.blackboard.claim_sector(self.robot_id, sector)

            if not claimed:
                self.sectors.current_sector = None
                sector = None
        # =====================================================
        # 5. ЕСЛИ СЕКТОР НАЙДЕН — ПРОВЕРЯЕМ ЭНЕРГИЮ НА СЕКТОР
        # =====================================================
        if sector is not None:
            ok_energy, required_energy = self.energy_predictor.has_energy_to_finish_sector(
                env,
                self.planner,
                self.sectors,
                sector,
                memory=self.memory
            )

            self.last_required_energy = required_energy

            if not ok_energy:
                self.mode = "RETURN_CHARGE"
                return env.start_pos

            cabbage = self.coverage.get_next_target(
                self.memory,
                env,
                self.sectors,
                sector
            )

        else:
            # ВАЖНО:
            # sector is None НЕ означает "ехать заряжаться"
            # это может значить, что секторный выбор не сработал
            self.last_required_energy = 0.0
            cabbage = None

        # =====================================================
        # 6. FALLBACK: ЕСЛИ В СЕКТОРЕ НЕТ ЦЕЛИ — ИЩЕМ КАПУСТУ НА ВСЕЙ КАРТЕ
        # =====================================================
        if cabbage is None:
            cabbage = self.nearest_cabbage(env)
        # =========================================
        #
        # ========================================
        if cabbage is None:
            frontier = self.frontiers.choose_frontier(
                env,
                self.memory,
                self.planner,
                self.energy_predictor
            )

            if frontier is not None:
                self.mode = "EXPLORE"
                return frontier

        # =====================================================
        # 7. ЕСЛИ КАПУСТЫ НЕТ ВООБЩЕ — ФИНИШ
        # =====================================================
        if cabbage is None:
            self.mode = "RETURN_FINISH"
            return env.start_pos

        # ===========================
        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        if self.goal in dynamic_positions:
            self.path = None
            self.goal = None
            self.replan_cooldown = 0

        # =====================================================
        # 8. СТРОИМ ПУТЬ ДО КАПУСТЫ
        # =====================================================
        path_to_cabbage = self.planner.find_path_oriented(
            env,
            env.pos,
            cabbage,
            memory=self.memory,
            unknown_policy="avoid",
            robot_id=self.robot_id,
            blackboard=self.blackboard
        )

        if path_to_cabbage is None:
            # не надо сразу RETURN_CHARGE
            # возможно именно эта капуста недоступна — fallback уже был,
            # но если A* не нашёл путь, тогда возвращаемся безопасно
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        to_cabbage_cost = self.estimate_path_cost(
            env,
            path_to_cabbage
        )

        # =====================================================
        # 9. ПРОВЕРЯЕМ ВОЗВРАТ ПОСЛЕ ЭТОЙ КАПУСТЫ
        # =====================================================
        path_back = self.planner.find_path_oriented(
            env,
            cabbage,
            env.start_pos,
            start_heading=env.heading,
            memory=self.memory,
            unknown_policy="avoid",
            robot_id=self.robot_id,
            blackboard=self.blackboard
        )

        if path_back is None:
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        back_cost = self.estimate_path_cost(
            env,
            path_back
        )

        total_mission_cost = to_cabbage_cost + back_cost

        if not env.energy_system.can_reach(total_mission_cost, reserve=5.0):
            self.mode = "RETURN_CHARGE"
            return env.start_pos

        # =====================================================
        # 10. ВСЁ ОК → СОБИРАЕМ
        # =====================================================
        self.mode = "COLLECT"
        return cabbage

    def action_from_path(self, env, path):
        if path is None or len(path) < 2:
            return 0

        x, y = env.pos
        nx, ny = path[1]

        dx = nx - x
        dy = ny - y

        for i, (adx, ady) in enumerate(ACTIONS):
            if (adx, ady) == (dx, dy):
                return i

        return 0

    def act(self, env, temp=0):

        self.blackboard.update_robot(self.robot_id, env.pos)
        # =====================================================
        # 1. UPDATE MEMORY
        # =====================================================
        if self.memory.map is None:
            self.memory.reset(env.grid.shape)

        # для partial observable режима
        self.memory.observe_local(env, radius=3)
        # если нужен полный доступ:
        # self.memory.observe_full(env)

        # ===== TEAM MEMORY =====
        self.blackboard.update_shared_memory(self.memory)
        self.memory = self.blackboard.sync_memory(self.memory)

        # синхронизация пути
        self.sync_path_with_position(env)

        # =====================================================
        # 2. INIT REPLAN STATE
        # =====================================================
        if not hasattr(self, "replan_cooldown"):
            self.replan_cooldown = 0

        if not hasattr(self, "replan_interval"):
            self.replan_interval = 5

        if not hasattr(self, "prev_pos"):
            self.prev_pos = None

        remaining = np.sum(env.grid == 1)

        env.allow_start_access = (
                self.mode == "RETURN_CHARGE"
                or (self.mode == "RETURN_FINISH" and remaining == 0)
        )



        # =====================================================
        # 3. DECIDE IF REPLAN IS NEEDED
        # =====================================================

        need_replan = False

        if self.goal is None:
            need_replan = True

        if self.path is None or len(self.path) < 2:
            need_replan = True

        blocked = self.path_is_blocked(env)

        if blocked:
            need_replan = True
            self.replan_cooldown = 0

        if self.replan_cooldown <= 0:
            need_replan = True

        # если цель уже достигнута
        if self.goal is not None and env.pos == self.goal:
            need_replan = True

        # =====================================================
        # 4. REPLAN ONLY IF NEEDED
        # =====================================================
        if need_replan:
            self.goal = self.choose_goal(env)

            unknown_policy = "allow"

            if self.mode == "EXPLORE":
                unknown_policy = "explore"

            elif self.mode in ["RETURN_CHARGE", "RETURN_FINISH"]:
                unknown_policy = "avoid"

            elif self.mode == "COLLECT":
                unknown_policy = "allow"

            if self.goal is not None:
                self.path = self.planner.find_path_oriented(
                    env,
                    env.pos,
                    self.goal,
                    memory=self.memory,
                    unknown_policy=unknown_policy,
                    robot_id=self.robot_id,
                    blackboard=self.blackboard
                )
            else:
                self.path = None

            self.replan_cooldown = self.replan_interval

        else:
            self.replan_cooldown -= 1

        # =====================================================
        # 5. ANTI BACK-AND-FORTH PROTECTION
        # =====================================================
        if (
                self.prev_pos is not None
                and self.path is not None
                and len(self.path) >= 2
                and self.path[1] == self.prev_pos
                and not self.path_is_blocked(env)
        ):
            # путь ведёт сразу назад без необходимости — принудительный replan
            self.path = None
            self.replan_cooldown = 0

            self.goal = self.choose_goal(env)

            if self.goal is not None:
                self.path = self.planner.find_path_oriented(
                    env,
                    env.pos,
                    self.goal,
                    memory=self.memory,
                    unknown_policy="allow",
                    robot_id=self.robot_id,
                    blackboard=self.blackboard
                )

        # =====================================================
        # 6. ACTION FROM PATH
        # =====================================================
        if self.path is not None and len(self.path) >= 2:
            action = self.action_from_path(env, self.path)
        else:
            action = self.safe_detour_action(env)

        # =====================================================
        # 7. SECTOR SWITCH METRICS
        # =====================================================
        sector = self.sectors.current_sector

        if not hasattr(self, "last_sector"):
            self.last_sector = None

        if not hasattr(self, "sector_switches"):
            self.sector_switches = 0

        if sector is not None and sector != self.last_sector:
            self.sector_switches += 1
            self.last_sector = sector

        # =====================================================
        # 8. METRICS
        # =====================================================
        total = np.sum(env.initial_grid == 1)
        remaining = np.sum(env.grid == 1)
        collected = total - remaining

        energy_used = getattr(env, "energy_used", 0.0)
        energy_per_cabbage = energy_used / max(1, collected)

        if hasattr(env, "visit_count"):
            overlap_cells = int(np.sum(env.visit_count > 1))
            visited_cells = int(np.sum(env.visit_count > 0))
            overlap_rate = overlap_cells / max(1, visited_cells)
        else:
            overlap_cells = 0
            overlap_rate = 0.0

        total_turns = getattr(env, "total_turns", 0)

        required_energy = getattr(self, "last_required_energy", 0.0)
        energy_margin = env.energy_system.energy - required_energy

        frontiers = self.memory.frontier_cells()

        # =====================================================
        # 9. DEBUG
        # =====================================================
        debug = {
            "mode": self.mode,
            "goal": self.goal,
            "path": self.path,

            "sector": self.sectors.current_sector,
            "sector_h": self.sectors.sector_h,
            "sector_w": self.sectors.sector_w,
            "sector_switches": self.sector_switches,

            "coverage_target": self.goal,

            "frontiers": frontiers,
            "frontier_count": len(frontiers),
            "frontier_clusters": getattr(self.frontiers, "frontier_clusters", []),
            "frontier_target": getattr(self.frontiers, "selected_frontier", None),

            "memory_map": self.memory.map.copy(),
            "memory_seen": self.memory.seen.copy(),
            "memory_coverage": self.memory.coverage_rate(),
            "memory_overlap": self.memory.visited_overlap_rate(),

            "energy": env.energy_system.energy,
            "max_energy": env.energy_system.max_energy,
            "energy_used": energy_used,
            "energy_per_cabbage": energy_per_cabbage,
            "required_energy": required_energy,
            "energy_margin": energy_margin,

            "knife_on": env.knife_on,
            "heading": env.heading,

            "total_turns": total_turns,
            "overlap_cells": overlap_cells,
            "overlap_rate": overlap_rate,

            "replan_cooldown": self.replan_cooldown,
            "need_replan": need_replan,

            "robot_id": self.robot_id,
            "claimed_sectors": dict(self.blackboard.claimed_sectors),

            "robot_id": self.robot_id,
            "claimed_sectors": dict(self.blackboard.claimed_sectors),

            "robot_positions": dict(self.blackboard.robot_positions)
        }

        # =====================================================
        # 10. UPDATE PREV POS
        # =====================================================
        self.prev_pos = env.pos

        return action, debug
    def estimate_path_cost(self, env, path, start_heading=None):
        if path is None or len(path) < 2:
            return 0.0

        heading = env.heading if start_heading is None else start_heading
        cost = 0.0

        for i in range(len(path) - 1):
            x, y = path[i]
            nx, ny = path[i + 1]

            dx = nx - x
            dy = ny - y

            # движение
            cost += MOVE_COST

            # поворот
            target_heading = heading

            for h, (hx, hy) in enumerate(DIRECTIONS):
                if (dx, dy) == (hx, hy):
                    target_heading = h
                    break

            diff = abs(target_heading - heading)
            diff = min(diff, 4 - diff)

            cost += diff * TURN_COST
            heading = target_heading

            # нож только если следующая клетка с капустой
            if env.grid[nx][ny] == 1:
                cost += CUT_COST

        return cost

    def path_is_blocked(self, env):
        if self.path is None or len(self.path) < 2:
            return False

        next_pos = self.path[1]

        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        if next_pos in dynamic_positions:
            return True

        if next_pos in env.obstacles:
            return True

        return False

    def sync_path_with_position(self, env):
        if self.path is None:
            return

        if env.pos in self.path:
            idx = self.path.index(env.pos)
            self.path = self.path[idx:]
        else:
            self.path = None

    def safe_detour_action(self, env):
        dynamic_positions = (
            env.dynamic_obstacles.positions()
            if hasattr(env, "dynamic_obstacles")
            else set()
        )

        x, y = env.pos
        h, w = env.grid.shape

        for a, (dx, dy) in enumerate(ACTIONS):
            nx = max(0, min(h - 1, x + dx))
            ny = max(0, min(w - 1, y + dy))
            np_ = (nx, ny)

            if np_ in env.obstacles:
                continue
            if np_ in dynamic_positions:
                continue
            if np_ == getattr(self, "prev_pos", None):
                continue

            return a

        return 0