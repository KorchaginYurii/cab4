import torch

# ===== ENV =====
MAP_H = 10
MAP_W = 10
VISION_SIZE = 7
#MAX_STEPS = 200

# ===== ACTIONS =====
ACTIONS = [
    (0, 1),   # right
    (0, -1),  # left
    (1, 0),   # down
    (-1, 0)   # up
]

NUM_ACTIONS = len(ACTIONS)

# ===== ENERGY =====
MOVE_COST = 0.1
TURN_COST = 0.1
TURN_COST_WEIGHT = 0.3
CUT_COST = 0.5
ENERGY_RESERVE = 5.0

# ===== SECTORS =====
SECTOR_H = 5
SECTOR_W = 5
SECTOR_SCORE_CABBAGE_WEIGHT = 10.0
SECTOR_SCORE_TRAVEL_WEIGHT = 1.0
SECTOR_SCORE_ENERGY_WEIGHT = 1.0

# ===== COVERAGE =====
COVERAGE_SWEEP_MODE = "boustrophedon"

# ===== REWARD =====
STEP_PENALTY = -0.05
OBSTACLE_PENALTY = -1.0
START_BLOCK_PENALTY = -0.5
ENERGY_REWARD_WEIGHT = 1.0
COLLECT_REWARD = 10.0
RETURN_REWARD = 100.0
FINAL_BONUS = 50.0

UNKNOWN_CELL_COST = 0.3
UNKNOWN_COST_AVOID = 1.0
UNKNOWN_COST_ALLOW = 0.3
UNKNOWN_COST_EXPLORE = 0.05


# ===== MISSION =====
RETURN_LIMIT_MULT = 2

DIRECTIONS = [
    (-1, 0),  # UP
    (0, 1),   # RIGHT
    (1, 0),   # DOWN
    (0, -1),  # LEFT
]

DIR_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]

# ===== RL =====
GAMMA = 0.99
TAU = 0.005

# ===== TRAIN =====
EPISODES = 1000
BATCH_SIZE = 64
MEMORY_SIZE = 50000

# ===== STATE =====
STATE_CHANNELS = 16

# ===== DEVICE =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")