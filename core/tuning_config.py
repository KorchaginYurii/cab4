from core.config import *

DEFAULT_TUNABLES = {
    "TURN_COST_WEIGHT": TURN_COST_WEIGHT,

    "UNKNOWN_COST_AVOID": UNKNOWN_COST_AVOID,
    "UNKNOWN_COST_ALLOW": UNKNOWN_COST_ALLOW,
    "UNKNOWN_COST_EXPLORE": UNKNOWN_COST_EXPLORE,

    "DYNAMIC_NEAR_COST": DYNAMIC_NEAR_COST,
    "DYNAMIC_MID_COST": DYNAMIC_MID_COST,
    "DYNAMIC_FAR_COST": DYNAMIC_FAR_COST,

    "REPLAN_INTERVAL": REPLAN_INTERVAL,
}

class RuntimeConfig:
    def __init__(self):
        self.values = {}
        self.values = DEFAULT_TUNABLES.copy()

    def set(self, key, value):
        self.values[key] = value

    def get(self, key, default):
        return self.values.get(key, default)

    def update(self, d):
        self.values.update(d)

    def as_dict(self):
        return dict(self.values)


runtime_config = RuntimeConfig()