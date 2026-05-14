import time
import pygame

from env.cabbage_env import CabbageEnv
from agents.cabbage_agent import CabbageAgent
from agents.hybrid_agent import HybridAgent
from ui.pygame_renderer import Renderer

from core.checkpoint import CheckpointManager
from core.replay_recorder import ReplayRecorder
from core.config import MAP_H, MAP_W

recorder = ReplayRecorder()
env = CabbageEnv(MAP_H, MAP_W)
env.reset()

# ======================================
# RL AGENT
# ======================================
local_agent = CabbageAgent()

# ======================================
# LOAD CHECKPOINT
# ======================================
ckpt = CheckpointManager()

start_ep, best = ckpt.load_checkpoint(local_agent)

# ======================================
# HYBRID WRAPPER
# ======================================
agent = HybridAgent(local_agent=local_agent)
#agent = HybridAgent()
agent.reset()

renderer = Renderer()

while True:
    for event in pygame.event.get():
        renderer.handle_mouse(event)

        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    action, debug = agent.act(env)
    reward, done = env.step(action)
    recorder.record(env, debug)
    renderer.draw(env, debug)

    time.sleep(0.1)

    if done:
        recorder.save()
        print("DONE")
        break