# Emergent Physics Simulator - Rule-Based Grid Engine
# Industry-Level Modular Version with Multiple Physical Behaviors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import torch
import torch.nn as nn

# --- Configuration ---
GRID_SIZE = 100        # Grid dimensions (100x100)
STEPS = 200            # Number of simulation steps
DECAY = 0.95           # Diffusion decay factor
THRESHOLD = 50         # Explosion threshold value
CHAOS_FACTOR = 0.1     # Random chaos multiplier
MUTATION_RATE = 0.1    # Mutation rate for rule evolution

# --- Initialize Grid ---
def initialize_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    center = GRID_SIZE // 2
    grid[center, center] = 100  # Hotspot
    return grid

# --- Rule 1: Simple Diffusion ---
def rule_diffusion(grid):
    new_grid = grid.copy()
    for i in range(1, GRID_SIZE-1):
        for j in range(1, GRID_SIZE-1):
            avg = (grid[i+1,j] + grid[i-1,j] + grid[i,j+1] + grid[i,j-1]) / 4
            new_grid[i,j] = avg * DECAY
    return new_grid

# --- Rule 2: Reaction-Diffusion (Gray-Scott model approximation) ---
def rule_reaction_diffusion(U, V):
    Du, Dv = 0.16, 0.08
    F, k = 0.035, 0.065
    Lu = (np.roll(U, 1, 0) + np.roll(U, -1, 0) + np.roll(U, 1, 1) + np.roll(U, -1, 1) - 4 * U)
    Lv = (np.roll(V, 1, 0) + np.roll(V, -1, 0) + np.roll(V, 1, 1) + np.roll(V, -1, 1) - 4 * V)
    UVV = U * V * V
    U += (Du * Lu - UVV + F * (1 - U))
    V += (Dv * Lv + UVV - (F + k) * V)
    return U, V

# --- Rule 3: Threshold-Based Explosions ---
def rule_explosive(grid):
    explosion_value = 200
    new_grid = grid.copy()
    for i in range(1, GRID_SIZE-1):
        for j in range(1, GRID_SIZE-1):
            if grid[i, j] > THRESHOLD:
                new_grid[i-1:i+2, j-1:j+2] += explosion_value
                new_grid[i, j] = 0
    return new_grid

# --- Rule 4: Chaotic Energy Loop ---
def rule_chaos(grid):
    new_grid = grid.copy()
    for i in range(1, GRID_SIZE-1):
        for j in range(1, GRID_SIZE-1):
            chaos = random.uniform(-CHAOS_FACTOR, CHAOS_FACTOR)
            avg = (grid[i+1,j] + grid[i-1,j] + grid[i,j+1] + grid[i,j-1]) / 4
            new_grid[i,j] = (avg + chaos) * DECAY
    return new_grid

# --- Mutation System for Rule Evolution ---
def mutate_rule_params(params):
    return {k: v + random.uniform(-0.1, 0.1) * v if random.random() < MUTATION_RATE else v for k, v in params.items()}

# --- AI Predictor (Example: Predict next state from current) ---
class SimplePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# --- Visualization Setup ---
def run_simulation():
    grid = initialize_grid()
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='inferno', interpolation='nearest')

    # Optional: Reaction-Diffusion setup
    U = np.ones((GRID_SIZE, GRID_SIZE))
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    U[GRID_SIZE//2-10:GRID_SIZE//2+10, GRID_SIZE//2-10:GRID_SIZE//2+10] = 0.5
    V[GRID_SIZE//2-10:GRID_SIZE//2+10, GRID_SIZE//2-10:GRID_SIZE//2+10] = 0.25

    def update(frame):
        nonlocal grid, U, V
        # Choose any rule to apply here:
        grid = rule_diffusion(grid)
        grid = rule_explosive(grid)
        grid = rule_chaos(grid)
        U, V = rule_reaction_diffusion(U, V)
        im.set_array(grid + V * 100)  # Combine for visualization
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=STEPS, blit=True, interval=50)
    plt.title("Multi-Rule Emergent Physics Simulator")
    plt.show()

# --- Entry Point ---
if __name__ == '__main__':
    run_simulation()
