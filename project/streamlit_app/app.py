# app.py - Streamlit UI for Interactive Physics Simulation

import sys
import os
# Add the parent directory to the Python path to find emergent_simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from emergent_simulator import (
    initialize_grid, rule_diffusion, rule_explosive, rule_chaos,
    rule_reaction_diffusion, GRID_SIZE
)

def run_physics_app():
    """Main function to run the physics simulation app"""
    st.title("🌌 Emergent Physics Simulator")

    # --- Sliders for user input ---
    decay = st.slider("Diffusion Decay", 0.8, 1.0, 0.95, 0.01)
    threshold = st.slider("Explosion Threshold", 10, 200, 50, 10)
    chaos_factor = st.slider("Chaos Factor", 0.0, 1.0, 0.1, 0.05)
    steps = st.slider("Simulation Steps", 50, 500, 100, 10)

    # --- Initialize grids ---
    grid = initialize_grid()
    U = np.ones((GRID_SIZE, GRID_SIZE))
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    # Add initial pattern for reaction-diffusion
    center = GRID_SIZE // 2
    U[center-5:center+5, center-5:center+5] = 0.5
    V[center-5:center+5, center-5:center+5] = 0.25

    frames = []

    # --- Run simulation ---
    progress_bar = st.progress(0)
    for i in range(steps):
        # Apply diffusion with custom decay
        grid_copy = grid.copy()
        for x in range(1, GRID_SIZE-1):
            for y in range(1, GRID_SIZE-1):
                avg = (grid[x+1,y] + grid[x-1,y] + grid[x,y+1] + grid[x,y-1]) / 4
                grid_copy[x,y] = avg * decay
        grid = grid_copy
        
        # Apply explosive rule with custom threshold
        explosion_value = 200
        new_grid = grid.copy()
        for x in range(1, GRID_SIZE-1):
            for y in range(1, GRID_SIZE-1):
                if grid[x, y] > threshold:
                    new_grid[x-1:x+2, y-1:y+2] += explosion_value
                    new_grid[x, y] = 0
        grid = new_grid
        
        # Apply chaos with custom factor
        chaos_grid = grid.copy()
        for x in range(1, GRID_SIZE-1):
            for y in range(1, GRID_SIZE-1):
                chaos = np.random.uniform(-chaos_factor, chaos_factor)
                avg = (grid[x+1,y] + grid[x-1,y] + grid[x,y+1] + grid[x,y-1]) / 4
                chaos_grid[x,y] = (avg + chaos) * decay
        grid = chaos_grid
        
        # Apply reaction-diffusion
        U, V = rule_reaction_diffusion(U, V)
        
        # Combine for visualization
        combined = grid + V * 100
        frames.append(combined.copy())
        
        # Update progress
        progress_bar.progress((i + 1) / steps)

    # --- Display results ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Final State")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(frames[-1], cmap='inferno', interpolation='nearest')
        ax1.set_title("Combined Simulation Result")
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        st.subheader("Reaction-Diffusion Pattern")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(V, cmap='viridis', interpolation='nearest')
        ax2.set_title("Reaction-Diffusion Component")
        ax2.axis('off')
        st.pyplot(fig2)

    # --- Animation option ---
    if st.checkbox("Show Animation"):
        st.subheader("Evolution Over Time")
        frame_slider = st.slider("Frame", 0, len(frames)-1, len(frames)-1)
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        ax3.imshow(frames[frame_slider], cmap='inferno', interpolation='nearest')
        ax3.set_title(f"Step {frame_slider + 1}/{len(frames)}")
        ax3.axis('off')
        st.pyplot(fig3)

    st.markdown("🎮 **Try adjusting the sliders above and rerunning to see different emergent behaviors!**")
    st.markdown("- **Diffusion Decay**: Controls how fast energy spreads and decays")
    st.markdown("- **Explosion Threshold**: Energy level that triggers explosive events")
    st.markdown("- **Chaos Factor**: Amount of randomness in the system")
    st.markdown("- **Simulation Steps**: How long the simulation runs")

# For standalone running
if __name__ == "__main__":
    run_physics_app()