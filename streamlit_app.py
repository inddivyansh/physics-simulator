# streamlit_app.py - Main entry point for Streamlit Community Cloud

import sys
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set page config first
st.set_page_config(
    page_title="🌌 Physics Simulator", 
    page_icon="🌌", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project directory to path
project_dir = os.path.join(os.path.dirname(__file__), 'project')
sys.path.insert(0, project_dir)

# Import the simulator components
try:
    from emergent_simulator import (
        initialize_grid, rule_diffusion, rule_explosive, rule_chaos,
        rule_reaction_diffusion, GRID_SIZE
    )
except ImportError as e:
    st.error(f"Failed to import simulator: {e}")
    st.stop()

def run_physics_app():
    """Main function to run the physics simulation app"""
    st.title("🌌 Emergent Physics Simulator")
    
    st.markdown("""
    Welcome to the **Emergent Physics Simulator**! This interactive app demonstrates complex emergent behaviors 
    using simple physics rules. Adjust the parameters below and watch how small changes can create dramatically 
    different patterns.
    """)

    # --- Sidebar Controls ---
    st.sidebar.header("🎮 Simulation Controls")
    
    decay = st.sidebar.slider("Diffusion Decay", 0.8, 1.0, 0.95, 0.01, 
                              help="Controls how fast energy spreads and decays")
    threshold = st.sidebar.slider("Explosion Threshold", 10, 200, 50, 10,
                                  help="Energy level that triggers explosive events") 
    chaos_factor = st.sidebar.slider("Chaos Factor", 0.0, 1.0, 0.1, 0.05,
                                     help="Amount of randomness in the system")
    steps = st.sidebar.slider("Simulation Steps", 50, 500, 100, 10,
                              help="How long the simulation runs")

    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        run_simulation(decay, threshold, chaos_factor, steps)
    else:
        st.info("👈 Adjust the parameters in the sidebar and click 'Run Simulation' to begin!")
        
        # Show example with default parameters
        st.subheader("Example Output")
        st.markdown("*This is what the simulation looks like with default parameters:*")
        run_simulation(0.95, 50, 0.1, 50, show_progress=False)

def run_simulation(decay, threshold, chaos_factor, steps, show_progress=True):
    """Run the physics simulation with given parameters"""
    
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
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i in range(steps):
        if show_progress:
            status_text.text(f'Step {i+1}/{steps}')
            
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
        if show_progress:
            progress_bar.progress((i + 1) / steps)
    
    if show_progress:
        status_text.text('Simulation complete!')

    # --- Display results ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔥 Final State")
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        ax1.imshow(frames[-1], cmap='inferno', interpolation='nearest')
        ax1.set_title("Combined Simulation Result")
        ax1.axis('off')
        st.pyplot(fig1)

    with col2:
        st.subheader("🌊 Reaction-Diffusion Pattern")
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.imshow(V, cmap='viridis', interpolation='nearest')
        ax2.set_title("Reaction-Diffusion Component")
        ax2.axis('off')
        st.pyplot(fig2)

    # --- Animation option ---
    if len(frames) > 1:
        with st.expander("🎬 Show Animation Timeline"):
            frame_slider = st.slider("Frame", 0, len(frames)-1, len(frames)-1, key="frame_slider")
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            ax3.imshow(frames[frame_slider], cmap='inferno', interpolation='nearest')
            ax3.set_title(f"Step {frame_slider + 1}/{len(frames)}")
            ax3.axis('off')
            st.pyplot(fig3)

    # --- Information ---
    with st.expander("ℹ️ About the Simulation"):
        st.markdown("""
        This simulation combines multiple physics rules:
        
        - **🌊 Diffusion**: Energy spreads across the grid and decays over time
        - **💥 Explosions**: High-energy cells trigger explosive events that spread energy
        - **🎲 Chaos**: Random perturbations create unpredictable behaviors
        - **🔄 Reaction-Diffusion**: Chemical-like patterns emerge from local interactions
        
        **Parameters:**
        - **Diffusion Decay**: Lower values = faster energy loss
        - **Explosion Threshold**: Lower values = more frequent explosions
        - **Chaos Factor**: Higher values = more randomness
        - **Simulation Steps**: More steps = longer evolution
        """)

if __name__ == "__main__":
    run_physics_app()
