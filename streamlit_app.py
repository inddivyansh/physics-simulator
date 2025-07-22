# streamlit_app.py - Enhanced Physics Simulator with Advanced Features

import sys
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page config first
st.set_page_config(
    page_title="🌌 Advanced Physics Simulator", 
    page_icon="🌌", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project directory to path
project_dir = os.path.join(os.path.dirname(__file__), 'project')
sys.path.insert(0, project_dir)

# Import the enhanced simulator components
try:
    from enhanced_physics import (
        PhysicsGrid, EnhancedPhysicsSimulator, create_composite_visualization,
        generate_time_lapse_frames, GRID_SIZE
    )
    from emergent_simulator import initialize_grid, rule_diffusion
    ENHANCED_MODE = True
except ImportError as e:
    st.warning(f"Enhanced physics not available, using basic mode: {e}")
    from emergent_simulator import (
        initialize_grid, rule_diffusion, rule_explosive, rule_chaos,
        rule_reaction_diffusion, GRID_SIZE
    )
    ENHANCED_MODE = False

def create_custom_colormap():
    """Create beautiful custom colormaps for different physics layers"""
    # Temperature colormap (blue -> red -> white)
    temp_colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#FFFFFF']
    temp_cmap = LinearSegmentedColormap.from_list('temperature', temp_colors)
    
    # Velocity colormap (black -> blue -> green -> yellow)
    vel_colors = ['#000000', '#000080', '#0080FF', '#00FF80', '#80FF00', '#FFFF00']
    vel_cmap = LinearSegmentedColormap.from_list('velocity', vel_colors)
    
    # Life colormap (dark -> green -> bright green)
    life_colors = ['#000000', '#004000', '#008000', '#00FF00', '#80FF80']
    life_cmap = LinearSegmentedColormap.from_list('life', life_colors)
    
    return temp_cmap, vel_cmap, life_cmap

def run_physics_app():
    """Main function to run the enhanced physics simulation app"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .physics-rule {
        background: #e8f4fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<div class="main-header"><h1 style="color: white; text-align: center;">🌌 Advanced Physics Simulator</h1><p style="color: white; text-align: center;">Real-world physics phenomena with emergent behaviors</p></div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Physics Controls")
    
    # Simulation mode selection
    if ENHANCED_MODE:
        mode = st.sidebar.selectbox(
            "Simulation Mode",
            ["Enhanced Multi-Physics", "Basic Physics", "Comparison Mode"],
            help="Choose between different physics simulation modes"
        )
    else:
        mode = "Basic Physics"
        st.sidebar.info("Enhanced mode not available - using basic physics")
    
    # Physics rules selection
    if mode == "Enhanced Multi-Physics" and ENHANCED_MODE:
        st.sidebar.subheader("🔬 Physics Rules")
        
        available_rules = {
            "Fluid Dynamics": "fluid_dynamics",
            "Heat Transfer": "heat_transfer", 
            "Turbulence": "turbulence",
            "Chemical Reactions": "chemical_reaction",
            "Life Evolution": "conways_life",
            "Magnetic Fields": "magnetic_field",
            "Pressure Waves": "pressure_waves",
            "Energy Cascade": "energy_cascade"
        }
        
        selected_rules = []
        for rule_name, rule_key in available_rules.items():
            if st.sidebar.checkbox(rule_name, value=True, help=f"Enable {rule_name.lower()} simulation"):
                selected_rules.append(rule_key)
        
        # Initial pattern selection
        pattern_type = st.sidebar.selectbox(
            "Initial Pattern",
            ["Standard", "Vortex", "Waves", "Ecosystem"],
            help="Choose the initial conditions for the simulation"
        )
        
        # Advanced parameters
        with st.sidebar.expander("🔧 Advanced Parameters"):
            chaos_intensity = st.slider("Chaos Intensity", 0.0, 1.0, 0.1, 0.01)
            thermal_conductivity = st.slider("Thermal Conductivity", 0.1, 2.0, 0.5, 0.1)
            magnetic_strength = st.slider("Magnetic Field Strength", 0.0, 1.0, 0.3, 0.1)
            reaction_rate = st.slider("Chemical Reaction Rate", 0.01, 0.2, 0.05, 0.01)
            
    else:
        # Basic physics parameters
        st.sidebar.subheader("🎮 Basic Controls")
        decay = st.sidebar.slider("Diffusion Decay", 0.8, 1.0, 0.95, 0.01)
        threshold = st.sidebar.slider("Explosion Threshold", 10, 200, 50, 10)
        chaos_intensity = st.sidebar.slider("Chaos Factor", 0.0, 1.0, 0.1, 0.05)
        selected_rules = ["basic"]
        pattern_type = "Standard"
    
    # Simulation parameters
    st.sidebar.subheader("⚙️ Simulation Settings")
    steps = st.sidebar.slider("Simulation Steps", 50, 1000, 200, 50)
    time_lapse_speed = st.sidebar.slider("Time-lapse Speed", 1, 20, 5, 1)
    show_intermediate = st.sidebar.checkbox("Show Intermediate Steps", value=True)
    
    # Visualization options
    st.sidebar.subheader("🎨 Visualization")
    viz_layers = st.sidebar.multiselect(
        "Display Layers",
        ["Temperature", "Velocity", "Pressure", "Chemical A", "Chemical B", "Life Cells", "Turbulence", "Energy"],
        default=["Temperature", "Chemical B", "Velocity"]
    )
    
    use_3d_view = st.sidebar.checkbox("3D Surface View", value=False)
    show_vectors = st.sidebar.checkbox("Show Vector Fields", value=False)
    
    # Run simulation
    if st.sidebar.button("🚀 Run Advanced Simulation", type="primary"):
        run_enhanced_simulation(
            mode, selected_rules, pattern_type, steps, time_lapse_speed,
            show_intermediate, viz_layers, use_3d_view, show_vectors,
            chaos_intensity, thermal_conductivity if 'thermal_conductivity' in locals() else 0.5,
            magnetic_strength if 'magnetic_strength' in locals() else 0.3,
            reaction_rate if 'reaction_rate' in locals() else 0.05,
            decay if 'decay' in locals() else 0.95,
            threshold if 'threshold' in locals() else 50
        )
    else:
        # Show example/demo
        st.info("👈 Configure your physics simulation in the sidebar and click 'Run Advanced Simulation' to begin!")
        show_demo_information()

def run_enhanced_simulation(mode, selected_rules, pattern_type, steps, time_lapse_speed,
                           show_intermediate, viz_layers, use_3d_view, show_vectors,
                           chaos_intensity, thermal_conductivity, magnetic_strength,
                           reaction_rate, decay, threshold):
    """Run the enhanced physics simulation with all features"""
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.subheader("🏃‍♂️ Simulation Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_placeholder = st.empty()
    
    # Initialize simulation
    start_time = time.time()
    
    if mode == "Enhanced Multi-Physics" and ENHANCED_MODE:
        simulator = EnhancedPhysicsSimulator(GRID_SIZE)
        
        # Add selected rules
        for rule in selected_rules:
            simulator.add_rule(rule)
        
        # Initialize pattern
        if pattern_type.lower() != "standard":
            simulator.initialize_pattern(pattern_type.lower())
            
        # Run simulation with progress updates
        frames = []
        intermediate_frames = []
        sample_every = max(1, steps // 20)  # Sample 20 frames maximum
        
        for step in range(steps):
            simulator.step()
            
            # Update progress
            progress = (step + 1) / steps
            progress_bar.progress(progress)
            status_text.text(f'Step {step + 1}/{steps} - {progress:.1%} Complete')
            
            # Sample frames for visualization
            if step % sample_every == 0:
                composite = create_composite_visualization(simulator.grid)
                frames.append(composite.copy())
                
                if show_intermediate and len(intermediate_frames) < 5:
                    intermediate_frames.append((step, composite.copy()))
            
            # Update metrics every 10 steps
            if step % 10 == 0:
                with metrics_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Energy", f"{np.sum(simulator.grid.layers['energy']):.2f}")
                    with col2:
                        st.metric("Avg Temperature", f"{np.mean(simulator.grid.layers['temperature']):.1f}")
                    with col3:
                        st.metric("Life Cells", f"{int(np.sum(simulator.grid.layers['life_cells']))}")
                    with col4:
                        st.metric("Max Velocity", f"{np.max(np.sqrt(simulator.grid.layers['velocity_x']**2 + simulator.grid.layers['velocity_y']**2)):.2f}")
        
        final_grid = simulator.grid
        
    else:
        # Basic physics simulation
        grid = initialize_grid()
        frames = []
        intermediate_frames = []
        sample_every = max(1, steps // 20)
        
        for step in range(steps):
            # Apply basic rules
            grid = rule_diffusion(grid)
            if np.random.random() < 0.1:  # Occasional explosions
                grid = rule_explosive(grid)
            
            progress = (step + 1) / steps
            progress_bar.progress(progress)
            status_text.text(f'Step {step + 1}/{steps} - {progress:.1%} Complete')
            
            if step % sample_every == 0:
                frames.append(grid.copy())
                if show_intermediate and len(intermediate_frames) < 5:
                    intermediate_frames.append((step, grid.copy()))
        
        final_grid = grid
    
    # Simulation complete
    elapsed_time = time.time() - start_time
    status_text.text(f'✅ Simulation Complete! ({elapsed_time:.1f}s)')
    
    # Display results
    display_simulation_results(final_grid, frames, intermediate_frames, viz_layers, 
                             use_3d_view, show_vectors, mode, time_lapse_speed)

def display_simulation_results(final_grid, frames, intermediate_frames, viz_layers,
                              use_3d_view, show_vectors, mode, time_lapse_speed):
    """Display comprehensive simulation results"""
    
    st.subheader("📊 Simulation Results")
    
    # Main visualization area
    if mode == "Enhanced Multi-Physics" and ENHANCED_MODE:
        display_enhanced_results(final_grid, frames, viz_layers, use_3d_view, show_vectors)
    else:
        display_basic_results(final_grid, frames)
    
    # Intermediate steps visualization
    if intermediate_frames:
        st.subheader("🎬 Evolution Timeline")
        
        # Create tabs for different time points
        tab_names = [f"Step {step}" for step, _ in intermediate_frames]
        tabs = st.tabs(tab_names)
        
        for i, (step, frame) in enumerate(intermediate_frames):
            with tabs[i]:
                fig, ax = plt.subplots(figsize=(8, 8))
                if mode == "Enhanced Multi-Physics":
                    ax.imshow(frame, interpolation='nearest')
                else:
                    ax.imshow(frame, cmap='inferno', interpolation='nearest')
                ax.set_title(f"Simulation State at Step {step}")
                ax.axis('off')
                st.pyplot(fig)
    
    # Time-lapse animation
    if len(frames) > 1:
        st.subheader("⏩ Time-lapse Animation")
        
        # Animation controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            frame_index = st.slider("Frame", 0, len(frames)-1, 0, 1, key="timelapse_slider")
        
        with col2:
            auto_play = st.checkbox("Auto-play", value=False)
            if auto_play:
                # Auto-advance frames
                if 'frame_counter' not in st.session_state:
                    st.session_state.frame_counter = 0
                
                time.sleep(1.0 / time_lapse_speed)
                st.session_state.frame_counter = (st.session_state.frame_counter + 1) % len(frames)
                frame_index = st.session_state.frame_counter
                st.rerun()
        
        # Display selected frame
        fig, ax = plt.subplots(figsize=(10, 8))
        if mode == "Enhanced Multi-Physics":
            ax.imshow(frames[frame_index], interpolation='nearest')
        else:
            ax.imshow(frames[frame_index], cmap='inferno', interpolation='nearest')
        ax.set_title(f"Frame {frame_index + 1}/{len(frames)}")
        ax.axis('off')
        st.pyplot(fig)
        
        # Frame statistics
        st.caption(f"🎯 Frame {frame_index + 1} of {len(frames)} | Progress: {(frame_index + 1) / len(frames) * 100:.1f}%")

def display_enhanced_results(final_grid, frames, viz_layers, use_3d_view, show_vectors):
    """Display results from enhanced physics simulation"""
    
    # Create layer visualizations
    layer_mapping = {
        "Temperature": "temperature",
        "Velocity": "velocity_magnitude",
        "Pressure": "pressure", 
        "Chemical A": "chemical_a",
        "Chemical B": "chemical_b",
        "Life Cells": "life_cells",
        "Turbulence": "turbulence",
        "Energy": "energy"
    }
    
    # Multi-layer visualization
    num_layers = len(viz_layers)
    if num_layers > 0:
        cols = st.columns(min(num_layers, 3))
        
        for i, layer_name in enumerate(viz_layers):
            col_idx = i % 3
            layer_key = layer_mapping.get(layer_name, "temperature")
            
            with cols[col_idx]:
                if layer_key == "velocity_magnitude":
                    # Calculate velocity magnitude
                    vx = final_grid.layers['velocity_x']
                    vy = final_grid.layers['velocity_y']
                    data = np.sqrt(vx**2 + vy**2)
                else:
                    data = final_grid.layers[layer_key]
                
                if use_3d_view:
                    # 3D surface plot
                    fig = go.Figure(data=[go.Surface(z=data, colorscale='Viridis')])
                    fig.update_layout(
                        title=f"{layer_name} - 3D View",
                        scene=dict(
                            xaxis_title="X",
                            yaxis_title="Y", 
                            zaxis_title=layer_name
                        ),
                        width=400,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # 2D heatmap
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(data, cmap='viridis', interpolation='nearest')
                    ax.set_title(f"{layer_name}")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                    st.pyplot(fig)
                
                # Show statistics
                st.caption(f"Min: {data.min():.2f} | Max: {data.max():.2f} | Mean: {data.mean():.2f}")
    
    # Vector field visualization
    if show_vectors:
        st.subheader("🌊 Vector Field Visualization")
        
        vx = final_grid.layers['velocity_x']
        vy = final_grid.layers['velocity_y']
        
        # Downsample for better visualization
        step = 5
        x_indices = np.arange(0, GRID_SIZE, step)
        y_indices = np.arange(0, GRID_SIZE, step)
        X, Y = np.meshgrid(x_indices, y_indices)
        
        U = vx[::step, ::step]
        V = vy[::step, ::step]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Background heatmap
        ax.imshow(final_grid.layers['temperature'], cmap='hot', alpha=0.7)
        
        # Vector field
        ax.quiver(X, Y, U, V, scale=50, alpha=0.8, color='white', width=0.003)
        
        ax.set_title("Velocity Vector Field over Temperature")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        st.pyplot(fig)

def display_basic_results(final_grid, frames):
    """Display results from basic physics simulation"""
    
    # Final state
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Final State")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(final_grid, cmap='inferno', interpolation='nearest')
        ax.set_title("Final Simulation State")
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Statistics")
        st.metric("Total Energy", f"{np.sum(final_grid):.2f}")
        st.metric("Max Temperature", f"{np.max(final_grid):.2f}")
        st.metric("Average Value", f"{np.mean(final_grid):.2f}")
        st.metric("Standard Deviation", f"{np.std(final_grid):.2f}")

def show_demo_information():
    """Show information about the simulator when not running"""
    
    st.subheader("🔬 About This Advanced Physics Simulator")
    
    # Feature overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌊 **Fluid Dynamics**
        - Navier-Stokes equations
        - Viscosity and turbulence
        - Realistic flow patterns
        
        ### 🔥 **Heat Transfer**
        - Thermal diffusion
        - Convection effects
        - Temperature gradients
        
        ### ⚡ **Electromagnetic**
        - Magnetic field interactions
        - Charged particle dynamics
        - Lorentz forces
        
        ### 🧬 **Chemical Reactions**
        - Multi-species diffusion
        - Reaction-diffusion patterns
        - Concentration gradients
        """)
    
    with col2:
        st.markdown("""
        ### 🌀 **Turbulence**
        - Vorticity calculations
        - Energy cascade
        - Chaotic behavior
        
        ### 🔊 **Wave Propagation**
        - Pressure waves
        - Sound propagation
        - Wave interference
        
        ### 🦠 **Life Evolution**
        - Conway's Game of Life
        - Population dynamics
        - Emergence patterns
        
        ### ⚡ **Energy Systems**
        - Energy conservation
        - Dissipation effects
        - Multi-scale interactions
        """)
    
    # Physics principles
    st.subheader("📚 Physics Principles")
    
    st.markdown("""
    This simulator implements real-world physics equations:
    
    - **Heat Equation**: ∂T/∂t = α∇²T + convection terms
    - **Navier-Stokes**: ∂v/∂t + (v·∇)v = -∇p/ρ + ν∇²v + f
    - **Reaction-Diffusion**: ∂u/∂t = D∇²u + R(u,v)
    - **Wave Equation**: ∂²p/∂t² = c²∇²p
    - **Maxwell's Equations**: ∇×E = -∂B/∂t, ∇×B = μ₀J + μ₀ε₀∂E/∂t
    """)
    
    # Example visualization
    st.subheader("🎨 Example Patterns")
    
    # Create some example patterns
    x = np.linspace(0, 4*np.pi, 100)
    y = np.linspace(0, 4*np.pi, 100)
    X, Y = np.meshgrid(x, y)
    
    # Wave interference pattern
    wave_pattern = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X + Y)
    
    # Reaction-diffusion-like pattern
    rd_pattern = np.exp(-((X-2*np.pi)**2 + (Y-2*np.pi)**2)/4) * np.sin(3*X) * np.cos(3*Y)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(wave_pattern, cmap='viridis', interpolation='nearest')
        ax.set_title("Wave Interference Pattern")
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rd_pattern, cmap='plasma', interpolation='nearest')
        ax.set_title("Reaction-Diffusion Pattern")
        ax.axis('off')
        st.pyplot(fig)

if __name__ == "__main__":
    run_physics_app()
