# Enhanced Emergent Physics Simulator - Advanced Reality-Based Engine
# Multi-Physics System with Real-World Phenomena

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.signal import convolve2d
import colorsys

# --- Enhanced Configuration ---
GRID_SIZE = 100
DEFAULT_STEPS = 200
PHYSICS_DT = 0.1  # Time step for physics calculations

# Physical Constants (scaled for simulation)
GRAVITY = 9.81 * 0.01
VISCOSITY = 0.1
THERMAL_CONDUCTIVITY = 0.5
MAGNETIC_PERMEABILITY = 0.3
CHEMICAL_REACTION_RATE = 0.05

# --- Multi-Layer Grid System ---
class PhysicsGrid:
    """Advanced multi-layer physics grid supporting multiple phenomena"""
    
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.layers = {
            'temperature': np.zeros((size, size)),
            'pressure': np.ones((size, size)),
            'velocity_x': np.zeros((size, size)),
            'velocity_y': np.zeros((size, size)),
            'density': np.ones((size, size)),
            'chemical_a': np.zeros((size, size)),
            'chemical_b': np.zeros((size, size)),
            'magnetic_field': np.zeros((size, size)),
            'life_cells': np.zeros((size, size)),
            'energy': np.zeros((size, size)),
            'turbulence': np.zeros((size, size))
        }
        self.initialize_systems()
    
    def initialize_systems(self):
        """Initialize different physical systems"""
        center = self.size // 2
        
        # Temperature hotspot
        self.layers['temperature'][center-5:center+5, center-5:center+5] = 100
        
        # Chemical gradients
        self.layers['chemical_a'][center-10:center+10, center-10:center+10] = 0.5
        self.layers['chemical_b'][center-3:center+3, center-3:center+3] = 0.8
        
        # Magnetic field pattern
        x, y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        self.layers['magnetic_field'] = np.sin(x * 0.1) * np.cos(y * 0.1)
        
        # Random life cells (Conway's Game of Life)
        self.layers['life_cells'] = np.random.choice([0, 1], (self.size, self.size), p=[0.8, 0.2])

# --- Enhanced Physics Rules ---

def rule_fluid_dynamics(grid, dt=PHYSICS_DT):
    """Navier-Stokes inspired fluid dynamics"""
    vx = grid.layers['velocity_x']
    vy = grid.layers['velocity_y']
    p = grid.layers['pressure']
    rho = grid.layers['density']
    
    # Gradient calculations
    grad_p_x = np.gradient(p, axis=1)
    grad_p_y = np.gradient(p, axis=0)
    
    # Laplacian for viscosity
    laplacian_vx = ndimage.laplace(vx)
    laplacian_vy = ndimage.laplace(vy)
    
    # Update velocities (simplified Navier-Stokes)
    vx_new = vx - dt * (grad_p_x / rho + GRAVITY) + VISCOSITY * dt * laplacian_vx
    vy_new = vy - dt * (grad_p_y / rho + GRAVITY) + VISCOSITY * dt * laplacian_vy
    
    # Apply boundary conditions
    vx_new[0, :] = vx_new[-1, :] = 0
    vx_new[:, 0] = vx_new[:, -1] = 0
    vy_new[0, :] = vy_new[-1, :] = 0
    vy_new[:, 0] = vy_new[:, -1] = 0
    
    grid.layers['velocity_x'] = vx_new
    grid.layers['velocity_y'] = vy_new
    
    return grid

def rule_heat_transfer(grid, dt=PHYSICS_DT):
    """Realistic heat diffusion with convection"""
    T = grid.layers['temperature']
    vx = grid.layers['velocity_x']
    vy = grid.layers['velocity_y']
    
    # Heat diffusion (Fourier's law)
    laplacian_T = ndimage.laplace(T)
    diffusion = THERMAL_CONDUCTIVITY * dt * laplacian_T
    
    # Convection (simplified)
    grad_T_x = np.gradient(T, axis=1)
    grad_T_y = np.gradient(T, axis=0)
    convection = -dt * (vx * grad_T_x + vy * grad_T_y)
    
    T_new = T + diffusion + convection
    
    # Cooling to environment
    T_new *= 0.999
    
    grid.layers['temperature'] = np.clip(T_new, 0, 200)
    return grid

def rule_turbulence_generation(grid, chaos_factor=0.1):
    """Generate realistic turbulence patterns"""
    vx = grid.layers['velocity_x']
    vy = grid.layers['velocity_y']
    
    # Calculate vorticity (curl of velocity field)
    dvx_dy = np.gradient(vx, axis=0)
    dvy_dx = np.gradient(vy, axis=1)
    vorticity = dvy_dx - dvx_dy
    
    # Turbulence intensity based on velocity gradients
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    turbulence = np.abs(vorticity) * velocity_magnitude * chaos_factor
    
    # Add random perturbations
    noise = np.random.normal(0, 0.1, grid.layers['turbulence'].shape)
    turbulence += noise * chaos_factor
    
    grid.layers['turbulence'] = turbulence
    return grid

def rule_multi_chemical_reaction(grid, dt=PHYSICS_DT):
    """Advanced multi-chemical reaction-diffusion system"""
    A = grid.layers['chemical_a']
    B = grid.layers['chemical_b']
    
    # Multiple reaction-diffusion parameters for different patterns
    Da, Db = 0.16, 0.08  # Diffusion coefficients
    f1, k1 = 0.035, 0.065  # Feed and kill rates (spots)
    f2, k2 = 0.055, 0.062  # Feed and kill rates (stripes)
    
    # Mixing parameter based on position
    x, y = np.meshgrid(np.arange(grid.size), np.arange(grid.size))
    mix = 0.5 + 0.3 * np.sin(x * 0.05) * np.cos(y * 0.05)
    
    f = f1 * mix + f2 * (1 - mix)
    k = k1 * mix + k2 * (1 - mix)
    
    # Laplacian for diffusion
    La = ndimage.laplace(A)
    Lb = ndimage.laplace(B)
    
    # Reaction terms
    ABB = A * B * B
    
    # Update equations
    dA_dt = Da * La - ABB + f * (1 - A)
    dB_dt = Db * Lb + ABB - (f + k) * B
    
    A_new = A + dt * dA_dt
    B_new = B + dt * dB_dt
    
    grid.layers['chemical_a'] = np.clip(A_new, 0, 1)
    grid.layers['chemical_b'] = np.clip(B_new, 0, 1)
    
    return grid

def rule_conways_life(grid):
    """Conway's Game of Life for emergent life-like behavior"""
    life = grid.layers['life_cells']
    
    # Count neighbors
    kernel = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
    
    neighbors = convolve2d(life, kernel, mode='same', boundary='wrap')
    
    # Apply Conway's rules
    life_new = np.zeros_like(life)
    life_new[(life == 1) & ((neighbors == 2) | (neighbors == 3))] = 1  # Survive
    life_new[(life == 0) & (neighbors == 3)] = 1  # Birth
    
    grid.layers['life_cells'] = life_new
    return grid

def rule_magnetic_field_interaction(grid, dt=PHYSICS_DT):
    """Magnetic field effects on charged particles"""
    B = grid.layers['magnetic_field']
    vx = grid.layers['velocity_x']
    vy = grid.layers['velocity_y']
    rho = grid.layers['density']
    
    # Lorentz force (simplified)
    charge_density = rho - 1  # Assume neutral density = 1
    
    # Magnetic force affects velocity
    force_x = charge_density * vy * B * MAGNETIC_PERMEABILITY
    force_y = -charge_density * vx * B * MAGNETIC_PERMEABILITY
    
    grid.layers['velocity_x'] += dt * force_x
    grid.layers['velocity_y'] += dt * force_y
    
    # Update magnetic field based on current
    current_x = charge_density * vx
    current_y = charge_density * vy
    current_magnitude = np.sqrt(current_x**2 + current_y**2)
    
    dB_dt = 0.01 * ndimage.laplace(B) + 0.001 * current_magnitude
    grid.layers['magnetic_field'] += dt * dB_dt
    
    return grid

def rule_pressure_waves(grid, dt=PHYSICS_DT):
    """Sound/pressure wave propagation"""
    p = grid.layers['pressure']
    vx = grid.layers['velocity_x']
    vy = grid.layers['velocity_y']
    rho = grid.layers['density']
    
    # Wave equation (simplified)
    sound_speed = 10.0
    laplacian_p = ndimage.laplace(p)
    
    # Pressure waves
    dp_dt = -sound_speed**2 * (np.gradient(rho * vx, axis=1) + np.gradient(rho * vy, axis=0))
    
    p_new = p + dt * dp_dt + 0.1 * dt * laplacian_p
    
    grid.layers['pressure'] = p_new
    return grid

def rule_energy_cascade(grid, dt=PHYSICS_DT):
    """Energy cascade and conservation"""
    energy = grid.layers['energy']
    T = grid.layers['temperature']
    vx = grid.layers['velocity_x']
    vy = grid.layers['velocity_y']
    turbulence = grid.layers['turbulence']
    
    # Kinetic energy
    kinetic = 0.5 * (vx**2 + vy**2)
    
    # Thermal energy
    thermal = T * 0.01
    
    # Turbulent energy
    turbulent = turbulence * 0.1
    
    # Total energy with conservation
    total_energy = kinetic + thermal + turbulent
    
    # Energy dissipation
    dissipation = 0.01 * total_energy
    
    energy_new = total_energy - dt * dissipation
    grid.layers['energy'] = energy_new
    
    return grid

# --- Advanced Initialization Patterns ---

def initialize_vortex_pattern(grid):
    """Initialize with vortex velocity field"""
    center_x, center_y = grid.size // 2, grid.size // 2
    x, y = np.meshgrid(np.arange(grid.size), np.arange(grid.size))
    
    # Distance from center
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)
    
    # Vortex velocity field
    theta = np.arctan2(dy, dx)
    v_magnitude = 5 * np.exp(-r / 20)
    
    grid.layers['velocity_x'] = -v_magnitude * np.sin(theta)
    grid.layers['velocity_y'] = v_magnitude * np.cos(theta)
    
    return grid

def initialize_wave_interference(grid):
    """Initialize with interfering wave patterns"""
    x, y = np.meshgrid(np.arange(grid.size), np.arange(grid.size))
    
    # Multiple wave sources
    wave1 = np.sin(0.3 * x + 0.2 * y)
    wave2 = np.cos(0.2 * x - 0.3 * y)
    wave3 = np.sin(0.25 * (x - grid.size//3)**2 + 0.25 * (y - grid.size//3)**2)
    
    interference = wave1 + wave2 + wave3
    
    grid.layers['pressure'] = 1 + 0.1 * interference
    grid.layers['temperature'] = 50 + 20 * interference
    
    return grid

def initialize_ecosystem(grid):
    """Initialize ecosystem-like patterns"""
    # Create patches of life
    for _ in range(5):
        x, y = np.random.randint(10, grid.size-10, 2)
        size = np.random.randint(5, 15)
        grid.layers['life_cells'][x-size:x+size, y-size:y+size] = 1
    
    # Chemical gradients for nutrients
    x, y = np.meshgrid(np.arange(grid.size), np.arange(grid.size))
    grid.layers['chemical_a'] = 0.3 + 0.2 * np.sin(x * 0.1) * np.cos(y * 0.1)
    
    return grid

# --- Enhanced Visualization Functions ---

def create_composite_visualization(grid, layer_weights=None):
    """Create a composite visualization of multiple layers"""
    if layer_weights is None:
        layer_weights = {
            'temperature': 0.3,
            'chemical_b': 0.3,
            'velocity_magnitude': 0.2,
            'life_cells': 0.1,
            'turbulence': 0.1
        }
    
    # Calculate velocity magnitude
    vx = grid.layers['velocity_x']
    vy = grid.layers['velocity_y']
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    
    # Normalize all layers to 0-1
    layers_normalized = {}
    for key in layer_weights.keys():
        if key == 'velocity_magnitude':
            data = velocity_magnitude
        else:
            data = grid.layers[key]
        
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            layers_normalized[key] = (data - data_min) / (data_max - data_min)
        else:
            layers_normalized[key] = np.zeros_like(data)
    
    # Create RGB composite
    composite = np.zeros((grid.size, grid.size, 3))
    
    # Map layers to color channels
    composite[:, :, 0] = (layers_normalized['temperature'] * layer_weights['temperature'] + 
                         layers_normalized['turbulence'] * layer_weights['turbulence'])
    
    composite[:, :, 1] = (layers_normalized['chemical_b'] * layer_weights['chemical_b'] + 
                         layers_normalized['velocity_magnitude'] * layer_weights['velocity_magnitude'])
    
    composite[:, :, 2] = (layers_normalized['life_cells'] * layer_weights['life_cells'] + 
                         layers_normalized['chemical_b'] * layer_weights['chemical_b'] * 0.5)
    
    # Normalize final composite
    composite = np.clip(composite, 0, 1)
    
    return composite

def generate_time_lapse_frames(grid, rules, steps, sample_every=10):
    """Generate frames for time-lapse visualization"""
    frames = []
    frame_info = []
    
    for step in range(steps):
        # Apply all rules
        for rule in rules:
            grid = rule(grid)
        
        # Sample frames
        if step % sample_every == 0:
            composite = create_composite_visualization(grid)
            frames.append(composite.copy())
            
            # Calculate metrics for this frame
            energy_total = np.sum(grid.layers['energy'])
            temp_avg = np.mean(grid.layers['temperature'])
            life_count = np.sum(grid.layers['life_cells'])
            
            frame_info.append({
                'step': step,
                'energy': energy_total,
                'temperature': temp_avg,
                'life_cells': life_count
            })
    
    return frames, frame_info

# --- Main Enhanced Simulator Class ---

class EnhancedPhysicsSimulator:
    """Advanced physics simulator with multiple rule systems"""
    
    def __init__(self, grid_size=GRID_SIZE):
        self.grid = PhysicsGrid(grid_size)
        self.available_rules = {
            'fluid_dynamics': rule_fluid_dynamics,
            'heat_transfer': rule_heat_transfer,
            'turbulence': rule_turbulence_generation,
            'chemical_reaction': rule_multi_chemical_reaction,
            'conways_life': rule_conways_life,
            'magnetic_field': rule_magnetic_field_interaction,
            'pressure_waves': rule_pressure_waves,
            'energy_cascade': rule_energy_cascade
        }
        self.active_rules = []
        self.history = []
    
    def add_rule(self, rule_name):
        """Add a physics rule to the active simulation"""
        if rule_name in self.available_rules:
            self.active_rules.append(self.available_rules[rule_name])
    
    def initialize_pattern(self, pattern_type):
        """Initialize with different starting patterns"""
        if pattern_type == 'vortex':
            self.grid = initialize_vortex_pattern(self.grid)
        elif pattern_type == 'waves':
            self.grid = initialize_wave_interference(self.grid)
        elif pattern_type == 'ecosystem':
            self.grid = initialize_ecosystem(self.grid)
    
    def step(self):
        """Execute one simulation step"""
        for rule in self.active_rules:
            self.grid = rule(self.grid)
        
        # Store history for analysis
        self.history.append({
            'energy': np.sum(self.grid.layers['energy']),
            'temperature': np.mean(self.grid.layers['temperature']),
            'life_count': np.sum(self.grid.layers['life_cells'])
        })
    
    def run_simulation(self, steps):
        """Run complete simulation"""
        frames = []
        for _ in range(steps):
            self.step()
            if len(frames) < 50:  # Limit frames for memory
                frames.append(create_composite_visualization(self.grid))
        return frames
    
    def get_layer_data(self, layer_name):
        """Get data from specific physics layer"""
        return self.grid.layers.get(layer_name, np.zeros((self.grid.size, self.grid.size)))

# --- Export the enhanced functions for the Streamlit app ---

def initialize_enhanced_grid():
    """Initialize the enhanced physics grid"""
    return PhysicsGrid()

# Make the enhanced simulator available
__all__ = [
    'PhysicsGrid', 'EnhancedPhysicsSimulator', 'initialize_enhanced_grid',
    'rule_fluid_dynamics', 'rule_heat_transfer', 'rule_turbulence_generation',
    'rule_multi_chemical_reaction', 'rule_conways_life', 'rule_magnetic_field_interaction',
    'rule_pressure_waves', 'rule_energy_cascade', 'create_composite_visualization',
    'generate_time_lapse_frames', 'GRID_SIZE'
]
