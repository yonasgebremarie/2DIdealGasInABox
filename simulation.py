"""
2D Ideal Gas Simulation

Simulates N particles in a 2D box with elastic wall collisions.
Tracks velocity distribution, temperature, and pressure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec


class Particle:
    """Represents a single particle with position and velocity."""
    
    def __init__(self, x, y, vx, vy, mass=1.0, radius=0.02):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass
        self.radius = radius
    
    def update_position(self, dt):
        """Update particle position based on velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def kinetic_energy(self):
        """Calculate kinetic energy of the particle."""
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)
    
    def speed(self):
        """Calculate speed (magnitude of velocity)."""
        return np.sqrt(self.vx**2 + self.vy**2)


class Box2D:
    """Represents a 2D box containing particles."""
    
    def __init__(self, width=1.0, height=1.0, gravity_x=0.0, gravity_y=0.0,
                 interaction_epsilon=0.0, interaction_sigma=0.04, interaction_cutoff=None):
        self.width = width
        self.height = height
        self.particles = []
        self.time = 0.0
        self.wall_collision_count = 0
        self.momentum_transfer = 0.0
        self.time_elapsed = 0.0
        self.gravity_x = gravity_x
        self.gravity_y = gravity_y
        self.interaction_epsilon = interaction_epsilon
        self.interaction_sigma = interaction_sigma
        self.interaction_cutoff = interaction_cutoff or (5 * interaction_sigma)
        self.potential_energy = 0.0
    
    def add_particle(self, particle):
        """Add a particle to the box."""
        self.particles.append(particle)
    
    def _compute_interaction_forces(self):
        """
        Compute pairwise forces and potential energy using a Lennard-Jones-like potential.
        Returns an array of forces for each particle.
        """
        n = len(self.particles)
        forces = np.zeros((n, 2))
        potential_energy = 0.0
        
        if self.interaction_epsilon <= 0.0 or n == 0:
            self.potential_energy = 0.0
            return forces
        
        sigma = self.interaction_sigma
        cutoff_sq = self.interaction_cutoff * self.interaction_cutoff
        
        for i in range(n):
            pi = self.particles[i]
            for j in range(i + 1, n):
                pj = self.particles[j]
                
                dx = pi.x - pj.x
                dy = pi.y - pj.y
                r_sq = dx * dx + dy * dy
                
                if r_sq < 1e-12 or r_sq > cutoff_sq:
                    continue
                
                inv_r2 = 1.0 / r_sq
                sig2_over_r2 = (sigma * sigma) * inv_r2
                inv_r6 = sig2_over_r2 * sig2_over_r2 * sig2_over_r2
                inv_r12 = inv_r6 * inv_r6
                
                # Lennard-Jones force magnitude: 24*epsilon/r^2 * (2*(sigma/r)^12 - (sigma/r)^6)
                force_scalar = 24 * self.interaction_epsilon * inv_r2 * (2 * inv_r12 - inv_r6)
                fx = force_scalar * dx
                fy = force_scalar * dy
                
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                
                # Potential energy: 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
                potential_energy += 4 * self.interaction_epsilon * (inv_r12 - inv_r6)
        
        self.potential_energy = potential_energy
        return forces
    
    def update(self, dt):
        """Update all particles and handle wall collisions."""
        forces = self._compute_interaction_forces()
        
        for idx, particle in enumerate(self.particles):
            # Store velocity before gravity for elastic reflections
            vx_before = particle.vx
            vy_before = particle.vy
            
            # Update position
            particle.update_position(dt)
            
            # Apply gravity and pairwise interaction forces (explicit Euler on velocity)
            particle.vx += (self.gravity_x + forces[idx, 0] / particle.mass) * dt
            particle.vy += (self.gravity_y + forces[idx, 1] / particle.mass) * dt
            
            # Check for wall collisions and bounce
            # Left/right walls
            if particle.x - particle.radius < 0:
                particle.x = particle.radius
                momentum_change = abs(2 * particle.mass * vx_before)
                self.momentum_transfer += momentum_change
                particle.vx = abs(vx_before)
                self.wall_collision_count += 1
            elif particle.x + particle.radius > self.width:
                particle.x = self.width - particle.radius
                momentum_change = abs(2 * particle.mass * vx_before)
                self.momentum_transfer += momentum_change
                particle.vx = -abs(vx_before)
                self.wall_collision_count += 1
            
            # Top/bottom walls
            if particle.y - particle.radius < 0:
                particle.y = particle.radius
                momentum_change = abs(2 * particle.mass * vy_before)
                self.momentum_transfer += momentum_change
                particle.vy = abs(vy_before)
                self.wall_collision_count += 1
            elif particle.y + particle.radius > self.height:
                particle.y = self.height - particle.radius
                momentum_change = abs(2 * particle.mass * vy_before)
                self.momentum_transfer += momentum_change
                particle.vy = -abs(vy_before)
                self.wall_collision_count += 1
        
        self.time += dt
        self.time_elapsed += dt
    
    def total_kinetic_energy(self):
        """Calculate total kinetic energy of all particles."""
        return sum(p.kinetic_energy() for p in self.particles)
    
    def temperature(self):
        """
        Calculate temperature from kinetic energy.
        Using equipartition theorem for 2D: <KE> = k_B * T per particle
        where <KE> is the average kinetic energy per particle.
        With k_B = 1 (natural units), T = <KE> = total_KE / N
        """
        if len(self.particles) == 0:
            return 0.0
        return self.total_kinetic_energy() / len(self.particles)
    
    def pressure(self):
        """
        Calculate pressure from momentum transfer on walls.
        Pressure = (momentum transfer) / (wall area * time)
        In 2D, "area" is actually perimeter: 2*(width + height)
        """
        if self.time_elapsed == 0:
            return 0.0
        perimeter = 2 * (self.width + self.height)
        return self.momentum_transfer / (perimeter * self.time_elapsed)
    
    def get_speeds(self):
        """Get array of all particle speeds."""
        return np.array([p.speed() for p in self.particles])
    
    def reset_measurements(self):
        """Reset time-averaged measurements."""
        self.momentum_transfer = 0.0
        self.time_elapsed = 0.0
        self.wall_collision_count = 0


def initialize_particles(n_particles=20, box_width=1.0, box_height=1.0, 
                        max_speed=1.0, particle_radius=0.02,
                        x_min=0.0, x_max=None, y_min=0.0, y_max=None):
    """
    Initialize particles with random positions and velocities.
    
    Args:
        n_particles: Number of particles to create
        box_width: Width of the box
        box_height: Height of the box
        max_speed: Maximum initial speed
        particle_radius: Radius of each particle
        x_min/x_max/y_min/y_max: Bounds for initial position sampling
    
    Returns:
        List of Particle objects
    """
    if x_max is None:
        x_max = box_width
    if y_max is None:
        y_max = box_height
    
    # Clamp ranges to remain inside the box with a radius margin
    def clamp_range(low, high, minimum, maximum, margin):
        lo = min(low, high)
        hi = max(low, high)
        lo = max(minimum + margin, min(lo, maximum - margin))
        hi = max(minimum + margin, min(hi, maximum - margin))
        if hi <= lo:
            lo = minimum + margin
            hi = maximum - margin
        return lo, hi
    
    x_low, x_high = clamp_range(x_min, x_max, 0.0, box_width, particle_radius)
    y_low, y_high = clamp_range(y_min, y_max, 0.0, box_height, particle_radius)
    
    particles = []
    for _ in range(n_particles):
        # Random position (with margin from walls)
        x = np.random.uniform(x_low, x_high)
        y = np.random.uniform(y_low, y_high)
        
        # Random velocity from uniform distribution
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.3 * max_speed, max_speed)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        particles.append(Particle(x, y, vx, vy, radius=particle_radius))
    
    return particles


class GasSimulation:
    """Main simulation class with visualization."""
    
    def __init__(self, n_particles=20, box_width=1.0, box_height=1.0, dt=0.01,
                 max_speed=1.0, particle_radius=0.02, interaction_epsilon=0.0):
        self.box = Box2D(box_width, box_height, interaction_epsilon=interaction_epsilon,
                         interaction_sigma=2 * particle_radius)
        self.dt = dt
        self.n_particles = n_particles
        self.max_speed = max_speed
        self.particle_radius = particle_radius
        self.steps_per_frame = 5
        self.x_min_init = particle_radius
        self.x_max_init = box_width - particle_radius
        self.y_min_init = particle_radius
        self.y_max_init = box_height - particle_radius
        self.gravity_y = 0.0
        self.gravity_x = 0.0
        self.interaction_epsilon = interaction_epsilon
        
        # Initialize particles
        particles = initialize_particles(
            n_particles, box_width, box_height,
            max_speed=max_speed, particle_radius=particle_radius,
            x_min=self.x_min_init, x_max=self.x_max_init,
            y_min=self.y_min_init, y_max=self.y_max_init
        )
        for p in particles:
            self.box.add_particle(p)
        
        # Setup figure and subplots
        self.fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Main particle view
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_main.set_xlim(0, box_width)
        self.ax_main.set_ylim(0, box_height)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_xlabel('X Position')
        self.ax_main.set_ylabel('Y Position')
        self.ax_main.set_title('2D Ideal Gas Simulation')
        
        # Velocity distribution histogram
        self.ax_hist = self.fig.add_subplot(gs[0, 1])
        self.ax_hist.set_xlabel('Speed')
        self.ax_hist.set_ylabel('Count')
        self.ax_hist.set_title('Velocity Distribution')
        
        # Temperature and pressure text
        self.ax_text = self.fig.add_subplot(gs[1, 1])
        self.ax_text.axis('off')
        
        # Initialize scatter plot for particles
        positions = np.array([[p.x, p.y] for p in self.box.particles])
        self.scatter = self.ax_main.scatter(positions[:, 0], positions[:, 1], 
                                           s=100, c='blue', alpha=0.6)
        
        # Draw box boundaries
        self.ax_main.plot([0, box_width, box_width, 0, 0], 
                         [0, 0, box_height, box_height, 0], 
                         'k-', linewidth=2)
        
        self.frame_count = 0
        self.measurement_interval = 100  # Reset measurements every N frames
        
        # Add interactive controls
        self._setup_controls()
    
    def _setup_controls(self):
        """Create sliders and buttons for interactive parameter control."""
        self.fig.subplots_adjust(bottom=0.5)
        axcolor = '#f0f0f0'
        
        # Number of particles slider
        ax_particles = self.fig.add_axes([0.08, 0.23, 0.32, 0.03], facecolor=axcolor)
        self.slider_particles = Slider(
            ax=ax_particles, label='Particles', valmin=1, valmax=300,
            valinit=self.n_particles, valstep=1
        )
        
        # Max initial speed slider (applied on reset)
        ax_speed = self.fig.add_axes([0.08, 0.18, 0.32, 0.03], facecolor=axcolor)
        self.slider_max_speed = Slider(
            ax=ax_speed, label='Max Init Speed', valmin=0.2, valmax=3.0,
            valinit=self.max_speed, valstep=0.05
        )
        
        # Time step slider (applies immediately)
        ax_dt = self.fig.add_axes([0.08, 0.13, 0.32, 0.03], facecolor=axcolor)
        self.slider_dt = Slider(
            ax=ax_dt, label='dt', valmin=0.001, valmax=0.05,
            valinit=self.dt, valstep=0.001
        )
        
        # Initial position range sliders
        ax_x_min = self.fig.add_axes([0.58, 0.23, 0.32, 0.03], facecolor=axcolor)
        self.slider_x_min = Slider(
            ax=ax_x_min, label='X Min', valmin=0.0, valmax=self.box.width,
            valinit=self.x_min_init, valstep=0.01
        )
        
        ax_x_max = self.fig.add_axes([0.58, 0.18, 0.32, 0.03], facecolor=axcolor)
        self.slider_x_max = Slider(
            ax=ax_x_max, label='X Max', valmin=0.0, valmax=self.box.width,
            valinit=self.x_max_init, valstep=0.01
        )
        
        ax_y_min = self.fig.add_axes([0.58, 0.13, 0.32, 0.03], facecolor=axcolor)
        self.slider_y_min = Slider(
            ax=ax_y_min, label='Y Min', valmin=0.0, valmax=self.box.height,
            valinit=self.y_min_init, valstep=0.01
        )
        
        ax_y_max = self.fig.add_axes([0.58, 0.08, 0.32, 0.03], facecolor=axcolor)
        self.slider_y_max = Slider(
            ax=ax_y_max, label='Y Max', valmin=0.0, valmax=self.box.height,
            valinit=self.y_max_init, valstep=0.01
        )
        
        # Gravity (y-acceleration) slider
        ax_gravity = self.fig.add_axes([0.08, 0.08, 0.32, 0.03], facecolor=axcolor)
        self.slider_gravity = Slider(
            ax=ax_gravity, label='Gravity (y-acc)', valmin=-10.0, valmax=10.0,
            valinit=self.gravity_y, valstep=0.05
        )
        
        # Interaction strength (LJ epsilon) slider
        ax_eps = self.fig.add_axes([0.08, 0.03, 0.32, 0.03], facecolor=axcolor)
        self.slider_interaction = Slider(
            ax=ax_eps,
            label='Interaction ε (log10)',
            valmin=-1000.0,
            valmax=0.0,
            valinit=np.clip(np.log10(max(self.interaction_epsilon, 1e-300)), -1000.0, 0.0),
            valstep=0.05,
        )
        
        # Button to rebuild the simulation with current controls
        ax_reset = self.fig.add_axes([0.58, 0.03, 0.14, 0.045])
        self.button_reset = Button(ax_reset, 'Reset with Controls')
        
        # Hook up callbacks
        self.slider_dt.on_changed(self._on_dt_change)
        self.slider_max_speed.on_changed(self._on_max_speed_change)
        self.slider_particles.on_changed(self._on_particle_change)
        self.slider_x_min.on_changed(self._on_x_min_change)
        self.slider_x_max.on_changed(self._on_x_max_change)
        self.slider_y_min.on_changed(self._on_y_min_change)
        self.slider_y_max.on_changed(self._on_y_max_change)
        self.slider_gravity.on_changed(self._on_gravity_change)
        self.slider_interaction.on_changed(self._on_interaction_change)
        self.button_reset.on_clicked(self.reset_simulation)
    
    def _on_dt_change(self, val):
        """Update timestep from slider."""
        self.dt = float(val)
    
    def _on_max_speed_change(self, val):
        """Update stored max speed (used when resetting)."""
        self.max_speed = float(val)
    
    def _on_particle_change(self, val):
        """Keep particle count in sync with slider."""
        self.n_particles = int(val)
    
    def _on_x_min_change(self, val):
        self.x_min_init = float(val)
    
    def _on_x_max_change(self, val):
        self.x_max_init = float(val)
    
    def _on_y_min_change(self, val):
        self.y_min_init = float(val)
    
    def _on_y_max_change(self, val):
        self.y_max_init = float(val)
    
    def _on_gravity_change(self, val):
        """Update gravity immediately."""
        self.gravity_y = float(val)
        self.box.gravity_y = self.gravity_y
    
    def _on_interaction_change(self, val):
        """Update interaction strength immediately (log slider -> linear epsilon)."""
        epsilon = 10 ** float(val)
        self.interaction_epsilon = epsilon
        self.box.interaction_epsilon = epsilon
    
    def reset_simulation(self, event=None):
        """Rebuild the simulation state using current control settings."""
        # Ensure ranges are well-ordered before initialization
        x_min = min(self.slider_x_min.val, self.slider_x_max.val)
        x_max = max(self.slider_x_min.val, self.slider_x_max.val)
        y_min = min(self.slider_y_min.val, self.slider_y_max.val)
        y_max = max(self.slider_y_min.val, self.slider_y_max.val)
        
        self.box = Box2D(
            self.box.width,
            self.box.height,
            gravity_x=self.gravity_x,
            gravity_y=self.gravity_y,
            interaction_epsilon=self.interaction_epsilon,
            interaction_sigma=2 * self.particle_radius,
        )
        particles = initialize_particles(
            n_particles=self.n_particles,
            box_width=self.box.width,
            box_height=self.box.height,
            max_speed=self.max_speed,
            particle_radius=self.particle_radius,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )
        for p in particles:
            self.box.add_particle(p)
        
        # Update scatter plot to new positions
        positions = np.array([[p.x, p.y] for p in self.box.particles])
        self.scatter.set_offsets(positions)
        
        # Reset counters and measurements
        self.box.reset_measurements()
        self.frame_count = 0
    
    def update_frame(self, frame):
        """Update function for animation."""
        # Update physics multiple times per frame for smoother simulation
        for _ in range(self.steps_per_frame):
            self.box.update(self.dt)
        
        # Update particle positions in plot
        positions = np.array([[p.x, p.y] for p in self.box.particles])
        self.scatter.set_offsets(positions)
        
        # Update velocity distribution histogram
        speeds = self.box.get_speeds()
        temperature = self.box.temperature()
        pressure = self.box.pressure()
        self.ax_hist.clear()
        counts, bins, _ = self.ax_hist.hist(
            speeds, bins=15, color='skyblue', edgecolor='black', alpha=0.7, label='Observed'
        )
        self.ax_hist.set_xlabel('Speed')
        self.ax_hist.set_ylabel('Count')
        self.ax_hist.set_title('Velocity Distribution')
        
        # Maxwell-Boltzmann overlay (2D) based on current temperature
        if len(speeds) > 0 and temperature > 0 and len(bins) > 1:
            bin_width = bins[1] - bins[0]
            mass = self.box.particles[0].mass if self.box.particles else 1.0
            v_vals = np.linspace(0, max(bins[-1], speeds.max() * 1.2), 200)
            pdf = (mass / temperature) * v_vals * np.exp(-mass * v_vals**2 / (2 * temperature))
            scale = len(speeds) * bin_width
            self.ax_hist.plot(v_vals, pdf * scale, 'r--', linewidth=2, label='Maxwell-Boltzmann')
            self.ax_hist.legend()
        
        # Update measurements text
        self.ax_text.clear()
        self.ax_text.axis('off')
        info_text = f"""
        Simulation Statistics
        ══════════════════════════
        
        Time:             {self.box.time:.2f} s
        Particles:        {len(self.box.particles)}
        Time Step (dt):   {self.dt:.4f}
        Gravity (y-acc):  {self.gravity_y:.3f}
        Interaction ε:    {self.interaction_epsilon:.3f}
        
        Temperature:      {temperature:.4f}
        Pressure:         {pressure:.4f}
        
        Total KE:         {self.box.total_kinetic_energy():.4f}
        Potential U:      {self.box.potential_energy:.4f}
        Wall Collisions:  {self.box.wall_collision_count}
        
        Mean Speed:       {np.mean(speeds):.4f}
        Std Speed:        {np.std(speeds):.4f}
        """
        self.ax_text.text(0.1, 0.5, info_text, fontsize=10, 
                         verticalalignment='center', family='monospace')
        
        self.frame_count += 1
        
        # Periodically reset measurements for running averages
        if self.frame_count % self.measurement_interval == 0:
            self.box.reset_measurements()
        
        return self.scatter,
    
    def run(self, frames=1000, interval=20):
        """Run the animation."""
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, frames=frames, 
            interval=interval, blit=False, repeat=True
        )
        plt.show()
        return anim


def main():
    """Main function to run the simulation."""
    print("Starting 2D Ideal Gas Simulation...")
    print("Parameters:")
    print("  - 20 particles")
    print("  - 1.0 x 1.0 box")
    print("  - Elastic wall collisions only")
    print("  - Random initial positions and velocities")
    print("\nClose the window to exit.")
    
    # Create and run simulation
    sim = GasSimulation(n_particles=100, box_width=1.0, box_height=1.0, dt=0.01)
    sim.run(frames=2000, interval=20)


if __name__ == "__main__":
    main()
