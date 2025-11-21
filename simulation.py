"""
2D Ideal Gas Simulation

Simulates N particles in a 2D box with elastic wall collisions.
Tracks velocity distribution, temperature, and pressure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    
    def __init__(self, width=1.0, height=1.0):
        self.width = width
        self.height = height
        self.particles = []
        self.time = 0.0
        self.wall_collision_count = 0
        self.momentum_transfer = 0.0
        self.time_elapsed = 0.0
    
    def add_particle(self, particle):
        """Add a particle to the box."""
        self.particles.append(particle)
    
    def update(self, dt):
        """Update all particles and handle wall collisions."""
        for particle in self.particles:
            # Update position
            particle.update_position(dt)
            
            # Check for wall collisions and bounce
            # Left/right walls
            if particle.x - particle.radius < 0:
                particle.x = particle.radius
                momentum_change = abs(2 * particle.mass * particle.vx)
                self.momentum_transfer += momentum_change
                particle.vx = abs(particle.vx)
                self.wall_collision_count += 1
            elif particle.x + particle.radius > self.width:
                particle.x = self.width - particle.radius
                momentum_change = abs(2 * particle.mass * particle.vx)
                self.momentum_transfer += momentum_change
                particle.vx = -abs(particle.vx)
                self.wall_collision_count += 1
            
            # Top/bottom walls
            if particle.y - particle.radius < 0:
                particle.y = particle.radius
                momentum_change = abs(2 * particle.mass * particle.vy)
                self.momentum_transfer += momentum_change
                particle.vy = abs(particle.vy)
                self.wall_collision_count += 1
            elif particle.y + particle.radius > self.height:
                particle.y = self.height - particle.radius
                momentum_change = abs(2 * particle.mass * particle.vy)
                self.momentum_transfer += momentum_change
                particle.vy = -abs(particle.vy)
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
                        max_speed=1.0, particle_radius=0.02):
    """
    Initialize particles with random positions and velocities.
    
    Args:
        n_particles: Number of particles to create
        box_width: Width of the box
        box_height: Height of the box
        max_speed: Maximum initial speed
        particle_radius: Radius of each particle
    
    Returns:
        List of Particle objects
    """
    particles = []
    for _ in range(n_particles):
        # Random position (with margin from walls)
        x = np.random.uniform(particle_radius, box_width - particle_radius)
        y = np.random.uniform(particle_radius, box_height - particle_radius)
        
        # Random velocity from uniform distribution
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.3 * max_speed, max_speed)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        particles.append(Particle(x, y, vx, vy, radius=particle_radius))
    
    return particles


class GasSimulation:
    """Main simulation class with visualization."""
    
    def __init__(self, n_particles=20, box_width=1.0, box_height=1.0, dt=0.01):
        self.box = Box2D(box_width, box_height)
        self.dt = dt
        self.n_particles = n_particles
        
        # Initialize particles
        particles = initialize_particles(n_particles, box_width, box_height)
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
    
    def update_frame(self, frame):
        """Update function for animation."""
        # Update physics multiple times per frame for smoother simulation
        steps_per_frame = 5
        for _ in range(steps_per_frame):
            self.box.update(self.dt)
        
        # Update particle positions in plot
        positions = np.array([[p.x, p.y] for p in self.box.particles])
        self.scatter.set_offsets(positions)
        
        # Update velocity distribution histogram
        speeds = self.box.get_speeds()
        self.ax_hist.clear()
        self.ax_hist.hist(speeds, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        self.ax_hist.set_xlabel('Speed')
        self.ax_hist.set_ylabel('Count')
        self.ax_hist.set_title('Velocity Distribution')
        
        # Update measurements text
        temperature = self.box.temperature()
        pressure = self.box.pressure()
        
        self.ax_text.clear()
        self.ax_text.axis('off')
        info_text = f"""
        Simulation Statistics
        ══════════════════════════
        
        Time:             {self.box.time:.2f} s
        Particles:        {self.n_particles}
        
        Temperature:      {temperature:.4f}
        Pressure:         {pressure:.4f}
        
        Total KE:         {self.box.total_kinetic_energy():.4f}
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
    sim = GasSimulation(n_particles=20, box_width=1.0, box_height=1.0, dt=0.01)
    sim.run(frames=2000, interval=20)


if __name__ == "__main__":
    main()
