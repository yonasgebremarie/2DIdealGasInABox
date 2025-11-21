"""
Demo script to generate a visualization snapshot without animation.
This is useful for testing the visualization setup.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from simulation import Box2D, initialize_particles


def create_snapshot():
    """Create a snapshot of the simulation at a single point in time."""
    # Setup simulation
    box_width, box_height = 1.0, 1.0
    box = Box2D(box_width, box_height)
    
    # Initialize particles
    particles = initialize_particles(n_particles=20, box_width=box_width, box_height=box_height)
    for p in particles:
        box.add_particle(p)
    
    # Run simulation for a bit
    for _ in range(100):
        box.update(0.01)
    
    # Create figure
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main particle view
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.set_xlim(0, box_width)
    ax_main.set_ylim(0, box_height)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('X Position')
    ax_main.set_ylabel('Y Position')
    ax_main.set_title('2D Ideal Gas Simulation - Snapshot')
    
    # Draw particles
    positions = np.array([[p.x, p.y] for p in box.particles])
    ax_main.scatter(positions[:, 0], positions[:, 1], s=100, c='blue', alpha=0.6)
    
    # Draw box boundaries
    ax_main.plot([0, box_width, box_width, 0, 0], 
                 [0, 0, box_height, box_height, 0], 
                 'k-', linewidth=2)
    
    # Velocity distribution histogram
    ax_hist = fig.add_subplot(gs[0, 1])
    speeds = box.get_speeds()
    ax_hist.hist(speeds, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax_hist.set_xlabel('Speed')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Velocity Distribution')
    
    # Temperature and pressure text
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')
    
    temperature = box.temperature()
    pressure = box.pressure()
    
    info_text = f"""
    Simulation Statistics
    ══════════════════════════
    
    Time:             {box.time:.2f} s
    Particles:        {len(box.particles)}
    
    Temperature:      {temperature:.4f}
    Pressure:         {pressure:.4f}
    
    Total KE:         {box.total_kinetic_energy():.4f}
    Wall Collisions:  {box.wall_collision_count}
    
    Mean Speed:       {np.mean(speeds):.4f}
    Std Speed:        {np.std(speeds):.4f}
    """
    ax_text.text(0.1, 0.5, info_text, fontsize=10, 
                 verticalalignment='center', family='monospace')
    
    # Save figure
    plt.savefig('simulation_snapshot.png', dpi=150, bbox_inches='tight')
    print("Snapshot saved to simulation_snapshot.png")
    
    # Print some stats
    print("\nSimulation Statistics:")
    print(f"  Time: {box.time:.2f} s")
    print(f"  Particles: {len(box.particles)}")
    print(f"  Temperature: {temperature:.4f}")
    print(f"  Pressure: {pressure:.4f}")
    print(f"  Total KE: {box.total_kinetic_energy():.4f}")
    print(f"  Wall Collisions: {box.wall_collision_count}")
    print(f"  Mean Speed: {np.mean(speeds):.4f}")
    print(f"  Std Speed: {np.std(speeds):.4f}")
    
    plt.close()


if __name__ == "__main__":
    print("Generating visualization snapshot...")
    create_snapshot()
    print("\nVisualization test complete!")
