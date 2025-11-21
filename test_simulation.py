"""
Test script for 2D ideal gas simulation.
Tests core functionality without GUI.
"""

import numpy as np
from simulation import Particle, Box2D, initialize_particles


def test_particle_creation():
    """Test creating a particle."""
    print("Testing particle creation...")
    p = Particle(0.5, 0.5, 1.0, 0.5)
    assert p.x == 0.5
    assert p.y == 0.5
    assert p.vx == 1.0
    assert p.vy == 0.5
    assert p.mass == 1.0
    print("  ✓ Particle creation works")


def test_particle_movement():
    """Test particle position update."""
    print("Testing particle movement...")
    p = Particle(0.5, 0.5, 1.0, 0.5)
    initial_x, initial_y = p.x, p.y
    dt = 0.1
    p.update_position(dt)
    assert abs(p.x - (initial_x + 1.0 * dt)) < 1e-10
    assert abs(p.y - (initial_y + 0.5 * dt)) < 1e-10
    print("  ✓ Particle movement works")


def test_kinetic_energy():
    """Test kinetic energy calculation."""
    print("Testing kinetic energy...")
    p = Particle(0.5, 0.5, 1.0, 0.0, mass=2.0)
    ke = p.kinetic_energy()
    expected_ke = 0.5 * 2.0 * (1.0**2 + 0.0**2)
    assert abs(ke - expected_ke) < 1e-10
    print(f"  ✓ Kinetic energy: {ke} (expected: {expected_ke})")


def test_speed():
    """Test speed calculation."""
    print("Testing speed calculation...")
    p = Particle(0.5, 0.5, 3.0, 4.0)
    speed = p.speed()
    expected_speed = 5.0  # 3-4-5 triangle
    assert abs(speed - expected_speed) < 1e-10
    print(f"  ✓ Speed: {speed} (expected: {expected_speed})")


def test_box_creation():
    """Test box creation."""
    print("Testing box creation...")
    box = Box2D(1.0, 1.0)
    assert box.width == 1.0
    assert box.height == 1.0
    assert len(box.particles) == 0
    print("  ✓ Box creation works")


def test_wall_collisions():
    """Test wall collision detection and bounce."""
    print("Testing wall collisions...")
    box = Box2D(1.0, 1.0)
    
    # Test right wall collision
    p = Particle(0.95, 0.5, 1.0, 0.0, radius=0.02)
    box.add_particle(p)
    
    # Move particle to hit right wall
    for _ in range(10):
        box.update(0.01)
    
    # Particle should have bounced and be moving left
    assert p.vx < 0, f"Expected negative vx after right wall collision, got {p.vx}"
    assert 0 < p.x < box.width, f"Particle escaped box: x={p.x}"
    print("  ✓ Right wall collision works")
    
    # Test left wall collision
    box2 = Box2D(1.0, 1.0)
    p2 = Particle(0.05, 0.5, -1.0, 0.0, radius=0.02)
    box2.add_particle(p2)
    
    for _ in range(10):
        box2.update(0.01)
    
    assert p2.vx > 0, f"Expected positive vx after left wall collision, got {p2.vx}"
    assert 0 < p2.x < box2.width, f"Particle escaped box: x={p2.x}"
    print("  ✓ Left wall collision works")
    
    # Test top wall collision
    box3 = Box2D(1.0, 1.0)
    p3 = Particle(0.5, 0.95, 0.0, 1.0, radius=0.02)
    box3.add_particle(p3)
    
    for _ in range(10):
        box3.update(0.01)
    
    assert p3.vy < 0, f"Expected negative vy after top wall collision, got {p3.vy}"
    assert 0 < p3.y < box3.height, f"Particle escaped box: y={p3.y}"
    print("  ✓ Top wall collision works")
    
    # Test bottom wall collision
    box4 = Box2D(1.0, 1.0)
    p4 = Particle(0.5, 0.05, 0.0, -1.0, radius=0.02)
    box4.add_particle(p4)
    
    for _ in range(10):
        box4.update(0.01)
    
    assert p4.vy > 0, f"Expected positive vy after bottom wall collision, got {p4.vy}"
    assert 0 < p4.y < box4.height, f"Particle escaped box: y={p4.y}"
    print("  ✓ Bottom wall collision works")


def test_temperature_calculation():
    """Test temperature calculation."""
    print("Testing temperature calculation...")
    box = Box2D(1.0, 1.0)
    
    # Add particles with known velocities
    box.add_particle(Particle(0.3, 0.3, 1.0, 0.0, mass=1.0))
    box.add_particle(Particle(0.7, 0.7, 0.0, 1.0, mass=1.0))
    
    # Total KE = 0.5*1*(1^2 + 0^2) + 0.5*1*(0^2 + 1^2) = 0.5 + 0.5 = 1.0
    # Temperature = KE / N = 1.0 / 2 = 0.5
    temp = box.temperature()
    expected_temp = 0.5
    assert abs(temp - expected_temp) < 1e-10
    print(f"  ✓ Temperature: {temp} (expected: {expected_temp})")


def test_pressure_measurement():
    """Test pressure measurement from wall collisions."""
    print("Testing pressure measurement...")
    box = Box2D(1.0, 1.0)
    
    # Add a fast particle that will hit walls
    p = Particle(0.5, 0.5, 2.0, 0.0, mass=1.0, radius=0.02)
    box.add_particle(p)
    
    # Run simulation
    for _ in range(100):
        box.update(0.01)
    
    # Should have some pressure from collisions
    pressure = box.pressure()
    assert pressure > 0, f"Expected positive pressure, got {pressure}"
    assert box.wall_collision_count > 0, "Expected some wall collisions"
    print(f"  ✓ Pressure: {pressure:.4f}")
    print(f"  ✓ Wall collisions: {box.wall_collision_count}")


def test_initialize_particles():
    """Test particle initialization."""
    print("Testing particle initialization...")
    particles = initialize_particles(n_particles=20, box_width=1.0, box_height=1.0)
    
    assert len(particles) == 20
    
    # Check all particles are within bounds
    for p in particles:
        assert 0 < p.x < 1.0, f"Particle x={p.x} out of bounds"
        assert 0 < p.y < 1.0, f"Particle y={p.y} out of bounds"
        assert p.speed() > 0, "Particle should have non-zero speed"
    
    print(f"  ✓ Created {len(particles)} particles")
    print(f"  ✓ All particles within bounds")
    print(f"  ✓ Mean speed: {np.mean([p.speed() for p in particles]):.4f}")


def test_energy_conservation():
    """Test that energy is conserved (approximately) in elastic collisions."""
    print("Testing energy conservation...")
    box = Box2D(1.0, 1.0)
    
    # Add particles
    particles = initialize_particles(n_particles=5, box_width=1.0, box_height=1.0)
    for p in particles:
        box.add_particle(p)
    
    initial_energy = box.total_kinetic_energy()
    
    # Run simulation
    for _ in range(100):
        box.update(0.01)
    
    final_energy = box.total_kinetic_energy()
    
    # Energy should be conserved (within numerical tolerance)
    energy_change = abs(final_energy - initial_energy) / initial_energy
    assert energy_change < 0.01, f"Energy changed by {energy_change*100:.2f}%"
    print(f"  ✓ Initial energy: {initial_energy:.6f}")
    print(f"  ✓ Final energy: {final_energy:.6f}")
    print(f"  ✓ Energy change: {energy_change*100:.4f}%")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running 2D Ideal Gas Simulation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_particle_creation,
        test_particle_movement,
        test_kinetic_energy,
        test_speed,
        test_box_creation,
        test_wall_collisions,
        test_temperature_calculation,
        test_pressure_measurement,
        test_initialize_particles,
        test_energy_conservation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
