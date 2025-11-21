# 2D Ideal Gas Simulation - Usage Examples

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Simulation
```bash
python simulation.py
```

This will open an animated window showing:
- 20 particles bouncing in a 2D box
- Real-time velocity distribution histogram
- Live measurements: temperature, pressure, kinetic energy, and collision count

## Running Tests

### Unit Tests
```bash
python test_simulation.py
```

Expected output:
- 10 tests covering particle physics, collisions, energy conservation
- All tests should pass

### Visual Demo (Non-Interactive)
```bash
python demo_visualization.py
```

This creates a snapshot image showing the simulation state at a single point in time.

## Understanding the Output

### Main Window Layout

**Left Panel - Particle View:**
- Blue dots represent particles
- Black box outline shows the boundaries
- Particles bounce elastically off walls

**Top Right - Velocity Distribution:**
- Histogram showing the distribution of particle speeds
- As the simulation runs, this evolves toward equilibrium

**Bottom Right - Statistics:**
- **Time**: Simulation time elapsed (seconds)
- **Particles**: Number of particles (20)
- **Temperature**: Calculated as mean kinetic energy per particle (k_B = 1)
- **Pressure**: Calculated from momentum transfer on walls
- **Total KE**: Sum of all particle kinetic energies
- **Wall Collisions**: Cumulative count of wall bounces
- **Mean Speed**: Average particle speed
- **Std Speed**: Standard deviation of speeds

## Customizing the Simulation

Edit `simulation.py` and modify the `main()` function:

```python
# Change number of particles
sim = GasSimulation(n_particles=50, box_width=1.0, box_height=1.0)

# Change box dimensions
sim = GasSimulation(n_particles=20, box_width=2.0, box_height=1.5)

# Change time step
sim = GasSimulation(n_particles=20, box_width=1.0, box_height=1.0, dt=0.005)
```

## Physics Explained

### Elastic Wall Collisions
When a particle hits a wall:
1. Position is corrected to keep particle inside
2. Velocity component perpendicular to wall is reversed
3. Momentum transfer is recorded for pressure calculation

### Temperature
```
T = <KE> = (Σ 0.5 * m * v²) / N
```
Where:
- <KE> is average kinetic energy per particle
- N is number of particles
- k_B = 1 (natural units)

### Pressure
```
P = Δp / (perimeter × Δt)
```
Where:
- Δp is total momentum transfer from wall collisions
- perimeter = 2 × (width + height) in 2D
- Δt is time interval

### Energy Conservation
The simulation conserves total kinetic energy (within numerical precision):
- No friction or damping
- Elastic collisions only
- Tests verify energy conservation < 0.01% error

## Expected Behavior

### At Startup:
- Particles have random positions (avoiding walls)
- Particles have random velocities from uniform distribution
- Initial velocity distribution is uniform

### During Simulation:
- Particles bounce off walls continuously
- Energy remains constant
- Velocity distribution gradually evolves
- Temperature stabilizes
- Pressure fluctuates around mean value

### At Equilibrium:
- Velocity distribution approaches Maxwell-Boltzmann
- Statistical properties become stable
- System reaches steady state

## Troubleshooting

### Animation Too Slow
- Reduce `n_particles`
- Increase `interval` parameter in `sim.run()`
- Reduce `steps_per_frame` in `update_frame()`

### Animation Too Fast
- Decrease `interval` parameter
- Increase `steps_per_frame`

### Particles Escaping Box
- This should not happen if code is correct
- Check that dt is small enough
- Verify wall collision logic

## Further Exploration

Try these experiments:
1. Increase particles to 50 or 100
2. Change initial velocity distributions
3. Modify box aspect ratio
4. Add particle-particle collisions (future feature)
5. Compare velocity distribution to Maxwell-Boltzmann theory

## References

- Statistical Mechanics and Thermodynamics
- Kinetic Theory of Gases
- Equipartition Theorem
- Ideal Gas Law: PV = NkT
