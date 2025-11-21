# 2D Ideal Gas in a Box

Simulate N particles in a 2D box, bouncing elastically off the walls (and optionally off each other). This simulation allows you to:
- Watch how velocity distribution evolves
- Measure temperature from kinetic energy
- Measure pressure from momentum transfer on the walls
- Compare to Maxwell–Boltzmann distribution and the ideal gas law

## Features

- **20 particles** in a 2D box (1.0 × 1.0 units)
- **Elastic wall collisions** (no particle-particle collisions in basic version)
- **Random initial conditions**: positions and velocities from simple distributions
- **Real-time visualization** using matplotlib animation
- **Live measurements**:
  - Temperature calculated from kinetic energy
  - Pressure calculated from momentum transfer on walls
  - Velocity distribution histogram
  - Statistical metrics (mean speed, standard deviation)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yonasgebremarie/2DIdealGasInABox.git
cd 2DIdealGasInABox
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python simulation.py
```

The simulation will open a window with:
- **Left panel**: Animated view of particles in the 2D box
- **Top-right panel**: Histogram of velocity distribution
- **Bottom-right panel**: Real-time statistics (temperature, pressure, kinetic energy, etc.)

Close the window to exit the simulation.

## Physics Background

### Temperature
Temperature is calculated from the average kinetic energy of particles:
```
T = <KE> / k_B
```
where we use k_B = 1 for simplicity (natural units).

### Pressure
Pressure is calculated from the momentum transfer on the walls:
```
P = Δp / (A × Δt)
```
where Δp is the total momentum transfer, A is the wall perimeter (in 2D), and Δt is the time interval.

### Velocity Distribution
The simulation tracks the distribution of particle speeds, which should approach a Maxwell-Boltzmann distribution at equilibrium.

## Customization

You can modify the simulation parameters in `simulation.py`:
- `n_particles`: Number of particles (default: 20)
- `box_width`, `box_height`: Box dimensions (default: 1.0 × 1.0)
- `max_speed`: Maximum initial particle speed (default: 1.0)
- `dt`: Time step for physics updates (default: 0.01)

## License

See LICENSE file for details.
