# Quick Start Guide

## Installation

AtomTwin is a Julia package. The easiest way to use it is via the Julia package manager.

### Requirements

- Julia **1.11** or later
- Internet access to download dependencies from the General registry


### Install Julia with juliaup  
>
> `juliaup` is the recommended way to install and manage Julia on all major platforms.
>
> - **Windows**: Install “Julia” from the Microsoft Store or via the Windows package manager (`winget`), which also installs `juliaup`. Then open a new terminal and run `julia` to start the Julia REPL.
> - **Linux**: Use your distribution’s instructions for installing `juliaup` (typically a single shell command from the official Julia download page), then open a terminal and run `julia`.
>
> If you want Julia to use multiple CPU threads, start it with
> `julia --threads auto` (or set the `JULIA_NUM_THREADS` environment variable), then follow the steps below to add `AtomTwin`.

### Install AtomTwin from the registry

Start Julia, enter the package mode with `]`, and add AtomTwin:

```julia
julia> ]
pkg> add AtomTwin
```

Then, in your code:

```julia
using AtomTwin
```

---

## Your first simulation: driven two‑level atom

An AtomTwin simulation usually follows three steps:

1. **Define the system** (atoms, tweezers, interactions, noise, …)
2. **Build a sequence** of operations and **attach detectors**
3. **Run the simulation** with `play` and inspect the outputs

### 1. Define a two‑level atom and system

```@example quickstart
using AtomTwin

g, e = Level(; label = "g"), Level(; label = "e")
atom  = Atom(; levels = [g, e])
system = System(atom)
```

### 2. Add physics and a detector

Add a resonant coupling between |g⟩ and |e⟩ with a Rabi frequency of 1 MHz:

```@example quickstart
add_coupling!(system, atom, g => e, 2π * 1e6)
```

Register a population detector on the excited state so we can plot the dynamics later:

```@example quickstart
add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))
```

Now define a sequence that records every 1 ns over a 5 µs wait:

```@example quickstart
seq = Sequence(1e-9)  # output every nanosecond
@sequence seq begin
    Wait(5e-6)        # duration of 5 microseconds
end
```

### 3. Run the simulation

```@example quickstart
out = play(system, seq; initial_state = g)
```

Detector outputs are stored in `out.detectors` by name. For the excited‑state population:

```@example quickstart
out.detectors["P_e"]
```

You can now plot `out.detectors["P_e"]` against `out.times` using your plotting package of choice.

---

## Using AtomTwin in a project environment

For a new project directory:

```bash
mkdir my_atom_project
cd my_atom_project
julia --project=.
```

Then in the Julia REPL:

```julia
julia> ]
pkg> add AtomTwin
```

Your `Project.toml` will now list AtomTwin as a dependency, and you can use it in any script inside this directory:

```julia
using AtomTwin
```

---

## Example workflows

Once AtomTwin is installed, you can:

- Explore **more examples** such as Rabi oscillations with noise and dissipation.
- Design and optimize your own sequences including **quantum** and **classical** dynamics of atoms in tweezer arrays.
- Prototype your own **quantum instructions**, estimate fidelities and error budgets.

