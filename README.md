# AtomTwin.jl

[![CI](https://github.com/aQCess/AtomTwin.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/aQCess/AtomTwin.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://aQCess.github.io/AtomTwin.jl/stable/)

AtomTwin is an open‑source **quantum EDA and physics level digital‑twin framework for neutral‑atom quantum processors**, focused on realistic, physics‑first simulation of atoms, tweezers, and laser control, including atomic motion, realistic level structures, laser noise, and more. While the primary focus is on neutral‑atom arrays, AtomTwin can model a broad class of atomic quantum systems driven by laser fields, including trapped ions, atomic clocks, quantum memories, interfaces, and sensors. It enables hardware and firmware developers to design, test, and validate low‑level, firmware‑like instructions without manually specifying Hamiltonians, bridging ideas to hardware‑realistic control sequences in a single environment. **AtomTwin sits between textbook models and lab hardware, providing vendor‑agnostic, physics‑level digital twins of atomic quantum devices rather than a generic circuit‑level or Hamiltonian‑only simulator.**

> **Status:** beta
> **Homepage / Docs:** https://aQCess.github.io/AtomTwin.jl/stable/
> **Repository:** https://github.com/aQCess/AtomTwin.jl
> **License:** Apache‑2.0
---

## Who is this for?

- Quantum hardware and firmware developers, to design and validate instruction sets, pulse sequences, and control stacks for neutral‑atom quantum processors, atomic clocks, interfaces, quantum sensors, and related devices.
- Experimental atomic physicists, for fast and accurate simulation of neutral‑atom experiments and other laser‑driven atomic systems.
- Quantum error correction and compiler researchers, to access hardware‑realistic noise and error models derived from atomic‑level simulations.
- Quantum technology educators, to demonstrate hands‑on concepts in realistic experimental setups using neutral‑atom and other atom‑like platforms.
- Students in quantum physics and quantum engineering, to explore atomic quantum hardware using approachable, research‑grade tools.

## Features

- **Physics‑first modeling in Julia** for fast, composable simulation code with an accessible, research‑friendly syntax.
- **Atom / tweezer / laser primitives** so you build simulations directly from physical components instead of abstract Hamiltonians.
- **High‑performance engines for coupled quantum and classical dynamics** at the pulse level.
- **Firmware‑like instruction sequences** that mimic execution on real neutral‑atom processors.
- **Ready‑to‑use models for common atomic species** (e.g. ytterbium), easily extended to new level schemes.
- **Visualization hooks** for atomic trajectories and quantum state evolution (ideal for debugging and teaching).
- Interactive notebook workflows for exploring key quantum protocols in a reproducible way.
- Suitable for independent research and for use in courses, labs, and workshops on neutral‑atom platforms.

Planned future directions include a **quantum SDK for instruction‑level control**, a **CAE/EDA‑style environment** for hardware engineers, and a **physics backend for digital twins** of neutral‑atom processors such as the aQCess QPU in Strasbourg.


---

## Installation

AtomTwin is a Julia package distributed via the General registry.

### Requirements

- Julia 1.11 or later
- Internet access to download dependencies from the General registry

### Install Julia with juliaup  
>
> `juliaup` is the recommended way to install and manage Julia on all major platforms.[1][2][3]
>
> - **Windows**: Install “Julia” from the Microsoft Store or via the Windows package manager (`winget`), which also installs `juliaup`. Then open a new terminal and run `julia` to start the Julia REPL.
> - **Linux**: Use your distribution’s instructions for installing `juliaup` (typically a single shell command from the official Julia download page), then open a terminal and run `julia`.
> 
> Once `julia` starts, you can follow the steps below to add `AtomTwin`.

### Install AtomTwin from the Julia REPL

```julia
julia> ]
pkg> add AtomTwin
```

Then load the package in your code:

```julia
using AtomTwin
```

To use AtomTwin in a project environment:

```
mkdir my_atom_project
cd my_atom_project
julia --project=.
```

Then, inside the Julia REPL:

```
julia> ]
pkg> add AtomTwin
```

Your `Project.toml` will now list AtomTwin as a dependency, and you can use it in any script in that directory:

```julia
using AtomTwin
```

---

## Quickstart example

A typical AtomTwin workflow has three steps:

1. Define the system (atoms, tweezers, interactions, noise, …).
2. Build a sequence of operations and attach detectors.
3. Run the simulation with `play` and inspect the outputs.

Below is a minimal driven two‑level atom example (simplified from the documentation):

```julia
using AtomTwin

# 1. Define a two-level atom and system
g, e = Level(; label = "g"), Level(; label = "e")
atom   = Atom(; levels = [g, e])
system = System(atom)

# 2. Add physics and a detector
add_coupling!(system, atom, g => e, 2π * 1e6)  # 1 MHz Rabi frequency
add_detector!(system, PopulationDetectorSpec(atom, e; name = "P_e"))

seq = Sequence(1e-9)   # 1 ns time step
@sequence seq begin
    Wait(5e-6)         # 5 μs
end

# 3. Run the simulation
out = play(system, seq; initial_state = g)

# Access detector data
pe = out.detectors["P_e"]
times = out.times
```

You can now plot `pe` versus `times` with your preferred plotting package (e.g. Makie, Plots.jl).

---

## Use cases

AtomTwin is being developed as a physics‑first **quantum EDA environment** for neutral‑atom processors, analogous to classical chip‑design tools. Typical use cases include:

- Designing and validating **device architectures and instruction sets**, from trap architecture and species choices to pulse schemes and firmware‑like instruction sequences.
- Prototyping and calibrating **control stacks** under realistic hardware constraints, including motion, level structure, and laser noise.
- Building **circuit‑level digital twins** of specific processors and extracting calibrated noise and timing models for higher‑level SDKs, compilers, and QEC studies.
- Teaching neutral‑atom quantum computing and atomic quantum technologies through hands‑on simulations and laboratory‑style exercises.

---

## Project roadmap

AtomTwin aims to grow into:

- A quantum SDK for instruction‑level control and algorithm–hardware co‑design.
- A quantum CAE/EDA platform for hardware engineers to prototype and validate instruction sets under realistic constraints.
- A physics backend for digital twins that inform noise models and instruction‑level errors for aQCess (Strasbourg) and compatible platforms.

---

## Documentation

Full documentation, including tutorials, API reference, and example notebooks, is available at: https://aQCess.github.io/AtomTwin.jl/stable/

Key entry points:

- **Getting started**: basic workflows and core concepts (atoms, tweezers, lasers, sequences, detectors).
- **Examples**: driven two‑level systems, Rabi oscillations with noise, tweezer arrays, instruction‑level protocols.
- **API reference**: types and functions for building systems, sequences, and detectors.

---

## Contributing

AtomTwin welcomes contributors, especially users working with neutral‑atom hardware or teaching with these platforms. Contributions are welcome in the following areas:

- Expanding atomic physics data to include new atomic species and level schemes.
- Validation of models for traps, lasers, and noise.
- Performance improvements, e.g. adaptive steps and GPU acceleration.
- Documentation, tutorials, and teaching materials.
- Integrations with other quantum‑software ecosystems.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards, testing guidelines, and how to propose changes.

Bug reports and feature requests can be submitted via [GitHub Issues](https://github.com/aQCess/AtomTwin.jl/issues). When reporting a problem, include:

- A minimal reproducible example.
- Your Julia version and AtomTwin version.
- OS and relevant hardware details.

---

## Acknowledgements

AtomTwin has received support from:

- **DigiQ – Digitally Enhanced Quantum Technology Master**, funded by the EU Digital Europe Programme (grant No. 101084035).
- **EuRyQa – European infrastructure for Rydberg Quantum Computing**, funded by the EU Horizon Europe programme (grant No. 101070144).
- **aQCess – Atomic Quantum Computing as a Service**, funded under the French *Programme d’Investissements d’Avenir* framework (Université de Strasbourg / CNRS).

---

## Citation

If you use AtomTwin in research or teaching, please cite the software directly:

```
@misc{whitlock2026atomtwinjlphysicsnativedigitaltwin,
      title={AtomTwin.jl: a physics-native digital twin framework for neutral-atom quantum processors}, 
      author={Shannon Whitlock},
      year={2026},
      eprint={2604.18531},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2604.18531}, 
}
```

## License

AtomTwin is released under the Apache-2.0 license; see `LICENSE` for details.

---

## Contact

- Lead maintainer: Shannon Whitlock, University of Strasbourg 
- Project website: https://github.com/aQCess/AtomTwin.jl
