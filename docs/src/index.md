# AtomTwin.jl
A quantum EDA and physics-level digital‑twin for atomic quantum hardware.

AtomTwin is an open‑source **quantum EDA and digital‑twin environment for neutral‑atom quantum processors**, focused on realistic, physics‑first modeling of atoms, tweezers, and laser control, including atomic motion, realistic level structures, laser noise, and more. While the primary application is on neutral‑atom arrays, AtomTwin can model a broad class of atomic quantum systems driven by laser fields, including trapped ions, atomic clocks, quantum memories, interfaces, and sensors. It enables hardware and firmware teams to design, validate, and calibrate low‑level, firmware‑like instructions without manually defining Hamiltonians, bridging ideas to hardware‑realistic control sequences in a single environment. AtomTwin sits between textbook models and lab hardware, providing vendor‑agnostic, pulse‑level digital twins of atomic quantum devices rather than a generic circuit‑level or Hamiltonian‑only simulator.

---

## What is AtomTwin?

Neutral‑atom quantum computers use arrays of individually trapped atoms, manipulated by optical tweezers and lasers, to implement qubits and quantum gates. AtomTwin is being developed as a physics‑first environment for rapid prototyping and validation of such processors, as well as other atomic quantum technologies (clocks, sensors, quantum memories and interfaces). It enables researchers, hardware engineers, and students to develop and validate low‑level instruction sets, quantum gates, control stacks, and processor architectures through hands‑on exploration, translating concepts and design ideas into realistic models of actual quantum hardware with real‑world constraints. AtomTwin is aligned with ongoing development of the aQCess neutral‑atom QPU in Strasbourg and is intended to support multiple atomic platforms over time.

---

## Key features

- Built in Julia for fast, composable simulation code with an accessible, research‑friendly syntax.
- **Atom / tweezer / laser primitives** so you build simulations directly from physical components instead of abstract Hamiltonians.
- **Error-controlled engines for coupled quantum and classical dynamics** at the pulse level: Schrödinger, wavefunction Monte Carlo and master-equation solvers, each with semiclassical atomic motion (see [Solvers and accuracy](solvers.md)).
- **Firmware‑like instruction sequences** to mimic execution on real neutral‑atom processors.
- **Light–matter physics from atomic data**: trap light shifts with full tensor polarizability, state-dependent potentials, radiation pressure and photon recoil for laser cooling and fluorescence imaging.
- Ready‑to‑use species (Yb‑171, Yb‑174, Sr‑88, Rb‑87, K‑39) with polarizability models, easily extended to new level schemes.
- Visualization hooks for atomic trajectories and quantum state evolution for debugging and teaching.
- Worked examples covering gates, noise, state preparation, transport, spectroscopy and laser cooling.
- Suitable for independent research, device and instruction‑set design, and structured teaching (courses, labs, workshops).

---

## Roadmap and vision

AtomTwin is intended as a physics‑first **quantum EDA environment** for neutral‑atom processors, analogous to the tools used to design classical integrated circuits and chips.

Planned directions include:

- A **quantum SDK** for instruction‑level control and algorithm–hardware co‑design.
- A **quantum CAE/EDA platform** for hardware engineers to prototype and validate device architectures and instruction sets under realistic constraints.
- A physics backend for circuit‑level **digital twins**, informing noise and timing models and instruction‑level errors for aQCess (Strasbourg) and compatible platforms, and feeding higher‑level SDKs and compilers.

AtomTwin is actively looking for contributors to help shape features, develop atomic libraries, improve performance, and integrations—especially from users working with neutral‑atom hardware, quantum control/firmware, or teaching with these platforms.

---

## Acknowledgements

AtomTwin has received support from the [EuRyQa – European infrastructure for Rydberg Quantum Computing](https://www.euryqa.eu/) project, funded by the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101070144.

It has also benefited from support provided through the [DigiQ – Digitally Enhanced Quantum Technology Master](https://digiq.eu/) project, funded by the European Union’s Digital Europe Programme under grant agreement No. 101084035.

In addition, AtomTwin has been supported by the [aQCess project (Atomic Quantum Computing as a Service)](https://aqcess.cesq.fr), funded under the French *Programme d’Investissements d’Avenir* framework (Université de Strasbourg / CNRS).

