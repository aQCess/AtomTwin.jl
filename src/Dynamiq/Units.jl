module Units

export Hz, kHz, MHz, GHz, THz,
       s, ms, µs, ns, ps,
       m, cm, mm, µm, nm,
       kg, g, mg,
       W, mW, µW,
       J, mJ, µJ,
       K, mK, µK, nK,
       G
export hbar, h, c, kb, ε0, μ0, e,
       amu, a0, m_e, m_p, µB

#------------------------------------------------------------------------------
# Units
#------------------------------------------------------------------------------

"""
    Hz

Frequency unit: hertz.

- Definition: \\(1\\,\\text{Hz} = 1\\,\\text{s}^{-1}\\).
"""
const Hz = 1.0

"""
    kHz

Frequency unit: kilohertz.

- Definition: \\(1\\,\\text{kHz} = 10^3\\,\\text{Hz}\\).
"""
const kHz = 1.0e3

"""
    MHz

Frequency unit: megahertz.

- Definition: \\(1\\,\\text{MHz} = 10^6\\,\\text{Hz}\\).
"""
const MHz = 1.0e6

"""
    GHz

Frequency unit: gigahertz.

- Definition: \\(1\\,\\text{GHz} = 10^9\\,\\text{Hz}\\).
"""
const GHz = 1.0e9

"""
    THz

Frequency unit: terahertz.

- Definition: \\(1\\,\\text{THz} = 10^{12}\\,\\text{Hz}\\).
"""
const THz = 1.0e12

"""
    s

Time unit: second.

- Definition: SI base unit of time.
"""
const s = 1.0

"""
    ms

Time unit: millisecond.

- Definition: \\(1\\,\\text{ms} = 10^{-3}\\,\\text{s}\\).
"""
const ms = 1.0e-3

"""
    µs

Time unit: microsecond.

- Definition: \\(1\\,\\text{µs} = 10^{-6}\\,\\text{s}\\).
"""
const µs = 1.0e-6

"""
    ns

Time unit: nanosecond.

- Definition: \\(1\\,\\text{ns} = 10^{-9}\\,\\text{s}\\).
"""
const ns = 1.0e-9

"""
    ps

Time unit: picosecond.

- Definition: \\(1\\,\\text{ps} = 10^{-12}\\,\\text{s}\\).
"""
const ps = 1.0e-12

"""
    m

Length unit: meter.

- Definition: SI base unit of length.
"""
const m = 1.0

"""
    cm

Length unit: centimeter.

- Definition: \\(1\\,\\text{cm} = 10^{-2}\\,\\text{m}\\).
"""
const cm = 1.0e-2

"""
    mm

Length unit: millimeter.

- Definition: \\(1\\,\\text{mm} = 10^{-3}\\,\\text{m}\\).
"""
const mm = 1.0e-3

"""
    µm

Length unit: micrometer.

- Definition: \\(1\\,\\text{µm} = 10^{-6}\\,\\text{m}\\).
"""
const µm = 1.0e-6

"""
    nm

Length unit: nanometer.

- Definition: \\(1\\,\\text{nm} = 10^{-9}\\,\\text{m}\\).
"""
const nm = 1.0e-9

"""
    kg

Mass unit: kilogram.

- Definition: SI base unit of mass.
"""
const kg = 1.0

"""
    g

Mass unit: gram.

- Definition: \\(1\\,\\text{g} = 10^{-3}\\,\\text{kg}\\).
"""
const g = 1.0e-3

"""
    mg

Mass unit: milligram.

- Definition: \\(1\\,\\text{mg} = 10^{-6}\\,\\text{kg}\\).
"""
const mg = 1.0e-6

"""
    W

Power unit: watt.

- Definition: \\(1\\,\\text{W} = 1\\,\\text{J/s}\\).
"""
const W = 1.0

"""
    mW

Power unit: milliwatt.

- Definition: \\(1\\,\\text{mW} = 10^{-3}\\,\\text{W}\\).
"""
const mW = 1.0e-3

"""
    µW

Power unit: microwatt.

- Definition: \\(1\\,\\text{µW} = 10^{-6}\\,\\text{W}\\).
"""
const µW = 1.0e-6

"""
    J

Energy unit: joule.

- Definition: \\(1\\,\\text{J} = 1\\,\\text{kg}\\,\\text{m}^2\\,\\text{s}^{-2}\\).
"""
const J = 1.0

"""
    mJ

Energy unit: millijoule.

- Definition: \\(1\\,\\text{mJ} = 10^{-3}\\,\\text{J}\\).
"""
const mJ = 1.0e-3

"""
    µJ

Energy unit: microjoule.

- Definition: \\(1\\,\\text{µJ} = 10^{-6}\\,\\text{J}\\).
"""
const µJ = 1.0e-6


"""
    G

Energy unit: Gauss.

- Definition: \\(1\\,\\text{G} = 10^{-4}\\,\\text{T}\\).
"""
const G = 1.0e-4

"""
    K

Temperature unit: kelvin.

- Definition: SI base unit of thermodynamic temperature.
"""
const K = 1.0

"""
    mK

Temperature unit: millikelvin.

- Definition: \\(1\\,\\text{mK} = 10^{-3}\\,\\text{K}\\).
"""
const mK = 1.0e-3

"""
    µK

Temperature unit: microkelvin.

- Definition: \\(1\\,\\text{µK} = 10^{-6}\\,\\text{K}\\).
"""
const µK = 1.0e-6

"""
    nK

Temperature unit: nanokelvin.

- Definition: \\(1\\,\\text{nK} = 10^{-9}\\,\\text{K}\\).
"""
const nK = 1.0e-9

#------------------------------------------------------------------------------
# Fundamental constants (SI)
#------------------------------------------------------------------------------

"""
    hbar

Reduced Planck constant \\(\\hbar\\).

- Physical meaning: quantum of action divided by \\(2\\pi\\).
- SI units: joule seconds (J·s).
"""
const hbar = 1.054_571_817e-34

"""
    h

Planck constant \\(h = 2\\pi\\hbar\\).

- Physical meaning: quantum of action.
- SI units: joule seconds (J·s).
"""
const h = 2π * hbar

"""
    c

Speed of light in vacuum.

- SI units: meters per second (m/s).
"""
const c = 2.997_924_58e8

"""
    kb

Boltzmann constant \\(k_B\\).

- Physical meaning: conversion between temperature and energy.
- SI units: joules per kelvin (J/K).
"""
const kb = 1.380_649e-23

"""
    ε0

Vacuum permittivity \\(\\varepsilon_0\\).

- SI units: farads per meter (F/m).
"""
const ε0 = 8.854_187_812_8e-12

"""
    μ0

Vacuum permeability \\(\\mu_0\\).

- SI units: newtons per ampere squared (N/A²).
"""
const μ0 = 4π * 1.0e-7

"""
    e

Elementary charge.

- SI units: coulombs (C).
"""
const e = 1.602_176_634e-19

"""
    amu

Atomic mass unit.

- Physical meaning: 1 unified atomic mass unit.
- SI units: kilograms (kg).
"""
const amu = 1.660_539_066_60e-27

"""
    a0

Bohr radius \\(a_0\\).

- Physical meaning: characteristic length scale of the hydrogen atom.
- SI units: meters (m).
"""
const a0 = 5.291_772_109_03e-11

"""
    m_e

Electron mass.

- SI units: kilograms (kg).
"""
const m_e = 9.109_383_7015e-31

"""
    m_p

Proton mass.

- SI units: kilograms (kg).
"""
const m_p = 1.672_621_923_69e-27

"""
Bohr magneton \\(\\mu_B\\).

- Physical meaning: quantum of action.
- SI units: joule / T (J·T⁻¹).
"""
const µB = 9.274_010_065_7e-24

end # module Units
