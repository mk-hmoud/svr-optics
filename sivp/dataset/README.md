# PCF-SPR confinement-loss dataset (432 FV-FEM samples)

Full-vectorial finite element (FV-FEM) solutions for a photonic crystal fiber
based surface plasmon resonance sensor: a fused-silica host with a hexagonal
air-hole cladding, a 40 nm gold film, and an outer analyte channel.

Generated in the course of Zelaci et al., *Generative Adversarial Neural
Networks Model of Photonic Crystal Fiber Based Surface Plasmon Resonance
Sensor*, J. Lightwave Technol. 39(5):1515-1522 (2021),
doi:10.1109/JLT.2020.3035580.

Reused in Hammoud, Kalyoncu and Yasli, *Confinement-loss prediction in photonic
crystal fiber SPR sensors: kernel surrogates versus generative data
augmentation*, Signal, Image and Video Processing (submitted).

## File

`data.xlsx` — 432 rows, 10 columns, one row per FV-FEM solve.
Nine geometric configurations x three analyte indices x a 16-point wavelength
grid.

## Columns

| Column       | Meaning                        | Unit                | Range        |
|--------------|--------------------------------|---------------------|--------------|
| `Analyte`    | analyte refractive index        | RIU                 | 1.33 – 1.35  |
| `Re(eff)`    | real part of effective index    | dimensionless       | 1.42 – 1.46  |
| `lambda`     | wavelength                      | **100 nm** (see below) | 5.0 – 8.0 |
| `Pitch (um)` | lattice pitch                   | **10 µm** (see below)  | 0.15 – 0.24 |
| `d1 (um)`    | inner cladding hole DIAMETER    | µm                  | 0.25 – 0.45  |
| `d2 (um)`    | outer cladding hole DIAMETER    | µm                  | 0.55 – 0.75  |
| `d3 (um)`    | interstitial hole DIAMETER      | µm                  | 0.15 – 0.35  |
| `dc (um)`    | central defect hole DIAMETER    | µm                  | 0.15 (fixed) |
| `loss`       | confinement loss                | dB/cm               | 1.07e-3 – 59.1 |
| `Im(neff)`   | imaginary part of effective index | dimensionless     | 1.06e-9 – 7.15e-5 |

## Unit conventions — read before use

Two columns are not in the unit their header suggests. Both have caused errors.

**`lambda` is in units of 100 nm, not µm.** A value of 5.0 means 500 nm, and the
grid spans 500–800 nm. Confirmed independently: the Sellmeier index of fused
silica over 500–800 nm is n = 1.462–1.453, which matches the `Re(eff)` column
(1.42–1.46). Reading the column as µm (5–8 µm) gives n = 1.34–0.64, and silica
is opaque there.

Consequence: a peak shift of 0.2763 in column units is 27.63 nm, so a shift per
0.01 RIU step corresponds to a sensitivity of 2763 nm/RIU.

**`Pitch (um)` is in units of 10 µm.** A value of 0.20 means a 2 µm pitch.

**`d1`, `d2`, `d3`, `dc` are DIAMETERS, not radii.** The nominal design is often
quoted by radius (r1 = 0.225, r2 = 0.375, r3 = 0.175 µm); the corresponding
column values are 0.45, 0.75 and 0.35.

Substituting radii into the diameter columns, or a pitch in µm into `Pitch (um)`,
places the point outside the sampled design space and makes any surrogate
extrapolate.

## Sampled design space

Nine geometries, listed as (`Pitch`, `d1`, `d2`, `d3`) in column units:

    (0.24, 0.25, 0.75, 0.35)   (0.24, 0.25, 0.55, 0.15)   (0.15, 0.45, 0.75, 0.35)
    (0.15, 0.25, 0.75, 0.35)   (0.15, 0.25, 0.55, 0.15)   (0.20, 0.45, 0.75, 0.35)
    (0.20, 0.25, 0.75, 0.35)   (0.24, 0.45, 0.75, 0.35)   (0.20, 0.25, 0.55, 0.15)

Each is solved at `Analyte` = 1.33, 1.34, 1.35 across the 16-point wavelength
grid.

## Suggested modelling target

`loss` spans nearly five orders of magnitude and is dominated by the resonance
peak. The associated article regresses log10(loss x 1e8) rather than `loss`.

Because rows within one geometry are a single wavelength sweep, they are not
independent. A random train/test split leaks a spectrum across the partition;
evaluate by leaving out whole geometries.

## Licence

CC BY 4.0. Please cite both the originating article and this dataset.
