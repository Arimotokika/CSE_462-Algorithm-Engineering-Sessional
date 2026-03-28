# Weapon-Target Assignment Problem - Presentation

## Files

- `presentation.tex` - Main combined LaTeX beamer presentation
- `iron_dome_example.gif` - Animated GIF showing Iron Dome concept

## Compilation Instructions

### Option 1: Using pdfLaTeX (Static Image)

If you want to use a static image instead of animation:

1. Convert the GIF to PNG using any image converter
2. Rename to `iron_dome_example.png`
3. Compile with: `pdflatex presentation.tex`

### Option 2: Using animate package (Animated PDF)

To include the animated GIF in PDF:

1. Convert GIF to individual frames: Use `ffmpeg` or online tools
   ```
   ffmpeg -i iron_dome_example.gif -vsync 0 frame%03d.png
   ```
2. Use the `animate` package in LaTeX (already included)
3. Replace the `\includegraphics` with `\animategraphics`

### Option 3: Online LaTeX Editors

- Upload to Overleaf
- Convert GIF to PNG first for compatibility

## Presentation Structure (15 minutes total)

| Section | Topic                                 | Time    |
| ------- | ------------------------------------- | ------- |
| 1       | Problem Definition                    | 2.5 min |
| 2       | NP-Hardness Proofs                    | 2.5 min |
| 3       | Algorithm Survey                      | 2.5 min |
| 4       | Implementation Algorithms (MMR & ACO) | 2.5 min |
| 5       | Experimental Design                   | 2.5 min |
| 6       | Applications                          | 2.5 min |

## Content Sources

Content compiled from team member files:

- `definition.tex` - Problem definition and formulation
- `np-hard.tex` - Polynomial-time reductions
- `algorithm-table.tex` - Algorithm survey
- `proposed algo/mmr.tex` - MMR algorithm details
- `proposed algo/antcolony.tex` - ACO algorithm details
- `experiment.tex` - Experimental design
- `application.tex` - Real-world applications

## Quick Start

```bash
pdflatex presentation.tex
pdflatex presentation.tex  # Run twice for TOC/references
```

## Notes

- Uses Madrid beamer theme with seahorse color theme
- 16:9 aspect ratio (aspectratio=169)
- Professional blue color scheme
- Clean, minimal design for academic presentation
