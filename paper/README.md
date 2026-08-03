# Paper Draft

This directory contains a single-file LaTeX paper framework:

- `ula_stein_evt_gen.tex`: manuscript body
- `sim2science2026.tex`: NeurIPS 2026 double-blind workshop version
- `neurips_2026.sty`: official NeurIPS 2026 style file
- `checklist.tex`: official NeurIPS 2026 paper checklist (currently TODO)
- `refs.bib`: bibliography
- `Makefile`: build helpers
- `latexmkrc`: local latexmk configuration

Build with:

```bash
make
```

Build the Sim2Science workshop version with:

```bash
make workshop
```

Clean generated LaTeX intermediates with:

```bash
make clean
```
