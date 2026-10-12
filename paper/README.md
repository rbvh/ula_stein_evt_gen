# Paper Draft

This directory contains a single-file LaTeX paper framework:

- `ula_stein_evt_gen.tex`: manuscript body
- `sim2science2026.tex`: NeurIPS 2026 camera-ready workshop version
- `neurips_2026.sty`: official NeurIPS 2026 style file
- `checklist.tex`: completed official NeurIPS 2026 paper checklist, including all questions and guidelines
- `refs.bib`: bibliography
- `Makefile`: build helpers
- `latexmkrc`: local latexmk configuration

Both manuscripts include the VEGAS unweighting comparison and the clarification
that the transverse-momentum cut applies only for positive gluon multiplicity.
The camera-ready workshop main text occupies five pages. Acknowledgements and
references begin on page six, followed by the complete checklist and appendices.
See the repository README for the efficiency runner and saved measurements.

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

## Camera-ready submission

The workshop source uses `dblblindworkshop, final` and the title
`Sim2Science: ML with Imperfect Scientific Models`, as requested in the
camera-ready email. Author details and acknowledgements are visible, the code
link points to the public repository, and review line numbers are removed.

The style file was verified byte-for-byte against the
[official NeurIPS 2026 template](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
on 2026-10-12 and is unchanged. All 16 checklist questions and their guidelines
match that template; the instruction block has been removed and every answer
has a justification. The VEGAS caption and checklist identify errors as standard
errors over ten independent surveys of each fitted grid. The compute statement
covers all experiments in the Results section.

Run `make workshop` to rebuild `sim2science2026.pdf`. The checked camera-ready
PDF has 19 pages in total: five main-text pages, acknowledgements and references
on pages 6--10, the checklist on pages 11--17, and appendices on pages 18--19.
The build has no layout or reference warnings and all fonts are embedded.

Upload `sim2science2026.pdf` in OpenReview via **Edit > Camera Ready Revision**
by **October 30, 2026, AoE**, according to the program chairs' email. This is a
manual upload; preparing the local files does not submit a revision.
