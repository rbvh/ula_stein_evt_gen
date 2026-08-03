MAIN = ula_stein_evt_gen
WORKSHOP = sim2science2026

.PHONY: all workshop clean distclean

all:
	latexmk -pdf $(MAIN).tex

workshop:
	latexmk -pdf $(WORKSHOP).tex

clean:
	latexmk -c $(MAIN).tex
	latexmk -c $(WORKSHOP).tex

distclean:
	latexmk -C $(MAIN).tex
	latexmk -C $(WORKSHOP).tex
