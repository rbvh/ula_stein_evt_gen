MAIN = main

.PHONY: all clean distclean

all:
	latexmk -pdf $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

distclean:
	latexmk -C $(MAIN).tex
