all: paper.pdf

paper.pdf: *.tex 
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm -f paper.ilg paper.aux paper.log paper.dvi paper.idx paper.toc paper.lof paper.lot paper.out 
	rm -f paper.blg paper.bbl paper~
	rm -f paper.ps
	rm -f paper.pdf
