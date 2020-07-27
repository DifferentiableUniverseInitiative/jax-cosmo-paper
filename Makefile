

paper.pdf: paper.tex refs.bib figures/*
	pdflatex paper
	bibtex paper
	pdflatex paper
	pdflatex paper
