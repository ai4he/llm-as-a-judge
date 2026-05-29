# PRISMA review: LLM-as-a-Judge vs Human Annotation -- reproducible pipeline
PY := python3
.PHONY: all data figures bib paper search notebook clean veryclean

all: data figures bib paper        ## rebuild dataset, figures, bib, and compile the paper

search:                            ## (re)run the systematic searches (needs network + keys)
	$(PY) data/openalex_search.py
	$(PY) data/semanticscholar_search.py
	$(PY) data/filter_candidates.py
	$(PY) data/study2_openalex_search.py

data:                              ## build both study corpora + PRISMA counts
	$(PY) data/build_corpus.py
	$(PY) data/build_study2_corpus.py

figures: data                      ## regenerate all figures (PNG+PDF) and data-driven tables (both studies)
	$(PY) analysis/make_figures.py
	$(PY) analysis/make_study2_figures.py

bib: data                          ## (re)generate paper/references.bib
	$(PY) data/generate_bib.py

paper: figures bib                 ## compile the LaTeX paper (needs a TeX engine)
	cd paper && ( latexmk -pdf -interaction=nonstopmode main.tex \
	  || ( pdflatex -interaction=nonstopmode main.tex && bibtex main \
	       && pdflatex -interaction=nonstopmode main.tex \
	       && pdflatex -interaction=nonstopmode main.tex ) )

notebook: data                     ## execute the analysis notebook in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/llm_judge_review.ipynb

clean:                             ## remove LaTeX build artifacts
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.fls paper/*.fdb_latexmk paper/*.toc

veryclean: clean                   ## also remove generated figures/tables (kept under git otherwise)
	@echo "kept data/ and figures/ ; use with care"
