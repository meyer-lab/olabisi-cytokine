SHELL := /bin/bash

.PHONY: clean test

flist = $(wildcard olabisi/figures/figure*.py)

all: $(patsubst olabisi/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: olabisi/figures/figure%.py
	@ mkdir -p ./output
	XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 poetry run fbuild $*

clean:
	rm -rf output