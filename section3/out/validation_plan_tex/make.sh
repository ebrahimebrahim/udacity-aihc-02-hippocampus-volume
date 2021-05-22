#!/bin/bash

set -e

pdflatex validation_plan.tex
bibtex validation_plan.aux
pdflatex validation_plan.tex
pdflatex validation_plan.tex

