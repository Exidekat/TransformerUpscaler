#!/usr/bin/env bash

# Compile LaTeX source to PDF
pdflatex architecture.tex

# Convert PDF to PNG, flattening the background to white
magick -density 300 architecture.pdf \
        -quality 90 \
        architecture_temp.png

# Adjust logo size
magick logo.png -scale 500 logo_temp.png


# Append logo to the right side of the diagram
magick architecture_temp.png logo_temp.png  \
       -gravity northeast -geometry +80+20  -composite \
       -flatten \
       architecture.png

# Clean up intermediate files
rm logo_temp.png architecture_temp.png architecture.aux architecture.log architecture.pdf
