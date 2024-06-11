# Makefile for converting a Jupyter notebook to HTML and PDF via Markdown and Pandoc

# Variables
NOTEBOOK = weakLight_seminar.ipynb
MARKDOWN_FILE = weakLight_seminar.md
HTML_FILE = weakLight_seminar.html
PDF_FILE = weakLight_seminar.pdf

# Default target
all: $(HTML_FILE) 

# Convert Jupyter notebook to Markdown
$(MARKDOWN_FILE): $(NOTEBOOK)
	jupyter nbconvert --to markdown $(NOTEBOOK) --no-input

# Convert Markdown to HTML using Pandoc
$(HTML_FILE): $(NOTEBOOK)
	jupyter nbconvert --to html $(NOTEBOOK) --no-input

# Convert Markdown to PDF using Pandoc
$(PDF_FILE): $(MARKDOWN_FILE)
	pandoc $(MARKDOWN_FILE) -o $(PDF_FILE) --pdf-engine=xelatex --standalone

# Clean up intermediate files
clean:
	rm -f $(MARKDOWN_FILE) $(HTML_FILE)

.PHONY: all clean
