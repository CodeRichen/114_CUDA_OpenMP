CC = gcc
CFLAGS = -std=c99

all: 	for-dynamic-chunk-1-omp \
	for-dynamic-chunk-1-reverse-omp \
	for-dynamic-chunk-2-omp \
	for-dynamic-chunk-4-reverse-omp \
	for-guided-chunk-1-omp \
	for-guided-chunk-1-reverse-omp \
	for-guided-chunk-4-reverse-omp \
	for-runtime-omp \
	for-static-chunk-4-omp \
	for-static-chunk-1-omp \
	for-static-chunk-1-reverse-omp
	


%-omp: %.c
	$(CC) -fopenmp $(CFLAGS) $< -lm -o $@ 

%-uni: %.c
	$(CC) $(CFLAGS) $< -lm -o $@

%.pdf: %.tex
	pdflatex $<


clean:
	rm *-uni *-omp *.aux *.log *.nav *.snm *.toc *.vrb *.out *~

