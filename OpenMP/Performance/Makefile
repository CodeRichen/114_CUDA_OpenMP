CC = gcc
CFLAGS = -Wall -std=c99 -DVERBOSE
LIBS = -fopenmp -lm

.PHONY: all exe openmpexe
all: assign-uni assign-shared-omp assign-private-omp assign-heap-omp

%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)


run-assign:
	time ./assign-uni
	time ./assign-shared-omp
	time ./assign-private-omp
	time ./assign-heap-omp



