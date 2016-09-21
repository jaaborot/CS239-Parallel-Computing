/*
 ============================================================================
 Name        : Matrix_Multiplication.c
 Author      : Jeffrey A. Aborot
 Version     :
 Copyright   : This work is open-source but requires proper citation if to be used.
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(void) {
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	int n, k, m, execution_count;

	// user input for matrix dimensions n, k and m
	printf("\nSpecify matrix dimensions...\n");
	printf(">> Matrix A row count: ");
	scanf("%d", &n);
	printf(">> Matrix A col / Matrix B row count: ");
	scanf("%d", &k);
	printf(">> Matrix B col count: ");
	scanf("%d", &m);

	// user input for number of iteration of program execution
	printf("\nSpecify number of program execution iterations: ");
	scanf("%d", &execution_count);

	printf("\nn = %d, k = %d, m = %d, execution count = %d\n\n", n, k, m, execution_count);
	for(int execution_counter = 0; execution_counter < execution_count; execution_counter++){

		printf("==== Program execution iteration #%d ====\n", execution_counter + 1);

		// randomly generate matrix A with dimension n x k
		printf(">> Generating random matrix A...\n");

		// randomly generate matrix B with dimension k x m
		printf(">> Generating random matrix B...\n");

		// call intermediate function for multiplying matrices A and B using global memory
		printf(">> Executing kernel function matmul_rec_glob...\n");

		// record execution time of kernel function
		printf(">> Recording execution time of kernel function matmul_rec_glob...\n");

		// clear device memory
		printf(">> Resetting device memory...\n");

		// call intermediate function for multiplying matrices A and B using shared memory
		printf(">> Executing kernel function matmul_rec_shar...\n");

		// record execution time of kernel function
		printf(">> Recording execution time of kernel function matmul_rec_shar...\n");

		// clear device memory
		printf(">> Resetting device memory...\n");

		// write to CSV file the execution time of each kernel for current iteration
		printf(">> Writing execution time of each kernel into CSV file...\n\n");

	}

	return EXIT_SUCCESS;
}
