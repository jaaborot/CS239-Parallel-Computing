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

// write kernel function matmul_rec_shar

// write intermediate function for kernel function matmul_rec_shar

// write kernel function matmul_rec_glob
void matmul_rec_glob(float* A_device, float* B_device, float* C_device, int blockDim_x, int blockDim_y, int blockIdx_x, int blockIdx_y, int threadIdx_x, int threadIdx_y){
	printf(">>>>>>>> Executing kernel function matmul_rec_glob...");

}

// write intermediate function for kernel function matmul_rec_glob
int host_matmul_rec_glob(float* A_host, float* B_host, float* C_host, int n_, int k_, int m_){
	printf(">>>> Executing intermediate function host_matmul_rec_glob...\n");

	// set execution configuration parameters
	printf(">>>> Setting execution configuration parameters...\n");

	// allocate device memory
	printf(">>>> Allocating device memory...\n");

	// copy data from host to device
	printf(">>>> Copying data from host to device...\n");

	// execute kernel function matmul_rec_glob
	printf(">>>> Executing kernel function matmul_rec_glob...\n");

	// record execution time of kernel function
	printf(">>>> Recording execution time of kernel function matmul_rec_glob...\n");

	// copy data from device to host
	printf(">>>> Copying data from device to host\n");

	// free device memory
	printf(">>>> Freeing device memory\n");

	// return kernel execution time in milliseconds
	return 0;
}

int main(void) {
	puts("!!!Hello World!!!"); /* prints !!!Hello World!!! */

	int n, k, m, execution_count;
	time_t t;

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

	float A[n][k];
	float B[k][m];
	float C[n][m];
	float kernel_times[execution_count][2];

	printf("\nn = %d, k = %d, m = %d, execution count = %d\n\n", n, k, m, execution_count);
	for(int execution_counter = 0; execution_counter < execution_count; execution_counter++){

		printf("==== Program execution iteration #%d ====\n", execution_counter + 1);

		// randomly generate matrix A with dimension n x k
		printf(">> Generating random matrix A...\n");
		srand((unsigned) time(&t));
		for (int row = 0; row < n; row++) {
			for (int col = 0; col < k; col++) {
				A[row][col] = rand() % 101;
				printf("%f ", A[row][col]);
			}
			printf("\n");
		}

		// randomly generate matrix B with dimension k x m
		printf(">> Generating random matrix B...\n");
		srand((unsigned) time(&t));
		for (int row = 0; row < k; row++) {
			for (int col = 0; col < m; col++) {
				B[row][col] = rand() % 101;
				printf("%f ", B[row][col]);
			}
			printf("\n");
		}

		// call intermediate function for multiplying matrices A and B using global memory
		printf(">> Executing intermediate function for kernel function matmul_rec_glob...\n");
		kernel_times[execution_counter][0] = host_matmul_rec_glob((float*) A, (float*) B, (float*) C, n, k, m);

		// store execution time of kernel function matmul_rec_glob
		printf(">> Storing execution time of kernel function matmul_rec_glob\n");

		// clear device memory
		printf(">> Resetting device memory...\n");

		// call intermediate function for multiplying matrices A and B using shared memory
		printf(">> Executing intermediate function for kernel function matmul_rec_shar...\n");

		// store execution time of kernel function matmul_rec_shar
		printf(">> Storing execution time of kernel function matmul_rec_shar\n");

		// clear device memory
		printf(">> Resetting device memory...\n");

	}

	// write to CSV file the execution time of each kernel for current iteration
	printf(">> Writing execution time of each kernel into CSV file...\n\n");

	return EXIT_SUCCESS;
}
