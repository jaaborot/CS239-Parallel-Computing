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

#define TILE_WIDTH 32

__global__ void matmul_rec_shar(float* A_device, float* B_device, float* C_device, int n_rows, int k_cols_rows, int m_cols, int tile_width, int phase_counter){
//	printf(">>>>>>>> Executing kernel function matmul_rec_shar...\n");

	// block scope: shared sub-matrices within a block
	__shared__ float A_device_shared[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_device_shared[TILE_WIDTH][TILE_WIDTH];
	int A_block_start_element = (phase_counter * tile_width) + (blockIdx.y * (tile_width * k_cols_rows));
	int B_block_start_element = (blockIdx.x * tile_width) + (phase_counter * (tile_width * m_cols));
	/*
	 * Load corresponding element of A to shared memory
	 */
	// thread scope: only visible within each thread
	int A_row = blockIdx.y * tile_width + threadIdx.y;
	int A_col = phase_counter * tile_width + threadIdx.x;
	if(A_row < n_rows && A_col < k_cols_rows){
		// fetch tile data from global memory to block's shared memory
//		A_device_shared[threadIdx.y * tile_width + threadIdx.x] = A_device[A_block_start_element + (threadIdx.y * k_cols_rows) + threadIdx.x];
		A_device_shared[threadIdx.y][threadIdx.x] = A_device[A_block_start_element + (threadIdx.y * k_cols_rows) + threadIdx.x];
//		printf("if: A_device_shared[%d][%d] = A_device_shared[%d + (%d * %d) + %d] = A_device[%d] = %f \n", threadIdx.y, threadIdx.x, A_block_start_element, threadIdx.y, k_cols_rows, threadIdx.x, A_block_start_element + (threadIdx.y * k_cols_rows) + threadIdx.x, A_device[A_block_start_element + (threadIdx.y * k_cols_rows) + threadIdx.x]);
	} else {
//		A_device_shared[threadIdx.y * tile_width + threadIdx.x] = 0.0;
		A_device_shared[threadIdx.y][threadIdx.x] = 0.0;
//		printf("else: A_device_shared[%d][%d] = A_device_shared[%d][%d] = 0.0 \n", threadIdx.y, threadIdx.x, threadIdx.y * tile_width, threadIdx.x);
	}

	/* The number of columns in A is equal to the number of rows in B, k_cols_rows,
	 * and so we flip the propagation of block in B from column priority (propagation
	 * of block ID through all columns and then to the next row) to row priority (propagation
	 * of block ID through all rows and then to the next column).
	 */
	/*
	 * Load corresponding element of B to shared memory
	 */
	// thread scope: only visible within each thread
	int B_row = phase_counter * tile_width + threadIdx.y;
	int B_col = blockIdx.x * tile_width + threadIdx.x;

	if(B_row < k_cols_rows && B_col < m_cols){
		// fetch tile data from global memory to block's shared memory
//		B_device_shared[threadIdx.y * tile_width + threadIdx.x] = B_device[B_block_start_element + (threadIdx.y * m_cols) + threadIdx.x];
		B_device_shared[threadIdx.y][threadIdx.x] = B_device[B_block_start_element + (threadIdx.y * m_cols) + threadIdx.x];
//		printf("if: B_device_shared[%d][%d] = B_device[%d + (%d * %d) + %d] = B_device[%d] = %f \n", threadIdx.y, threadIdx.x, B_block_start_element, threadIdx.y, m_cols, threadIdx.x, B_block_start_element + (threadIdx.y * m_cols) + threadIdx.x, B_device[B_block_start_element + (threadIdx.y * m_cols) + threadIdx.x]);
	}else{
//		B_device_shared[threadIdx.y * tile_width + threadIdx.x] = 0.0;
		B_device_shared[threadIdx.y][threadIdx.x] = 0.0;
//		printf("else: B_device_shared[%d][%d] = B_device_shared[%d][%d] = 0.0 \n", threadIdx.y, threadIdx.x, threadIdx.y * tile_width, threadIdx.x);
	}
	__syncthreads(); // wait for all threads to finish loading their corresponding sub-matrices prior to computation of corresponding sub-matrices in C

	/*
	 * Each thread of the block will load the elements in the corresponding blocks in A and B to the shared memory of the thread's block.
	 * We can then proceed with the computation after loading the elements in corresponding blocks in A and B in the shared block.
	 */
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if(row < n_rows && col < m_cols){
		C_device[row * m_cols + col] = 0.0;
		for(int counter = 0; counter < tile_width; counter++){
//			printf("C_device[%d][%d] = C_device[%d * %d + %d] += A_device_shared[%d][%d] * B_device_shared[%d][%d] = %f + (%f * %f) = %f\n", row, col, row, m_cols, col, threadIdx.y, counter, counter, threadIdx.y, C_device[row * m_cols + col], A_device_shared[threadIdx.y][counter], B_device_shared[counter][threadIdx.x], C_device[row * m_cols + col] + A_device_shared[threadIdx.y][counter] * B_device_shared[counter][threadIdx.x]);
			C_device[row * m_cols + col] += A_device_shared[threadIdx.y][counter] * B_device_shared[counter][threadIdx.x];
		}
	}
}

// write intermediate function for kernel function matmul_rec_shar
int host_matmul_rec_shar(float* A_host, float* B_host, float* C_host, int n_rows, int k_cols_rows, int m_cols){
	printf(">>>> Executing intermediate function host_matmul_rec_shar...\n");
	printf(">>>> Setting execution configuration parameters...\n");
	// set execution configuration parameters
	// Reference: CUDA by Example, Sanders et al., 2011
	cudaDeviceProp prop;
	int cuda_device_count; // number of CUDA-enabled devices
	int gridDim_x; /* number of blocks in the grid's x dimension */
	int gridDim_y; /* number of blocks in the grid's y dimension */
	int blockDim_x; /* number of threads in a block's x dimension */
	int blockDim_y; /* number of threads in a block's y dimension */

	cudaGetDeviceCount(&cuda_device_count);
	if(cuda_device_count > 0){
		cudaGetDeviceProperties(&prop, 0);
		printf(">> Device properties ==\n");
		printf("CUDA-enabled device count: %d\n", cuda_device_count);
		printf("Maximum number of threads per 2D block: %d\n", prop.maxThreadsPerBlock); // maximum number of threads which can be alloted to a single block
		printf("Maximum number of threads in x, y and z dimension of a 3D block: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); // maximum number of threads which can be alloted along the x, y and z dimension of each block
		printf("Maximum number of threads in x, y and z dimension of a 3D grid: %d, %d, %d\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		/* The programmer may design each block with varying number of threads along the x and y dimension as long as
		 * the total number of threads in each block does not exceed maxThreadsPerBlock. For example, if maxThreadsPerBlock = 1024,
		 * the programmer may design the block to be a 32 rows x 32 columns block, 1 row by 1024 columns block, 1024 row by 1 column block, etc.
		 */

		/* maxGridSize refers to the maximum number of threads a grid can hold along the x (maxGridSize[0]), y (maxGridSize[1]) and z (maxGridSize[2]) dimensions.
		 * If the programmer wants to compute for the output matrix in a single step, then the maximum dimensions of the input matrices will be limited by the
		 * maximum number of threads a grid can have, i.e. maxGridSize[0] > dimension and maxGridSize[1] > dimension.
		 */
		if(prop.maxGridSize[0] > m_cols && prop.maxGridSize[1] > n_rows){ // if matrix size less than number of threads in a grid

			// determine block dimension of the grid
			gridDim_x = (int)ceil((float) m_cols / prop.warpSize); // necessary number of blocks in x dimension in the grid
			gridDim_y = (int)ceil((float) n_rows / prop.warpSize); // necessary number of blocks in y dimension in the grid

			// determine thread dimension of each block
			blockDim_x = prop.warpSize;
			blockDim_y = prop.warpSize;

			printf("warp size: %d\n", prop.warpSize);
			printf("Number of blocks along x dim in grid: %d\n", gridDim_x);
			printf("Number of blocks along y dim in grid: %d\n", gridDim_y);
			printf("Number of threads along x dim in block: %d\n", blockDim_x);
			printf("Number of threads along y dim in block: %d\n", blockDim_y);

		}else{

			// exit with max-thread-limit-reached exception
			printf(">> Number of matrix elements exceeded maximum number of threads available.\n");
			return 1;

		}
	}else{
		printf(">> No CUDA-enabled device found.\n");
		return 1;
	}

	dim3 blocks_per_grid(gridDim_x, gridDim_y, 1); // how many blocks to utilize in the grid
	dim3 threads_per_block(blockDim_x, blockDim_y, 1); // how many threads to utilize per block in the grid

	// configure execution parameters
	int size = n_rows * m_cols * sizeof(float);
	int tile_width = blockDim_x;
	float* A_device;
	float* B_device;
	float* C_device;

	// allocate device memory
	printf(">>>> Allocating device memory...\n");
	cudaMalloc((void**) &A_device, n_rows * k_cols_rows * sizeof(float));
	cudaMalloc((void**) &B_device, k_cols_rows * m_cols * sizeof(float));
	cudaMalloc((void**) &C_device, n_rows * m_cols * sizeof(float));

	// copy data from host to device
	printf(">>>> Copying data from host to device...\n");
	cudaMemcpy(A_device, A_host, n_rows * k_cols_rows * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, k_cols_rows * m_cols * sizeof(float), cudaMemcpyHostToDevice);

	// execute kernel function matmul_rec_glob
	printf(">>>> Executing kernel function matmul_rec_shar...\n");
	// define CUDA event recorder
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	// execute kernel function
	int phase_max = (int) ceil((float) k_cols_rows / prop.warpSize); /* maximum number of phases */
	printf("phase_max = %d\n", phase_max);
	for(int phase_counter = 0; phase_counter < phase_max; phase_counter++){
		matmul_rec_shar<<<blocks_per_grid, threads_per_block>>>(A_device, B_device, C_device, n_rows, k_cols_rows, m_cols, tile_width, phase_counter);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// compute for elapsed time (execution time of kernel)
	float runtime = 0;
	cudaEventElapsedTime(&runtime, start, stop);
	cudaDeviceSynchronize();

	// copy data from device to host
	printf(">>>> Copying data from device to host\n");
	cudaMemcpy(C_host, C_device, n_rows * m_cols * sizeof(float), cudaMemcpyDeviceToHost);

	// free device memory
	printf(">>>> Freeing device memory\n");
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);

	// return kernel execution time in milliseconds
	return (int)ceil(runtime);
}

// kernel function matmul_rec_glob
__global__ void matmul_rec_glob(float* A_device, float* B_device, float* C_device, int n_rows, int k_cols_rows, int m_cols){
//	printf(">>>>>>>> Executing kernel function matmul_rec_glob...\n");
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if(row < n_rows && col < m_cols){
		C_device[row * m_cols + col] = 0.0;
		for(int cols_rows_counter = 0; cols_rows_counter < k_cols_rows; cols_rows_counter++){
//			printf("C[%d] = C[%d * %d + %d] + A[ %d * %d + %d] * B[%d * %d + %d] = C[%d] + A[%d] * B[%d] = %f + %f * %f\n",
//					(row * m_cols + col),
//					row,
//					m_cols,
//					col,
//					row,
//					k_cols_rows,
//					cols_rows_counter,
//					cols_rows_counter,
//					m_cols,
//					col,
//					(row * m_cols + col),
//					(row * k_cols_rows + cols_rows_counter),
//					(cols_rows_counter * m_cols + col),
//					C_device[row * m_cols + col],
//					A_device[row * k_cols_rows + cols_rows_counter],
//					B_device[cols_rows_counter * m_cols + col]);
			C_device[row * m_cols + col] += A_device[row * k_cols_rows + cols_rows_counter] * B_device[cols_rows_counter * m_cols + col];
		}
		__syncthreads();
//		printf("C[%d * %d + %d] = C[%d]: %f \n", row, m_cols, col, row * m_cols + col, C_device[row * m_cols + col]);
	}
}

/*
 * Intermediate function for kernel function matmul_rec_glob.
 */
int host_matmul_rec_glob(float* A_host, float* B_host, float* C_host, int n_rows, int k_cols_rows, int m_cols){
	printf(">>>> Executing intermediate function host_matmul_rec_glob...\n");
	printf(">>>> Setting execution configuration parameters...\n");
	// set execution configuration parameters
	// Reference: CUDA by Example, Sanders et al., 2011
	cudaDeviceProp prop;
	int cuda_device_count; // number of CUDA-enabled devices
	int gridDim_x; /* number of blocks in the grid's x dimension */
	int gridDim_y; /* number of blocks in the grid's y dimension */
	int blockDim_x; /* number of threads in a block's x dimension */
	int blockDim_y; /* number of threads in a block's y dimension */

	cudaGetDeviceCount(&cuda_device_count);
	if(cuda_device_count > 0){
		cudaGetDeviceProperties(&prop, 0);
		printf(">> Device properties ==\n");
		printf("CUDA-enabled device count: %d\n", cuda_device_count);
		printf("Maximum number of threads per 2D block: %d\n", prop.maxThreadsPerBlock); // maximum number of threads which can be alloted to a single block
		printf("Maximum number of threads in x, y and z dimension of a 3D block: %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); // maximum number of threads which can be alloted along the x, y and z dimension of each block
		printf("Maximum number of threads in x, y and z dimension of a 3D grid: %d, %d, %d\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		/* The programmer may design each block with varying number of threads along the x and y dimension as long as
		 * the total number of threads in each block does not exceed maxThreadsPerBlock. For example, if maxThreadsPerBlock = 1024,
		 * the programmer may design the block to be a 32 rows x 32 columns block, 1 row by 1024 columns block, 1024 row by 1 column block, etc.
		 */

		/* maxGridSize refers to the maximum number of threads a grid can hold along the x (maxGridSize[0]), y (maxGridSize[1]) and z (maxGridSize[2]) dimensions.
		 * If the programmer wants to compute for the output matrix in a single step, then the maximum dimensions of the input matrices will be limited by the
		 * maximum number of threads a grid can have, i.e. maxGridSize[0] > dimension and maxGridSize[1] > dimension.
		 */
		if(prop.maxGridSize[0] > m_cols && prop.maxGridSize[1] > n_rows){ // if matrix size less than number of threads in a grid

			// determine block dimension of the grid
			gridDim_x = (int)ceil((float) m_cols / prop.warpSize); // necessary number of blocks in x dimension in the grid
			gridDim_y = (int)ceil((float) n_rows / prop.warpSize); // necessary number of blocks in y dimension in the grid

			// determine thread dimension of each block
			blockDim_x = prop.warpSize;
			blockDim_y = prop.warpSize;

			printf("warp size: %d\n", prop.warpSize);
			printf("Number of blocks along x dim in grid: %d\n", gridDim_x);
			printf("Number of blocks along y dim in grid: %d\n", gridDim_y);
			printf("Number of threads along x dim in block: %d\n", blockDim_x);
			printf("Number of threads along y dim in block: %d\n", blockDim_y);

		}else{

			// exit with max-thread-limit-reached exception
			printf(">> Number of matrix elements exceeded maximum number of threads available.\n");
			return 1;

		}
	}else{
		printf(">> No CUDA-enabled device found.\n");
		return 1;
	}

	dim3 blocks_per_grid(gridDim_x, gridDim_y, 1); // how many blocks to utilize in the grid
	dim3 threads_per_block(blockDim_x, blockDim_y, 1); // how many threads to utilize per block in the grid

	// configure execution parameters
	int size = n_rows * m_cols * sizeof(float);
	float* A_device;
	float* B_device;
	float* C_device;

	// allocate device memory
	printf(">>>> Allocating device memory...\n");
	cudaMalloc((void**) &A_device, n_rows * k_cols_rows * sizeof(float));
	cudaMalloc((void**) &B_device, k_cols_rows * m_cols * sizeof(float));
	cudaMalloc((void**) &C_device, n_rows * m_cols * sizeof(float));

	// copy data from host to device
	printf(">>>> Copying data from host to device...\n");
	cudaMemcpy(A_device, A_host, n_rows * k_cols_rows * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, k_cols_rows * m_cols * sizeof(float), cudaMemcpyHostToDevice);

	// execute kernel function matmul_rec_glob
	printf(">>>> Executing kernel function matmul_rec_glob...\n");
	// define CUDA event recorder
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	// execute kernel function
	matmul_rec_glob<<<blocks_per_grid, threads_per_block>>>(A_device, B_device, C_device, n_rows, k_cols_rows, m_cols);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// compute for elapsed time (execution time of kernel)
	float runtime = 0;
	cudaEventElapsedTime(&runtime, start, stop);
	cudaDeviceSynchronize();

	// copy data from device to host
	printf(">>>> Copying data from device to host\n");
	cudaMemcpy(C_host, C_device, n_rows * m_cols * sizeof(float), cudaMemcpyDeviceToHost);

	// free device memory
	printf(">>>> Freeing device memory\n");
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);

	// return kernel execution time in milliseconds
	return (int)ceil(runtime);
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
	float C_glob[n][m];
	float C_shar[n][m];
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
				B[row][col] = rand() % 100;
				printf("%f ", B[row][col]);
			}
			printf("\n");
		}

		// initializing matrix C
		for (int row = 0; row < n; row++) {
			for (int col = 0; col < m; col++) {
				C_glob[row][col] = 0.0;
				C_shar[row][col] = 0.0;
			}
		}

		// call intermediate function for multiplying matrices A and B using global memory
		printf(">> Executing intermediate function for kernel function matmul_rec_glob...\n");
		kernel_times[execution_counter][0] = host_matmul_rec_glob((float*) A, (float*) B, (float*) C_glob, n, k, m);
		printf("C_glob:\n");
		for(int row = 0; row < n; row++){
			for(int col = 0; col < m; col++){
				printf("%f ", C_glob[row][col]);
			}
			printf("\n");
		}

		// clear device memory
		printf(">> Resetting device memory...\n");

		// call intermediate function for multiplying matrices A and B using shared memory
		printf(">> Executing intermediate function for kernel function matmul_rec_shar...\n");
		kernel_times[execution_counter][0] = host_matmul_rec_shar((float*) A, (float*) B, (float*) C_shar, n, k, m);
		printf("C_shar:\n");
		for(int row = 0; row < n; row++){
			for(int col = 0; col < m; col++){
				printf("%f ", C_shar[row][col]);
			}
			printf("\n");
		}

		// check for mismatch between to output matrices
		printf(">> Checking for any mismatches between C_glob and C_shar...\n");
		for(int row = 0; row < n; row++){
			for(int col = 0; col < m; col++){
				if(C_glob[row][col] != C_shar[row][col]){
					printf("A mismatch occurs between C_glob and C_shar in element at coordinate (%d,%d).\n", row, col);
					return 1;
				}
			}
		}

		// store execution time of kernel function matmul_rec_shar
		printf(">> Storing execution time of kernel function matmul_rec_shar\n");

		// clear device memory
		printf(">> Resetting device memory...\n");

	}

	// write to CSV file the execution time of each kernel for current iteration
	printf(">> Writing execution time of each kernel into CSV file...\n\n");

	return EXIT_SUCCESS;
}
