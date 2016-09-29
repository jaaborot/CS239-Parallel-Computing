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
void matmul_rec_shar(float* A_device, float* A_device_shared, float* B_device, float* B_device_shared, float* C_device, int A_block_start_element, int B_block_start_element, int tile_width, int n_rows, int k_cols_rows, int m_cols, int blockDim_x, int blockDim_y, int blockIdx_x, int blockIdx_y, int threadIdx_x, int threadIdx_y, int phase_counter){
//	printf(">>>>>>>> Executing kernel function matmul_rec_shar...\n");

	int A_row = blockIdx_y * tile_width + threadIdx_y;
	int A_col = phase_counter * tile_width + threadIdx_x;

	if(A_row < n_rows && A_col < k_cols_rows){
//		printf("A if: row: %d < n_rows: %d && col: %d < k_cols_rows: %d\n", A_row, n_rows, A_col, k_cols_rows);
		// fetch tile data from global memory to block's shared memory
		A_device_shared[threadIdx_y * tile_width + threadIdx_x] = A_device[A_block_start_element + (threadIdx_y * k_cols_rows) + threadIdx_x];
	} else {
//		printf("A else: row: %d < n_rows: %d && col: %d < k_cols_rows: %d\n", A_row, n_rows, A_col, k_cols_rows);
		A_device_shared[threadIdx_y * tile_width + threadIdx_x] = 0.0;
	}

	/* The number of columns in A is equal to the number of rows in B, k_cols_rows,
	 * and so we flip the propagation of block in B from column priority (propagation
	 * of block ID through all columns and then to the next row) to row priority (propagation
	 * of block ID through all rows and then to the next column).
	 */
	int B_row = phase_counter * tile_width + threadIdx_y;
	int B_col = blockIdx_x * tile_width + threadIdx_x;

	if(B_row < k_cols_rows && B_col < m_cols){
//		printf("B if: row: %d < k_cols_rows: %d && col: %d < m_rows: %d\n", B_row, k_cols_rows, B_col, m_cols);
		B_device_shared[threadIdx_y * tile_width + threadIdx_x] = B_device[B_block_start_element + (threadIdx_y * m_cols) + threadIdx_x];
	}else{
//		printf("B else: row: %d < k_cols_rows: %d && col: %d < m_rows: %d\n", B_row, k_cols_rows, B_col, m_cols);
		B_device_shared[threadIdx_y * tile_width + threadIdx_x] = 0.0;
	}
	// synchronize threads here using __syncthreads();

	/* Each thread of the block will load the elements in the corresponding blocks in A and B to the shared memory of the thread's block.
	 * We can then proceed with the computation after loading the elements in corresponding blocks in A and B in the shared block.
	 */
//	int row = blockDim_y * blockIdx_y + threadIdx_y;
//	int col = blockDim_x * blockIdx_x + threadIdx_x;
//	if(row < n_rows && col < m_cols){
//		for(int counter = 0; counter < tile_width; counter++){
//			printf("C_device[%d,%d] = C_device[%d * %d + %d] += A_device_shared[%d][%d] * B_device_shared[%d][%d] = %f + (%f * %f) = %f\n", row, col, row, m_cols, col, threadIdx_y, counter, counter, threadIdx_y, C_device[row * m_cols + col], A_device_shared[threadIdx_y][counter], B_device_shared[counter][threadIdx_x], C_device[row * m_cols + col] + A_device_shared[threadIdx_y][counter] * B_device_shared[counter][threadIdx_x]);
//			C_device[row * m_cols + col] += A_device_shared[threadIdx_y][counter] * B_device_shared[counter][threadIdx_x];
//		}
//	}
}

// write intermediate function for kernel function matmul_rec_shar
int host_matmul_rec_shar(float* A_host, float* B_host, float* C_host, int n_rows, int k_cols_rows, int m_cols){
	printf(">>>> Executing intermediate function host_matmul_rec_shar...\n");

	// set execution configuration parameters
	printf(">>>> Setting execution configuration parameters...\n");
	int warp_size = 32;
	int gridDim_x = (int)ceil((float) m_cols / warp_size); /* number of blocks in the grid's x dimension */
	int gridDim_y = (int)ceil((float) n_rows / warp_size); /* number of blocks in the grid's y dimension */
	int blockDim_x = warp_size; /* number of threads in a block's x dimension */
	int blockDim_y = warp_size; /* number of threads in a block's y dimension */
	int phase_max = (int) ceil((float) k_cols_rows / warp_size); /* maximum number of phases */

	printf("warp size: %d\n", warp_size);
	printf("Number of blocks along x dim in grid: %d\n", gridDim_x);
	printf("Number of blocks along y dim in grid: %d\n", gridDim_y);
	printf("Number of threads along x dim in block: %d\n", blockDim_x);
	printf("Number of threads along y dim in block: %d\n", blockDim_y);

	// allocate device memory
	printf(">>>> Allocating device memory...\n");

	// copy data from host to device
	printf(">>>> Copying data from host to device...\n");

	// execute kernel function matmul_rec_glob
	printf(">>>> Executing kernel function matmul_rec_shar...\n");
	/* variables in device global memory
	 */
	float* A_device = A_host;
	float* B_device = B_host;
	float* C_device = C_host;
	int tile_width = blockDim_x;

	for(int phase_counter = 0; phase_counter < phase_max; phase_counter++){
		for(int blockIdx_y = 0; blockIdx_y < gridDim_y; blockIdx_y++){
			for(int blockIdx_x = 0; blockIdx_x < gridDim_x; blockIdx_x++){ // for each block in the grid

				/* variables in device shared memory (to be declared as __shared__)
				 * tile_width = blockDim_x = blockDim_y = warp_size
				 */
				float A_device_shared[tile_width][tile_width];
				float B_device_shared[tile_width][tile_width];

				// load all subset data from current tile in input matrices A and B
//				printf(">>>> Loading data from A[%d,%d]-A[%d,%d]-A[%d,%d]-A[%d,%d] and B[%d,%d]-B[%d,%d]-B[%d,%d]-B[%d,%d] into shared memory...\n",
//						(blockDim_y * blockIdx_y), (blockDim_x * blockIdx_x),
//						(blockDim_y * blockIdx_y), (blockDim_x * blockIdx_x + (tile_width - 1)),
//						(blockDim_y * blockIdx_y + (tile_width - 1)), (blockDim_x * blockIdx_x),
//						(blockDim_y * blockIdx_y + (tile_width - 1)), (blockDim_x * blockIdx_x + (tile_width - 1)),
//						(blockDim_y * blockIdx_y), (blockDim_x * blockIdx_x),
//						(blockDim_y * blockIdx_y), (blockDim_x * blockIdx_x + (tile_width - 1)),
//						(blockDim_y * blockIdx_y + (tile_width - 1)), (blockDim_x * blockIdx_x),
//						(blockDim_y * blockIdx_y + (tile_width - 1)), (blockDim_x * blockIdx_x + (tile_width - 1))
//					  );


//				printf("C block [%d,%d], Phase #%d...\n", blockIdx_x, blockIdx_y, phase_counter);
				int A_block_start_element = (phase_counter * tile_width) + (blockIdx_y * (tile_width * k_cols_rows));
				int B_block_start_element = (blockIdx_x * tile_width) + (phase_counter * (tile_width * m_cols));
//				printf("blockIdx_y: %d\n", blockIdx_y);
//				printf("blockIdx_x: %d\n", blockIdx_x);
//				printf("phase_counter: %d\n", phase_counter);
//				printf("Matrix A start element: %d\n", A_block_start_element);
//				printf("Matrix B start element: %d\n", B_block_start_element);

				for(int threadIdx_y = 0; threadIdx_y < tile_width; threadIdx_y++){
					for(int threadIdx_x = 0; threadIdx_x < tile_width; threadIdx_x++){

						matmul_rec_shar((float*) A_device, (float*) A_device_shared, (float*) B_device, (float*) B_device_shared, (float*) C_device, A_block_start_element, B_block_start_element, tile_width, n_rows, k_cols_rows, m_cols, blockDim_x, blockDim_y, blockIdx_x, blockIdx_y, threadIdx_x, threadIdx_y, phase_counter);
//						int A_row = blockIdx_y * tile_width + threadIdx_y;
//						int A_col = phase_counter * tile_width + threadIdx_x;
//
//						if(A_row < n_rows && A_col < k_cols_rows){
//							printf("A if: row: %d < n_rows: %d && col: %d < k_cols_rows: %d\n", A_row, n_rows, A_col, k_cols_rows);
//							// fetch tile data from global memory to block's shared memory
//							A_device_shared[threadIdx_y][threadIdx_x] = A_device[A_block_start_element + (threadIdx_y * k_cols_rows) + threadIdx_x];
//						} else {
//							printf("A else: row: %d < n_rows: %d && col: %d < k_cols_rows: %d\n", A_row, n_rows, A_col, k_cols_rows);
//							A_device_shared[threadIdx_y][threadIdx_x] = 0.0;
//						}
//
//						/* The number of columns in A is equal to the number of rows in B, k_cols_rows,
//						 * and so we flip the propagation of block in B from column priority (propagation
//						 * of block ID through all columns and then to the next row) to row priority (propagation
//						 * of block ID through all rows and then to the next column).
//						 */
//						int B_row = phase_counter * tile_width + threadIdx_y;
//						int B_col = blockIdx_x * tile_width + threadIdx_x;
//
//						if(B_row < k_cols_rows && B_col < m_cols){
//							printf("B if: row: %d < k_cols_rows: %d && col: %d < m_rows: %d\n", B_row, k_cols_rows, B_col, m_cols);
//							B_device_shared[threadIdx_y][threadIdx_x] = B_device[B_block_start_element + (threadIdx_y * m_cols) + threadIdx_x];
//						}else{
//							printf("B else: row: %d < k_cols_rows: %d && col: %d < m_rows: %d\n", B_row, k_cols_rows, B_col, m_cols);
//							B_device_shared[threadIdx_y][threadIdx_x] = 0.0;
//						}
					}
				}

//				printf("A_device_shared:\n");
//				for(int threadIdx_y = 0; threadIdx_y < blockDim_y; threadIdx_y++){
//					for(int threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++){
//						printf("%f ", (float) A_device_shared[threadIdx_y][threadIdx_x]);
//					}
//					printf("\n");
//				}
//				printf("B_device_shared:\n");
//				for(int threadIdx_y = 0; threadIdx_y < blockDim_y; threadIdx_y++){
//					for(int threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++){
//						printf("%f ", (float) B_device_shared[threadIdx_y][threadIdx_x]);
//					}
//					printf("\n");
//				}

				// compute using loaded subset data on current tile of input matrices A and B
//				printf(">>>> Computing subset of output matrix using data in block\n");
				for(int threadIdx_y = 0; threadIdx_y < tile_width; threadIdx_y++){
					for(int threadIdx_x = 0; threadIdx_x < tile_width; threadIdx_x++){

						int row = blockDim_y * blockIdx_y + threadIdx_y;
						int col = blockDim_x * blockIdx_x + threadIdx_x;
						if(row < n_rows && col < m_cols){
							for(int counter = 0; counter < tile_width; counter++){
//								printf("C_device[%d,%d] = C_device[%d * %d + %d] += A_device_shared[%d][%d] * B_device_shared[%d][%d] = %f + (%f * %f) = %f\n", row, col, row, m_cols, col, threadIdx_y, counter, counter, threadIdx_y, C_device[row * m_cols + col], A_device_shared[threadIdx_y][counter], B_device_shared[counter][threadIdx_x], C_device[row * m_cols + col] + A_device_shared[threadIdx_y][counter] * B_device_shared[counter][threadIdx_x]);
								C_device[row * m_cols + col] += A_device_shared[threadIdx_y][counter] * B_device_shared[counter][threadIdx_x];
							}
						}
					}
				}

//				printf("C_device:\n");
//				for(int threadIdx_y = 0; threadIdx_y < blockDim_y; threadIdx_y++){
//					for(int threadIdx_x = 0; threadIdx_x < blockDim_x; threadIdx_x++){
//						int row = blockDim_y * blockIdx_y + threadIdx_y;
//						int col = blockDim_x * blockIdx_x + threadIdx_x;
//						if(row < n_rows && col < m_cols){
//							printf("%f ", C_device[row * m_cols + col]);
//						}
//					}
//					printf("\n");
//				}
			}
		}
	}
	// record execution time of kernel function
	printf(">>>> Recording execution time of kernel function matmul_rec_shar...\n");

	// copy data from device to host
	printf(">>>> Copying data from device to host\n");

	// free device memory
	printf(">>>> Freeing device memory\n");

	// return kernel execution time in milliseconds
	return 0;
}

// write kernel function matmul_rec_glob
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
			// C[4 * 3 + 0] = 0 + A[4 * 4 + 0] * B[0 * 3 + 0] = A[16]*B[0]
			// C[4 * 3 + 0] = (A[16] * B[0]) + A[4 * 4 + 1] * B[1 * 3 + 0] = A[16]*B[0] + A[17]*B[3]
			// C[4 * 3 + 0] = (A[16]*B[0] + A[17]*B[3]) + A[4 * 4 + 2] * B[2 * 3 + 0] = A[16]*B[0] + A[17]*B[3] + A[18]*B[6]
			// C[4 * 3 + 0] = (A[16]*B[0] + A[17]*B[3] + A[18]*B[6]) + A[4 * 4 + 3] * B[3 * 3 + 0] = A[16]*B[0] + A[17]*B[3] + A[18]*B[6] + A[19]*B[9]

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
//	printf("A_host:\n");
//	for(int row = 0; row < n_rows; row++){
//		for(int col = 0; col < k_cols_rows; col++){
//			printf("%f", A_host[row * k_cols_rows + col]);
//		}
//		printf("\n");
//	}
//	printf("\n");
//	printf("A_host:\n");
//	for(int row = 0; row < n_rows; row++){
//		for(int col = 0; col < k_cols_rows; col++){
//			printf("%f", A_host[row * k_cols_rows + col]);
//		}
//		printf("\n");
//	}
//	printf("\n");

	// set execution configuration parameters
	// Reference: CUDA by Example, Sanders et al., 2011

	printf(">>>> Setting execution configuration parameters...\n");
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
				printf("%f ", ((float*)C_shar)[row * m + col]);
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
