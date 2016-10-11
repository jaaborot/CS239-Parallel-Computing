/*
 ============================================================================
 Name        : Matrix_Multiplication.c
 Author      : Jeffrey A. Aborot
 Version     :
 Copyright   : This work is open-source but requires proper citation if to be used.
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//#define MULTIPLE_DIMENSIONS
#define TILE_WIDTH 32
#define FILE_WRITE_MODE "a"

// Define this to turn on error checking
#define CUDA_ERROR_CHECK // check for any error when executing CUDA code; // https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
//#define PRINT_EXECUTION_CONFIG // print kernel execution configuration parameter values
//#define TRACE_LOGS // print program logs helpful in debugging the program
#define WRITE_RUNNING_TIMES_TO_FILE // write running time results of program execution to file
#define WRITE_AVERAGE_TIMES_TO_FILE // write averages of running time results of program execution to file
//#define EXECUTE_ON_CPU // multiply matrices on CPU

#define CudaSafeCall( err )       __cudaSafeCall( err, __FILE__, __LINE__ ) // https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
#define CudaCheckError()          __cudaCheckError( __FILE__, __LINE__ ) // https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
#define WriteRunningTimesToFile(table, execution_count, n, k, m, file_name, mode) __writeRunningTimesToFile(table, execution_count, n, k, m, file_name, mode)
#define WriteAverageTimesToFile(table, max_exponent, file_name, mode) __writeAverageTimesToFile(table, max_exponent, file_name, mode)

/*
 * Write into exercise CSV file the running times of the CPU and kernel functions.
 */
inline int __writeRunningTimesToFile(float* table, int execution_count, int n,
		int k, int m, char* file_name, char* mode)
{
#ifdef WRITE_RUNNING_TIMES_TO_FILE
	// open file
	FILE* csv_file_ptr;
	csv_file_ptr = fopen(file_name, mode);

	// handle exception when opening file
	if (csv_file_ptr == NULL)
	{
		printf("Unable to open %s.", file_name);
		return 1;
	}

	// write content of table to file
	fprintf(csv_file_ptr, "\nMatrix dimensions: n = %d, k = %d, m = %d\n", n, k,
			m);
	fprintf(csv_file_ptr,
			"iter #, matmul_rec_cpu, matmul_rec_glob,matmul_rec_shar\n");
	for (int row = 0; row < execution_count; row++)
	{
		fprintf(csv_file_ptr, "%d,", (int) table[row * 4]);
		fprintf(csv_file_ptr, "%f,", table[row * 4 + 1]);
		fprintf(csv_file_ptr, "%f,", table[row * 4 + 2]);
		fprintf(csv_file_ptr, "%f,", table[row * 4 + 3]);
		fprintf(csv_file_ptr, "\n");
	}
	fprintf(csv_file_ptr, "\n\n");

	// close file
	fclose(csv_file_ptr);
#endif

	return 0;
}

/*
 * Write into exercise CSV file the average running times of the CPU and kernel functions.
 */
inline int __writeAverageTimesToFile(float* table, int max_exponent,
		char* file_name, char* mode)
{
#ifdef WRITE_AVERAGE_TIMES_TO_FILE
	// open file
	FILE* csv_file_ptr;
	csv_file_ptr = fopen(file_name, mode);

	// handle exception when opening file
	if (csv_file_ptr == NULL)
	{
		printf("Unable to open %s.", file_name);
		return 1;
	}

	// write content of table to file
	fprintf(csv_file_ptr,
			"n = m,matmul_rec_cpu,matmul_rec_glob,matmul_rec_shar\n");
	for (int row = 0; row < max_exponent; row++)
	{
		fprintf(csv_file_ptr, "%d,", (int) table[row * 4]);
		fprintf(csv_file_ptr, "%f,", table[row * 4 + 1]);
		fprintf(csv_file_ptr, "%f,", table[row * 4 + 2]);
		fprintf(csv_file_ptr, "%f,", table[row * 4 + 3]);
		fprintf(csv_file_ptr, "\n");
	}
	fprintf(csv_file_ptr, "\n\n");

	// close file
	fclose(csv_file_ptr);
#endif

	return 0;
}

/*
 * Execute a code in CUDA and catch any error return by the device during execution.
 */
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

/*
 * Execute a code in CUDA and catch any error return by the device during execution.
 */
inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

/*
 * Multiply matrices A and B in GPU using a tiling method.
 */
__global__ void matmul_rec_shar(float** A_device, float** B_device,
		float** C_device, int n_rows, int k_cols_rows, int m_cols,
		int tile_width, int phase_count)
{
//	printf(">>>>>>>> Executing kernel function matmul_rec_shar...\n");

	// block scope: shared sub-matrices within a block
	__shared__
	float A_device_shared[TILE_WIDTH][TILE_WIDTH];
	__shared__
	float B_device_shared[TILE_WIDTH][TILE_WIDTH];
	// thread scope: only visible within each thread
	int A_col;
	int B_row;

	// element in C to be computed by this thread
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	/*
	 * Each thread of the block will load the elements in the corresponding blocks in A and B to the shared memory of the thread's block.
	 * We can then proceed with the computation after loading the elements in corresponding blocks in A and B in the shared block.
	 */

	for (int phase_counter = 0; phase_counter < phase_count; phase_counter++)
	{
		A_col = phase_counter * tile_width + threadIdx.x;
		if (row < n_rows && A_col < k_cols_rows)
		{
			A_device_shared[threadIdx.y][threadIdx.x] = A_device[row][A_col];
		}

		B_row = phase_counter * tile_width + threadIdx.y;
		if (B_row < k_cols_rows && col < m_cols)
		{
			B_device_shared[threadIdx.y][threadIdx.x] = B_device[B_row][col];
		}
		__syncthreads();

		/* After all threads in the block has already loaded their corresponding
		 * element in A_device and B_device, compute for the partial product element
		 * in C_device.
		 */
		if (row < n_rows && col < m_cols)
		{
			for (int i = 0; i < tile_width; i++)
			{
				if ((phase_counter * tile_width + i) < k_cols_rows)
				{
//					printf("phase: %d | i: %d | C_device[%d][%d] = C_device[%d][%d] + A_device_shared[%d][%d] * B_device_shared[%d][%d] = %f + %f * %f = %f \n", phase_counter, i, row, col, row, col, threadIdx.y, i, i, threadIdx.x, C_device[row][col], A_device_shared[threadIdx.y][i], B_device_shared[i][threadIdx.x], C_device[row][col] + A_device_shared[threadIdx.y][i] * B_device_shared[i][threadIdx.x]);
					C_device[row][col] += A_device_shared[threadIdx.y][i]
							* B_device_shared[i][threadIdx.x];
				}
			}
		}
		__syncthreads();
	}
}

/*
 * This function is a intermediate function for executing kernel matmul_rec_shar for setting up execution configuration parameters values.
 */
float host_matmul_rec_shar(float** A_host, float** B_host, float** C_host,
		int n_rows, int k_cols_rows, int m_cols)
{
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
	int tile_width;

	cudaGetDeviceCount(&cuda_device_count);
	if (cuda_device_count > 0)
	{
		cudaGetDeviceProperties(&prop, 0);

#ifdef PRINT_EXECUTION_CONFIG
		printf(">> Device properties ==\n");
		printf("CUDA-enabled device count: %d\n", cuda_device_count);
		printf("Maximum number of threads per 2D block: %d\n",
				prop.maxThreadsPerBlock); // maximum number of threads which can be alloted to a single block
		printf(
				"Maximum number of threads in x, y and z dimension of a 3D block: %d, %d, %d\n",
				prop.maxThreadsDim[0], prop.maxThreadsDim[1],
				prop.maxThreadsDim[2]); // maximum number of threads which can be alloted along the x, y and z dimension of each block
		printf(
				"Maximum number of threads in x, y and z dimension of a 3D grid: %d, %d, %d\n\n",
				prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
#endif
		/* The programmer may design each block with varying number of threads along the x and y dimension as long as
		 * the total number of threads in each block does not exceed maxThreadsPerBlock. For example, if maxThreadsPerBlock = 1024,
		 * the programmer may design the block to be a 32 rows x 32 columns block, 1 row by 1024 columns block, 1024 row by 1 column block, etc.
		 */

		/* maxGridSize refers to the maximum number of threads a grid can hold along the x (maxGridSize[0]), y (maxGridSize[1]) and z (maxGridSize[2]) dimensions.
		 * If the programmer wants to compute for the output matrix in a single step, then the maximum dimensions of the input matrices will be limited by the
		 * maximum number of threads a grid can have, i.e. maxGridSize[0] > dimension and maxGridSize[1] > dimension.
		 */
		if (prop.maxGridSize[0] > m_cols && prop.maxGridSize[1] > n_rows)
		{ // if matrix size less than number of threads in a grid

			// determine block dimension of the grid
			gridDim_x = (int) ceil((float) m_cols / prop.warpSize); // necessary number of blocks in x dimension in the grid
			gridDim_y = (int) ceil((float) n_rows / prop.warpSize); // necessary number of blocks in y dimension in the grid

			// determine thread dimension of each block
			blockDim_x = prop.warpSize;
			blockDim_y = prop.warpSize;

#ifdef PRINT_EXECUTION_CONFIG
			printf("warp size: %d\n", prop.warpSize);
			printf("Number of blocks along x dim in grid: %d\n", gridDim_x);
			printf("Number of blocks along y dim in grid: %d\n", gridDim_y);
			printf("Number of threads along x dim in block: %d\n", blockDim_x);
			printf("Number of threads along y dim in block: %d\n", blockDim_y);
#endif
		}
		else
		{

			// exit with max-thread-limit-reached exception
			printf(
					">> Number of matrix elements exceeded maximum number of threads available.\n");
			return EXIT_FAILURE;

		}
	}
	else
	{
		printf(">> No CUDA-enabled device found.\n");
		return EXIT_FAILURE;
	}

	dim3
	blocks_per_grid(gridDim_x, gridDim_y, 1); // how many blocks to utilize in the grid
	dim3
	threads_per_block(blockDim_x, blockDim_y, 1); // how many threads to utilize per block in the grid

	// configure execution parameters
	tile_width = prop.warpSize;
	// device pointers stored in device memory
	float** A_pointers_in_device_memory;
	float** B_pointers_in_device_memory;
	float** C_pointers_in_device_memory;

	// device pointers stored in host memory
	float* A_pointers_in_host_memory[n_rows];
	float* B_pointers_in_host_memory[k_cols_rows];
	float* C_pointers_in_host_memory[n_rows];

	// allocate memory in device for device pointers
	printf(">>>> Allocating device memory...\n");
	CudaSafeCall(
			cudaMalloc((void** ) &A_pointers_in_device_memory,
					n_rows * sizeof(float*)));
	CudaSafeCall(
			cudaMalloc((void** ) &B_pointers_in_device_memory,
					k_cols_rows * sizeof(float*)));
	CudaSafeCall(
			cudaMalloc((void** ) &C_pointers_in_device_memory,
					n_rows * sizeof(float*)));

	// allocate memory in device for host pointers
	for (int i = 0; i < n_rows; i++)
	{
		CudaSafeCall(
				cudaMalloc((void** ) &A_pointers_in_host_memory[i],
						k_cols_rows * sizeof(float)));
	}

	for (int i = 0; i < k_cols_rows; i++)
	{
		CudaSafeCall(
				cudaMalloc((void** ) &B_pointers_in_host_memory[i],
						m_cols * sizeof(float)));
	}

	for (int i = 0; i < n_rows; i++)
	{
		CudaSafeCall(
				cudaMalloc((void** ) &C_pointers_in_host_memory[i],
						m_cols * sizeof(float)));
	}

	// copy data from host memory to device memory
	printf(">>>> Copying data from host to device...\n");
	for (int i = 0; i < n_rows; i++)
	{
		// copy int*s in A_host to A_pointers_in_host_memory
		CudaSafeCall(
				cudaMemcpy(A_pointers_in_host_memory[i], A_host[i],
						k_cols_rows * sizeof(float), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < k_cols_rows; i++)
	{
		// copy int*s in B_host to B_pointers_in_host_memory
		CudaSafeCall(
				cudaMemcpy(B_pointers_in_host_memory[i], B_host[i],
						m_cols * sizeof(float), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < n_rows; i++)
	{
		// copy int*s in C_host to C_pointers_in_host_memory
		CudaSafeCall(
				cudaMemcpy(C_pointers_in_host_memory[i], C_host[i],
						m_cols * sizeof(float), cudaMemcpyHostToDevice));
	}

	/* Copy addresses in C_pointers_in_host_memory to C_pointers_in_device_memory.
	 * Recall that pointers in C_pointers_in_device_memory points to pointers in C_pointers_in_host_memory
	 */
	CudaSafeCall(
			cudaMemcpy(A_pointers_in_device_memory, A_pointers_in_host_memory,
					n_rows * sizeof(float*), cudaMemcpyHostToDevice));
	CudaSafeCall(
			cudaMemcpy(B_pointers_in_device_memory, B_pointers_in_host_memory,
					k_cols_rows * sizeof(float*), cudaMemcpyHostToDevice));
	CudaSafeCall(
			cudaMemcpy(C_pointers_in_device_memory, C_pointers_in_host_memory,
					n_rows * sizeof(float*), cudaMemcpyHostToDevice));

	// execute kernel function matmul_rec_glob
	printf(">>>> Executing kernel function matmul_rec_shar...\n");
	int phase_count = (int) ceil((float) k_cols_rows / tile_width); // number of phases for complete computation

	float runtime = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	matmul_rec_shar<<<blocks_per_grid, threads_per_block>>>(A_pointers_in_device_memory, B_pointers_in_device_memory, C_pointers_in_device_memory, n_rows, k_cols_rows, m_cols, tile_width, phase_count);
	CudaCheckError();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&runtime, start, stop);
//	cudaDeviceSynchronize();

	// copy data from device to host
	printf(">>>> Copying data from device to host\n");
	for (int i = 0; i < n_rows; i++)
	{
		CudaSafeCall(
				cudaMemcpy(C_host[i], C_pointers_in_host_memory[i],
						m_cols * sizeof(float), cudaMemcpyDeviceToHost));
	}

	// free device memory
	printf(">>>> Freeing device memory\n");
	CudaSafeCall(cudaFree(A_pointers_in_device_memory));
	CudaSafeCall(cudaFree(B_pointers_in_device_memory));
	CudaSafeCall(cudaFree(C_pointers_in_device_memory));

	// return kernel execution time in milliseconds
	return runtime;
}

/*
 * Multiply matrices A and B in GPU using a single thread for a single element in the output matrix C.
 */
__global__ void matmul_rec_glob(float** A_device, float** B_device,
		float** C_device, int n_rows, int k_cols_rows, int m_cols)
{
//	printf(">>>>>>>> Executing kernel function matmul_rec_glob...\n");
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < n_rows && col < m_cols)
	{
		for (int cols_rows_counter = 0; cols_rows_counter < k_cols_rows;
				cols_rows_counter++)
		{
//			printf("*(*(C_device + %d) + %d) += *(*(A_device + %d) + %d)) * *(*(B_device + %d) + %d) = %f + %f * %f\n", row, col, row, cols_rows_counter, cols_rows_counter, col, *(*(C_device + row) + col), *(*(A_device + row) + cols_rows_counter), *(*(B_device + cols_rows_counter) + col));
			C_device[row][col] += A_device[row][cols_rows_counter]
					* B_device[cols_rows_counter][col];
		}
	}
	__syncthreads();
}

/*
 * This is an intermediate function for kernel matmul_rec_glob for setting up execution configuration parameters values.
 */
float host_matmul_rec_glob(float** A_host, float** B_host, float** C_host,
		int n_rows, int k_cols_rows, int m_cols)
{
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
	if (cuda_device_count > 0)
	{
		cudaGetDeviceProperties(&prop, 0);

#ifdef PRINT_EXECUTION_CONFIG
		printf(">> Device properties ==\n");
		printf("CUDA-enabled device count: %d\n", cuda_device_count);
		printf("Maximum number of threads per 2D block: %d\n",
				prop.maxThreadsPerBlock); // maximum number of threads which can be alloted to a single block
		printf(
				"Maximum number of threads in x, y and z dimension of a 3D block: %d, %d, %d\n",
				prop.maxThreadsDim[0], prop.maxThreadsDim[1],
				prop.maxThreadsDim[2]); // maximum number of threads which can be alloted along the x, y and z dimension of each block
		printf(
				"Maximum number of threads in x, y and z dimension of a 3D grid: %d, %d, %d\n\n",
				prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
#endif
		/* The programmer may design each block with varying number of threads along the x and y dimension as long as
		 * the total number of threads in each block does not exceed maxThreadsPerBlock. For example, if maxThreadsPerBlock = 1024,
		 * the programmer may design the block to be a 32 rows x 32 columns block, 1 row by 1024 columns block, 1024 row by 1 column block, etc.
		 */

		/* maxGridSize refers to the maximum number of threads a grid can hold along the x (maxGridSize[0]), y (maxGridSize[1]) and z (maxGridSize[2]) dimensions.
		 * If the programmer wants to compute for the output matrix in a single step, then the maximum dimensions of the input matrices will be limited by the
		 * maximum number of threads a grid can have, i.e. maxGridSize[0] > dimension and maxGridSize[1] > dimension.
		 */
		if (prop.maxGridSize[0] > m_cols && prop.maxGridSize[1] > n_rows)
		{ // if matrix size less than number of threads in a grid

			// determine block dimension of the grid
			gridDim_x = (int) ceil((float) m_cols / prop.warpSize); // necessary number of blocks in x dimension in the grid
			gridDim_y = (int) ceil((float) n_rows / prop.warpSize); // necessary number of blocks in y dimension in the grid

			// determine thread dimension of each block
			blockDim_x = prop.warpSize;
			blockDim_y = prop.warpSize;

#ifdef PRINT_EXECUTION_CONFIG
			printf("warp size: %d\n", prop.warpSize);
			printf("Number of blocks along x dim in grid: %d\n", gridDim_x);
			printf("Number of blocks along y dim in grid: %d\n", gridDim_y);
			printf("Number of threads along x dim in block: %d\n", blockDim_x);
			printf("Number of threads along y dim in block: %d\n", blockDim_y);
#endif
		}
		else
		{

			// exit with max-thread-limit-reached exception
			printf(
					">> Number of matrix elements exceeded maximum number of threads available.\n");
			return EXIT_FAILURE;

		}
	}
	else
	{
		printf(">> No CUDA-enabled device found.\n");
		return EXIT_FAILURE;
	}

	// configure execution parameters
	dim3
	blocks_per_grid(gridDim_x, gridDim_y, 1); // how many blocks to utilize in the grid
	dim3
	threads_per_block(blockDim_x, blockDim_y, 1); // how many threads to utilize per block in the grid

	// allocate device memory
	printf(">>>> Allocating device memory...\n");

	// device pointers stored in device memory
	float** A_pointers_in_device_memory;
	CudaSafeCall(
			cudaMalloc((void** ) &A_pointers_in_device_memory,
					n_rows * sizeof(float*)));

	float** B_pointers_in_device_memory;
	CudaSafeCall(
			cudaMalloc((void** ) &B_pointers_in_device_memory,
					k_cols_rows * sizeof(float*)));

	float** C_pointers_in_device_memory;
	CudaSafeCall(
			cudaMalloc((void** ) &C_pointers_in_device_memory,
					n_rows * sizeof(float*)));

	/* Copy pointers to pointers stored in host memory (C_host) to allocated memory location in device (C_pointers_in_device_memory)
	 * Make sure that C_host has been allocated sizeof(float*) * n amount of memory in calling function.
	 */

	// device pointers stored in host memory
	float* A_pointers_in_host_memory[n_rows];
	for (int i = 0; i < n_rows; i++)
	{
		CudaSafeCall(
				cudaMalloc((void** ) &A_pointers_in_host_memory[i],
						k_cols_rows * sizeof(float)));
		// copy int*s in A_host to A_pointers_in_host_memory
		CudaSafeCall(
				cudaMemcpy(A_pointers_in_host_memory[i], A_host[i],
						k_cols_rows * sizeof(float), cudaMemcpyHostToDevice));
	}

	float* B_pointers_in_host_memory[k_cols_rows];
	for (int i = 0; i < k_cols_rows; i++)
	{
		CudaSafeCall(
				cudaMalloc((void** ) &B_pointers_in_host_memory[i],
						m_cols * sizeof(float)));
		// copy int*s in B_host to B_pointers_in_host_memory
		CudaSafeCall(
				cudaMemcpy(B_pointers_in_host_memory[i], B_host[i],
						m_cols * sizeof(float), cudaMemcpyHostToDevice));
	}

	float* C_pointers_in_host_memory[n_rows];
	for (int i = 0; i < n_rows; i++)
	{
		CudaSafeCall(
				cudaMalloc((void** ) &C_pointers_in_host_memory[i],
						m_cols * sizeof(float)));
		// copy int*s in C_host to C_pointers_in_host_memory
		CudaSafeCall(
				cudaMemcpy(C_pointers_in_host_memory[i], C_host[i],
						m_cols * sizeof(float), cudaMemcpyHostToDevice));
	}

	// copy data from host to device
	printf(">>>> Copying data from host to device...\n");
	/* Copy pointers to pointers defined in host memory (A_host, B_host) to pointers to pointers defined in device memory
	 * (A_pointers_in_device_memory, B_pointers_in_device_memory).
	 * Make sure that A_host has been allocated with sizeof(float*) * n_rows amount of memory in calling function.
	 * Make sure that B_host has been allocated with sizeof(float*) * k_cols_rows amoutn of memory in calling function.
	 */
	// copy int**s in C_pointers_in_host_memory to C_pointers_in_device_memory
	CudaSafeCall(
			cudaMemcpy(A_pointers_in_device_memory, A_pointers_in_host_memory,
					n_rows * sizeof(float*), cudaMemcpyHostToDevice));
	CudaSafeCall(
			cudaMemcpy(B_pointers_in_device_memory, B_pointers_in_host_memory,
					k_cols_rows * sizeof(float*), cudaMemcpyHostToDevice));
	CudaSafeCall(
			cudaMemcpy(C_pointers_in_device_memory, C_pointers_in_host_memory,
					n_rows * sizeof(float*), cudaMemcpyHostToDevice));

	// execute kernel function matmul_rec_glob
	printf(">>>> Executing kernel function matmul_rec_glob...\n");
	// define CUDA event recorder
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	// execute kernel function
	matmul_rec_glob<<<blocks_per_grid, threads_per_block>>>(A_pointers_in_device_memory, B_pointers_in_device_memory, C_pointers_in_device_memory, n_rows, k_cols_rows, m_cols);
	/* To block all succeeding commands in the CPU until all commands in the GPU has finished.
	 * How to Implement Performance Metrics in CUDA C/C++
	 * URL: https://devblogs.nvidia.com/parallelforall/how-implement-performance-metrics-cuda-cc/
	 */
//	cudaDeviceSynchronize(); // replace with cudaEventSynchronize()
	CudaCheckError(); // utility function for checking any error which might have occurred during kernel execution
	cudaEventRecord(stop);
	cudaEventSynchronize(stop); // blocks execution of any CPU command until event stop has been recorded

	// compute for elapsed time (execution time of kernel)
	float runtime = 0;
	cudaEventElapsedTime(&runtime, start, stop);

	// copy data from device to host
	printf(">>>> Copying data from device to host\n");
	for (int i = 0; i < n_rows; i++)
	{
		CudaSafeCall(
				cudaMemcpy(C_host[i], C_pointers_in_host_memory[i],
						m_cols * sizeof(float), cudaMemcpyDeviceToHost));
	}

	// free device memory
	printf(">>>> Freeing device memory\n");
	CudaSafeCall(cudaFree(A_pointers_in_device_memory));
	CudaSafeCall(cudaFree(B_pointers_in_device_memory));
	CudaSafeCall(cudaFree(C_pointers_in_device_memory));

	// return kernel execution time in milliseconds
	return runtime;
}

/*
 * Multiply matrices A and B using CPU only.
 */
float matmul_rec_cpu(float** A_host, float** B_host, float** C_host, int n_rows,
		int k_cols_rows, int m_cols)
{
	time_t t_start, t_end;

	t_start = time(NULL);
	for (int row = 0; row < n_rows; row++)
	{
		for (int col = 0; col < m_cols; col++)
		{
			C_host[row][col] = 0.0;
			for (int i = 0; i < k_cols_rows; i++)
			{
				C_host[row][col] += A_host[row][i] * B_host[i][col];
			}
		}
	}
	t_end = time(NULL);
	printf("matmul_rec_cpu: t_start = %ld s, t_end = %ld s, total_time = %f\n",
			t_start, t_end, (float) ((t_end - t_start) * 1000));

	return (float) ((t_end - t_start) * 1000);
}

#if MULTIPLE_DIMENSIONS
int main(void)
{
	puts("Start.\n"); /* prints !!!Hello World!!! */

	int n, k, m, execution_count;
	int max_exponent;
	time_t t;

	// user input for matrix dimensions n, k and m
	printf(
			"Let A be a n x k matrix, B a k x m matrix. Output matrix C = A dot B.\n");
	printf(
			"Specify maximum exponent x where n = m = 2^x (square matrix output C)...\n>> ");
	scanf("%d", &max_exponent);
	printf("The program will be executed for each n = m = 2");
	if ((int) max_exponent > 1)
	{
		printf(" to %d...\n", (int) pow(2, max_exponent));
	}
	else
	{
		printf(".\n");
	}
	printf("max_exponent: %d\n", max_exponent);
	printf("Specify k...\n>> ");
	scanf("%d", &k);

	// user input for number of iteration of program execution
	printf(
			"\nSpecify number of program execution iteration per dimension of C: ");
	scanf("%d", &execution_count);

	float kernel_times[execution_count][4];
	float average_times[max_exponent][4];
	int exec_code;

	for (int power_counter = 1; power_counter <= max_exponent; power_counter++)
	{
		n = m = (int) pow(2, power_counter);

		/* Since we vary the values for n and m in each iteration for each power of 2,
		 * we use dynamic allocation for the dimensions of the 2D arrays for A, B,
		 * C_glob and C_shar using double pointer method.
		 */
		// dynamically allocate memory for an array of row number of pointers
		float** A_ptr = (float**) malloc(n * sizeof(float*));
		// for each row, dynamically allocate memory enough for each row
		for (int i = 0; i < n; ++i)
		{
			A_ptr[i] = (float*) malloc(k * sizeof(float));
		}

		// dynamically allocate memory for an array of row number of pointers
		float** B_ptr = (float**) malloc(k * sizeof(float*));
		// for each row, dynamically allocate memory enough for each row
		for (int i = 0; i < k; ++i)
		{
			B_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		// dynamically allocate memory for an array of row number of pointers
		float** C_cpu_ptr = (float**) malloc(n * sizeof(float*));
		// for each row, dynamically allocate memory enough for each row
		for (int i = 0; i < n; ++i)
		{
			C_cpu_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		// dynamically allocate memory for an array of row number of pointers
		float** C_glob_ptr = (float**) malloc(n * sizeof(float*));
		// for each row, dynamically allocate memory enough for each row
		for (int i = 0; i < n; ++i)
		{
			C_glob_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		// dynamically allocate memory for an array of row number of pointers
		float** C_shar_ptr = (float**) malloc(n * sizeof(float*));
		// for each row, dynamically allocate memory enough for each row
		for (int i = 0; i < n; ++i)
		{
			C_shar_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		printf(
				"\nParameter values: n = %d, k = %d, m = %d, execution_count = %d\n\n",
				n, k, m, execution_count);

		for (int execution_counter = 0; execution_counter < execution_count;
				execution_counter++)
		{

			printf("==== Program execution iteration #%d ====\n",
					execution_counter + 1);
			kernel_times[execution_counter][0] =
					(float) (execution_counter + 1);

			// randomly generate matrix A with dimension n x k
			printf(">> Generating random matrix A...\n");
			srand((unsigned) time(&t));
			for (int row = 0; row < n; row++)
			{
				for (int col = 0; col < k; col++)
				{
					A_ptr[row][col] = (float) (rand() % 101);
				}
			}

#ifdef TRACE_LOGS
			for (int row = 0; row < n; row++)
			{
				for (int col = 0; col < k; col++)
				{
					printf("%f ", A_ptr[row][col]);
				}
				printf("\n");
			}
#endif

			// randomly generate matrix B with dimension k x m
			printf(">> Generating random matrix B...\n");
			srand((unsigned) time(&t));
			for (int row = 0; row < k; row++)
			{
				for (int col = 0; col < m; col++)
				{
					B_ptr[row][col] = (float) (rand() % 100);
				}
			}
#ifdef TRACE_LOGS
			for (int row = 0; row < k; row++)
			{
				for (int col = 0; col < m; col++)
				{
					printf("%f ", B_ptr[row][col]);
				}
				printf("\n");
			}
#endif

			// initializing matrix C
			for (int row = 0; row < n; row++)
			{
				for (int col = 0; col < m; col++)
				{
					C_cpu_ptr[row][col] = 0.0;
					C_glob_ptr[row][col] = 0.0;
					C_shar_ptr[row][col] = 0.0;
				}
			}

			// call function for multiplying matrices A and B using CPU only
			printf(">> Executing computation on CPU...\n");
#ifdef EXECUTE_ON_CPU
			kernel_times[execution_counter][1] = matmul_rec_cpu((float**) A_ptr, (float**) B_ptr, (float**) C_cpu_ptr, n, k, m);
#else
			kernel_times[execution_counter][1] = 0.0;
#endif

#ifdef TRACE_LOGS
			printf("C_cpu:\n");
			for(int row = 0; row < n; row++)
			{
				for(int col = 0; col < m; col++)
				{
					printf("%f ", C_cpu_ptr[row][col]);
				}
				printf("\n");
			}
#endif

			// call intermediate function for multiplying matrices A and B using global memory
			printf(
					">> Executing intermediate function for kernel function matmul_rec_glob...\n");
			kernel_times[execution_counter][2] = host_matmul_rec_glob(
					(float**) A_ptr, (float**) B_ptr, (float**) C_glob_ptr, n,
					k, m); // using pointers
#ifdef TRACE_LOGS
					printf("C_glob:\n");
					for(int row = 0; row < n; row++)
					{
						for(int col = 0; col < m; col++)
						{
							printf("%f ", C_glob_ptr[row][col]);
						}
						printf("\n");
					}
#endif

			// call intermediate function for multiplying matrices A and B using shared memory
			printf(
					">> Executing intermediate function for kernel function matmul_rec_shar...\n");
			kernel_times[execution_counter][3] = host_matmul_rec_shar(
					(float**) A_ptr, (float**) B_ptr, (float**) C_shar_ptr, n,
					k, m); // using pointers
#ifdef TRACE_LOGS
					printf("C_shar:\n");
					for(int row = 0; row < n; row++)
					{
						for(int col = 0; col < m; col++)
						{
							printf("%f ", C_shar_ptr[row][col]);
						}
						printf("\n");
					}
#endif
			// check for mismatch between to output matrices
			printf(
					">> Checking for any mismatches between C_cpu, C_glob and C_shar...\n");
#ifdef EXECUTE_ON_CPU
			for(int row = 0; row < n; row++)
			{
				for(int col = 0; col < m; col++)
				{
					if(C_cpu_ptr[row][col] != C_glob_ptr[row][col])
					{
						printf("A mismatch occurs between C_cpu and C_glob in element at coordinate (%d,%d).\n", row, col);
						return 1;
					}
					else if(C_cpu_ptr[row][col] != C_shar_ptr[row][col])
					{
						printf("A mismatch occurs between C_cpu and C_shar in element at coordinate (%d,%d).\n", row, col);
						return 1;
					}
				}
			}
#else
			for (int row = 0; row < n; row++)
			{
				for (int col = 0; col < m; col++)
				{
					if (C_glob_ptr[row][col] != C_shar_ptr[row][col])
					{
						printf(
								"A mismatch occurs between C_glob and C_shar in element at coordinate (%d,%d).\n",
								row, col);
						return 1;
					}
				}
			}
#endif
			printf(">> Resulting matrices matched...\n");
		}

		// free memory allocation for A, B, C_glob and C_shar
		free(A_ptr);
		free(B_ptr);
		free(C_cpu_ptr);
		free(C_glob_ptr);
		free(C_shar_ptr);

#ifdef WRITE_RUNNING_TIMES_TO_FILE
		// write to CSV file the execution time of each kernel for current iteration
		printf(">> Writing execution time of each kernel into CSV file...\n\n");
		exec_code = WriteRunningTimesToFile((float* ) kernel_times,
				execution_count, n, k, m, "aborot-exer2.csv", FILE_WRITE_MODE);
		if (exec_code == 1)
		{
			return EXIT_FAILURE;
		}
#endif
		// compute for average execution time of kernel functions for current n and m
		float total_times_cpu = 0.0;
		float total_times_glob = 0.0;
		float total_times_shar = 0.0;
		for (int i = 0; i < execution_count; i++)
		{
			total_times_cpu += kernel_times[i][1];
			total_times_glob += kernel_times[i][2];
			total_times_shar += kernel_times[i][3];
		}
		average_times[power_counter - 1][0] = pow(2, power_counter);
		average_times[power_counter - 1][1] = total_times_cpu
				/ (float) execution_count;
		average_times[power_counter - 1][2] = total_times_glob
				/ (float) execution_count;
		average_times[power_counter - 1][3] = total_times_shar
				/ (float) execution_count;

		printf(
				"\npower: %d | total_times_cpu: %f ms | total_times_glob: %f ms | total_times_shar: %f ms | average_cpu: %f ms | average_glob: %f ms | average_shar: %f ms\n\n",
				power_counter, total_times_cpu, total_times_glob,
				total_times_shar, average_times[power_counter - 1][1],
				average_times[power_counter - 1][2],
				average_times[power_counter - 1][3]);
	}

	// write to CSV file the average execution times for all n and m
#ifdef WRITE_AVERAGE_TIMES_TO_FILE
	printf(
			"Writing average execution times of each kernel per dimension n, m...\n");
	for (int i = 0; i < max_exponent; i++)
	{
		printf("n = m = %d: %f ms, %f ms, %f ms\n", (int) average_times[i][0],
				average_times[i][1], average_times[i][2], average_times[i][3]);
	}
	exec_code = WriteAverageTimesToFile((float* ) average_times, max_exponent,
			"aborot-exer2.csv", FILE_WRITE_MODE);
	if (exec_code == 1)
	{
		return EXIT_FAILURE;
	}
#endif

	printf(">> End.\n");

	return EXIT_SUCCESS;
}
#else
int main(void)
{
	int n, k, m, n_init, n_last, execution_count, exec_code;
	time_t t;

	printf("Specify initial row count n: ");
	scanf("%d", &n_init);
	printf("Specify final row count n: ");
	scanf("%d", &n_last);

	printf("Specify initial row count n: ");
	scanf("%d", &m_init);
	printf("Specify final row count n: ");
	scanf("%d", &m_last);

	printf("Specify number of rows of A: ");
	scanf("%d", &n);
	printf("Specify number of columns of B: ");
	scanf("%d", &m);
	printf("Specify number of columns of A and number of rows of B: ");
	scanf("%d", &k);
	printf("Specify number of times to execute the computation: ");
	scanf("%d", &execution_count);
	printf("n: %d, k: %d, m: %d, execution_count: %d\n", n, k, m, execution_count);

	float kernel_times[execution_count][4];
	float average_times[execution_count][4];

	for (int execution_counter = 0; execution_counter < execution_count; ++execution_counter)
	{
		/*
		 * Allocate memory locations in host.
		 */
		float** A_ptr = (float**) malloc(n * sizeof(float*));
		for (int i = 0; i < n; ++i)
		{
			A_ptr[i] = (float*) malloc(k * sizeof(float));
		}

		float** B_ptr = (float**) malloc(k * sizeof(float*));
		for (int i = 0; i < k; ++i)
		{
			B_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		float** C_cpu_ptr = (float**) malloc(n * sizeof(float*));
		for (int i = 0; i < n; ++i)
		{
			C_cpu_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		float** C_glob_ptr = (float**) malloc(n * sizeof(float*));
		for (int i = 0; i < n; ++i)
		{
			C_glob_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		float** C_shar_ptr = (float**) malloc(n * sizeof(float*));
		for (int i = 0; i < n; ++i)
		{
			C_shar_ptr[i] = (float*) malloc(m * sizeof(float));
		}

		// randomly generate matrix A with dimension n x k
		printf(">> Generating random matrix A...\n");
		srand((unsigned) time(&t));
		for (int row = 0; row < n; row++)
		{
			for (int col = 0; col < k; col++)
			{
				A_ptr[row][col] = (float) (rand() % 101);
			}
		}

#ifdef TRACE_LOGS
		for (int row = 0; row < n; row++)
		{
			for (int col = 0; col < k; col++)
			{
				printf("%f ", A_ptr[row][col]);
			}
			printf("\n");
		}
#endif

		// randomly generate matrix B with dimension k x m
		printf(">> Generating random matrix B...\n");
		srand((unsigned) time(&t));
		for (int row = 0; row < k; row++)
		{
			for (int col = 0; col < m; col++)
			{
				B_ptr[row][col] = (float) (rand() % 100);
			}
		}
#ifdef TRACE_LOGS
		for (int row = 0; row < k; row++)
		{
			for (int col = 0; col < m; col++)
			{
				printf("%f ", B_ptr[row][col]);
			}
			printf("\n");
		}
#endif

		// initializing matrix C
		for (int row = 0; row < n; row++)
		{
			for (int col = 0; col < m; col++)
			{
				C_cpu_ptr[row][col] = 0.0;
				C_glob_ptr[row][col] = 0.0;
				C_shar_ptr[row][col] = 0.0;
			}
		}

		kernel_times[execution_counter][0] = (float) execution_counter;

#ifdef EXECUTE_ON_CPU
		// execute kernel matmul_rec_cpu
		printf("if: kernel_times[%d][1] = %f\n", execution_counter);
		kernel_times[execution_counter][1] = matmul_rec_cpu(A_ptr, B_ptr, C_cpu_ptr, n, k, m);
#else
//		printf("else: kernel_times[%d][1] = 0.0\n", execution_counter);
		kernel_times[execution_counter][1] = 0.0;
#endif
		// execute kernel matmul_rec_glob
		kernel_times[execution_counter][2] = host_matmul_rec_glob(A_ptr, B_ptr, C_glob_ptr, n, k, m);
		// execute kernel matmul_rec_shar
		kernel_times[execution_counter][3] = host_matmul_rec_shar(A_ptr, B_ptr, C_shar_ptr, n, k, m);

		// check for mismatches between C_cpu_ptr, C_glob_ptr, and C_shar_ptr
		printf(
				">> Checking for any mismatches between C_cpu, C_glob and C_shar...\n");
#ifdef EXECUTE_ON_CPU
		for(int row = 0; row < n; row++)
		{
			for(int col = 0; col < m; col++)
			{
				if(C_cpu_ptr[row][col] != C_glob_ptr[row][col])
				{
					printf("A mismatch occurs between C_cpu and C_glob in element at coordinate (%d,%d).\n", row, col);
					return 1;
				}
				else if(C_cpu_ptr[row][col] != C_shar_ptr[row][col])
				{
					printf("A mismatch occurs between C_cpu and C_shar in element at coordinate (%d,%d).\n", row, col);
					return 1;
				}
			}
		}
#else
		for (int row = 0; row < n; row++)
		{
			for (int col = 0; col < m; col++)
			{
				if (C_glob_ptr[row][col] != C_shar_ptr[row][col])
				{
					printf(
							"A mismatch occurs between C_glob and C_shar in element at coordinate (%d,%d).\n",
							row, col);
					return 1;
				}
			}
		}
#endif
		printf(">> Resulting matrices matched...\n");

		// free memory allocation for A, B, C_glob and C_shar
		free(A_ptr);
		free(B_ptr);
		free(C_cpu_ptr);
		free(C_glob_ptr);
		free(C_shar_ptr);

	}

#ifdef WRITE_RUNNING_TIMES_TO_FILE
	// write to CSV file the execution time of each kernel for current iteration
	printf(">> Writing execution time of each kernel into CSV file...\n\n");
	exec_code = WriteRunningTimesToFile((float*) kernel_times,
			execution_count, n, k, m, "aborot-exer2.csv", FILE_WRITE_MODE);
	if (exec_code == 1)
	{
		return EXIT_FAILURE;
	}
#endif

	// record average execution times
	float total_cpu_time = 0.0;
	float total_glob_time = 0.0;
	float total_shar_time = 0.0;
	for (int i = 0; i < execution_count; ++i)
	{
		total_cpu_time += kernel_times[i][1];
		total_glob_time += kernel_times[i][2];
		total_shar_time += kernel_times[i][3];
	}
//	printf("total_cpu_time: %f ms | total_glob_time: %f ms | total_shar_time: %f ms\n", total_cpu_time, total_glob_time, total_shar_time);
	average_times[0][0] = 0;
	average_times[0][1] = total_cpu_time / (float) execution_count;
	average_times[0][2] = total_glob_time / (float) execution_count;
	average_times[0][3] = total_shar_time / (float) execution_count;

	// write to average kernel execution times to file
#ifdef WRITE_AVERAGE_TIMES_TO_FILE
	printf(
			"Writing average execution times of each kernel per dimension n, m...\n");
	printf("CPU: %f ms | Global: %f ms, Shared: %f ms\n", average_times[0][1],
			average_times[0][2], average_times[0][3]);
	exec_code = WriteAverageTimesToFile((float*) average_times, 1,
			"aborot-exer2.csv", FILE_WRITE_MODE);
	if (exec_code == 1)
	{
		return EXIT_FAILURE;
	}
#endif

	printf(">> End");
	return EXIT_SUCCESS;
}
#endif
