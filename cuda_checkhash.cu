/**
 * This code compares final hash against target
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"

__constant__ uint32_t pTarget[8]; // 32 bytes

// store MAX_GPUS device arrays of 8 nonces
static uint32_t* h_resNonces[MAX_GPUS];
static uint32_t* d_resNonces[MAX_GPUS];

__host__
void cuda_check_cpu_init(int thr_id, uint32_t threads)
{
    CUDA_SAFE_CALL(cudaMallocHost(&h_resNonces[thr_id], 8*sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], 8 * sizeof(uint32_t)));
}

// Target Difficulty

__host__
void cuda_check_cpu_setTarget(const void *ptarget, int thr_id)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(pTarget, ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
}

/* --------------------------------------------------------------------------------------------- */

__device__ __forceinline__
static bool hashbelowtarget(const uint32_t *const __restrict__ hash, const uint32_t *const __restrict__ target)
{
	if (hash[7] > target[7])
		return false;
	if (hash[7] < target[7])
		return true;
	if (hash[6] > target[6])
		return false;
	if (hash[6] < target[6])
		return true;

	if (hash[5] > target[5])
		return false;
	if (hash[5] < target[5])
		return true;
	if (hash[4] > target[4])
		return false;
	if (hash[4] < target[4])
		return true;

	if (hash[3] > target[3])
		return false;
	if (hash[3] < target[3])
		return true;
	if (hash[2] > target[2])
		return false;
	if (hash[2] < target[2])
		return true;

	if (hash[1] > target[1])
		return false;
	if (hash[1] < target[1])
		return true;
	if (hash[0] > target[0])
		return false;

	return true;
}

__global__ __launch_bounds__(512, 2)
void cuda_checkhash_64(uint32_t threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// shl 4 = *16 x 4 (uint32) = 64 bytes
		// todo: use only 32 bytes * threads if possible
		uint32_t *inpHash = &hash[thread << 4];

		if (resNonces[0] == UINT32_MAX) {
			if (hashbelowtarget(inpHash, pTarget))
				resNonces[0] = (startNounce + thread);
		}
	}
}

__host__
uint32_t cuda_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash)
{
	CUDA_SAFE_CALL(cudaMemsetAsync(d_resNonces[thr_id], 0xff, sizeof(uint32_t), gpustream[thr_id]));

	const uint32_t threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	cuda_checkhash_64 <<<grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_inputHash, d_resNonces[thr_id]);

	cudaMemcpyAsync(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id]);
	cudaStreamSynchronize(gpustream[thr_id]);

	return h_resNonces[thr_id][0];
}

/* --------------------------------------------------------------------------------------------- */

__global__ __launch_bounds__(512, 2)
void cuda_checkhash_64_suppl(uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t *inpHash = &hash[thread << 4];

	if (hashbelowtarget(inpHash, pTarget)) {
		int resNum = atomicAdd(resNonces,1)+1;
		if (resNum < 8)
			resNonces[resNum] = (startNounce + thread);
	}
}

__host__
uint32_t cuda_check_hash_suppl(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint32_t foundnonce)
{
	uint32_t rescnt, result = 0;

	const uint32_t threadsperblock = 512;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	// first element stores the count of found nonces
	cudaMemsetAsync(d_resNonces[thr_id], 0, sizeof(uint32_t), gpustream[thr_id]);

	cuda_checkhash_64_suppl <<<grid, block, 0, gpustream[thr_id]>>> (startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaMemcpyAsync(h_resNonces[thr_id], d_resNonces[thr_id], 8*sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id]);
	cudaStreamSynchronize(gpustream[thr_id]);

	rescnt = h_resNonces[thr_id][0];
	if (rescnt > 1)
	{
		do
		{
			if (h_resNonces[thr_id][rescnt] != foundnonce)
			{
				result = h_resNonces[thr_id][rescnt];
				break;
			}
			rescnt--;
		} while (rescnt > 0);
	}
	return result;
}

/* --------------------------------------------------------------------------------------------- */

__global__
void cuda_check_hash_branch_64(uint32_t threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = g_nonceVector[thread];
		uint32_t hashPosition = (nounce - startNounce) << 4;
		const uint32_t *const inpHash = &g_hash[hashPosition];

		if (hashbelowtarget(inpHash, pTarget))
		{
			if (resNounce[0] > nounce)
				resNounce[0] = nounce;
		}
	}
}

__global__
void cuda_check_quarkcoin_64(uint32_t threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t nounce = g_nonceVector[thread];
		uint32_t hashPosition = (nounce - startNounce) << 4;
		const uint32_t *const inpHash = &g_hash[hashPosition];

		if (inpHash[7] <= pTarget[7])
		{
			uint32_t tmp = atomicExch(resNounce, nounce);
			if (tmp != 0xffffffff)
				resNounce[1] = tmp;
		}
	}
}

__host__
uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash)
{
	uint32_t result = 0xffffffff;
	cudaMemsetAsync(d_resNonces[thr_id], 0xff, sizeof(uint32_t), gpustream[thr_id]);

	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_check_hash_branch_64 <<<grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_nonceVector, d_inputHash, d_resNonces[thr_id]);

	cudaMemcpyAsync(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id]);
	cudaStreamSynchronize(gpustream[thr_id]);

	result = *h_resNonces[thr_id];

	return result;
}
__host__
void cuda_check_quarkcoin(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, uint32_t *resNonces)
{
	CUDA_SAFE_CALL(cudaMemsetAsync(d_resNonces[thr_id], 0xff, 2 * sizeof(uint32_t), gpustream[thr_id]));

	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	cuda_check_quarkcoin_64 << <grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_nonceVector, d_inputHash, d_resNonces[thr_id]);

	cudaMemcpyAsync(resNonces, d_resNonces[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id]);
	cudaStreamSynchronize(gpustream[thr_id]);
}

int cuda_arch[MAX_GPUS];
__global__ void get_cuda_arch_gpu(int *d_version)
{
#ifdef __CUDA_ARCH__
	*d_version = __CUDA_ARCH__;
#endif
}

extern sha_algos opt_algo;

__host__ void get_cuda_arch(int *version)
{
	int *d_version;
	cudaMalloc(&d_version, sizeof(int));
	get_cuda_arch_gpu << < 1, 1 >> > (d_version);
	cudaMemcpy(version, d_version, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_version);
}
