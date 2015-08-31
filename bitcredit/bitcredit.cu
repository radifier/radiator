extern "C"
{
#include "sph/sph_types.h"
#include "sph/sph_sha2.h"
}
#include "miner.h"
#include "cuda_helper.h"

extern void bitcredit_setBlockTarget(int thr_id, uint32_t * data, const uint32_t * midstate, const void *ptarget);
extern void bitcredit_cpu_init(int thr_id, uint32_t threads, uint32_t* hash);
extern void bitcredit_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *result);

void credithash(void *state, const void *input)
{

	sph_sha256_context sha1, sha2;
	uint32_t hash[8], hash2[8];

	sph_sha256_init(&sha1);
	sph_sha256(&sha1, input, 168);
	sph_sha256_close(&sha1, hash);


	sph_sha256_init(&sha2);
	sph_sha256(&sha2, hash, 32);
	sph_sha256_close(&sha2, hash2);


	memcpy(state, hash2, 32);
}

int scanhash_bitcredit(int thr_id, uint32_t *pdata,
					   uint32_t *ptarget, const uint32_t *midstate, uint32_t max_nonce,
					   uint32_t *hashes_done)
{
	const uint32_t first_nonce = pdata[35];
	if(opt_benchmark)
		ptarget[7] = 0x00000001;

	int intensity = 256 * 256 * 64 * 8;
	const uint32_t throughput = min(device_intensity(device_map[thr_id], __func__, intensity), (max_nonce - first_nonce)) & 0xfffffc00; // 19=256*256*8;

	static THREAD uint32_t *d_hash = nullptr;
	static THREAD uint32_t *foundNonce = nullptr;

	static THREAD bool init = false;
	if(!init)
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash, 8 * sizeof(uint32_t) * throughput));
		CUDA_SAFE_CALL(cudaMallocHost(&foundNonce, 2 * sizeof(uint32_t)));
		bitcredit_cpu_init(thr_id, throughput, d_hash);
		init = true;
	}

	uint32_t endiandata[42];
	for(int k = 0; k < 42; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	bitcredit_setBlockTarget(thr_id, pdata, midstate, ptarget);
	do
	{
		bitcredit_cpu_hash(thr_id, throughput, pdata[35], foundNonce);
		cudaStreamSynchronize(gpustream[thr_id]);
		if(stop_mining)
		{
			mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);
		}
		if(foundNonce[0] != 0xffffffff)
		{
			int res = 1;
			if(opt_benchmark)
				applog(LOG_INFO, "GPU #%d: Found nonce %08x", thr_id, foundNonce[0]);
			*hashes_done = pdata[35] - first_nonce + throughput;
			pdata[35] = foundNonce[0];
			if(foundNonce[1] != 0xffffffff)
			{
				res = 2;
				if(opt_benchmark)
					applog(LOG_INFO, "GPU #%d: Found second nonce %08x", thr_id, foundNonce[1]);
				pdata[37] = foundNonce[1];
			}
			return res;
		}
		pdata[35] += throughput;
	} while(!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[35]) + (uint64_t)throughput)));
	*hashes_done = pdata[35] - first_nonce + 1;
	return 0;
}
