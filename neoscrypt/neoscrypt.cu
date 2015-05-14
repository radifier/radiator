
extern "C"
{
#include "sph/neoscrypt.h"
}

#include "cuda_helper.h"
#include "miner.h"
#include <string.h>

static uint32_t *d_hash[MAX_GPUS] ;
 

extern void neoscrypt_setBlockTarget(int thr_id, uint32_t * data, const void *ptarget);
extern void neoscrypt_cpu_init(int thr_id, int threads,uint32_t* hash);
extern void neoscrypt_cpu_hash_k4(int stratum,int thr_id, int threads, uint32_t startNounce, int order, uint32_t* foundnonce);
  

int scanhash_neoscrypt(bool stratum, int thr_id, uint32_t *pdata,
    uint32_t *ptarget, uint32_t max_nonce,
    uint32_t *hashes_done)
{
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		ptarget[7] = 0x01ff;

//	const int throughput = gpus_intensity[thr_id] ? 256 * 64 * gpus_intensity[thr_id] : 256 * 64 * 3.5;
	int intensity = (256 * 64 * 3.5);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device_map[thr_id]);
	if (strstr(props.name, "970"))
	{
		intensity = (256 * 64 * 4);
	}
	else if (strstr(props.name, "980"))
	{
		intensity = (256 * 64 * 4);
	}
	else if (strstr(props.name, "750 Ti"))
	{
		intensity = (256 * 64 * 3.5);
	}
	else if (strstr(props.name, "750"))
	{
		intensity = ((256 * 64 * 3.5) / 2);
	}
	else if (strstr(props.name, "960"))
	{
		intensity = (256 * 64 * 3.5);
	}

	uint32_t throughput = device_intensity(device_map[thr_id], __func__, intensity);

	throughput = min(throughput, (max_nonce - first_nonce));



	static volatile bool init[MAX_GPUS] = { false };
	if (!init[thr_id])
	{
		if(thr_id%opt_n_gputhreads == 0)
		{
			CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
			
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}
		else
		{
			while(!init[thr_id - thr_id%opt_n_gputhreads])
			{
			}
			CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		}
		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], 32 * 130 * sizeof(uint64_t) * throughput));
		neoscrypt_cpu_init(thr_id, throughput,d_hash[thr_id]);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	if (stratum) {
		for (int k = 0; k < 20; k++)
			be32enc(&endiandata[k], pdata[k]);
	}
	else {
		for (int k = 0; k < 20; k++)
			endiandata[k] = pdata[k];
	}
	


	neoscrypt_setBlockTarget(thr_id, endiandata, ptarget);
	uint32_t foundNonce[2];

	do {
		int order = 0;
		neoscrypt_cpu_hash_k4(stratum, thr_id, throughput, pdata[19], order++, foundNonce);
		if(stop_mining) {mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);}
		if(foundNonce[0] != 0xffffffff)
		{
			uint32_t vhash64[8];

			if(stratum)
			{
				be32enc(&endiandata[19], foundNonce[0]);
			}
			else
			{
				endiandata[19] = foundNonce[0];
			}
			neoscrypt((unsigned char*)endiandata, (unsigned char*)vhash64, 0x80000620);

			if(vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
			{
				int res = 1;
				*hashes_done = pdata[19] - first_nonce + throughput;
				if(foundNonce[1] != 0xffffffff)
				{
					if(stratum)
					{
						be32enc(&endiandata[19], foundNonce[1]);
					}
					else
					{
						endiandata[19] = foundNonce[1];
					}
					neoscrypt((unsigned char*)endiandata, (unsigned char*)vhash64, 0x80000620);
					if(vhash64[7] <= ptarget[7] && fulltest(vhash64, ptarget))
					{

						pdata[21] = foundNonce[1];
						res++;
						if(opt_benchmark)
							applog(LOG_INFO, "GPU #%d: Found second nounce %08x", device_map[thr_id], foundNonce[1]);
					}
					else
					{
						if(vhash64[7] != ptarget[7])
						{
							applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], foundNonce[1]);
						}
					}

				}
				pdata[19] = foundNonce[0];
				if(opt_benchmark)
					applog(LOG_INFO, "GPU #%d: Found nounce %08x", device_map[thr_id], foundNonce[0]);
				return res;
			}
			else
			{
				if(vhash64[7] != ptarget[7])
				   applog(LOG_WARNING, "GPU #%d: result for nonce $%08X does not validate on CPU!", device_map[thr_id], foundNonce);
			}

		}

		pdata[19] += throughput;
} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}
