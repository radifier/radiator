#include "sph/sph_sha2.h"
#include "miner.h"
#include "cuda_helper.h"

extern void rad_cpu_init(int thr_id);
extern void rad_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, const uint64_t *const data, uint32_t *const h_nounce);
extern void rad_midstate(const uint32_t *data, uint32_t *midstate);

uint32_t rotr64(uint32_t x, unsigned int n)
{
	return (x >> n) | (x << (32 - n));
}

void rad_hash(uint32_t *output, const uint32_t *data, uint32_t nonce)
{
	uint32_t header[20];
	for (int i = 0; i < 20; i++) {
		header[i] = swab32(data[i]);
	}
	header[19] = swab32(nonce);
	sph_sha512_context ctx;
	sph_sha512_256_init(&ctx);
	sph_sha512(&ctx, header, 80);
	uint32_t hash1[16];
	sph_sha512_close(&ctx, hash1);

	unsigned char hex2[160];
	cbin2hex((char*)hex2, (const char*)header, 80);

	uint32_t hash2[16];
	sph_sha512_256_init(&ctx);
	sph_sha512(&ctx, hash1, 32);
	sph_sha512_close(&ctx, hash2);

	for (int i = 0; i < 8; i++) {
		output[i] = hash2[i];
	}

	unsigned char hex4[32];
	cbin2hex((char*)hex4, (const char*)output, 32);
}


int scanhash_rad(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done)
{
	static THREAD uint32_t *h_nounce = nullptr;

	const uint32_t first_nonce = pdata[19];
	uint32_t throughputmax = device_intensity(device_map[thr_id], __func__, 1U << 28);
	uint32_t throughput = min(throughputmax, (max_nonce - first_nonce)) & 0xfffffc00;

	if (opt_benchmark)
		ptarget[7] = 0x0005;

	static THREAD volatile bool init = false;
	if(!init)
	{
		if(throughputmax == 1<<28)
			applog(LOG_INFO, "GPU #%d: using default intensity 28", device_map[thr_id]);
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		CUDA_SAFE_CALL(cudaDeviceReset());
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaschedule));
		CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

		CUDA_SAFE_CALL(cudaStreamCreate(&gpustream[thr_id]));
		rad_cpu_init(thr_id);
		CUDA_SAFE_CALL(cudaMallocHost(&h_nounce, 2 * sizeof(uint32_t)));
		mining_has_stopped[thr_id] = false;
		init = true;
	}

	do
	{
		rad_cpu_hash(thr_id, throughput, pdata[19], (uint64_t *)pdata, h_nounce);
		if(stop_mining) {mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);}
		if(h_nounce[0] != UINT32_MAX)
		{
			uint32_t vhash64[8]={0};

			rad_hash(vhash64, pdata, h_nounce[0]);
			if (!opt_verify || (vhash64[7] == 0 && fulltest(vhash64, ptarget)))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (h_nounce[1] != 0xffffffff)
				{
					rad_hash(vhash64, pdata, h_nounce[1]);
					if (!opt_verify || (vhash64[7] == 0 && fulltest(vhash64, ptarget)))
					{
						pdata[21] = h_nounce[1];
						res++;
						if (opt_benchmark)
							applog(LOG_INFO, "GPU #%d Found second nounce %08x", device_map[thr_id], h_nounce[1]);
					}
					else
					{
						if (vhash64[7] > 0)
						{
							applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], h_nounce[1]);
						}
					}
				}
				pdata[19] = h_nounce[0];
				if (opt_benchmark)
					applog(LOG_INFO, "GPU #%d Found nounce %08x", device_map[thr_id], h_nounce[0]);
				return res;
			}
			else
			{
				if (vhash64[7] > 0)
				{
					applog(LOG_WARNING, "GPU #%d: result for %08x does not validate on CPU!", device_map[thr_id], h_nounce[0]);
				}
			}
		}

		pdata[19] += throughput; CUDA_SAFE_CALL(cudaGetLastError());
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce ;
	return 0;
}
