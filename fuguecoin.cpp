#include <string.h>
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include "uint256.h"
#include "sph/sph_fugue.h"

#include "miner.h"
#include "cuda_fugue256.h"
#include <cuda_runtime.h>
extern bool stop_mining;
extern bool mining_has_stopped[MAX_GPUS];

extern "C" void my_fugue256_init(void *cc);
extern "C" void my_fugue256(void *cc, const void *data, size_t len);
extern "C" void my_fugue256_close(void *cc, void *dst);
extern "C" void my_fugue256_addbits_and_close(void *cc, unsigned ub, unsigned n, void *dst);

// vorbereitete Kontexte nach den ersten 80 Bytes
// sph_fugue256_context  ctx_fugue_const[MAX_GPUS];

#define SWAP32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u)   | \
      (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

static volatile bool init[MAX_GPUS] = { false };

extern int scanhash_fugue256(int thr_id, uint32_t *pdata, uint32_t *ptarget,
	uint32_t max_nonce, uint32_t *hashes_done)
{
	uint32_t start_nonce = pdata[19]++;
	unsigned int intensity = (device_sm[device_map[thr_id]] > 500) ? 22 : 19;
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, 1 << intensity); // 256*256*8
	throughput = min(throughput, max_nonce - start_nonce);

	if (opt_benchmark)
		ptarget[7] = 0xf;

	// init
	if(!init[thr_id])
	{
		fugue256_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	// Endian Drehung ist notwendig
	uint32_t endiandata[20];
	for (int kk=0; kk < 20; kk++)
		be32enc(&endiandata[kk], pdata[kk]);

	// Context mit dem Endian gedrehten Blockheader vorbereiten (Nonce wird später ersetzt)
	fugue256_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		// GPU
		uint32_t foundNounce = 0xFFFFFFFF;
		fugue256_cpu_hash(thr_id, throughput, pdata[19], NULL, &foundNounce);

		if(stop_mining) {mining_has_stopped[thr_id] = true; pthread_exit(nullptr);}
		if(foundNounce < 0xffffffff)
		{
			uint32_t hash[8];
			const uint32_t Htarg = ptarget[7];

			endiandata[19] = SWAP32(foundNounce);
			sph_fugue256_context ctx_fugue;
			sph_fugue256_init(&ctx_fugue);
			sph_fugue256 (&ctx_fugue, endiandata, 80);
			sph_fugue256_close(&ctx_fugue, &hash);

			if (hash[7] <= Htarg && fulltest(hash, ptarget))
			{
				pdata[19] = foundNounce;
				*hashes_done = foundNounce - start_nonce + 1;
				return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", device_map[thr_id], foundNounce);
			}
		}

		pdata[19] += throughput;
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			applog(LOG_ERR, "GPU #%d: %s", device_map[thr_id], cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	} while (!work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - start_nonce + 1;
	return 0;
}

void fugue256_hash(unsigned char* output, const unsigned char* input, int len)
{
	sph_fugue256_context ctx;

	sph_fugue256_init(&ctx);
	sph_fugue256(&ctx, input, len);
	sph_fugue256_close(&ctx, (void *)output);
}
