#include "miner.h"
#include "cuda_helper.h"

extern void novo_cpu_init(int thr_id);
extern void novo_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, const uint32_t *const ms, uint32_t merkle, uint32_t time, uint32_t compacttarget, uint32_t *const h_nounce);
extern void novo_midstate(const uint32_t *data, uint32_t *midstate);

uint32_t rotr(uint32_t x, unsigned int n)
{
	return (x >> n) | (x << (32 - n));
}

void novo_hash(uint32_t *output, const uint32_t *data, uint32_t nonce, const uint32_t *midstate)
{
	int i;
	uint32_t s0, s1, t1, t2, maj, ch, a, b, c, d, e, f, g, h;
	uint32_t w[64];

	const uint32_t k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};
	const uint32_t hc[8] = {
		0xdfa9bf2cU, 0xb72074d4U, 0x6bb01122U, 0xd338e869U,
		0xaa3ff126U, 0x475bbf30U, 0x8fd52e5bU, 0x9f75c9adU
	};

	for (i = 0; i <= 15; i++)
	{
		w[i] = data[i + 16];
	}
	w[3] = nonce;
	for (i = 16; i <= 63; i++)
	{
		s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = midstate[0];
	b = midstate[1];
	c = midstate[2];
	d = midstate[3];
	e = midstate[4];
	f = midstate[5];
	g = midstate[6];
	h = midstate[7];
	for (i = 0; i <= 63; i++)
	{
		s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
		ch = (e & f) ^ ((~e) & g);
		t1 = h + s1 + ch + k[i] + w[i];
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	w[0] = a + midstate[0];
	w[1] = b + midstate[1];
	w[2] = c + midstate[2];
	w[3] = d + midstate[3];
	w[4] = e + midstate[4];
	w[5] = f + midstate[5];
	w[6] = g + midstate[6];
	w[7] = h + midstate[7];
	w[8] = 0x80000000U;
	for (i = 9; i <= 14; i++)
		w[i] = 0U;
	w[15] = 0x300U;
	for (i = 16; i <= 63; i++)
	{
		s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
		s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
		w[i] = w[i - 16] + s0 + w[i - 7] + s1;
	}
	a = hc[0];
	b = hc[1];
	c = hc[2];
	d = hc[3];
	e = hc[4];
	f = hc[5];
	g = hc[6];
	h = hc[7];
	for (i = 0; i <= 63; i++)
	{
		s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
		maj = (a & b) ^ (a & c) ^ (b & c);
		t2 = s0 + maj;
		s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
		ch = (e & f) ^ ((~e) & g);
		t1 = h + s1 + ch + k[i] + w[i];
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	be32enc(&output[0], a + hc[0]);
	be32enc(&output[1], b + hc[1]);
	be32enc(&output[2], c + hc[2]);
	be32enc(&output[3], d + hc[3]);
	be32enc(&output[4], e + hc[4]);
	be32enc(&output[5], f + hc[5]);
	be32enc(&output[6], g + hc[6]);
	be32enc(&output[7], h + hc[7]);
}


int scanhash_novo(int thr_id, uint32_t *pdata,
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
		novo_cpu_init(thr_id);
		CUDA_SAFE_CALL(cudaMallocHost(&h_nounce, 2 * sizeof(uint32_t)));
		mining_has_stopped[thr_id] = false;
		init = true;
	}

	uint32_t ms[8];

	sha256t_init(ms);
	sha256_transform(ms, pdata, 0);

	do
	{
		novo_cpu_hash(thr_id, throughput, pdata[19], ms, pdata[16], pdata[17], pdata[18], h_nounce);
		if(stop_mining) {mining_has_stopped[thr_id] = true; cudaStreamDestroy(gpustream[thr_id]); pthread_exit(nullptr);}
		if(h_nounce[0] != UINT32_MAX)
		{
			uint32_t vhash64[8]={0};

			novo_hash(vhash64, pdata, h_nounce[0], ms);
			if (!opt_verify || (vhash64[7] == 0 && fulltest(vhash64, ptarget)))
			{
				int res = 1;
				// check if there was some other ones...
				*hashes_done = pdata[19] - first_nonce + throughput;
				if (h_nounce[1] != 0xffffffff)
				{
					novo_hash(vhash64, pdata, h_nounce[1], ms);
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
