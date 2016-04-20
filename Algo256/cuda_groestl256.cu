#include <memory.h>
#include "cuda_helper.h"

static uint32_t *d_GNonce[MAX_GPUS];

__constant__ uint32_t pTarget[8];

#define B32_0(x)    __byte_perm(x, 0, 0x4440)
//((x) & 0xFF)
#define B32_1(x)    __byte_perm(x, 0, 0x4441)
//(((x) >> 8) & 0xFF)
#define B32_2(x)    __byte_perm(x, 0, 0x4442)
//(((x) >> 16) & 0xFF)
#define B32_3(x)    __byte_perm(x, 0, 0x4443)
//((x) >> 24)

#define MAXWELL_OR_FERMI 1
#if MAXWELL_OR_FERMI
	#define USE_SHARED 1
	// Maxwell and Fermi cards get the best speed with SHARED access it seems.
	#if USE_SHARED
	#define T0up(x) (*(mixtabs + (    (x))))
	#define T0dn(x) (*(mixtabs + (256+(x))))
	#define T1up(x) (*(mixtabs + (512+(x))))
	#define T1dn(x) (*(mixtabs + (768+(x))))
	#define T2up(x) (*(mixtabs + (1024+(x))))
	#define T2dn(x) (*(mixtabs + (1280+(x))))
	#define T3up(x) (*(mixtabs + (1536+(x))))
	#define T3dn(x) (*(mixtabs + (1792+(x))))
	#else
	#define T0up(x) tex1Dfetch(t0up2, x)
	#define T0dn(x) tex1Dfetch(t0dn2, x)
	#define T1up(x) tex1Dfetch(t1up2, x)
	#define T1dn(x) tex1Dfetch(t1dn2, x)
	#define T2up(x) tex1Dfetch(t2up2, x)
	#define T2dn(x) tex1Dfetch(t2dn2, x)
	#define T3up(x) tex1Dfetch(t3up2, x)
	#define T3dn(x) tex1Dfetch(t3dn2, x)
	#endif
#else
	#define USE_SHARED 1
	// a healthy mix between shared and textured access provides the highest speed on Compute 3.0 and 3.5!
	#define T0up(x) (*((uint32_t*)mixtabs + (    (x))))
	#define T0dn(x) tex1Dfetch(t0dn2, x)
	#define T1up(x) tex1Dfetch(t1up2, x)
	#define T1dn(x) (*((uint32_t*)mixtabs + (768+(x))))
	#define T2up(x) tex1Dfetch(t2up2, x)
	#define T2dn(x) (*((uint32_t*)mixtabs + (1280+(x))))
	#define T3up(x) (*((uint32_t*)mixtabs + (1536+(x))))
	#define T3dn(x) tex1Dfetch(t3dn2, x)
#endif

texture<unsigned int, 1, cudaReadModeElementType> t0up2;
texture<unsigned int, 1, cudaReadModeElementType> t0dn2;
texture<unsigned int, 1, cudaReadModeElementType> t1up2;
texture<unsigned int, 1, cudaReadModeElementType> t1dn2;
texture<unsigned int, 1, cudaReadModeElementType> t2up2;
texture<unsigned int, 1, cudaReadModeElementType> t2dn2;
texture<unsigned int, 1, cudaReadModeElementType> t3up2;
texture<unsigned int, 1, cudaReadModeElementType> t3dn2;

#define RSTT(d0, d1, a, b0, b1, b2, b3, b4, b5, b6, b7) do { \
	t[d0] = T0up(B32_0(a[b0])) \
		^ T1up(B32_1(a[b1])) \
		^ T2up(B32_2(a[b2])) \
		^ T3up(B32_3(a[b3])) \
		^ T0dn(B32_0(a[b4])) \
		^ T1dn(B32_1(a[b5])) \
		^ T2dn(B32_2(a[b6])) \
		^ T3dn(B32_3(a[b7])); \
	t[d1] = T0dn(B32_0(a[b0])) \
		^ T1dn(B32_1(a[b1])) \
		^ T2dn(B32_2(a[b2])) \
		^ T3dn(B32_3(a[b3])) \
		^ T0up(B32_0(a[b4])) \
		^ T1up(B32_1(a[b5])) \
		^ T2up(B32_2(a[b6])) \
		^ T3up(B32_3(a[b7])); \
	} while (0)


extern uint32_t T0up_cpu[];
extern uint32_t T0dn_cpu[];
extern uint32_t T1up_cpu[];
extern uint32_t T1dn_cpu[];
extern uint32_t T2up_cpu[];
extern uint32_t T2dn_cpu[];
extern uint32_t T3up_cpu[];
extern uint32_t T3dn_cpu[];

__device__ __forceinline__
void groestl256_perm_P(uint32_t *const __restrict__ a, const uint32_t *const __restrict__ mixtabs)
{
	#pragma unroll 10
	for (int r = 0; r<10; r++)
	{
		uint32_t t[16];

		a[0x0] ^= 0x00 + r;
		a[0x2] ^= 0x10 + r;
		a[0x4] ^= 0x20 + r;
		a[0x6] ^= 0x30 + r;
		a[0x8] ^= 0x40 + r;
		a[0xA] ^= 0x50 + r;
		a[0xC] ^= 0x60 + r;
		a[0xE] ^= 0x70 + r;
		RSTT(0x0, 0x1, a, 0x0, 0x2, 0x4, 0x6, 0x9, 0xB, 0xD, 0xF);
		RSTT(0x2, 0x3, a, 0x2, 0x4, 0x6, 0x8, 0xB, 0xD, 0xF, 0x1);
		RSTT(0x4, 0x5, a, 0x4, 0x6, 0x8, 0xA, 0xD, 0xF, 0x1, 0x3);
		RSTT(0x6, 0x7, a, 0x6, 0x8, 0xA, 0xC, 0xF, 0x1, 0x3, 0x5);
		RSTT(0x8, 0x9, a, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7);
		RSTT(0xA, 0xB, a, 0xA, 0xC, 0xE, 0x0, 0x3, 0x5, 0x7, 0x9);
		RSTT(0xC, 0xD, a, 0xC, 0xE, 0x0, 0x2, 0x5, 0x7, 0x9, 0xB);
		RSTT(0xE, 0xF, a, 0xE, 0x0, 0x2, 0x4, 0x7, 0x9, 0xB, 0xD);

		#pragma unroll 16
		for (int k = 0; k<16; k++)
			a[k] = t[k];
	}
}
__device__ __forceinline__

void groestl256_perm_P_final(uint32_t *const __restrict__ a, const uint32_t *const __restrict__ mixtabs)
{
	uint32_t t[16];
#pragma unroll
	for(int r = 0; r<9; r++)
	{
		a[0x0] ^= 0x00 + r;
		a[0x2] ^= 0x10 + r;
		a[0x4] ^= 0x20 + r;
		a[0x6] ^= 0x30 + r;
		a[0x8] ^= 0x40 + r;
		a[0xA] ^= 0x50 + r;
		a[0xC] ^= 0x60 + r;
		a[0xE] ^= 0x70 + r;
		RSTT(0x0, 0x1, a, 0x0, 0x2, 0x4, 0x6, 0x9, 0xB, 0xD, 0xF);
		RSTT(0x2, 0x3, a, 0x2, 0x4, 0x6, 0x8, 0xB, 0xD, 0xF, 0x1);
		RSTT(0x4, 0x5, a, 0x4, 0x6, 0x8, 0xA, 0xD, 0xF, 0x1, 0x3);
		RSTT(0x6, 0x7, a, 0x6, 0x8, 0xA, 0xC, 0xF, 0x1, 0x3, 0x5);
		RSTT(0x8, 0x9, a, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7);
		RSTT(0xA, 0xB, a, 0xA, 0xC, 0xE, 0x0, 0x3, 0x5, 0x7, 0x9);
		RSTT(0xC, 0xD, a, 0xC, 0xE, 0x0, 0x2, 0x5, 0x7, 0x9, 0xB);
		RSTT(0xE, 0xF, a, 0xE, 0x0, 0x2, 0x4, 0x7, 0x9, 0xB, 0xD);

#pragma unroll 16
		for(int k = 0; k<16; k++)
			a[k] = t[k];
	}
	a[15] = T0dn(B32_0(a[14] ^ 0x79))
		^   T1dn(B32_1(a[ 0] ^ 0x09))
		^   T2dn(B32_2(a[ 2] ^ 0x19))
		^   T3dn(B32_3(a[ 4] ^ 0x29))
		^   T0up(B32_0(a[ 7]))
		^   T1up(B32_1(a[ 9]))
		^   T2up(B32_2(a[11]))
		^   T3up(B32_3(a[13]));
}

__device__ __forceinline__
void groestl256_perm_Q(uint32_t *const __restrict__ a, const uint32_t *const __restrict__ mixtabs)
{
	#pragma unroll
	for (uint32_t r = 0; r<0x0a000000; r+=0x01000000)
	{
		uint32_t t[16];

		a[0x0] ^= 0xFFFFFFFF;
		a[0x1] ^= ~r;
		a[0x2] ^= 0xFFFFFFFF;
		a[0x3] ^= r ^ 0xefffffff;
		a[0x4] ^= 0xFFFFFFFF;
		a[0x5] ^= r ^ 0xdfffffff;
		a[0x6] ^= 0xFFFFFFFF;
		a[0x7] ^= r ^ 0xcfffffff;
		a[0x8] ^= 0xFFFFFFFF;
		a[0x9] ^= r ^ 0xbfffffff;
		a[0xA] ^= 0xFFFFFFFF;
		a[0xB] ^= r ^ 0xafffffff;
		a[0xC] ^= 0xFFFFFFFF;
		a[0xD] ^= r ^ 0x9fffffff;
		a[0xE] ^= 0xFFFFFFFF;
		a[0xF] ^= r ^ 0x8fffffff;
		RSTT(0x0, 0x1, a, 0x2, 0x6, 0xA, 0xE, 0x1, 0x5, 0x9, 0xD);
		RSTT(0x2, 0x3, a, 0x4, 0x8, 0xC, 0x0, 0x3, 0x7, 0xB, 0xF);
		RSTT(0x4, 0x5, a, 0x6, 0xA, 0xE, 0x2, 0x5, 0x9, 0xD, 0x1);
		RSTT(0x6, 0x7, a, 0x8, 0xC, 0x0, 0x4, 0x7, 0xB, 0xF, 0x3);
		RSTT(0x8, 0x9, a, 0xA, 0xE, 0x2, 0x6, 0x9, 0xD, 0x1, 0x5);
		RSTT(0xA, 0xB, a, 0xC, 0x0, 0x4, 0x8, 0xB, 0xF, 0x3, 0x7);
		RSTT(0xC, 0xD, a, 0xE, 0x2, 0x6, 0xA, 0xD, 0x1, 0x5, 0x9);
		RSTT(0xE, 0xF, a, 0x0, 0x4, 0x8, 0xC, 0xF, 0x3, 0x7, 0xB);

		#pragma unroll
		for (int k = 0; k<16; k++)
			a[k] = t[k];
	}
}

__global__ __launch_bounds__(256,1)
void groestl256_gpu_hash32(uint32_t threads, uint32_t startNounce, const uint64_t *const __restrict__ outputHash, uint32_t *const __restrict__ nonceVector)
{
#if USE_SHARED
	__shared__ uint32_t mixtabs[2048];

	if (threadIdx.x < 256) {
		*(mixtabs + (threadIdx.x)) = tex1Dfetch(t0up2, threadIdx.x);
		*(mixtabs + (256 + threadIdx.x)) = tex1Dfetch(t0dn2, threadIdx.x);
		*(mixtabs + (512 + threadIdx.x)) = tex1Dfetch(t1up2, threadIdx.x);
		*(mixtabs + (768 + threadIdx.x)) = tex1Dfetch(t1dn2, threadIdx.x);
		*(mixtabs + (1024 + threadIdx.x)) = tex1Dfetch(t2up2, threadIdx.x);
		*(mixtabs + (1280 + threadIdx.x)) = tex1Dfetch(t2dn2, threadIdx.x);
		*(mixtabs + (1536 + threadIdx.x)) = tex1Dfetch(t3up2, threadIdx.x);
		*(mixtabs + (1792 + threadIdx.x)) = tex1Dfetch(t3dn2, threadIdx.x);
	}

	__syncthreads();
#endif

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		// GROESTL
		uint32_t message[16];
		uint32_t state[16];

		#pragma unroll
		for (int k = 0; k<4; k++)
			LOHI(message[2*k], message[2*k+1], outputHash[k*threads+thread]);

		#pragma unroll
		for (int k = 9; k<15; k++)
			message[k] = 0;

		message[8] = 0x80;
		message[15] = 0x01000000;

		#pragma unroll 16
		for (int u = 0; u<16; u++)
			state[u] = message[u];

		state[15] ^= 0x10000;

		// Perm

#if USE_SHARED
		groestl256_perm_P(state, mixtabs);
		state[15] ^= 0x10000;
		groestl256_perm_Q(message, mixtabs);
#else
		groestl256_perm_P(state, NULL);
		state[15] ^= 0x10000;
		groestl256_perm_P(message, NULL);
#endif
		#pragma unroll 16
		for (int u = 0; u<16; u++) state[u] ^= message[u];
		#pragma unroll 16
		for (int u = 0; u<16; u++) message[u] = state[u];
#if USE_SHARED
		groestl256_perm_P_final(message, mixtabs);
#else
		groestl256_perm_P(message, NULL);
#endif
		state[15] ^= message[15];

		if (state[15] <= pTarget[7])
		{
			uint32_t tmp = atomicExch(&nonceVector[0], startNounce + thread);
			if(tmp!=0)
				nonceVector[1] = tmp;
		}
	}
}

#define texDef(texname, texmem, texsource, texsize) \
	unsigned int *texmem; \
	CUDA_SAFE_CALL(cudaMalloc(&texmem, texsize)); \
	CUDA_SAFE_CALL(cudaMemcpyAsync(texmem, texsource, texsize, cudaMemcpyHostToDevice, gpustream[thr_id])); \
	texname.normalized = 0; \
	texname.filterMode = cudaFilterModePoint; \
	texname.addressMode[0] = cudaAddressModeClamp; \
	{ cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>(); \
	  CUDA_SAFE_CALL(cudaBindTexture(NULL, &texname, texmem, &channelDesc, texsize)); }

__host__
void groestl256_cpu_init(int thr_id, uint32_t threads)
{

	// Texturen mit obigem Makro initialisieren
	texDef(t0up2, d_T0up, T0up_cpu, sizeof(uint32_t) * 256);
	texDef(t0dn2, d_T0dn, T0dn_cpu, sizeof(uint32_t) * 256);
	texDef(t1up2, d_T1up, T1up_cpu, sizeof(uint32_t) * 256);
	texDef(t1dn2, d_T1dn, T1dn_cpu, sizeof(uint32_t) * 256);
	texDef(t2up2, d_T2up, T2up_cpu, sizeof(uint32_t) * 256);
	texDef(t2dn2, d_T2dn, T2dn_cpu, sizeof(uint32_t) * 256);
	texDef(t3up2, d_T3up, T3up_cpu, sizeof(uint32_t) * 256);
	texDef(t3dn2, d_T3dn, T3dn_cpu, sizeof(uint32_t) * 256);

	CUDA_SAFE_CALL(cudaMalloc(&d_GNonce[thr_id], 2 * sizeof(uint32_t)));
}

__host__
void groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t *resultnonces)
{
	CUDA_SAFE_CALL(cudaMemsetAsync(d_GNonce[thr_id], 0, 2 * sizeof(uint32_t), gpustream[thr_id]));
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	groestl256_gpu_hash32<<<grid, block, 0, gpustream[thr_id]>>>(threads, startNounce, d_outputHash, d_GNonce[thr_id]);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaMemcpyAsync(resultnonces, d_GNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id]));
	CUDA_SAFE_CALL(cudaStreamSynchronize(gpustream[thr_id]));
}

__host__
void groestl256_setTarget(int thr_id, const void *pTargetIn)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(pTarget, pTargetIn, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
}
