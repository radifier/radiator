#include "cuda_helper.h"

#define CUBEHASH_ROUNDS 16 /* this is r for CubeHashr/b */
#define CUBEHASH_BLOCKBYTES 32 /* this is b for CubeHashr/b */
#define TPB 1024

#define ROTATEUPWARDS7(a)  ROTL32(a,7)
#define ROTATEUPWARDS11(a) ROTL32(a,11)

//#define SWAP(a,b) { uint32_t u = a; a = b; b = u; }
#define SWAP(a,b) { a ^= b; b ^=a; a ^=b;}
__device__ __forceinline__ void rrounds(uint32_t x[2][2][2][2][2])
{
	int r;
	int j;
	int k;
	int l;
	int m;

	#pragma unroll 2
	for (r = 0; r < CUBEHASH_ROUNDS; ++r) {

		/* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

		/* "swap x_00klm with x_01klm" */
#pragma unroll 2
		for (k = 0; k < 2; ++k)
#pragma unroll 2
			for (l = 0; l < 2; ++l)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[0][0][k][l][m], x[0][1][k][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[1][j][k][0][m], x[1][j][k][1][m])

					/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

		/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (l = 0; l < 2; ++l)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[0][j][0][l][m], x[0][j][1][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
					SWAP(x[1][j][k][l][0], x[1][j][k][l][1])

	}
}

__device__ __forceinline__ void rrounds_final(uint32_t x[2][2][2][2][2])
{
	int r;
	int j;
	int k;
	int l;
	int m;

#pragma unroll 2
	for(r = 0; r < CUBEHASH_ROUNDS-1; ++r)
	{

		/* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
		for(j = 0; j < 2; ++j)
#pragma unroll 2
			for(k = 0; k < 2; ++k)
#pragma unroll 2
				for(l = 0; l < 2; ++l)
#pragma unroll 2
					for(m = 0; m < 2; ++m)
						x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
		for(j = 0; j < 2; ++j)
#pragma unroll 2
			for(k = 0; k < 2; ++k)
#pragma unroll 2
				for(l = 0; l < 2; ++l)
#pragma unroll 2
					for(m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

		/* "swap x_00klm with x_01klm" */
#pragma unroll 2
		for(k = 0; k < 2; ++k)
#pragma unroll 2
			for(l = 0; l < 2; ++l)
#pragma unroll 2
				for(m = 0; m < 2; ++m)
					SWAP(x[0][0][k][l][m], x[0][1][k][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for(j = 0; j < 2; ++j)
#pragma unroll 2
						for(k = 0; k < 2; ++k)
#pragma unroll 2
							for(l = 0; l < 2; ++l)
#pragma unroll 2
								for(m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
		for(j = 0; j < 2; ++j)
#pragma unroll 2
			for(k = 0; k < 2; ++k)
#pragma unroll 2
				for(m = 0; m < 2; ++m)
					SWAP(x[1][j][k][0][m], x[1][j][k][1][m])

					/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
					for(j = 0; j < 2; ++j)
#pragma unroll 2
						for(k = 0; k < 2; ++k)
#pragma unroll 2
							for(l = 0; l < 2; ++l)
#pragma unroll 2
								for(m = 0; m < 2; ++m)
									x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
		for(j = 0; j < 2; ++j)
#pragma unroll 2
			for(k = 0; k < 2; ++k)
#pragma unroll 2
				for(l = 0; l < 2; ++l)
#pragma unroll 2
					for(m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

		/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
		for(j = 0; j < 2; ++j)
#pragma unroll 2
			for(l = 0; l < 2; ++l)
#pragma unroll 2
				for(m = 0; m < 2; ++m)
					SWAP(x[0][j][0][l][m], x[0][j][1][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for(j = 0; j < 2; ++j)
#pragma unroll 2
						for(k = 0; k < 2; ++k)
#pragma unroll 2
							for(l = 0; l < 2; ++l)
#pragma unroll 2
								for(m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
		for(j = 0; j < 2; ++j)
#pragma unroll 2
			for(k = 0; k < 2; ++k)
#pragma unroll 2
				for(l = 0; l < 2; ++l)
					SWAP(x[1][j][k][l][0], x[1][j][k][l][1])

	}
	/* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
		for(k = 0; k < 2; ++k)
#pragma unroll 2
			for(l = 0; l < 2; ++l)
#pragma unroll 2
				for(m = 0; m < 2; ++m)
					x[1][0][k][l][m] += x[0][0][k][l][m];

	/* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
		for(k = 0; k < 2; ++k)
#pragma unroll 2
			for(l = 0; l < 2; ++l)
#pragma unroll 2
				for(m = 0; m < 2; ++m)
					x[0][1][k][l][m] = ROTATEUPWARDS7(x[0][1][k][l][m]);

	/* "swap x_00klm with x_01klm" */
#pragma unroll 2
	for(k = 0; k < 2; ++k)
#pragma unroll 2
		for(l = 0; l < 2; ++l)
#pragma unroll 2
			for(m = 0; m < 2; ++m)
				x[0][0][k][l][m] = x[0][1][k][l][m];

				/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for(k = 0; k < 2; ++k)
#pragma unroll 2
						for(l = 0; l < 2; ++l)
#pragma unroll 2
							for(m = 0; m < 2; ++m)
								x[0][0][k][l][m] ^= x[1][0][k][l][m];

	/* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
		for(k = 0; k < 2; ++k)
#pragma unroll 2
			for(m = 0; m < 2; ++m)
				SWAP(x[1][0][k][0][m], x[1][0][k][1][m])

				/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
					for(k = 0; k < 2; ++k)
#pragma unroll 2
						for(l = 0; l < 2; ++l)
#pragma unroll 2
							for(m = 0; m < 2; ++m)
								x[1][0][k][l][m] += x[0][0][k][l][m];

	/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
		for(k = 0; k < 2; ++k)
#pragma unroll 2
			for(l = 0; l < 2; ++l)
#pragma unroll 2
				for(m = 0; m < 2; ++m)
					x[0][0][k][l][m] = ROTATEUPWARDS11(x[0][0][k][l][m]);

	/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
		for(l = 0; l < 2; ++l)
#pragma unroll 2
			for(m = 0; m < 2; ++m)
				SWAP(x[0][0][0][l][m], x[0][0][1][l][m])

				/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for(k = 0; k < 2; ++k)
#pragma unroll 2
						for(l = 0; l < 2; ++l)
#pragma unroll 2
							for(m = 0; m < 2; ++m)
								x[0][0][k][l][m] ^= x[1][0][k][l][m];
}

__device__ __forceinline__ void block_tox(const uint32_t *in, uint32_t x[2][2][2][2][2])
{
	x[0][0][0][0][0] ^= in[0];
	x[0][0][0][0][1] ^= in[1];
	x[0][0][0][1][0] ^= in[2];
	x[0][0][0][1][1] ^= in[3];
	x[0][0][1][0][0] ^= in[4];
	x[0][0][1][0][1] ^= in[5];
	x[0][0][1][1][0] ^= in[6];
	x[0][0][1][1][1] ^= in[7];
}

__device__ __forceinline__ void hash_fromx(uint32_t *out, uint32_t x[2][2][2][2][2])
{
	out[0] = x[0][0][0][0][0];
	out[1] = x[0][0][0][0][1];
	out[2] = x[0][0][0][1][0];
	out[3] = x[0][0][0][1][1];
	out[4] = x[0][0][1][0][0];
	out[5] = x[0][0][1][0][1];
	out[6] = x[0][0][1][1][0];
	out[7] = x[0][0][1][1][1];

}

void __device__ __forceinline__ Update32(uint32_t x[2][2][2][2][2], const uint32_t *data)
{
	/* "xor the block into the first b bytes of the state" */
	/* "and then transform the state invertibly through r identical rounds" */
	block_tox(data, x);
	rrounds(x);
}

void __device__ __forceinline__ Update32_const(uint32_t x[2][2][2][2][2])
{
	x[0][0][0][0][0] ^= 0x80;
	rrounds(x);
}



void __device__ __forceinline__ Final(uint32_t x[2][2][2][2][2], uint32_t *hashval)
{
	int i;

	/* "the integer 1 is xored into the last state word x_11111" */
	x[1][1][1][1][1] ^= 1;

	/* "the state is then transformed invertibly through 10r identical rounds" */
	#pragma unroll 2
	for (i = 0; i < 9; ++i)
		rrounds(x);
	rrounds_final(x);
	/* "output the first h/8 bytes of the state" */
	hash_fromx(hashval, x);
}


#if __CUDA_ARCH__ <500
__global__	__launch_bounds__(TPB, 1)
#else 
__global__	__launch_bounds__(TPB, 1)
#endif
void cubehash256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
    uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//    if (thread < threads)
    {

        uint2 Hash[4];

		
		Hash[0]= __ldg(&g_hash[thread]);
		Hash[1] = __ldg(&g_hash[thread + 1 * threads]);	//	LOHI(Hash[2], Hash[3], __ldg(&g_hash[thread + 1 * threads]));
		Hash[2] = __ldg(&g_hash[thread + 2 * threads]);	//	LOHI(Hash[4], Hash[5], __ldg(&g_hash[thread + 2 * threads]));
		Hash[3] = __ldg(&g_hash[thread + 3 * threads]);	//	LOHI(Hash[6], Hash[7], __ldg(&g_hash[thread + 3 * threads]));

		uint32_t x[2][2][2][2][2];

		x[1][0][0][1][0] = 0xD89041C3 + (0xEA2BD4B4 ^ Hash[0].x);
		x[1][0][0][1][1] = 0x6107FBD5 + (0xCCD6F29F ^ Hash[0].y);
		x[1][0][0][0][0] = 0x6C859D41 + (0x63117E71 ^ Hash[1].x);
		x[1][0][0][0][1] = 0xF0B26679 + (0x35481EAE ^ Hash[1].y);
		x[1][0][1][1][0] = 0x09392549 + (0x22512D5B ^ Hash[2].x);
		x[1][0][1][1][1] = 0x5FA25603 + (0xE5D94E63 ^ Hash[2].y);
		x[1][0][1][0][0] = 0x65C892FD + (0x7E624131 ^ Hash[3].x);
		x[1][0][1][0][1] = 0x93CB6285 + (0xF4CC12BE ^ Hash[3].y);

		x[1][1][0][1][0] = 0xEDC36C44;
		x[1][1][0][1][1] = 0xE0FA6ED0;
		x[1][1][0][0][0] = 0x47BCCC12;
		x[1][1][0][0][1] = 0xB88721B1;
		x[1][1][1][1][0] = 0x3E4E478F;
		x[1][1][1][1][1] = 0xD9AF5859;
		x[1][1][1][0][0] = 0xE35BA4AF;
		x[1][1][1][0][1] = 0x16E927B5;

		x[0][0][0][0][0] = ROTATEUPWARDS7(0xC2D0B696) ^ x[1][0][0][1][0];
		x[0][0][0][0][1] = ROTATEUPWARDS7(0x42AF2070) ^ x[1][0][0][1][1];
		x[0][0][0][1][0] = ROTATEUPWARDS7(0xD0720C35) ^ x[1][0][0][0][0];
		x[0][0][0][1][1] = ROTATEUPWARDS7(0x3361DA8C) ^ x[1][0][0][0][1];
		x[0][0][1][0][0] = ROTATEUPWARDS7(0x28CCECA4) ^ x[1][0][1][1][0];
		x[0][0][1][0][1] = ROTATEUPWARDS7(0x8EF8AD83) ^ x[1][0][1][1][1];
		x[0][0][1][1][0] = ROTATEUPWARDS7(0x4680AC00) ^ x[1][0][1][0][0];
		x[0][0][1][1][1] = ROTATEUPWARDS7(0x40E5FBAB) ^ x[1][0][1][0][1];

		x[0][1][0][0][0] = ROTATEUPWARDS7(0xEA2BD4B4 ^ Hash[0].x) ^ 0xEDC36C44;
		x[0][1][0][0][1] = ROTATEUPWARDS7(0xCCD6F29F ^ Hash[0].y) ^ 0xE0FA6ED0;
		x[0][1][0][1][0] = ROTATEUPWARDS7(0x63117E71 ^ Hash[1].x) ^ 0x47BCCC12;
		x[0][1][0][1][1] = ROTATEUPWARDS7(0x35481EAE ^ Hash[1].y) ^ 0xB88721B1;
		x[0][1][1][0][0] = ROTATEUPWARDS7(0x22512D5B ^ Hash[2].x) ^ 0x3E4E478F;
		x[0][1][1][0][1] = ROTATEUPWARDS7(0xE5D94E63 ^ Hash[2].y) ^ 0xD9AF5859;
		x[0][1][1][1][0] = ROTATEUPWARDS7(0x7E624131 ^ Hash[3].x) ^ 0xE35BA4AF;
		x[0][1][1][1][1] = ROTATEUPWARDS7(0xF4CC12BE ^ Hash[3].y) ^ 0x16E927B5;
		int r;
		int j,k,l,m;

		/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
		for(j = 0; j < 2; ++j)
#pragma unroll 2
			for(k = 0; k < 2; ++k)
#pragma unroll 2
				for(l = 0; l < 2; ++l)
#pragma unroll 2
					for(m = 0; m < 2; ++m)
						x[1][j][k][l][m] += x[0][j][k][l][m];

		/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

		/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (l = 0; l < 2; ++l)
#pragma unroll 2
				for (m = 0; m < 2; ++m)
					SWAP(x[0][j][0][l][m], x[0][j][1][l][m])

					/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
					for (j = 0; j < 2; ++j)
#pragma unroll 2
						for (k = 0; k < 2; ++k)
#pragma unroll 2
							for (l = 0; l < 2; ++l)
#pragma unroll 2
								for (m = 0; m < 2; ++m)
									x[0][j][k][l][m] ^= x[1][j][k][l][m];

		/* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
		for (j = 0; j < 2; ++j)
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
					SWAP(x[1][j][k][l][0], x[1][j][k][l][1])


#pragma unroll
		for (r = 1; r < CUBEHASH_ROUNDS; ++r) 
		{

			/* "add x_0jklm into x_1jklmn modulo 2^32" */
#pragma unroll 2
			for (j = 0; j < 2; ++j)
#pragma unroll 2
				for (k = 0; k < 2; ++k)
#pragma unroll 2
					for (l = 0; l < 2; ++l)
#pragma unroll 2
						for (m = 0; m < 2; ++m)
							x[1][j][k][l][m] += x[0][j][k][l][m];

			/* "rotate x_0jklm upwards by 7 bits" */
#pragma unroll 2
			for (j = 0; j < 2; ++j)
#pragma unroll 2
				for (k = 0; k < 2; ++k)
#pragma unroll 2
					for (l = 0; l < 2; ++l)
#pragma unroll 2
						for (m = 0; m < 2; ++m)
							x[0][j][k][l][m] = ROTATEUPWARDS7(x[0][j][k][l][m]);

			/* "swap x_00klm with x_01klm" */
#pragma unroll 2
			for (k = 0; k < 2; ++k)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						SWAP(x[0][0][k][l][m], x[0][1][k][l][m])

						/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
						for (j = 0; j < 2; ++j)
#pragma unroll 2
							for (k = 0; k < 2; ++k)
#pragma unroll 2
								for (l = 0; l < 2; ++l)
#pragma unroll 2
									for (m = 0; m < 2; ++m)
										x[0][j][k][l][m] ^= x[1][j][k][l][m];

			/* "swap x_1jk0m with x_1jk1m" */
#pragma unroll 2
			for (j = 0; j < 2; ++j)
#pragma unroll 2
				for (k = 0; k < 2; ++k)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						SWAP(x[1][j][k][0][m], x[1][j][k][1][m])

						/* "add x_0jklm into x_1jklm modulo 2^32" */
#pragma unroll 2
						for (j = 0; j < 2; ++j)
#pragma unroll 2
							for (k = 0; k < 2; ++k)
#pragma unroll 2
								for (l = 0; l < 2; ++l)
#pragma unroll 2
									for (m = 0; m < 2; ++m)
										x[1][j][k][l][m] += x[0][j][k][l][m];

			/* "rotate x_0jklm upwards by 11 bits" */
#pragma unroll 2
			for (j = 0; j < 2; ++j)
#pragma unroll 2
				for (k = 0; k < 2; ++k)
#pragma unroll 2
					for (l = 0; l < 2; ++l)
#pragma unroll 2
						for (m = 0; m < 2; ++m)
							x[0][j][k][l][m] = ROTATEUPWARDS11(x[0][j][k][l][m]);

			/* "swap x_0j0lm with x_0j1lm" */
#pragma unroll 2
			for (j = 0; j < 2; ++j)
#pragma unroll 2
				for (l = 0; l < 2; ++l)
#pragma unroll 2
					for (m = 0; m < 2; ++m)
						SWAP(x[0][j][0][l][m], x[0][j][1][l][m])

						/* "xor x_1jklm into x_0jklm" */
#pragma unroll 2
						for (j = 0; j < 2; ++j)
#pragma unroll 2
							for (k = 0; k < 2; ++k)
#pragma unroll 2
								for (l = 0; l < 2; ++l)
#pragma unroll 2
									for (m = 0; m < 2; ++m)
										x[0][j][k][l][m] ^= x[1][j][k][l][m];

			/* "swap x_1jkl0 with x_1jkl1" */
#pragma unroll 2
			for (j = 0; j < 2; ++j)
#pragma unroll 2
				for (k = 0; k < 2; ++k)
#pragma unroll 2
					for (l = 0; l < 2; ++l)
						SWAP(x[1][j][k][l][0], x[1][j][k][l][1])

		}



		x[0][0][0][0][0] ^= 0x80;
		rrounds(x);

		Final(x, (uint32_t *)Hash);

		g_hash[thread] =               ((uint2*)Hash)[0];
		g_hash[1 * threads + thread] = ((uint2*)Hash)[1];
		g_hash[2 * threads + thread] = ((uint2*)Hash)[2];
		g_hash[3 * threads + thread] = ((uint2*)Hash)[3];
    }
}


__host__
void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash)
{

    // berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + (TPB) - 1) / (TPB));
	dim3 block(TPB);

	cubehash256_gpu_hash_32 << <grid, block, 0, gpustream[thr_id] >> >(threads, startNounce, (uint2 *)d_hash);
	CUDA_SAFE_CALL(cudaGetLastError());
}
