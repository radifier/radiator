#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include "miner.h"
#include "cuda_helper.h"

#define rotr(x, y) ((x >> y) | (x << (64 -  y)))

#define SHA512_256_R(a, b, c, d, e, f, g, h, w, k) \
	h = h + (rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41)) + (g ^ (e & (f ^ g))) + k + w; \
	d = d + h; \
	h = h + (rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39)) + ((a & b) | (c & (a | b)))

#define sha512_S0(x) (rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39))
#define sha512_S1(x) (rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41))
#define sha512_s0(x) (rotr(x,  1) ^ rotr(x,  8) ^ (x >> 7))
#define sha512_s1(x) (rotr(x, 19) ^ rotr(x, 61) ^ (x >>  6))
#define ch(x, y, z) ((x & (y ^ z)) ^ z)
#define Ma(x, y, z) ((x & (y | z)) | (y & z))

__constant__ uint64_t K[80] = {
	0x428a2f98d728ae22ull, 0x7137449123ef65cdull,
	0xb5c0fbcfec4d3b2full, 0xe9b5dba58189dbbcull,
	0x3956c25bf348b538ull, 0x59f111f1b605d019ull,
	0x923f82a4af194f9bull, 0xab1c5ed5da6d8118ull,
	0xd807aa98a3030242ull, 0x12835b0145706fbeull,
	0x243185be4ee4b28cull, 0x550c7dc3d5ffb4e2ull,
	0x72be5d74f27b896full, 0x80deb1fe3b1696b1ull,
	0x9bdc06a725c71235ull, 0xc19bf174cf692694ull,
	0xe49b69c19ef14ad2ull, 0xefbe4786384f25e3ull,
	0x0fc19dc68b8cd5b5ull, 0x240ca1cc77ac9c65ull,
	0x2de92c6f592b0275ull, 0x4a7484aa6ea6e483ull,
	0x5cb0a9dcbd41fbd4ull, 0x76f988da831153b5ull,
	0x983e5152ee66dfabull, 0xa831c66d2db43210ull,
	0xb00327c898fb213full, 0xbf597fc7beef0ee4ull,
	0xc6e00bf33da88fc2ull, 0xd5a79147930aa725ull,
	0x06ca6351e003826full, 0x142929670a0e6e70ull,
	0x27b70a8546d22ffcull, 0x2e1b21385c26c926ull,
	0x4d2c6dfc5ac42aedull, 0x53380d139d95b3dfull,
	0x650a73548baf63deull, 0x766a0abb3c77b2a8ull,
	0x81c2c92e47edaee6ull, 0x92722c851482353bull,
	0xa2bfe8a14cf10364ull, 0xa81a664bbc423001ull,
	0xc24b8b70d0f89791ull, 0xc76c51a30654be30ull,
	0xd192e819d6ef5218ull, 0xd69906245565a910ull,
	0xf40e35855771202aull, 0x106aa07032bbd1b8ull,
	0x19a4c116b8d2d0c8ull, 0x1e376c085141ab53ull,
	0x2748774cdf8eeb99ull, 0x34b0bcb5e19b48a8ull,
	0x391c0cb3c5c95a63ull, 0x4ed8aa4ae3418acbull,
	0x5b9cca4f7763e373ull, 0x682e6ff3d6b2b8a3ull,
	0x748f82ee5defb2fcull, 0x78a5636f43172f60ull,
	0x84c87814a1f0ab72ull, 0x8cc702081a6439ecull,
	0x90befffa23631e28ull, 0xa4506cebde82bde9ull,
	0xbef9a3f7b2c67915ull, 0xc67178f2e372532bull,
	0xca273eceea26619cull, 0xd186b8c721c0c207ull,
	0xeada7dd6cde0eb1eull, 0xf57d4f7fee6ed178ull,
	0x06f067aa72176fbaull, 0x0a637dc5a2c898a6ull,
	0x113f9804bef90daeull, 0x1b710b35131c471bull,
	0x28db77f523047d84ull, 0x32caab7b40c72493ull,
	0x3c9ebe0a15c9bebcull, 0x431d67c49c100d4cull,
	0x4cc5d4becb3e42b6ull, 0x597f299cfc657e2aull,
	0x5fcb6fab3ad6faecull, 0x6c44198c4a475817ull
};

__constant__ uint64_t H[80] = {
	0x22312194fc2bf72cull, 0x9f555fa3c84c64c2ull,
	0x2393b86b6f53b151ull, 0x963877195940eabdull,
	0x96283ee2a88effe3ull, 0xbe5e1e2553863992ull,
	0x2b0199fc2c85b8aaull, 0x0eb72ddc81c52ca2ull
};

void rad_cpu_init(int thr_id);
void rad_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, const uint64_t *const data, uint32_t *const h_nounce);


__constant__ uint32_t pTarget[8];
static uint32_t *d_result[MAX_GPUS];

#define TPB 512
#define NONCES_PER_THREAD 16

__global__ __launch_bounds__(TPB, 2)

void rad_gpu_hash(const uint32_t threads, const uint32_t startNounce, uint32_t *const result, uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3, uint64_t w4, uint64_t w5, uint64_t w6, uint64_t w7, uint64_t w8, uint64_t w9, uint64_t ctx_a, uint64_t ctx_b, uint64_t ctx_c, uint64_t ctx_d, uint64_t ctx_e, uint64_t ctx_f, uint64_t ctx_g, uint64_t ctx_h, uint64_t dXORe, uint64_t w3s1)
{
	const uint32_t threadindex = (blockDim.x * blockIdx.x + threadIdx.x);
	if (threadindex < threads)
	{
		uint64_t Vals[24];
		uint64_t *W = &Vals[8];
		const uint32_t numberofthreads = blockDim.x*gridDim.x;
		const uint32_t maxnonce = startNounce + threadindex + numberofthreads*NONCES_PER_THREAD - 1;

		#pragma unroll 
		for (uint32_t nonce = startNounce + threadindex; nonce-1 < maxnonce; nonce += numberofthreads)
		{
			W[0] = w0;
			W[1] = w1;
			W[2] = w2;
			W[3] = w3;
			W[4] = w4;
			W[5] = w5;
			W[6] = w6;
			W[7] = w7;
			W[8] = w8;
			W[9] = w9 + nonce;
			W[10] = 0x8000000000000000ull;
			W[14] = 0x8000000000000147ull;
			W[15] = 0x0000000000000280ull;

			Vals[0] = ctx_d;
			Vals[1] = ctx_e;
			Vals[2] = ctx_h;
			Vals[3] = ctx_g;
			Vals[4] = ctx_f;
			Vals[5] = ctx_a;
			Vals[6] = ctx_c;
			Vals[7] = ctx_b;
			
			Vals[3]+=W[9];
			Vals[6]+=W[9];

			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=((Vals[6] & dXORe) ^ Vals[1]);
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[11];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[12];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[13];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[14];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[15];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=W[9];
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[16];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[17];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[18];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[19];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[20];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=w3s1;
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[21];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[22];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=W[0];
			W[7]+=(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[23];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=(sha512_s0(W[9]));
			W[8]+=W[1];
			W[8]+=(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[24];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=0x4180000000000000ull;
			W[9]+=W[2];
			W[9]+=(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[25];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=W[3];
			W[10]+=(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[26];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]=W[4];
			W[11]+=(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[27];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]=W[5];
			W[12]+=(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[28];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]=W[6];
			W[13]+=(sha512_s1(W[11]));
			Vals[6]+=W[13];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[29];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=W[7];
			W[14]+=(sha512_s1(W[12]));
			Vals[7]+=W[14];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[30];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=(sha512_s0(W[0]));
			W[15]+=W[8];
			W[15]+=(sha512_s1(W[13]));
			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[31];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=(sha512_s0(W[1]));
			W[0]+=W[9];
			W[0]+=(sha512_s1(W[14]));
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[32];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=(sha512_s0(W[2]));
			W[1]+=W[10];
			W[1]+=(sha512_s1(W[15]));
			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[33];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s0(W[3]));
			W[2]+=W[11];
			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[34];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=(sha512_s0(W[4]));
			W[3]+=W[12];
			W[3]+=(sha512_s1(W[1]));
			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[35];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s0(W[5]));
			W[4]+=W[13];
			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[36];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=(sha512_s0(W[6]));
			W[5]+=W[14];
			W[5]+=(sha512_s1(W[3]));
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[37];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=(sha512_s0(W[7]));
			W[6]+=W[15];
			W[6]+=(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[38];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=(sha512_s0(W[8]));
			W[7]+=W[0];
			W[7]+=(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[39];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=(sha512_s0(W[9]));
			W[8]+=W[1];
			W[8]+=(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[40];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=(sha512_s0(W[10]));
			W[9]+=W[2];
			W[9]+=(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[41];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=(sha512_s0(W[11]));
			W[10]+=W[3];
			W[10]+=(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[42];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=(sha512_s0(W[12]));
			W[11]+=W[4];
			W[11]+=(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[43];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=(sha512_s0(W[13]));
			W[12]+=W[5];
			W[12]+=(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[44];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=(sha512_s0(W[14]));
			W[13]+=W[6];
			W[13]+=(sha512_s1(W[11]));
			Vals[6]+=W[13];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[45];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=(sha512_s0(W[15]));
			W[14]+=W[7];
			W[14]+=(sha512_s1(W[12]));
			Vals[7]+=W[14];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[46];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=(sha512_s0(W[0]));
			W[15]+=W[8];
			W[15]+=(sha512_s1(W[13]));
			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[47];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=(sha512_s0(W[1]));
			W[0]+=W[9];
			W[0]+=(sha512_s1(W[14]));
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[48];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=(sha512_s0(W[2]));
			W[1]+=W[10];
			W[1]+=(sha512_s1(W[15]));
			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[49];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s0(W[3]));
			W[2]+=W[11];
			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[50];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=(sha512_s0(W[4]));
			W[3]+=W[12];
			W[3]+=(sha512_s1(W[1]));
			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[51];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s0(W[5]));
			W[4]+=W[13];
			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[52];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=(sha512_s0(W[6]));
			W[5]+=W[14];
			W[5]+=(sha512_s1(W[3]));
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[53];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=(sha512_s0(W[7]));
			W[6]+=W[15];
			W[6]+=(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[54];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=(sha512_s0(W[8]));
			W[7]+=W[0];
			W[7]+=(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[55];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=(sha512_s0(W[9]));
			W[8]+=W[1];
			W[8]+=(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[56];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=(sha512_s0(W[10]));
			W[9]+=W[2];
			W[9]+=(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[57];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=(sha512_s0(W[11]));
			W[10]+=W[3];
			W[10]+=(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[58];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=(sha512_s0(W[12]));
			W[11]+=W[4];
			W[11]+=(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[59];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=(sha512_s0(W[13]));
			W[12]+=W[5];
			W[12]+=(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[60];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=(sha512_s0(W[14]));
			W[13]+=W[6];
			W[13]+=(sha512_s1(W[11]));
			Vals[6]+=W[13];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[61];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=(sha512_s0(W[15]));
			W[14]+=W[7];
			W[14]+=(sha512_s1(W[12]));
			Vals[7]+=W[14];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[62];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=(sha512_s0(W[0]));
			W[15]+=W[8];
			W[15]+=(sha512_s1(W[13]));
			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[63];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=(sha512_s0(W[1]));
			W[0]+=W[9];
			W[0]+=(sha512_s1(W[14]));
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[64];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=(sha512_s0(W[2]));
			W[1]+=W[10];
			W[1]+=(sha512_s1(W[15]));
			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[65];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s0(W[3]));
			W[2]+=W[11];
			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[66];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=(sha512_s0(W[4]));
			W[3]+=W[12];
			W[3]+=(sha512_s1(W[1]));
			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[67];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s0(W[5]));
			W[4]+=W[13];
			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[68];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=(sha512_s0(W[6]));
			W[5]+=W[14];
			W[5]+=(sha512_s1(W[3]));
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[69];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=(sha512_s0(W[7]));
			W[6]+=W[15];
			W[6]+=(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[70];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=(sha512_s0(W[8]));
			W[7]+=W[0];
			W[7]+=(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[71];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=(sha512_s0(W[9]));
			W[8]+=W[1];
			W[8]+=(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[72];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=(sha512_s0(W[10]));
			W[9]+=W[2];
			W[9]+=(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[73];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=(sha512_s0(W[11]));
			W[10]+=W[3];
			W[10]+=(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[74];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=(sha512_s0(W[12]));
			W[11]+=W[4];
			W[11]+=(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[75];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=(sha512_s0(W[13]));
			W[12]+=W[5];
			W[12]+=(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[76];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=(sha512_s0(W[14]));
			W[13]+=W[6];
			W[13]+=(sha512_s1(W[11]));
			Vals[6]+=W[13];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[77];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=(sha512_s0(W[15]));
			W[14]+=W[7];
			W[14]+=(sha512_s1(W[12]));
			Vals[7]+=W[14];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[78];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=(sha512_s0(W[0]));
			W[15]+=W[8];
			W[15]+=(sha512_s1(W[13]));
			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[79];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]=H[0]+Vals[5];
			W[1]=H[1]+Vals[7];
			W[2]=H[2]+Vals[6];
			W[3]=H[3]+Vals[0];
			W[4] = 0x8000000000000000ul;
			W[14] = 0x0000000000000083ul;
			W[15] = 0x0000000000000100ul;

			Vals[5]=H[0];
			Vals[7]=H[1];
			Vals[6]=H[2];
			Vals[0]=H[3];
			Vals[1]=H[4];
			Vals[4]=H[5];
			Vals[3]=H[6];
			Vals[2]=H[7];

			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[0];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[1];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[2];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[3];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=0xb956c25bf348b538ul;
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[5];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[6];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[7];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[8];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[9];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[10];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[11];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[12];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[13];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[14];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=0xc19bf174cf692794ul;
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=(sha512_s0(W[1]));
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[16];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=(sha512_s0(W[2]));
			W[1]+=0x20000000000804ul;
			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[17];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s0(W[3]));
			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[18];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=0x4180000000000000ul;
			W[3]+=(sha512_s1(W[1]));
			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[19];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[20];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]=(sha512_s1(W[3]));
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[21];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]=W[15]+(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[22];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]=W[0]+(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[23];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]=W[1]+(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[24];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]=W[2]+(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[25];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]=W[3]+(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[26];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]=W[4]+(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[27];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]=W[5]+(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[28];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]=W[6]+(sha512_s1(W[11]));
			Vals[6]+=W[13];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[29];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=W[7];
			W[14]+=(sha512_s1(W[12]));
			Vals[7]+=W[14];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[30];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=(sha512_s0(W[0]));
			W[15]+=W[8];
			W[15]+=(sha512_s1(W[13]));
			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[31];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=(sha512_s0(W[1]));
			W[0]+=W[9];
			W[0]+=(sha512_s1(W[14]));
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[32];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=(sha512_s0(W[2]));
			W[1]+=W[10];
			W[1]+=(sha512_s1(W[15]));
			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[33];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s0(W[3]));
			W[2]+=W[11];
			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[34];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=(sha512_s0(W[4]));
			W[3]+=W[12];
			W[3]+=(sha512_s1(W[1]));
			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[35];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s0(W[5]));
			W[4]+=W[13];
			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[36];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=(sha512_s0(W[6]));
			W[5]+=W[14];
			W[5]+=(sha512_s1(W[3]));
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[37];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=(sha512_s0(W[7]));
			W[6]+=W[15];
			W[6]+=(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[38];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=(sha512_s0(W[8]));
			W[7]+=W[0];
			W[7]+=(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[39];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=(sha512_s0(W[9]));
			W[8]+=W[1];
			W[8]+=(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[40];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=(sha512_s0(W[10]));
			W[9]+=W[2];
			W[9]+=(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[41];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=(sha512_s0(W[11]));
			W[10]+=W[3];
			W[10]+=(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[42];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=(sha512_s0(W[12]));
			W[11]+=W[4];
			W[11]+=(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[43];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=(sha512_s0(W[13]));
			W[12]+=W[5];
			W[12]+=(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[44];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=(sha512_s0(W[14]));
			W[13]+=W[6];
			W[13]+=(sha512_s1(W[11]));
			Vals[6]+=W[13];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[45];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=(sha512_s0(W[15]));
			W[14]+=W[7];
			W[14]+=(sha512_s1(W[12]));
			Vals[7]+=W[14];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[46];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=(sha512_s0(W[0]));
			W[15]+=W[8];
			W[15]+=(sha512_s1(W[13]));
			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[47];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=(sha512_s0(W[1]));
			W[0]+=W[9];
			W[0]+=(sha512_s1(W[14]));
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[48];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=(sha512_s0(W[2]));
			W[1]+=W[10];
			W[1]+=(sha512_s1(W[15]));
			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[49];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s0(W[3]));
			W[2]+=W[11];
			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[50];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=(sha512_s0(W[4]));
			W[3]+=W[12];
			W[3]+=(sha512_s1(W[1]));
			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[51];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s0(W[5]));
			W[4]+=W[13];
			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[52];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=(sha512_s0(W[6]));
			W[5]+=W[14];
			W[5]+=(sha512_s1(W[3]));
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[53];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=(sha512_s0(W[7]));
			W[6]+=W[15];
			W[6]+=(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[54];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=(sha512_s0(W[8]));
			W[7]+=W[0];
			W[7]+=(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[55];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=(sha512_s0(W[9]));
			W[8]+=W[1];
			W[8]+=(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[56];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=(sha512_s0(W[10]));
			W[9]+=W[2];
			W[9]+=(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[57];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=(sha512_s0(W[11]));
			W[10]+=W[3];
			W[10]+=(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[58];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=(sha512_s0(W[12]));
			W[11]+=W[4];
			W[11]+=(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[59];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=(sha512_s0(W[13]));
			W[12]+=W[5];
			W[12]+=(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[60];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=(sha512_s0(W[14]));
			W[13]+=W[6];
			W[13]+=(sha512_s1(W[11]));
			Vals[6]+=W[13];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[61];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=(sha512_s0(W[15]));
			W[14]+=W[7];
			W[14]+=(sha512_s1(W[12]));
			Vals[7]+=W[14];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[62];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=(sha512_s0(W[0]));
			W[15]+=W[8];
			W[15]+=(sha512_s1(W[13]));
			Vals[5]+=W[15];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[63];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=(sha512_s0(W[1]));
			W[0]+=W[9];
			W[0]+=(sha512_s1(W[14]));
			Vals[2]+=W[0];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[64];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=(sha512_s0(W[2]));
			W[1]+=W[10];
			W[1]+=(sha512_s1(W[15]));
			Vals[3]+=W[1];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[65];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=(sha512_s0(W[3]));
			W[2]+=W[11];
			W[2]+=(sha512_s1(W[0]));
			Vals[4]+=W[2];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[66];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=(sha512_s0(W[4]));
			W[3]+=W[12];
			W[3]+=(sha512_s1(W[1]));
			Vals[1]+=W[3];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[67];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=(sha512_s0(W[5]));
			W[4]+=W[13];
			W[4]+=(sha512_s1(W[2]));
			Vals[0]+=W[4];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[68];
			Vals[2]+=Vals[0];
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=(sha512_s0(W[6]));
			W[5]+=W[14];
			W[5]+=(sha512_s1(W[3]));
			Vals[6]+=W[5];
			Vals[6]+=(sha512_S1(Vals[2]));
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[69];
			Vals[3]+=Vals[6];
			Vals[6]+=(sha512_S0(Vals[0]));
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=(sha512_s0(W[7]));
			W[6]+=W[15];
			W[6]+=(sha512_s1(W[4]));
			Vals[7]+=W[6];
			Vals[7]+=(sha512_S1(Vals[3]));
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[70];
			Vals[4]+=Vals[7];
			Vals[7]+=(sha512_S0(Vals[6]));
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=(sha512_s0(W[8]));
			W[7]+=W[0];
			W[7]+=(sha512_s1(W[5]));
			Vals[5]+=W[7];
			Vals[5]+=(sha512_S1(Vals[4]));
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[71];
			Vals[1]+=Vals[5];
			Vals[5]+=(sha512_S0(Vals[7]));
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=(sha512_s0(W[9]));
			W[8]+=W[1];
			W[8]+=(sha512_s1(W[6]));
			Vals[2]+=W[8];
			Vals[2]+=(sha512_S1(Vals[1]));
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[72];
			Vals[0]+=Vals[2];
			Vals[2]+=(sha512_S0(Vals[5]));
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=(sha512_s0(W[10]));
			W[9]+=W[2];
			W[9]+=(sha512_s1(W[7]));
			Vals[3]+=W[9];
			Vals[3]+=(sha512_S1(Vals[0]));
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[73];
			Vals[6]+=Vals[3];
			Vals[3]+=(sha512_S0(Vals[2]));
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=(sha512_s0(W[11]));
			W[10]+=W[3];
			W[10]+=(sha512_s1(W[8]));
			Vals[4]+=W[10];
			Vals[4]+=(sha512_S1(Vals[6]));
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[74];
			Vals[7]+=Vals[4];
			Vals[4]+=(sha512_S0(Vals[3]));
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=(sha512_s0(W[12]));
			W[11]+=W[4];
			W[11]+=(sha512_s1(W[9]));
			Vals[1]+=W[11];
			Vals[1]+=(sha512_S1(Vals[7]));
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[75];
			Vals[5]+=Vals[1];
			Vals[1]+=(sha512_S0(Vals[4]));
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=(sha512_s0(W[13]));
			W[12]+=W[5];
			W[12]+=(sha512_s1(W[10]));
			Vals[0]+=W[12];
			Vals[0]+=(sha512_S1(Vals[5]));
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=(sha512_S0(Vals[1]));
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			uint32_t r = Vals[0];
			if (r == 0xdb80d28du)
			{
				uint32_t tmp = atomicCAS(result, 0xffffffff, nonce);
				if (tmp != 0xffffffff)
					result[1] = nonce;
			}
		} // nonce loop
	} // if thread<threads
}

__host__
void rad_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, const uint64_t *const data, uint32_t *const h_nounce)
{
	cudaMemsetAsync(d_result[thr_id], 0xff, 2 * sizeof(uint32_t), gpustream[thr_id]);

	uint64_t w0 = data[0] << 32 | data[0] >> 32;
	uint64_t w1 = data[1] << 32 | data[1] >> 32;
	uint64_t w2 = data[2] << 32 | data[2] >> 32;
	uint64_t w3 = data[3] << 32 | data[3] >> 32;
	uint64_t w4 = data[4] << 32 | data[4] >> 32;
	uint64_t w5 = data[5] << 32 | data[5] >> 32;
	uint64_t w6 = data[6] << 32 | data[6] >> 32;
	uint64_t w7 = data[7] << 32 | data[7] >> 32;
	uint64_t w8 = data[8] << 32 | data[8] >> 32;
	uint64_t w9 = data[9] << 32;

	uint64_t A, B, C, D, E, F, G, H;

	A = 0x22312194fc2bf72cull;
	B = 0x9f555fa3c84c64c2ull;
	C = 0x2393b86b6f53b151ull;
	D = 0x963877195940eabdull;
	E = 0x96283ee2a88effe3ull;
	F = 0xbe5e1e2553863992ull;
	G = 0x2b0199fc2c85b8aaull;
	H = 0x0eb72ddc81c52ca2ull;


	SHA512_256_R(A, B, C, D, E, F, G, H, w0, 0x428a2f98d728ae22ull);
	SHA512_256_R(H, A, B, C, D, E, F, G, w1, 0x7137449123ef65cdull);
	SHA512_256_R(G, H, A, B, C, D, E, F, w2, 0xb5c0fbcfec4d3b2full);
	SHA512_256_R(F, G, H, A, B, C, D, E, w3, 0xe9b5dba58189dbbcull);
	SHA512_256_R(E, F, G, H, A, B, C, D, w4, 0x3956c25bf348b538ull);
	SHA512_256_R(D, E, F, G, H, A, B, C, w5, 0x59f111f1b605d019ull);
	SHA512_256_R(C, D, E, F, G, H, A, B, w6, 0x923f82a4af194f9bull);
	SHA512_256_R(B, C, D, E, F, G, H, A, w7, 0xab1c5ed5da6d8118ull);
	SHA512_256_R(A, B, C, D, E, F, G, H, w8, 0xd807aa98a3030242ull);

	w0 += sha512_s0(w1);
	w1 += sha512_s0(w2) + 0x805000000000140aull;
	w2 += sha512_s0(w3);
	w3 += sha512_s0(w4) + sha512_s1(w1);
	w4 += sha512_s0(w5);
	w5 += sha512_s0(w6);
	w6 += sha512_s0(w7) + 0x0000000000000280ull;
	w7 += sha512_s0(w8);

	uint64_t dXORe = D ^ E;
	uint64_t w3s1 = sha512_s1(w3);

	G += sha512_S1(D) + 0x12835b0145706fbeull + ch(D, E, F);
	C += G;
	G += sha512_S0(H) + Ma(B, H, A);
	F += 0x8000000000000000ull + 0x243185be4ee4b28cull;

	dim3 grid((threads + TPB*NONCES_PER_THREAD - 1) / TPB / NONCES_PER_THREAD);
	dim3 block(TPB);
	rad_gpu_hash << <grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_result[thr_id], w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, A, B, C, D, E, F, G, H, dXORe, w3s1);
	cudaStreamSynchronize(gpustream[thr_id]);
	CUDA_SAFE_CALL(cudaMemcpyAsync(h_nounce, d_result[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id])); cudaStreamSynchronize(gpustream[thr_id]);
}

__host__
void rad_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_result[thr_id], 4 * sizeof(uint32_t)));
}
