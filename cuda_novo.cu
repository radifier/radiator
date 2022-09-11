#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#include "miner.h"
#include "cuda_helper.h"

#define rotr(x, y) ((x >> y) | (x << (32 -  y)))

#define rotate(x,y) ((x<<y) | (x>>(sizeof(x)*8-y)))
#define R(a, b, c, d, e, f, g, h, w, k) \
	h = h + (rotate(e, 26) ^ rotate(e, 21) ^ rotate(e, 7)) + (g ^ (e & (f ^ g))) + k + w; \
	d = d + h; \
	h = h + (rotate(a, 30) ^ rotate(a, 19) ^ rotate(a, 10)) + ((a & b) | (c & (a | b)))

#define ch(x, y, z)    ((x & (y ^ z)) ^ z)
#define Ma(x, y, z)    ((x & (y | z)) | (y & z))
#define S0(x)          (rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22))
#define S1(x)          (rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25))
#define s0(x)          (rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3))
#define s1(x)          (rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10))

__device__ const uint32_t K[64] = {
	0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
	0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
	0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
	0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
	0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
	0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
	0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
	0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

void novo_cpu_init(int thr_id);
void novo_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, const uint32_t *const ms, uint32_t merkle, uint32_t time, uint32_t compacttarget, uint32_t *const h_nounce);


__constant__ uint32_t pTarget[8];
static uint32_t *d_result[MAX_GPUS];

#define TPB 512
#define NONCES_PER_THREAD 32

__global__ __launch_bounds__(TPB, 2)

void novo_gpu_hash(const uint32_t threads, const uint32_t startNounce, uint32_t *const result, const uint32_t state0, const uint state1, const uint state2, const uint state3, const uint32_t state4, const uint state5, const uint state6, const uint state7, const uint32_t b1, const uint c1, const uint32_t f1, const uint g1, const uint h1, const uint32_t fw0, const uint fw1, const uint fw2, const uint fw3, const uint fw15, const uint fw01r, const uint32_t D1A, const uint C1addK5, const uint B1addK6, const uint32_t W16addK16, const uint W17addK17, const uint32_t PreVal4addT1, const uint Preval0)
{
	const uint32_t threadindex = (blockDim.x * blockIdx.x + threadIdx.x);
	if (threadindex < threads)
	{
		uint32_t Vals[24];
		uint32_t *W = &Vals[8];
		const uint32_t numberofthreads = blockDim.x*gridDim.x;
		const uint32_t maxnonce = startNounce + threadindex + numberofthreads*NONCES_PER_THREAD - 1;

		#pragma unroll 
		for (uint32_t nonce = startNounce + threadindex; nonce-1 < maxnonce; nonce += numberofthreads)
		{

			Vals[5]=Preval0;
			Vals[5]+=nonce;

			Vals[0]=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],b1,c1);
			Vals[0]+=D1A;

			Vals[2]=Vals[0];
			Vals[2]+=h1;

			Vals[1]=PreVal4addT1;
			Vals[1]+=nonce;
			Vals[0]+=S0(Vals[1]);

			Vals[6]=C1addK5;
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],b1);

			Vals[3]=Vals[6];
			Vals[3]+=g1;
			Vals[0]+=Ma(g1,Vals[1],f1);
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(f1,Vals[0],Vals[1]);

			Vals[7]=B1addK6;
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);

			Vals[4]=Vals[7];
			Vals[4]+=f1;

			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[7];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[8];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[9];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[10];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[11];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[12];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[13];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[14];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=0xc19bf5f4U;
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=W16addK16;
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=W17addK17;
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]=s0(nonce);
			W[2]+=fw2;
			Vals[4]+=W[2];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[18];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]=nonce;
			W[3]+=fw3;
			Vals[1]+=W[3];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[19];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]=s1(W[2]);
			W[4]+=0x80000000U;
			Vals[0]+=W[4];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[20];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]=s1(W[3]);
			Vals[6]+=W[5];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[21];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]=s1(W[4]);
			W[6]+=0x00000480U;
			Vals[7]+=W[6];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[22];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]=s1(W[5]);
			W[7]+=fw0;
			Vals[5]+=W[7];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[23];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]=s1(W[6]);
			W[8]+=fw1;
			Vals[2]+=W[8];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[24];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]=W[2];
			W[9]+=s1(W[7]);
			Vals[3]+=W[9];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[25];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]=W[3];
			W[10]+=s1(W[8]);
			Vals[4]+=W[10];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[26];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]=W[4];
			W[11]+=s1(W[9]);
			Vals[1]+=W[11];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[27];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]=W[5];
			W[12]+=s1(W[10]);
			Vals[0]+=W[12];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[28];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]=W[6];
			W[13]+=s1(W[11]);
			Vals[6]+=W[13];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[29];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]=0x01200099U;
			W[14]+=W[7];
			W[14]+=s1(W[12]);
			Vals[7]+=W[14];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[30];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]=fw15;
			W[15]+=W[8];
			W[15]+=s1(W[13]);
			Vals[5]+=W[15];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[31];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]=fw01r;
			W[0]+=W[9];
			W[0]+=s1(W[14]);
			Vals[2]+=W[0];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[32];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]=fw1;
			W[1]+=s0(W[2]);
			W[1]+=W[10];
			W[1]+=s1(W[15]);
			Vals[3]+=W[1];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[33];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=s0(W[3]);
			W[2]+=W[11];
			W[2]+=s1(W[0]);
			Vals[4]+=W[2];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[34];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=s0(W[4]);
			W[3]+=W[12];
			W[3]+=s1(W[1]);
			Vals[1]+=W[3];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[35];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=s0(W[5]);
			W[4]+=W[13];
			W[4]+=s1(W[2]);
			Vals[0]+=W[4];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[36];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=s0(W[6]);
			W[5]+=W[14];
			W[5]+=s1(W[3]);
			Vals[6]+=W[5];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[37];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=s0(W[7]);
			W[6]+=W[15];
			W[6]+=s1(W[4]);
			Vals[7]+=W[6];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[38];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=s0(W[8]);
			W[7]+=W[0];
			W[7]+=s1(W[5]);
			Vals[5]+=W[7];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[39];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=s0(W[9]);
			W[8]+=W[1];
			W[8]+=s1(W[6]);
			Vals[2]+=W[8];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[40];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=s0(W[10]);
			W[9]+=W[2];
			W[9]+=s1(W[7]);
			Vals[3]+=W[9];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[41];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=s0(W[11]);
			W[10]+=W[3];
			W[10]+=s1(W[8]);
			Vals[4]+=W[10];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[42];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=s0(W[12]);
			W[11]+=W[4];
			W[11]+=s1(W[9]);
			Vals[1]+=W[11];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[43];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=s0(W[13]);
			W[12]+=W[5];
			W[12]+=s1(W[10]);
			Vals[0]+=W[12];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[44];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=s0(W[14]);
			W[13]+=W[6];
			W[13]+=s1(W[11]);
			Vals[6]+=W[13];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[45];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=s0(W[15]);
			W[14]+=W[7];
			W[14]+=s1(W[12]);
			Vals[7]+=W[14];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[46];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=s0(W[0]);
			W[15]+=W[8];
			W[15]+=s1(W[13]);
			Vals[5]+=W[15];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[47];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=s0(W[1]);
			W[0]+=W[9];
			W[0]+=s1(W[14]);
			Vals[2]+=W[0];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[48];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=s0(W[2]);
			W[1]+=W[10];
			W[1]+=s1(W[15]);
			Vals[3]+=W[1];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[49];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=s0(W[3]);
			W[2]+=W[11];
			W[2]+=s1(W[0]);
			Vals[4]+=W[2];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[50];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=s0(W[4]);
			W[3]+=W[12];
			W[3]+=s1(W[1]);
			Vals[1]+=W[3];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[51];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=s0(W[5]);
			W[4]+=W[13];
			W[4]+=s1(W[2]);
			Vals[0]+=W[4];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[52];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=s0(W[6]);
			W[5]+=W[14];
			W[5]+=s1(W[3]);
			Vals[6]+=W[5];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[53];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=s0(W[7]);
			W[6]+=W[15];
			W[6]+=s1(W[4]);
			Vals[7]+=W[6];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[54];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=s0(W[8]);
			W[7]+=W[0];
			W[7]+=s1(W[5]);
			Vals[5]+=W[7];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[55];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=s0(W[9]);
			W[8]+=W[1];
			W[8]+=s1(W[6]);
			Vals[2]+=W[8];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[56];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=s0(W[10]);
			W[9]+=W[2];
			W[9]+=s1(W[7]);
			Vals[3]+=W[9];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[57];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=s0(W[11]);
			W[10]+=W[3];
			W[10]+=s1(W[8]);
			Vals[4]+=W[10];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[58];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=s0(W[12]);
			W[11]+=W[4];
			W[11]+=s1(W[9]);
			Vals[1]+=W[11];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[59];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=s0(W[13]);
			W[12]+=W[5];
			W[12]+=s1(W[10]);
			Vals[0]+=W[12];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[60];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=s0(W[14]);
			W[13]+=W[6];
			W[13]+=s1(W[11]);
			Vals[6]+=W[13];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[61];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			Vals[7]+=W[14];
			Vals[7]+=s0(W[15]);
			Vals[7]+=W[7];
			Vals[7]+=s1(W[12]);
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[62];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=W[15];
			Vals[5]+=s0(W[0]);
			Vals[5]+=W[8];
			Vals[5]+=s1(W[13]);
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[63];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			Vals[5]+=state0;

			W[7]=state7;
			W[7]+=Vals[2];

			Vals[2]=0x8b60e42dU;
			Vals[2]+=Vals[5];
			W[0]=Vals[5];
			Vals[5]=0xdfa9bf2cU;

			W[3]=state3;
			W[3]+=Vals[0];

			Vals[0]=0xd338e869U;
			Vals[0]+=Vals[2];
			Vals[2]+=0x6810571cU;

			W[6]=state6;
			W[6]+=Vals[3];

			Vals[3]=0x010c72ecU;
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=(0x475bbf30U^(Vals[0]&0xed644e16));

			Vals[7]+=state1;
			Vals[3]+=Vals[7];
			W[1]=Vals[7];
			Vals[7]=0xb72074d4U;

			W[2]=state2;
			W[2]+=Vals[6];

			Vals[6]=0x6bb01122U;
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[5]=state5;
			W[5]+=Vals[4];

			Vals[4]=0xfd1cbaffU;
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],0xaa3ff126U);
			Vals[4]+=W[2];

			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[4]=state4;
			W[4]+=Vals[1];

			Vals[1]=0x93f5cccbU;
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=W[3];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[4];
			Vals[0]+=W[4];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[5];
			Vals[6]+=W[5];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[6];
			Vals[7]+=W[6];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[7];
			Vals[5]+=W[7];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=0x5807aa98U;
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[9];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[10];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[11];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[12];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[13];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[14];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=0xc19bf474U;
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=s0(W[1]);
			Vals[2]+=W[0];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[16];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=s0(W[2]);
			W[1]+=0x01e00000U;
			Vals[3]+=W[1];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[17];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=s0(W[3]);
			W[2]+=s1(W[0]);
			Vals[4]+=W[2];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[18];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=s0(W[4]);
			W[3]+=s1(W[1]);
			Vals[1]+=W[3];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[19];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=s0(W[5]);
			W[4]+=s1(W[2]);
			Vals[0]+=W[4];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[20];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=s0(W[6]);
			W[5]+=s1(W[3]);
			Vals[6]+=W[5];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[21];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=s0(W[7]);
			W[6]+=0x00000300U;
			W[6]+=s1(W[4]);
			Vals[7]+=W[6];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[22];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=0x11002000U;
			W[7]+=W[0];
			W[7]+=s1(W[5]);
			Vals[5]+=W[7];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[23];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]=0x80000000U;
			W[8]+=W[1];
			W[8]+=s1(W[6]);
			Vals[2]+=W[8];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[24];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]=W[2];
			W[9]+=s1(W[7]);
			Vals[3]+=W[9];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[25];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]=W[3];
			W[10]+=s1(W[8]);
			Vals[4]+=W[10];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[26];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]=W[4];
			W[11]+=s1(W[9]);
			Vals[1]+=W[11];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[27];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]=W[5];
			W[12]+=s1(W[10]);
			Vals[0]+=W[12];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[28];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]=W[6];
			W[13]+=s1(W[11]);
			Vals[6]+=W[13];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[29];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]=0x00c00066U;
			W[14]+=W[7];
			W[14]+=s1(W[12]);
			Vals[7]+=W[14];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[30];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]=0x00000300U;
			W[15]+=s0(W[0]);
			W[15]+=W[8];
			W[15]+=s1(W[13]);
			Vals[5]+=W[15];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[31];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=s0(W[1]);
			W[0]+=W[9];
			W[0]+=s1(W[14]);
			Vals[2]+=W[0];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[32];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=s0(W[2]);
			W[1]+=W[10];
			W[1]+=s1(W[15]);
			Vals[3]+=W[1];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[33];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=s0(W[3]);
			W[2]+=W[11];
			W[2]+=s1(W[0]);
			Vals[4]+=W[2];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[34];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=s0(W[4]);
			W[3]+=W[12];
			W[3]+=s1(W[1]);
			Vals[1]+=W[3];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[35];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=s0(W[5]);
			W[4]+=W[13];
			W[4]+=s1(W[2]);
			Vals[0]+=W[4];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[36];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=s0(W[6]);
			W[5]+=W[14];
			W[5]+=s1(W[3]);
			Vals[6]+=W[5];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[37];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=s0(W[7]);
			W[6]+=W[15];
			W[6]+=s1(W[4]);
			Vals[7]+=W[6];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[38];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=s0(W[8]);
			W[7]+=W[0];
			W[7]+=s1(W[5]);
			Vals[5]+=W[7];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[39];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=s0(W[9]);
			W[8]+=W[1];
			W[8]+=s1(W[6]);
			Vals[2]+=W[8];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[40];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[9]+=s0(W[10]);
			W[9]+=W[2];
			W[9]+=s1(W[7]);
			Vals[3]+=W[9];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[41];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[10]+=s0(W[11]);
			W[10]+=W[3];
			W[10]+=s1(W[8]);
			Vals[4]+=W[10];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[42];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[11]+=s0(W[12]);
			W[11]+=W[4];
			W[11]+=s1(W[9]);
			Vals[1]+=W[11];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[43];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[12]+=s0(W[13]);
			W[12]+=W[5];
			W[12]+=s1(W[10]);
			Vals[0]+=W[12];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[44];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[13]+=s0(W[14]);
			W[13]+=W[6];
			W[13]+=s1(W[11]);
			Vals[6]+=W[13];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[45];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[14]+=s0(W[15]);
			W[14]+=W[7];
			W[14]+=s1(W[12]);
			Vals[7]+=W[14];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[46];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[15]+=s0(W[0]);
			W[15]+=W[8];
			W[15]+=s1(W[13]);
			Vals[5]+=W[15];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[47];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[0]+=s0(W[1]);
			W[0]+=W[9];
			W[0]+=s1(W[14]);
			Vals[2]+=W[0];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[48];
			Vals[0]+=Vals[2];
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);

			W[1]+=s0(W[2]);
			W[1]+=W[10];
			W[1]+=s1(W[15]);
			Vals[3]+=W[1];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[49];
			Vals[6]+=Vals[3];
			Vals[3]+=S0(Vals[2]);
			Vals[3]+=Ma(Vals[7],Vals[2],Vals[5]);

			W[2]+=s0(W[3]);
			W[2]+=W[11];
			W[2]+=s1(W[0]);
			Vals[4]+=W[2];
			Vals[4]+=S1(Vals[6]);
			Vals[4]+=ch(Vals[6],Vals[0],Vals[1]);
			Vals[4]+=K[50];
			Vals[7]+=Vals[4];
			Vals[4]+=S0(Vals[3]);
			Vals[4]+=Ma(Vals[5],Vals[3],Vals[2]);

			W[3]+=s0(W[4]);
			W[3]+=W[12];
			W[3]+=s1(W[1]);
			Vals[1]+=W[3];
			Vals[1]+=S1(Vals[7]);
			Vals[1]+=ch(Vals[7],Vals[6],Vals[0]);
			Vals[1]+=K[51];
			Vals[5]+=Vals[1];
			Vals[1]+=S0(Vals[4]);
			Vals[1]+=Ma(Vals[2],Vals[4],Vals[3]);

			W[4]+=s0(W[5]);
			W[4]+=W[13];
			W[4]+=s1(W[2]);
			Vals[0]+=W[4];
			Vals[0]+=S1(Vals[5]);
			Vals[0]+=ch(Vals[5],Vals[7],Vals[6]);
			Vals[0]+=K[52];
			Vals[2]+=Vals[0];
			Vals[0]+=S0(Vals[1]);
			Vals[0]+=Ma(Vals[3],Vals[1],Vals[4]);

			W[5]+=s0(W[6]);
			W[5]+=W[14];
			W[5]+=s1(W[3]);
			Vals[6]+=W[5];
			Vals[6]+=S1(Vals[2]);
			Vals[6]+=ch(Vals[2],Vals[5],Vals[7]);
			Vals[6]+=K[53];
			Vals[3]+=Vals[6];
			Vals[6]+=S0(Vals[0]);
			Vals[6]+=Ma(Vals[4],Vals[0],Vals[1]);

			W[6]+=s0(W[7]);
			W[6]+=W[15];
			W[6]+=s1(W[4]);
			Vals[7]+=W[6];
			Vals[7]+=S1(Vals[3]);
			Vals[7]+=ch(Vals[3],Vals[2],Vals[5]);
			Vals[7]+=K[54];
			Vals[4]+=Vals[7];
			Vals[7]+=S0(Vals[6]);
			Vals[7]+=Ma(Vals[1],Vals[6],Vals[0]);

			W[7]+=s0(W[8]);
			W[7]+=W[0];
			W[7]+=s1(W[5]);
			Vals[5]+=W[7];
			Vals[5]+=S1(Vals[4]);
			Vals[5]+=ch(Vals[4],Vals[3],Vals[2]);
			Vals[5]+=K[55];
			Vals[1]+=Vals[5];
			Vals[5]+=S0(Vals[7]);
			Vals[5]+=Ma(Vals[0],Vals[7],Vals[6]);

			W[8]+=s0(W[9]);
			W[8]+=W[1];
			W[8]+=s1(W[6]);
			Vals[2]+=W[8];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);
			Vals[2]+=K[56];
			Vals[0]+=Vals[2];

			W[9]+=s0(W[10]);
			W[9]+=W[2];
			W[9]+=s1(W[7]);
			Vals[3]+=W[9];
			Vals[3]+=S1(Vals[0]);
			Vals[3]+=ch(Vals[0],Vals[1],Vals[4]);
			Vals[3]+=K[57];
			Vals[3]+=Vals[6];

			W[10]+=s0(W[11]);
			W[10]+=W[3];
			W[10]+=s1(W[8]);
			Vals[4]+=W[10];
			Vals[4]+=S1(Vals[3]);
			Vals[4]+=ch(Vals[3],Vals[0],Vals[1]);
			Vals[4]+=K[58];
			Vals[4]+=Vals[7];
			Vals[1]+=S1(Vals[4]);
			Vals[1]+=ch(Vals[4],Vals[3],Vals[0]);
			Vals[1]+=W[11];
			Vals[1]+=s0(W[12]);
			Vals[1]+=W[4];
			Vals[1]+=s1(W[9]);
			Vals[1]+=K[59];
			Vals[1]+=Vals[5];

			Vals[2]+=Ma(Vals[6],Vals[5],Vals[7]);
			Vals[2]+=S0(Vals[5]);
			Vals[2]+=W[12];
			Vals[2]+=s0(W[13]);
			Vals[2]+=W[5];
			Vals[2]+=s1(W[10]);
			Vals[2]+=Vals[0];
			Vals[2]+=S1(Vals[1]);
			Vals[2]+=ch(Vals[1],Vals[4],Vals[3]);

			if (Vals[2] == -0x3034c9a7U)
			{
				uint32_t tmp = atomicCAS(result, 0xffffffff, nonce);
				if (tmp != 0xffffffff)
					result[1] = nonce;
			}
		} // nonce loop
	} // if thread<threads
}

__host__
void novo_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, const uint32_t *const ms, uint32_t merkle, uint32_t time, uint32_t compacttarget, uint32_t *const h_nounce)
{
	cudaMemsetAsync(d_result[thr_id], 0xff, 2 * sizeof(uint32_t), gpustream[thr_id]);

	uint32_t A = ms[0];
	uint32_t B = ms[1];
	uint32_t C = ms[2];
	uint32_t D = ms[3];
	uint32_t E = ms[4];
	uint32_t F = ms[5];
	uint32_t G = ms[6];
	uint32_t H = ms[7];

	R(A, B, C, D, E, F, G, H, merkle, 0x428a2f98U);
	R(H, A, B, C, D, E, F, G, time, 0x71374491U);
	R(G, H, A, B, C, D, E, F, compacttarget, 0xb5c0fbcfU);

	uint32_t D1A = D + 0xb956c25b;
	uint32_t fW0 = merkle + (rotr(time, 7) ^ rotr(time, 18) ^ (time >> 3));
	uint32_t fW1 = time + (rotr(compacttarget, 7) ^ rotr(compacttarget, 18) ^ (compacttarget >> 3)) + 0x2d00001;
	uint32_t PreVal4 = ms[4] + (rotr(B, 6) ^ rotr(B, 11) ^ rotr(B, 25)) + (D ^ (B & (C ^ D))) + 0xe9b5dba5;
	uint32_t T1 = (rotr(F, 2) ^ rotr(F, 13) ^ rotr(F, 22)) + ((F & G) | (H & (F | G)));
	uint32_t PreVal0 = PreVal4 + ms[0];
	uint32_t fW2 = compacttarget + (rotr(fW0, 17) ^ rotr(fW0, 19) ^ (fW0 >> 10));
	uint32_t fW3 = 0x11002000 + (rotr(fW1, 17) ^ rotr(fW1, 19) ^ (fW1 >> 10));
	uint32_t fW15 = 0x00000480 + (rotr(fW0, 7) ^ rotr(fW0, 18) ^ (fW0 >> 3));
	uint32_t fW01r = fW0 + (rotr(fW1, 7) ^ rotr(fW1, 18) ^ (fW1 >> 3));
	uint32_t PreVal4addT1 = PreVal4 + T1;
	uint32_t C1addK5 = C + 0x59f111f1U;
	uint32_t B1addK6 = B + 0x923f82a4U;
	uint32_t W16addK16 = fW0 + 0xe49b69c1U;
	uint32_t W17addK17 = fW1 + 0xefbe4786U;

	dim3 grid((threads + TPB*NONCES_PER_THREAD - 1) / TPB / NONCES_PER_THREAD);
	dim3 block(TPB);
	novo_gpu_hash << <grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_result[thr_id], ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6], ms[7], B, C, F, G, H, fW0, fW1, fW2, fW3, fW15, fW01r, D1A, C1addK5, B1addK6, W16addK16, W17addK17, PreVal4addT1, PreVal0);
	CUDA_SAFE_CALL(cudaMemcpyAsync(h_nounce, d_result[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id])); cudaStreamSynchronize(gpustream[thr_id]);
}

__host__
void novo_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_result[thr_id], 4 * sizeof(uint32_t)));
}
