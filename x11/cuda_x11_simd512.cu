/***************************************************************************************************
 * SIMD512 SM3+ CUDA IMPLEMENTATION (require cuda_x11_simd512_func.cuh)
 */

#include "miner.h"
#include "cuda_helper.h"

#define TPB 128

uint32_t *d_state[MAX_GPUS];
uint4 *d_temp4[MAX_GPUS];

// texture bound to d_temp4[thr_id], for read access in Compaction kernel
texture<uint4, 1, cudaReadModeElementType> texRef1D_128;

__constant__ uint8_t c_perm[8][8] =
{
	{ 2, 3, 6, 7, 0, 1, 4, 5 },
	{ 6, 7, 2, 3, 4, 5, 0, 1 },
	{ 7, 6, 5, 4, 3, 2, 1, 0 },
	{ 1, 0, 3, 2, 5, 4, 7, 6 },
	{ 0, 1, 4, 5, 6, 7, 2, 3 },
	{ 6, 7, 2, 3, 0, 1, 4, 5 },
	{ 6, 7, 0, 1, 4, 5, 2, 3 },
	{ 4, 5, 2, 3, 6, 7, 0, 1 }
};

/* used in cuda_x11_simd512_func.cuh (SIMD_Compress2) */
__constant__ uint32_t c_IV_512[32] =
{
	0x0ba16b95, 0x72f999ad, 0x9fecc2ae, 0xba3264fc, 0x5e894929, 0x8e9f30e5, 0x2f1daa37, 0xf0f2c558,
	0xac506643, 0xa90635a5, 0xe25b878b, 0xaab7878f, 0x88817f7a, 0x0a02892b, 0x559a7550, 0x598f657e,
	0x7eef60a1, 0x6b70e3e8, 0x9c1714d1, 0xb958e2a8, 0xab02675e, 0xed1c014f, 0xcd8d65bb, 0xfdb7a257,
	0x09254899, 0xd699c7bc, 0x9019b6dc, 0x2b9022e4, 0x8fa14956, 0x21bf9bd3, 0xb94d0943, 0x6ffddc22
};

__constant__ int c_FFT128_8_16_Twiddle[128] =
{
	1,   1,   1,   1,   1,    1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
	1,  60,   2, 120,   4,  -17,   8, -34,  16, -68,  32, 121,  64, -15, 128, -30,
	1,  46,  60, -67,   2,   92, 120, 123,   4, -73, -17, -11,   8, 111, -34, -22,
	1, -67, 120, -73,   8,  -22, -68, -70,  64,  81, -30, -46,  -2,-123,  17,-111,
	1,-118,  46, -31,  60,  116, -67, -61,   2,  21,  92, -62, 120, -25, 123,-122,
	1, 116,  92,-122, -17,   84, -22,  18,  32, 114, 117, -49, -30, 118,  67,  62,
	1, -31, -67,  21, 120, -122, -73, -50,   8,   9, -22, -89, -68,  52, -70, 114,
	1, -61, 123, -50, -34,   18, -70, -99, 128, -98,  67,  25,  17,  -9,  35, -79
};

__constant__ int c_FFT256_2_128_Twiddle[128] =
{
	  1,  41,-118,  45,  46,  87, -31,  14,
	 60,-110, 116,-127, -67,  80, -61,  69,
	  2,  82,  21,  90,  92, -83, -62,  28,
	120,  37, -25,   3, 123, -97,-122,-119,
	  4, -93,  42, -77, -73,  91,-124,  56,
	-17,  74, -50,   6, -11,  63,  13,  19,
	  8,  71,  84, 103, 111, -75,   9, 112,
	-34,-109,-100,  12, -22, 126,  26,  38,
	 16,-115, -89, -51, -35, 107,  18, -33,
	-68,  39,  57,  24, -44,  -5,  52,  76,
	 32,  27,  79,-102, -70, -43,  36, -66,
	121,  78, 114,  48, -88, -10, 104,-105,
	 64,  54, -99,  53, 117, -86,  72, 125,
	-15,-101, -29,  96,  81, -20, -49,  47,
	128, 108,  59, 106, -23,  85,-113,  -7,
	-30,  55, -58, -65, -95, -40, -98,  94
};

/************* the round function ****************/
#define IF(x, y, z) (((y ^ z) & x) ^ z)
#define MAJ(x, y, z) ((z & y) | ((z | y) & x))

#include "cuda_x11_simd512_func.cuh"

/********************* Message expansion ************************/

/*
 * Reduce modulo 257; result is in [-127; 383]
 * REDUCE(x) := (x&255) - (x>>8)
 */
#define REDUCE(x) (((x)&255) - ((x)>>8))

/*
 * Reduce from [-127; 383] to [-128; 128]
 * EXTRA_REDUCE_S(x) := x<=128 ? x : x-257
 */
#define EXTRA_REDUCE_S(x) ((x)<=128 ? (x) : (x)-257)

/*
 * Reduce modulo 257; result is in [-128; 128]
 */
#define REDUCE_FULL_S(x) EXTRA_REDUCE_S(REDUCE(x))

// Parallelization:
//
// FFT_8  wird 2 times 8-fach parallel ausgeführt (in FFT_64)
//        and  1 time 16-fach parallel (in FFT_128_full)
//
// STEP8_IF and STEP8_MAJ beinhalten je 2x 8-fach parallel Operations

/**
 * FFT_8 using w=4 as 8th root of unity
 * Unrolled decimation in frequency (DIF) radix-2 NTT.
 * Output data is in revbin_permuted order.
 */
static __device__ __forceinline__
void FFT_8(int *y, int stripe)
{
#define X(i) y[stripe*i]

#define DO_REDUCE(i) \
	X(i) = REDUCE(X(i))

#define DO_REDUCE_FULL_S(i) \
do { \
	X(i) = REDUCE(X(i)); \
	X(i) = EXTRA_REDUCE_S(X(i)); \
} while(0)

#define BUTTERFLY(i,j,n) \
do { \
	int u = X(i); \
	X(i) += X(j); \
	X(j) = (u - X(j)) << (2 * n); \
} while(0)

	BUTTERFLY(0, 4, 0);
	BUTTERFLY(1, 5, 1);
	BUTTERFLY(2, 6, 2);
	BUTTERFLY(3, 7, 3);

	DO_REDUCE(6);
	DO_REDUCE(7);

	BUTTERFLY(0, 2, 0);
	BUTTERFLY(4, 6, 0);
	BUTTERFLY(1, 3, 2);
	BUTTERFLY(5, 7, 2);

	DO_REDUCE(7);

	BUTTERFLY(0, 1, 0);
	BUTTERFLY(2, 3, 0);
	BUTTERFLY(4, 5, 0);
	BUTTERFLY(6, 7, 0);

	DO_REDUCE_FULL_S(0);
	DO_REDUCE_FULL_S(1);
	DO_REDUCE_FULL_S(2);
	DO_REDUCE_FULL_S(3);
	DO_REDUCE_FULL_S(4);
	DO_REDUCE_FULL_S(5);
	DO_REDUCE_FULL_S(6);
	DO_REDUCE_FULL_S(7);

#undef X
#undef DO_REDUCE
#undef DO_REDUCE_FULL_S
#undef BUTTERFLY
}

/**
 * FFT_16 using w=2 as 16th root of unity
 * Unrolled decimation in frequency (DIF) radix-2 NTT.
 * Output data is in revbin_permuted order.
 */
static __device__ __forceinline__
void FFT_16(int *y)
{
#define DO_REDUCE_FULL_S(i) \
	do { \
		y[i] = REDUCE(y[i]); \
		y[i] = EXTRA_REDUCE_S(y[i]); \
	} while(0)

	int u,v;

	u = y[0];
	y[0] += y[1];
	y[1] = (u - y[1]) << (threadIdx.x&7);

	if ((threadIdx.x&7) >=3)
		y[1] = REDUCE(y[1]);  // 11...15

	u = SHFL((int)y[0],  (threadIdx.x&3),8); // 0,1,2,3  0,1,2,3
	v = SHFL((int)y[0],4+(threadIdx.x&3),8); // 4,5,6,7  4,5,6,7
	y[0] = ((threadIdx.x&7) < 4) ? (u+v) : ((u-v) << (2*(threadIdx.x&3)));

	u = SHFL((int)y[1],  (threadIdx.x&3),8); // 8,9,10,11    8,9,10,11
	v = SHFL((int)y[1],4+(threadIdx.x&3),8); // 12,13,14,15  12,13,14,15
	y[1] = ((threadIdx.x&7) < 4) ? (u+v) : ((u-v) << (2*(threadIdx.x&3)));


	if ((threadIdx.x&1) && (threadIdx.x&7) >= 4)
	{
		y[0] = REDUCE(y[0]);  // 5, 7
		y[1] = REDUCE(y[1]);  // 13, 15
	}

	u = SHFL((int)y[0],  (threadIdx.x&5),8); // 0,1,0,1  4,5,4,5
	v = SHFL((int)y[0],2+(threadIdx.x&5),8); // 2,3,2,3  6,7,6,7
	y[0] = ((threadIdx.x&3) < 2) ? (u+v) : ((u-v) << (4*(threadIdx.x&1)));

	u = SHFL((int)y[1],  (threadIdx.x&5),8); // 8,9,8,9      12,13,12,13
	v = SHFL((int)y[1],2+(threadIdx.x&5),8); // 10,11,10,11  14,15,14,15
	y[1] = ((threadIdx.x&3) < 2) ? (u+v) : ((u-v) << (4*(threadIdx.x&1)));

	u = SHFL((int)y[0],  (threadIdx.x&6),8); // 0,0,2,2      4,4,6,6
	v = SHFL((int)y[0],1+(threadIdx.x&6),8); // 1,1,3,3      5,5,7,7
	y[0] = ((threadIdx.x&1) < 1) ? (u+v) : (u-v);

	u = SHFL((int)y[1],  (threadIdx.x&6),8); // 8,8,10,10    12,12,14,14
	v = SHFL((int)y[1],1+(threadIdx.x&6),8); // 9,9,11,11    13,13,15,15
	y[1] = ((threadIdx.x&1) < 1) ? (u+v) : (u-v);

	DO_REDUCE_FULL_S( 0); // 0...7
	DO_REDUCE_FULL_S( 1); // 8...15

#undef DO_REDUCE_FULL_S
}

static __device__ __forceinline__
void FFT_128_full(int y[128])
{
	int i;

	FFT_8(y+0,2); // eight parallel FFT8's
	FFT_8(y+1,2); // eight parallel FFT8's

#pragma unroll 16
	for (i=0; i<16; i++)
	/*if (i & 7)*/ y[i] = REDUCE(y[i]*c_FFT128_8_16_Twiddle[i*8+(threadIdx.x&7)]);

#pragma unroll 8
	for (i=0; i<16; i+=2)
		FFT_16(y + i);  // eight sequential FFT16's, each one executed in parallel by 8 threads
}

static __device__ __forceinline__
void FFT_256_halfzero(int y[256])
{
	/*
	 * FFT_256 using w=41 as 256th root of unity.
	 * Decimation in frequency (DIF) NTT.
	 * Output data is in revbin_permuted order.
	 * In place.
	 */
	const int tmp = y[15];

#pragma unroll 8
	for (int i=0; i<8; i++)
		y[16+i] = REDUCE(y[i] * c_FFT256_2_128_Twiddle[8*i+(threadIdx.x&7)]);
#pragma unroll 8
	for (int i=24; i<32; i++)
		y[i] = 0;

	/* handle X^255 with an additional butterfly */
	if ((threadIdx.x&7) == 7)
	{
		y[15] = REDUCE(tmp + 1);
		y[31] = REDUCE((tmp - 1) * c_FFT256_2_128_Twiddle[127]);
	}

	FFT_128_full(y);
	FFT_128_full(y+16);
}

/***************************************************/

static __device__ __forceinline__
void Expansion(const uint32_t *data, uint4 *g_temp4)
{
	/* Message Expansion using Number Theoretical Transform similar to FFT */
	int expanded[32];
#pragma unroll 4
	for (int i=0; i < 4; i++) {
		expanded[  i] = __byte_perm(SHFL((int)data[0], 2*i, 8), SHFL((int)data[0], (2*i)+1, 8), threadIdx.x&7)&0xff;
		expanded[4+i] = __byte_perm(SHFL((int)data[1], 2*i, 8), SHFL((int)data[1], (2*i)+1, 8), threadIdx.x&7)&0xff;
	}
#pragma unroll 8
	for (int i=8; i < 16; i++)
		expanded[i] = 0;

	FFT_256_halfzero(expanded);

	// store w matrices in global memory

#define mul_185(x) ( (x)*185 )
#define mul_233(x) ( (x)*233 )

	uint4 vec0;
	int P, Q, P1, Q1, P2, Q2;
	bool even = (threadIdx.x & 1) == 0;

	P1 = expanded[ 0]; P2 = SHFL(expanded[ 2], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[16]; Q2 = SHFL(expanded[18], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[0][threadIdx.x&7], 8);
	P1 = expanded[ 8]; P2 = SHFL(expanded[10], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[24]; Q2 = SHFL(expanded[26], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[0][threadIdx.x&7], 8);
	P1 = expanded[ 4]; P2 = SHFL(expanded[ 6], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[20]; Q2 = SHFL(expanded[22], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[0][threadIdx.x&7], 8);
	P1 = expanded[12]; P2 = SHFL(expanded[14], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[28]; Q2 = SHFL(expanded[30], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[0][threadIdx.x&7], 8);
	g_temp4[threadIdx.x&7] = vec0;

	P1 = expanded[ 1]; P2 = SHFL(expanded[ 3], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[17]; Q2 = SHFL(expanded[19], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[1][threadIdx.x&7], 8);
	P1 = expanded[ 9]; P2 = SHFL(expanded[11], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[25]; Q2 = SHFL(expanded[27], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[1][threadIdx.x&7], 8);
	P1 = expanded[ 5]; P2 = SHFL(expanded[ 7], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[21]; Q2 = SHFL(expanded[23], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[1][threadIdx.x&7], 8);
	P1 = expanded[13]; P2 = SHFL(expanded[15], (threadIdx.x-1)&7, 8); P = even ? P1 : P2;
	Q1 = expanded[29]; Q2 = SHFL(expanded[31], (threadIdx.x-1)&7, 8); Q = even ? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[1][threadIdx.x&7], 8);
	g_temp4[8+(threadIdx.x&7)] = vec0;

	bool hi = (threadIdx.x&7)>=4;

	P1 = hi?expanded[ 1]:expanded[ 0]; P2 = SHFL(hi?expanded[ 3]:expanded[ 2], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = hi?expanded[17]:expanded[16]; Q2 = SHFL(hi?expanded[19]:expanded[18], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[2][threadIdx.x&7], 8);
	P1 = hi?expanded[ 9]:expanded[ 8]; P2 = SHFL(hi?expanded[11]:expanded[10], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = hi?expanded[25]:expanded[24]; Q2 = SHFL(hi?expanded[27]:expanded[26], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[2][threadIdx.x&7], 8);
	P1 = hi?expanded[ 5]:expanded[ 4]; P2 = SHFL(hi?expanded[ 7]:expanded[ 6], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = hi?expanded[21]:expanded[20]; Q2 = SHFL(hi?expanded[23]:expanded[22], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[2][threadIdx.x&7], 8);
	P1 = hi?expanded[13]:expanded[12]; P2 = SHFL(hi?expanded[15]:expanded[14], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = hi?expanded[29]:expanded[28]; Q2 = SHFL(hi?expanded[31]:expanded[30], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[2][threadIdx.x&7], 8);
	g_temp4[16+(threadIdx.x&7)] = vec0;

	bool lo = (threadIdx.x&7)<4;

	P1 = lo?expanded[ 1]:expanded[ 0]; P2 = SHFL(lo?expanded[ 3]:expanded[ 2], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = lo?expanded[17]:expanded[16]; Q2 = SHFL(lo?expanded[19]:expanded[18], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[3][threadIdx.x&7], 8);
	P1 = lo?expanded[ 9]:expanded[ 8]; P2 = SHFL(lo?expanded[11]:expanded[10], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = lo?expanded[25]:expanded[24]; Q2 = SHFL(lo?expanded[27]:expanded[26], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[3][threadIdx.x&7], 8);
	P1 = lo?expanded[ 5]:expanded[ 4]; P2 = SHFL(lo?expanded[ 7]:expanded[ 6], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = lo?expanded[21]:expanded[20]; Q2 = SHFL(lo?expanded[23]:expanded[22], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[3][threadIdx.x&7], 8);
	P1 = lo?expanded[13]:expanded[12]; P2 = SHFL(lo?expanded[15]:expanded[14], (threadIdx.x+1)&7, 8); P = !even ? P1 : P2;
	Q1 = lo?expanded[29]:expanded[28]; Q2 = SHFL(lo?expanded[31]:expanded[30], (threadIdx.x+1)&7, 8); Q = !even ? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_185(P),  mul_185(Q) , 0x5410), c_perm[3][threadIdx.x&7], 8);
	g_temp4[24+(threadIdx.x&7)] = vec0;

	bool sel = ((threadIdx.x+2)&7) >= 4;  // 2,3,4,5

	P1 = sel?expanded[0]:expanded[1]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[2]:expanded[3]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[4][threadIdx.x&7], 8);
	P1 = sel?expanded[8]:expanded[9]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[10]:expanded[11]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[4][threadIdx.x&7], 8);
	P1 = sel?expanded[4]:expanded[5]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[6]:expanded[7]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[4][threadIdx.x&7], 8);
	P1 = sel?expanded[12]:expanded[13]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[14]:expanded[15]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[4][threadIdx.x&7], 8);

	g_temp4[32+(threadIdx.x&7)] = vec0;

	P1 = sel?expanded[1]:expanded[0]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[3]:expanded[2]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[5][threadIdx.x&7], 8);
	P1 = sel?expanded[9]:expanded[8]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[11]:expanded[10]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[5][threadIdx.x&7], 8);
	P1 = sel?expanded[5]:expanded[4]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[7]:expanded[6]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[5][threadIdx.x&7], 8);
	P1 = sel?expanded[13]:expanded[12]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	Q2 = sel?expanded[15]:expanded[14]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[5][threadIdx.x&7], 8);

	g_temp4[40+(threadIdx.x&7)] = vec0;

	int t;
	t = SHFL(expanded[17],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[16]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[19],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[18]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[6][threadIdx.x&7], 8);
	t = SHFL(expanded[25],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[24]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[27],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[26]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[6][threadIdx.x&7], 8);
	t = SHFL(expanded[21],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[20]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[23],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[22]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[6][threadIdx.x&7], 8);
	t = SHFL(expanded[29],(threadIdx.x+4)&7,8); P1 = sel?t:expanded[28]; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[31],(threadIdx.x+4)&7,8); Q2 = sel?t:expanded[30]; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[6][threadIdx.x&7], 8);

	g_temp4[48+(threadIdx.x&7)] = vec0;

	t = SHFL(expanded[16],(threadIdx.x+4)&7,8); P1 = sel?expanded[17]:t; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[18],(threadIdx.x+4)&7,8); Q2 = sel?expanded[19]:t; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.x = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[7][threadIdx.x&7], 8);
	t = SHFL(expanded[24],(threadIdx.x+4)&7,8); P1 = sel?expanded[25]:t; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[26],(threadIdx.x+4)&7,8); Q2 = sel?expanded[27]:t; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.y = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[7][threadIdx.x&7], 8);
	t = SHFL(expanded[20],(threadIdx.x+4)&7,8); P1 = sel?expanded[21]:t; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[22],(threadIdx.x+4)&7,8); Q2 = sel?expanded[23]:t; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.z = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[7][threadIdx.x&7], 8);
	t = SHFL(expanded[28],(threadIdx.x+4)&7,8); P1 = sel?expanded[29]:t; Q1 = SHFL(P1, threadIdx.x^1, 8);
	t = SHFL(expanded[30],(threadIdx.x+4)&7,8); Q2 = sel?expanded[31]:t; P2 = SHFL(Q2, threadIdx.x^1, 8);
	P = even? P1 : P2; Q = even? Q1 : Q2;
	vec0.w = SHFL((int)__byte_perm(mul_233(P),  mul_233(Q) , 0x5410), c_perm[7][threadIdx.x&7], 8);

	g_temp4[56+(threadIdx.x&7)] = vec0;

#undef mul_185
#undef mul_233
}

/***************************************************/

__global__ __launch_bounds__(TPB, 4)
void x11_simd512_gpu_expand_64(uint32_t threads, uint32_t *g_hash, uint4 *g_temp4)
{
	int threadBloc = (blockDim.x * blockIdx.x + threadIdx.x) / 8;
	if (threadBloc < threads)
	{
		int hashPosition = threadBloc * 16;
		uint32_t *inpHash = &g_hash[hashPosition];

		// Read hash per 8 threads
		uint32_t Hash[2];
		int ndx = threadIdx.x & 7;
		Hash[0] = inpHash[ndx];
		Hash[1] = inpHash[ndx + 8];

		// Puffer für expandierte Nachricht
		uint4 *temp4 = &g_temp4[hashPosition * 4];

		Expansion(Hash, temp4);
	}
}

__global__ __launch_bounds__(TPB, 1)
void x11_simd512_gpu_compress1_64(uint32_t threads, uint32_t *g_hash, uint4 *g_fft4, uint32_t *g_state)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *Hash = &g_hash[thread * 16];
		Compression1(Hash, thread, g_fft4, g_state);
	}
}

__global__ __launch_bounds__(TPB, 1)
void x11_simd512_gpu_compress2_64(uint32_t threads, uint4 *g_fft4, uint32_t *g_state)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		Compression2(thread, g_fft4, g_state);
	}
}

__global__ __launch_bounds__(TPB, 2)
void x11_simd512_gpu_compress_64_maxwell(uint32_t threads, uint32_t *g_hash, uint4 *g_fft4, uint32_t *g_state)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *Hash = &g_hash[thread * 16];
		Compression1(Hash, thread, g_fft4, g_state);
		Compression2(thread, g_fft4, g_state);
	}
}

__global__ __launch_bounds__(TPB, 2)
void x11_simd512_gpu_final_64(uint32_t threads, uint32_t *g_hash, uint4 *g_fft4, uint32_t *g_state)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *Hash = &g_hash[thread * 16];
		Final(Hash, thread, g_fft4, g_state);
	}
}


__host__
int x11_simd512_cpu_init(int thr_id, uint32_t threads)
{
	size_t temp4size = sizeof(uint4) * 64 * threads;
	CUDA_SAFE_CALL(cudaMalloc(&d_temp4[thr_id], temp4size));
	CUDA_SAFE_CALL(cudaMalloc(&d_state[thr_id], sizeof(int) * 32 * threads));

	// Texture for 128-Bit Zugriffe
	cudaChannelFormatDesc channelDesc128 = cudaCreateChannelDesc<uint4>();
	texRef1D_128.normalized = 0;
	texRef1D_128.filterMode = cudaFilterModePoint;
	texRef1D_128.addressMode[0] = cudaAddressModeClamp;

	CUDA_SAFE_CALL(cudaBindTexture(0, &texRef1D_128, d_temp4[thr_id], &channelDesc128, temp4size));

	return 0;
}

__host__
void x11_simd512_cpu_free(int thr_id)
{
	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] >= 300 && cuda_arch[dev_id] >= 300) {
		cudaFree(d_temp4[thr_id]);
		cudaFree(d_state[thr_id]);
	}
}

__host__
void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB;
	int dev_id = device_map[thr_id];

	dim3 block(threadsperblock);
	dim3 grid((threads + threadsperblock-1) / threadsperblock);
	dim3 gridX8(grid.x * 8);

	x11_simd512_gpu_expand_64 <<<gridX8, block>>> (threads, d_hash, d_temp4[thr_id]);
	CUDA_SAFE_CALL(cudaGetLastError());

	if (device_sm[dev_id] >= 500 && cuda_arch[dev_id] >= 500)
	{
		x11_simd512_gpu_compress_64_maxwell <<< grid, block, 0, gpustream[thr_id] >>> (threads, d_hash, d_temp4[thr_id], d_state[thr_id]);
		CUDA_SAFE_CALL(cudaGetLastError());
	}
	else
	{
		x11_simd512_gpu_compress1_64 <<< grid, block, 0, gpustream[thr_id] >>> (threads, d_hash, d_temp4[thr_id], d_state[thr_id]);
		CUDA_SAFE_CALL(cudaGetLastError());
		x11_simd512_gpu_compress2_64 <<< grid, block, 0, gpustream[thr_id] >>> (threads, d_temp4[thr_id], d_state[thr_id]);
		CUDA_SAFE_CALL(cudaGetLastError());
	}

	x11_simd512_gpu_final_64 <<<grid, block, 0, gpustream[thr_id] >>> (threads, d_hash, d_temp4[thr_id], d_state[thr_id]);
	CUDA_SAFE_CALL(cudaGetLastError());

	//MyStreamSynchronize(NULL, order, thr_id);
}
