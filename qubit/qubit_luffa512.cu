/*
 * luffa_for_32.c
 * Version 2.0 (Sep 15th 2009)
 *
 * Copyright (C) 2008-2009 Hitachi, Ltd. All rights reserved.
 *
 * Hitachi, Ltd. is the owner of this software and hereby grant
 * the U.S. Government and any interested party the right to use
 * this software for the purposes of the SHA-3 evaluation process,
 * notwithstanding that this software is copyrighted.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */
#ifdef __cplusplus
#include <cstdint>
#include <cstdio>
#else
#include <stdint.h>
#include <stdio.h>
#endif
#include <memory.h>
#include "miner.h"
#include "cuda_helper.h"



#ifndef UINT32_MAX
#define UINT32_MAX UINT_MAX
#endif

static THREAD unsigned char PaddedMessage[128];
__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)
__constant__ uint32_t c_Target[8];
__constant__ uint32_t statebufferpre[8];
__constant__ uint32_t statechainvpre[40];



static uint32_t *h_resNounce[MAX_GPUS];
static uint32_t *d_resNounce[MAX_GPUS];

#define NBN 1 /* max results, could be 2, see blake32.cu */
#if NBN > 1
static uint32_t extra_results[2] = { UINT32_MAX, UINT32_MAX };
#endif

#define BYTES_SWAP32(x) cuda_swab32(x)

#define MULT2(a,j)\
	tmp = a[7+(8*j)];\
	a[7+(8*j)] = a[6+(8*j)];\
	a[6+(8*j)] = a[5+(8*j)];\
	a[5+(8*j)] = a[4+(8*j)];\
	a[4+(8*j)] = a[3+(8*j)] ^ tmp;\
	a[3+(8*j)] = a[2+(8*j)] ^ tmp;\
	a[2+(8*j)] = a[1+(8*j)];\
	a[1+(8*j)] = a[0+(8*j)] ^ tmp;\
	a[0+(8*j)] = tmp;

#define TWEAK(a0,a1,a2,a3,j)\
	a0 = ROTL32(a0,j);\
	a1 = ROTL32(a1,j);\
	a2 = ROTL32(a2,j);\
	a3 = ROTL32(a3,j);

#define STEP(c0,c1)\
	SUBCRUMB(chainv[0],chainv[1],chainv[2],chainv[3],tmp);\
	SUBCRUMB(chainv[5],chainv[6],chainv[7],chainv[4],tmp);\
	MIXWORD(chainv[0],chainv[4]);\
	MIXWORD(chainv[1],chainv[5]);\
	MIXWORD(chainv[2],chainv[6]);\
	MIXWORD(chainv[3],chainv[7]);\
	ADD_CONSTANT(chainv[0],chainv[4],c0,c1);

#define SUBCRUMB(a0,a1,a2,a3,a4)\
	a4  = a0;\
	a0 |= a1;\
	a2 ^= a3;\
	a1  = ~a1;\
	a0 ^= a3;\
	a3 &= a4;\
	a1 ^= a3;\
	a3 ^= a2;\
	a2 &= a0;\
	a0  = ~a0;\
	a2 ^= a1;\
	a1 |= a3;\
	a4 ^= a1;\
	a3 ^= a2;\
	a2 &= a1;\
	a1 ^= a0;\
	a0  = a4;

#define MIXWORD(a0,a4)\
	a4 ^= a0;\
	a0  = (a0<<2) | (a0>>(30));\
	a0 ^= a4;\
	a4  = (a4<<14) | (a4>>(18));\
	a4 ^= a0;\
	a0  = (a0<<10) | (a0>>(22));\
	a0 ^= a4;\
	a4  = (a4<<1) | (a4>>(31));

#define ADD_CONSTANT(a0,b0,c0,c1)\
	a0 ^= c0;\
	b0 ^= c1;

/* initial values of chaining variables */
__constant__ uint32_t c_CNS[80] = {
	0x303994a6,0xe0337818,0xc0e65299,0x441ba90d,
	0x6cc33a12,0x7f34d442,0xdc56983e,0x9389217f,
	0x1e00108f,0xe5a8bce6,0x7800423d,0x5274baf4,
	0x8f5b7882,0x26889ba7,0x96e1db12,0x9a226e9d,
	0xb6de10ed,0x01685f3d,0x70f47aae,0x05a17cf4,
	0x0707a3d4,0xbd09caca,0x1c1e8f51,0xf4272b28,
	0x707a3d45,0x144ae5cc,0xaeb28562,0xfaa7ae2b,
	0xbaca1589,0x2e48f1c1,0x40a46f3e,0xb923c704,
	0xfc20d9d2,0xe25e72c1,0x34552e25,0xe623bb72,
	0x7ad8818f,0x5c58a4a4,0x8438764a,0x1e38e2e7,
	0xbb6de032,0x78e38b9d,0xedb780c8,0x27586719,
	0xd9847356,0x36eda57f,0xa2c78434,0x703aace7,
	0xb213afa5,0xe028c9bf,0xc84ebe95,0x44756f91,
	0x4e608a22,0x7e8fce32,0x56d858fe,0x956548be,
	0x343b138f,0xfe191be2,0xd0ec4e3d,0x3cb226e5,
	0x2ceb4882,0x5944a28e,0xb3ad2208,0xa1c4c355,
	0xf0d2e9e3,0x5090d577,0xac11d7fa,0x2d1925ab,
	0x1bcb66f2,0xb46496ac,0x6f2d9bc9,0xd1925ab0,
	0x78602649,0x29131ab6,0x8edae952,0x0fc053c3,
	0x3b6ba548,0x3f014f0c,0xedae9520,0xfc053c31};

static uint32_t h_CNS[80] = {
	0x303994a6, 0xe0337818, 0xc0e65299, 0x441ba90d,
	0x6cc33a12, 0x7f34d442, 0xdc56983e, 0x9389217f,
	0x1e00108f, 0xe5a8bce6, 0x7800423d, 0x5274baf4,
	0x8f5b7882, 0x26889ba7, 0x96e1db12, 0x9a226e9d,
	0xb6de10ed, 0x01685f3d, 0x70f47aae, 0x05a17cf4,
	0x0707a3d4, 0xbd09caca, 0x1c1e8f51, 0xf4272b28,
	0x707a3d45, 0x144ae5cc, 0xaeb28562, 0xfaa7ae2b,
	0xbaca1589, 0x2e48f1c1, 0x40a46f3e, 0xb923c704,
	0xfc20d9d2, 0xe25e72c1, 0x34552e25, 0xe623bb72,
	0x7ad8818f, 0x5c58a4a4, 0x8438764a, 0x1e38e2e7,
	0xbb6de032, 0x78e38b9d, 0xedb780c8, 0x27586719,
	0xd9847356, 0x36eda57f, 0xa2c78434, 0x703aace7,
	0xb213afa5, 0xe028c9bf, 0xc84ebe95, 0x44756f91,
	0x4e608a22, 0x7e8fce32, 0x56d858fe, 0x956548be,
	0x343b138f, 0xfe191be2, 0xd0ec4e3d, 0x3cb226e5,
	0x2ceb4882, 0x5944a28e, 0xb3ad2208, 0xa1c4c355,
	0xf0d2e9e3, 0x5090d577, 0xac11d7fa, 0x2d1925ab,
	0x1bcb66f2, 0xb46496ac, 0x6f2d9bc9, 0xd1925ab0,
	0x78602649, 0x29131ab6, 0x8edae952, 0x0fc053c3,
	0x3b6ba548, 0x3f014f0c, 0xedae9520, 0xfc053c31 };


static __device__ __forceinline__
void rnd512(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

#pragma unroll 8
	for (i = 0; i<8; i++)
	{
		t[i] = statechainv[i];
#pragma unroll
		for (j = 1; j<5; j++)
		{
			t[i] ^= statechainv[i + 8 * j];
		}
	}

	MULT2(t, 0);

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			statechainv[i + 8 * j] ^= t[i];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			statechainv[i + 8 * j] ^= statebuffer[i];
		}
		MULT2(statebuffer, 0);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		chainv[i] = statechainv[i];
	}

#pragma unroll 1
	for (i = 0; i<8; i++) 
	{
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) 
	{
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

#pragma unroll 1
	for (i = 0; i<8; i++) 
	{
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

#pragma unroll 1
	for (i = 0; i<8; i++) 
	{
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

#pragma unroll 1
	for (i = 0; i<8; i++) 
	{
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) 
	{
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

#pragma unroll 1
	for (i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 32] = chainv[i];
	}
}


static __device__ __forceinline__
void rnd512_first(uint32_t *statebuffer, uint32_t *statechainv)
{
	uint32_t chainv[8];
	uint32_t tmp;
	int i;

	statechainv[0 + 8 * 0] ^= statebuffer[0];
	statechainv[1 + 8 * 0] ^= statebuffer[1];
	statechainv[2 + 8 * 0] ^= statebuffer[2];
	statechainv[3 + 8 * 0] ^= statebuffer[3];
	statechainv[4 + 8 * 0] ^= statebuffer[4];


	statechainv[1 + 8 * 1] ^= statebuffer[0];
	statechainv[2 + 8 * 1] ^= statebuffer[1];
	statechainv[3 + 8 * 1] ^= statebuffer[2];
	statechainv[4 + 8 * 1] ^= statebuffer[3];
	statechainv[5 + 8 * 1] ^= statebuffer[4];


	statechainv[2 + 8 * 2] ^= statebuffer[0];
	statechainv[3 + 8 * 2] ^= statebuffer[1];
	statechainv[4 + 8 * 2] ^= statebuffer[2];
	statechainv[5 + 8 * 2] ^= statebuffer[3];
	statechainv[6 + 8 * 2] ^= statebuffer[4];


	statechainv[3 + 8 * 3] ^= statebuffer[0];
	statechainv[4 + 8 * 3] ^= statebuffer[1];
	statechainv[5 + 8 * 3] ^= statebuffer[2];
	statechainv[6 + 8 * 3] ^= statebuffer[3];
	statechainv[7 + 8 * 3] ^= statebuffer[4];

	statechainv[4 + 8 * 4] ^= statebuffer[0] ^ statebuffer[4];
	statechainv[5 + 8 * 4] ^= statebuffer[1];
	statechainv[6 + 8 * 4] ^= statebuffer[2];
	statechainv[7 + 8 * 4] ^= statebuffer[3];
	statechainv[0 + 8 * 4] ^= statebuffer[4];

	statechainv[1 + 8 * 4] = (statechainv[1 + 8 * 4] ^ statebuffer[4]);
	statechainv[3 + 8 * 4] = (statechainv[3 + 8 * 4] ^ statebuffer[4]);

#pragma unroll 8
	for (i = 0; i<8; i++) {
		chainv[i] = statechainv[i];
	}

#pragma unroll 1
	for (i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++)
	{
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

#pragma unroll 1
	for (i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

#pragma unroll 1
	for (i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

#pragma unroll 1
	for (i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++)
	{
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

#pragma unroll 1
	for (i = 0; i<8; i++)
	{
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		statechainv[i + 32] = chainv[i];
	}
}


void rnd512cpu(uint32_t *statebuffer, uint32_t *statechainv)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

	for (i = 0; i<8; i++)
	{
		t[i] = statechainv[i];
		for (j = 1; j<5; j++)
		{
			t[i] ^= statechainv[i + 8 * j];
		}
	}

	MULT2(t, 0);

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[i + 8 * j] ^= t[i];
		}
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++)
	{
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++)
	{
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}

	for (j = 0; j<5; j++)
	{
		for (i = 0; i<8; i++)
		{
			statechainv[i + 8 * j] ^= statebuffer[i];
		}
		MULT2(statebuffer, 0);
	}

	for (i = 0; i<8; i++)
	{
		chainv[i] = statechainv[i];
	}

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i)], h_CNS[(2 * i) + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i] = chainv[i];
		chainv[i] = statechainv[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 16], h_CNS[(2 * i) + 16 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 8] = chainv[i];
		chainv[i] = statechainv[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 32], h_CNS[(2 * i) + 32 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 16] = chainv[i];
		chainv[i] = statechainv[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 48], h_CNS[(2 * i) + 48 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 24] = chainv[i];
		chainv[i] = statechainv[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

	for (i = 0; i<8; i++)
	{
		STEP(h_CNS[(2 * i) + 64], h_CNS[(2 * i) + 64 + 1]);
	}

	for (i = 0; i<8; i++)
	{
		statechainv[i + 32] = chainv[i];
	}
}

static __device__ __forceinline__
void Update512(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv, const uint32_t *const __restrict__ data)
{
#pragma unroll 8
	for (int i = 0; i<8; i++)
		statebuffer[i] = BYTES_SWAP32((data)[i]);
	rnd512(statebuffer, statechainv);

#pragma unroll 8
	for(int i=0;i<8;i++)
		statebuffer[i] = BYTES_SWAP32(((data))[i+8]);
	rnd512(statebuffer, statechainv);
#pragma unroll 4
	for(int i=0;i<4;i++)
		statebuffer[i] = BYTES_SWAP32(((data))[i+16]);
}

/***************************************************/
static __device__ __forceinline__
void rnd512_nullhash(uint32_t *state)
{
	int i, j;
	uint32_t t[40];
	uint32_t chainv[8];
	uint32_t tmp;

#pragma unroll 8
	for (i = 0; i<8; i++) {
		t[i] = state[i + 8 * 0];
#pragma unroll 4
		for (j = 1; j<5; j++) {
			t[i] ^= state[i + 8 * j];
		}
	}

	MULT2(t, 0);

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			state[i + 8 * j] ^= t[i];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = state[i + 8 * j];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
		MULT2(state, j);
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			state[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = state[i + 8 * j];
		}
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
		MULT2(state, j);
	}

#pragma unroll 5
	for (j = 0; j<5; j++) {
#pragma unroll 8
		for (i = 0; i<8; i++) {
			state[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		chainv[i] = state[i];
	}

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i)], c_CNS[(2 * i) + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		state[i] = chainv[i];
		chainv[i] = state[i + 8];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 1);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 16], c_CNS[(2 * i) + 16 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		state[i + 8] = chainv[i];
		chainv[i] = state[i + 16];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 2);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 32], c_CNS[(2 * i) + 32 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		state[i + 16] = chainv[i];
		chainv[i] = state[i + 24];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 3);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 48], c_CNS[(2 * i) + 48 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		state[i + 24] = chainv[i];
		chainv[i] = state[i + 32];
	}

	TWEAK(chainv[4], chainv[5], chainv[6], chainv[7], 4);

#pragma unroll 1
	for (i = 0; i<8; i++) {
		STEP(c_CNS[(2 * i) + 64], c_CNS[(2 * i) + 64 + 1]);
	}

#pragma unroll 8
	for (i = 0; i<8; i++) {
		state[i + 32] = chainv[i];
	}
}


static __device__ __forceinline__
void finalization512(uint32_t *const __restrict__ statebuffer, uint32_t *const __restrict__ statechainv, uint32_t *const __restrict__ b)
{
	int i,j;

	statebuffer[4] = 0x80000000;
#pragma unroll 3
	for(int i=5;i<8;i++)
		statebuffer[i] = 0;
	rnd512(statebuffer, statechainv);

	rnd512_nullhash(statechainv);

#pragma unroll 8
	for(i=0;i<8;i++) {
		b[i] = 0;
#pragma unroll 5
		for(j=0;j<5;j++) {
			b[i] ^= statechainv[i+8*j];
		}
		b[i] = BYTES_SWAP32((b[i]));
	}

	rnd512_nullhash(statechainv);

#pragma unroll 8
	for(i=0;i<8;i++)
	{
		b[8+i] = 0;
#pragma unroll 5
		for(j=0;j<5;j++)
		{
			b[8+i] ^= statechainv[i+8*j];
		}
		b[8+i] = BYTES_SWAP32((b[8+i]));
	}
}

/***************************************************/
// Die Hash-Funktion
__global__
#if __CUDA_ARCH__ == 500
__launch_bounds__(256, 4)
#endif
void qubit_luffa512_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		int i, j;
		const uint32_t nounce = startNounce + thread;
		uint64_t buff[16];

#pragma unroll
		for (int i=8; i < 16; ++i)
			buff[i] = c_PaddedMessage80[i];

		// die Nounce durch die thread-spezifische ersetzen
		buff[9] = REPLACE_HIWORD(buff[9], cuda_swab32(nounce));

		uint32_t statebuffer[8];
		uint32_t statechainv[40];

#pragma unroll 4
		for (int i = 0; i<4; i++)
			statebuffer[i] = BYTES_SWAP32(((uint32_t*)buff)[i + 16]);
#pragma unroll 4
		for (int i = 4; i<8; i++)
			statebuffer[i] = statebufferpre[i];
#pragma unroll 
		for (int i = 0; i<40; i++)
			statechainv[i] = statechainvpre[i];

		uint32_t *outHash = outputHash + 16 * thread;

		statebuffer[4] = 0x80000000;

		rnd512_first(statebuffer, statechainv);
		rnd512_nullhash(statechainv);


		#pragma unroll
		for (i = 0; i<8; i++) 
		{
			buff[i] = statechainv[i];
			#pragma unroll
			for (j = 1; j<5; j++) {
				buff[i] ^= statechainv[i + 8 * j];
			}
			outHash[i] = BYTES_SWAP32((buff[i]));
		}

		rnd512_nullhash(statechainv);

#pragma unroll 8
		for (i = 0; i<8; i++)
		{
			buff[8 + i] = statechainv[i];
#pragma unroll 5
			for (j = 1; j<5; j++)
			{
				buff[8 + i] ^= statechainv[i + 8 * j];
			}
			outHash[8 + i] = BYTES_SWAP32((buff[8 + i]));
		}


	}
}

__global__  __launch_bounds__(256,4)
void qubit_luffa512_gpu_finalhash_80(uint32_t threads, uint32_t startNounce, void *outputHash, uint32_t *resNounce, int thr_id)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		const uint32_t nounce = startNounce + thread;
		union {
			uint64_t buf64[16];
			uint32_t buf32[32];
		} buff;
		uint32_t Hash[16];

		#pragma unroll 16
		for (int i=0; i < 16; ++i)
			buff.buf64[i] = c_PaddedMessage80[i];

		// Tested nonce
		buff.buf64[9] = REPLACE_HIWORD(buff.buf64[9], cuda_swab32(nounce));

		uint32_t statebuffer[8];
		uint32_t statechainv[40] =
		{
			0x6d251e69, 0x44b051e0, 0x4eaa6fb4, 0xdbf78465,
			0x6e292011, 0x90152df4, 0xee058139, 0xdef610bb,
			0xc3b44b95, 0xd9d2f256, 0x70eee9a0, 0xde099fa3,
			0x5d9b0557, 0x8fc944b3, 0xcf1ccf0e, 0x746cd581,
			0xf7efc89d, 0x5dba5781, 0x04016ce5, 0xad659c05,
			0x0306194f, 0x666d1836, 0x24aa230a, 0x8b264ae7,
			0x858075d5, 0x36d79cce, 0xe571f7d7, 0x204b1f67,
			0x35870c6a, 0x57e9e923, 0x14bcb808, 0x7cde72ce,
			0x6c68e9be, 0x5ec41e22, 0xc825b7c7, 0xaffb4363,
			0xf5df3999, 0x0fc688f1, 0xb07224cc, 0x03e86cea
		};

		Update512(statebuffer, statechainv, buff.buf32);
		finalization512(statebuffer, statechainv, Hash);

		/* dont ask me why not a simple if (Hash[i] > c_Target[i]) return;
		 * we lose 20% in perfs without the position test */
		int position = -1;
		#pragma unroll 8
		for (int i = 7; i >= 0; i--) {
			if (Hash[i] > c_Target[i]) {
				if (position < i) {
					return;
				}
			}
			if (Hash[i] < c_Target[i]) {
				if (position < i) {
					position = i;
					//break; /* impact perfs, unroll ? */
				}
			}
		}

#if NBN == 1
		if (resNounce[0] > nounce) {
			resNounce[0] = nounce;
		}
#else
		/* keep the smallest nounce, + extra one if found */
		if (resNounce[0] > nounce) {
			resNounce[1] = resNounce[0];
			resNounce[0] = nounce;
		} else {
			resNounce[1] = nounce;
		}
#endif
	}
}

__host__
void qubit_luffa512_cpu_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_resNounce[thr_id], NBN * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMallocHost(&h_resNounce[thr_id], NBN * sizeof(uint32_t)));
}

__host__
uint32_t qubit_luffa512_cpu_finalhash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{
	uint32_t result = UINT32_MAX;
	cudaMemsetAsync(d_resNounce[thr_id], 0xff, NBN * sizeof(uint32_t), gpustream[thr_id]);
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	qubit_luffa512_gpu_finalhash_80 <<<grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_outputHash, d_resNounce[thr_id], thr_id);
	//MyStreamSynchronize(NULL, order, thr_id);
	CUDA_SAFE_CALL(cudaMemcpyAsync(h_resNounce[thr_id], d_resNounce[thr_id], NBN * sizeof(uint32_t), cudaMemcpyDeviceToHost, gpustream[thr_id]));
	cudaStreamSynchronize(gpustream[thr_id]);
	result = h_resNounce[thr_id][0];
#if NBN > 1
	extra_results[0] = h_resNounce[thr_id][1];
#endif
	return result;
}

__host__ void qubit_cpu_precalc(int thr_id)
{
	uint32_t tmp,i,j;
	uint32_t statebuffer[8];
	uint32_t t[40];
	uint32_t statechainv[40] =
	{
		0x6d251e69, 0x44b051e0, 0x4eaa6fb4, 0xdbf78465,
		0x6e292011, 0x90152df4, 0xee058139, 0xdef610bb,
		0xc3b44b95, 0xd9d2f256, 0x70eee9a0, 0xde099fa3,
		0x5d9b0557, 0x8fc944b3, 0xcf1ccf0e, 0x746cd581,
		0xf7efc89d, 0x5dba5781, 0x04016ce5, 0xad659c05,
		0x0306194f, 0x666d1836, 0x24aa230a, 0x8b264ae7,
		0x858075d5, 0x36d79cce, 0xe571f7d7, 0x204b1f67,
		0x35870c6a, 0x57e9e923, 0x14bcb808, 0x7cde72ce,
		0x6c68e9be, 0x5ec41e22, 0xc825b7c7, 0xaffb4363,
		0xf5df3999, 0x0fc688f1, 0xb07224cc, 0x03e86cea
	};

	for (i = 0; i<8; i++)
		statebuffer[i] = BYTES_SWAP32(*(((uint32_t*)PaddedMessage) + i));
	rnd512cpu(statebuffer, statechainv);

	for (i = 0; i<8; i++)
		statebuffer[i] = BYTES_SWAP32(*(((uint32_t*)PaddedMessage) + i + 8));

	rnd512cpu(statebuffer, statechainv);


	for (i = 0; i<8; i++)
	{
		t[i] = statechainv[i];
		for (j = 1; j<5; j++)
		{
			t[i] ^= statechainv[i + 8 * j];
		}
	}

	MULT2(t, 0);

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			statechainv[i + 8 * j] ^= t[i];
		}
	}
	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 1) % 5) + i];
		}
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			t[i + 8 * j] = statechainv[i + 8 * j];
		}
	}

	for (j = 0; j<5; j++) {
		MULT2(statechainv, j);
	}

	for (j = 0; j<5; j++) {
		for (i = 0; i<8; i++) {
			statechainv[8 * j + i] ^= t[8 * ((j + 4) % 5) + i];
		}
	}



	cudaMemcpyToSymbolAsync(statebufferpre, statebuffer, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]);
	cudaMemcpyToSymbolAsync(statechainvpre, statechainv, 40 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]);
}

__host__
void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	qubit_luffa512_gpu_hash_80 <<<grid, block, 0, gpustream[thr_id]>>> (threads, startNounce, d_outputHash);
}

__host__
void qubit_luffa512_cpu_setBlock_80(int thr_id, void *pdata)
{
	memcpy(PaddedMessage, pdata, 80);
	memset(PaddedMessage + 80, 0, 48);
	PaddedMessage[80] = 0x80;
	PaddedMessage[111] = 1;
	PaddedMessage[126] = 0x02;
	PaddedMessage[127] = 0x80;

	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(c_PaddedMessage80, PaddedMessage, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
	qubit_cpu_precalc(thr_id);
}

__host__
void qubit_luffa512_cpufinal_setBlock_80(int thr_id, void *pdata, const void *ptarget)
{
	unsigned char PaddedMsg[128];

	memcpy(PaddedMsg, pdata, 80);
	memset(PaddedMsg+80, 0, 48);
	PaddedMsg[80] = 0x80;
	PaddedMsg[111] = 1;
	PaddedMsg[126] = 0x02;
	PaddedMsg[127] = 0x80;

	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(c_Target, ptarget, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
	CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(c_PaddedMessage80, PaddedMsg, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice, gpustream[thr_id]));
}
