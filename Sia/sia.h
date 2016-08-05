#pragma once

#define MAXRESULTS 8
#define npt 128
#define blocksize 256

void sia_gpu_init(int thr_id);
void sia_precalc(cudaStream_t cudastream, const uint64_t *blockHeader);
void sia_gpu_hash(cudaStream_t cudastream, int thr_id, uint32_t threads, uint64_t *headerHash, uint32_t *nonceOut, uint64_t target, uint64_t startnonce);
