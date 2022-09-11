#ifndef __MINER_H__
#define __MINER_H__

#ifdef __cplusplus
#include <algorithm>
#include <cstring>
#include <cinttypes>
#include <cstdlib>
#include <cstddef>
#else
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#endif
#include <sys/time.h>

#include <pthread.h>
#include <jansson.h>
#include <curl/curl.h>

#ifndef WIN32
#include "ccminer-config.h"
#else
#include "ccminer-config-win.h"
#endif

#ifdef WIN32
#ifndef __cplusplus
#define inline __inline
#define snprintf(...) _snprintf(__VA_ARGS__)
#endif
#undef strdup
#define strdup(x) _strdup(x)
#define strncasecmp(x,y,z) _strnicmp(x,y,z)
#define strcasecmp(x,y) _stricmp(x,y)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#undef HAVE_ALLOCA_H
#undef HAVE_SYSLOG_H
#endif

#ifdef HAVE_ALLOCA_H
# include <alloca.h>
#elif !defined alloca
# ifdef __GNUC__
#  define alloca __builtin_alloca
# elif defined _AIX
#  define alloca __alloca
# elif defined _MSC_VER
#  include <malloc.h>
#  define alloca _alloca
# elif !defined HAVE_ALLOCA
#  ifdef  __cplusplus
extern "C"
#  endif
void *alloca (size_t);
# endif
#endif

#include "compat.h"

#ifdef _MSC_VER
#define THREAD __declspec(thread)
#else
#define THREAD __thread
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
# define _ALIGN(x) __align__(x)
#elif _MSC_VER
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
#endif

#ifdef HAVE_SYSLOG_H
#include <syslog.h>
#define LOG_BLUE 0x10
#define LOG_HW 0x20
#define LOG_RAW  0x99
#else
enum
{
	LOG_ERR,
	LOG_WARNING,
	LOG_NOTICE,
	LOG_INFO,
	LOG_DEBUG,
	/* custom notices */
	LOG_BLUE = 0x10,
	LOG_HW = 0x20,
	LOG_RAW = 0x99
};
#endif
#ifdef __cplusplus
using namespace std;
#endif

typedef unsigned char uchar;

#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifndef __cplusplus
#ifndef max
# define max(a, b)  ((a) > (b) ? (a) : (b))
#endif
#ifndef min
# define min(a, b)  ((a) < (b) ? (a) : (b))
#endif
#endif

#ifndef UINT32_MAX
/* for gcc 4.4 */
#define UINT32_MAX UINT_MAX
#endif

static inline bool is_windows(void) {
#ifdef WIN32
        return true;
#else
        return false;
#endif
}

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define WANT_BUILTIN_BSWAP
#endif

static inline uint32_t swab32(uint32_t x)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap32(x);
#else
#ifdef _MSC_VER
	return _byteswap_ulong(x);
#else
	return ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu));
#endif
#endif
}

static inline uint64_t swab64(uint64_t x)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap64(x);
#else
#ifdef _MSC_VER
	return _byteswap_uint64(x);
#else
	return (((uint64_t)swab32((uint32_t)((x)& 0xffffffffu)) << 32) | (uint64_t)swab32((uint32_t)((x) >> 32)));
#endif
#endif
}

static inline void swab256(void *dest_p, const void *src_p)
{
	uint32_t *dest = (uint32_t *) dest_p;
	const uint32_t *src = (const uint32_t *) src_p;

	dest[0] = swab32(src[7]);
	dest[1] = swab32(src[6]);
	dest[2] = swab32(src[5]);
	dest[3] = swab32(src[4]);
	dest[4] = swab32(src[3]);
	dest[5] = swab32(src[2]);
	dest[6] = swab32(src[1]);
	dest[7] = swab32(src[0]);
}

#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

#if !HAVE_DECL_BE32DEC
static inline uint32_t be32dec(const void *pp)
{
	return swab32(*((uint32_t*)pp));
}
#endif

#if !HAVE_DECL_LE32DEC
static inline uint32_t le32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
	    ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}
#endif

#if !HAVE_DECL_BE32ENC
static inline void be32enc(void *pp, uint32_t x)
{
	*((uint32_t*)pp) = swab32(x);
}
#endif

#if !HAVE_DECL_LE32ENC
static inline void le32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
	p[2] = (x >> 16) & 0xff;
	p[3] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_BE16DEC
static inline uint16_t be16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[1]) + ((uint16_t)(p[0]) << 8));
}
#endif

#if !HAVE_DECL_BE16ENC
static inline void be16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[1] = x & 0xff;
	p[0] = (x >> 8) & 0xff;
}
#endif

#if !HAVE_DECL_LE16DEC
static inline uint16_t le16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[0]) + ((uint16_t)(p[1]) << 8));
}
#endif

#if !HAVE_DECL_LE16ENC
static inline void le16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
}
#endif

/* used for struct work */
void *aligned_calloc(int size);
void aligned_free(void *ptr);

#if JANSSON_MAJOR_VERSION >= 2
#define JSON_LOADS(str, err_ptr) json_loads((str), 0, (err_ptr))
#else
#define JSON_LOADS(str, err_ptr) json_loads((str), (err_ptr))
#endif

#define USER_AGENT PACKAGE_NAME "/" PACKAGE_VERSION

#ifdef __cplusplus
extern "C" {
#endif

	void sha256_init(uint32_t *state);
	void sha256t_init(uint32_t *state);
	void sha256_transform(uint32_t *state, const uint32_t *block, int swap);
	void sha256d(unsigned char *hash, const unsigned char *data, int len);

#ifdef __cplusplus
}
#endif

struct work_restart
{
	volatile unsigned long	restart;
	char			padding[128 - sizeof(unsigned long)];
};
extern struct work_restart *work_restart;

bool fulltest(const uint32_t *hash, const uint32_t *target);

extern int scanhash_deep(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_doom(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_fugue256(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_groestlcoin(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_c11(int thr_id, uint32_t *pdata,
						uint32_t *ptarget, uint32_t max_nonce,
						uint32_t *hashes_done);

extern int scanhash_keccak256(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_myriad(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_jackpot(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_quark(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);


extern int scanhash_blake256(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done, int8_t blakerounds);

extern int scanhash_fresh(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_lyra2v2(int thr_id, uint32_t *pdata,
	const uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_lyra2v3(int thr_id, uint32_t *pdata,
							const uint32_t *ptarget, uint32_t max_nonce,
							uint32_t *hashes_done);

extern int scanhash_nist5(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_pentablake(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_qubit(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);


extern int scanhash_skeincoin(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_s3(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_whc(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_whirlpoolx(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_x11(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_x13(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_x14(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_x15(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_x17(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_bitcoin(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_neoscrypt(bool stratum, int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_sia(int thr_id, uint32_t *pdata,
												uint32_t *ptarget, uint32_t max_nonce,
												uint32_t *hashes_done);

extern int scanhash_novo(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

extern int scanhash_rad(int thr_id, uint32_t *pdata,
	uint32_t *ptarget, uint32_t max_nonce,
	uint32_t *hashes_done);

/* api related */
void *api_thread(void *userdata);
void api_set_throughput(int thr_id, uint32_t throughput);
void bench_set_throughput(int thr_id, uint32_t throughput);

struct monitor_info
{
	uint32_t gpu_temp;
	uint32_t gpu_fan;
	uint32_t gpu_clock;
	uint32_t gpu_memclock;
	uint32_t gpu_power;

	pthread_mutex_t lock;
	pthread_cond_t sampling_signal;
	volatile bool sampling_flag;
	time_t tm_displayed;
};

struct cgpu_info
{
	uint8_t gpu_id;
	uint8_t thr_id;
	uint16_t hw_errors;
	unsigned accepted;
	uint32_t rejected;
	double khashes;
	int has_monitoring;
	float gpu_temp;
	uint16_t gpu_fan;
	uint16_t gpu_fan_rpm;
	uint16_t gpu_arch;
	uint32_t gpu_clock;
	uint32_t gpu_memclock;
	uint64_t gpu_mem;
	uint64_t gpu_memfree;
	uint32_t gpu_power;
	uint32_t gpu_plimit;
	double gpu_vddc;
	int16_t gpu_pstate;
	int16_t gpu_bus;
	uint16_t gpu_vid;
	uint16_t gpu_pid;

	int8_t nvml_id;
	int8_t nvapi_id;

	char gpu_sn[64];
	char gpu_desc[64];
	double intensity;
	uint32_t throughput;

	struct monitor_info monitor;
};

struct thr_api {
	int id;
	pthread_t pth;
	struct thread_q	*q;
};

struct stats_data {
	uint32_t uid;
	uint32_t tm_stat;
	uint32_t hashcount;
	uint32_t height;
	double difficulty;
	double hashrate;
	uint8_t thr_id;
	uint8_t gpu_id;
	uint8_t hashfound;
	uint8_t ignored;
};

struct hashlog_data {
	uint32_t tm_sent;
	uint32_t height;
	uint32_t njobid;
	uint32_t nonce;
	uint32_t scanned_from;
	uint32_t scanned_to;
	uint32_t last_from;
	uint32_t tm_add;
	uint32_t tm_upd;
};

/* end of api */

struct thr_info {
	int		id;
	pthread_t	pth;
	struct thread_q	*q;
	struct cgpu_info gpu;
};

extern int cuda_num_devices();
extern int cuda_version();
extern int cuda_gpu_clocks(struct cgpu_info *gpu);
int cuda_gpu_info(struct cgpu_info *gpu);
extern bool opt_verify;
extern bool opt_benchmark;
extern bool opt_debug;
extern bool opt_quiet;
extern bool opt_protocol;
extern bool opt_tracegpu;
extern int opt_n_threads;
extern int num_cpus;
extern int active_gpus;
extern int opt_timeout;
extern bool want_longpoll;
extern bool have_longpoll;
extern bool want_stratum;
extern bool have_stratum;
extern bool opt_stratum_stats;
extern char *opt_cert;
extern char *opt_proxy;
extern long opt_proxy_type;
extern bool use_syslog;
extern bool use_colors;
extern pthread_mutex_t applog_lock;
extern struct thr_info *thr_info;
extern int longpoll_thr_id;
extern int stratum_thr_id;
extern int api_thr_id;
extern bool opt_trust_pool;
extern volatile bool abort_flag;
extern uint64_t global_hashrate;
extern double   global_diff;
extern unsigned int cudaschedule;

#define MAX_GPUS 16
extern char* device_name[MAX_GPUS];
extern int device_map[MAX_GPUS];
extern long  device_sm[MAX_GPUS];
extern uint32_t device_plimit[MAX_GPUS];
extern uint32_t gpus_intensity[MAX_GPUS];
double throughput2intensity(uint32_t throughput);
extern void gpulog(int prio, int thr_id, const char *fmt, ...);

#define CL_N    "\x1B[0m"
#define CL_RED  "\x1B[31m"
#define CL_GRN  "\x1B[32m"
#define CL_YLW  "\x1B[33m"
#define CL_BLU  "\x1B[34m"
#define CL_MAG  "\x1B[35m"
#define CL_CYN  "\x1B[36m"

#define CL_BLK  "\x1B[22;30m" /* black */
#define CL_RD2  "\x1B[22;31m" /* red */
#define CL_GR2  "\x1B[22;32m" /* green */
#define CL_YL2  "\x1B[22;33m" /* dark yellow */
#define CL_BL2  "\x1B[22;34m" /* blue */
#define CL_MA2  "\x1B[22;35m" /* magenta */
#define CL_CY2  "\x1B[22;36m" /* cyan */
#define CL_SIL  "\x1B[22;37m" /* gray */

#ifdef WIN32
#define CL_GRY  "\x1B[01;30m" /* dark gray */
#else
#define CL_GRY  "\x1B[90m"    /* dark gray selectable in putty */
#endif
#define CL_LRD  "\x1B[01;31m" /* light red */
#define CL_LGR  "\x1B[01;32m" /* light green */
#define CL_LYL  "\x1B[01;33m" /* tooltips */
#define CL_LBL  "\x1B[01;34m" /* light blue */
#define CL_LMA  "\x1B[01;35m" /* light magenta */
#define CL_LCY  "\x1B[01;36m" /* light cyan */

#define CL_WHT  "\x1B[01;37m" /* white */

void format_hashrate(double hashrate, char *output);
void applog(int prio, const char *fmt, ...);
json_t *json_rpc_call(CURL *curl, const char *url, const char *userpass, const char *rpc_req, bool, bool, int *);
void cbin2hex(char *out, const char *in, size_t len);
char *bin2hex(const unsigned char *in, size_t len);
bool hex2bin(unsigned char *p, const char *hexstr, size_t len);
int timeval_subtract(struct timeval *result, struct timeval *x, struct timeval *y);
void diff_to_target(uint32_t *target, double diff);
void get_currentalgo(char* buf, int sz);
uint32_t device_intensity(int thr_id, const char *func, uint32_t defcount);

struct stratum_job {
	char *job_id;
	unsigned char prevhash[32];
	size_t coinbase_size;
	unsigned char *coinbase;
	unsigned char *xnonce2;
	int merkle_count;
	unsigned char **merkle;
	unsigned char version[4];
	unsigned char nbits[4];
	unsigned char ntime[4];
	bool clean;
	unsigned char nreward[2];
	uint32_t height;
	double diff;
};

struct stratum_ctx {
	char *url;

	CURL *curl;
	char *curl_url;
	curl_socket_t sock;
	size_t sockbuf_size;
	char *sockbuf;
	pthread_mutex_t sock_lock;

	double next_diff;

	char *session_id;
	size_t xnonce1_size;
	unsigned char *xnonce1;
	size_t xnonce2_size;
	struct stratum_job job;
	pthread_mutex_t work_lock;

	struct timeval tv_submit;
	uint32_t answer_msec;
	uint32_t disconnects;
	time_t tm_connected;

	int srvtime_diff;
};

struct work {
	uint32_t data[64];
	size_t datasize;
	uint32_t midstate[8];
	uint32_t target[8];

	char job_id[128];
	size_t xnonce2_len;
	uchar xnonce2[32];

	union {
		uint32_t u32[2];
		uint64_t u64[1];
	} noncerange;

	double difficulty;
	uint32_t height;

	uint32_t scanned_from;
	uint32_t scanned_to;
};

enum sha_algos
{
	ALGO_INVALID,
	ALGO_NOVO,
	ALGO_RAD
};

bool stratum_socket_full(struct stratum_ctx *sctx, int timeout);
bool stratum_send_line(struct stratum_ctx *sctx, char *s);
char *stratum_recv_line(struct stratum_ctx *sctx);
bool stratum_connect(struct stratum_ctx *sctx, const char *url);
void stratum_disconnect(struct stratum_ctx *sctx);
bool stratum_subscribe(struct stratum_ctx *sctx);
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass,bool extranonce);
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s);

void hashlog_remember_submit(struct work* work, uint32_t nonce);
void hashlog_remember_scan_range(struct work* work);
uint32_t hashlog_already_submittted(char* jobid, uint32_t nounce);
uint32_t hashlog_get_last_sent(char* jobid);
uint64_t hashlog_get_scan_range(char* jobid);
int  hashlog_get_history(struct hashlog_data *data, int max_records);
void hashlog_purge_old(void);
void hashlog_purge_job(char* jobid);
void hashlog_purge_all(void);
void hashlog_dump_job(char* jobid);
void hashlog_getmeminfo(uint64_t *mem, uint32_t *records);

void stats_remember_speed(int thr_id, uint32_t hashcount, double hashrate, uint8_t found, uint32_t height);
double stats_get_speed(int thr_id, double def_speed);
int  stats_get_history(int thr_id, struct stats_data *data, int max_records);
void stats_purge_old(void);
void stats_purge_all(void);
void stats_getmeminfo(uint64_t *mem, uint32_t *records);

struct thread_q;

extern struct thread_q *tq_new(void);
extern void tq_free(struct thread_q *tq);
extern bool tq_push(struct thread_q *tq, void *data);
extern void *tq_pop(struct thread_q *tq, const struct timespec *abstime);
extern void tq_freeze(struct thread_q *tq);
extern void tq_thaw(struct thread_q *tq);

void proper_exit(int reason);
void restart_threads(void);

size_t time2str(char* buf, time_t timer);
char* atime2str(time_t timer);

void applog_hash(unsigned char *hash);
void applog_compare_hash(unsigned char *hash, unsigned char *hash2);

void print_hash_tests(void);

void blake256hash(void *output, const void *input, int8_t rounds);
void deephash(void *state, const void *input);
void doomhash(void *state, const void *input);
void fresh_hash(void *state, const void *input);
void fugue256_hash(unsigned char* output, const unsigned char* input, int len);
void keccak256_hash(void *state, const void *input);
unsigned int jackpothash(void *state, const void *input);
void groestlhash(void *state, const void *input);
void myriadhash(void *state, const void *input);
void nist5hash(void *state, const void *input);
void pentablakehash(void *output, const void *input);
void quarkhash(void *state, const void *input);
void qubithash(void *state, const void *input);
void skeincoinhash(void *output, const void *input);
void s3hash(void *output, const void *input);
void wcoinhash(void *state, const void *input);
void x11hash(void *output, const void *input);
void x13hash(void *output, const void *input);
void x14hash(void *output, const void *input);
void x15hash(void *output, const void *input);
void x17hash(void *output, const void *input);

#endif /* __MINER_H__ */
