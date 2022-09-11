/*
* Copyright 2010 Jeff Garzik
* Copyright 2012-2014 pooler
* Copyright 2014 ccminer team
*
* This program is free software; you can redistribute it and/or modify it
* under the terms of the GNU General Public License as published by the Free
* Software Foundation; either version 2 of the License, or (at your option)
* any later version.  See COPYING for more details.
*/

#include <cstdio>
#include <cstdlib>
#include <ctype.h>
#include <cstdarg>
#include <cstring>
#include <cinttypes>

#include <unistd.h>
#include <jansson.h>
#include <curl/curl.h>
#include <ctime>
#ifdef WIN32
#include "compat/winansi.h"
#include <winsock2.h>
#include <mstcpip.h>
#else
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif
#include "elist.h"
#include "miner.h"


extern enum sha_algos opt_algo;
extern char curl_err_str[];
extern bool stop_mining;
extern bool send_stale;
extern const char *algo_names[];

extern bool opt_logfile;
extern FILE *logfilepointer;

bool opt_tracegpu = false;

struct data_buffer
{
	void		*buf;
	size_t		len;
};

struct upload_buffer
{
	const void	*buf;
	size_t		len;
	size_t		pos;
};

struct header_info
{
	char		*lp_path;
	char		*reason;
	char		*stratum_url;
};

struct tq_ent
{
	void			*data;
	struct list_head	q_node;
};

struct thread_q
{
	struct list_head	q;

	bool frozen;

	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
};

// input and output may point to the same location
void hexstringreverse(void *output, const void *input, size_t length)
{
	uint16_t tmp1;
	uint16_t tmp2;
	for(size_t i = 0; i < length / 4; i++)
	{
		tmp1 = *(((uint16_t*)input) + i);
		tmp2 = *(((uint16_t*)output) + (length / 2 - i));
		*(((uint16_t*)input) + i) = tmp2;
		*(((uint16_t*)output) + (length / 2 - i)) = tmp1;
	}
}

void applog(int prio, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);

#ifdef HAVE_SYSLOG_H
	if(use_syslog)
	{
		va_list ap2;
		char *buf;
		int len;

		/* custom colors to syslog prio */
		if(prio > LOG_DEBUG)
		{
			switch(prio)
			{
			case LOG_BLUE: prio = LOG_NOTICE; break;
			case LOG_HW:   prio = LOG_INFO; break;
			}
		}

		va_copy(ap2, ap);
		len = vsnprintf(NULL, 0, fmt, ap2) + 1;
		va_end(ap2);
		buf = (char*)malloc(len);
		if(vsnprintf(buf, len, fmt, ap) >= 0)
			syslog(prio, "%s", buf);
		free(buf);
	}
#else
	if(0)
	{
	}
#endif
	else
	{
		const char* color = "";
		char f[16000];
		struct tm tm, *tm_p;
		time_t now = time(NULL);

		pthread_mutex_lock(&applog_lock);
		tm_p = localtime(&now);
		memcpy(&tm, tm_p, sizeof(tm));
		pthread_mutex_unlock(&applog_lock);

		switch(prio)
		{
		case LOG_ERR:     color = CL_RED; break;
		case LOG_WARNING: color = CL_YLW; break;
		case LOG_NOTICE:  color = CL_WHT; break;
		case LOG_INFO:    color = ""; break;
		case LOG_DEBUG:   color = CL_GRY; break;

		case LOG_HW:
			color = CL_MAG;
			prio = LOG_INFO;
			break;

		case LOG_BLUE:
			prio = LOG_NOTICE;
			color = CL_CYN;
			break;
		}
		if(!use_colors)
			color = "";

		pthread_mutex_lock(&applog_lock);

		snprintf(f, sizeof(f), "[%d-%02d-%02d %02d:%02d:%02d]%s %s%s\n",
				tm.tm_year + 1900,
				tm.tm_mon + 1,
				tm.tm_mday,
				tm.tm_hour,
				tm.tm_min,
				tm.tm_sec,
				color,
				fmt,
				use_colors ? CL_N : ""
				);
		vfprintf(stdout, f, ap);	/* atomic write to stderr */
		fflush(stdout);
		if (opt_logfile)
		{
			snprintf(f, sizeof(f), "[%d-%02d-%02d %02d:%02d:%02d] %s\n",
					tm.tm_year + 1900,
					tm.tm_mon + 1,
					tm.tm_mday,
					tm.tm_hour,
					tm.tm_min,
					tm.tm_sec,
					fmt
			);
			vfprintf(logfilepointer, f, ap);	/* atomic write to logfile */
			fflush(logfilepointer);
		}
		pthread_mutex_unlock(&applog_lock);
	}
	va_end(ap);
}

extern int gpu_threads;
// Use different prefix if multiple cpu threads per gpu
// Also, auto hide LOG_DEBUG if --debug (-D) is not used
void gpulog(int prio, int thr_id, const char *fmt, ...)
{
	char _ALIGN(128) pfmt[128];
	char _ALIGN(128) line[256];
	int len, dev_id = device_map[thr_id % MAX_GPUS];
	va_list ap;

	if(prio == LOG_DEBUG && !opt_debug)
		return;

	if(gpu_threads > 1)
		len = snprintf(pfmt, 128, "GPU T%d: %s", thr_id, fmt);
	else
		len = snprintf(pfmt, 128, "GPU #%d: %s", dev_id, fmt);
	pfmt[sizeof(pfmt) - 1] = '\0';

	va_start(ap, fmt);

	if(len && vsnprintf(line, sizeof(line), pfmt, ap))
	{
		line[sizeof(line) - 1] = '\0';
		applog(prio, "%s", line);
	}
	else
	{
		fprintf(stdout, "%s OOM!\n", __func__);
	}

	va_end(ap);
}

void format_hashrate(double hashrate, char *output)
{
	char prefix = '\0';

	if(hashrate < 10000)
	{
		// nop
	}
	else if(hashrate < 1e7)
	{
		prefix = 'k';
		hashrate *= 1e-3;
	}
	else if(hashrate < 1e10)
	{
		prefix = 'M';
		hashrate *= 1e-6;
	}
	else if(hashrate < 1e13)
	{
		prefix = 'G';
		hashrate *= 1e-9;
	}
	else
	{
		prefix = 'T';
		hashrate *= 1e-12;
	}

	sprintf(
		output,
		prefix ? "%.2f %cH/s" : "%.2f H/s%c",
		hashrate, prefix
		);
}

static void databuf_free(struct data_buffer *db)
{
	if(!db)
		return;

	free(db->buf);

	memset(db, 0, sizeof(*db));
}

static size_t all_data_cb(const void *ptr, size_t size, size_t nmemb,
						  void *user_data)
{
	struct data_buffer *db = (struct data_buffer *)user_data;
	size_t len = size * nmemb;
	size_t oldlen, newlen;
	void *newmem;
	static const uchar zero = 0;

	oldlen = db->len;
	newlen = oldlen + len;

	newmem = realloc(db->buf, newlen + 1);
	if(newmem == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}

	db->buf = newmem;
	db->len = newlen;
	memcpy((char*)db->buf + oldlen, ptr, len);
	memcpy((char*)db->buf + newlen, &zero, 1);	/* null terminate */

	return len;
}

static size_t upload_data_cb(void *ptr, size_t size, size_t nmemb,
							 void *user_data)
{
	struct upload_buffer *ub = (struct upload_buffer *)user_data;
	unsigned int len = (unsigned int)(size * nmemb);

	if(len > ub->len - ub->pos)
		len = (unsigned int)(ub->len - ub->pos);

	if(len)
	{
		memcpy(ptr, (char*)ub->buf + ub->pos, len);
		ub->pos += len;
	}

	return len;
}

#if LIBCURL_VERSION_NUM >= 0x071200
static int seek_data_cb(void *user_data, curl_off_t offset, int origin)
{
	struct upload_buffer *ub = (struct upload_buffer *)user_data;

	switch(origin)
	{
	case SEEK_SET:
		ub->pos = (size_t)offset;
		break;
	case SEEK_CUR:
		ub->pos += (size_t)offset;
		break;
	case SEEK_END:
		ub->pos = ub->len + (size_t)offset;
		break;
	default:
		return 1; /* CURL_SEEKFUNC_FAIL */
	}

	return 0; /* CURL_SEEKFUNC_OK */
}
#endif

static size_t resp_hdr_cb(void *ptr, size_t size, size_t nmemb, void *user_data)
{
	struct header_info *hi = (struct header_info *)user_data;
	size_t remlen, slen, ptrlen = size * nmemb;
	char *rem, *val = NULL, *key = NULL;
	void *tmp;

	val = (char*)calloc(1, ptrlen);
	if(val == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	key = (char*)calloc(1, ptrlen);
	if(key == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}

	tmp = memchr(ptr, ':', ptrlen);
	if(!tmp || (tmp == ptr))	/* skip empty keys / blanks */
		goto out;
	slen = (size_t)((char*)tmp - (char*)ptr);
	if((slen + 1) == ptrlen)	/* skip key w/ no value */
		goto out;
	memcpy(key, ptr, slen);		/* store & nul term key */
	key[slen] = 0;

	rem = (char*)ptr + slen + 1;		/* trim value's leading whitespace */
	remlen = ptrlen - slen - 1;
	while((remlen > 0) && (isspace(*rem)))
	{
		remlen--;
		rem++;
	}

	memcpy(val, rem, remlen);	/* store value, trim trailing ws */
	val[remlen] = 0;
	while((*val) && (isspace(val[strlen(val) - 1])))
	{
		val[strlen(val) - 1] = 0;
	}
	if(!*val)			/* skip blank value */
		goto out;

	if(!strcasecmp("X-Long-Polling", key))
	{
		hi->lp_path = val;	/* X-Mining-Extensions: longpoll */
		val = NULL;
	}

	if(!strcasecmp("X-Reject-Reason", key))
	{
		hi->reason = val;	/* X-Mining-Extensions: reject-reason */
		//applog(LOG_WARNING, "%s:%s", key, val);
		val = NULL;
	}

	if(!strcasecmp("X-Stratum", key))
	{
		hi->stratum_url = val;	/* steal memory reference */
		val = NULL;
	}

	if(!strcasecmp("X-Nonce-Range", key))
	{
		/* todo when available: X-Mining-Extensions: noncerange */
	}
out:
	free(key);
	free(val);
	return ptrlen;
}

#if LIBCURL_VERSION_NUM >= 0x070f06
static int sockopt_keepalive_cb(void *userdata, curl_socket_t fd,
								curlsocktype purpose)
{
	int keepalive = 1;
	int tcp_keepcnt = 3;
	int tcp_keepidle = 50;
	int tcp_keepintvl = 50;
#ifdef WIN32
	DWORD outputBytes;
#endif

#ifndef WIN32	
	if(unlikely(setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive,
		sizeof(keepalive))))
		return 1;
#ifdef __linux
	if(unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPCNT,
		&tcp_keepcnt, sizeof(tcp_keepcnt))))
		return 1;
	if(unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPIDLE,
		&tcp_keepidle, sizeof(tcp_keepidle))))
		return 1;
	if(unlikely(setsockopt(fd, SOL_TCP, TCP_KEEPINTVL,
		&tcp_keepintvl, sizeof(tcp_keepintvl))))
		return 1;
#endif /* __linux */
#ifdef __APPLE_CC__
	if(unlikely(setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE,
		&tcp_keepintvl, sizeof(tcp_keepintvl))))
		return 1;
#endif /* __APPLE_CC__ */
#else /* WIN32 */
	struct tcp_keepalive vals;
	vals.onoff = 1;
	vals.keepalivetime = tcp_keepidle * 1000;
	vals.keepaliveinterval = tcp_keepintvl * 1000;
	if(unlikely(WSAIoctl(fd, SIO_KEEPALIVE_VALS, &vals, sizeof(vals),
		NULL, 0, &outputBytes, NULL, NULL)))
		return 1;
#endif /* WIN32 */

	return 0;
}
#endif

json_t *json_rpc_call(CURL *curl, const char *url,
					  const char *userpass, const char *rpc_req,
					  bool longpoll_scan, bool longpoll, int *curl_err)
{
	json_t *val, *err_val, *res_val;
	CURLcode rc;
	struct data_buffer all_data = { 0 };
	struct upload_buffer upload_data;
	json_error_t err;
	struct curl_slist *headers = NULL;
	char* httpdata;
	char len_hdr[64], hashrate_hdr[64];
	long timeout = opt_timeout;
	struct header_info hi = { 0 };
	bool lp_scanning = longpoll_scan && !have_longpoll;

	/* it is assumed that 'curl' is freshly [re]initialized at this pt */

	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_err_str);
	if(opt_protocol)
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	rc = curl_easy_setopt(curl, CURLOPT_URL, url);
	if(rc != CURLE_OK)
	{
		if(strlen(curl_err_str)>0)
			applog(LOG_ERR, "CURLOPT_URL error: %s", curl_err_str);
		else
			applog(LOG_ERR, "CURLOPT_URL error: %s", curl_easy_strerror(rc));
		curl_easy_reset(curl);
		return NULL;
	}
	if(opt_cert)
		curl_easy_setopt(curl, CURLOPT_CAINFO, opt_cert);
	curl_easy_setopt(curl, CURLOPT_ENCODING, "");
	curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
	curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, all_data_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &all_data);
	curl_easy_setopt(curl, CURLOPT_READFUNCTION, upload_data_cb);
	curl_easy_setopt(curl, CURLOPT_READDATA, &upload_data);
#if LIBCURL_VERSION_NUM >= 0x071200
	curl_easy_setopt(curl, CURLOPT_SEEKFUNCTION, &seek_data_cb);
	curl_easy_setopt(curl, CURLOPT_SEEKDATA, &upload_data);
#endif
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
	curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, resp_hdr_cb);
	curl_easy_setopt(curl, CURLOPT_HEADERDATA, &hi);
	if(opt_proxy)
	{
		curl_easy_setopt(curl, CURLOPT_PROXY, opt_proxy);
		curl_easy_setopt(curl, CURLOPT_PROXYTYPE, opt_proxy_type);
	}
	if(userpass)
	{
		curl_easy_setopt(curl, CURLOPT_USERPWD, userpass);
		curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
	}
#if LIBCURL_VERSION_NUM >= 0x070f06
	if(longpoll)
		curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_keepalive_cb);
#endif
	curl_easy_setopt(curl, CURLOPT_POST, 1);

	if(opt_protocol)
		applog(LOG_DEBUG, "JSON protocol request:\n%s", rpc_req);

	upload_data.buf = rpc_req;
	upload_data.len = strlen(rpc_req);
	upload_data.pos = 0;
	sprintf(len_hdr, "Content-Length: %lu", (unsigned long)upload_data.len);
	sprintf(hashrate_hdr, "X-Mining-Hashrate: %llu", (unsigned long long) global_hashrate);

	headers = curl_slist_append(headers, "Content-Type: application/json");
	headers = curl_slist_append(headers, len_hdr);
	headers = curl_slist_append(headers, "User-Agent: " USER_AGENT);
	headers = curl_slist_append(headers, "X-Mining-Extensions: longpoll noncerange reject-reason");
	headers = curl_slist_append(headers, hashrate_hdr);
	headers = curl_slist_append(headers, "Accept:"); /* disable Accept hdr*/
	headers = curl_slist_append(headers, "Expect:"); /* disable Expect hdr*/

	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_err_str[0] = 0;
	rc = curl_easy_perform(curl);
	if(curl_err != NULL)
		*curl_err = rc;
	if(rc != CURLE_OK)
	{
		if(!(longpoll && rc == CURLE_OPERATION_TIMEDOUT))
		{
			if(strlen(curl_err_str)>0)
				applog(LOG_ERR, "HTTP request failed: %s", curl_err_str);
			else
				applog(LOG_ERR, "HTTP request failed: %s", curl_easy_strerror(rc));
			goto err_out;
		}
	}

	/* If X-Stratum was found, activate Stratum */
	if(want_stratum && hi.stratum_url &&
	   !strncasecmp(hi.stratum_url, "stratum+tcp://", 14) &&
	   !(opt_proxy && opt_proxy_type == CURLPROXY_HTTP))
	{
		have_stratum = true;
		tq_push(thr_info[stratum_thr_id].q, hi.stratum_url);
		hi.stratum_url = NULL;
	}

	/* If X-Long-Polling was found, activate long polling */
	if(lp_scanning && hi.lp_path && !have_stratum)
	{
		have_longpoll = true;
		tq_push(thr_info[longpoll_thr_id].q, hi.lp_path);
		hi.lp_path = NULL;
	}

	if(!all_data.buf || !all_data.len)
	{
		applog(LOG_ERR, "Empty data received in json_rpc_call.");
		goto err_out;
	}

	httpdata = (char*)all_data.buf;

	if(*httpdata != '{' && *httpdata != '[')
	{
		long errcode = 0;
		CURLcode c = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &errcode);
		if(c == CURLE_OK && errcode == 401)
		{
			applog(LOG_ERR, "You are not authorized, check your login and password.");
			goto err_out;
		}
	}

	val = JSON_LOADS(httpdata, &err);
	if(!val)
	{
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		if(opt_protocol)
			applog(LOG_DEBUG, "%s", httpdata);
		goto err_out;
	}

	if(opt_protocol)
	{
		char *s = json_dumps(val, JSON_INDENT(3));
		applog(LOG_DEBUG, "JSON protocol response:\n%s\n", s);
		free(s);
	}

	/* JSON-RPC valid response returns a non-null 'result',
	* and a null 'error'. */
	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if(!res_val || json_is_null(res_val) ||
	   (err_val && !json_is_null(err_val)))
	{
		char *s = NULL;

		if(err_val)
		{
			s = json_dumps(err_val, 0);
			json_t *msg = json_object_get(err_val, "message");
			json_t *err_code = json_object_get(err_val, "code");
			if(curl_err && json_integer_value(err_code))
				*curl_err = (int)json_integer_value(err_code);

			if(json_is_string(msg))
			{
				free(s);
				s = strdup(json_string_value(msg));
				if(have_longpoll && s && !strcmp(s, "method not getwork"))
				{
					json_decref(err_val);
					free(s);
					goto err_out;
				}
			}
			json_decref(err_val);
		}
		else
			s = strdup("(unknown reason)");

		if(!curl_err || opt_debug)
			applog(LOG_ERR, "JSON-RPC call failed: %s", s);

		free(s);

		goto err_out;
	}

	if(hi.reason)
		json_object_set_new(val, "reject-reason", json_string(hi.reason));

	databuf_free(&all_data);
	curl_slist_free_all(headers);
	curl_easy_reset(curl);
	return val;

err_out:
	free(hi.lp_path);
	free(hi.reason);
	free(hi.stratum_url);
	databuf_free(&all_data);
	curl_slist_free_all(headers);
	curl_easy_reset(curl);
	return NULL;
}

/**
* Unlike malloc, calloc set the memory to zero

void *aligned_calloc(int size)
{
	const int ALIGN = 64; // cache line
#ifdef _MSC_VER
	void* res = _aligned_malloc(size, ALIGN);
	if(res == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	memset(res, 0, size);
	return res;
#else
	void *mem = calloc(1, size + ALIGN + sizeof(uintptr_t));
	if(mem == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	void **ptr = (void**)((size_t)(((uintptr_t)(mem)) + ALIGN + sizeof(uintptr_t)) & ~(ALIGN - 1));
	ptr[-1] = mem;
	return ptr;
#endif
}

void aligned_free(void *ptr)
{
#ifdef _MSC_VER
	_aligned_free(ptr);
#else
	free(((void**)ptr)[-1]);
#endif
}
*/

void cbin2hex(char *out, const char *in, size_t len)
{
	if(out)
	{
		unsigned int i;
		for(i = 0; i < len; i++)
			sprintf(out + (i * 2), "%02x", (uint8_t)in[i]);
	}
}

char *bin2hex(const uchar *in, size_t len)
{
	char *s = (char*)malloc((len * 2) + 1);
	if(s == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}

	cbin2hex(s, (const char *)in, len);

	return s;
}

bool hex2bin(uchar *p, const char *hexstr, size_t len)
{
	char hex_byte[3];
	char *ep;

	hex_byte[2] = '\0';

	while(*hexstr && len)
	{
		if(!hexstr[1])
		{
			applog(LOG_ERR, "hex2bin str truncated");
			return false;
		}
		hex_byte[0] = hexstr[0];
		hex_byte[1] = hexstr[1];
		*p = (uchar)strtol(hex_byte, &ep, 16);
		if(*ep)
		{
			applog(LOG_ERR, "hex2bin failed on '%s'", hex_byte);
			return false;
		}
		p++;
		hexstr += 2;
		len--;
	}

	return (len == 0 && *hexstr == 0) ? true : false;
}

/* Subtract the `struct timeval' values X and Y,
storing the result in RESULT.
Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *x,
struct timeval *y)
{
	uint64_t start, end;

	end = x->tv_usec + 1000000 * x->tv_sec;
	start = y->tv_usec + 1000000 * y->tv_sec;
	if(start <= end)
	{
		uint64_t diff = end - start;
		result->tv_sec = (long)(diff / 1000000);
		result->tv_usec = diff % 1000000;
	}
	else
	{
		result->tv_sec = 0;
		result->tv_usec = 0;
	}

	/* Return 1 if result is negative. */
	return (start > end);
}

bool fulltest(const uint32_t *hash, const uint32_t *target)
{
	int i;
	bool rc = true;

	for(i = 7; i >= 0; i--)
	{
		if(hash[i] > target[i])
		{
			rc = false;
			break;
		}
		if(hash[i] < target[i])
		{
			rc = true;
			break;
		}
		if(hash[1] == target[1])
		{
			applog(LOG_NOTICE, "We found a close match!");
		}
	}

	if(!rc && opt_debug)
	{
		uint32_t hash_be[8], target_be[8];
		char *hash_str, *target_str;

		for(i = 0; i < 8; i++)
		{
			be32enc(hash_be + i, hash[7 - i]);
			be32enc(target_be + i, target[7 - i]);
		}
		hash_str = bin2hex((uchar *)hash_be, 32);
		target_str = bin2hex((uchar *)target_be, 32);

		applog(LOG_DEBUG, "DEBUG: %s\nHash:   %s\nTarget: %s",
			   rc ? "hash <= target"
			   : CL_YLW "hash > target (false positive)" CL_N,
			   hash_str,
			   target_str);

		free(hash_str);
		free(target_str);
	}

	return rc;
}

bool fulltest_sia(const uint64_t *hash, const uint64_t *target)
{
	int i;
	bool rc = true;

	for(i = 0; i < 4; i++)
	{
		if(swab64(hash[i]) > target[3 - i])
		{
			rc = false;
			break;
		}
		if(swab64(hash[i]) < target[3 - i])
		{
			rc = true;
			break;
		}
	}
	return rc;
}

void diff_to_target(uint32_t *target, double diff)
{
	uint64_t m;
	int k;

	for(k = 6; k > 0 && diff > 1.0; k--)
		diff /= 4294967296.0;
	m = (uint64_t)(4294901760.0 / diff);
	if(m == 0 && k == 6)
		memset(target, 0xff, 32);
	else
	{
		memset(target, 0, 32);
		target[k] = (uint32_t)m;
		target[k + 1] = (uint32_t)(m >> 32);
	}
}

#ifdef WIN32
#define socket_blocks() (WSAGetLastError() == WSAEWOULDBLOCK)
#else
#define socket_blocks() (errno == EAGAIN || errno == EWOULDBLOCK)
#endif

static bool send_line(curl_socket_t sock, char *s)
{
	ssize_t len, sent = 0;

	len = (ssize_t)strlen(s);
	s[len++] = '\n';

	while(len > 0)
	{
		struct timeval timeout;
		int n;
		fd_set wd;

		timeout.tv_sec = 3;
		timeout.tv_usec = 0;
		FD_ZERO(&wd);
		FD_SET(sock, &wd);
		if(select((int)sock + 1, NULL, &wd, NULL, &timeout) < 1)
			return false;
		n = send(sock, s + sent, (int)len, 0);
		if(n < 0)
		{
			if(!socket_blocks())
				return false;
			n = 0;
		}
		sent += n;
		len -= n;
	}

	return true;
}

bool stratum_send_line(struct stratum_ctx *sctx, char *s)
{
	bool ret = false;

	if(opt_protocol)
		applog(LOG_DEBUG, "> %s", s);

	pthread_mutex_lock(&sctx->sock_lock);
	ret = send_line(sctx->sock, s);
	pthread_mutex_unlock(&sctx->sock_lock);

	return ret;
}

static bool socket_full(curl_socket_t sock, int timeout)
{
	struct timeval tv;
	fd_set rd;

	FD_ZERO(&rd);
	FD_SET(sock, &rd);
	tv.tv_sec = timeout;
	tv.tv_usec = 0;
	if(select((int)sock + 1, &rd, NULL, NULL, &tv) > 0)
		return true;
	return false;
}

bool stratum_socket_full(struct stratum_ctx *sctx, int timeout)
{
	if(!sctx->sockbuf)
		return false;
	else
		return strlen(sctx->sockbuf) || socket_full(sctx->sock, timeout);
}

#define RBUFSIZE 2048
#define RECVSIZE (RBUFSIZE - 4)

static void stratum_buffer_append(struct stratum_ctx *sctx, const char *s)
{
	size_t old, snew;

	old = strlen(sctx->sockbuf);
	snew = old + strlen(s) + 1;
	if(snew >= sctx->sockbuf_size)
	{
		sctx->sockbuf_size = snew + (RBUFSIZE - (snew % RBUFSIZE));
		sctx->sockbuf = (char*)realloc(sctx->sockbuf, sctx->sockbuf_size);
		if(sctx->sockbuf == NULL)
		{
			applog(LOG_ERR, "Out of memory!");
			proper_exit(EXIT_FAILURE);
		}
	}
	strcpy(sctx->sockbuf + old, s);
}

char *stratum_recv_line(struct stratum_ctx *sctx)
{
	size_t len, buflen;
	char *tok, *sret = NULL;
	int timeout = opt_timeout;

	if(!sctx->sockbuf)
		return NULL;

	if(!strstr(sctx->sockbuf, "\n"))
	{
		bool ret = true;
		time_t rstart = time(NULL);
		if(!socket_full(sctx->sock, timeout))
		{
			if(!stop_mining)
				applog(LOG_ERR, "stratum_recv_line timed out");
			goto out;
		}
		do
		{
			char s[RBUFSIZE];
			ssize_t n;

			memset(s, 0, RBUFSIZE);
			n = recv(sctx->sock, s, RECVSIZE, 0);
			if(!n)
			{
				ret = false;
				break;
			}
			if(n < 0)
			{
				if(!socket_blocks() || !socket_full(sctx->sock, 10))
				{
					ret = false;
					break;
				}
			}
			else
				stratum_buffer_append(sctx, s);
		} while(time(NULL) - rstart < timeout && !strstr(sctx->sockbuf, "\n"));

		if(!ret)
		{
			if(opt_debug) applog(LOG_ERR, "stratum_recv_line failed");
			goto out;
		}
	}

	buflen = strlen(sctx->sockbuf);
	tok = strtok(sctx->sockbuf, "\n");
	if(!tok)
	{
		applog(LOG_ERR, "stratum_recv_line failed to parse a newline-terminated string");
		goto out;
	}
	sret = strdup(tok);
	len = strlen(sret);

	if(buflen > len + 1)
		memmove(sctx->sockbuf, sctx->sockbuf + len + 1, buflen - len + 1);
	else
		sctx->sockbuf[0] = '\0';

out:
	if(sret && opt_protocol)
		applog(LOG_DEBUG, "< %s", sret);
	return sret;
}

#if LIBCURL_VERSION_NUM >= 0x071101
static curl_socket_t opensocket_grab_cb(void *clientp, curlsocktype purpose,
struct curl_sockaddr *addr)
{
	curl_socket_t *sock = (curl_socket_t *)clientp;
	*sock = socket(addr->family, addr->socktype, addr->protocol);
	return *sock;
}
#endif

bool stratum_connect(struct stratum_ctx *sctx, const char *url)
{
	CURL *curl;
	CURLcode rc;

	pthread_mutex_lock(&sctx->sock_lock);
	if(sctx->curl)
		curl_easy_cleanup(sctx->curl);
	sctx->curl = curl_easy_init();
	if(!sctx->curl)
	{
		applog(LOG_ERR, "CURL initialization failed");
		pthread_mutex_unlock(&sctx->sock_lock);
		return false;
	}
	curl = sctx->curl;
	if(!sctx->sockbuf)
	{
		sctx->sockbuf = (char*)calloc(RBUFSIZE, 1);
		if(sctx->sockbuf == NULL)
		{
			applog(LOG_ERR, "Out of memory!");
			proper_exit(EXIT_FAILURE);
		}
		sctx->sockbuf_size = RBUFSIZE;
	}
	sctx->sockbuf[0] = '\0';
	pthread_mutex_unlock(&sctx->sock_lock);

	if(url != sctx->url)
	{
		free(sctx->url);
		sctx->url = strdup(url);
	}
	free(sctx->curl_url);
	sctx->curl_url = (char*)malloc(strlen(url));
	if(sctx->curl_url == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		pthread_mutex_unlock(&sctx->sock_lock);
		proper_exit(EXIT_FAILURE);
	}
	sprintf(sctx->curl_url, "http%s", strstr(url, "://"));

	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_err_str);
	if(opt_protocol)
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	rc = curl_easy_setopt(curl, CURLOPT_URL, sctx->curl_url);
	if(rc != CURLE_OK)
	{
		if(strlen(curl_err_str)>0)
			applog(LOG_ERR, "CURLOPT_URL error: %s", curl_err_str);
		else
			applog(LOG_ERR, "CURLOPT_URL error: %s", curl_easy_strerror(rc));
		curl_easy_cleanup(curl);
		sctx->curl = NULL;
		pthread_mutex_unlock(&sctx->sock_lock);
		return false;
	}
	curl_easy_setopt(curl, CURLOPT_FRESH_CONNECT, 1);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, opt_timeout);
	curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1);
	curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1);
	if(opt_proxy && opt_proxy_type != CURLPROXY_HTTP)
	{
		curl_easy_setopt(curl, CURLOPT_PROXY, opt_proxy);
		curl_easy_setopt(curl, CURLOPT_PROXYTYPE, opt_proxy_type);
	}
	else if(getenv("http_proxy"))
	{
		if(getenv("all_proxy"))
			curl_easy_setopt(curl, CURLOPT_PROXY, getenv("all_proxy"));
		else if(getenv("ALL_PROXY"))
			curl_easy_setopt(curl, CURLOPT_PROXY, getenv("ALL_PROXY"));
		else
			curl_easy_setopt(curl, CURLOPT_PROXY, "");
	}
#if LIBCURL_VERSION_NUM >= 0x070f06
	curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_keepalive_cb);
#endif
#if LIBCURL_VERSION_NUM >= 0x071101
	curl_easy_setopt(curl, CURLOPT_OPENSOCKETFUNCTION, opensocket_grab_cb);
	curl_easy_setopt(curl, CURLOPT_OPENSOCKETDATA, &sctx->sock);
#endif
	curl_easy_setopt(curl, CURLOPT_CONNECT_ONLY, 1);
	curl_err_str[0] = 0;
	rc = curl_easy_perform(curl);
	if(rc != CURLE_OK)
	{
		if(strlen(curl_err_str)>0)
			applog(LOG_ERR, "Stratum connect failed: %s", curl_err_str);
		else
			applog(LOG_ERR, "Stratum connect failed: %s", curl_easy_strerror(rc));
		curl_easy_cleanup(curl);
		sctx->curl = NULL;
		pthread_mutex_unlock(&sctx->sock_lock);
		return false;
	}

#if LIBCURL_VERSION_NUM < 0x071101
	/* CURLINFO_LASTSOCKET is broken on Win64; only use it as a last resort */
	curl_easy_getinfo(curl, CURLINFO_LASTSOCKET, (long *)&sctx->sock);
#endif
	pthread_mutex_unlock(&sctx->sock_lock);
	return true;
}

static void stratum_free_job(struct stratum_ctx *sctx)
{
	pthread_mutex_lock(&sctx->work_lock);
	if(sctx->job.job_id)
	{
		free(sctx->job.job_id);
	}
	if(sctx->job.merkle_count)
	{
		for(int i = 0; i < sctx->job.merkle_count; i++)
		{
			free(sctx->job.merkle[i]);
			sctx->job.merkle[i] = NULL;
		}
		free(sctx->job.merkle);
	}
	free(sctx->job.coinbase);
	// note: xnonce2 is not allocated
	memset(&(sctx->job.job_id), 0, sizeof(struct stratum_job));
	pthread_mutex_unlock(&sctx->work_lock);
}

void stratum_disconnect(struct stratum_ctx *sctx)
{
	pthread_mutex_lock(&sctx->sock_lock);
	if(sctx->curl)
	{
		sctx->disconnects++;
		curl_easy_cleanup(sctx->curl);
		sctx->curl = NULL;
		if(sctx->sockbuf)
			sctx->sockbuf[0] = '\0';
	}
	if(sctx->job.job_id)
	{
		stratum_free_job(sctx);
	}
	pthread_mutex_unlock(&sctx->sock_lock);
}

static const char *get_stratum_session_id(const json_t *val)
{
	json_t *arr_val;
	int i, n;

	arr_val = json_array_get(val, 0);
	if(!arr_val || !json_is_array(arr_val))
		return NULL;
	n = (int)json_array_size(arr_val);
	for(i = 0; i < n; i++)
	{
		const char *notify;
		json_t *arr = json_array_get(arr_val, i);
		if(!arr || !json_is_array(arr))
			break;
		notify = json_string_value(json_array_get(arr, 0));
		if(!notify)
			continue;
		if(!strcasecmp(notify, "mining.notify"))
			return json_string_value(json_array_get(arr, 1));
	}
	return NULL;
}

static bool stratum_parse_extranonce(struct stratum_ctx *sctx, const json_t *params, int pndx)
{
	const char* xnonce1;
	int xn2_size;

	xnonce1 = json_string_value(json_array_get(params, pndx));
	if(!xnonce1)
	{
		applog(LOG_ERR, "Failed to get extranonce1");
		goto out;
	}
	xn2_size = (int)json_integer_value(json_array_get(params, pndx + 1));
	if(!xn2_size)
	{
		applog(LOG_ERR, "Failed to get extranonce2_size");
		goto out;
	}
	if(xn2_size < 2 || xn2_size > 16)
	{
		applog(LOG_ERR, "invalid n2size in parse_extranonce: size=%d", xn2_size);
		goto out;
	}

	pthread_mutex_lock(&sctx->work_lock);
	if(sctx->xnonce1)
		free(sctx->xnonce1);
	sctx->xnonce1_size = strlen(xnonce1) / 2;
	sctx->xnonce1 = (uchar*)calloc(1, sctx->xnonce1_size);
	if(sctx->xnonce1 == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	hex2bin(sctx->xnonce1, xnonce1, sctx->xnonce1_size);
	sctx->xnonce2_size = xn2_size;
	pthread_mutex_unlock(&sctx->work_lock);

	if(pndx == 0 && opt_debug) /* pool dynamic change */
		applog(LOG_DEBUG, "Stratum set nonce %s with extranonce2 size=%d",
		xnonce1, xn2_size);

	return true;
out:
	return false;
}

bool stratum_subscribe(struct stratum_ctx *sctx)
{
	json_error_t err;
	json_t *val = NULL;
	json_t *res_val;
	json_t *err_val;
	bool ret = false, retry = false;
	char *sret = NULL;
	char *sid;

start:
	char *s = (char*)malloc(128 + (sctx->session_id ? strlen(sctx->session_id) : 0));
	if(s == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	if(retry)
		sprintf(s, "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}");
	else if(sctx->session_id)
		sprintf(s, "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"" USER_AGENT "\", \"%s\"]}", sctx->session_id);
	else
		sprintf(s, "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": [\"" USER_AGENT "\"]}");

	if(!stratum_send_line(sctx, s))
		goto out;

	if(!socket_full(sctx->sock, 10))
	{
		applog(LOG_ERR, "stratum_subscribe timed out");
		goto out;
	}

	sret = stratum_recv_line(sctx);
	if(!sret)
		goto out;

	val = JSON_LOADS(sret, &err);
	free(sret);
	if(!val)
	{
		applog(LOG_ERR, "Stratum subscribe: JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	if(json_integer_value(json_object_get(val, "id")) != 1)
	{
		applog(LOG_WARNING, "Stratum subscribe answer id is not correct!");
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if(!res_val || json_is_null(res_val) || (err_val && !json_is_null(err_val)))
	{
		if(retry)
		{
			free(s);
			if(err_val)
				s = json_dumps(err_val, JSON_INDENT(3));
			else
				s = strdup("(unknown reason)");
			applog(LOG_ERR, "Stratum subscribe failed: %s", s);
		}
		goto out;
	}

	// sid is param 1, extranonce params are 2 and 3
	if(!stratum_parse_extranonce(sctx, res_val, 1))
	{
		goto out;
	}

	ret = true;

	// session id (optional)
	sid = (char*)get_stratum_session_id(res_val);
	if(opt_debug && sid)
		applog(LOG_DEBUG, "Stratum session id: %s", sid);

	pthread_mutex_lock(&sctx->work_lock);
	if(sctx->session_id)
		free(sctx->session_id);
	sctx->session_id = sid ? strdup(sid) : NULL;
	sctx->next_diff = 1.0;
	pthread_mutex_unlock(&sctx->work_lock);

out:
	free(s);
	if(val)
		json_decref(val);

	if(!ret && sret && !retry)
	{
		retry = true;
		goto start;
	}

	return ret;
}

bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass, bool extranonce)
{
	json_t *val = NULL, *res_val, *err_val;
	char *s, *sret;
	json_error_t err;
	bool ret = false;

	s = (char*)malloc(80 + strlen(user) + strlen(pass));
	if(s == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	sprintf(s, "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"%s\", \"%s\"]}",
			user, pass);

	if(!stratum_send_line(sctx, s))
	{
		applog(LOG_ERR, "Error: couldn't send stratum authorization request");
		goto out;
	}

	while(1)
	{
		sret = stratum_recv_line(sctx);
		if(!sret)
		{
			applog(LOG_ERR, "Error: no answer to Stratum authorization request");
			goto out;
		}
		if(!stratum_handle_method(sctx, sret))
			break;
		free(sret);
	}

	val = JSON_LOADS(sret, &err);
	free(sret);
	if(!val)
	{
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}
	if(json_integer_value(json_object_get(val, "id")) != 2)
	{
		applog(LOG_WARNING, "Stratum authorize answer id is not correct!");
	}
	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if (!res_val || json_is_false(res_val) ||
	    (err_val && !json_is_null(err_val)))  {
		if (err_val && json_is_array(err_val)) {
			const char* reason = json_string_value(json_array_get(err_val, 1));
			applog(LOG_ERR, "Stratum authentication failed (%s)", reason);
		}
		else applog(LOG_ERR, "Stratum authentication failed");
		goto out;
	}
	sctx->tm_connected = time(NULL);
	ret = true;
	if(extranonce)
	{
		// subscribe to extranonce (optional)
		sprintf(s, "{\"id\": 3, \"method\": \"mining.extranonce.subscribe\", \"params\": []}");
		if(!stratum_send_line(sctx, s))
			goto out;
		// reduced timeout to handle pools ignoring this method without answer (like xpool.ca)
		if(!socket_full(sctx->sock, 10))
		{
			if(opt_debug)
				applog(LOG_DEBUG, "stratum extranonce subscribe timed out");
			goto out;
		}

		sret = stratum_recv_line(sctx);
		if(sret)
		{
			json_t *extra = JSON_LOADS(sret, &err);
			if(!extra)
			{
				applog(LOG_WARNING, "JSON decode failed(%d): %s", err.line, err.text);
			}
			else
			{
				if(json_integer_value(json_object_get(extra, "id")) != 3)
				{
					// we receive a standard method if extranonce is ignored
					if(!stratum_handle_method(sctx, sret))
						applog(LOG_WARNING, "Stratum extranonce answer id was not correct!");
				}
				else
				{
					res_val = json_object_get(extra, "result");
					if(opt_debug && (!res_val || json_is_false(res_val)))
						applog(LOG_DEBUG, "extranonce subscribe not supported");
				}
				json_decref(extra);
			}
			free(sret);
		}
	}

out:
	free(s);
	if(val)
		json_decref(val);

	return ret;
}

/**
* Extract block height     L H... here len=3, height=0x1333e8
* "...0000000000ffffffff2703e83313062f503253482f043d61105408"
*/
static uint32_t getblockheight(struct stratum_ctx *sctx)
{
	uint32_t height = 0;
	uint8_t hlen = 0, *p, *m;

	// find 0xffff tag
	p = (uint8_t*)sctx->job.coinbase + 32;
	m = p + 128;
	while(*p != 0xff && p < m) p++;
	while(*p == 0xff && p < m) p++;
	if(*(p - 1) == 0xff && *(p - 2) == 0xff)
	{
		p++; hlen = *p;
		p++; height = le16dec(p);
		p += 2;
		switch(hlen)
		{
		case 4:
			height += 0x10000UL * le16dec(p);
			break;
		case 3:
			height += 0x10000UL * (*p);
			break;
		}
	}
	return height;
}

static bool stratum_notify(struct stratum_ctx *sctx, json_t *params)
{
	const char *job_id, *prevhash, *coinb1, *coinb2, *version, *nbits, *nreward;
	char *stime;
	size_t coinb1_size, coinb2_size;
	bool clean, ret = false;
	int merkle_count, i;
	json_t *merkle_arr;
	uchar **merkle = NULL;
	int32_t ntime;

	job_id = json_string_value(json_array_get(params, 0));
	prevhash = json_string_value(json_array_get(params, 1));
	coinb1 = json_string_value(json_array_get(params, 2));
	coinb2 = json_string_value(json_array_get(params, 3));
	merkle_arr = json_array_get(params, 4);
	if(!merkle_arr || !json_is_array(merkle_arr))
		goto out;
	merkle_count = (int)json_array_size(merkle_arr);
	version = json_string_value(json_array_get(params, 5));
	nbits = json_string_value(json_array_get(params, 6));
	stime = (char *)json_string_value(json_array_get(params, 7));
	clean = json_is_true(json_array_get(params, 8));
	nreward = json_string_value(json_array_get(params, 9));

	if(!job_id || !prevhash || !coinb1 || !coinb2 || !version || !nbits || !stime ||
		 strlen(prevhash) != 64 || strlen(version) != 8 || strlen(nbits) != 8)
	{
		applog(LOG_ERR, "Stratum notify: invalid parameters");
		goto out;
	}
	if(strlen(stime) != 8)
	{
		applog(LOG_ERR, "Stratum notify: invalid time parameter");
		goto out;
	}

	/* store stratum server time diff */
	hex2bin((uchar *)&ntime, stime, 4);
	ntime = swab32(ntime) - (uint32_t)time(0);

	pthread_mutex_lock(&sctx->work_lock);

	if(ntime > sctx->srvtime_diff)
	{
		sctx->srvtime_diff = ntime;
		if(!opt_quiet && ntime > 20)
			applog(LOG_DEBUG, "stratum time is at least %d seconds in the future", ntime);
	}

	if(merkle_count)
	{
		merkle = (uchar**)malloc(merkle_count * sizeof(char *));
		if(merkle == NULL)
		{
			applog(LOG_ERR, "Out of memory!");
			proper_exit(EXIT_FAILURE);
		}
	}
	for(i = 0; i < merkle_count; i++)
	{
		const char *s = json_string_value(json_array_get(merkle_arr, i));
		if(!s || strlen(s) != 64)
		{
			while(i--)
				free(merkle[i]);
			free(merkle);
			applog(LOG_ERR, "Stratum notify: invalid Merkle branch");
			pthread_mutex_unlock(&sctx->work_lock);
			goto out;
		}
		merkle[i] = (uchar*)malloc(32);
		if(merkle[i] == NULL)
		{
			applog(LOG_ERR, "Out of memory!");
			proper_exit(EXIT_FAILURE);
		}
		hex2bin(merkle[i], s, 32);
	}

	coinb1_size = strlen(coinb1) / 2;
	coinb2_size = strlen(coinb2) / 2;
	sctx->job.coinbase_size = coinb1_size + sctx->xnonce1_size +
		sctx->xnonce2_size + coinb2_size;

	sctx->job.coinbase = (uchar*)realloc(sctx->job.coinbase, sctx->job.coinbase_size);
	if(sctx->job.coinbase == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	sctx->job.xnonce2 = sctx->job.coinbase + coinb1_size + sctx->xnonce1_size;
	hex2bin(sctx->job.coinbase, coinb1, coinb1_size);
	memcpy(sctx->job.coinbase + coinb1_size, sctx->xnonce1, sctx->xnonce1_size);

	if(!sctx->job.job_id || strcmp(sctx->job.job_id, job_id))
		memset(sctx->job.xnonce2, 0, sctx->xnonce2_size);
	hex2bin(sctx->job.xnonce2 + sctx->xnonce2_size, coinb2, coinb2_size);

	free(sctx->job.job_id);
	sctx->job.job_id = strdup(job_id);
	hex2bin(sctx->job.prevhash, prevhash, 32);

	sctx->job.height = getblockheight(sctx);
	if(!opt_quiet)
	{
		applog(LOG_BLUE, "Received new %s block header", algo_names[opt_algo]);
		applog(LOG_BLUE, "block height %d, %d transactions", sctx->job.height, merkle_count);
	}
	for(i = 0; i < sctx->job.merkle_count; i++)
		free(sctx->job.merkle[i]);
	free(sctx->job.merkle);
	sctx->job.merkle = merkle;
	sctx->job.merkle_count = merkle_count;

	hex2bin(sctx->job.version, version, 4);
	hex2bin(sctx->job.nbits, nbits, 4);
	hex2bin(sctx->job.ntime, stime, 4);
	if(nreward != NULL)
	{
		if(strlen(nreward) == 4)
			hex2bin(sctx->job.nreward, nreward, 2);
	}
	sctx->job.clean = clean;
	send_stale = !clean;

	sctx->job.diff = sctx->next_diff;

	pthread_mutex_unlock(&sctx->work_lock);

	ret = true;

out:
	return ret;
}

extern time_t g_work_time;
static bool stratum_set_difficulty(struct stratum_ctx *sctx, json_t *params)
{
	double diff;

	diff = json_number_value(json_array_get(params, 0));
	if(diff <= 0.0)
		return false;

	pthread_mutex_lock(&sctx->work_lock);
	sctx->next_diff = diff;
	pthread_mutex_unlock(&sctx->work_lock);

	/* store for api stats */
	if(diff != global_diff)
	{
		global_diff = diff;
		applog(LOG_WARNING, "Stratum difficulty set to %g", diff);
		g_work_time = 0;
	}

	return true;
}

static bool stratum_reconnect(struct stratum_ctx *sctx, json_t *params)
{
	json_t *port_val;
	const char *host;
	int port;

	host = json_string_value(json_array_get(params, 0));
	port_val = json_array_get(params, 1);
	if(json_is_string(port_val))
		port = atoi(json_string_value(port_val));
	else
		port = (int)json_integer_value(port_val);
	if(!host || !port)
		return false;

	free(sctx->url);
	sctx->url = (char*)malloc(32 + strlen(host));
	if(sctx->url == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	sprintf(sctx->url, "stratum+tcp://%s:%d", host, port);

	applog(LOG_NOTICE, "Server requested reconnection to %s", sctx->url);

	stratum_disconnect(sctx);

	return true;
}
static bool stratum_pong(struct stratum_ctx *sctx, json_t *id)
{
	char buf[64];
	bool ret = false;

	if(!id || json_is_null(id))
		return ret;

	sprintf(buf, "{\"id\":%d,\"result\":\"pong\",\"error\":null}",
					(int)json_integer_value(id));
	ret = stratum_send_line(sctx, buf);

	return ret;
}

static bool stratum_get_algo(struct stratum_ctx *sctx, json_t *id, json_t *params)
{
	char algo[64] = {0};
	char *s;
	json_t *val;
	bool ret = true;

	if(!id || json_is_null(id))
		return false;

	get_currentalgo(algo, sizeof(algo));

	val = json_object();
	json_object_set(val, "id", id);
	json_object_set_new(val, "error", json_null());
	json_object_set_new(val, "result", json_string(algo));

	s = json_dumps(val, 0);
	ret = stratum_send_line(sctx, s);
	json_decref(val);
	free(s);

	return ret;
}

#include "nvml.h"
extern char driver_version[32];
extern int cuda_arch[MAX_GPUS];

static bool json_object_set_error(json_t *result, int code, const char *msg)
{
	json_t *val = json_object();
	json_object_set_new(val, "code", json_integer(code));
	json_object_set_new(val, "message", json_string(msg));
	return json_object_set_new(result, "error", val) != -1;
}

/* allow to report algo/device perf to the pool for algo stats */
static bool stratum_benchdata(json_t *result, json_t *params, int thr_id)
{
	char algo[64] = { 0 };
	char vid[32], arch[8], driver[32];
	char *card;
	char os[8];
	uint32_t watts = 0, plimit = 0;
	int dev_id = device_map[thr_id];
	int cuda_ver = cuda_version();
	struct cgpu_info *cgpu = &thr_info[thr_id].gpu;
	json_t *val;

	if(!cgpu || !opt_stratum_stats) return false;

#ifdef _WIN64
	strcpy(os, "win64");
#else
#ifdef WIN32
	strcpy(os, "win32");
#else
#ifdef __APPLE__
	strcpy(os, "OSX");
#else
	strcpy(os, "linux");
#endif
#endif
#endif

#ifdef USE_WRAPNVML
	cgpu->has_monitoring = true;
	if(cgpu->monitor.gpu_power)
		cgpu->gpu_power = cgpu->monitor.gpu_power;
	else
		cgpu->gpu_power = gpu_power(cgpu); // mWatts
	watts = (cgpu->gpu_power >= 1000) ? cgpu->gpu_power / 1000 : 0; // ignore nvapi %
	plimit = device_plimit[dev_id] > 0 ? device_plimit[dev_id] : 0;
	gpu_info(cgpu); // vid/pid
#endif
	get_currentalgo(algo, sizeof(algo));

	card = device_name[dev_id];
	cgpu->khashes = stats_get_speed(thr_id, 0.0) / 1000.0;

	sprintf(vid, "%04hx:%04hx", cgpu->gpu_vid, cgpu->gpu_pid);
	sprintf(arch, "%d", (int)cgpu->gpu_arch);
	if(cuda_arch[dev_id] > 0 && cuda_arch[dev_id] != cgpu->gpu_arch)
	{
		// if binary was not compiled for the highest cuda arch, add it
		snprintf(arch, 8, "%d@%d", (int)cgpu->gpu_arch, cuda_arch[dev_id]);
	}
	snprintf(driver, 32, "CUDA %d.%d %s", cuda_ver / 1000, (cuda_ver % 1000) / 10, driver_version);
	driver[31] = '\0';

	val = json_object();
	json_object_set_new(val, "algo", json_string(algo));
	json_object_set_new(val, "type", json_string("gpu"));
	json_object_set_new(val, "device", json_string(card));
	json_object_set_new(val, "vendorid", json_string(vid));
	json_object_set_new(val, "arch", json_string(arch));
	json_object_set_new(val, "freq", json_integer(cgpu->gpu_clock / 1000));
	json_object_set_new(val, "memf", json_integer(cgpu->gpu_memclock / 1000));
	json_object_set_new(val, "curr_freq", json_integer(cgpu->monitor.gpu_clock));
	json_object_set_new(val, "curr_memf", json_integer(cgpu->monitor.gpu_memclock));
	json_object_set_new(val, "power", json_integer(watts));
	json_object_set_new(val, "plimit", json_integer(plimit));
	json_object_set_new(val, "khashes", json_real(cgpu->khashes));
	json_object_set_new(val, "intensity", json_real(cgpu->intensity));
	json_object_set_new(val, "throughput", json_integer(cgpu->throughput));
	json_object_set_new(val, "client", json_string(PACKAGE_NAME "/" PACKAGE_VERSION));
	json_object_set_new(val, "os", json_string(os));
	json_object_set_new(val, "driver", json_string(driver));

	json_object_set_new(result, "result", val);

	return true;
}

static bool stratum_get_stats(struct stratum_ctx *sctx, json_t *id, json_t *params)
{
	char *s;
	json_t *val;
	bool ret;

	if(!id || json_is_null(id))
		return false;

	val = json_object();
	json_object_set(val, "id", id);

	ret = stratum_benchdata(val, params, 0);

	if(!ret)
	{
		json_object_set_error(val, 1, "disabled"); //EPERM
	}
	else
	{
		json_object_set_new(val, "error", json_null());
	}

	s = json_dumps(val, 0);
	ret = stratum_send_line(sctx, s);
	json_decref(val);
	free(s);

	return ret;
}
static bool stratum_get_version(struct stratum_ctx *sctx, json_t *id)
{
	char *s;
	json_t *val;
	bool ret;

	if(!id || json_is_null(id))
		return false;

	val = json_object();
	json_object_set(val, "id", id);
	json_object_set_new(val, "result", json_string(USER_AGENT));
	json_object_set_new(val, "error", json_null());
	s = json_dumps(val, 0);
	ret = stratum_send_line(sctx, s);
	json_decref(val);
	free(s);

	return ret;
}

static bool stratum_show_message(struct stratum_ctx *sctx, json_t *id, json_t *params)
{
	char *s;
	json_t *val;
	bool ret;

	val = json_array_get(params, 0);
	if(val)
		applog(LOG_NOTICE, "MESSAGE FROM SERVER: %s", json_string_value(val));

	if(!id || json_is_null(id))
		return true;

	val = json_object();
	json_object_set(val, "id", id);
	json_object_set_new(val, "error", json_null());
	json_object_set_new(val, "result", json_true());
	s = json_dumps(val, 0);
	ret = stratum_send_line(sctx, s);
	json_decref(val);
	free(s);

	return ret;
}
static bool stratum_unknown_method(struct stratum_ctx *sctx, json_t *id)
{
	char *s;
	json_t *val;
	bool ret = false;

	if(!id || json_is_null(id))
		return ret;

	val = json_object();
	json_object_set(val, "id", id);
	json_object_set_new(val, "result", json_false());
	json_object_set_error(val, 38, "unknown method"); // ENOSYS

	s = json_dumps(val, 0);
	ret = stratum_send_line(sctx, s);
	json_decref(val);
	free(s);

	return ret;
}

bool stratum_handle_method(struct stratum_ctx *sctx, const char *s)
{
	json_t *val, *id, *params;
	json_error_t err;
	const char *method;
	bool ret = false;

	val = JSON_LOADS(s, &err);
	if(!val)
	{
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	method = json_string_value(json_object_get(val, "method"));
	if(!method)
		goto out;
	id = json_object_get(val, "id");
	params = json_object_get(val, "params");

	if(!strcasecmp(method, "mining.notify"))
	{
		ret = stratum_notify(sctx, params);
		goto out;
	}
	if(!strcasecmp(method, "mining.ping"))
	{ // cgminer 4.7.1+
		if(opt_debug) applog(LOG_DEBUG, "Pool ping");
		ret = stratum_pong(sctx, id);
		goto out;
	}
	if(!strcasecmp(method, "mining.set_difficulty"))
	{
		ret = stratum_set_difficulty(sctx, params);
		goto out;
	}
	if(!strcasecmp(method, "mining.set_extranonce"))
	{
		ret = stratum_parse_extranonce(sctx, params, 0);
		goto out;
	}
	if(!strcasecmp(method, "client.reconnect"))
	{
		ret = stratum_reconnect(sctx, params);
		goto out;
	}
	if(!strcasecmp(method, "client.get_algo"))
	{ // ccminer only yet!
		// will prevent wrong algo parameters on a pool, will be used as test on rejects
		if(!opt_quiet) applog(LOG_NOTICE, "Pool asked your algo parameter");
		ret = stratum_get_algo(sctx, id, params);
		goto out;
	}
	if(!strcasecmp(method, "client.get_stats"))
	{ // ccminer/yiimp only yet!
		// optional to fill device benchmarks
		ret = stratum_get_stats(sctx, id, params);
		goto out;
	}
	if(!strcasecmp(method, "client.get_version"))
	{
		ret = stratum_get_version(sctx, id);
		goto out;
	}
	if(!strcasecmp(method, "client.show_message"))
	{
		ret = stratum_show_message(sctx, id, params);
		goto out;
	}
	if(!ret)
	{
		// don't fail = disconnect stratum on unknown (and optional?) methods
		if(opt_debug) applog(LOG_WARNING, "unknown stratum method %s!", method);
		ret = stratum_unknown_method(sctx, id);
	}
out:
	if(val)
		json_decref(val);

	return ret;
}

struct thread_q *tq_new(void)
{
	struct thread_q *tq = (struct thread_q *)calloc(1, sizeof(struct thread_q));
	if(tq == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}

	INIT_LIST_HEAD(&tq->q);
	int err = pthread_mutex_init(&tq->mutex, NULL);
	if(err != 0)
	{
		applog(LOG_ERR, "pthread_mutex_init error %d", err);
		proper_exit(EXIT_FAILURE);
	}
	err = pthread_cond_init(&tq->cond, NULL);
	if(err != 0)
	{
		applog(LOG_ERR, "pthread_cond_init error %d", err);
		proper_exit(EXIT_FAILURE);
	}

	return tq;
}

void tq_free(struct thread_q *tq)
{
	struct tq_ent *ent, *iter;

	if(!tq)
		return;

	list_for_each_entry_safe(ent, iter, &tq->q, q_node, struct tq_ent, struct tq_ent)
	{
		list_del(&ent->q_node);
		free(ent);
	}

	pthread_cond_destroy(&tq->cond);
	pthread_mutex_destroy(&tq->mutex);

	memset(tq, 0, sizeof(*tq));	/* poison */
	free(tq);
	if(errno)
	{
		applog(LOG_ERR, "free() error in tq_free: %s", strerror(errno));
		errno = 0;
	}
}

static void tq_freezethaw(struct thread_q *tq, bool frozen)
{
	pthread_mutex_lock(&tq->mutex);

	tq->frozen = frozen;

	pthread_cond_signal(&tq->cond);
	pthread_mutex_unlock(&tq->mutex);
}

void tq_freeze(struct thread_q *tq)
{
	tq_freezethaw(tq, true);
}

void tq_thaw(struct thread_q *tq)
{
	tq_freezethaw(tq, false);
}

bool tq_push(struct thread_q *tq, void *data)
{
	bool rc = true;

	struct tq_ent* ent = (struct tq_ent *)calloc(1, sizeof(*ent));
	if(ent == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	else
	{
		ent->data = data;

		pthread_mutex_lock(&tq->mutex);

		INIT_LIST_HEAD(&ent->q_node);

		if (!tq->frozen)
		{
			list_add_tail(&ent->q_node, &tq->q);
		}
		else
		{
			free(ent);
			if (errno)
			{
				applog(LOG_ERR, "free() error in tq_push: %s", strerror(errno));
				errno = 0;
			}
			rc = false;
		}

		pthread_cond_signal(&tq->cond);
		pthread_mutex_unlock(&tq->mutex);
	}
	return rc;
}

void *tq_pop(struct thread_q *tq, const struct timespec *abstime)
{
	pthread_mutex_lock(&tq->mutex);

	if (list_empty(&tq->q))
	{
		int rc;
		if (abstime)
			rc = pthread_cond_timedwait(&tq->cond, &tq->mutex, abstime);
		else
			rc = pthread_cond_wait(&tq->cond, &tq->mutex);
		if (rc)
		{
			pthread_mutex_unlock(&tq->mutex);
			return NULL;
		}
		if (list_empty(&tq->q))
		{
			pthread_mutex_unlock(&tq->mutex);
			return NULL;
		}
	}

	struct tq_ent* ent = list_entry(tq->q.next, struct tq_ent, q_node);
	void* rval = ent->data;

	list_del(&ent->q_node);
	free(ent);
	if(errno)
	{
		applog(LOG_ERR, "free() error in tq_pop: %s", strerror(errno));
		errno = 0;
	}

	pthread_mutex_unlock(&tq->mutex);
	return rval;
}

/**
* @param buf char[9] mini
* @param time_t timer to convert
*/
size_t time2str(char* buf, time_t timer)
{
	struct tm* tm_info;
	tm_info = localtime(&timer);
	return strftime(buf, 19, "%H:%M:%S", tm_info);
}

/**
* Alloc and returns time string (to be freed)
* @param time_t timer to convert
*/
char* atime2str(time_t timer)
{
	char* buf = (char*)malloc(16);
	if(buf == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	memset(buf, 0, 16);
	time2str(buf, timer);
	return buf;
}

/* sprintf can be used in applog */
static char* format_hash(char* buf, uchar *hash)
{
	int len = 0;
	for(int i = 0; i < 32; i += 4)
	{
		len += sprintf(buf + len, "%02x%02x%02x%02x ",
					   hash[i], hash[i + 1], hash[i + 2], hash[i + 3]);
	}
	return buf;
}

/* to debug diff in data */
extern void applog_compare_hash(uchar *hash, uchar *hash2)
{
	char s[256] = "";
	int len = 0;
	for(int i = 0; i < 32; i += 4)
	{
		const char *color = memcmp(hash + i, hash2 + i, 4) ? CL_WHT : CL_GRY;
		len += sprintf(s + len, "%s%02x%02x%02x%02x " CL_GRY, color,
					   hash[i], hash[i + 1], hash[i + 2], hash[i + 3]);
		s[len] = '\0';
	}
	applog(LOG_DEBUG, "%s", s);
}

extern void applog_hash(uchar *hash)
{
	char s[128] = { '\0' };
	applog(LOG_DEBUG, "%s", format_hash(s, hash));
}

#define printpfx(n,h) \
	printf("%s%12s%s: %s\n", CL_BLU, n, CL_N, format_hash(s, h))

void do_gpu_tests(void)
{
#ifdef _DEBUG
	unsigned long done;
	char s[128] = { '\0' };
	uchar buf[128];
	uint32_t tgt[8] = { 0 };

	opt_tracegpu = true;
	work_restart = (struct work_restart*) malloc(sizeof(struct work_restart));
	work_restart[0].restart = 1;
	tgt[7] = 0xffff;

	memset(buf, 0, sizeof buf);
	scanhash_x11(0, (uint32_t*)buf, tgt, 1, &done);

	//memset(buf, 0, sizeof buf);
	// buf[0] = 1; buf[64] = 2; // for endian tests
	scanhash_blake256(0, (uint32_t*)buf, tgt, 1, &done, 14);

	memset(buf, 0, sizeof buf);

	free(work_restart);
	work_restart = NULL;
	opt_tracegpu = false;
#endif
}

void print_hash_tests(void)
{
	char s[128] = { '\0' };
	uchar buf[128], hash[128];
	memset(buf, 0, sizeof buf);
	// buf[0] = 1; buf[64] = 2; // for endian tests

	printf(CL_WHT "CPU HASH ON EMPTY BUFFER RESULTS:" CL_N "\n");

 /*
	memset(hash, 0, sizeof hash);
	blake256hash(&hash[0], &buf[0], 8);
	printpfx("blakecoin", hash);

	memset(hash, 0, sizeof hash);
	blake256hash(&hash[0], &buf[0], 14);
	printpfx("blake", hash);

	memset(hash, 0, sizeof hash);
	deephash(&hash[0], &buf[0]);
	printpfx("deep", hash);

	memset(hash, 0, sizeof hash);
	fresh_hash(&hash[0], &buf[0]);
	printpfx("fresh", hash);

	memset(hash, 0, sizeof hash);
	fugue256_hash(&hash[0], &buf[0], 32);
	printpfx("fugue256", hash);

	memset(hash, 0, sizeof hash);
	groestlhash(&hash[0], &buf[0]);
	printpfx("groestl", hash);

	memset(hash, 0, sizeof hash);
	jackpothash(&hash[0], &buf[0]);
	printpfx("jackpot", hash);

	memset(hash, 0, sizeof hash);
	keccak256_hash(&hash[0], &buf[0]);
	printpfx("keccak", hash);

	memset(hash, 0, sizeof hash);
	doomhash(&hash[0], &buf[0]);
	printpfx("luffa", hash);

	memset(hash, 0, sizeof hash);
	myriadhash(&hash[0], &buf[0]);
	printpfx("myriad", hash);

	memset(hash, 0, sizeof hash);
	nist5hash(&hash[0], &buf[0]);
	printpfx("nist5", hash);

	memset(hash, 0, sizeof hash);
	pentablakehash(&hash[0], &buf[0]);
	printpfx("pentablake", hash);

	memset(hash, 0, sizeof hash);
	quarkhash(&hash[0], &buf[0]);
	printpfx("quark", hash);

	memset(hash, 0, sizeof hash);
	qubithash(&hash[0], &buf[0]);
	printpfx("qubit", hash);

	skeincoinhash(&hash[0], &buf[0]);
	printpfx("skein", hash);

	memset(hash, 0, sizeof hash);
	s3hash(&hash[0], &buf[0]);
	printpfx("S3", hash);

	memset(hash, 0, sizeof hash);
	wcoinhash(&hash[0], &buf[0]);
	printpfx("whirl", hash);

	memset(hash, 0, sizeof hash);
	x11hash(&hash[0], &buf[0]);
	printpfx("X11", hash);

	memset(hash, 0, sizeof hash);
	x13hash(&hash[0], &buf[0]);
	printpfx("X13", hash);

	memset(hash, 0, sizeof hash);
	x14hash(&hash[0], &buf[0]);
	printpfx("X14", hash);

	memset(hash, 0, sizeof hash);
	x15hash(&hash[0], &buf[0]);
	printpfx("X15", hash);

	memset(hash, 0, sizeof hash);
	x17hash(&hash[0], &buf[0]);
	printpfx("X17", hash);
	*/

	printf("\n");

	do_gpu_tests();
}

void bin2hex(char *s, const unsigned char *p, size_t len)
{
	for(size_t i = 0; i < len; i++)
		sprintf(s + (i * 2), "%02x", (unsigned int)p[i]);
}

char *abin2hex(const unsigned char *p, size_t len)
{
	char *s = (char*)malloc((len * 2) + 1);
	if(s == NULL)
	{
		applog(LOG_ERR, "Out of memory!");
		proper_exit(EXIT_FAILURE);
	}
	bin2hex(s, p, len);
	return s;
}
void applog_hex(void *data, int len)
{
	char* hex = abin2hex((uchar*)data, len);
	applog(LOG_INFO, "%s", hex);
	free(hex);
}
