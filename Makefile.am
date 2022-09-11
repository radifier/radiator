# allow to use Host cuda functions in C/C++
DEF_INCLUDES = @CUDA_INCLUDES@

if WANT_JANSSON
JANSSON_INCLUDES = -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES =
endif

EXTRA_DIST = autogen.sh README.txt LICENSE.txt \
			  cudaminer.sln cudaminer.vcxproj cudaminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES = elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  crc32.c \
			  ccminer.cpp util.cpp \
			  api.cpp hashlog.cpp stats.cpp sysinfos.cpp cuda.cpp \
			  nvml.cpp nvml.h nvsettings.cpp \
			  cuda_helper.h cuda_vector.h \
			  sph/sha2big.c sph/sha256_Y.h sph/sha256_Y.c sph/sph_sha2.c sph/sha2.c \
			  cuda_checkhash.cu \
			  novo.cpp cuda_novo.cu \
			  rad.cpp cuda_rad.cu

# scrypt
# ccminer_SOURCES += scrypt.cpp scrypt-jane.cpp \
#     scrypt/blake.cu scrypt/keccak.cu scrypt/sha256.cu \
#     scrypt/salsa_kernel.cu scrypt/test_kernel.cu \
#     scrypt/fermi_kernel.cu scrypt/kepler_kernel.cu \
#     scrypt/nv_kernel.cu scrypt/nv_kernel2.cu scrypt/titan_kernel.cu

			  
if HAVE_NVML
nvml_defs = -DUSE_WRAPNVML
nvml_libs = -ldl
endif

ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ $(nvml_libs)
ccminer_CPPFLAGS = @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(CPPFLAGS) $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) $(DEF_INCLUDES) $(nvml_defs) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME

if HAVE_OSX
ccminer_CPPFLAGS += -I/usr/local/llvm/lib/clang/4.0.0/include
ccminer_LDFLAGS += -L/usr/local/llvm/lib
ccminer_LDADD += -lomp
endif

nvcc_ARCH  = -gencode=arch=compute_61,code=sm_61
nvcc_ARCH += -gencode=arch=compute_60,code=sm_60
nvcc_ARCH += -gencode=arch=compute_52,code=sm_52
nvcc_ARCH += -gencode=arch=compute_70,code=sm_70
nvcc_ARCH += -gencode=arch=compute_75,code=sm_75
nvcc_ARCH += -gencode=arch=compute_80,code=sm_80
nvcc_ARCH += -gencode=arch=compute_86,code=sm_86

nvcc_FLAGS = $(nvcc_ARCH) @CUDA_INCLUDES@ -I. @CUDA_CFLAGS@
nvcc_FLAGS += $(JANSSON_INCLUDES)

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) $(nvcc_FLAGS) --maxrregcount=128 -o $@ -c $<

