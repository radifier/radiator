#!/bin/bash

# Arch Linux build

make distclean || echo clean

rm -f Makefile.in
rm -f config.status

./autogen.sh
./configure \
    CPPFLAGS='-I/usr/include/openssl-1.0' \
    LDFLAGS='-L/usr/lib/openssl-1.0' \
    CUDA_CFLAGS='--shared --compiler-options "-fPIC"' \
    --prefix=/usr \
    --sysconfdir=/etc \
    --libdir=/usr/lib \
    --with-cuda=/opt/cuda
make
