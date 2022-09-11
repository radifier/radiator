#!/bin/bash
version=$1
docker build . -t radiator
docker run -it -v $(pwd):/root/radiator radiator /root/radiator/build.sh
mkdir -p build/linux/radiator
cp ccminer build/linux/radiator
cp hive/* build/linux/radiator
cp LICENSE.txt README.md build/linux/radiator
tar -C build/linux -czvf build/radiator-$version-linux.tar.gz radiator
