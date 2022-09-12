# Radiator

Fork of KlausT ccminer with Radiant and Novo support.

Compile:
```
./build.sh
```

Mine Radiant:
```
./ccminer -a rad -o stratum+tcp://radpool.net:3052 -u ADDRESS.WORKER -p PASSWORD
```

Mine Novo:
```
./ccminer -a novo -o stratum+tcp://m.novopool.net:3333 -u ADDRESS.WORKER -p PASSWORD
```

License: GPLv3.  See LICENSE.txt for details.
