
ccMiner release 8.01(KlausT-mod) (August 11th, 2016)
---------------------------------------------------------------

***************************************************************
If you find this tool useful and like to support its continued 
          development, then consider a donation.

tpruvot@github:
  BTC donation address: 1AJdfCpLWPNoAMDfHF1wD5y8VgKSSTHxPo
  DRK  : XeVrkPrWB7pDbdFLfKhF1Z3xpqhsx6wkH3
  NEOS : NaEcVrdzoCWHUYXb7X8QoafoKS9UV69Yk4
  XST  : S9TqZucWgT6ajZLDBxQnHUtmkotCEHn9z9

sp-hash@github:
  BTC: 1CTiNJyoUmbdMRACtteRWXhGqtSETYd6Vd
  DRK: XdgfWywdxABwMdrGUd2xseb6CYy1UKi9jX
  
DJM34:
  BTC donation address: 1NENYmxwZGHsKFmyjTc5WferTn5VTFb7Ze

KlausT @github:
  BTC 1H2BHSyuwLP9vqt2p3bK9G3mDJsAi7qChw
  DRK XcM9FXrvZS275pGyGmfJmS98tHPZ1rjErM
  
cbuchner v1.2:
  LTC donation address: LKS1WDKGED647msBQfLBHV3Ls8sveGncnm
  BTC donation address: 16hJF5mceSojnTD3ZTUDqdRhDyPJzoRakM

***************************************************************

>>> Introduction <<<

This is a CUDA accelerated mining application which handle :

Bitcoin
HeavyCoin & MjollnirCoin
FugueCoin
GroestlCoin & Myriad-Groestl
JackpotCoin
QuarkCoin family & AnimeCoin
TalkCoin
DarkCoin and other X11 coins
NEOS blake (256 14-rounds)
BlakeCoin (256 8-rounds)
Deep, Doom and Qubit
Keccak (Maxcoin)
Pentablake (Blake 512 x5)
S3 (OneCoin)
Skein (Skein + SHA)
Lyra2RE (new VertCoin algo)
Neoscrypt

where some of these coins have a VERY NOTABLE nVidia advantage
over competing AMD (OpenCL Only) implementations.

We did not take a big effort on improving usability, so please set
your parameters carefuly.

THIS PROGRAMM IS PROVIDED "AS-IS", USE IT AT YOUR OWN RISK!

If you're interessted and read the source-code, please excuse
that the most of our comments are in german.

>>> Command Line Interface <<<

This code is based on the pooler cpuminer 2.3.2 release and inherits
its command line interface and options.

  -a, --algo=ALGO       specify the algorithm to use
                          anime       use to mine Animecoin
						  bitcoin     use to mine Bitcoin
                          blake       use to mine NEOS (Blake 256)
                          blakecoin   use to mine Old Blake 256
                          deep        use to mine Deepcoin
                          dmd-gr      use to mine Diamond-Groestl
                          fresh       use to mine Freshcoin
                          fugue256    use to mine Fuguecoin
                          groestl     use to mine Groestlcoin
                          heavy       use to mine Heavycoin
                          jackpot     use to mine Jackpotcoin
                          keccak      use to mine Maxcoin
                          luffa       use to mine Doomcoin
                          mjollnir    use to mine Mjollnircoin
                          myr-gr      use to mine Myriad-Groestl
                          neoscrypt   use to mine FeatherCoin
                          nist5       use to mine TalkCoin
                          penta       use to mine Joincoin / Pentablake
                          quark       use to mine Quarkcoin
                          qubit       use to mine Qubit Algo
                          s3          use to mine 1coin
                          sia         use to mine Siacoin (at siamining.com pool)
                          skein       use to mine Skeincoin
                          whirl       use to mine Whirlcoin
                          x11         use to mine DarkCoin
                          x14         use to mine X14Coin
                          x15         use to mine Halcyon
                          x17         use to mine X17
						  vanilla     use to mine Vanillacoin
                          lyra2v2     use to mine Vertcoin

  -d, --devices         gives a comma separated list of CUDA device IDs
                        to operate on. Device IDs start counting from 0!
                        Alternatively give string names of your card like
                        gtx780ti or gt640#2 (matching 2nd gt640 in the PC).

  -i, --intensity       GPU threads per call 8-31 (default: 0=auto)
                        Decimals are allowed for fine tuning
  -f, --diff            Divide difficulty by this factor (std is 1)
  -v, --vote            Heavycoin block vote (default: 512)
  -o, --url=URL         URL of mining server
  -O, --userpass=U:P    username:password pair for mining server
  -u, --user=USERNAME   username for mining server
  -p, --pass=PASSWORD   password for mining server
      --cert=FILE       certificate for mining server using SSL
  -x, --proxy=[PROTOCOL://]HOST[:PORT]  connect through a proxy
  -t, --threads=N       number of miner threads (default: number of nVidia GPUs in your system)
  -g                    number of mining threads per GPU (default: 1)
  -r, --retries=N       number of times to retry if a network call fails
                          (default: retry indefinitely)
  -R, --retry-pause=N   time to pause between retries, in seconds (default: 15)
  -T, --timeout=N       network timeout, in seconds (default: 270)
  -s, --scantime=N      upper bound on time spent scanning current work when
                        long polling is unavailable, in seconds (default: 5)
  -N, --statsavg        number of samples used to display hashrate (default: 30)
      --no-gbt          disable getblocktemplate support (height check in solo)
      --no-longpoll     disable X-Long-Polling support
      --no-stratum      disable X-Stratum support
  -q, --quiet           disable per-thread hashmeter output
  -D, --debug           enable debug output
  -P, --protocol-dump   verbose dump of protocol-level activities
  -b, --api-bind        IP/Port for the miner API (default: 127.0.0.1:4068)
      --benchmark       run in offline benchmark mode
      --cputest         debug hashes from cpu algorithms
      --cpu-affinity    set process affinity to specific cpu core(s) mask
      --cpu-priority    set process priority (default: 0 idle, 2 normal to 5 highest)
  -c, --config=FILE     load a JSON-format configuration file
      --no-color        disable colored console output
  -V, --version         display version information and exit
  -h, --help            display this help text and exit


>>> Examples <<<


Example for Heavycoin Mining on heavycoinpool.com with a single gpu in your system
    ccminer -t 1 -a heavy -o stratum+tcp://stratum01.heavycoinpool.com:5333 -u <<username.worker>> -p <<workerpassword>> -v 8


Example for Heavycoin Mining on hvc.1gh.com with a dual gpu in your system
    ccminer -t 2 -a heavy -o stratum+tcp://hvcpool.1gh.com:5333/ -u <<WALLET>> -p x -v 8


Example for Fuguecoin solo-mining with 4 gpu's in your system and a Fuguecoin-wallet running on localhost
    ccminer -q -s 1 -t 4 -a fugue256 -o http://localhost:9089/ -u <<myusername>> -p <<mypassword>>


Example for Fuguecoin pool mining on dwarfpool.com with all your GPUs
    ccminer -q -a fugue256 -o stratum+tcp://erebor.dwarfpool.com:3340/ -u YOURWALLETADDRESS.1 -p YOUREMAILADDRESS


Example for Groestlcoin solo mining
    ccminer -q -s 1 -a groestl -o http://127.0.0.1:1441/ -u USERNAME -p PASSWORD


For solo-mining you typically use -o http://127.0.0.1:xxxx where xxxx represents
the rpcport number specified in your wallet's .conf file and you have to pass the same username
and password with -O (or -u -p) as specified in the wallet config.

The wallet must also be started with the -server option and/or with the server=1 flag in the .conf file


>>> API and Monitoring <<<

With the -b parameter you can open your ccminer to your network, use -b 0.0.0.0:4068 if required.
On windows, setting 0.0.0.0 will ask firewall permissions on the first launch. Its normal.

Default API feature is only enabled for localhost queries by default, on port 4068.

You can test this api on linux with "telnet <miner-ip> 4068" and type "help" to list the commands.
Default api format is delimited text. If required a php json wrapper is present in api/ folder.

I plan to add a json format later, if requests are formatted in json too..


>>> Additional Notes <<<

This code should be running on nVidia GPUs ranging from compute capability
3.0 up to compute capability 5.2. Support for Compute 2.0 has been dropped
so we can more efficiently implement new algorithms using the latest hardware
features.

>>> RELEASE HISTORY <<<

2015-02-01 Release 1.0, forked from tpruvot and sp-hash
2015-02-03 v1.01: bug fix for cards with compute capability 3.0 (untested)
2015-02-09 v1.02: various bug fixes and optimizations
2015-03-08 v2.00: added whirlpoolx algo (Vanillacoin), also various optimizations and bug fixes
2015-03-30 v3.00: added skein (for Myriadcoin for example)
2015-05-06 v4.00: added Neoscrypt
2015-05-15 v4.01: fixed crash after ctrl-c (Windows), fixed -g option
2015-07-06 v5.00: -g option removed, some bug fixes and optimizations
2015-07-08 v5.01: lyra2 optimization
2015-08-22 v6.00: remove Lyra2RE, add Lyra2REv2, remove Animecoin, remove yesscrypt
2016-05-03 v6.01: various bug fixes and optimizations
2016-05-12 v6.02: faster x17 and quark
2016-05-16 v7.00: added Vanillacoin, optimized blake and blakecoin,
                  added stratum methods used by yiimp.ccminer.org
2016-05-16 v7.01: stratum.get_stats bug fix
2016-06-02 v7.02: fix default intensity for Nist5
                  fix power usage statistics
2016-06-11 v7.03: faster lyra2v2
2016-06-18 v7.04: Neoscrypt optimization
                  Bug Fixes 
2016-08-11 v8.00: added Siacoin

>>> AUTHORS <<<

Notable contributors to this application are:

Christian Buchner, Christian H. (Germany): Initial CUDA implementation

djm34, tsiv, sp and KlausT for cuda algos implementation and optimisation

Tanguy Pruvot : 750Ti tuning, blake, colors, general code cleanup/opts
                API monitoring, linux Config/Makefile and vstudio stuff...

and also many thanks to anyone else who contributed to the original
cpuminer application (Jeff Garzik, pooler), it's original HVC-fork
and the HVC-fork available at hvc.1gh.com

Source code is included to satisfy GNU GPL V3 requirements.


With kind regards,

   Christian Buchner ( Christian.Buchner@gmail.com )
   Christian H. ( Chris84 )
   Tanguy Pruvot ( tpruvot@github )
