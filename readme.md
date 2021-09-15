# Maeve
## Xilinx Open Hardware 2020

###### Directory organization
The folder src contains the source code of the entire project.
In order to build the bitstream run the command:
```
make all TARGET=hw
```
The folder HW_build contains the bitstream ready to go and the host executable.
In order to run it, launch the command:
```
./host source_kmp.xclbin
```
