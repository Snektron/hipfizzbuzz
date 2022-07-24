# hipfizzbuzz

## Compiling

```
$ hipcc -O3 -DNDEBUG -march=native -x hip --offload-arch=gfx908 -std=c++20 -o hipfizzbuzz hipfizzbuzz.cpp
```

Some configurations can be applies while compiling, by setting these variables using `-DVARNAME=value`:

`HIPFIZZBUZZ_VMSPLICE` (0/1) (default: 1)
* If 1, use vmsplice() instead of write() to print the output. If the output is not consumed at sufficient rate, it might be corrupted. Disable hwen piping the output to slower programs.

`HIPFIZZBUZZ_PRINT_SYNC` (0/1) (default: 0)
* Print a buffer directly after computing it instead of swapping buffers. Useful for debugging.

`HIPFIZZBUZZ_OVERRIDE_PIPESZ` (0/1) (default: 1)
* Override the pipe size. When this is enabled, some programs cannot consume the pipe correctly. Useful to disable for debugging.

`HIPFIZZBUZZ_COPY_MEM` (0/1) (default: 0)
* Manually copy vram to main memory before printing it, instead of directly printing automatically migrating data.

`HIPFIZZBUZZ_DEBUG` (0/1) (default: 0)
* Enable "debug mode": This enables HIPFIZZBUZZ_PRINT_SYNC and HIPFIZZBUZZ_COPY_MEM, disables HIPFIZZBUZZ_VMSPLICE and HIPFIZZBUZZ_OVERRIDE_PIPESZ.
