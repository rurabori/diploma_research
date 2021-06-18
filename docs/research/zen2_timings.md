
### Cache
#### L0 Op Cache: 
+ 4,096 Ops
+ 8-way set associative
+ 64 sets, 8 Op line size
+ Parity protected

#### L1I Cache:
+ 32 KiB, 8-way set associative
+ 64 sets, 64 B line size
+ Shared by the two threads, per core
+ Parity protected

#### L1D Cache:
+ 32 KiB, 8-way set associative
+ 64 sets, 64 B line size
+ Write-back policy
+ 4-5 cycles latency for Int
+ 7-8 cycles latency for FP
+ ECC

#### L2 Cache:
+ 512 KiB, 8-way set associative
+ 1,024 sets, 64 B line size
+ Write-back policy
+ Inclusive of L1
+ ≥ 12 cycles latency
+ ECC

#### L3 Cache:
+ Matisse, Castle Peak, Rome: 16 MiB/CCX, shared across all cores
+ Renoir: 4 MiB/CCX, shared across all cores
+ 16-way set associative
+ 16,384 sets, 64 B line size
+ Write-back policy, Victim cache
+ 39 cycles average latency
+ ECC
+ QoS Monitoring and Enforcement

### RAM

+ Corsair 32GB KIT DDR4 3200MHz CL16 Vengeance RGB PRO Series
+ CL16-18-18-36

### Example run results

```
Sequential SpMV took: 1715000ns
omega = 4, sigma = 16. #partition = 26244
CSR->CSR5 malloc time = 1.121000 ms
CSR->CSR5 tile_ptr time = 0.891000 ms
CSR->CSR5 tile_desc time = 1.078000 ms
CSR->CSR5 transpose time = 2.461000 ms
CSR5 conversion took: 6587600ns
CSR5 SpMV took: 1150500ns
```

### calculations

+ L1   -> ~ 3ns
+ L2   -> ~ 6ns
+ L3   -> ~ 22ns
+ DRAM -> ~ 35ns ? (RAM itself 5ns + all misses ~30ns)

#### total data needed:

+ non zero: 1679599
+ matrix dims: 20414x20414

total data:

+ 1679599 * (8 + 4) = 20155188  (values + column indices)
+ 20415   * 4       = 81660     (row starts)
+ 20414   * 8       = 163312    (right hand side)
+ total               20400160

+ transfers of 4KiB -> ~ 4981 transfers worst case
+ 4981 * 35ns = 174335ns spent on memory transfers ? 

(assuming RHS stays in cache)