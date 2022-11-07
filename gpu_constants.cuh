#pragma once

#define WARP_SIZE       32 
#define MIN_OCCUPANCY   512 // threads per block
#define MAX_BLOCK_SIZE  1024 
#define CONST_MEM_SIZE  65536

#define WARPS_COUNT(x) ((x + (WARP_SIZE - 1)) / WARP_SIZE)


