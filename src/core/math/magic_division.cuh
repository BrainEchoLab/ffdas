#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

struct magic_pair {
  uint32_t m;
  int32_t  s;
};


// Call only for d >= 2.  If d == 1, just know q=x/1 == x.
static magic_pair compute_magic_pair_unsigned(uint32_t d) {
    const uint64_t two31 = 0x80000000ull;  // = 2^31
    // nc = (2^31 - 1) - ((2^31 - 1) mod d)
    uint64_t anc = (two31 - 1) - ((two31 - 1) % d);

    uint64_t p  = 31;
    uint64_t q1 =  two31 / anc;  // floor(2^31 / nc)
    uint64_t r1 = (two31 - q1*anc);
    uint64_t q2 =  two31 / d;  // floor(2^31 / d)
    uint64_t r2 = (two31 - q2*d);

    // iterate until q1 >= d-r2 (or equal && r1==0)
    uint64_t delta;
    do {
      p++;
      // advance the nc‐side
      q1 <<= 1;  r1 <<= 1;
      if (r1 >= anc) { q1++;  r1 -= anc; }
      // advance the d‐side
      q2 <<= 1;  r2 <<= 1;
      if (r2 >= d  ) { q2++;  r2 -= d;   }

      delta = d - r2;
    } while (q1 < delta || (q1 == delta && r1 == 0));

    magic_pair mag;
    mag.m = uint32_t(q2 + 1);
    mag.s = int32_t(p - 32);  // note: subtract 32, not 31

    return mag;
}

static __device__ uint32_t magic_divide_unsigned(uint32_t x, const magic_pair &magic) {
    return __umulhi(x, magic.m) >> magic.s;
}

