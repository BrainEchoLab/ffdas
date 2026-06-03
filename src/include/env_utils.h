#pragma once

#include <cstdlib>
#include <cstdio>


inline int getenv_int(const char* key, int fallback) {
    if (const char* s = std::getenv(key)) {
        int v;
        if (std::sscanf(s, "%d", &v) == 1) return v;
    }
    return fallback;
}
