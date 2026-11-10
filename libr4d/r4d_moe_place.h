// r4d_moe_place.h -- moe_scatter_placed's table and dispatch (r4d_moe_place.hip), on the host: the
// op's init builds the table with r4d_place_parse, and r4d_place_owner is the dispatch's reference
// (the emulator test and r4d_selftest hold the kernel to it).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

/* Ranks a placement names: one hex digit an expert. */
constexpr int kPlaceMaxRanks = 4;

/* One rank's table: [0, E) each model expert's copy mask, [E, 2E) its local index on this rank or
 * -1, [2E, 3E) its load units. */
struct R4dPlaceTab {
    long long n_all = 0, n_local = 0;
    int world = 0, rank = 0;
    std::vector<int32_t> tab;
};

/* The op's parameters into `out` for `rank` (r4d_moe_place.hip says what each means). RAD_OK, or
 * RAD_E_SHAPE with `why` naming the first fault. */
int r4d_place_parse(const char* place, const char* extras, const char* cost, long long n_all,
                    long long n_home, long long n_local, int world, int rank, R4dPlaceTab* out,
                    std::string* why);

/* The dispatch on the host: owner[i] = the rank serving slot i of `ids` (T of them), -1 for an id
 * outside the model's experts. The same on every rank's table (the masks and costs are shared). */
void r4d_place_owner(const R4dPlaceTab& t, const int32_t* ids, long long T, int32_t* owner);

/* The kernel over a caller's device copy of `t.tab`, for the emulator and the selftest. */
int r4d_moe_scatter_placed_launch(const int32_t* ids, const int32_t* dtab, int32_t* sorted, int32_t* off,
                                  int32_t* cnt, long long T, long long n_local, long long n_all, int world,
                                  int rank, void* stream);
