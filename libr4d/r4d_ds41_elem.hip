// R4D -- DeepSeek-V4.1's elem ops. Each kernel rounds where its libref function does.
//
//   token_map            mapped[m] = table[ids[m]]                         (ref_sparse_kv.cpp)
//   stream_gate_add      Engram's normalised-dot gate into hc streams      (ref_sparse_kv.cpp)
//   rope_rows            rotary from a rope_table plane over a head window (ref_attn.cpp)
//   quant_roundtrip      block-scaled fp8 / fp4 quantise-dequantise        (ref_sparse_kv.cpp)
//   mhc_mix              mHC projection, gates and Sinkhorn, pre out       (ref_mhc.cpp)
//   mhc_collapse         x = sum_j pre[j] h[j]                             (ref_mhc.cpp)
//   softmax_pool         channel-wise softmax pooling of ratio rows        (ref_sparse_kv.cpp)
//   entry_store          one entry a block into a paged cache              (ref_sparse_kv.cpp)
//   router_scored_topk   correction-bias routing over named scores         (ref_glm.cpp)
//   embed_lookup_planes  gather from affine code/scale(/table) planes      (ref_gemm.cpp)
//
// Every product and sum is the one libref forms, unfused (-ffp-contract=off), in f32; only the
// order of a long sum differs (a lane's strided partial, then a fixed butterfly). Stores narrow as
// rad_f32_to_bf16 does (RNE, a NaN quieted). libref's transcendentals are glibc's float expf /
// log1pf or ref_exp (exp in double, rounded once); here each is the double routine rounded once,
// which is the correctly rounded value glibc returns in all but near-tie cases.
#include "r4d.h"
#include "r4d_ika_x.h"   // lever IKA (DRAFTER track round 50): R4D_IKA_ENTRY(); lever CIK-X (tp8 loop R49c): R4D_IKA_ENTRY_X()
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_ds41_f32pf.h"   /* lever SPD-F: ds41_f32dec's router form (rgs_emit) */
#include "r4d_ds41_pcn.h"
#include "r4d_ds41_qrt.h"
#include "r4d_gemm_f32_tile.h"
#include "r4d_moe_sort.h"
#include "r4d_plugin.h"

#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

#ifndef R4D_EMU_HOST
/* split_ctr's counter blocks (r4d_gemm_fp8a8.hip), for mhc_mix's one-launch form */
unsigned int* r4d_8a8_split_ctr(RadStream s, const void* out);
#endif

namespace {

/* ==================================================================== libref's scalar rules
 * ds_bf16, ds_bf, ds_e4m3, ds_to_e4m3, ds_e8m0, ds_max, ds_clamp, quant_roundtrip's block_scale and
 * roundtrip, and FzRope live in r4d_ds41_qrt.h (one definition, shared with r4d_ds41_sparse.hip's fused
 * epilogue). */
using namespace r4d_ds41q;

// rad_f16_to_f32, bit for bit (subnormals renormalised, a NaN's payload kept).
__device__ __forceinline__ float ds_f16(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    return __uint_as_float(u);
}

// The double routines rounded once, as libref's contract states them.
__device__ __forceinline__ float ds_exp_ref(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float ds_log1p_ref(float x) { return (float)log1p((double)x); }

/* The same two values without FP64 on the common path (gfx1201 runs a wave's double exp + log1p
 * in ~4 us: the router's sqrt(softplus) over 12 experts a lane was 48 us of FP64). Each is formed
 * in float-float (|error| ~2^-46 for exp, ~2^-40 for log1p, relative), and kept only when every
 * real within a bound of that (2^-43, 2^-38: the double routine's own error is far below) rounds
 * to the same float; otherwise (about one input in 2^17 / 2^13, and outside the reduced ranges)
 * the lane runs the double routine itself. So the float returned is the double routine's, bit for
 * bit, for every input: checked exhaustively on gfx1201 over all 2^32 floats against
 * ds_exp_ref / ds_log1p_ref (r4d-port/bench/decsmall). Tables: 2^(j/64) as hi + lo;
 * R_j = RN(1/(1 + j/256)) and -log(R_j) as hi + lo. */
__device__ __constant__ uint32_t kDsExpHi[64] = {
    0x3f800000u, 0x3f8164d2u, 0x3f82cd87u, 0x3f843a29u, 0x3f85aac3u, 0x3f871f62u, 0x3f88980fu, 0x3f8a14d5u,
    0x3f8b95c2u, 0x3f8d1adfu, 0x3f8ea43au, 0x3f9031dcu, 0x3f91c3d3u, 0x3f935a2bu, 0x3f94f4f0u, 0x3f96942du,
    0x3f9837f0u, 0x3f99e046u, 0x3f9b8d3au, 0x3f9d3edau, 0x3f9ef532u, 0x3fa0b051u, 0x3fa27043u, 0x3fa43516u,
    0x3fa5fed7u, 0x3fa7cd94u, 0x3fa9a15bu, 0x3fab7a3au, 0x3fad583fu, 0x3faf3b79u, 0x3fb123f6u, 0x3fb311c4u,
    0x3fb504f3u, 0x3fb6fd92u, 0x3fb8fbafu, 0x3fbaff5bu, 0x3fbd08a4u, 0x3fbf179au, 0x3fc12c4du, 0x3fc346cdu,
    0x3fc5672au, 0x3fc78d75u, 0x3fc9b9beu, 0x3fcbec15u, 0x3fce248cu, 0x3fd06334u, 0x3fd2a81eu, 0x3fd4f35bu,
    0x3fd744fdu, 0x3fd99d16u, 0x3fdbfbb8u, 0x3fde60f5u, 0x3fe0ccdfu, 0x3fe33f89u, 0x3fe5b907u, 0x3fe8396au,
    0x3feac0c7u, 0x3fed4f30u, 0x3fefe4bau, 0x3ff28177u, 0x3ff5257du, 0x3ff7d0dfu, 0x3ffa83b3u, 0x3ffd3e0cu,
};
__device__ __constant__ uint32_t kDsExpLo[64] = {
    0x00000000u, 0xb1c43fd0u, 0xb34ea7a9u, 0xb2f14c87u, 0x334f9891u, 0xb352c2e6u, 0xb37eda4bu, 0x336a92deu,
    0xb260aba1u, 0x3336fcb7u, 0xb3697465u, 0x330628cdu, 0x33675624u, 0x32bc4f9cu, 0xb32e0212u, 0x32dc8061u,
    0x33231b71u, 0xb359be90u, 0xb30c5563u, 0xb331a601u, 0x33412342u, 0x31fb9715u, 0x30c3125au, 0xb323ec33u,
    0xb32c9d5eu, 0xb3162d36u, 0xb3162b08u, 0xb314ad82u, 0xb22deaf6u, 0xb3252debu, 0xb37c5aa8u, 0x32154889u,
    0x32cfe77au, 0xb266b974u, 0x330ec5f7u, 0xb31bd983u, 0xb3414fe8u, 0xb3130b1au, 0xb2d6663eu, 0xb2976da2u,
    0x320aa837u, 0xb2dd5119u, 0xb37323a2u, 0xb006c6c2u, 0x3228fc24u, 0xb2944353u, 0xb35c1daau, 0xb3286024u,
    0xb2d4a58au, 0xb2f61d41u, 0xb3504a1cu, 0xb37b43e3u, 0xb21eab59u, 0x33657d15u, 0xb2441be6u, 0x33207898u,
    0xb24116deu, 0x3276cca1u, 0xb348464au, 0x32f167ffu, 0x32292436u, 0x336615a2u, 0xb2923758u, 0x31cf486cu,
};
__device__ __constant__ uint32_t kDsLogR[256] = {
    0x3f800000u, 0x3f7f00ffu, 0x3f7e03f8u, 0x3f7d08e5u, 0x3f7c0fc1u, 0x3f7b1885u, 0x3f7a232du, 0x3f792fb2u,
    0x3f783e10u, 0x3f774e40u, 0x3f76603eu, 0x3f757404u, 0x3f74898du, 0x3f73a0d5u, 0x3f72b9d6u, 0x3f71d48cu,
    0x3f70f0f1u, 0x3f700f01u, 0x3f6f2eb7u, 0x3f6e500fu, 0x3f6d7304u, 0x3f6c9791u, 0x3f6bbdb3u, 0x3f6ae564u,
    0x3f6a0ea1u, 0x3f693965u, 0x3f6865acu, 0x3f679373u, 0x3f66c2b4u, 0x3f65f36du, 0x3f652598u, 0x3f645933u,
    0x3f638e39u, 0x3f62c4a7u, 0x3f61fc78u, 0x3f6135aau, 0x3f607038u, 0x3f5fac1fu, 0x3f5ee95cu, 0x3f5e27ebu,
    0x3f5d67c9u, 0x3f5ca8f1u, 0x3f5beb62u, 0x3f5b2f17u, 0x3f5a740eu, 0x3f59ba42u, 0x3f5901b2u, 0x3f584a5au,
    0x3f579436u, 0x3f56df44u, 0x3f562b81u, 0x3f5578e9u, 0x3f54c77bu, 0x3f541733u, 0x3f53680du, 0x3f52ba08u,
    0x3f520d21u, 0x3f516154u, 0x3f50b6a0u, 0x3f500d01u, 0x3f4f6475u, 0x3f4ebcf9u, 0x3f4e168au, 0x3f4d7127u,
    0x3f4ccccdu, 0x3f4c2978u, 0x3f4b8728u, 0x3f4ae5d8u, 0x3f4a4588u, 0x3f49a634u, 0x3f4907dau, 0x3f486a79u,
    0x3f47ce0cu, 0x3f473294u, 0x3f46980cu, 0x3f45fe74u, 0x3f4565c8u, 0x3f44ce08u, 0x3f443730u, 0x3f43a13eu,
    0x3f430c31u, 0x3f427806u, 0x3f41e4bcu, 0x3f415250u, 0x3f40c0c1u, 0x3f40300cu, 0x3f3fa030u, 0x3f3f112bu,
    0x3f3e82fau, 0x3f3df59du, 0x3f3d6910u, 0x3f3cdd53u, 0x3f3c5264u, 0x3f3bc841u, 0x3f3b3ee7u, 0x3f3ab656u,
    0x3f3a2e8cu, 0x3f39a786u, 0x3f392144u, 0x3f389bc3u, 0x3f381703u, 0x3f379301u, 0x3f370fbbu, 0x3f368d31u,
    0x3f360b61u, 0x3f358a48u, 0x3f3509e7u, 0x3f348a3au, 0x3f340b41u, 0x3f338cfau, 0x3f330f63u, 0x3f32927cu,
    0x3f321643u, 0x3f319ab6u, 0x3f311fd4u, 0x3f30a59bu, 0x3f302c0bu, 0x3f2fb322u, 0x3f2f3adeu, 0x3f2ec33eu,
    0x3f2e4c41u, 0x3f2dd5e6u, 0x3f2d602bu, 0x3f2ceb10u, 0x3f2c7692u, 0x3f2c02b0u, 0x3f2b8f6au, 0x3f2b1cbeu,
    0x3f2aaaabu, 0x3f2a392fu, 0x3f29c84au, 0x3f2957fbu, 0x3f28e83fu, 0x3f287917u, 0x3f280a81u, 0x3f279c7bu,
    0x3f272f05u, 0x3f26c21eu, 0x3f2655c4u, 0x3f25e9f7u, 0x3f257eb5u, 0x3f2513fdu, 0x3f24a9cfu, 0x3f244029u,
    0x3f23d70au, 0x3f236e72u, 0x3f23065eu, 0x3f229ecfu, 0x3f2237c3u, 0x3f21d13au, 0x3f216b31u, 0x3f2105a9u,
    0x3f20a0a1u, 0x3f203c17u, 0x3f1fd80au, 0x3f1f747au, 0x3f1f1166u, 0x3f1eaecdu, 0x3f1e4cadu, 0x3f1deb07u,
    0x3f1d89d9u, 0x3f1d2922u, 0x3f1cc8e1u, 0x3f1c6917u, 0x3f1c09c1u, 0x3f1baadfu, 0x3f1b4c70u, 0x3f1aee73u,
    0x3f1a90e8u, 0x3f1a33cdu, 0x3f19d723u, 0x3f197ae7u, 0x3f191f1au, 0x3f18c3bbu, 0x3f1868c8u, 0x3f180e41u,
    0x3f17b426u, 0x3f175a75u, 0x3f17012eu, 0x3f16a850u, 0x3f164fdau, 0x3f15f7ccu, 0x3f15a025u, 0x3f1548e5u,
    0x3f14f209u, 0x3f149b93u, 0x3f144581u, 0x3f13efd2u, 0x3f139a86u, 0x3f13459cu, 0x3f12f114u, 0x3f129cecu,
    0x3f124925u, 0x3f11f5bdu, 0x3f11a2b4u, 0x3f115009u, 0x3f10fdbcu, 0x3f10abccu, 0x3f105a38u, 0x3f100901u,
    0x3f0fb824u, 0x3f0f67a2u, 0x3f0f177au, 0x3f0ec7abu, 0x3f0e7835u, 0x3f0e2918u, 0x3f0dda52u, 0x3f0d8be3u,
    0x3f0d3dcbu, 0x3f0cf009u, 0x3f0ca29cu, 0x3f0c5584u, 0x3f0c08c1u, 0x3f0bbc51u, 0x3f0b7034u, 0x3f0b246bu,
    0x3f0ad8f3u, 0x3f0a8dcdu, 0x3f0a42f8u, 0x3f09f874u, 0x3f09ae41u, 0x3f09645cu, 0x3f091ac7u, 0x3f08d181u,
    0x3f088889u, 0x3f083fdeu, 0x3f07f781u, 0x3f07af70u, 0x3f0767abu, 0x3f072033u, 0x3f06d905u, 0x3f069223u,
    0x3f064b8au, 0x3f06053cu, 0x3f05bf37u, 0x3f05797cu, 0x3f053408u, 0x3f04eeddu, 0x3f04a9fau, 0x3f04655eu,
    0x3f042108u, 0x3f03dcf9u, 0x3f039930u, 0x3f0355adu, 0x3f03126fu, 0x3f02cf75u, 0x3f028cc0u, 0x3f024a4eu,
    0x3f020821u, 0x3f01c636u, 0x3f01848eu, 0x3f014328u, 0x3f010204u, 0x3f00c122u, 0x3f008081u, 0x3f004020u,
};
__device__ __constant__ uint32_t kDsLogHi[256] = {
    0x00000000u, 0x3b7f8056u, 0x3bff015bu, 0x3c3ee24fu, 0x3c7e0545u, 0x3c9e752fu, 0x3cbdc8d6u, 0x3cdcfe05u,
    0x3cfc14c8u, 0x3d0d86c8u, 0x3d1cf437u, 0x3d2c52dbu, 0x3d3ba2ceu, 0x3d4ae41bu, 0x3d5a16f0u, 0x3d693b59u,
    0x3d785185u, 0x3d83acc1u, 0x3d8b29b9u, 0x3d929fb1u, 0x3d9a0ebau, 0x3da176e7u, 0x3da8d837u, 0x3db032c5u,
    0x3db78694u, 0x3dbed3b5u, 0x3dc61a33u, 0x3dcd5a11u, 0x3dd4936cu, 0x3ddbc63fu, 0x3de2f2a6u, 0x3dea189du,
    0x3df1383au, 0x3df85182u, 0x3dff648au, 0x3e0338a8u, 0x3e06bbf4u, 0x3e0a3c2eu, 0x3e0db958u, 0x3e11337au,
    0x3e14aa96u, 0x3e181ebau, 0x3e1b8fe1u, 0x3e1efe16u, 0x3e22695au, 0x3e25d1b9u, 0x3e29372fu, 0x3e2c99c5u,
    0x3e2ff983u, 0x3e33566au, 0x3e36b07eu, 0x3e3a07cau, 0x3e3d5c48u, 0x3e40ae03u, 0x3e43fd04u, 0x3e474948u,
    0x3e4a92d4u, 0x3e4dd9b3u, 0x3e511de0u, 0x3e545f67u, 0x3e579e49u, 0x3e5ada8cu, 0x3e5e1436u, 0x3e614b44u,
    0x3e647fbdu, 0x3e67b1adu, 0x3e6ae10au, 0x3e6e0de7u, 0x3e71383bu, 0x3e746013u, 0x3e778570u, 0x3e7aa850u,
    0x3e7dc8c6u, 0x3e807362u, 0x3e82012eu, 0x3e838dc3u, 0x3e851928u, 0x3e86a35au, 0x3e882c5fu, 0x3e89b438u,
    0x3e8b3ae5u, 0x3e8cc06au, 0x3e8e44c6u, 0x3e8fc7fdu, 0x3e914a0fu, 0x3e92cb01u, 0x3e944ad0u, 0x3e95c981u,
    0x3e974716u, 0x3e98c38du, 0x3e9a3eeeu, 0x3e9bb934u, 0x3e9d3263u, 0x3e9eaa7cu, 0x3ea02185u, 0x3ea19779u,
    0x3ea30c5du, 0x3ea48034u, 0x3ea5f2fdu, 0x3ea764bbu, 0x3ea8d56cu, 0x3eaa4515u, 0x3eabb3bau, 0x3ead2156u,
    0x3eae8dedu, 0x3eaff984u, 0x3eb16416u, 0x3eb2cdabu, 0x3eb43640u, 0x3eb59dd8u, 0x3eb70476u, 0x3eb86a18u,
    0x3eb9cebfu, 0x3ebb3270u, 0x3ebc952au, 0x3ebdf6f1u, 0x3ebf57c2u, 0x3ec0b7a0u, 0x3ec2168eu, 0x3ec3748du,
    0x3ec4d19du, 0x3ec62dbfu, 0x3ec788f5u, 0x3ec8e33eu, 0x3eca3c9fu, 0x3ecb951au, 0x3eccecacu, 0x3ece4358u,
    0x3ecf991eu, 0x3ed0ee04u, 0x3ed24205u, 0x3ed39524u, 0x3ed4e766u, 0x3ed638c6u, 0x3ed78949u, 0x3ed8d8f1u,
    0x3eda27bdu, 0x3edb75adu, 0x3edcc2c5u, 0x3ede0f04u, 0x3edf5a6du, 0x3ee0a501u, 0x3ee1eebeu, 0x3ee337a7u,
    0x3ee47fbfu, 0x3ee5c702u, 0x3ee70d79u, 0x3ee8531eu, 0x3ee997f4u, 0x3eeadbfbu, 0x3eec1f3au, 0x3eed61abu,
    0x3eeea34fu, 0x3eefe42bu, 0x3ef12441u, 0x3ef2638eu, 0x3ef3a213u, 0x3ef4dfd2u, 0x3ef61ccfu, 0x3ef75906u,
    0x3ef8947au, 0x3ef9cf2du, 0x3efb0921u, 0x3efc4250u, 0x3efd7ac3u, 0x3efeb278u, 0x3effe96fu, 0x3f008fd6u,
    0x3f012a95u, 0x3f01c4f8u, 0x3f025efdu, 0x3f02f8a7u, 0x3f0391f3u, 0x3f042ae4u, 0x3f04c37bu, 0x3f055bb7u,
    0x3f05f397u, 0x3f068b1fu, 0x3f07224cu, 0x3f07b921u, 0x3f084f9eu, 0x3f08e5c2u, 0x3f097b8du, 0x3f0a1100u,
    0x3f0aa61fu, 0x3f0b3ae5u, 0x3f0bcf55u, 0x3f0c6370u, 0x3f0cf735u, 0x3f0d8aa5u, 0x3f0e1dc0u, 0x3f0eb088u,
    0x3f0f42fau, 0x3f0fd51au, 0x3f1066e6u, 0x3f10f860u, 0x3f118987u, 0x3f121a5bu, 0x3f12aadfu, 0x3f133b0fu,
    0x3f13caf0u, 0x3f145a80u, 0x3f14e9c0u, 0x3f1578b0u, 0x3f160750u, 0x3f1695a0u, 0x3f1723a2u, 0x3f17b155u,
    0x3f183ebau, 0x3f18cbd0u, 0x3f19589au, 0x3f19e516u, 0x3f1a7144u, 0x3f1afd27u, 0x3f1b88beu, 0x3f1c1407u,
    0x3f1c9f07u, 0x3f1d29bau, 0x3f1db423u, 0x3f1e3e41u, 0x3f1ec812u, 0x3f1f519du, 0x3f1fdaddu, 0x3f2063d2u,
    0x3f20ec7eu, 0x3f2174e3u, 0x3f21fcfeu, 0x3f2284d2u, 0x3f230c5fu, 0x3f2393a1u, 0x3f241a9fu, 0x3f24a154u,
    0x3f2527c4u, 0x3f25adecu, 0x3f2633ceu, 0x3f26b969u, 0x3f273ec1u, 0x3f27c3d2u, 0x3f28489eu, 0x3f28cd25u,
    0x3f29516au, 0x3f29d569u, 0x3f2a5924u, 0x3f2adc9au, 0x3f2b5fceu, 0x3f2be2c0u, 0x3f2c656du, 0x3f2ce7dau,
    0x3f2d6a01u, 0x3f2debe9u, 0x3f2e6d8eu, 0x3f2eeef3u, 0x3f2f7015u, 0x3f2ff0f6u, 0x3f307197u, 0x3f30f1f8u,
};
__device__ __constant__ uint32_t kDsLogLo[256] = {
    0x00000000u, 0x2dac42f0u, 0x2f310679u, 0x2f70215cu, 0xaff03fc2u, 0xae395a38u, 0x307d5b12u, 0x307af91au,
    0x30678338u, 0x30923d99u, 0x2f7f5ec7u, 0xb08371b2u, 0xaf66916eu, 0x30cc8e3fu, 0x3091971du, 0x2fb6c2d7u,
    0x2d0b153bu, 0x30931c8eu, 0xb175c852u, 0xb15f5ebeu, 0x30c37a3cu, 0xb0cb3722u, 0xb03e79ebu, 0x3153750cu,
    0xb151a94bu, 0xb1784ed3u, 0xafe7325du, 0x30a40676u, 0x3124ad57u, 0xb154f125u, 0xaf2438b8u, 0x3116cb36u,
    0x3162af2eu, 0xb0ddd3a7u, 0x2fedf55eu, 0xb1c35a69u, 0x31cd08e6u, 0x315ce858u, 0x3017b3d4u, 0xb1d75a53u,
    0x3103fea6u, 0x31c647feu, 0xb19e1709u, 0x309b318bu, 0xb166905bu, 0x31534abeu, 0x31343687u, 0xb1967df7u,
    0x2fa793d5u, 0x2f655aa2u, 0x31e19d22u, 0xaeff2116u, 0x31021b21u, 0xb1479500u, 0x31441923u, 0x30a21b51u,
    0x2ff456b8u, 0xb1b82047u, 0xae6a54c7u, 0xaff7d0e4u, 0xb16fe800u, 0xb1bf04a5u, 0xb069398au, 0x3190c18du,
    0x31735345u, 0xb1df63a1u, 0x30d3eee6u, 0xb0d504ddu, 0xb06a868du, 0xb1f55a5au, 0xb088e9b0u, 0x31ecb7b5u,
    0xb0a8706fu, 0xb1ea55fcu, 0xb214b2fbu, 0x304d590du, 0x3227390fu, 0xb119522cu, 0xb2651b52u, 0xb1bd8aeau,
    0xb205459fu, 0xb23abdc9u, 0xb2496660u, 0x311228bdu, 0x31d9ef2du, 0xb262089cu, 0x323de86au, 0xb248762du,
    0xb08f7168u, 0x31828832u, 0xb256782au, 0x311ff619u, 0xb1d2d743u, 0xb0bf0472u, 0xb25f9582u, 0x32784e81u,
    0x310717b3u, 0x326c009du, 0xb2488876u, 0xb1fbffb1u, 0xb17481f5u, 0x3226dd26u, 0xb213aa59u, 0x314d33f9u,
    0x31ab013au, 0x326a793eu, 0x32139c7au, 0x3160f87du, 0x3166c52du, 0x31b6b288u, 0x31f5743du, 0xb26876ddu,
    0x312ef402u, 0xb151890bu, 0x31bba8f5u, 0x30fbd7e2u, 0xb15f540fu, 0xb19be56cu, 0x31a3d163u, 0x31b66174u,
    0x320c1426u, 0xb1a4e96eu, 0x31f6cc5au, 0xb03a57b3u, 0x32621171u, 0x31a26a2bu, 0xb21e81f7u, 0xb25a3f7au,
    0x324bf985u, 0xb1dd48cdu, 0x323e5bb1u, 0xb1ba84c2u, 0xb18cbef6u, 0x31b3161du, 0xb2738879u, 0xb208203au,
    0xb263370au, 0xb1c86194u, 0x32493110u, 0x31736b43u, 0xb0ac7242u, 0xb1a541f9u, 0xb24b252bu, 0x3223f700u,
    0xafcacbb4u, 0x327cd7adu, 0x3190be7eu, 0xb20cde1au, 0x32700ebdu, 0x32262c69u, 0xb1feba18u, 0xb2593a6bu,
    0xb09f1daau, 0x32146d5cu, 0xb1d0727bu, 0xb2717890u, 0xb2029787u, 0xb20a79e0u, 0x30468cc2u, 0xb27ff220u,
    0xb2050f90u, 0x318023ccu, 0x30f8145cu, 0x325be9edu, 0x31cf1e65u, 0xb25c9ea8u, 0x2fafe94au, 0xb2f7a007u,
    0xb1968bc0u, 0x3225cc68u, 0xb2a16d93u, 0xb2c79ba6u, 0x32d1cdf4u, 0xb1e52ebcu, 0xb2ea8caeu, 0xb2f11eb3u,
    0x2f94aa0cu, 0xb2fe6294u, 0x324a3991u, 0xb1325b41u, 0xb2ad2b36u, 0xb2ff8330u, 0x32bd3efcu, 0x32ec43d5u,
    0x31fd357cu, 0x320974c2u, 0x32131335u, 0x31cd73c2u, 0x326cf92eu, 0x32ffda2au, 0x320e2786u, 0xb2031200u,
    0x324e081bu, 0xb2031ed6u, 0x3212556eu, 0x3102df76u, 0xb2482818u, 0x32fc8393u, 0x32131efau, 0x3285430fu,
    0x32e89b12u, 0x3276c0f0u, 0xb29bd4f7u, 0xb1be6fd4u, 0x323891d3u, 0xb23a246cu, 0xb289bfd9u, 0x327612a9u,
    0xb2ab0ef4u, 0x3210ffceu, 0xb23dbc52u, 0x30b28995u, 0x31c73878u, 0x3119b751u, 0x3248e8f7u, 0x31ae1e63u,
    0xb2da9d5eu, 0x323bd7bfu, 0x31ffff0bu, 0xb27119beu, 0x32ef1570u, 0x323d65ecu, 0xb2d6e90cu, 0xb1d7c8c5u,
    0x32c4672cu, 0x32a1f80cu, 0x325739ceu, 0x32667761u, 0xb26c7427u, 0x32b53aabu, 0x318aa239u, 0xb2efd03fu,
    0xb209f827u, 0xb2a291d9u, 0x32619dc8u, 0xb00c05e9u, 0x3236eb77u, 0x32c3026fu, 0xb13f4bc9u, 0x3287a345u,
    0xb29a43a5u, 0xb26ccc90u, 0xb20ccd6cu, 0x32dd40bdu, 0xb1f30317u, 0xb2be90e7u, 0x3239af32u, 0xb19145deu,
    0x32d359f4u, 0x2f9c71deu, 0x325d82edu, 0xb2feb66eu, 0x32b24167u, 0x32723124u, 0xb2b78771u, 0x31523233u,
};

__device__ __forceinline__ void ds_two_sum(float a, float b, float& s, float& e) {
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void ds_two_prod(float a, float b, float& p, float& e) {
    p = a * b;
    e = fmaf(a, b, -p);
}
// Every real within rel*zh of zh + zl (zh > 0, normal) rounds to zh.
__device__ __forceinline__ bool ds_settled(float zh, float zl, float rel) {
    const uint32_t u = __float_as_uint(zh);
    const float nb = __uint_as_float(zl < 0.0f ? u - 1u : u + 1u);
    return fabsf(zl) + rel * zh < fabsf(nb - zh) * 0.5f;
}
__device__ __forceinline__ float ds_tab(const uint32_t* t, int i) { return __uint_as_float(t[i]); }

// Branch-free, so a caller's independent evaluations interleave; false = take the double routine.
__device__ __forceinline__ bool ds_exp_fast(float xi, float& y) {
    const bool dom = fabsf(xi) <= 87.0f;
    const float x = dom ? xi : 0.0f;
    const float kf = rintf(x * __uint_as_float(0x42b8aa3bu));     // x * 64 / ln2
    const int k = (int)kf;
    const float c1 = __uint_as_float(0x3c316000u), c2 = __uint_as_float(0x3690bfbfu),
                c3 = __uint_as_float(0xaa630865u);                 // ln2/64 = c1 + c2 + c3
    const float rh0 = x - kf * c1;                                 // both steps exact
    float ph, pl, rh, rl;
    ds_two_prod(kf, c2, ph, pl);
    ds_two_sum(rh0, -ph, rh, rl);
    rl = rl - (pl + kf * c3);
    ds_two_sum(rh, rl, rh, rl);                                    // r = rh + rl, |r| <= ln2/128
    float qh, ql, eh, el;
    ds_two_prod(rh, rh, qh, ql);
    const float t = rh * qh * (0.16666667f + rh * (0.041666668f + rh * 0.008333334f));
    ds_two_sum(rh, 0.5f * qh, eh, el);                             // e^r - 1 = eh + el
    el = el + (t + (0.5f * ql + (rh * rl + rl)));
    const float th = ds_tab(kDsExpHi, k & 63), tl = ds_tab(kDsExpLo, k & 63);
    float p2h, p2l, yh, yl, zh, zl;
    ds_two_prod(th, eh, p2h, p2l);
    ds_two_sum(th, p2h, yh, yl);
    yl = yl + (tl + (p2l + (th * el + tl * eh)));
    ds_two_sum(yh, yl, zh, zl);
    y = ldexpf(zh, k >> 6);
    return dom & ds_settled(zh, zl, 0x1p-43f);  // `&`: no branch
}

__device__ __forceinline__ bool ds_log1p_fast(float xi, float& y) {
    const bool dom = xi >= 0x1p-60f && xi <= 0x1p100f;
    const float x = dom ? xi : 1.0f;
    float uh, ul;
    ds_two_sum(1.0f, x, uh, ul);                                   // 1 + x = uh + ul exactly
    const uint32_t bu = __float_as_uint(uh);
    const int m = (int)(bu >> 23) - 127, j = (int)((bu >> 15) & 255u);
    const float f = __uint_as_float((bu & 0x007fffffu) | 0x3f800000u), r = ds_tab(kDsLogR, j);
    float ph, pl, sh, sl;
    ds_two_prod(f, r, ph, pl);
    ds_two_sum(ph - 1.0f, pl, sh, sl);                             // f r - 1, ph - 1 exact
    sl = sl + ldexpf(ul, -m) * r;                                  // s = (1 + x) 2^-m r - 1
    ds_two_sum(sh, sl, sh, sl);
    float qh, ql, lh, ll;
    ds_two_prod(sh, sh, qh, ql);
    ql = ql + 2.0f * sh * sl;
    const float p3 = sh * qh *
        (0.33333334f + sh * (-0.25f + sh * (0.2f + sh * (-0.16666667f + sh * 0.14285715f))));
    ds_two_sum(sh, -0.5f * qh, lh, ll);                            // log1p(s) = lh + ll
    ll = ll + (p3 + (sl - 0.5f * ql));
    const float mf = (float)m, l2h = __uint_as_float(0x3f317200u),
                l2l = __uint_as_float(0x35bfbe8eu), l2l2 = __uint_as_float(0x29779abdu);
    float bh, bl, a0, e0, a1, e1, zh, zl;
    ds_two_prod(mf, l2l, bh, bl);
    ds_two_sum(mf * l2h, ds_tab(kDsLogHi, j), a0, e0);             // m ln2 + -log(R_j) + log1p(s)
    ds_two_sum(a0, lh, a1, e1);
    const float lo = e1 + (e0 + (ll + (bh + (ds_tab(kDsLogLo, j) + (bl + mf * l2l2)))));
    ds_two_sum(a1, lo, zh, zl);
    y = zh;
    return dom & ds_settled(zh, zl, 0x1p-38f);
}

__device__ __forceinline__ float ds_exp(float x) {
    float y;
    return ds_exp_fast(x, y) ? y : ds_exp_ref(x);
}
// N independent values in place: every fast path first, then the rare double routines.
// (One branch for all N, so the N fast paths interleave ahead of it.)
template <int N>
__device__ __forceinline__ void ds_exp_n(float (&v)[N]) {
    float y[N];
    bool ok[N], all = true;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        ok[i] = ds_exp_fast(v[i], y[i]);
        all &= ok[i];
    }
    if (!all)
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (!ok[i]) y[i] = ds_exp_ref(v[i]);
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = y[i];
}
template <int N>
__device__ __forceinline__ void ds_log1p_n(float (&v)[N]) {
    float y[N];
    bool ok[N], all = true;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        ok[i] = ds_log1p_fast(v[i], y[i]);
        all &= ok[i];
    }
    if (!all)
#pragma unroll
        for (int i = 0; i < N; ++i)
            if (!ok[i]) y[i] = ds_log1p_ref(v[i]);
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] = y[i];
}

// An element of a float plane of any of the dtypes ld_dt reads that these ops accept.
__device__ __forceinline__ float ds_ld(uint32_t dt, const void* p, long i) {
    switch (dt) {
        case RAD_F32:    return ((const float*)p)[i];
        case RAD_F16:    return ds_f16(((const uint16_t*)p)[i]);
        case RAD_BF16:   return ds_bf(((const uint16_t*)p)[i]);
        case RAD_F8E4M3: return ds_e4m3(((const uint8_t*)p)[i]);
        case RAD_E8M0:   return ds_e8m0(((const uint8_t*)p)[i]);
        case RAD_I8:     return (float)((const int8_t*)p)[i];
        case RAD_U8:     return (float)((const uint8_t*)p)[i];
        default:         return 0.0f;
    }
}

/* ==================================================================== host-side operand checks */

// An index operand read as libref's offlin(t, m): element m at m * step. Contiguous, or one
// non-unit axis with its stride.
bool lin_step(const RadTensor& t, long* step) {
    if (!t.data || t.rank == 0) return false;
    if (rad_tensor_is_contiguous(&t)) { *step = 1; return true; }
    int nd = 0, d = 0;
    for (uint32_t i = 0; i < t.rank; ++i)
        if (t.shape[i] != 1) { ++nd; d = (int)i; }
    if (nd != 1) return false;
    *step = (long)t.stride[d];
    return true;
}
// [rows, cols] with a unit inner stride; the row pitch is stride[0].
bool mat(const RadTensor& t, uint32_t dt, long rows, long cols) {
    return t.data && t.dtype == dt && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols &&
           t.stride[1] == 1 && t.stride[0] >= cols;
}
// [rows, cols] of any inner stride (the few f32 coefficients a row of mhc_mix writes).
bool mat_s(const RadTensor& t, uint32_t dt, long rows, long cols) {
    return t.data && t.dtype == dt && t.rank == 2 && t.shape[0] == rows && t.shape[1] == cols;
}
long last_stride(const RadTensor& t) { return t.rank ? (long)t.stride[t.rank - 1] : 1; }
unsigned grid_rows(long rows) { return (unsigned)(rows < 65535 ? rows : 65535); }
/* Lever WQ (DRAFTER track, 18.11.2026 round 5): r4d_common.h's full-card cliff (r4d_cliff_rows, dffec0b: a launch whose
 * waves fall in the last 32 before a multiple of 2048 runs ~5x slower) for the row-looped launches that did not step past it:
 * quant_roundtrip, rope_rows_roundtrip, ds41_rope_rows (`for (r = blockIdx.y; r < rows; r += gridDim.y)`: extra y blocks do
 * nothing -- the same work, the same bits). Measured where it was missed: the drafter's o_q quant_group at [16, 8192] = 512
 * workgroups 19-23 us a stage in serving (lq145), 18.6 -> 4.6 us on the xeon (abench a9); the trunk's rope_rt_kernel<32> ran
 * 16-22 us on ~24 % of its lq145 calls; tools/dbench/wgq re-measured the cliff on both xeon cards (24.5-26.3 vs 5.4-6.4 us). */
std::atomic<int> g_wq_pin{-1};   /* bench pin (abench): 0 = no step past the cliff */
unsigned ds_dodge_gy(unsigned gx, unsigned gy, unsigned waves_per_wg) {
    if (g_wq_pin.load(std::memory_order_relaxed) == 0) return gy;
    return r4d_cliff_rows(gy, gx, waves_per_wg);
}

/* ==================================================================== token_map */

__global__ __launch_bounds__(256) void token_map_kernel(const int32_t* ids, long is,
                                                        const int32_t* tab, long ts, int32_t* y,
                                                        long ys, long M, long nv) { R4D_IKA_ENTRY();
    for (long m = (long)blockIdx.x * 256 + threadIdx.x; m < M; m += (long)gridDim.x * 256) {
        const long id = ids[m * is];
        y[m * ys] = id < 0 || id >= nv ? -1 : tab[id * ts];
    }
}

int token_map_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &tab = a->t[1], &y = a->t[2];
    const long nv = (long)r4d_geti_or(a, "n_vocab", 0);
    const long M = (long)rad_tensor_numel(&x);
    if (nv < 1 || rad_tensor_numel(&tab) != nv || M < 1 || rad_tensor_numel(&y) != M)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_I32 || tab.dtype != RAD_I32 || y.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    long is, ts, ys;
    if (!lin_step(x, &is) || !lin_step(tab, &ts) || !lin_step(y, &ys)) return R4D_NO(RAD_E_SHAPE);
    const long blocks = (M + 255) / 256;
    return R4D_EMIT(stream, token_map_kernel, dim3((unsigned)(blocks < 65535 ? blocks : 65535)),
                    dim3(256), 0, (const int32_t*)x.data, is, (const int32_t*)tab.data, ts,
                    (int32_t*)y.data, ys, M, nv);
}

/* ==================================================================== bi_rowhash (lever BI's tracer, ds41-bi)
 * A 64-bit fingerprint of each row's bytes: lane t folds its 16-bit units u (at index i) as XOR of a mix of (i, u), the
 * workgroup XORs the lanes' folds -- an order-free reduction, so the value is a function of the row's bytes alone. A
 * debug op: declared only under RADIANCE_DS41_BITRACE (deepseek41), never in a served pass otherwise. */
__device__ __forceinline__ unsigned long long bi_mix(unsigned long long x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return x;
}
__global__ __launch_bounds__(256) void bi_rowhash_kernel(const uint8_t* x, long xs, long nu, int odd, uint32_t* h,
                                                         long hs, long rows) { R4D_IKA_ENTRY();
    __shared__ unsigned long long sh[256];
    for (long r = blockIdx.x; r < rows; r += gridDim.x) {
        const uint16_t* p = (const uint16_t*)(x + r * xs);
        unsigned long long acc = 0;
        for (long i = threadIdx.x; i < nu; i += 256)
            acc ^= bi_mix(((unsigned long long)i << 16 | p[i]) ^ 0x9E3779B97F4A7C15ull);
        if (odd && threadIdx.x == 0) acc ^= bi_mix(((unsigned long long)nu << 16 | x[r * xs + 2 * nu]) ^ 0x51ED27ull);
        sh[threadIdx.x] = acc;
        __syncthreads();
        for (int w = 128; w > 0; w >>= 1) {
            if ((int)threadIdx.x < w) sh[threadIdx.x] ^= sh[threadIdx.x + w];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const unsigned long long v = bi_mix(sh[0] ^ (unsigned long long)(2 * nu + odd));
            h[r * hs] = (uint32_t)v;
            h[r * hs + 1] = (uint32_t)(v >> 32);
        }
        __syncthreads();
    }
}

int bi_rowhash_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 2 || !a->t[0].data || !a->t[1].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &h = a->t[1];
    if (x.rank < 1 || h.rank != 2 || h.dtype != RAD_I32 || h.shape[1] != 2 || h.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = (long)x.shape[0];
    if (rows < 1 || (long)h.shape[0] != rows) return R4D_NO(RAD_E_SHAPE);
    /* the row's bytes: the trailing extents contiguous (a column slice keeps its row pitch in stride[0]) */
    long inner = 1;
    for (uint32_t d = x.rank; d-- > 1;) {
        if ((long)x.stride[d] != inner) return R4D_NO(RAD_E_SHAPE);
        inner *= (long)x.shape[d];
    }
    const long eb = (long)rad_dtype_bytes(x.dtype, 1);
    if (eb < 1) return R4D_NO(RAD_E_DTYPE);
    const long bytes = inner * eb, xs = (x.rank > 1 ? (long)x.stride[0] : 1) * eb;
    if (((uintptr_t)x.data & 1u) || (xs & 1)) return R4D_NO(RAD_E_ALIGN);
    return R4D_EMIT(stream, bi_rowhash_kernel, dim3(grid_rows(rows)), dim3(256), 0, (const uint8_t*)x.data, xs,
                    bytes / 2, (int)(bytes & 1), (uint32_t*)h.data, (long)h.stride[0], rows);
}

/* ==================================================================== stream_gate_add
 * One workgroup a (row, stream): the three sums over the stream's n values (a lane's strided
 * partial, the wave butterfly, the eight waves in order), the gate on one lane, then the write. */

__global__ __launch_bounds__(256) void stream_gate_kernel(const uint16_t* h, long hp,
                                                          const uint16_t* k, long kp,
                                                          const uint16_t* v, long vp,
                                                          const float* w, long wst, uint16_t* out,
                                                          long op, int n, long rows, float eps,
                                                          float ge) { R4D_IKA_ENTRY();
    __shared__ float red[3][8];
    __shared__ float gate_s;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, s = blockIdx.x;
    const float* ws = w + (long)s * n * wst;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        const uint16_t* hr = h + r * hp + (long)s * n;
        const uint16_t* kr = k + r * kp + (long)s * n;
        float sh = 0, sk = 0, dot = 0;
        for (int i = tid; i < n; i += 256) {
            const float hv = ds_bf(hr[i]), kv = ds_bf(kr[i]);
            sh += hv * hv;
            sk += kv * kv;
            dot += hv * ws[(long)i * wst] * kv;
        }
        sh = r4d_wave_sum(sh);
        sk = r4d_wave_sum(sk);
        dot = r4d_wave_sum(dot);
        if (!lane) {
            red[0][wave] = sh;
            red[1][wave] = sk;
            red[2][wave] = dot;
        }
        __syncthreads();
        if (!tid) {
            float a = 0, b = 0, c = 0;
            for (int q = 0; q < 8; ++q) {
                a += red[0][q];
                b += red[1][q];
                c += red[2][q];
            }
            const float rstd =
                (1.0f / sqrtf(a / (float)n + eps)) * (1.0f / sqrtf(b / (float)n + eps));
            c = c * rstd * (1.0f / sqrtf((float)n));
            const float t = copysignf(sqrtf(ds_max(fabsf(c), ge)), c);
            gate_s = 1.0f / (1.0f + ds_exp(-t));
        }
        __syncthreads();
        const float g = gate_s;
        const uint16_t* vr = v + r * vp;
        uint16_t* orow = out + r * op + (long)s * n;
        for (int i = tid; i < n; i += 256) orow[i] = ds_bf16(ds_bf(hr[i]) + g * ds_bf(vr[i]));
        __syncthreads();
    }
}

int stream_gate_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor &h = a->t[0], &k = a->t[1], &v = a->t[2], &w = a->t[3], &out = a->t[4];
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    const float eps = (float)r4d_getf(a, "eps", -1.0), ge = (float)r4d_getf(a, "gate_eps", -1.0);
    if (n < 1 || hc < 1 || hc > 16 || n * hc > INT_MAX || !(eps >= 0) || !(ge >= 0) ||
        h.rank != 2 || h.shape[0] < 1)
        return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || k.dtype != RAD_BF16 || v.dtype != RAD_BF16 || w.dtype != RAD_F32 ||
        out.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat(k, RAD_BF16, rows, hc * n) ||
        !mat(v, RAD_BF16, rows, n) || !mat(out, RAD_BF16, rows, hc * n) || !w.data ||
        rad_tensor_numel(&w) != hc * n || !(w.rank == 1 || rad_tensor_is_contiguous(&w)))
        return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, stream_gate_kernel, dim3((unsigned)hc, grid_rows(rows)), dim3(256), 0,
                    (const uint16_t*)h.data, (long)h.stride[0], (const uint16_t*)k.data,
                    (long)k.stride[0], (const uint16_t*)v.data, (long)v.stride[0],
                    (const float*)w.data, last_stride(w), (uint16_t*)out.data, (long)out.stride[0],
                    (int)n, rows, eps, ge);
}

/* ==================================================================== rope_rows
 * One thread a rotated pair of a row, rows over grid.y. */

__global__ __launch_bounds__(256) void rope_rows_kernel(uint16_t* x, long xp, const float* cs,
                                                        long cp, long cst, const int32_t* pos,
                                                        long ps, long M, long tab_rows, int hd,
                                                        int half, int off, int pairs, int gptj,
                                                        float sign) { R4D_IKA_ENTRY();
    const int q = blockIdx.x * 256 + threadIdx.x;
    if (q >= pairs) return;
    const int hh = q / half, i = q - hh * half;
    const long ia = (long)hh * hd + off + (gptj ? 2 * i : i), ib = gptj ? ia + 1 : ia + half;
    for (long m = blockIdx.y; m < M; m += gridDim.y) {
        const long p = pos[m * ps];
        if (p < 0 || p >= tab_rows) continue;
        const float c = cs[p * cp + (long)i * cst], s = sign * cs[p * cp + (long)(half + i) * cst];
        uint16_t* row = x + m * xp;
        const float va = ds_bf(row[ia]), vb = ds_bf(row[ib]);
        row[ia] = ds_bf16(va * c - vb * s);
        row[ib] = ds_bf16(va * s + vb * c);
    }
}

int rope_rows_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &cs = a->t[1], &pos = a->t[2];
    const long nh = (long)r4d_geti_or(a, "n_head", 0), hd = (long)r4d_geti_or(a, "head_dim", 0);
    const long rot = (long)r4d_geti_or(a, "rot", 0), off = (long)r4d_geti_or(a, "off", -1);
    const long inv = (long)r4d_geti_or(a, "inverse", -1);
    const char* mode = r4d_gets(a, "mode");
    if (!mode) return R4D_NO(RAD_E_UNSUPPORTED);
    const bool gptj = !std::strcmp(mode, "gptj");
    if (!gptj && std::strcmp(mode, "neox")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (nh < 1 || hd < 2 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd ||
        (inv != 0 && inv != 1) || nh * hd > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || cs.dtype != RAD_F32 || pos.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (x.rank != 2 || x.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long M = x.shape[0];
    long ps;
    if (!mat(x, RAD_BF16, M, nh * hd) || cs.rank != 2 || cs.shape[1] != rot ||
        rad_tensor_numel(&pos) != M || !lin_step(pos, &ps))
        return R4D_NO(RAD_E_SHAPE);
    const long half = rot / 2, pairs = nh * half;
    const unsigned gxp = (unsigned)((pairs + 255) / 256);   /* lever WQ: y outside the dispatch window */
    return R4D_EMIT(stream, rope_rows_kernel, dim3(gxp, ds_dodge_gy(gxp, grid_rows(M), 8)),
                    dim3(256), 0, (uint16_t*)x.data, (long)x.stride[0], (const float*)cs.data,
                    (long)cs.stride[0], (long)cs.stride[1], (const int32_t*)pos.data, ps, M,
                    (long)cs.shape[0], (int)hd, (int)half, (int)off, (int)pairs, gptj ? 1 : 0,
                    inv ? -1.0f : 1.0f);
}

/* ==================================================================== quant_roundtrip
 * A block's amax, its scale and the round trip of each value, as ref_quant_roundtrip's
 * block_scale and roundtrip. A power-of-two block up to 32 is a lane group (one value a thread, the
 * max over the group's lanes); any other block is one thread a block. amax is a max, so both give
 * the same bits. The scalar rules (block_scale, roundtrip) are r4d_ds41_qrt.h's. */

template <int B>
__global__ __launch_bounds__(256) void quant_group_kernel(uint16_t* x, long xp, int n, long rows,
                                                          int fmt, int sc) { R4D_IKA_ENTRY();
    const int col = blockIdx.x * 256 + threadIdx.x;
    const bool in = col < n;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* row = x + r * xp;
        const float v = in ? ds_bf(row[col]) : 0.0f;
        float a = fabsf(v);
        a = a == a ? a : 0.0f;  // std::max(amax, NaN) keeps amax: a NaN never wins
        const float amax = r4d_xor_max<B>(a);
        if (in) row[col] = ds_bf16(roundtrip(v, block_scale(amax, fmt, sc), fmt));
    }
}

/* Lever QG8 (DRAFTER track, 18.11.2026 round 5): eight consecutive columns a thread (one 16-byte load and
 * store), a block of B = 8 / 16 / 32 over B / 8 lanes. A value's amax term is quant_group_kernel's (|v|, a NaN as
 * 0: never negative, never NaN), so the block's maximum is the same number whatever grouping forms it (the
 * thread's eight by fmaxf, then the lanes' xor maximum), and each value takes the same block_scale and
 * roundtrip: the same bits. Why: quant_group_kernel's one column a thread makes [M, 8192] at 16 rows 512
 * workgroups of one value a thread, which measured 22.9 us on the R9700 (xeon abench a4: [8, 8192] 24.2,
 * [32, 8192] 8.3, [4, 8192] 4.1) -- the drafter's o_q at its 16-row pass, 19-23 us a stage in serving (lq145). */
template <int B>
__global__ __launch_bounds__(256) void quant_group8_kernel(uint16_t* x, long xp, int n, long rows,
                                                           int fmt, int sc) { R4D_IKA_ENTRY();
    static_assert(B == 8 || B == 16 || B == 32, "a block of whole 8-column pieces");
    const int c0 = (blockIdx.x * 256 + threadIdx.x) * 8;
    const bool in = c0 < n;   /* n % B == 0: a block's lanes are all in or all out */
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* row = x + r * xp;
        uint4 w = in ? *(const uint4*)(row + c0) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t wv[4] = { w.x, w.y, w.z, w.w };
        float v[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[2 * i] = ds_bf((uint16_t)(wv[i] & 0xffffu));
            v[2 * i + 1] = ds_bf((uint16_t)(wv[i] >> 16));
        }
        float a = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) a = fmaxf(a, ds_amax_term(v[i]));
        const float amax = r4d_xor_max<B / 8>(a);
        const float s = block_scale(amax, fmt, sc);
        uint32_t o[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            o[i] = (uint32_t)ds_bf16(roundtrip(v[2 * i], s, fmt)) |
                   ((uint32_t)ds_bf16(roundtrip(v[2 * i + 1], s, fmt)) << 16);
        if (in) *(uint4*)(row + c0) = make_uint4(o[0], o[1], o[2], o[3]);
    }
}
/* QG8's bench pin (abench): -1 the rule (QG8 wherever it applies), 0 never. */
std::atomic<int> g_qg8_pin{-1};

__global__ __launch_bounds__(256) void quant_serial_kernel(uint16_t* x, long xp, int n, int block,
                                                           long rows, int fmt, int sc) { R4D_IKA_ENTRY();
    const int b = blockIdx.x * 256 + threadIdx.x;
    if (b >= n / block) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        uint16_t* p = x + r * xp + (long)b * block;
        float amax = 0;
        for (int i = 0; i < block; ++i) amax = ds_max(amax, fabsf(ds_bf(p[i])));
        const float s = block_scale(amax, fmt, sc);
        for (int i = 0; i < block; ++i) p[i] = ds_bf16(roundtrip(ds_bf(p[i]), s, fmt));
    }
}

int quant_rt_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 1 || !a->t[0].data) return RAD_E_INVAL;
    const RadTensor& x = a->t[0];
    const long n = (long)r4d_geti_or(a, "n", 0), block = (long)r4d_geti_or(a, "block", 0);
    const char* fs = r4d_gets(a, "format");
    const char* ss = r4d_gets(a, "scale");
    if (!fs || !ss) return R4D_NO(RAD_E_UNSUPPORTED);
    int fmt, sc;
    if (!std::strcmp(fs, "fp8_e4m3")) fmt = kFp8;
    else if (!std::strcmp(fs, "fp4_e2m1")) fmt = kFp4;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (!std::strcmp(ss, "ue8m0")) sc = kUe8m0;
    else if (!std::strcmp(ss, "e4m3") && fmt == kFp4) sc = kE4m3;
    else if (!std::strcmp(ss, "f32") && fmt == kFp8) sc = kScaleF32;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    if (n < 1 || block < 1 || n % block || n > INT_MAX || x.rank != 2 || x.shape[0] < 1 ||
        x.shape[1] != n)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long rows = x.shape[0];
    if (!mat(x, RAD_BF16, rows, n)) return R4D_NO(RAD_E_SHAPE);
    uint16_t* xd = (uint16_t*)x.data;
    const long xp = (long)x.stride[0];
    /* lever QG8: eight columns a thread where the rows are 16-byte pieces (the same bits), from 32 x 8192 values (abench a6:
     * [32 / 48 / 80 / 160, 8192] 8.2 / 9.2 / 11.3 / 13.3 us vs 8.4 / 10.1 / 13.6 / 18.7; below it the one-column kernel is faster) */
    if ((block == 8 || block == 16 || block == 32) && n % 8 == 0 && xp % 8 == 0 && rows * n >= 262144 &&
        ((uintptr_t)xd & 15u) == 0 && g_qg8_pin.load(std::memory_order_relaxed) != 0) {
        const unsigned gx8 = (unsigned)((n / 8 + 255) / 256);
        const dim3 g8(gx8, ds_dodge_gy(gx8, grid_rows(rows), 8));
        if (block == 8) return R4D_EMIT(stream, quant_group8_kernel<8>, g8, dim3(256), 0, xd, xp, (int)n, rows, fmt, sc);
        if (block == 16) return R4D_EMIT(stream, quant_group8_kernel<16>, g8, dim3(256), 0, xd, xp, (int)n, rows, fmt, sc);
        return R4D_EMIT(stream, quant_group8_kernel<32>, g8, dim3(256), 0, xd, xp, (int)n, rows, fmt, sc);
    }
    const unsigned gxq = (unsigned)((n + 255) / 256);
    const dim3 grid(gxq, ds_dodge_gy(gxq, grid_rows(rows), 8));   /* lever WQ */
#define QRT_GROUP(B)                                                                               \
    case B:                                                                                        \
        return R4D_EMIT(stream, quant_group_kernel<B>, grid, dim3(256), 0, xd, xp, (int)n, rows,   \
                        fmt, sc)
    switch (block) {
        QRT_GROUP(1);
        QRT_GROUP(2);
        QRT_GROUP(4);
        QRT_GROUP(8);
        QRT_GROUP(16);
        QRT_GROUP(32);
        default: break;
    }
#undef QRT_GROUP
    const long nb = n / block;
    return R4D_EMIT(stream, quant_serial_kernel, dim3((unsigned)((nb + 255) / 256), grid_rows(rows)),
                    dim3(256), 0, xd, xp, (int)n, (int)block, rows, fmt, sc);
}

/* ==================================================================== mhc_mix
 * The projection is mhc_pre's: hc*(hc+2) dots of the row with fn and its sum of squares. The row
 * splits into 1024-column chunks; in a chunk each sum is lane L's columns L, L+32, .. in order,
 * closed with the wave butterfly, and the chunk partials are added in chunk order. With scratch
 * the chunks are separate workgroups whose waves split the sums (mix_split_kernel), then a
 * finishing wave a row; without, one wave a row walks the chunks itself, for the same bits. The
 * finish is ref finish_row lane-parallel: element (j,k) of the Sinkhorn matrix on lane j*hc+k (and
 * +32 from hc 6), every row and column sum gathered by shuffles in the scalar loop's order. */

constexpr int kMixCols = 1024;

struct MixOut {
    const float* scale;
    long sst;
    const float* base;
    long bst;
    float* pre;
    long prp, prs;
    float* post;
    long pop, pos;
    float* comb;
    long cp, cs;
    int width, iters;
    float eps, heps;
};

template <int HC, int R>
__device__ __forceinline__ void mix_chunk(const uint16_t* h, long hp, const float* fn, long fp,
                                          int width, long r0, int nr, int chunk, int lane,
                                          float (&acc)[HC * (HC + 2) + 1][R]) {
    constexpr int MIX = HC * (HC + 2);
#pragma unroll
    for (int j = 0; j <= MIX; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[j][r] = 0.0f;
    const int c0 = chunk * kMixCols;
    for (int t = 0; t < kMixCols / 32; ++t) {
        const int col = c0 + t * 32 + lane;
        const bool in = col < width;
        float v[R];
#pragma unroll
        for (int r = 0; r < R; ++r) v[r] = in && r < nr ? ds_bf(h[(r0 + r) * hp + col]) : 0.0f;
#pragma unroll
        for (int j = 0; j < MIX; ++j) {
            const float w = in ? fn[(long)j * fp + col] : 0.0f;
#pragma unroll
            for (int r = 0; r < R; ++r) acc[j][r] += v[r] * w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[MIX][r] += v[r] * v[r];
    }
}

// The scales and biases a lane of mix_finish reads, loaded before the partials so their latencies
// overlap.
template <int HC>
struct MixPre {
    float s0, s1, s2, bpre, bpost, bc[(HC * HC + 31) / 32];
};
template <int HC>
__device__ __forceinline__ MixPre<HC> mix_pre(int lane, const MixOut& o) {
    constexpr int NN = HC * HC, NE = (NN + 31) / 32;
    MixPre<HC> p;
    p.s0 = o.scale[0];
    p.s1 = o.scale[o.sst];
    p.s2 = o.scale[2 * o.sst];
    const int jl = lane < HC ? lane : 0;
    p.bpre = o.base[(long)jl * o.bst];
    p.bpost = o.base[(long)(HC + jl) * o.bst];
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u;
        p.bc[u] = o.base[(long)(2 * HC + (e < NN ? e : 0)) * o.bst];
    }
    return p;
}

// d[q] holds sum j = lane + 32q (the hc*(hc+2) dots, then the sum of squares) in every lane that
// owns one. Every lane must enter: the gathers are shuffles.
template <int HC>
__device__ __forceinline__ void mix_finish(const float (&d)[(HC * (HC + 2) + 32) / 32], int lane,
                                           long row, const MixOut& o, const MixPre<HC>& pr) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32, NN = HC * HC, NE = (NN + 31) / 32;
    const float sq = __shfl(d[MIX / 32], MIX % 32, 32);
    const float inv = 1.0f / sqrtf(sq / (float)o.width + o.eps);
    float cf[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) cf[q] = d[q] * inv;
    auto coeff = [&](int j) {
        float r = 0.0f;
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float t = __shfl(cf[q], j & 31, 32);
            if ((j >> 5) == q) r = t;
        }
        return r;
    };
    const float s0 = pr.s0, s1 = pr.s1, s2 = pr.s2;
    const int jl = lane < HC ? lane : 0;
    const float cpre = coeff(jl), cpost = coeff(HC + jl);
    const float zpre = -(cpre * s0 + pr.bpre), z = cpost * s1 + pr.bpost;
    float c[NE];
    int rb[NE], ck[NE];
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u, ee = e < NN ? e : 0;
        rb[u] = ee / HC * HC;
        ck[u] = ee % HC;
        const float cq = coeff(2 * HC + ee);
        c[u] = e < NN ? cq * s2 + pr.bc[u] : 0.0f;
    }
    auto get = [&](int idx) {
        float r = 0.0f;
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            const float t = __shfl(c[u], idx & 31, 32);
            if ((idx >> 5) == u) r = t;
        }
        return r;
    };
    {
        float mx[NE], sum[NE];
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            mx[u] = -INFINITY;
#pragma unroll
            for (int k = 0; k < HC; ++k) mx[u] = ds_max(mx[u], get(rb[u] + k));
        }
        float ex[2 + NE];  // every lane's three exps in one batch (pre and post kept on hc lanes)
        ex[0] = zpre;
        ex[1] = -z;
#pragma unroll
        for (int u = 0; u < NE; ++u) ex[2 + u] = c[u] - mx[u];
        ds_exp_n(ex);
        if (lane < HC) {
            const float pj = 1.0f / (1.0f + ex[0]) + o.heps;
            o.post[row * o.pop + lane * o.pos] = 2.0f / (1.0f + ex[1]);
            o.pre[row * o.prp + lane * o.prs] = pj;
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] = ex[2 + u];
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            sum[u] = 0.0f;
#pragma unroll
            for (int k = 0; k < HC; ++k) sum[u] += get(rb[u] + k);
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] = c[u] / sum[u] + o.heps;
    }
    for (int it = 0; it < o.iters; ++it) {
        float sum[NE];
        if (it) {
#pragma unroll
            for (int u = 0; u < NE; ++u) {
                sum[u] = 0.0f;
#pragma unroll
                for (int k = 0; k < HC; ++k) sum[u] += get(rb[u] + k);
            }
#pragma unroll
            for (int u = 0; u < NE; ++u) c[u] /= sum[u] + o.heps;
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) {
            sum[u] = 0.0f;
#pragma unroll
            for (int j = 0; j < HC; ++j) sum[u] += get(j * HC + ck[u]);
        }
#pragma unroll
        for (int u = 0; u < NE; ++u) c[u] /= sum[u] + o.heps;
    }
#pragma unroll
    for (int u = 0; u < NE; ++u) {
        const int e = lane + 32 * u;
        if (e < NN) o.comb[row * o.cp + (long)e * o.cs] = c[u];
    }
}

// The same partials at decode heights: the WV waves of a workgroup take the same R rows of one
// chunk and split its hc*(hc+2) dots between them (wave w forms dots w, w+WV, ..; wave
// hc*(hc+2) % WV also the sum of squares), each walked as mix_chunk walks it (lane L the columns
// L, L+32, .. in order, each product added unfused), so a partial is the same sum in the same
// order whichever wave forms it, for any R, WV and row count. A wave's loads for TS column steps
// go out before their adds (one wave of a row's 24 loads a step, as a chunk a workgroup with the
// rows over its waves had it at decode heights, was ~1 us a step: 34 us of a 36 us call).
constexpr int kMixSteps = 8;

template <int HC>
__device__ __forceinline__ void mix_finish_row(const float* part, int chunks, const MixOut& o, long row,
                                               int lane);

/* FIN (mhc_mix's one-launch form): after a workgroup's partials for its row group are written, it
 * adds one to the group's counter (ctr, a split_ctr block of this (stream, output): zero, and zeroed
 * again by the last arriver); the workgroup that brings it to `chunks` finishes the group's rows,
 * a wave a row, with mix_finish_kernel's row (mix_finish_row): the same partials summed in the same
 * chunk order, so the finish kernel's bits, one ordered dispatch fewer. */
template <int HC, int R, int WV, int TS, bool FIN = false>
__global__ __launch_bounds__(32 * WV) void mix_split_kernel(const uint16_t* h, long hp,
                                                            const float* fn, long fp, float* part,
                                                            int width, int chunks, long rows,
                                                            unsigned* ctr = nullptr, MixOut fo = {}) { R4D_IKA_ENTRY();
    constexpr int MIX = HC * (HC + 2), S = MIX + 1, JW = (MIX + WV - 1) / WV;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, chunk = blockIdx.x;
    const int c0 = chunk * kMixCols;
    const float* wr[JW];  // a slot past the last dot walks row 0 and is dropped
#pragma unroll
    for (int q = 0; q < JW; ++q) {
        const int j = wave + WV * q;
        wr[q] = fn + (long)(j < MIX ? j : 0) * fp;
    }
    for (long g = blockIdx.y; g * R < rows; g += gridDim.y) {
        const long r0 = g * R;
        const int nr = (int)(rows - r0 < R ? rows - r0 : R);
        const uint16_t* hr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) hr[r] = h + (r0 + (r < nr ? r : 0)) * hp;
        float acc[JW + 1][R];  // [JW]: the sum of squares
#pragma unroll
        for (int q = 0; q <= JW; ++q)
#pragma unroll
            for (int r = 0; r < R; ++r) acc[q][r] = 0.0f;
        auto walk = [&](auto full, auto sq) {
            constexpr bool FULL = decltype(full)::value, SQ = decltype(sq)::value;
#pragma unroll 1  // unrolled, the loads of every step went up front and spilled
            for (int t0 = 0; t0 < kMixCols / 32; t0 += TS) {
                float v[TS][R], w[TS][JW];
#pragma unroll
                for (int u = 0; u < TS; ++u) {
                    const int col = c0 + (t0 + u) * 32 + lane;
                    const bool in = FULL || col < width;
                    const int cc = in ? col : 0;  // a valid address; the value is replaced by 0
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const float x = ds_bf(hr[r][cc]);
                        v[u][r] = in && r < nr ? x : 0.0f;
                    }
#pragma unroll
                    for (int q = 0; q < JW; ++q) {
                        const float x = wr[q][cc];
                        w[u][q] = in ? x : 0.0f;
                    }
                }
#pragma unroll
                for (int u = 0; u < TS; ++u) {
#pragma unroll
                    for (int q = 0; q < JW; ++q)
#pragma unroll
                        for (int r = 0; r < R; ++r) acc[q][r] += v[u][r] * w[u][q];
                    if (SQ)
#pragma unroll
                        for (int r = 0; r < R; ++r) acc[JW][r] += v[u][r] * v[u][r];
                }
            }
        };
        const bool full = c0 + kMixCols <= width, sq = wave == MIX % WV;
        if (full) {
            if (sq) walk(std::true_type{}, std::true_type{});
            else walk(std::true_type{}, std::false_type{});
        } else {
            if (sq) walk(std::false_type{}, std::true_type{});
            else walk(std::false_type{}, std::false_type{});
        }
#pragma unroll
        for (int q = 0; q <= JW; ++q) {
            const int j = q < JW ? wave + WV * q : MIX;
            if (q < JW ? j >= MIX : !sq) continue;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float s = r4d_wave_sum(acc[q][r]);
                if (!lane && r < nr) part[((r0 + r) * S + j) * chunks + chunk] = s;
            }
        }
        if constexpr (FIN) {
            __shared__ unsigned s_last;
            __threadfence();   // this workgroup's partials, device-wide, before its arrival
            __syncthreads();
            if (threadIdx.x == 0) {
                const unsigned old =
                    __hip_atomic_fetch_add(&ctr[g], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
                s_last = old + 1u == (unsigned)chunks ? 1u : 0u;
                if (s_last) __hip_atomic_store(&ctr[g], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            __syncthreads();
            if (s_last) {
                __threadfence();   // every other workgroup's partials: no stale line
                for (int r = wave; r < nr; r += WV) mix_finish_row<HC>(part, chunks, fo, r0 + r, lane);
            }
            __syncthreads();   // s_last is rewritten for the next row group
        }
    }
}

template <int HC>
__device__ __forceinline__ void mix_finish_row(const float* part, int chunks, const MixOut& o, long row,
                                               int lane) {
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32, CU = 32;
    const MixPre<HC> pr = mix_pre<HC>(lane, o);
    const bool vec = (chunks & 3) == 0 && ((uintptr_t)part & 15) == 0;
    float d[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const int j = lane + 32 * q;
        const float* p = part + (row * (MIX + 1) + (j <= MIX ? j : 0)) * chunks;
        float s = 0.0f;
        if (vec) {  // CU chunks a batch as CU/4 16-byte loads, then their adds in chunk order
            for (int c = 0; c < chunks; c += CU) {
                float4 t[CU / 4];
#pragma unroll
                for (int u = 0; u < CU / 4; ++u)
                    t[u] = reinterpret_cast<const float4*>(p + c)[c + 4 * u < chunks ? u : 0];
#pragma unroll
                for (int u = 0; u < CU / 4; ++u) {
                    const float e[4] = { t[u].x, t[u].y, t[u].z, t[u].w };
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        if (c + 4 * u + k < chunks) s += e[k];
                }
            }
        } else {
            for (int c = 0; c < chunks; c += CU) {
                float t[CU];
#pragma unroll
                for (int u = 0; u < CU; ++u) t[u] = p[c + u < chunks ? c + u : c];
#pragma unroll
                for (int u = 0; u < CU; ++u)
                    if (c + u < chunks) s += t[u];
            }
        }
        d[q] = j <= MIX ? s : 0.0f;
    }
    mix_finish<HC>(d, lane, row, o, pr);
}

template <int HC>
__global__ __launch_bounds__(128) void mix_finish_kernel(const float* part, int chunks, MixOut o,
                                                         long rows) { R4D_IKA_ENTRY_X();
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    mix_finish_row<HC>(part, chunks, o, row, lane);
}

// Without scratch: one wave a row, the chunks in order, the same partials and the same sums.
template <int HC>
__global__ __launch_bounds__(128) void mix_fused_kernel(const uint16_t* h, long hp, const float* fn,
                                                        long fp, int chunks, MixOut o, long rows) { R4D_IKA_ENTRY_X();
    constexpr int MIX = HC * (HC + 2), Q = (MIX + 32) / 32;
    const int lane = threadIdx.x & 31;
    const long row = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= rows) return;
    const MixPre<HC> pr = mix_pre<HC>(lane, o);
    float tot[MIX + 1];
#pragma unroll
    for (int j = 0; j <= MIX; ++j) tot[j] = 0.0f;
    for (int c = 0; c < chunks; ++c) {
        float acc[MIX + 1][1];
        mix_chunk<HC, 1>(h, hp, fn, fp, o.width, row, 1, c, lane, acc);
#pragma unroll
        for (int j = 0; j <= MIX; ++j) tot[j] += r4d_wave_sum(acc[j][0]);
    }
    float d[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        d[q] = 0.0f;
#pragma unroll
        for (int j = 0; j <= MIX; ++j)
            if (j == lane + 32 * q) d[q] = tot[j];
    }
    mix_finish<HC>(d, lane, row, o, pr);
}

long mix_chunks(long width) { return (width + kMixCols - 1) / kMixCols; }

/* mhc_mix's one-launch form (mix_split_kernel FIN). R4D_DS41_MIX_FUSE: "1" = on; unset or anything
 * else = the split and finish kernels (the same bits). Default off since 10.11. (gpu02 item 77: with
 * the router fusion c4 -1.0..-1.4 %: rejected, the switch kept). r4d_ds41_mix_fuse_force pins it for
 * tests (-1 = the environment's). */
constexpr long kDsMixCtr = 8192;   // split_ctr's counters a block (r4d_gemm_fp8a8.hip kMaxCtr)
std::atomic<int> g_mix_fuse_force{-1};
bool ds_mix_fuse_on() {
    const int f = g_mix_fuse_force.load(std::memory_order_relaxed);
    if (f >= 0) return f != 0;
    static const bool on = [] { const char* e = std::getenv("R4D_DS41_MIX_FUSE"); return e && e[0] == '1' && !e[1]; }();
    return on;
}
/* Lever FUSE-M (ds41-fuse, 13.11.2026; zzzaql's AQL-stamped c8 on int14: dep mix_split_kernel -> mix_finish_kernel 346 us a
 * decode step, ~80 calls): libr4d's fn4 switch "ds41_mix_fin" 0 / 1 -- 1 = a launch of at most kMixFinRows rows takes mhc_mix's
 * one-launch FIN form (mix_split_kernel<.., FIN>: the last workgroup of each row group runs mix_finish_kernel's row,
 * mix_finish_row, over the same partials in the same chunk order -- the finish kernel's bits, one launch fewer; the R 1 split
 * at WV 8, so not with ds41_mix_wv's wide forms, which keep their two launches). 0 = today's two launches (R4D_DS41_MIX_FUSE=1
 * still takes FIN at every height, as before). RADIANCE_DS41_MIX_FIN=1 at load starts it on. Counters (rad_fuse_stats, every
 * rank's launches summed; a recorded pass counts when it is recorded): FIN launches the switch took and their rows. */
constexpr long kMixFinRows = 48;
std::atomic<int> g_mix_fin{-1};
std::atomic<long long> g_mf_launch{0}, g_mf_rows{0};
int mix_fin_now() {
    int v = g_mix_fin.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_DS41_MIX_FIN");
        int unread = -1;
        g_mix_fin.compare_exchange_strong(unread, e && e[0] == '1' && !e[1] ? 1 : 0);
        v = g_mix_fin.load(std::memory_order_relaxed);
    }
    return v;
}
/* Lever 1U-MX (ds41-int11): libr4d's fn4 switch "ds41_mix_wv" (rad_fn4_* in r4d_ds41_ggml.hip) -- the waves a chunk of the
 * split launch at decode heights (rows <= 8, R 1), 0 = the stock 8; 12 / 16 / 24 that many (hc 4 only: 24 dots, so two /
 * two / one a wave). The start value is RADIANCE_DS41_MIX_WV (12 / 16 / 24; anything else or unset: 0), read at the first
 * launch. Exact by mix_split_kernel's note: a partial (row, chunk, dot) is lane L's columns L, L + 32, .. from zero with
 * unfused adds and the same r4d_wave_sum whichever wave forms it, the sum of squares likewise on wave MIX % WV, and the
 * finish reads the same partials. Counters (rad_mx_stats, every rank's launches summed; a recorded pass counts when it is
 * recorded): the launches taken in a wide form and their rows. */
/* tp8 loop X8 step 1 (DECODE round 19): the hundreds of ds41_mix_wv (1 .. 5) pick the 9 .. 64-row split launch's form -- rows a
 * wave, waves a chunk, column steps a load batch -- beside the units (0 / 12 / 16 / 24, the <= 8-row form as before): 1 = 16 rows
 * a wave (WV 8, TS 4), 2 = WV 12, 3 = WV 24 (8 rows, TS 4), 4 = 4 rows a wave (WV 8, TS 4), 5 = TS 8 (8 rows, WV 8); 0 = today's
 * (8 rows, WV 8, TS 4 above 24 rows; 4 rows, TS 4 at 9 .. 24). The same partials by mix_split_kernel's note (any R, WV, TS and
 * row count), the finish unchanged; not the FIN form. */
bool mix_wv_known(int v) {
    const int u = v % 100, hv = v / 100;
    return v >= 0 && hv <= 5 && (u == 0 || u == 12 || u == 16 || u == 24);
}
std::atomic<int> g_mix_wv{-1};
std::atomic<long long> g_mx_launch{0}, g_mx_rows{0};
int mix_wv_now() {
    int v = g_mix_wv.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("RADIANCE_DS41_MIX_WV");
        const int ev = e ? (int)std::strtol(e, nullptr, 10) : 0;
        int unread = -1;
        g_mix_wv.compare_exchange_strong(unread, mix_wv_known(ev) ? ev : 0);
        v = g_mix_wv.load(std::memory_order_relaxed);
    }
    return v;
}

#ifdef R4D_EMU_HOST
/* tests/r4d_emu: device memory is host memory; a zeroed block a (stream, output), as split_ctr */
unsigned* ds_mix_ctr(RadStream s, const void* out) {
    static std::mutex m;
    static std::map<std::pair<RadStream, const void*>, unsigned*> blocks;
    std::lock_guard<std::mutex> lk(m);
    unsigned*& b = blocks[std::make_pair(s, out)];
    if (!b) b = (unsigned*)std::calloc((size_t)kDsMixCtr, sizeof(unsigned));
    return b;
}
#else
unsigned* ds_mix_ctr(RadStream s, const void* out) { return r4d_8a8_split_ctr(s, out); }
#endif


template <int HC>
int mix_emit(RadStream stream, const RadTensor& h, const RadTensor& fn, const MixOut& o, long rows,
             void* scratch, int64_t scratch_bytes) {
    constexpr int MIX = HC * (HC + 2);
    const long chunks = mix_chunks(o.width);
    const int64_t need = (int64_t)rows * chunks * (MIX + 1) * (int64_t)sizeof(float);
    const unsigned fin = (unsigned)((rows + 3) / 4);
    if (scratch && scratch_bytes >= need) {
        float* part = (float*)scratch;
        // Rows a wave keeps (its sums' registers), and the fastest split measured per height on
        // gfx1201 (n 5120, hc 4): one row up to 8, four rows 4 steps up to 24, then eight.
        constexpr int RB = HC <= 4 ? 8 : HC <= 6 ? 4 : 2, RM = RB < 4 ? RB : 4;
        // FIN: the finish in the split launch's last workgroup of each row group (its counters a
        // split_ctr block of this stream and `pre`; a grid past their count, or no block, takes the
        // finish kernel -- the same bits)
        const long RSel = rows <= 8 ? 1 : rows <= 24 ? RM : RB;
        const long groups = (rows + RSel - 1) / RSel, gy = grid_rows(groups);
        /* lever FUSE-M: ds41_mix_fin 1 takes FIN at decode heights (rows <= kMixFinRows) where mix_wv's wide form is not
         * asked for */
        const bool fin_sw = !ds_mix_fuse_on() && rows <= kMixFinRows && mix_fin_now() > 0 &&
                            !(HC == 4 && rows <= 8 && mix_wv_now() != 0);
        unsigned* const ctr = (ds_mix_fuse_on() || fin_sw) && gy <= (long)kDsMixCtr ? ds_mix_ctr(stream, o.pre) : nullptr;
#define MIX_SPLIT(RS, TS)                                                                          \
    (ctr ? R4D_EMIT(stream, (mix_split_kernel<HC, RS, 8, TS, true>),                               \
                    dim3((unsigned)chunks, grid_rows((rows + (RS) - 1) / (RS))), dim3(256), 0,     \
                    (const uint16_t*)h.data, (long)h.stride[0], (const float*)fn.data,             \
                    (long)fn.stride[0], part, o.width, (int)chunks, rows, ctr, o)                  \
         : R4D_EMIT(stream, (mix_split_kernel<HC, RS, 8, TS>),                                     \
                    dim3((unsigned)chunks, grid_rows((rows + (RS) - 1) / (RS))), dim3(256), 0,     \
                    (const uint16_t*)h.data, (long)h.stride[0], (const float*)fn.data,             \
                    (long)fn.stride[0], part, o.width, (int)chunks, rows, (unsigned*)nullptr, o))
        /* lever 1U-MX: the decode heights' launch with more waves a chunk (the same partials; not the FIN form) */
        const int wv = HC == 4 && rows <= 8 && !ctr ? mix_wv_now() % 100 : 0;
        /* tp8 loop X8 step 1: the 9 .. 64-row launch's form by ds41_mix_wv's hundreds (the same partials; not the FIN form) */
        const int xv = HC == 4 && rows > 8 && rows <= 64 && !ctr ? mix_wv_now() / 100 : 0;
#define MIX_X8(RS, WV, TS)                                                                                     \
    R4D_EMIT(stream, (mix_split_kernel<HC, RS, WV, TS>), dim3((unsigned)chunks, grid_rows((rows + (RS) - 1) / (RS))), \
             dim3(32 * (WV)), 0, (const uint16_t*)h.data, (long)h.stride[0], (const float*)fn.data,           \
             (long)fn.stride[0], part, o.width, (int)chunks, rows, (unsigned*)nullptr, o)
#define MIX_WIDE(WV)                                                                                       \
    R4D_EMIT(stream, (mix_split_kernel<HC, 1, WV, kMixSteps>), dim3((unsigned)chunks, grid_rows(rows)),    \
             dim3(32 * (WV)), 0, (const uint16_t*)h.data, (long)h.stride[0], (const float*)fn.data,       \
             (long)fn.stride[0], part, o.width, (int)chunks, rows, (unsigned*)nullptr, o)
        int rc;
        if (wv == 12 || wv == 16 || wv == 24) {
            rc = wv == 12 ? MIX_WIDE(12) : wv == 16 ? MIX_WIDE(16) : MIX_WIDE(24);
            if (rc >= 0) {
                g_mx_launch.fetch_add(1, std::memory_order_relaxed);
                g_mx_rows.fetch_add(rows, std::memory_order_relaxed);
            }
        } else if (xv >= 1 && xv <= 5) {
            if constexpr (HC == 4) {
                rc = xv == 1 ? MIX_X8(16, 8, 4) : xv == 2 ? MIX_X8(8, 12, 4) : xv == 3 ? MIX_X8(8, 24, 4)
                   : xv == 4 ? MIX_X8(4, 8, 4) : MIX_X8(8, 8, 8);
            } else {
                rc = -1;
            }
            if (rc >= 0) {
                g_mx_launch.fetch_add(1, std::memory_order_relaxed);
                g_mx_rows.fetch_add(rows, std::memory_order_relaxed);
            }
        } else {
            rc = rows <= 8 ? MIX_SPLIT(1, kMixSteps)
                 : rows <= 24 ? MIX_SPLIT(RM, 4)
                              : MIX_SPLIT(RB, 4);
        }
#undef MIX_X8
#undef MIX_WIDE
#undef MIX_SPLIT
        if (rc >= 0 && ctr && fin_sw) {
            g_mf_launch.fetch_add(1, std::memory_order_relaxed);
            g_mf_rows.fetch_add(rows, std::memory_order_relaxed);
        }
        if (rc < 0 || ctr) return rc;
        return R4D_EMIT(stream, mix_finish_kernel<HC>, dim3(fin), dim3(128), 0, (const float*)part,
                        (int)chunks, o, rows);
    }
    return R4D_EMIT(stream, mix_fused_kernel<HC>, dim3(fin), dim3(128), 0, (const uint16_t*)h.data,
                    (long)h.stride[0], (const float*)fn.data, (long)fn.stride[0], (int)chunks, o,
                    rows);
}

bool mix_params(const RadArgs* a, long* n, long* hc, long* iters, float* eps, float* heps) {
    *n = (long)r4d_geti_or(a, "n", 0);
    *hc = (long)r4d_geti_or(a, "hc", 0);
    *iters = (long)r4d_geti_or(a, "iterations", 0);
    *eps = (float)r4d_getf(a, "eps", 0);
    *heps = (float)r4d_getf(a, "hc_eps", 0);
    return *n > 0 && *hc >= 2 && *hc <= 8 && *iters >= 1 && *iters <= 256 &&
           *hc * *n <= (1L << 30) && std::isfinite(*eps) && std::isfinite(*heps) && *eps > 0 &&
           *heps >= 0;
}

int mhc_mix_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    long n, hc, iters;
    float eps, heps;
    if (!mix_params(a, &n, &hc, &iters, &eps, &heps)) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &fn = a->t[1], &sc = a->t[2], &bs = a->t[3];
    const RadTensor &pre = a->t[4], &post = a->t[5], &comb = a->t[6];
    const long width = hc * n, mix = hc * (hc + 2);
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || fn.dtype != RAD_F32 || sc.dtype != RAD_F32 || bs.dtype != RAD_F32 ||
        pre.dtype != RAD_F32 || post.dtype != RAD_F32 || comb.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, width) || !mat(fn, RAD_F32, mix, width) || !sc.data ||
        sc.rank != 1 || sc.shape[0] != 3 || !bs.data || bs.rank != 1 || bs.shape[0] != mix ||
        !mat_s(pre, RAD_F32, rows, hc) || !mat_s(post, RAD_F32, rows, hc) ||
        !mat_s(comb, RAD_F32, rows, hc * hc))
        return R4D_NO(RAD_E_SHAPE);
    MixOut o;
    o.scale = (const float*)sc.data;
    o.sst = (long)sc.stride[0];
    o.base = (const float*)bs.data;
    o.bst = (long)bs.stride[0];
    o.pre = (float*)pre.data;
    o.prp = (long)pre.stride[0];
    o.prs = (long)pre.stride[1];
    o.post = (float*)post.data;
    o.pop = (long)post.stride[0];
    o.pos = (long)post.stride[1];
    o.comb = (float*)comb.data;
    o.cp = (long)comb.stride[0];
    o.cs = (long)comb.stride[1];
    o.width = (int)width;
    o.iters = (int)iters;
    o.eps = eps;
    o.heps = heps;
    switch (hc) {
        case 2: return mix_emit<2>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 3: return mix_emit<3>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 4: return mix_emit<4>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 5: return mix_emit<5>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 6: return mix_emit<6>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 7: return mix_emit<7>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        case 8: return mix_emit<8>(stream, h, fn, o, rows, a->scratch, a->scratch_bytes);
        default: return R4D_NO(RAD_E_SHAPE);
    }
}

/* ==================================================================== mhc_collapse
 * One column a thread (eight with 16-byte rows), the streams summed from zero in order. */

__global__ __launch_bounds__(256) void collapse_kernel(const uint16_t* h, long hp, const float* pre,
                                                       long pp, long ps, uint16_t* x, long xp,
                                                       int n, int hc, long rows) { R4D_IKA_ENTRY();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        float v = 0;
        for (int j = 0; j < hc; ++j) v += pre[r * pp + j * ps] * ds_bf(h[r * hp + (long)j * n + i]);
        x[r * xp + i] = ds_bf16(v);
    }
}

__global__ __launch_bounds__(256) void collapse_vec_kernel(const uint16_t* h, long hp,
                                                           const float* pre, long pp, long ps,
                                                           uint16_t* x, long xp, int n, int hc,
                                                           long rows) { R4D_IKA_ENTRY();
    const int i = (blockIdx.x * 256 + threadIdx.x) * 8;
    if (i >= n) return;
    for (long r = blockIdx.y; r < rows; r += gridDim.y) {
        float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < hc; ++j) {
            const float pj = pre[r * pp + j * ps];
            const uint4 q = *reinterpret_cast<const uint4*>(h + r * hp + (long)j * n + i);
            const uint32_t w[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int e = 0; e < 8; ++e) v[e] += pj * ds_bf((uint16_t)(w[e >> 1] >> (16 * (e & 1))));
        }
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            o[e] = (uint32_t)ds_bf16(v[2 * e]) | ((uint32_t)ds_bf16(v[2 * e + 1]) << 16);
        *reinterpret_cast<uint4*>(x + r * xp + i) = uint4{ o[0], o[1], o[2], o[3] };
    }
}

int mhc_collapse_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8 || hc * n > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &pre = a->t[1], &x = a->t[2];
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat_s(pre, RAD_F32, rows, hc) ||
        !mat(x, RAD_BF16, rows, n))
        return R4D_NO(RAD_E_SHAPE);
    const auto al16 = [](const RadTensor& t) {
        return ((uintptr_t)t.data & 15) == 0 && t.stride[0] % 8 == 0;
    };
    if (n % 8 == 0 && al16(h) && al16(x))
        return R4D_EMIT(stream, collapse_vec_kernel,
                        dim3((unsigned)((n / 8 + 255) / 256), grid_rows(rows)), dim3(256), 0,
                        (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,
                        (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data,
                        (long)x.stride[0], (int)n, (int)hc, rows);
    return R4D_EMIT(stream, collapse_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(rows)),
                    dim3(256), 0, (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,
                    (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data, (long)x.stride[0],
                    (int)n, (int)hc, rows);
}

/* ==================================================================== softmax_pool
 * One thread a (entry, channel): the max, the exp sum and the weighted sum over the entry's rows,
 * each in row order; the exp is recomputed for the third pass (the same value). */

__global__ __launch_bounds__(256) void softmax_pool_kernel(const float* st, long sp, uint16_t* out,
                                                           long op, const int32_t* page, long pgs,
                                                           int n, int ratio, long W) { R4D_IKA_ENTRY();
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= n) return;
    for (long w = blockIdx.y; w < W; w += gridDim.y) {
        if (page && page[w * pgs] < 0) continue;
        const float* b = st + w * ratio * sp;
        float mx = -INFINITY;
        for (int i = 0; i < ratio; ++i) mx = ds_max(mx, b[i * sp + n + c]);
        float den = 0;
        for (int i = 0; i < ratio; ++i) den += ds_exp(b[i * sp + n + c] - mx);
        float acc = 0;
        for (int i = 0; i < ratio; ++i) acc += ds_exp(b[i * sp + n + c] - mx) / den * b[i * sp + c];
        out[w * op + c] = ds_bf16(acc);
    }
}

int softmax_pool_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 2 || !a->t[0].data || !a->t[1].data) return RAD_E_INVAL;
    const RadTensor &st = a->t[0], &out = a->t[1];
    const RadTensor* page = a->n_t > 2 && a->t[2].data ? &a->t[2] : nullptr;
    const long n = (long)r4d_geti_or(a, "n", 0), ratio = (long)r4d_geti_or(a, "ratio", 0);
    if (n < 1 || n > INT_MAX / 2 || ratio < 1 || ratio > 64 || st.rank != 2 || out.rank != 2 ||
        out.shape[0] < 1 || out.shape[1] != n || st.shape[0] != out.shape[0] * ratio ||
        st.shape[1] != 2 * n || (page && rad_tensor_numel(page) < out.shape[0]))
        return R4D_NO(RAD_E_SHAPE);
    if (st.dtype != RAD_F32 || out.dtype != RAD_BF16 || (page && page->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long W = out.shape[0];
    long pgs = 0;
    if (!mat(st, RAD_F32, W * ratio, 2 * n) || !mat(out, RAD_BF16, W, n) ||
        (page && !lin_step(*page, &pgs)))
        return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, softmax_pool_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(W)),
                    dim3(256), 0, (const float*)st.data, (long)st.stride[0], (uint16_t*)out.data,
                    (long)out.stride[0], page ? (const int32_t*)page->data : nullptr, pgs, (int)n,
                    (int)ratio, W);
}

/* ==================================================================== entry_store
 * One thread an element; a block's n values land at pg * n as libref writes them. */

__global__ __launch_bounds__(256) void entry_store_kernel(const uint16_t* x, long xp,
                                                          const int32_t* page, long pgs,
                                                          uint16_t* cache, long blocks, int n,
                                                          long W) { R4D_IKA_ENTRY();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    for (long w = blockIdx.y; w < W; w += gridDim.y) {
        const long pg = page[w * pgs];
        if (pg < 0 || pg >= blocks) continue;
        cache[pg * n + i] = ds_bf16(ds_bf(x[w * xp + i]));
    }
}

extern "C" int r4d_ds41_entry_store_mx4(const RadArgs* a, RadStream stream);   /* r4d_ds41_sparse.hip */
extern "C" int r4d_ds41_entry_store_nv4(const RadArgs* a, RadStream stream);   /* r4d_ds41_sparse.hip */

int entry_store_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    /* format mxfp4: the packed index-key form (r4d_ds41_sparse.hip); bf16 or absent: as stored below */
    if (const char* f = r4d_gets(a, "format")) {
        if (!std::strcmp(f, "mxfp4")) return r4d_ds41_entry_store_mx4(a, stream);
        if (!std::strcmp(f, "nvfp4")) return r4d_ds41_entry_store_nv4(a, stream);
        if (std::strcmp(f, "bf16")) return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const RadTensor &x = a->t[0], &page = a->t[1], &cache = a->t[2];
    const long n = (long)r4d_geti_or(a, "n", 0), bs = (long)r4d_geti_or(a, "block_size", 0);
    if (n < 1 || n > INT_MAX || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        rad_tensor_numel(&page) < x.shape[0] || cache.rank != 4 || cache.shape[2] != bs ||
        cache.shape[1] * cache.shape[2] * cache.shape[3] != n)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || page.dtype != RAD_I32 || cache.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long W = x.shape[0];
    long pgs;
    if (!mat(x, RAD_BF16, W, n) || !lin_step(page, &pgs)) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, entry_store_kernel, dim3((unsigned)((n + 255) / 256), grid_rows(W)),
                    dim3(256), 0, (const uint16_t*)x.data, (long)x.stride[0],
                    (const int32_t*)page.data, pgs, (uint16_t*)cache.data, (long)cache.shape[0],
                    (int)n, W);
}

/* ==================================================================== window_store (lever KVWIN)
 * x[m] (n values, n a multiple of 32) packed into the fp8 window slot (r4d_ds41_qrt.h ds_w8_*) at
 * slot_mapping[m] of a paged cache [n_blocks, 1, block_size, ds_w8_bytes(n) / 2] -- kv_store's slot rule: a
 * negative slot or s / block_size >= n_blocks stores nothing. 256 threads a row, a wave a block of 32
 * columns (its scale from the wave's amax), a byte a thread. */
__global__ __launch_bounds__(256) void window_store_kernel(const uint16_t* x, long xp, const int32_t* slot,
                                                           long ss, unsigned char* cache, long bs,
                                                           long n_blocks, int n, long M) { R4D_IKA_ENTRY();
    const int sb = ds_w8_bytes(n);
    for (long m = blockIdx.x; m < M; m += gridDim.x) {
        const long s = slot[m * ss];
        if (s < 0 || s / bs >= n_blocks) continue;    /* block-uniform */
        unsigned char* e = cache + s * (long)sb;
        for (int base = 0; base < n; base += 256) {   /* n % 32 == 0: a wave's lanes are all in or all out */
            const int col = base + (int)threadIdx.x;
            const bool in = col < n;
            const float v = in ? ds_bf(x[m * xp + col]) : 0.0f;
            const uint32_t sc = ds_w8_scale(r4d_xor_max<32>(ds_amax_term(v)));
            if (in) {
                e[col] = (unsigned char)ds_w8_code(v, sc);
                if ((col & 31) == 0) e[n + col / 32] = (unsigned char)sc;
            }
        }
        for (int z = n + n / 32 + (int)threadIdx.x; z < sb; z += 256) e[z] = 0;
    }
}

int window_store_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const char* f = r4d_gets(a, "format");
    if (!f || std::strcmp(f, "fp8_e4m3")) return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &x = a->t[0], &sm = a->t[1], &kc = a->t[2];
    const long n = (long)r4d_geti_or(a, "n", 0), bs = (long)r4d_geti_or(a, "block_size", 0);
    if (n < 32 || n % 32 || n > 16384 || bs < 1 || x.rank != 2 || x.shape[0] < 1 || x.shape[1] != n ||
        rad_tensor_numel(&sm) < x.shape[0] || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[2] != bs ||
        kc.shape[3] * 2 != ds_w8_bytes((int)n) || kc.shape[0] < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_BF16 || sm.dtype != RAD_I32 || kc.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(&kc)) return R4D_NO(RAD_E_STRIDE);
    const long M = x.shape[0];
    long ss;
    if (!mat(x, RAD_BF16, M, n) || !lin_step(sm, &ss)) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, window_store_kernel, dim3(grid_rows(M)), dim3(256), 0, (const uint16_t*)x.data,
                    (long)x.stride[0], (const int32_t*)sm.data, ss, (unsigned char*)kc.data, bs,
                    (long)kc.shape[0], (int)n, M);
}

/* ==================================================================== router_scored_topk
 * One wave a row, as r4d_glm.hip's router_sigmoid_topk: lane l holds experts l, l+32, ...; the
 * score is libref's (sigmoid, sqrt(softplus) or the softmax with its sum in expert order), round j
 * takes the largest (rank, ~expert) key below round j-1's and parks it on lane j, so equal ranks go
 * to the lower expert as libref's strict `>` scan takes them. A NaN or -inf rank is never taken.
 * The weights' sum runs in round order. */

enum { kSigmoid = 0, kSqrtSoftplus = 1, kSoftmax = 2 };

__device__ __forceinline__ unsigned long long ds_rank_key(float rank, int e) {
    uint32_t u = __float_as_uint(rank);
    if (u == 0x80000000u) u = 0;
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return rank > -INFINITY ? ((unsigned long long)k << 32) | (uint32_t)~e : 0ull;
}
__device__ __forceinline__ unsigned long long ds_wave_max_u64(unsigned long long b) {
    unsigned long long o;
    o = r4d_xor_lane_u64<16>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<8>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<4>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<2>(b);
    b = b > o ? b : o;
    o = r4d_xor_lane_u64<1>(b);
    b = b > o ? b : o;
    return b;
}

/* One row of router_scored_topk by one wave: writes the row's ids and weights and returns the
 * expert lane j < k wrote (the expert 0 of an empty slot, as `ids` holds it). Shared with
 * router_scored_topk_scatter, so the fused selection is this function's bits. */
template <int OWN, int SCORE>
__device__ __forceinline__ int router_scored_row(const float* logits, long lp, const float* bias,
                                                 long bp, int32_t* ids, long ip, float* weights,
                                                 long wp, long r, int ne, int k, int norm,
                                                 float factor, int lane) {
    float p[OWN], bv[OWN];  // every load first (a clamped index past ne, its value dropped)
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, ec = e < ne ? e : 0;
        const float x = logits[r * lp + ec];
        bv[i] = bias[(long)ec * bp];
        p[i] = e < ne ? x : 0.0f;
    }
    if (SCORE == kSoftmax) {
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            if (lane + 32 * i < ne) mx = fmaxf(mx, p[i]);
        mx = r4d_wave_max(mx);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = lane + 32 * i < ne ? ds_exp(p[i] - mx) : 0.0f;
        float den = 0.0f;
#pragma unroll
        for (int i = 0; i < OWN; ++i)
            for (int l = 0; l < 32; ++l)
                if (32 * i + l < ne)
                    den += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(p[i]), l));
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = p[i] / den;
    } else if (SCORE == kSigmoid) {
        float t[OWN];
#pragma unroll
        for (int i = 0; i < OWN; ++i) t[i] = -p[i];
        ds_exp_n(t);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = 1.0f / (1.0f + t[i]);
    } else {
        float t[OWN];
#pragma unroll
        for (int i = 0; i < OWN; ++i) t[i] = p[i];
        ds_exp_n(t);
        ds_log1p_n(t);
#pragma unroll
        for (int i = 0; i < OWN; ++i) p[i] = sqrtf(p[i] > 20.0f ? p[i] : t[i]);
    }
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < ne ? ds_rank_key(p[i] + bv[i], e) : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = ds_wave_max_u64(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[r * ip + lane] = e;
        weights[r * wp + lane] = chosen * mult;
    }
    return e;
}

template <int OWN, int SCORE>
__global__ __launch_bounds__(128) void router_scored_kernel(const float* logits, long lp,
                                                            const float* bias, long bp, int32_t* ids,
                                                            long ip, float* weights, long wp,
                                                            long m, int ne, int k, int norm,
                                                            float factor) { R4D_IKA_ENTRY();
    const int lane = threadIdx.x & 31;
    const long r = (long)blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    (void)router_scored_row<OWN, SCORE>(logits, lp, bias, bp, ids, ip, weights, wp, r, ne, k, norm,
                                        factor, lane);
}

/* router_scored_topk, then moe_scatter over its ids, then the `fill` of an ungated shared arm's
 * gate with 1, in one workgroup (router_sigmoid_topk_scatter's form, r4d_glm.hip, with the score
 * named). Rows wrap over the NW waves; each row is router_scored_row's, and its slots go to the
 * sort's LDS list instead of being read back from `ids`. A slot carries the expert its row wrote,
 * including the expert 0 of an empty slot, so every output equals the separate launches'. INV also
 * writes the inverse, as moe_scatter_inv does. */
template <int OWN, int SCORE, uint32_t NW, bool INV>
__global__ void __launch_bounds__(NW * 32)
    router_scored_scatter_kernel(const float* logits, long lp, const float* bias, long bp,
                                 int32_t* ids, long ip, float* weights, long wp, int32_t* sorted,
                                 int32_t* off, int32_t* cnt, uint16_t* gate, long gp, int32_t* inv,
                                 int m, int ne, int n_local, int k, int norm, float factor, int base,
                                 r4d_moesort::Protect prot) { R4D_IKA_ENTRY();
    __shared__ r4d_moesort::ScatterLds<NW> sl;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int e = threadIdx.x; e < n_local; e += (int)(NW * 32)) sl.cnt[e] = 0;
    for (int r = wave; r < m; r += (int)NW) {
        const int e = router_scored_row<OWN, SCORE>(logits, lp, bias, bp, ids, ip, weights, wp, r,
                                                    ne, k, norm, factor, lane) -
                      base;
        if (lane < k)
            sl.e[r * k + lane] =
                e >= 0 && e < n_local ? r4d_moesort::sort_place(e, (uint32_t)n_local, prot) : -1;
        if (gate && !lane) gate[(long)r * gp] = f32_to_bf16(1.0f);
    }
    lds_barrier();
    r4d_moesort::scatter_small_place<NW, INV>(sorted, off, cnt, (uint32_t)(m * k),
                                              (uint32_t)n_local, sl, inv);
}

template <int SCORE>
int router_emit(RadStream stream, const RadTensor& lg, const RadTensor& bias, const RadTensor& ids,
                const RadTensor& w, int ne, int k, int norm, float factor) {
    const long m = lg.shape[0];
    const dim3 grid((unsigned)((m + 3) / 4));
#define ROUTER(OWN)                                                                                \
    return R4D_EMIT(stream, (router_scored_kernel<OWN, SCORE>), grid, dim3(128), 0,                \
                    (const float*)lg.data, (long)lg.stride[0], (const float*)bias.data,            \
                    (long)bias.stride[0], (int32_t*)ids.data, (long)ids.stride[0],                 \
                    (float*)w.data, (long)w.stride[0], m, ne, k, norm, factor)
    if (ne <= 128) ROUTER(4);
    if (ne <= 256) ROUTER(8);
    if (ne <= 384) ROUTER(12);
    ROUTER(16);
#undef ROUTER
}

int router_scored_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return RAD_E_INVAL;
    const char* sc = r4d_gets(a, "score");
    if (!sc) return R4D_NO(RAD_E_UNSUPPORTED);
    int score;
    if (!std::strcmp(sc, "sigmoid")) score = kSigmoid;
    else if (!std::strcmp(sc, "sqrtsoftplus")) score = kSqrtSoftplus;
    else if (!std::strcmp(sc, "softmax")) score = kSoftmax;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = (long)r4d_geti_or(a, "n_expert", 0), k = (long)r4d_geti_or(a, "top_k", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    const int norm = r4d_geti_or(a, "norm", 0) != 0 ? 1 : 0;
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne || ids.rank != 2 ||
        w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k)
        return R4D_NO(RAD_E_SHAPE);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (lg.stride[1] != 1 || ids.stride[1] != 1 || w.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
    switch (score) {
        case kSigmoid:
            return router_emit<kSigmoid>(stream, lg, bias, ids, w, (int)ne, (int)k, norm, factor);
        case kSqrtSoftplus:
            return router_emit<kSqrtSoftplus>(stream, lg, bias, ids, w, (int)ne, (int)k, norm,
                                              factor);
        default:
            return router_emit<kSoftmax>(stream, lg, bias, ids, w, (int)ne, (int)k, norm, factor);
    }
}

/* router_scored_topk_scatter: router_scored_topk's four operands, moe_scatter's three outputs, then
 * the optional gate and the optional inverse (router_sigmoid_topk_scatter's operands, with the
 * score named). The sort is the single-workgroup form, so M * top_k must fit it; a larger pass is
 * refused rather than served at another cost. */
template <int SCORE>
int router_scored_scatter_emit(RadStream stream, const RadTensor& lg, const RadTensor& bias,
                               const RadTensor& ids, const RadTensor& w, const RadTensor* sorted,
                               const RadTensor* off, const RadTensor* cnt, const RadTensor* gate,
                               long long gp, const RadTensor* inv, int m, int ne, int nl, int k,
                               int norm, float factor, int base, const r4d_moesort::Protect& prot) {
    const int nw = m <= 8 ? 8 : m <= 16 ? 16 : 32;
#define R4D_RSC_I(OWN, NW, INV)                                                                    \
    return R4D_EMIT(stream, (router_scored_scatter_kernel<OWN, SCORE, NW, INV>), dim3(1),         \
                    dim3((NW) * 32), 0, (const float*)lg.data, (long)lg.stride[0],                 \
                    (const float*)bias.data, (long)bias.stride[0], (int32_t*)ids.data,             \
                    (long)ids.stride[0], (float*)w.data, (long)w.stride[0],                        \
                    (int32_t*)sorted->data, (int32_t*)off->data, (int32_t*)cnt->data,              \
                    gate ? (uint16_t*)gate->data : (uint16_t*)nullptr, (long)gp,                   \
                    inv ? (int32_t*)inv->data : (int32_t*)nullptr, m, ne, nl, k, norm, factor,     \
                    base, prot)
#define R4D_RSC(OWN, NW)                                                                           \
    do {                                                                                           \
        if (inv) R4D_RSC_I(OWN, NW, true);                                                         \
        R4D_RSC_I(OWN, NW, false);                                                                 \
    } while (0)
    if (ne <= 384) {
        if (nw == 8) R4D_RSC(12, 8u);
        if (nw == 16) R4D_RSC(12, 16u);
        R4D_RSC(12, 32u);
    }
    if (nw == 8) R4D_RSC(16, 8u);
    if (nw == 16) R4D_RSC(16, 16u);
    R4D_RSC(16, 32u);
#undef R4D_RSC
#undef R4D_RSC_I
}

int router_scored_scatter_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7) return RAD_E_INVAL;
    const char* sc = r4d_gets(a, "score");
    if (!sc) return R4D_NO(RAD_E_UNSUPPORTED);
    int score;
    if (!std::strcmp(sc, "sigmoid")) score = kSigmoid;
    else if (!std::strcmp(sc, "sqrtsoftplus")) score = kSqrtSoftplus;
    else if (!std::strcmp(sc, "softmax")) score = kSoftmax;
    else return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const RadTensor *sorted = &a->t[4], *off = &a->t[5], *cnt = &a->t[6];
    const RadTensor* gate = a->n_t > 7 ? r4d_opt(a, 7) : nullptr;
    const RadTensor* inv = a->n_t > 8 ? r4d_opt(a, 8) : nullptr;
    const long ne = (long)r4d_geti_or(a, "n_expert", 0), k = (long)r4d_geti_or(a, "top_k", 0);
    const long nl = (long)r4d_geti_or(a, "n_local", ne), base = (long)r4d_geti_or(a, "expert_base", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    const int norm = r4d_geti_or(a, "norm", 0) != 0 ? 1 : 0;
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 ||
        w.dtype != RAD_F32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32 || (gate && gate->dtype != RAD_BF16) ||
        (inv && inv->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || !sorted->data || !off->data ||
        !cnt->data || lg.rank != 2 || lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 ||
        bias.shape[0] != ne || ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] ||
        w.shape[0] != lg.shape[0] || ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 ||
        ids.stride[1] != 1 || w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k ||
        w.stride[0] < k || bias.stride[0] < 1)
        return R4D_NO(RAD_E_SHAPE);
    const long m = lg.shape[0], T = m * k;
    if (T > (long)r4d_moesort::kScatterSmall || nl < 1 || nl > ne || base < 0 || base + nl > ne ||
        !r4d_contig(sorted) || !r4d_contig(off) || !r4d_contig(cnt) || r4d_numel(sorted) < T ||
        r4d_numel(off) < nl + 1 || r4d_numel(cnt) < nl ||
        (inv && (!r4d_contig(inv) || r4d_numel(inv) < T)))
        return R4D_NO(RAD_E_SHAPE);
    long long grows = 0;
    const long long gp = gate ? r4d_rows_pitch(gate, 1, &grows) : 0;
    if (gate && (gp < 1 || grows < m)) return R4D_NO(RAD_E_SHAPE);
    r4d_moesort::Protect prot;
    const int prc = r4d_moesort::protect_of(a, nl, &prot);
    if (prc != RAD_OK) return prc;
    switch (score) {
        case kSigmoid:
            return router_scored_scatter_emit<kSigmoid>(stream, lg, bias, ids, w, sorted, off, cnt,
                                                        gate, gp, inv, (int)m, (int)ne, (int)nl,
                                                        (int)k, norm, factor, (int)base, prot);
        case kSqrtSoftplus:
            return router_scored_scatter_emit<kSqrtSoftplus>(stream, lg, bias, ids, w, sorted, off,
                                                             cnt, gate, gp, inv, (int)m, (int)ne,
                                                             (int)nl, (int)k, norm, factor,
                                                             (int)base, prot);
        default:
            return router_scored_scatter_emit<kSoftmax>(stream, lg, bias, ids, w, sorted, off, cnt,
                                                        gate, gp, inv, (int)m, (int)ne, (int)nl,
                                                        (int)k, norm, factor, (int)base, prot);
    }
}

/* ==================================================================== router_gemm_f32_scored_topk_scatter
 * Lever FUSE-R (ds41-fuse, 13.11.2026; zzzaql's AQL-stamped c8 on int14: the router chain's two dependent launch gaps,
 * gemm_nt_f32_tile -> router_scored 173 us and router_scored -> scatter_small 172 us a decode step): DeepSeek-V4.1's
 * router projection (gemm_nt_f32), its selection (router_scored_topk) and the sort (moe_scatter) in ONE launch, every
 * stage the stock code, only the launch boundaries removed:
 *   - the GEMM: r4d_gemm_f32_tile.h's tile_wave, gemm_nt_f32_tile_kernel's own body (moved there unchanged), a wave a
 *     C-column tile of R rows as there, NW waves a workgroup instead of 4 -- which wave of which workgroup forms a
 *     column changes none of its arithmetic (each logit: lane-strided fmaf over k ascending, the fixed butterfly);
 *   - the last arriver (router_gemm_topk_scatter's pattern, r4d_moe.hip): every workgroup publishes its logits
 *     (release fence) and adds one to its op instance's arrival counter; the workgroup that completes the count resets
 *     it, acquires the others' logits (fence) and runs router_scored_scatter_kernel's body -- router_scored_row a row a
 *     wave (the stock selection's function, so the ids and weights are router_scored_kernel's bytes) into the sort's
 *     LDS slot list, then scatter_small_place (the stock small sort, whose result does not depend on the wave count).
 * So every output -- logits, ids, weights, sorted_tok, expert_offset, expert_count (and the inverse) -- is the bytes the
 * three launches write, at any M: each row's logits and selection are its own (batch-invariant as the stock chain).
 * Decode band only (M <= 48: the forms below), DeepSeek's sqrtsoftplus score, <= 384 experts. */
/* 64 since lever RGP (DRAFTER track round 17): the core declares the fused launch up to 48 rows unless RADIANCE_RGS_ROWS=64 at load
 * (rad_block_moe_fp8.h), so a 49..64-row pass reaches it only then; every form handles 64 rows (T = 64 x top_k fits the small sort) */
constexpr int kRgsMaxRows = 64;

/* tp8 loop D28 (DECODE round 22): RW > 1 = a 2-D wave map -- a workgroup's NW waves are RW row groups x NW / RW expert groups
 * (wave w: expert group w % (NW / RW), row group w / (NW / RW)), so a workgroup covers R RW rows x C NW / RW experts and every
 * weight row is read by ceil(M / (R RW)) workgroups instead of ceil(M / R) (at 48 rows, R 4: 3 instead of 12). Which wave of
 * which workgroup forms a logit changes none of its arithmetic (tile_wave's note): the stock bytes. RW 1 = today's map. */
template <int R, int C, int U, typename WT, int OWN, int SCORE, uint32_t NW, bool INV, int RW = 1>
__global__ __launch_bounds__(NW * 32) void router_gemm_scored_scatter_kernel(
    const uint16_t* x, long xp, const WT* wt, long wtp, float* lg, long lp, int m, int ne, int K,
    const float* bias, long bp, int32_t* ids, long ip, float* weights, long wp, int32_t* sorted,
    int32_t* off, int32_t* cnt, int32_t* inv, int n_local, int k, int norm, float factor, int base,
    r4d_moesort::Protect prot, unsigned* ctr) { R4D_IKA_ENTRY();
    static_assert(RW >= 1 && NW % RW == 0, "the 2-D map's row groups divide the waves");
    constexpr int CW = (int)NW / RW;   /* expert groups a workgroup */
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int c0 = ((int)blockIdx.x * CW + wave % CW) * C, r0 = ((int)blockIdx.y * RW + wave / CW) * R;
    if (c0 < ne && r0 < m) r4d_gf32::tile_wave<R, C, U, WT>(x, wt, lg, xp, wtp, lp, m, ne, K, c0, r0, lane);
    __shared__ unsigned s_last;
    __threadfence();   // this workgroup's logits, device-wide, before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old + 1u == gridDim.x * gridDim.y ? 1u : 0u;
        if (s_last) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();   // every other workgroup's logits: no stale line
    /* router_scored_scatter_kernel's body from here (no gate) */
    __shared__ r4d_moesort::ScatterLds<NW> sl;
    for (int e = threadIdx.x; e < n_local; e += (int)(NW * 32)) sl.cnt[e] = 0;
    for (int r = wave; r < m; r += (int)NW) {
        const int e = router_scored_row<OWN, SCORE>(lg, lp, bias, bp, ids, ip, weights, wp, r, ne, k,
                                                    norm, factor, lane) -
                      base;
        if (lane < k)
            sl.e[r * k + lane] =
                e >= 0 && e < n_local ? r4d_moesort::sort_place(e, (uint32_t)n_local, prot) : -1;
    }
    lds_barrier();
    r4d_moesort::scatter_small_place<NW, INV>(sorted, off, cnt, (uint32_t)(m * k), (uint32_t)n_local,
                                              sl, inv);
}

/* Lever RGP (DRAFTER track round 17, tp8 loop 19.11.2026; ds41_f32dec 8, forms 19 / 20): the fused router with the SELECTION
 * spread over the row groups. The stock launch's last arriver selects every row of the pass in one workgroup after the whole
 * grid's logits (at 48 rows 3 rows a wave in a row, then the sort); here each row group's last arriving workgroup (its own
 * counter, ctr[1 + y]) selects that group's R rows as soon as the group's logits are in -- router_scored_row a row a wave, the
 * stock selection's function -- and parks each slot's sort place in `sorted` [r k + j] (the output the sort writes last;
 * nothing reads it before the launch ends); the last row group (ctr[0]) reads the slot list back into the sort's LDS (after
 * the acquire fence, as the stock launch reads the logits) and runs scatter_small_place. The GEMM is tile_wave on the 1-D map.
 * So every output -- logits, ids, weights, sorted_tok, expert_offset, expert_count (and the inverse) -- is the stock launch's
 * bytes at any M (tools/dbench/rbench r17: the 6 outputs + guard tails byte-identical at 16 .. 64 rows, each launched twice).
 * One card (xeon GPU1, K 5120, 384 experts, bf16; cold = after a 256 MiB read (TESTING.md rule 2), us a call, stock form 8 ->
 * form 19 / 20): 16 rows 36.2 -> 30.6 / 29.6, 24 40.4 -> 33.3 / 32.7, 32 41.7 -> 33.8 / 33.6, 40 45.8 -> 39.1 / 38.7, 48 56.2 ->
 * 44.1 / 42.4, 64 65.1 -> 51.1 / 50.8 (the served 49..64-row pair f32pf + router_scored_scatter: ~80 us in lq177's stamps; the
 * served 48-row call 61.1 us). Rows <= 16 groups x 4 = 64: ctr needs 17 slots. */
template <int R, int C, int U, typename WT, int OWN, int SCORE, uint32_t NW, bool INV>
__global__ __launch_bounds__(NW * 32) void router_gemm_rowsel_kernel(
    const uint16_t* x, long xp, const WT* wt, long wtp, float* lg, long lp, int m, int ne, int K,
    const float* bias, long bp, int32_t* ids, long ip, float* weights, long wp, int32_t* sorted,
    int32_t* off, int32_t* cnt, int32_t* inv, int n_local, int k, int norm, float factor, int base,
    r4d_moesort::Protect prot, unsigned* ctr) { R4D_IKA_ENTRY();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int c0 = ((int)blockIdx.x * (int)NW + wave) * C, r0 = (int)blockIdx.y * R;
    if (c0 < ne && r0 < m) r4d_gf32::tile_wave<R, C, U, WT>(x, wt, lg, xp, wtp, lp, m, ne, K, c0, r0, lane);
    __shared__ unsigned s_last;
    __threadfence();   // this workgroup's logits, device-wide, before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* gc = ctr + 1 + blockIdx.y;
        const unsigned old = __hip_atomic_fetch_add(gc, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old + 1u == gridDim.x ? 1u : 0u;
        if (s_last) __hip_atomic_store(gc, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();   // the row group's logits from its other workgroups
    for (int r = r0 + wave; r < r0 + R && r < m; r += (int)NW) {
        const int e = router_scored_row<OWN, SCORE>(lg, lp, bias, bp, ids, ip, weights, wp, r, ne, k, norm, factor, lane) -
                      base;
        if (lane < k)
            sorted[r * k + lane] = e >= 0 && e < n_local ? r4d_moesort::sort_place(e, (uint32_t)n_local, prot) : -1;
    }
    __threadfence();   // this row group's slots (and ids / weights) before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old + 1u == gridDim.y ? 1u : 0u;
        if (s_last) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();   // every row group's slots
    __shared__ r4d_moesort::ScatterLds<NW> sl;
    for (int e = threadIdx.x; e < n_local; e += (int)(NW * 32)) sl.cnt[e] = 0;
    for (int i = threadIdx.x; i < m * k; i += (int)(NW * 32)) sl.e[i] = sorted[i];
    lds_barrier();   // the slot list is in LDS before the sort writes `sorted`
    r4d_moesort::scatter_small_place<NW, INV>(sorted, off, cnt, (uint32_t)(m * k), (uint32_t)n_local, sl, inv);
}

/* Lever RGK (DRAFTER track round 39, tp8 loop 20.11.2026; ds41_f32dec 9, form 21): RGP with the selection's PER-EXPERT part moved
 * into the GEMM workgroups' epilogue. router_scored_row spends most of a row's time on its 12 scores a lane (the score's exp / log1p
 * with their table reads, then the rank key) before the 6 rounds; here, once a workgroup's logits are written (tile_wave, the stock
 * bytes), its threads form each of them's score and rank key with the same element functions (ds_exp_n / ds_log1p_n / sqrtf /
 * ds_rank_key: the same bits whichever thread forms them) into a scratch [rows][n_expert], and the row group's last arriver runs
 * router_keyed_row: the stock 6 rounds over the keys, the weights from the chosen experts' scores, the sum in round order. So every
 * output -- logits, ids, weights, sorted_tok, expert_offset, expert_count (and the inverse) -- is the stock launch's bytes at any M
 * (tools/dbench/rbench r39d: logits / ids / weights / the sort byte-identical to the served form 19 at 16 / 32 / 48 / 64 rows, each
 * launched twice). The scratch is libr4d's own, one per (stream, logits buffer), made at that buffer's first form-21 launch
 * (r4d_8a8_split_ctr's pattern; 64 x 384 x 12 B = 288 KiB); every key / score the tail reads is written earlier in the same launch.
 * One card (xeon GPU1, K 5120, 384 experts, bf16; cold = after a 256 MiB read, us a call, form 19 -> 21): 16 rows 30.6 -> 25.1,
 * 32 33.2 -> 28.6, 48 44.3 -> 38.2, 64 51.1 -> 43.8 (form 8, ds41_f32dec 1's rule at 17..48 rows: 56.2 at 48 rows). */
template <int SCORE>
__device__ __forceinline__ float router_score_one(float x) {
    static_assert(SCORE == kSqrtSoftplus || SCORE == kSigmoid, "a per-expert score (the softmax needs its row)");
    float t[1];
    if (SCORE == kSigmoid) {
        t[0] = -x;
        ds_exp_n(t);
        return 1.0f / (1.0f + t[0]);
    }
    t[0] = x;
    ds_exp_n(t);
    ds_log1p_n(t);
    return sqrtf(x > 20.0f ? x : t[0]);
}
/* router_scored_row from the 6 rounds on: the rank keys and scores come from the epilogue (keys / sc, pitch kp) */
template <int OWN>
__device__ __forceinline__ int router_keyed_row(const unsigned long long* keys, const float* sc, long kp, int32_t* ids,
                                                long ip, float* weights, long wp, long r, int ne, int k, int norm,
                                                float factor, int lane) {
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i;
        key[i] = e < ne ? keys[r * kp + e] : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = ds_wave_max_u64(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    const float chosen = sc[r * kp + e];
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[r * ip + lane] = e;
        weights[r * wp + lane] = chosen * mult;
    }
    return e;
}
template <int R, int C, int U, typename WT, int OWN, int SCORE, uint32_t NW, bool INV>
__global__ __launch_bounds__(NW * 32) void router_gemm_rowkey_kernel(
    const uint16_t* x, long xp, const WT* wt, long wtp, float* lg, long lp, int m, int ne, int K,
    const float* bias, long bp, int32_t* ids, long ip, float* weights, long wp, int32_t* sorted,
    int32_t* off, int32_t* cnt, int32_t* inv, int n_local, int k, int norm, float factor, int base,
    r4d_moesort::Protect prot, unsigned* ctr, unsigned long long* keys, float* sc) { R4D_IKA_ENTRY();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int cb = (int)blockIdx.x * (int)NW * C, c0 = cb + wave * C, r0 = (int)blockIdx.y * R;
    if (c0 < ne && r0 < m) r4d_gf32::tile_wave<R, C, U, WT>(x, wt, lg, xp, wtp, lp, m, ne, K, c0, r0, lane);
    __syncthreads();   /* the workgroup's logits (each wave's lane 0 wrote its tile) */
    for (int q = threadIdx.x; q < R * (int)NW * C; q += (int)NW * 32) {
        const int row = r0 + q / ((int)NW * C), col = cb + q % ((int)NW * C);
        if (row < m && col < ne) {
            const float p = router_score_one<SCORE>(lg[(long)row * lp + col]);
            sc[(long)row * ne + col] = p;
            keys[(long)row * ne + col] = ds_rank_key(p + bias[(long)col * bp], col);
        }
    }
    __shared__ unsigned s_last;
    __threadfence();   // this workgroup's logits, scores and keys, device-wide, before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* gc = ctr + 1 + blockIdx.y;
        const unsigned old = __hip_atomic_fetch_add(gc, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old + 1u == gridDim.x ? 1u : 0u;
        if (s_last) __hip_atomic_store(gc, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();   // the row group's keys from its other workgroups
    for (int r = r0 + wave; r < r0 + R && r < m; r += (int)NW) {
        const int e = router_keyed_row<OWN>(keys, sc, ne, ids, ip, weights, wp, r, ne, k, norm, factor, lane) - base;
        if (lane < k)
            sorted[r * k + lane] = e >= 0 && e < n_local ? r4d_moesort::sort_place(e, (uint32_t)n_local, prot) : -1;
    }
    __threadfence();   // this row group's slots (and ids / weights) before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old + 1u == gridDim.y ? 1u : 0u;
        if (s_last) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();   // every row group's slots
    __shared__ r4d_moesort::ScatterLds<NW> sl;
    for (int e = threadIdx.x; e < n_local; e += (int)(NW * 32)) sl.cnt[e] = 0;
    for (int i = threadIdx.x; i < m * k; i += (int)(NW * 32)) sl.e[i] = sorted[i];
    lds_barrier();   // the slot list is in LDS before the sort writes `sorted`
    r4d_moesort::scatter_small_place<NW, INV>(sorted, off, cnt, (uint32_t)(m * k), (uint32_t)n_local, sl, inv);
}
/* lever RGK's scratch: keys [64][384] u64 then scores [64][384] f32, one per (stream, logits buffer); nullptr = form 19 serves */
constexpr long kRgkRows = 64, kRgkExperts = 384;
unsigned long long* rgk_scratch(RadStream s, const void* lg) {
#ifdef R4D_EMU_HOST
    (void)s; (void)lg;
    return nullptr;
#else
    static std::mutex mu;
    static std::map<std::pair<RadStream, const void*>, unsigned long long*> blocks;
    const auto key = std::make_pair(s, lg);
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = blocks.find(key); it != blocks.end()) return it->second;
    void* p = nullptr;
    if (hipMalloc(&p, (size_t)kRgkRows * kRgkExperts * (8 + 4)) != hipSuccess) return nullptr;
    blocks.emplace(key, (unsigned long long*)p);
    return (unsigned long long*)p;
#endif
}

/* ==================================================================== lever RGM (RULE B)
 * Lever RGM (DRAFTER track round 40, tp8 loop 20.11.2026; ds41_f32dec 10): DeepSeek-V4.1's router projection (K 5120, bf16
 * weights) on v_wmma_f32_16x16x16_bf16 EVERYWHERE the router's logits are formed -- the fused decode launch (form 22 at 1..64
 * rows: + lever RGK's epilogue scores / keys, the per-row-group selection, the stock small sort) AND gemm_nt_f32 at the router's
 * shape at any M (code 40: a mixed / prefill pass's projection, glm_router's column split of 96 included). NOT the stock fmaf
 * chain: the logits move (bench: max |d| ~5e-6 vs the stock chain), so this is rule B -- deterministic and batch-invariant
 * against the build's own reference:
 *   a logit (row r, expert e) = (((p_0 + p_1) + p_2) + ... + p_7), p_w = wave w's slice k in [640 w, 640 (w + 1)) as 40 WMMA steps
 *   in ascending order from 0, step j feeding k = 640 w + 8 j + i and 640 w + 320 + 8 j + i (i < 8) for BOTH operands; the WMMA's
 *   element (r, e) is a function of x row r, weight row e and its accumulator alone. So a logit depends on its own x row and
 *   weight row only: not on M, on the row tiles / expert tiles a workgroup holds (RT / ET), on the unroll U, on which columns a
 *   call computes (the 96-column split), on aligned or element loads (AL), or on the kernel (rgm_gemm_kernel vs form 22).
 * Then everything downstream of the logits (scores, keys, the 6 rounds, the weights, the sort) is the stock function's: given the
 * logits, ids / weights / sorted / offsets / counts are the stock bytes. r4d_selftest --case spdf32 SPD_PART=rgm checks it. */
typedef __bf16 rgm_v8bf __attribute__((ext_vector_type(8)));
typedef float rgm_v8f __attribute__((ext_vector_type(8)));
constexpr int kRgmS = 8;        /* waves a workgroup = the K slices: part of every logit's arithmetic, never a tuning knob */
constexpr int kRgmK = 5120;     /* the trunk router's K: slices of 640, 40 WMMA steps a slice */
template <bool AL>
__device__ __forceinline__ uint4 rgm_ld8(const uint16_t* p) {
    if constexpr (AL) return *(const uint4*)(const void*)p;
    uint4 v;
    v.x = (uint32_t)p[0] | ((uint32_t)p[1] << 16);
    v.y = (uint32_t)p[2] | ((uint32_t)p[3] << 16);
    v.z = (uint32_t)p[4] | ((uint32_t)p[5] << 16);
    v.w = (uint32_t)p[6] | ((uint32_t)p[7] << 16);
    return v;
}
/* wave `wave`'s partials of the workgroup's RT 16-row x ET 16-expert tiles into sp[wave][RT ET 256] (D lane L: expert L % 16, rows
 * 8 (L / 16) + i). Rows past m feed zeros (their own outputs only); experts past n read the last one (not stored). */
template <int RT, int ET, int U, bool AL>
__device__ __forceinline__ void rgm_wave(const uint16_t* x, long xp, const uint16_t* wt, long wtp, int m, int n, int K, int e0,
                                         int r0, int wave, int lane, float* sp) {
    const int nl = lane & 15, h = lane >> 4;
    const int ks = K / kRgmS, half = ks >> 1, J = half >> 3;
    const long kb = (long)wave * ks + (long)h * half;
    const uint16_t* bp[ET];
#pragma unroll
    for (int q = 0; q < ET; ++q) {
        const int e = e0 + 16 * q + nl;
        bp[q] = wt + (long)(e < n ? e : n - 1) * wtp + kb;
    }
    const uint16_t* ap[RT];
    bool rv[RT];
#pragma unroll
    for (int t = 0; t < RT; ++t) {
        const int r = r0 + 16 * t + nl;
        rv[t] = r < m;
        ap[t] = x + (long)(r < m ? r : 0) * xp + kb;
    }
    rgm_v8f c[RT][ET];
#pragma unroll
    for (int t = 0; t < RT; ++t)
#pragma unroll
        for (int q = 0; q < ET; ++q) c[t][q] = (rgm_v8f){ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    int ntv = 0;   /* row tiles with a valid row: the rest skip their WMMAs */
#pragma unroll
    for (int t = 0; t < RT; ++t) ntv += r0 + 16 * t < m ? 1 : 0;
    for (int j = 0; j < J; j += U) {
        uint4 bv[U][ET], av[U][RT];
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < ET; ++q) bv[u][q] = rgm_ld8<AL>(bp[q] + 8 * (j + u));
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int t = 0; t < RT; ++t) {
                uint4 z = { 0u, 0u, 0u, 0u };
                if (t < ntv && rv[t]) z = rgm_ld8<AL>(ap[t] + 8 * (j + u));
                av[u][t] = z;
            }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int t = 0; t < RT; ++t)
                if (t < ntv)
#pragma unroll
                    for (int q = 0; q < ET; ++q)
                        c[t][q] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(rgm_v8bf, av[u][t]),
                                                                                   __builtin_bit_cast(rgm_v8bf, bv[u][q]), c[t][q]);
    }
#pragma unroll
    for (int t = 0; t < RT; ++t)
#pragma unroll
        for (int q = 0; q < ET; ++q)
#pragma unroll
            for (int i = 0; i < 8; ++i) sp[(wave * RT * ET + t * ET + q) * 256 + (8 * h + i) * 16 + nl] = c[t][q][i];
}
/* the slices' fold in wave order: element q of the workgroup's tiles */
template <int RT, int ET>
__device__ __forceinline__ float rgm_fold(const float* sp, int q) {
    float s = sp[q];
#pragma unroll
    for (int w = 1; w < kRgmS; ++w) s += sp[w * RT * ET * 256 + q];
    return s;
}
/* code 40: y[M, n] = the logits alone at any M (gemm_nt_f32's router shape) */
template <int RT, int ET, int U, bool AL>
__global__ __launch_bounds__(kRgmS * 32) void rgm_gemm_kernel(const uint16_t* x, long xp, const uint16_t* wt, long wtp, float* y,
                                                              long yp, int m, int n, int K) { R4D_IKA_ENTRY();
    __shared__ float sp[kRgmS * RT * ET * 256];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int e0 = (int)blockIdx.x * 16 * ET, r0 = (int)blockIdx.y * 16 * RT;
    rgm_wave<RT, ET, U, AL>(x, xp, wt, wtp, m, n, K, e0, r0, wave, lane, sp);
    __syncthreads();
    for (int q = threadIdx.x; q < RT * ET * 256; q += kRgmS * 32) {
        const int tq = q >> 8, row = r0 + 16 * (tq / ET) + ((q >> 4) & 15), col = e0 + 16 * (tq % ET) + (q & 15);
        if (row < m && col < n) y[(long)row * yp + col] = rgm_fold<RT, ET>(sp, q);
    }
}
/* form 22: the fused router at 1..64 rows -- the logits as code 40's, then lever RGK's epilogue (score + rank key of each logit,
 * the same element functions), the row group's (16 rows) last arriver runs router_keyed_row (the stock 6 rounds), the last row
 * group the stock small sort */
template <int ET, int U, bool AL, int OWN, int SCORE, bool INV>
__global__ __launch_bounds__(kRgmS * 32) void router_wm_rowkey_kernel(
    const uint16_t* x, long xp, const uint16_t* wt, long wtp, float* lg, long lp, int m, int ne, int K,
    const float* bias, long bp, int32_t* ids, long ip, float* weights, long wp, int32_t* sorted,
    int32_t* off, int32_t* cnt, int32_t* inv, int n_local, int k, int norm, float factor, int base,
    r4d_moesort::Protect prot, unsigned* ctr, unsigned long long* keys, float* sc) { R4D_IKA_ENTRY();
    __shared__ float sp[kRgmS * ET * 256];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int e0 = (int)blockIdx.x * 16 * ET, r0 = (int)blockIdx.y * 16;
    rgm_wave<1, ET, U, AL>(x, xp, wt, wtp, m, ne, K, e0, r0, wave, lane, sp);
    __syncthreads();
    for (int q = threadIdx.x; q < ET * 256; q += kRgmS * 32) {
        const int row = r0 + ((q >> 4) & 15), col = e0 + 16 * (q >> 8) + (q & 15);
        if (row < m && col < ne) {
            const float s = rgm_fold<1, ET>(sp, q);
            lg[(long)row * lp + col] = s;
            const float p = router_score_one<SCORE>(s);
            sc[(long)row * ne + col] = p;
            keys[(long)row * ne + col] = ds_rank_key(p + bias[(long)col * bp], col);
        }
    }
    __shared__ unsigned s_last;
    __threadfence();   // this workgroup's logits, scores and keys, device-wide, before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned* gc = ctr + 1 + blockIdx.y;
        const unsigned old = __hip_atomic_fetch_add(gc, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old + 1u == gridDim.x ? 1u : 0u;
        if (s_last) __hip_atomic_store(gc, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();   // the row group's keys from its other workgroups
    for (int r = r0 + wave; r < r0 + 16 && r < m; r += kRgmS) {
        const int e = router_keyed_row<OWN>(keys, sc, ne, ids, ip, weights, wp, r, ne, k, norm, factor, lane) - base;
        if (lane < k)
            sorted[r * k + lane] = e >= 0 && e < n_local ? r4d_moesort::sort_place(e, (uint32_t)n_local, prot) : -1;
    }
    __threadfence();   // this row group's slots (and ids / weights) before its arrival
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = old + 1u == gridDim.y ? 1u : 0u;
        if (s_last) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();   // every row group's slots
    __shared__ r4d_moesort::ScatterLds<(uint32_t)kRgmS> sl;
    for (int e = threadIdx.x; e < n_local; e += kRgmS * 32) sl.cnt[e] = 0;
    for (int i = threadIdx.x; i < m * k; i += kRgmS * 32) sl.e[i] = sorted[i];
    lds_barrier();   // the slot list is in LDS before the sort writes `sorted`
    r4d_moesort::scatter_small_place<(uint32_t)kRgmS, INV>(sorted, off, cnt, (uint32_t)(m * k), (uint32_t)n_local, sl, inv);
}
static bool rgm_aligned(const void* x, long xp, const void* w, long wp) {
    return ((uintptr_t)x & 15u) == 0 && ((uintptr_t)w & 15u) == 0 && (xp & 7) == 0 && (wp & 7) == 0;
}
}  // namespace

/* code 40 (r4d_ds41_f32pf.h): RAD_OK, a launch failure or R4D_F32PF_DECLINE (not bf16 weights, K not the router's, N not whole
 * 16-expert tiles). The tiles a workgroup by M (speed only: the arithmetic is the same at every choice). */
int r4d_rgm_gemm(RadStream s, const R4dF32PfCall& c) {
    if (!c.wbf16 || c.K != kRgmK || c.M < 1 || c.N < 16 || (c.N & 15) || !c.x || !c.w || !c.y) return R4D_F32PF_DECLINE;
    const bool al = rgm_aligned(c.x, c.xp, c.w, c.wp);
    const bool et2 = (c.N & 31) == 0;
#define RGM_G(RT, ET, U)                                                                                                  \
    (al ? R4D_EMIT(s, (rgm_gemm_kernel<RT, ET, U, true>),                                                                 \
                   dim3((unsigned)((c.N + 16 * (ET) - 1) / (16 * (ET))), (unsigned)((c.M + 16 * (RT) - 1) / (16 * (RT)))), \
                   dim3(kRgmS * 32), 0, c.x, c.xp, (const uint16_t*)c.w, c.wp, c.y, c.yp, c.M, c.N, c.K)                   \
        : R4D_EMIT(s, (rgm_gemm_kernel<RT, ET, U, false>),                                                                \
                   dim3((unsigned)((c.N + 16 * (ET) - 1) / (16 * (ET))), (unsigned)((c.M + 16 * (RT) - 1) / (16 * (RT)))), \
                   dim3(kRgmS * 32), 0, c.x, c.xp, (const uint16_t*)c.w, c.wp, c.y, c.yp, c.M, c.N, c.K))
    if ((long)c.M > 16L * 65535L) return R4D_F32PF_DECLINE;
    if (c.M <= 64) return et2 ? RGM_G(1, 2, 8) : RGM_G(1, 1, 8);
    /* above 64 rows: R4D_RGM_TILE=t (read once; bench knob, speed only -- every choice forms each logit by rgm_wave / rgm_fold):
     * 0 RT 2 ET 2 U 4, 1 RT 4 ET 1 U 4, 2 RT 1 ET 4 U 4, 3 RT 2 ET 2 U 8, 4 RT 4 ET 1 U 8, 5 RT 1 ET 2 U 8 */
    /* unset: by N (xeon card 1 r40e, cold us a call at 2112 rows vs tile 0): N 96 (the router's split) tile 5 89.5 vs 97.2 (256 rows 16.4 vs
     * 19.6); N 384 from 128 rows tile 2 231.0 vs 302.8; N 1024 (the compressor, value 11) tile 3 697.7 vs 758.0 */
    static const int tile_env = [] { const char* e = std::getenv("R4D_RGM_TILE"); return e && *e ? std::atoi(e) : -1; }();
    const int tile = tile_env >= 0 ? tile_env : c.N == 96 ? 5 : c.N == 384 && c.M >= 128 ? 2 : c.N == 1024 ? 3 : 0;
    if (!et2) return tile == 1 || tile == 4 ? RGM_G(4, 1, 4) : RGM_G(2, 1, 4);
    switch (tile) {
        case 1: return RGM_G(4, 1, 4);
        case 2: return (c.N & 63) == 0 ? RGM_G(1, 4, 4) : RGM_G(2, 2, 4);
        case 3: return RGM_G(2, 2, 8);
        case 4: return RGM_G(4, 1, 8);
        case 5: return RGM_G(1, 2, 8);
        default: return RGM_G(2, 2, 4);
    }
#undef RGM_G
}

namespace {

/* The op instance's arrival counter (router_gemm_topk_scatter's rule, r4d_moe.hip): taken at init -- once per resolved
 * instance during declare, before the engine measures free VRAM -- never on the launch path; zeroed, and zeroed again by
 * each launch's last arriver. One 64-byte slot an instance (op, band, domain, rank); each routed layer declares its own
 * op, and an instance's launches are ordered (they write the same declared outputs). Never reused while its block lives. */
constexpr uint32_t kRgsCtrStride = 32, kRgsCtrSlots = 4096;   /* 32 since lever RGP: ctr[0] + 16 row groups at 64 rows */
struct RgsInst {
    int dev = 0;
    unsigned* ctr = nullptr;
};
#ifndef R4D_EMU_HOST
struct RgsBlock {
    unsigned* base = nullptr;
    uint32_t next = 0, live = 0;
};
std::mutex g_rgs_mu;
std::map<int, RgsBlock> g_rgs_blocks;
#endif
std::atomic<long long> g_rgs_launch{0}, g_rgs_rows{0};

template <typename WT, int SCORE>
int rgs_emit(RadStream stream, const RadTensor& x, const RadTensor& w, const RadTensor& lg, const RadTensor& bias,
             const RadTensor& ids, const RadTensor& wts, const RadTensor& sorted, const RadTensor& off,
             const RadTensor& cnt, const RadTensor* inv, int m, int ne, int K, int nl, int k, int norm, float factor,
             int base, const r4d_moesort::Protect& prot, unsigned* ctr) {
#define RGS_L(R, C, U, NW, INV)                                                                                   \
    R4D_EMIT(stream, (router_gemm_scored_scatter_kernel<R, C, U, WT, 12, SCORE, NW, INV>),                       \
             dim3((unsigned)((ne + (NW) * (C) - 1) / ((NW) * (C))), (unsigned)((m + (R) - 1) / (R))),             \
             dim3((NW) * 32), 0, (const uint16_t*)x.data, (long)x.stride[0], (const WT*)w.data, (long)w.stride[0],  \
             (float*)lg.data, (long)lg.stride[0], m, ne, K, (const float*)bias.data, (long)bias.stride[0],         \
             (int32_t*)ids.data, (long)ids.stride[0], (float*)wts.data, (long)wts.stride[0], (int32_t*)sorted.data, \
             (int32_t*)off.data, (int32_t*)cnt.data, inv ? (int32_t*)inv->data : (int32_t*)nullptr, nl, k, norm,   \
             factor, base, prot, ctr)
#define RGS_I(R, C, U, NW) (inv ? RGS_L(R, C, U, NW, true) : RGS_L(R, C, U, NW, false))
    /* tp8 loop D28: the 2-D map's forms (RW row groups a workgroup; RGS_L's arguments, the grid by C NW / RW experts and R RW rows) */
#define RGS_L2(R, C, U, NW, RW, INV)                                                                              \
    R4D_EMIT(stream, (router_gemm_scored_scatter_kernel<R, C, U, WT, 12, SCORE, NW, INV, RW>),                   \
             dim3((unsigned)((ne + ((NW) / (RW)) * (C) - 1) / (((NW) / (RW)) * (C))),                            \
                  (unsigned)((m + (R) * (RW) - 1) / ((R) * (RW)))),                                               \
             dim3((NW) * 32), 0, (const uint16_t*)x.data, (long)x.stride[0], (const WT*)w.data, (long)w.stride[0],  \
             (float*)lg.data, (long)lg.stride[0], m, ne, K, (const float*)bias.data, (long)bias.stride[0],         \
             (int32_t*)ids.data, (long)ids.stride[0], (float*)wts.data, (long)wts.stride[0], (int32_t*)sorted.data, \
             (int32_t*)off.data, (int32_t*)cnt.data, inv ? (int32_t*)inv->data : (int32_t*)nullptr, nl, k, norm,   \
             factor, base, prot, ctr)
#define RGS_I2(R, C, U, NW, RW) (inv ? RGS_L2(R, C, U, NW, RW, true) : RGS_L2(R, C, U, NW, RW, false))
    /* lever RGP: router_gemm_rowsel_kernel on the 1-D map (RGS_L's grid and arguments) */
#define RGP_L(R, C, U, NW, INV)                                                                                   \
    R4D_EMIT(stream, (router_gemm_rowsel_kernel<R, C, U, WT, 12, SCORE, NW, INV>),                               \
             dim3((unsigned)((ne + (NW) * (C) - 1) / ((NW) * (C))), (unsigned)((m + (R) - 1) / (R))),             \
             dim3((NW) * 32), 0, (const uint16_t*)x.data, (long)x.stride[0], (const WT*)w.data, (long)w.stride[0],  \
             (float*)lg.data, (long)lg.stride[0], m, ne, K, (const float*)bias.data, (long)bias.stride[0],         \
             (int32_t*)ids.data, (long)ids.stride[0], (float*)wts.data, (long)wts.stride[0], (int32_t*)sorted.data, \
             (int32_t*)off.data, (int32_t*)cnt.data, inv ? (int32_t*)inv->data : (int32_t*)nullptr, nl, k, norm,   \
             factor, base, prot, ctr)
#define RGP_I(R, C, U, NW) (inv ? RGP_L(R, C, U, NW, true) : RGP_L(R, C, U, NW, false))
    /* lever SPD-F (ds41-etp-spd, r4d_ds41_f32pf.h): ds41_f32dec's form -- the tile R x C a wave (U steps of k) and the
     * waves a workgroup NW (the tail's waves too: a row a wave, wrapping; the sort's result does not depend on NW).
     * Which wave of which workgroup forms a logit changes none of its arithmetic: the stock bytes at every form. */
    switch (r4d_f32dec_rgs_form(m)) {
        case 1: return RGS_I(8, 2, 8, 16u);
        case 2: return RGS_I(8, 4, 8, 16u);
        case 3: return RGS_I(4, 4, 8, 16u);
        case 4: return RGS_I(4, 2, 8, 16u);
        case 5: return RGS_I(2, 4, 16, 16u);
        case 6: return RGS_I(4, 4, 8, 8u);
        case 7: return RGS_I(2, 4, 16, 8u);
        case 8: return RGS_I(4, 8, 8, 16u);
        case 9: return RGS_I(2, 2, 16, 8u);    /* the stock forms, for the bench */
        case 10: return RGS_I(2, 2, 16, 16u);
        case 11: return RGS_I(8, 1, 8, 16u);
        case 12: return RGS_I(4, 2, 16, 8u);
        case 13: return RGS_I(1, 4, 16, 8u);
        case 14: return RGS_I(2, 8, 16, 8u);
        /* tp8 loop D28: form 8's 4 x 8 U8 tile in 2-D maps -- 15: 4 row groups x 4 expert groups (16 rows x 32 experts a
         * workgroup), 16: form 3's 4 x 4 U8 at 4 x 4 (16 x 16), 17: 2 x 8 (8 rows x 64 experts), 18: 4 x 4 U8 at 8 waves as 2 x 4
         * (8 rows x 16 experts) */
        case 15: return RGS_I2(4, 8, 8, 16u, 4);
        case 16: return RGS_I2(4, 4, 8, 16u, 4);
        case 17: return RGS_I2(4, 8, 8, 16u, 2);
        case 18: return RGS_I2(4, 4, 8, 8u, 2);
        /* lever RGP (DRAFTER track round 17): form 3's 4 x 4 U8 tile at 8 waves (19) / U 16 (20), the selection a row group at a time */
        case 19: return RGP_I(4, 4, 8, 8u);
        case 20: return RGP_I(4, 4, 16, 8u);
        /* lever RGK (DRAFTER track round 39): form 19's map with the scores / keys in the GEMM epilogue (form 19 when the scratch
         * cannot be made or the expert count exceeds it) */
        case 21: {
            unsigned long long* sk = (SCORE == kSqrtSoftplus || SCORE == kSigmoid) && m <= kRgkRows && ne <= kRgkExperts
                                         ? rgk_scratch(stream, lg.data) : nullptr;
            if (!sk) return RGP_I(4, 4, 8, 8u);
            float* sc = (float*)(sk + kRgkRows * kRgkExperts);
#define RGK_L(R, C, U, NW, INV)                                                                                   \
    R4D_EMIT(stream, (router_gemm_rowkey_kernel<R, C, U, WT, 12, (SCORE == kSigmoid ? kSigmoid : kSqrtSoftplus), NW, INV>), \
             dim3((unsigned)((ne + (NW) * (C) - 1) / ((NW) * (C))), (unsigned)((m + (R) - 1) / (R))),             \
             dim3((NW) * 32), 0, (const uint16_t*)x.data, (long)x.stride[0], (const WT*)w.data, (long)w.stride[0],  \
             (float*)lg.data, (long)lg.stride[0], m, ne, K, (const float*)bias.data, (long)bias.stride[0],         \
             (int32_t*)ids.data, (long)ids.stride[0], (float*)wts.data, (long)wts.stride[0], (int32_t*)sorted.data, \
             (int32_t*)off.data, (int32_t*)cnt.data, inv ? (int32_t*)inv->data : (int32_t*)nullptr, nl, k, norm,   \
             factor, base, prot, ctr, sk, sc)
            return inv ? RGK_L(4, 4, 8, 8u, true) : RGK_L(4, 4, 8, 8u, false);
#undef RGK_L
        }
        /* lever RGM (DRAFTER track round 40, RULE B): the WMMA logits (gemm_nt_f32's code 40 arithmetic) + RGK's epilogue keys + the
         * per-row-group selection + the stock sort. The trunk router (384 experts) with bf16 weights at K 5120 only; f32 weights,
         * another K or another router take form 21 (the stock arithmetic), as code 40 / r4d_rgm_route leave them to the stock gemm:
         * a router's logits are one function in every pass either way. */
        case 22: {
            unsigned long long* sk = (SCORE == kSqrtSoftplus || SCORE == kSigmoid) && m <= kRgkRows && ne <= kRgkExperts
                                         ? rgk_scratch(stream, lg.data) : nullptr;
            /* the trunk router (384 experts, the shape r4d_rgm_route takes in gemm_nt_f32); the DSpark drafter's router (128 experts)
             * only under value 12 (lever RGD, DRAFTER round 41: r4d_rgm_route then takes its context pass's gemm_nt_f32 at N 128 too,
             * so its logits are one function in every pass); below 12 it stays in form 21 (the stock arithmetic) */
            const bool rgd = ne == 128 && r4d_f32dec_get() == 12;
            if (!std::is_same<WT, uint16_t>::value || K != kRgmK || (ne != 384 && !rgd)) {
                if (!sk) return RGP_I(4, 4, 8, 8u);
                float* sc = (float*)(sk + kRgkRows * kRgkExperts);
#define RGK_L(R, C, U, NW, INV)                                                                                   \
    R4D_EMIT(stream, (router_gemm_rowkey_kernel<R, C, U, WT, 12, (SCORE == kSigmoid ? kSigmoid : kSqrtSoftplus), NW, INV>), \
             dim3((unsigned)((ne + (NW) * (C) - 1) / ((NW) * (C))), (unsigned)((m + (R) - 1) / (R))),             \
             dim3((NW) * 32), 0, (const uint16_t*)x.data, (long)x.stride[0], (const WT*)w.data, (long)w.stride[0],  \
             (float*)lg.data, (long)lg.stride[0], m, ne, K, (const float*)bias.data, (long)bias.stride[0],         \
             (int32_t*)ids.data, (long)ids.stride[0], (float*)wts.data, (long)wts.stride[0], (int32_t*)sorted.data, \
             (int32_t*)off.data, (int32_t*)cnt.data, inv ? (int32_t*)inv->data : (int32_t*)nullptr, nl, k, norm,   \
             factor, base, prot, ctr, sk, sc)
                return inv ? RGK_L(4, 4, 8, 8u, true) : RGK_L(4, 4, 8, 8u, false);
#undef RGK_L
            }
            if (!sk) return R4D_NO(RAD_E_NOMEM);   /* the WMMA logits or nothing: never the stock arithmetic for a bf16 router */
            float* sc = (float*)(sk + kRgkRows * kRgkExperts);
            const bool al = rgm_aligned(x.data, (long)x.stride[0], w.data, (long)w.stride[0]);
            /* speed only: ET 2 above 32 rows at 384 experts; the DSpark router (128) by R4D_RGD_ET (bench knob, read once: 1 / 2; unset
             * = ET 1 at every height: 8 expert tiles x ceil(m / 16) row groups) */
            static const int rgd_et = [] { const char* e = std::getenv("R4D_RGD_ET"); return e && *e ? std::atoi(e) : 0; }();
            const bool et2 = rgd ? rgd_et == 2 : m > 32 && (ne & 31) == 0;
#define RGM_L(ET, AL, INV)                                                                                        \
    R4D_EMIT(stream, (router_wm_rowkey_kernel<ET, 8, AL, 12, (SCORE == kSigmoid ? kSigmoid : kSqrtSoftplus), INV>),    \
             dim3((unsigned)((ne + 16 * (ET) - 1) / (16 * (ET))), (unsigned)((m + 15) / 16)), dim3(kRgmS * 32), 0, \
             (const uint16_t*)x.data, (long)x.stride[0], (const uint16_t*)w.data, (long)w.stride[0],                \
             (float*)lg.data, (long)lg.stride[0], m, ne, K, (const float*)bias.data, (long)bias.stride[0],         \
             (int32_t*)ids.data, (long)ids.stride[0], (float*)wts.data, (long)wts.stride[0], (int32_t*)sorted.data, \
             (int32_t*)off.data, (int32_t*)cnt.data, inv ? (int32_t*)inv->data : (int32_t*)nullptr, nl, k, norm,   \
             factor, base, prot, ctr, sk, sc)
#define RGM_A(ET, INV) (al ? RGM_L(ET, true, INV) : RGM_L(ET, false, INV))
            if (inv) return et2 ? RGM_A(2, true) : RGM_A(1, true);
            return et2 ? RGM_A(2, false) : RGM_A(1, false);
#undef RGM_A
#undef RGM_L
        }
        default: break;
    }
    /* the stock gemm_f32_body's tile form for the height (2x2 with 16 steps up to 16 rows, 8x1 with 8 steps below
     * 64); the waves a workgroup router_scored_scatter_emit's (a wave a row up to 16 rows; 16 waves above, the rows
     * wrapping -- 32 would cap the GEMM's registers) */
    return m <= 8 ? RGS_I(2, 2, 16, 8u) : m <= 16 ? RGS_I(2, 2, 16, 16u) : RGS_I(8, 1, 8, 16u);
#undef RGP_I
#undef RGP_L
#undef RGS_I2
#undef RGS_L2
#undef RGS_I
#undef RGS_L
}

/* Operands: a [M, K] bf16, b [n_expert, K] f32 / bf16 (the router projection), bias [n_expert] f32 -> logits [M,
 * n_expert] f32, ids [M, top_k] i32, weights [M, top_k] f32, sorted_tok, expert_offset, expert_count i32, inverse? i32.
 * Every shim check of gemm_nt_f32 (the decode band's), router_scored_topk and moe_scatter's small form. */
int router_gemm_scored_scatter_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 9) return RAD_E_INVAL;
    const char* sc = r4d_gets(a, "score");
    if (!sc || std::strcmp(sc, "sqrtsoftplus")) return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &x = a->t[0], &w = a->t[1], &bias = a->t[2], &lg = a->t[3], &ids = a->t[4], &wts = a->t[5];
    const RadTensor &sorted = a->t[6], &off = a->t[7], &cnt = a->t[8];
    const RadTensor* inv = a->n_t > 9 ? r4d_opt(a, 9) : nullptr;
    const long ne = (long)r4d_geti_or(a, "n_expert", 0), k = (long)r4d_geti_or(a, "top_k", 0);
    const long K = (long)r4d_geti_or(a, "K", 0);
    const long nl = (long)r4d_geti_or(a, "n_local", ne), base = (long)r4d_geti_or(a, "expert_base", 0);
    const float factor = (float)r4d_getf(a, "route_scale", 0);
    const int norm = r4d_geti_or(a, "norm", 0) != 0 ? 1 : 0;
    if (x.dtype != RAD_BF16 || (w.dtype != RAD_F32 && w.dtype != RAD_BF16) || bias.dtype != RAD_F32 ||
        lg.dtype != RAD_F32 || ids.dtype != RAD_I32 || wts.dtype != RAD_F32 || sorted.dtype != RAD_I32 ||
        off.dtype != RAD_I32 || cnt.dtype != RAD_I32 || (inv && inv->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    if (!x.data || !w.data || !bias.data || !lg.data || !ids.data || !wts.data || !sorted.data || !off.data ||
        !cnt.data)
        return R4D_NO(RAD_E_INVAL);
    if (ne < 1 || ne > 384 || k < 1 || k > 32 || k > ne || K < 1 || K > 16384 || !std::isfinite(factor) ||
        factor <= 0 || x.rank != 2 || w.rank != 2 || lg.rank != 2 || ids.rank != 2 || wts.rank != 2 ||
        bias.rank != 1 || x.shape[0] < 1 || x.shape[0] > kRgsMaxRows || x.shape[1] != K || x.stride[1] != 1 ||
        x.stride[0] < K || w.shape[0] != ne || w.shape[1] != K || w.stride[1] != 1 || w.stride[0] < K ||
        bias.shape[0] != ne || bias.stride[0] < 1 || lg.shape[0] != x.shape[0] || lg.shape[1] != ne ||
        lg.stride[1] != 1 || lg.stride[0] < ne || ids.shape[0] != x.shape[0] || ids.shape[1] != k ||
        ids.stride[1] != 1 || ids.stride[0] < k || wts.shape[0] != x.shape[0] || wts.shape[1] != k ||
        wts.stride[1] != 1 || wts.stride[0] < k)
        return R4D_NO(RAD_E_SHAPE);
    const long m = x.shape[0], T = m * k;
    if (T > (long)r4d_moesort::kScatterSmall || nl < 1 || nl > ne || base < 0 || base + nl > ne ||
        !r4d_contig(&sorted) || !r4d_contig(&off) || !r4d_contig(&cnt) || r4d_numel(&sorted) < T ||
        r4d_numel(&off) < nl + 1 || r4d_numel(&cnt) < nl || (inv && (!r4d_contig(inv) || r4d_numel(inv) < T)))
        return R4D_NO(RAD_E_SHAPE);
    r4d_moesort::Protect prot;
    const int prc = r4d_moesort::protect_of(a, nl, &prot);
    if (prc != RAD_OK) return prc;
    const RgsInst* const in = (const RgsInst*)a->instance;
    if (!in || !in->ctr) return R4D_NO(RAD_E_STATE);
    const int rc = w.dtype == RAD_BF16
                       ? rgs_emit<uint16_t, kSqrtSoftplus>(stream, x, w, lg, bias, ids, wts, sorted, off, cnt, inv,
                                                           (int)m, (int)ne, (int)K, (int)nl, (int)k, norm, factor,
                                                           (int)base, prot, in->ctr)
                       : rgs_emit<float, kSqrtSoftplus>(stream, x, w, lg, bias, ids, wts, sorted, off, cnt, inv,
                                                        (int)m, (int)ne, (int)K, (int)nl, (int)k, norm, factor,
                                                        (int)base, prot, in->ctr);
    if (rc >= 0) {
        g_rgs_launch.fetch_add(1, std::memory_order_relaxed);
        g_rgs_rows.fetch_add(m, std::memory_order_relaxed);
    }
    return rc;
}

/* ==================================================================== embed_lookup_planes
 * One thread an element of a gathered row, rows over grid.y. The code rows are read in place, at
 * id * rb bytes as libref reads them; only the gathered rows are touched. */

__global__ __launch_bounds__(256) void embed_planes_kernel(
    const int32_t* tok, long ts, const uint8_t* codes, long rb, uint32_t cdt, const void* scale,
    uint32_t sdt, long sp, long ss, const void* table, uint32_t tdt, uint16_t* x, long xp, int E,
    int G, long V, long voff, float mult, long M) { R4D_IKA_ENTRY();
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= E) return;
    for (long m = blockIdx.y; m < M; m += gridDim.y) {
        const long id = (long)tok[m * ts] - voff;
        float val = 0.0f;
        if (id >= 0 && id < V) {
            const uint8_t* row = codes + id * rb;
            int code;
            if (cdt == RAD_I8) {
                code = (int)(int8_t)row[i];
            } else if (cdt == RAD_U8) {
                code = (int)row[i];
            } else {
                const int u = (row[i >> 1] >> ((i & 1) * 4)) & 15;
                code = cdt == RAD_I4 ? (u ^ 8) - 8 : u;
            }
            const float v = table ? ds_ld(tdt, table, code) : (float)code;
            val = v * ds_ld(sdt, scale, id * sp + (long)(i / G) * ss) * mult;
        }
        x[m * xp + i] = ds_bf16(val);
    }
}

bool float_plane_dt(uint32_t dt) {
    return dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16 || dt == RAD_F8E4M3 || dt == RAD_E8M0;
}
bool table_dt(uint32_t dt) {
    return dt == RAD_F32 || dt == RAD_F16 || dt == RAD_BF16 || dt == RAD_F8E4M3 || dt == RAD_I8 ||
           dt == RAD_U8;
}

int embed_planes_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5 || !a->t[0].data || !a->t[1].data || !a->t[2].data || !a->t[4].data)
        return RAD_E_INVAL;
    const RadTensor &tok = a->t[0], &cd = a->t[1], &sc = a->t[2], &tb = a->t[3], &x = a->t[4];
    const bool tab = tb.data != nullptr;
    const long E = (long)r4d_geti_or(a, "n_embd", 0), V = (long)r4d_geti_or(a, "n_vocab", 0);
    const long G = (long)r4d_geti_or(a, "group", 0), voff = (long)r4d_geti_or(a, "vocab_offset", 0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    if (E <= 0 || E > INT_MAX || V <= 0 || G <= 0 || E % G || voff < 0 || cd.rank != 2 ||
        cd.shape[0] < V || cd.shape[1] != E || sc.rank != 2 || sc.shape[0] < V ||
        sc.shape[1] != E / G)
        return R4D_NO(RAD_E_SHAPE);
    const uint32_t cdt = cd.dtype;
    const int bits = (cdt == RAD_U4 || cdt == RAD_I4) ? 4 : (cdt == RAD_U8 || cdt == RAD_I8) ? 8 : 0;
    if (!bits) return R4D_NO(RAD_E_DTYPE);
    if (tab && (cdt == RAD_I4 || cdt == RAD_I8 || rad_tensor_numel(&tb) < (int64_t)1 << bits))
        return R4D_NO(RAD_E_DTYPE);
    if (!float_plane_dt(sc.dtype) || (tab && !table_dt(tb.dtype)) || tok.dtype != RAD_I32 ||
        x.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    const long rb = (long)rad_dtype_bytes(cdt, E);
    const long nx = (long)rad_tensor_numel(&x);
    if (rb <= 0 || nx < E || nx % E) return R4D_NO(RAD_E_SHAPE);
    const long M = nx / E;
    long xp;
    if (x.rank == 2 && x.shape[1] == E) {
        if (x.stride[1] != 1) return R4D_NO(RAD_E_SHAPE);
        xp = (long)x.stride[0];
    } else if (rad_tensor_is_contiguous(&x)) {
        xp = E;
    } else {
        return R4D_NO(RAD_E_SHAPE);
    }
    long ts;
    if (rad_tensor_numel(&tok) < M || !lin_step(tok, &ts)) return R4D_NO(RAD_E_SHAPE);
    return R4D_EMIT(stream, embed_planes_kernel, dim3((unsigned)((E + 255) / 256), grid_rows(M)),
                    dim3(256), 0, (const int32_t*)tok.data, ts, (const uint8_t*)cd.data, rb, cdt,
                    (const void*)sc.data, sc.dtype, (long)sc.stride[0], (long)sc.stride[1],
                    tab ? (const void*)tb.data : nullptr, tab ? tb.dtype : 0u, (uint16_t*)x.data,
                    xp, (int)E, (int)G, V, voff, mult, M);
}

/* ==================================================================== fused chains
 * rmsnorm_rope_roundtrip  rmsnorm -> rope_rows -> quant_roundtrip   (a window or index key)
 * rope_rows_roundtrip     rope_rows -> quant_roundtrip              (attention output, keys)
 * mhc_collapse_rmsnorm    mhc_collapse -> rmsnorm                   (a sublayer's input)
 * mhc_post_collapse_rmsnorm  mhc_post -> mhc_collapse -> rmsnorm    (a sublayer's output into the
 *                         next sublayer's input)
 * One launch each where a decode step issued two or three dependent ones. Byte-identical to the
 * chain of rows (r4d_fuse.cpp declares it, rad-kbench's fusion oracle runs it): every step forms
 * its values as its own row does and rounds where that row stores. The norm is rmsnorm_bf16's
 * (r4d_model_bf16.hip rmsnorm_body): r4d_act_threads(M, n) threads a row, column i on thread
 * i % width, the same block_sum, rsqrtf, product order and f32_to_bf16 (no NaN quieting there);
 * rope_rows' pair expressions and quant_roundtrip's lane-group amax are the rows above. */

constexpr uint32_t kFzWave = 32;
// Columns a thread loads ahead per batch in the fused norms (rows of up to kFzCols * width stay in registers).
constexpr int kFzCols = 4;

// r4d_model_bf16.hip's block_sum: the wave butterfly, then the waves' sums in order from zero.
template <uint32_t NW>
__device__ __forceinline__ float fz_block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kFzWave - 1), wave = threadIdx.x / kFzWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

// quant_group_kernel's step over a row held in `buf`: lane groups of B columns, `width` threads.
// KV (rmsnorm_rope_roundtrip_kv_store): each value is also stored at kc[col] when the row has a cache
// slot (kc not null) -- the very bf16 written to yr[col], so the cache holds what kv_store would copy.
// W8 (kv_dtype fp8_e4m3, lever KVWIN): the cache slot is the fp8 window slot (ds_w8_*, r4d_ds41_qrt.h) at
// kc's bytes -- each wave's 32 columns are one block (width and n multiples of 32), packed from the bf16
// the step stores in y, with window_store's arithmetic (its scale from the wave's amax of those bf16).
template <int B, bool KV = false, bool W8 = false>
__device__ __forceinline__ void fz_quant_group(const uint16_t* buf, uint16_t* yr, int n,
                                               uint32_t width, int fmt, int sc,
                                               uint16_t* kc = nullptr) {
    for (int base = 0; base < n; base += (int)width) {
        const int col = base + (int)threadIdx.x;
        const bool in = col < n;
        const float v = in ? ds_bf(buf[col]) : 0.0f;
        float a = fabsf(v);
        a = a == a ? a : 0.0f;
        const float amax = r4d_xor_max<B>(a);
        [[maybe_unused]] uint16_t ow = 0;
        if (in) {
            const uint16_t o = ds_bf16(roundtrip(v, block_scale(amax, fmt, sc), fmt));
            yr[col] = o;
            if constexpr (KV && !W8) {
                if (kc) kc[col] = o;
            }
            ow = o;
        }
        if constexpr (W8) {   /* every lane of the wave, in or not (a block is all in or all out) */
            const float ov = ds_bf(ow);
            const uint32_t e = ds_w8_scale(r4d_xor_max<32>(ds_amax_term(ov)));
            if (kc && in) {
                unsigned char* k8 = (unsigned char*)kc;
                k8[col] = (unsigned char)ds_w8_code(ov, e);
                if ((col & 31) == 0) k8[n + col / 32] = (unsigned char)e;
            }
        }
    }
}

// quant_serial_kernel's step: one thread a block (KV as fz_quant_group).
template <bool KV = false>
__device__ __forceinline__ void fz_quant_serial(const uint16_t* buf, uint16_t* yr, int n, int block,
                                                uint32_t width, int fmt, int sc,
                                                uint16_t* kc = nullptr) {
    for (int b = (int)threadIdx.x; b < n / block; b += (int)width) {
        const uint16_t* p = buf + (long)b * block;
        float amax = 0;
        for (int i = 0; i < block; ++i) amax = ds_max(amax, fabsf(ds_bf(p[i])));
        const float s = block_scale(amax, fmt, sc);
        for (int i = 0; i < block; ++i) {
            const uint16_t o = ds_bf16(roundtrip(ds_bf(p[i]), s, fmt));
            yr[(long)b * block + i] = o;
            if constexpr (KV) {
                if (kc) kc[(long)b * block + i] = o;
            }
        }
    }
}

/* rmsnorm_rope_roundtrip_kv_store's cache (kv_store_bf16's operands at one head): the row's slot from the
 * slot mapping, the paged cache [n_blocks, 1, block_size, n] (an entry's K half then its V half, kv_store's
 * layout at n_head_kv 1 and head_dim n / 2). FzKvNone is the plain chain's (no cache). */
struct FzKv {
    const int32_t* slot;
    uint16_t* kv;
    uint32_t bs, n_blocks;
    uint32_t slotb;   /* W8: the fp8 window slot's bytes (ds_w8_bytes(n)) */
};
struct FzKvNone {};

/* rmsnorm_rope_roundtrip: one workgroup a row (rmsnorm's grid and width); the row passes between
 * the steps through LDS as the bf16 each step stores.
 * KV (rmsnorm_rope_roundtrip_kv_store, lever K1 fold c): the quant step also writes each bf16 it stores in
 * y to the row's cache slot, as kv_store_bf16 copies y's row there (kv_store_kernel: slot s < 0 or
 * s / block_size >= n_blocks stores nothing; else entry (s / block_size, s % block_size) of the one head,
 * K = y[0, n/2) at [0, n/2), V = y[n/2, n) at [n/2, n)): the same bytes, without the second launch. */
template <bool WF32, uint32_t THR, bool KV = false, bool W8 = false>
__global__ __launch_bounds__(THR) void norm_rope_rt_kernel(
    const uint16_t* __restrict__ x, uint32_t x_ld, const void* __restrict__ w,
    uint16_t* __restrict__ y, uint32_t y_ld, uint32_t n, float eps, float wadd, float mult,
    uint32_t M, FzRope r, int block, int fmt, int sc, std::conditional_t<KV, FzKv, FzKvNone> kv) { R4D_IKA_ENTRY_X();
#ifdef R4D_EMU_HOST   /* tests/r4d_emu: the launch's dynamic LDS (no extern __shared__ on the host) */
    uint16_t* const fz_row = (uint16_t*)emu_dyn_lds();
#else
    extern __shared__ uint16_t fz_row[];
#endif
    __shared__ float lds[THR / kFzWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;   // a row the launch added past the full-card cliff (r4d_cliff_rows)
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;

    // rmsnorm_body's sums; the row and its gains are loaded a batch of kFzCols columns ahead, and
    // stay in registers across the barrier when the row is one batch (n <= kFzCols * THR).
    const bool one = n <= (uint32_t)kFzCols * THR;
    float xv[kFzCols], gv[kFzCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint16_t xb[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            xv[c] = bf16_to_f32(xb[c]);
            if (i0 + (uint32_t)c * THR < n) ss += xv[c] * xv[c];
        }
    }
    ss = fz_block_sum<THR / kFzWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    if (one) {
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) fz_row[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            const float g = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
            fz_row[i] = f32_to_bf16(bf16_to_f32(xr[i]) * s * (g + wadd) * mult);
        }
    }
    __syncthreads();

    const long p = r.pos[(long)m * r.ps];
    if (p >= 0 && p < r.tab_rows)
        for (int q = (int)threadIdx.x; q < r.pairs; q += (int)THR) {
            const int hh = q / r.half, i = q - hh * r.half;
            const long ia = (long)hh * r.hd + r.off + (r.gptj ? 2 * i : i);
            const long ib = r.gptj ? ia + 1 : ia + r.half;
            const float c = r.cs[p * r.cp + (long)i * r.cst];
            const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
            const float va = ds_bf(fz_row[ia]), vb = ds_bf(fz_row[ib]);
            fz_row[ia] = ds_bf16(va * c - vb * sn);
            fz_row[ib] = ds_bf16(va * sn + vb * c);
        }
    __syncthreads();

    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    uint16_t* kc = nullptr;
    if constexpr (KV && !W8) {   // kv_store_kernel's slot rule and address, kv_heads 1, 2 * head_dim = n
        const int32_t s = kv.slot[m];
        if (s >= 0 && (uint32_t)s / kv.bs < kv.n_blocks)
            kc = kv.kv + ((size_t)((uint32_t)s / kv.bs) * kv.bs + (size_t)((uint32_t)s % kv.bs)) * n;
    } else if constexpr (KV && W8) {   // the same slot rule; the slot is ds_w8_bytes(n) bytes
        const int32_t s = kv.slot[m];
        if (s >= 0 && (uint32_t)s / kv.bs < kv.n_blocks)
            kc = (uint16_t*)((unsigned char*)kv.kv +
                             ((size_t)((uint32_t)s / kv.bs) * kv.bs + (size_t)((uint32_t)s % kv.bs)) * kv.slotb);
    } else {
        (void)kv;
    }
    if constexpr (W8) {   /* the launch admits W8 at a lane-group block only, n a multiple of 32 */
        switch (block) {
            case 1: fz_quant_group<1, KV, true>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
            case 2: fz_quant_group<2, KV, true>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
            case 4: fz_quant_group<4, KV, true>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
            case 8: fz_quant_group<8, KV, true>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
            case 16: fz_quant_group<16, KV, true>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
            default: fz_quant_group<32, KV, true>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
        }
        if (kc) {   /* the slot's pad bytes past the scales */
            unsigned char* k8 = (unsigned char*)kc;
            for (uint32_t z = n + n / 32 + threadIdx.x; z < kv.slotb; z += THR) k8[z] = 0;
        }
        return;
    }
    switch (block) {
        case 1: fz_quant_group<1, KV>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
        case 2: fz_quant_group<2, KV>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
        case 4: fz_quant_group<4, KV>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
        case 8: fz_quant_group<8, KV>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
        case 16: fz_quant_group<16, KV>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
        case 32: fz_quant_group<32, KV>(fz_row, yr, (int)n, THR, fmt, sc, kc); break;
        default: fz_quant_serial<KV>(fz_row, yr, (int)n, block, THR, fmt, sc, kc); break;
    }
}

/* rope_rows_roundtrip, in place: quant_group_kernel's grid (a thread a column, rows over grid.y);
 * a rotated column takes its gptj partner (the other lane of its even/odd pair, the same wave)
 * by a shuffle, forms its half of rope_rows' pair and rounds it as rope_rows stores it. */
template <int B>
__global__ __launch_bounds__(256) void rope_rt_kernel(uint16_t* x, long xp, int n, long rows,
                                                      FzRope r, int fmt, int sc) { R4D_IKA_ENTRY();
    const int col = blockIdx.x * 256 + threadIdx.x;
    const bool in = col < n;
    const int hh = col / r.hd, d = col - hh * r.hd - r.off;
    const bool rot = in && d >= 0 && d < 2 * r.half;
    const int i = d >> 1;
    const bool is_a = (d & 1) == 0;
    for (long m = blockIdx.y; m < rows; m += gridDim.y) {
        uint16_t* row = x + m * xp;
        const uint32_t h = in ? (uint32_t)row[col] : 0u;
        const uint32_t hp = (uint32_t)__shfl_xor((int)h, 1);
        float v = ds_bf((uint16_t)h);
        const long p = r.pos[m * r.ps];
        if (rot && p >= 0 && p < r.tab_rows) {
            const float c = r.cs[p * r.cp + (long)i * r.cst];
            const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
            const float va = is_a ? v : ds_bf((uint16_t)hp), vb = is_a ? ds_bf((uint16_t)hp) : v;
            v = ds_bf(is_a ? ds_bf16(va * c - vb * sn) : ds_bf16(va * sn + vb * c));
        }
        float a = fabsf(v);
        a = a == a ? a : 0.0f;
        const float amax = r4d_xor_max<B>(a);
        if (in) row[col] = ds_bf16(roundtrip(v, block_scale(amax, fmt, sc), fmt));
    }
}

/* tp8 loop P32d (PREFILL round 26; RADIANCE_DS41_ROPE8=1): rope_rt_kernel<32>'s work a thread EIGHT columns (one 16-byte load and
 * store) instead of one: a gptj pair is two adjacent columns of the thread (off and hd even), a block of 32 is four threads
 * (r4d_xor_max<4> of their 8-column maxima = the 32-lane butterfly's maximum: fmaxf of NaN-free non-negative values in any
 * grouping), every value the same expressions -- the same bytes. One card (P32's cell, the attention's fold at prefill heights):
 * the one-column form moved ~260 GB/s. Taken for blocks of 32 with n, the row pitch and the plane on 8 bf16. */
__global__ __launch_bounds__(256) void rope_rt8_kernel(uint16_t* x, long xp, int n, long rows, FzRope r, int fmt, int sc) { R4D_IKA_ENTRY();
    const int c8 = (blockIdx.x * 256 + threadIdx.x) * 8;
    const bool in = c8 < n;
    const int hh = c8 / r.hd, d8 = c8 - hh * r.hd - r.off;
    for (long m = blockIdx.y; m < rows; m += gridDim.y) {
        uint16_t* row = x + m * xp;
        const uint4 w = in ? *(const uint4*)(row + c8) : make_uint4(0u, 0u, 0u, 0u);
        const uint32_t ww[4] = { w.x, w.y, w.z, w.w };
        float v[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            v[2 * e] = ds_bf((uint16_t)(ww[e] & 0xffffu));
            v[2 * e + 1] = ds_bf((uint16_t)(ww[e] >> 16));
        }
        const long p = r.pos[m * r.ps];
        if (p >= 0 && p < r.tab_rows) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int d = d8 + 2 * e;
                if (in && d >= 0 && d < 2 * r.half) {
                    const int i = d >> 1;
                    const float c = r.cs[p * r.cp + (long)i * r.cst];
                    const float sn = r.sign * r.cs[p * r.cp + (long)(r.half + i) * r.cst];
                    const float va = v[2 * e], vb = v[2 * e + 1];
                    v[2 * e] = ds_bf(ds_bf16(va * c - vb * sn));
                    v[2 * e + 1] = ds_bf(ds_bf16(va * sn + vb * c));
                }
            }
        }
        float a = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float t = fabsf(v[e]);
            t = t == t ? t : 0.0f;
            a = fmaxf(a, t);
        }
        const float amax = r4d_xor_max<4>(a);
        const float bs = block_scale(amax, fmt, sc);
        uint32_t o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            o[e] = (uint32_t)ds_bf16(roundtrip(v[2 * e], bs, fmt)) | ((uint32_t)ds_bf16(roundtrip(v[2 * e + 1], bs, fmt)) << 16);
        if (in) *(uint4*)(row + c8) = make_uint4(o[0], o[1], o[2], o[3]);
    }
}
int rope8_env() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DS41_ROPE8");
        const int on = e && !std::strcmp(e, "1");
        if (on)
            std::fprintf(stderr, "radiance: libr4d: RADIANCE_DS41_ROPE8=1 (tp8 loop P32d): rope_rows_roundtrip at blocks of 32 runs "
                                 "eight columns a thread (the same bytes)\n");
        return on ? 1 : 0;
    }();
    return v;
}
std::atomic<int> g_rope8_pin{-1};
[[maybe_unused]] const int g_rope8_load = rope8_env();

/* mhc_collapse_rmsnorm: rmsnorm's grid and width. A thread owns columns i = tid + k*THR (the norm's
 * own split), collapses each (the streams from zero in order, collapse_kernel's sum), stores it and
 * folds the stored bf16 into the sum of squares in that order. Loads go out a batch of kFzCols
 * columns at a time (every stream and the gains, at clamped addresses) ahead of the arithmetic; when
 * the row is one batch (n <= kFzCols * THR: DeepSeek-V4.1's decode rows) the collapsed values and
 * gains stay in registers across the barrier, else the second pass re-reads what this thread stored
 * (as add_rmsnorm_body), so no barrier between. HCM: streams loaded a column (4 or 8, hc <= HCM). */
template <bool WF32, uint32_t THR, int HCM>
__global__ __launch_bounds__(THR) void collapse_norm_kernel(
    const uint16_t* __restrict__ h, long hp, const float* __restrict__ pre, long pp, long pst,
    uint16_t* __restrict__ x, uint32_t x_ld, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t y_ld, uint32_t n, int hc, float eps, float wadd, float mult, uint32_t M) { R4D_IKA_ENTRY();
    __shared__ float lds[THR / kFzWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const uint16_t* __restrict__ hr = h + (long)m * hp;
    const float* __restrict__ pr = pre + (long)m * pp;
    uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    float pj[HCM];
#pragma unroll
    for (int j = 0; j < HCM; ++j) pj[j] = pr[(long)(j < hc ? j : hc - 1) * pst];
    const bool one = n <= (uint32_t)kFzCols * THR;
    float xv[kFzCols], gv[kFzCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint32_t hv[kFzCols][HCM];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
#pragma unroll
            for (int j = 0; j < HCM; ++j) hv[c][j] = hr[(long)(j < hc ? j : hc - 1) * n + i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            float v = 0;
#pragma unroll
            for (int j = 0; j < HCM; ++j)
                if (j < hc) v += pj[j] * ds_bf((uint16_t)hv[c][j]);
            const uint16_t b = ds_bf16(v);
            xv[c] = bf16_to_f32(b);
            if (i < n) {
                xr[i] = b;
                ss += xv[c] * xv[c];
            }
        }
    }
    ss = fz_block_sum<THR / kFzWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    if (one) {
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
        return;
    }
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint16_t xb[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(bf16_to_f32(xb[c]) * s * (gv[c] + wadd) * mult);
        }
    }
}

/* mhc_post_collapse_rmsnorm: collapse_norm_kernel with mhc_post in front, column-local. For each of
 * its columns a thread forms every output stream k as mhc_post_kernel does (from zero, comb[j][k] *
 * h[j] over the source streams in order, then + post[k] * x, f32_to_bf16), stores it, and collapses
 * the stored values (collapse_kernel's sum over k, ds_bf16); the norm is collapse_norm_kernel's. */
template <bool WF32, uint32_t THR, int HCM>
__global__ __launch_bounds__(THR) void post_collapse_norm_kernel(
    const uint16_t* __restrict__ xin, long xp, const uint16_t* __restrict__ h, long hp,
    const float* __restrict__ post, long ptp, const float* __restrict__ comb, long cbp,
    uint16_t* __restrict__ out, long op, const float* __restrict__ pre, long pp, long pst,
    uint16_t* __restrict__ x, uint32_t x_ld, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t y_ld, uint32_t n, int hc, float eps, float wadd, float mult, uint32_t M) { R4D_IKA_ENTRY();
    __shared__ float lds[THR / kFzWave];
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const uint16_t* __restrict__ xir = xin + (long)m * xp;
    const uint16_t* __restrict__ hr = h + (long)m * hp;
    uint16_t* __restrict__ orow = out + (long)m * op;
    const float* __restrict__ pr = pre + (long)m * pp;
    uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    float pj[HCM], pk[HCM], cb[HCM][HCM];
#pragma unroll
    for (int k = 0; k < HCM; ++k) {
        const int kk = k < hc ? k : hc - 1;
        pj[k] = pr[(long)kk * pst];
        pk[k] = post[(long)m * ptp + kk];
#pragma unroll
        for (int j = 0; j < HCM; ++j) cb[j][k] = comb[(long)m * cbp + (j < hc ? j : hc - 1) * hc + kk];
    }
    const bool one = n <= (uint32_t)kFzCols * THR;
    float xv[kFzCols], gv[kFzCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint32_t hv[kFzCols][HCM], xi[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
#pragma unroll
            for (int j = 0; j < HCM; ++j) hv[c][j] = hr[(long)(j < hc ? j : hc - 1) * n + i];
            xi[c] = xir[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            float v = 0;
#pragma unroll
            for (int k = 0; k < HCM; ++k) {
                if (k >= hc) continue;
                float u = 0;
#pragma unroll
                for (int j = 0; j < HCM; ++j)
                    if (j < hc) u += cb[j][k] * bf16_to_f32((uint16_t)hv[c][j]);
                u += pk[k] * bf16_to_f32((uint16_t)xi[c]);
                const uint16_t o = f32_to_bf16(u);
                if (i < n) orow[(long)k * n + i] = o;
                v += pj[k] * ds_bf(o);
            }
            const uint16_t b = ds_bf16(v);
            xv[c] = bf16_to_f32(b);
            if (i < n) {
                xr[i] = b;
                ss += xv[c] * xv[c];
            }
        }
    }
    ss = fz_block_sum<THR / kFzWave>(ss, lds);
    const float s = rsqrtf(ss / (float)n + eps);
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    if (one) {
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
        return;
    }
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kFzCols * THR) {
        uint16_t xb[kFzCols];
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kFzCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(bf16_to_f32(xb[c]) * s * (gv[c] + wadd) * mult);
        }
    }
}

// quant_rt_body's format and scale names.
bool fz_quant_fmt(const RadArgs* a, int* fmt, int* sc) {
    const char* fs = r4d_gets(a, "format");
    const char* ss = r4d_gets(a, "scale");
    if (!fs || !ss) return false;
    if (!std::strcmp(fs, "fp8_e4m3")) *fmt = kFp8;
    else if (!std::strcmp(fs, "fp4_e2m1")) *fmt = kFp4;
    else return false;
    if (!std::strcmp(ss, "ue8m0")) *sc = kUe8m0;
    else if (!std::strcmp(ss, "e4m3") && *fmt == kFp4) *sc = kE4m3;
    else if (!std::strcmp(ss, "f32") && *fmt == kFp8) *sc = kScaleF32;
    else return false;
    return true;
}

// rope_rows_body's checks over `x` [M, nh*hd] and its cos_sin / positions, into an FzRope.
int fz_rope_args(const RadArgs* a, const RadTensor& x, const RadTensor& cs, const RadTensor& pos,
                 long M, FzRope* r) {
    const long nh = (long)r4d_geti_or(a, "n_head", 0), hd = (long)r4d_geti_or(a, "head_dim", 0);
    const long rot = (long)r4d_geti_or(a, "rot", 0), off = (long)r4d_geti_or(a, "off", -1);
    const long inv = (long)r4d_geti_or(a, "inverse", -1);
    const char* mode = r4d_gets(a, "mode");
    if (!mode) return RAD_E_UNSUPPORTED;
    const bool gptj = !std::strcmp(mode, "gptj");
    if (!gptj && std::strcmp(mode, "neox")) return RAD_E_UNSUPPORTED;
    if (nh < 1 || hd < 2 || rot < 2 || (rot & 1) || off < 0 || off + rot > hd ||
        (inv != 0 && inv != 1) || nh * hd > INT_MAX)
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || cs.dtype != RAD_F32 || pos.dtype != RAD_I32) return RAD_E_DTYPE;
    long ps;
    if (!mat(x, RAD_BF16, M, nh * hd) || !cs.data || cs.rank != 2 || cs.shape[1] != rot ||
        rad_tensor_numel(&pos) != M || !lin_step(pos, &ps))
        return RAD_E_SHAPE;
    r->cs = (const float*)cs.data;
    r->cp = (long)cs.stride[0];
    r->cst = (long)cs.stride[1];
    r->pos = (const int32_t*)pos.data;
    r->ps = ps;
    r->tab_rows = (long)cs.shape[0];
    r->hd = (int)hd;
    r->half = (int)(rot / 2);
    r->off = (int)off;
    r->pairs = (int)(nh * (rot / 2));
    r->gptj = gptj ? 1 : 0;
    r->sign = inv ? -1.0f : 1.0f;
    return RAD_OK;
}

// Most columns the fused norm holds in LDS (the row as bf16); the row's constraint says the same.
constexpr long kFzMaxN = 16384;

/* x [M, n] (rows as rmsnorm reads them), w, y [M, n], cos_sin, positions; with KV
 * (rmsnorm_rope_roundtrip_kv_store) then kv_store_bf16's slot_mapping [M] and kv_cache [n_blocks, 1,
 * block_size, n] (block_size and kv_dtype bf16 as kv_store declares them). */
template <bool KV>
int norm_rope_rt_launch(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < (KV ? 7 : 5)) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2], &cs = a->t[3], &pos = a->t[4];
    if (!x.data || !w.data || !y.data || !cs.data || !pos.data) return R4D_NO(RAD_E_INVAL);
    if (x.dtype != RAD_BF16 || y.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(&w)) return R4D_NO(RAD_E_STRIDE);
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0 || n > kFzMaxN || r4d_numel(&w) < n) return R4D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r4d_rows_pitch(&x, n, &M);
    if (x_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || M > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    FzRope r;
    const int rc = fz_rope_args(a, y, cs, pos, (long)M, &r);
    if (rc != RAD_OK) return R4D_NO(rc);
    if ((long long)r.hd * (r.pairs / r.half) != n) return R4D_NO(RAD_E_SHAPE);
    const long long block = r4d_geti_or(a, "block", 0);
    int fmt, sc;
    if (!fz_quant_fmt(a, &fmt, &sc)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (block < 1 || n % block) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    std::conditional_t<KV, FzKv, FzKvNone> kv{};
    [[maybe_unused]] bool w8 = false;
    if constexpr (KV) {   /* kv_store_geom's checks at one head of n / 2 (K) + n / 2 (V) */
        const RadTensor &sm = a->t[5], &kc = a->t[6];
        if (!sm.data || !kc.data) return R4D_NO(RAD_E_INVAL);
        const char* kd = r4d_gets(a, "kv_dtype");
        const long long bs = r4d_geti_or(a, "block_size", 0);
        /* kv_dtype fp8_e4m3 (lever KVWIN): the fp8 window slot, ds_w8_bytes(n) bytes in bf16 slots; packed
         * a wave a block, so n a multiple of 32 and the round trip's block a lane group (power of two <= 32) */
        w8 = kd && !std::strcmp(kd, "fp8_e4m3");
        if (!kd || (std::strcmp(kd, "bf16") && !w8)) return R4D_NO(RAD_E_UNSUPPORTED);
        if (w8 && (n % 32 || block > 32 || (block & (block - 1)))) return R4D_NO(RAD_E_UNSUPPORTED);
        if (sm.dtype != RAD_I32 || kc.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
        if (!r4d_contig(&sm) || !r4d_contig(&kc)) return R4D_NO(RAD_E_STRIDE);
        if (bs <= 0 || (n & 1) || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[2] != bs ||
            kc.shape[3] != (w8 ? ds_w8_bytes((int)n) / 2 : n) || kc.shape[0] < 1 || kc.shape[0] > UINT32_MAX ||
            bs > UINT32_MAX || r4d_numel(&sm) != M)
            return R4D_NO(RAD_E_SHAPE);
        kv.slot = (const int32_t*)sm.data;
        kv.kv = (uint16_t*)kc.data;
        kv.bs = (uint32_t)bs;
        kv.n_blocks = (uint32_t)kc.shape[0];
        kv.slotb = (uint32_t)ds_w8_bytes((int)n);
    }
    const int thr = r4d_act_threads((int)M, (int)n);
    const uint32_t lds = (uint32_t)(n * sizeof(uint16_t));
#define FZ_NRQ(WF, T, W)                                                                          \
    R4D_EMIT(stream, (norm_rope_rt_kernel<WF, T, KV, W>), dim3(r4d_cliff_rows((uint32_t)M, 1, T / kFzWave)), \
             dim3(T), lds, (const uint16_t*)x.data, (uint32_t)x_ld, (const void*)w.data,          \
             (uint16_t*)y.data, (uint32_t)y.stride[0], (uint32_t)n, eps, wadd, mult, (uint32_t)M, \
             r, (int)block, fmt, sc, kv)
#define FZ_NRQ_W(WF, W)                                                                           \
    return thr == 1024 ? FZ_NRQ(WF, 1024, W) : thr == 512 ? FZ_NRQ(WF, 512, W) : FZ_NRQ(WF, 256, W)
    if constexpr (KV) {
        if (w8) {
            if (w.dtype == RAD_F32) FZ_NRQ_W(true, true);
            FZ_NRQ_W(false, true);
        }
    }
    if (w.dtype == RAD_F32) FZ_NRQ_W(true, false);
    FZ_NRQ_W(false, false);
#undef FZ_NRQ_W
#undef FZ_NRQ
}
int norm_rope_rt_body(const RadArgs* a, RadStream stream) { return norm_rope_rt_launch<false>(a, stream); }
int norm_rope_rt_kv_body(const RadArgs* a, RadStream stream) { return norm_rope_rt_launch<true>(a, stream); }

/* x [M, n_head*head_dim] in place, cos_sin, positions. The lane-group form needs a gptj pair on
 * one wave's even/odd lanes (head_dim and off even, the row's constraints) and a power-of-two block
 * up to 32; any other geometry runs rope_rows then quant_roundtrip as two launches, the same bits. */
int rope_rt_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data) return RAD_E_INVAL;
    const RadTensor &x = a->t[0], &cs = a->t[1], &pos = a->t[2];
    if (x.rank != 2 || x.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long M = (long)x.shape[0];
    FzRope r;
    const int rc = fz_rope_args(a, x, cs, pos, M, &r);
    if (rc != RAD_OK) return R4D_NO(rc);
    const long n = (long)r.hd * (r.pairs / r.half);
    const long block = (long)r4d_geti_or(a, "block", 0);
    int fmt, sc;
    if (!fz_quant_fmt(a, &fmt, &sc)) return R4D_NO(RAD_E_UNSUPPORTED);
    if (block < 1 || n % block) return R4D_NO(RAD_E_SHAPE);
    const bool lanes = r.gptj && !(r.hd & 1) && !(r.off & 1) &&
                       (block == 1 || block == 2 || block == 4 || block == 8 || block == 16 ||
                        block == 32);
    if (!lanes) {
        const int rr = rope_rows_body(a, stream);
        if (rr != RAD_OK) return rr;
        RadParam pp[24];
        if (a->n_p > 23) return R4D_NO(RAD_E_INVAL);
        std::memcpy(pp, a->p, sizeof(RadParam) * (size_t)a->n_p);
        RadParam& pn = pp[a->n_p];
        std::memset(&pn, 0, sizeof pn);
        pn.key = "n";
        pn.kind = RAD_P_INT;
        pn.ival = n;
        RadArgs q = *a;
        q.p = pp;
        q.n_p = a->n_p + 1;
        return quant_rt_body(&q, stream);
    }
    uint16_t* xd = (uint16_t*)x.data;
    const long xp = (long)x.stride[0];
    {   /* tp8 loop P32d */
        const int pin = g_rope8_pin.load(std::memory_order_relaxed);
        if ((pin >= 0 ? pin : rope8_env()) > 0 && block == 32 && n % 8 == 0 && r.hd % 8 == 0 && xp % 8 == 0 && x.stride[1] == 1 &&
            ((uintptr_t)xd & 15u) == 0) {
            const unsigned gx8 = (unsigned)((n / 8 + 255) / 256);
            const dim3 g8(gx8, ds_dodge_gy(gx8, grid_rows(M), 8));
            return R4D_EMIT(stream, rope_rt8_kernel, g8, dim3(256), 0, xd, xp, (int)n, M, r, fmt, sc);
        }
    }
    const unsigned gxr = (unsigned)((n + 255) / 256);
    const dim3 grid(gxr, ds_dodge_gy(gxr, grid_rows(M), 8));   /* lever WQ */
#define FZ_RRT(B)                                                                                  \
    case B:                                                                                        \
        return R4D_EMIT(stream, rope_rt_kernel<B>, grid, dim3(256), 0, xd, xp, (int)n, M, r, fmt, sc)
    switch (block) {
        FZ_RRT(1);
        FZ_RRT(2);
        FZ_RRT(4);
        FZ_RRT(8);
        FZ_RRT(16);
        FZ_RRT(32);
        default: break;
    }
#undef FZ_RRT
    return R4D_NO(RAD_E_SHAPE);
}

/* h [M, hc*n], pre [M, hc] f32, x [M, n] (the collapsed row, stored), w, y (rows as rmsnorm). */
int collapse_norm_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || hc < 2 || hc > 8 || hc * n > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &h = a->t[0], &pre = a->t[1], &x = a->t[2], &w = a->t[3], &y = a->t[4];
    if (!w.data || !y.data) return R4D_NO(RAD_E_INVAL);
    if (h.rank != 2 || h.shape[0] < 1) return R4D_NO(RAD_E_SHAPE);
    const long rows = h.shape[0];
    if (h.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16 || y.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(h, RAD_BF16, rows, hc * n) || !mat_s(pre, RAD_F32, rows, hc) ||
        !mat(x, RAD_BF16, rows, n))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(&w)) return R4D_NO(RAD_E_STRIDE);
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(&w) < n) return R4D_NO(RAD_E_SHAPE);
    long long My = 0;
    const long long y_ld = r4d_rows_pitch(&y, n, &My);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < rows || rows > INT_MAX) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    const int thr = r4d_act_threads((int)rows, (int)n);
#define FZ_CN(WF, T, HM)                                                                           \
    R4D_EMIT(stream, (collapse_norm_kernel<WF, T, HM>),                                            \
             dim3(r4d_cliff_rows((uint32_t)rows, 1, T / kFzWave)), dim3(T), 0,                     \
             (const uint16_t*)h.data, (long)h.stride[0], (const float*)pre.data,                   \
             (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data, (uint32_t)x.stride[0],   \
             (const void*)w.data, (uint16_t*)y.data, (uint32_t)y_ld, (uint32_t)n, (int)hc, eps,    \
             wadd, mult, (uint32_t)rows)
#define FZ_CN_T(WF, HM)                                                                            \
    return thr == 1024 ? FZ_CN(WF, 1024, HM) : thr == 512 ? FZ_CN(WF, 512, HM) : FZ_CN(WF, 256, HM)
#define FZ_CN_W(WF)                                                                                \
    do { if (hc <= 4) FZ_CN_T(WF, 4); FZ_CN_T(WF, 8); } while (0)
    if (w.dtype == RAD_F32) FZ_CN_W(true);
    FZ_CN_W(false);
#undef FZ_CN_W
#undef FZ_CN_T
#undef FZ_CN
}

/* x [M, n], h [M, hc*n], post [M, hc], comb [M, hc*hc] f32 -> out [M, hc*n] (mhc_post_bf16's
 * checks: n <= 16384, out not h); pre, xc, w, y as mhc_collapse_rmsnorm. */
/* ---- the all-reduce epilogue form of mhc_post_collapse_rmsnorm (fusion 2's single-card step, 6.11.2026).
 * The one-shot all-reduce splits a pass's M * n / 8 sixteen-byte words over its nb workgroups (w = ceil(words / nb) a
 * block, never row-aligned). As its epilogue: each block forms its words' out[k] and xc exactly as
 * post_collapse_norm_kernel forms a column (from zero, comb[j][k] * h[j] in j order, + post[k] * x, f32_to_bf16; then
 * pre[k] * out[k] in k order, ds_bf16) and stores them; per row a counter (zeroed by the row's last arriver) takes the
 * block's words in it, and the block that completes a row runs the norm over the stored xc in the norm kernel's own
 * order: virtual thread v of THR = r4d_act_threads(M, n) adds columns v, v + THR, .. from zero, each wave of 32 virtual
 * threads closes with r4d_wave_sum (run on real lanes), the wave sums are added from zero in order, then rsqrtf and
 * y = f32_to_bf16(xv * s * (g + wadd) * mult). Same values in the same order: post_collapse_norm_kernel's bytes, at any
 * nb and block width. Here without the collective (x is the reduced input); tests/r4d_emu holds it to the norm kernel. */
template <int HCM>
__global__ __launch_bounds__(256) void pcn_words_kernel(const uint16_t* __restrict__ xin, long xp,
                                                        r4d_ds41pcn::PcnArgs A) { R4D_IKA_ENTRY();
    __shared__ float vss[1024];          // the virtual threads' partial sums (THR <= 1024)
    __shared__ float wsum[32];
    __shared__ unsigned s_done;
    const uint32_t wpr = A.n / 8, words = A.M * wpr, nb = gridDim.x, b = blockIdx.x, nt = 256;
    const uint32_t wd = (words + nb - 1) / nb, w0 = b * wd, w1 = min(words, w0 + wd);
    for (uint32_t k = w0 + threadIdx.x; k < w1; k += nt) {
        const uint32_t m = k / wpr, i0 = (k - m * wpr) * 8;
        uint16_t xw[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) xw[e] = xin[(long)m * xp + i0 + (uint32_t)e];
        r4d_ds41pcn::pcn_word<HCM>(A, m, i0, xw);
    }
    r4d_ds41pcn::pcn_arrive(A, w0, w1, nt, vss, wsum, &s_done);
}

/* tests: post_collapse_norm_body's operands, the epilogue form over `nb` blocks of 256 (ctr: M zeroed words) */
int pcn_words_body(const RadArgs* a, RadStream stream, int nb, unsigned* ctr) {
    if (!a || a->n_t < 9 || nb < 1 || !ctr) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || n > 16384 || n % 8 || hc < 2 || hc > 8) return RAD_E_SHAPE;
    const RadTensor &xi = a->t[0], &h = a->t[1], &po = a->t[2], &cb = a->t[3], &o = a->t[4];
    const RadTensor &pre = a->t[5], &x = a->t[6], &w = a->t[7], &y = a->t[8];
    const long rows = xi.shape[0];
    long long My = 0;
    const long long y_ld = r4d_rows_pitch(&y, n, &My);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    r4d_ds41pcn::PcnArgs A{};
    A.h = (const uint16_t*)h.data; A.hp = (long)h.stride[0];
    A.post = (const float*)po.data; A.ptp = (long)po.stride[0];
    A.comb = (const float*)cb.data; A.cbp = (long)cb.stride[0];
    A.out = (uint16_t*)o.data; A.op = (long)o.stride[0];
    A.pre = (const float*)pre.data; A.pp = (long)pre.stride[0]; A.pst = (long)pre.stride[1];
    A.xc = (uint16_t*)x.data; A.x_ld = (uint32_t)x.stride[0];
    A.w = w.data; A.wf32 = w.dtype == RAD_F32 ? 1 : 0;
    A.y = (uint16_t*)y.data; A.y_ld = (uint32_t)y_ld;
    A.n = (uint32_t)n; A.M = (uint32_t)rows; A.hc = (int)hc; A.thr_v = r4d_act_threads((int)rows, (int)n);
    A.eps = eps; A.wadd = wadd; A.mult = mult; A.ctr = ctr;
    if (hc <= 4)
        return R4D_EMIT(stream, (pcn_words_kernel<4>), dim3((unsigned)nb), dim3(256), 0, (const uint16_t*)xi.data,
                        (long)xi.stride[0], A);
    return R4D_EMIT(stream, (pcn_words_kernel<8>), dim3((unsigned)nb), dim3(256), 0, (const uint16_t*)xi.data,
                    (long)xi.stride[0], A);
}

int post_collapse_norm_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 9) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    if (n <= 0 || n > 16384 || hc < 2 || hc > 8) return R4D_NO(RAD_E_SHAPE);
    const RadTensor &xi = a->t[0], &h = a->t[1], &po = a->t[2], &cb = a->t[3], &o = a->t[4];
    const RadTensor &pre = a->t[5], &x = a->t[6], &w = a->t[7], &y = a->t[8];
    if (!w.data || !y.data) return R4D_NO(RAD_E_INVAL);
    if (xi.rank != 2 || xi.shape[0] < 1 || xi.shape[0] > 65535) return R4D_NO(RAD_E_SHAPE);
    const long rows = xi.shape[0];
    if (xi.dtype != RAD_BF16 || h.dtype != RAD_BF16 || po.dtype != RAD_F32 || cb.dtype != RAD_F32 ||
        o.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16 || y.dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(xi, RAD_BF16, rows, n) || !mat(h, RAD_BF16, rows, hc * n) ||
        !mat(po, RAD_F32, rows, hc) || !mat(cb, RAD_F32, rows, hc * hc) ||
        !mat(o, RAD_BF16, rows, hc * n) || o.data == h.data || !mat_s(pre, RAD_F32, rows, hc) ||
        !mat(x, RAD_BF16, rows, n))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(&w)) return R4D_NO(RAD_E_STRIDE);
    if (w.dtype != RAD_F32 && w.dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (r4d_numel(&w) < n) return R4D_NO(RAD_E_SHAPE);
    long long My = 0;
    const long long y_ld = r4d_rows_pitch(&y, n, &My);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < rows) return R4D_NO(RAD_E_SHAPE);
    const float eps = (float)r4d_getf(a, "eps", 1e-6), wadd = (float)r4d_getf(a, "wadd", 0.0);
    const float mult = (float)r4d_getf(a, "mult", 1.0);
    const int thr = r4d_act_threads((int)rows, (int)n);
#define FZ_PCN(WF, T, HM)                                                                          \
    R4D_EMIT(stream, (post_collapse_norm_kernel<WF, T, HM>),                                       \
             dim3(r4d_cliff_rows((uint32_t)rows, 1, T / kFzWave)), dim3(T), 0,                     \
             (const uint16_t*)xi.data, (long)xi.stride[0], (const uint16_t*)h.data,                \
             (long)h.stride[0], (const float*)po.data, (long)po.stride[0],                         \
             (const float*)cb.data, (long)cb.stride[0], (uint16_t*)o.data, (long)o.stride[0],      \
             (const float*)pre.data, (long)pre.stride[0], (long)pre.stride[1], (uint16_t*)x.data,  \
             (uint32_t)x.stride[0], (const void*)w.data, (uint16_t*)y.data, (uint32_t)y_ld,        \
             (uint32_t)n, (int)hc, eps, wadd, mult, (uint32_t)rows)
#define FZ_PCN_T(WF, HM)                                                                           \
    return thr == 1024 ? FZ_PCN(WF, 1024, HM) : thr == 512 ? FZ_PCN(WF, 512, HM) : FZ_PCN(WF, 256, HM)
#define FZ_PCN_W(WF)                                                                               \
    do { if (hc <= 4) FZ_PCN_T(WF, 4); FZ_PCN_T(WF, 8); } while (0)
    if (w.dtype == RAD_F32) FZ_PCN_W(true);
    FZ_PCN_W(false);
#undef FZ_PCN_W
#undef FZ_PCN_T
#undef FZ_PCN
}

/* ==================================================================== mhc_post_collapse_rmsnorm_mix (lever M1)
 * mhc_post_collapse_rmsnorm (x, h, post, comb -> out; out, pre -> xc; xc, w -> y) and then mhc_mix over its out (fn,
 * scale, base -> pre_o, post, comb), as one op of three launches, byte-identical to the two ops (ds41-int9b, lever M1:
 * the pcn_a -> mix_a pair of DeepSeek-V4.1's attention connection; kbench2 F1R8 12.11.2026, 1874.7 -> 1384.1 us a pair
 * at 4096 rows, the bytes the image's at 4096 / 8 / 1 rows with real blk.9 / blk.10 weights).
 *  K1 (grid n / 1024 column blocks x ceil(M / 8) row groups, 256 threads): its 8 rows and its 1024-column block of
 *     each of the 4 streams, a 128-column slab at a time. Phase A: out[k][i] = f32_to_bf16(sum_j comb[j][k] h[j][i]
 *     (from zero, j ascending) + post[k] x[i]) and xc[i] = ds_bf16(sum_k pre[k] ds_bf(out[k][i])) --
 *     post_collapse_norm_kernel's column arithmetic, the same expressions -- stored, and the slab's out kept in LDS.
 *     Phase B: mhc_mix's partials over the slab: wave w the dots of stream w / 2 with parity w & 1 (dots (w & 1) + 2 q)
 *     and the sum of squares on parity 0; lane L adds columns L, L + 32, .. of the chunk in column order from zero,
 *     each product added unfused (-ffp-contract=off, as the whole file) -- mix_split_kernel's per-partial sequence
 *     (which wave forms a partial does not change its bits: mix_split_kernel's own note) -- closed with r4d_wave_sum
 *     into mix's scratch layout part[(row * 25 + j) * chunks + chunk], chunk = stream * (n / 1024) + block, the
 *     1024-column chunks of the flattened 4n row (n a multiple of 1024: a chunk never straddles two streams).
 *  FIN: the stock mix_finish_kernel<4> over those partials (pre_o, post, comb): mhc_mix's finish, the same bits.
 *  K2 (a workgroup a row): the norm over the stored xc in post_collapse_norm_kernel's order: virtual thread v of
 *     THR = r4d_act_threads(M, n) adds ds_bf(xc)^2 over columns v, v + THR, .. from zero (the stock thread v's
 *     sequence), each wave of 32 virtual threads closes with r4d_wave_sum, the wave sums are added from zero in order
 *     (fz_block_sum); y = f32_to_bf16(bf(xc) * rsqrtf(ss / n + eps) * (w + wadd) * mult).
 * post and comb are read by K1 (the previous connection's gates) and written by FIN (this connection's): FIN runs after
 * K1 on the stream (one op's launches stay ordered: Ctx::haz_mark's "cont"). hc 4, n a multiple of 1024 up to 16384, f32
 * w, mix's scratch; anything else is refused and the caller issues the pair. REV / K2 virtual-thread count: the
 * selftest's mutation controls only (r4d_i9b_mutate), never the production launch. */
constexpr int kPmHC = 4, kPmMix = kPmHC * (kPmHC + 2), kPmS = kPmMix + 1, kPmSlab = 128, kPmR = 8;

struct PmArgs {
    const uint16_t* xin; long xp;     /* [M, n] x */
    const uint16_t* h;   long hp;     /* [M, 4n] the streams before */
    const float* post;   long ptp;    /* [M, 4] */
    const float* comb;   long cbp;    /* [M, 16] */
    const float* pre;    long pp, pst;/* [M, 4] the collapse weights */
    uint16_t* out;       long op;     /* [M, 4n] the streams after */
    uint16_t* xc;        long xcp;    /* [M, n] */
    const float* fn;     long fp;     /* [24, 4n] the next mix's projection */
    float* part;                      /* [M][25][chunks] mix's scratch layout */
    int n, chunks;
    long M;
};

/* the selftest's mutation hook (r4d_i9b_mutate): 0 = the production forms; see the forms that read it */
std::atomic<int> g_i9b_mut{0};

template <bool REV>
__global__ __launch_bounds__(256) void pcn_mix_k1_kernel(PmArgs A) { R4D_IKA_ENTRY();
    constexpr int R = kPmR, WV = 8, NCG = WV / R, CPW = kPmSlab / NCG, UA = CPW / 32, JD = kPmMix / 2;
    static_assert(WV == 2 * kPmHC && WV % R == 0 && CPW % 32 == 0, "two waves a stream, R divides the 8 waves");
    __shared__ uint16_t ob[R][kPmHC][kPmSlab];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int b = blockIdx.x, n = A.n;
    const long g0 = (long)blockIdx.y * R;
    if (g0 >= A.M) return;   /* block-uniform */
    const int nr = (int)(A.M - g0 < R ? A.M - g0 : R);
    /* phase A's thread: row ra (wave % R) of the group, column group cg of the slab */
    const int ra = wave % R, cg = wave / R;
    const long ma = g0 + (ra < nr ? ra : 0);
    float pj[kPmHC], pk[kPmHC], cb[kPmHC][kPmHC];
#pragma unroll
    for (int k = 0; k < kPmHC; ++k) {
        pj[k] = A.pre[ma * A.pp + (long)k * A.pst];
        pk[k] = A.post[ma * A.ptp + k];
#pragma unroll
        for (int j = 0; j < kPmHC; ++j) cb[j][k] = A.comb[ma * A.cbp + j * kPmHC + k];
    }
    /* phase B's wave: stream kw, dots par + 2 q (q < 12), the squares on par 0 */
    const int kw = wave >> 1, par = wave & 1;
    const bool sq = par == 0;
    const float* wr = A.fn + (long)par * A.fp + (long)kw * n;
    float acc[JD + 1][R];
#pragma unroll
    for (int q = 0; q <= JD; ++q)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[q][r] = 0.0f;
#pragma unroll 1
    for (int si = 0; si < kMixCols / kPmSlab; ++si) {
        const int sb = REV ? kMixCols / kPmSlab - 1 - si : si;
        const int c0 = b * kMixCols + sb * kPmSlab;
        if (ra < nr) {
            const uint16_t* hr = A.h + ma * A.hp;
#pragma unroll
            for (int u = 0; u < UA; ++u) {
                const int cl = cg * CPW + lane + 32 * u, i = c0 + cl;
                uint32_t hv[kPmHC];
#pragma unroll
                for (int j = 0; j < kPmHC; ++j) hv[j] = hr[(long)j * n + i];
                const uint32_t xi = A.xin[ma * A.xp + i];
                float v = 0;
#pragma unroll
                for (int k = 0; k < kPmHC; ++k) {
                    float uu = 0;
#pragma unroll
                    for (int j = 0; j < kPmHC; ++j) uu += cb[j][k] * bf16_to_f32((uint16_t)hv[j]);
                    uu += pk[k] * bf16_to_f32((uint16_t)xi);
                    const uint16_t o = f32_to_bf16(uu);
                    A.out[ma * A.op + (long)k * n + i] = o;
                    ob[ra][k][cl] = o;
                    v += pj[k] * ds_bf(o);
                }
                A.xc[ma * A.xcp + i] = ds_bf16(v);
            }
        }
        __syncthreads();
#pragma unroll
        for (int u = 0; u < kPmSlab / 32; ++u) {
            const int cl = 32 * u + lane, i = c0 + cl;
            float v[R];
#pragma unroll
            for (int r = 0; r < R; ++r) v[r] = r < nr ? ds_bf(ob[r][kw][cl]) : 0.0f;
            float w[JD];
#pragma unroll
            for (int q = 0; q < JD; ++q) w[q] = wr[(long)(2 * q) * A.fp + i];
#pragma unroll
            for (int q = 0; q < JD; ++q)
#pragma unroll
                for (int r = 0; r < R; ++r) acc[q][r] += v[r] * w[q];
            if (sq)
#pragma unroll
                for (int r = 0; r < R; ++r) acc[JD][r] += v[r] * v[r];
        }
        __syncthreads();
    }
    const int chunk = kw * (n / kMixCols) + b;
#pragma unroll
    for (int q = 0; q <= JD; ++q) {
        if (q == JD && !sq) continue;
        const int j = q < JD ? par + 2 * q : kPmMix;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float s = r4d_wave_sum(acc[q][r]);
            if (!lane && r < nr) A.part[((g0 + r) * kPmS + j) * A.chunks + chunk] = s;
        }
    }
}

/* K2: the norm over the stored xc, a workgroup of 256 a row, in post_collapse_norm_kernel's order (tv = its THR) */
__global__ __launch_bounds__(256) void pcn_mix_k2_kernel(const uint16_t* xc, long xcp, const float* w, uint16_t* y,
                                                         long yp, int n, int tv, float eps, float wadd, float mult,
                                                         long M) { R4D_IKA_ENTRY();
    __shared__ float vss[1024];
    __shared__ float wsum[32];
    const long m = blockIdx.x;
    if (m >= M) return;
    const uint16_t* xr = xc + m * xcp;
    for (int v = (int)threadIdx.x; v < tv; v += 256) {
        float ss = 0.0f;
        for (int i = v; i < n; i += tv) {
            const float xv = bf16_to_f32(xr[i]);
            ss += xv * xv;
        }
        vss[v] = ss;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, nwv = tv / 32;
    for (int vw = wave; vw < nwv; vw += 8) {
        const float t = r4d_wave_sum(vss[vw * 32 + lane]);
        if (lane == 0) wsum[vw] = t;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int q = 0; q < nwv; ++q) sum += wsum[q];
    const float sc = rsqrtf(sum / (float)n + eps);
    uint16_t* yr = y + m * yp;
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * sc * (w[i] + wadd) * mult);
}

int64_t pcn_mix_scratch(long rows, long n) { return (int64_t)rows * mix_chunks(kPmHC * n) * kPmS * (int64_t)sizeof(float); }

/* Operands: mhc_post_collapse_rmsnorm's [0] x [1] h [2] post (inout) [3] comb (inout) [4] out [5] pre [6] xc [7] w [8] y,
 * then mhc_mix's [9] fn [10] scale [11] base [12] pre_o; post and comb are mhc_mix's post / comb outputs too. */
int pcn_mix_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 13) return RAD_E_INVAL;
    const long n = (long)r4d_geti_or(a, "n", 0), hc = (long)r4d_geti_or(a, "hc", 0);
    const long iters = (long)r4d_geti_or(a, "iterations", 0);
    if (hc != kPmHC || n < kMixCols || n % kMixCols || n > 16384 || iters < 1 || iters > 256)
        return R4D_NO(RAD_E_UNSUPPORTED);
    const RadTensor &xi = a->t[0], &h = a->t[1], &po = a->t[2], &cb = a->t[3], &o = a->t[4];
    const RadTensor &pre = a->t[5], &x = a->t[6], &w = a->t[7], &y = a->t[8];
    const RadTensor &fn = a->t[9], &sc = a->t[10], &bs = a->t[11], &pro = a->t[12];
    if (!w.data || !y.data || !fn.data || !sc.data || !bs.data || !pro.data) return R4D_NO(RAD_E_INVAL);
    if (xi.rank != 2 || xi.shape[0] < 1 || xi.shape[0] > 65535) return R4D_NO(RAD_E_SHAPE);
    const long rows = xi.shape[0], width = hc * n, mix = hc * (hc + 2);
    if (xi.dtype != RAD_BF16 || h.dtype != RAD_BF16 || po.dtype != RAD_F32 || cb.dtype != RAD_F32 ||
        o.dtype != RAD_BF16 || pre.dtype != RAD_F32 || x.dtype != RAD_BF16 || y.dtype != RAD_BF16 ||
        w.dtype != RAD_F32 || fn.dtype != RAD_F32 || sc.dtype != RAD_F32 || bs.dtype != RAD_F32 || pro.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (!mat(xi, RAD_BF16, rows, n) || !mat(h, RAD_BF16, rows, width) || !mat(po, RAD_F32, rows, hc) ||
        !mat(cb, RAD_F32, rows, hc * hc) || !mat(o, RAD_BF16, rows, width) || o.data == h.data ||
        !mat_s(pre, RAD_F32, rows, hc) || !mat(x, RAD_BF16, rows, n) || !mat(fn, RAD_F32, mix, width) ||
        sc.rank != 1 || sc.shape[0] != 3 || bs.rank != 1 || bs.shape[0] != mix || !mat_s(pro, RAD_F32, rows, hc))
        return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(&w) || r4d_numel(&w) < n) return R4D_NO(RAD_E_STRIDE);
    long long My = 0;
    const long long y_ld = r4d_rows_pitch(&y, n, &My);
    if (y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (My < rows) return R4D_NO(RAD_E_SHAPE);
    const int64_t need = pcn_mix_scratch(rows, n);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_INVAL);
    const float eps = (float)r4d_getf(a, "eps", 0), heps = (float)r4d_getf(a, "hc_eps", 0);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0), mult = (float)r4d_getf(a, "mult", 1.0);
    if (!std::isfinite(eps) || !std::isfinite(heps) || eps <= 0 || heps < 0) return R4D_NO(RAD_E_SHAPE);
    PmArgs A{};
    A.xin = (const uint16_t*)xi.data; A.xp = (long)xi.stride[0];
    A.h = (const uint16_t*)h.data; A.hp = (long)h.stride[0];
    A.post = (const float*)po.data; A.ptp = (long)po.stride[0];
    A.comb = (const float*)cb.data; A.cbp = (long)cb.stride[0];
    A.pre = (const float*)pre.data; A.pp = (long)pre.stride[0]; A.pst = (long)pre.stride[1];
    A.out = (uint16_t*)o.data; A.op = (long)o.stride[0];
    A.xc = (uint16_t*)x.data; A.xcp = (long)x.stride[0];
    A.fn = (const float*)fn.data; A.fp = (long)fn.stride[0];
    A.part = (float*)a->scratch;
    A.n = (int)n;
    A.chunks = (int)mix_chunks(width);
    A.M = rows;
    MixOut mo;
    mo.scale = (const float*)sc.data;
    mo.sst = (long)sc.stride[0];
    mo.base = (const float*)bs.data;
    mo.bst = (long)bs.stride[0];
    mo.pre = (float*)pro.data;
    mo.prp = (long)pro.stride[0];
    mo.prs = (long)pro.stride[1];
    mo.post = (float*)po.data;
    mo.pop = (long)po.stride[0];
    mo.pos = (long)po.stride[1];
    mo.comb = (float*)cb.data;
    mo.cp = (long)cb.stride[0];
    mo.cs = (long)cb.stride[1];
    mo.width = (int)width;
    mo.iters = (int)iters;
    mo.eps = eps;
    mo.heps = heps;
    const int mut = g_i9b_mut.load(std::memory_order_relaxed);
    int tv = r4d_act_threads((int)rows, (int)n);
    if (mut == 2) tv = tv < 1024 ? tv * 2 : tv / 2;   /* mutation control: another virtual-thread count (NOT exact) */
    const dim3 g1((unsigned)(n / kMixCols), (unsigned)((rows + kPmR - 1) / kPmR));
    int rc = mut == 1 ? R4D_EMIT(stream, pcn_mix_k1_kernel<true>, g1, dim3(256), 0, A)
                      : R4D_EMIT(stream, pcn_mix_k1_kernel<false>, g1, dim3(256), 0, A);
    if (rc < 0) return rc;
    rc = R4D_EMIT(stream, mix_finish_kernel<kPmHC>, dim3((unsigned)((rows + 3) / 4)), dim3(128), 0,
                  (const float*)A.part, A.chunks, mo, rows);
    if (rc < 0) return rc;
    return R4D_EMIT(stream, pcn_mix_k2_kernel, dim3((unsigned)rows), dim3(256), 0, (const uint16_t*)x.data,
                    (long)x.stride[0], (const float*)w.data, (uint16_t*)y.data, (long)y_ld, (int)n, tv, eps, wadd,
                    mult, rows);
}

}  // namespace

extern "C" int r4d_rad_token_map(const RadArgs* a, RadStream s) { return token_map_body(a, s); }
extern "C" int r4d_rad_bi_rowhash(const RadArgs* a, RadStream s) { return bi_rowhash_body(a, s); }
extern "C" int r4d_rad_stream_gate_add(const RadArgs* a, RadStream s) {
    return stream_gate_body(a, s);
}
extern "C" int r4d_rad_ds41_rope_rows(const RadArgs* a, RadStream s) { return rope_rows_body(a, s); }
extern "C" int r4d_rad_quant_roundtrip(const RadArgs* a, RadStream s) { return quant_rt_body(a, s); }
/* Lever WQ's bench pin (g_wq_pin: -1 the rule, 0 no dodge). */
extern "C" void r4d_wq_pin(int v) { g_wq_pin.store(v, std::memory_order_relaxed); }
/* Lever QG8's bench pin (g_qg8_pin: -1 the rule, 0 the one-column kernel). */
extern "C" void r4d_qg8_pin(int v) { g_qg8_pin.store(v, std::memory_order_relaxed); }
extern "C" int r4d_rad_mhc_mix(const RadArgs* a, RadStream s) { return mhc_mix_body(a, s); }
extern "C" void r4d_ds41_mix_fuse_force(int on) { g_mix_fuse_force.store(on, std::memory_order_relaxed); }
/* Lever 1U-MX (ds41-int11): the fn4 switch ds41_mix_wv (r4d_ds41_ggml.hip's rad_fn4_*) and its counters. */
bool r4d_mix_wv_known(int v) { return mix_wv_known(v); }
int r4d_mix_wv_get() { return mix_wv_now(); }
void r4d_mix_wv_set(int v) { g_mix_wv.store(mix_wv_known(v) ? v : 0, std::memory_order_relaxed); }
/* out[0] mhc_mix launches in a wide-wave form, [1] their rows; every rank's launches summed, cumulative. */
extern "C" int rad_mx_stats(long long* out, int n) {
    if (!out || n < 2) return 0;
    out[0] = g_mx_launch.load(std::memory_order_relaxed);
    out[1] = g_mx_rows.load(std::memory_order_relaxed);
    return 2;
}
/* Partials of every chunk of every row of the band, so the projection spreads over workgroups. */
extern "C" int64_t r4d_rad_mhc_mix_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), n = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M < 1 || n < 1 || hc < 2 || hc > 8 || hc * n > (1LL << 30)) return 0;
    return (int64_t)(M * mix_chunks((long)(hc * n)) * (hc * (hc + 2) + 1) * (long long)sizeof(float));
}
extern "C" int r4d_rad_mhc_collapse(const RadArgs* a, RadStream s) {
    return mhc_collapse_body(a, s);
}
extern "C" int r4d_rad_softmax_pool(const RadArgs* a, RadStream s) {
    return softmax_pool_body(a, s);
}
extern "C" int r4d_rad_entry_store(const RadArgs* a, RadStream s) { return entry_store_body(a, s); }
extern "C" int r4d_rad_window_store(const RadArgs* a, RadStream s) { return window_store_body(a, s); }
extern "C" int r4d_rad_router_scored_topk(const RadArgs* a, RadStream s) {
    return router_scored_body(a, s);
}
extern "C" int r4d_rad_router_scored_topk_scatter(const RadArgs* a, RadStream s) {
    return router_scored_scatter_body(a, s);
}
/* lever FUSE-R (ds41-fuse): router_gemm_f32_scored_topk_scatter and its instance's arrival counter */
extern "C" int r4d_rad_router_gemm_scored_scatter(const RadArgs* a, RadStream s) {
    return router_gemm_scored_scatter_body(a, s);
}
extern "C" int r4d_rad_router_gemm_scored_scatter_init(const RadParam*, int, int, int, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
#ifdef R4D_EMU_HOST
    /* tests/r4d_emu: device memory is host memory; a zeroed slot an instance */
    RgsInst* in = new RgsInst;
    in->ctr = (unsigned*)std::calloc(kRgsCtrStride, sizeof(unsigned));
    if (!in->ctr) { delete in; return RAD_E_NOMEM; }
    *out = in;
    return RAD_OK;
#else
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
    std::lock_guard<std::mutex> lk(g_rgs_mu);
    RgsBlock& blk = g_rgs_blocks[dev];
    if (!blk.base) {
        const size_t bytes = (size_t)kRgsCtrSlots * kRgsCtrStride * sizeof(unsigned);
        void* p = nullptr;
        if (hipMalloc(&p, bytes) != hipSuccess) return RAD_E_NOMEM;
        /* zeroed before any launch reads it (r4d_moe.hip's note: the engine's hipMemset has finished when it returns;
         * under HIP's own runtime, r4d_selftest, the null stream is waited for here) */
        if (hipMemset(p, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            (void)hipFree(p);
            return RAD_E_DEVICE;
        }
        blk.base = (unsigned*)p;
        blk.next = 0;
    }
    if (blk.next >= kRgsCtrSlots) return RAD_E_FULL;
    RgsInst* in = new RgsInst;
    in->dev = dev;
    in->ctr = blk.base + (size_t)blk.next * kRgsCtrStride;
    ++blk.next;
    ++blk.live;
    *out = in;
    return RAD_OK;
#endif
}
extern "C" void r4d_rad_router_gemm_scored_scatter_fini(void* inst) {
    RgsInst* in = (RgsInst*)inst;
    if (!in) return;
#ifdef R4D_EMU_HOST
    std::free(in->ctr);
#else
    std::lock_guard<std::mutex> lk(g_rgs_mu);
    auto it = g_rgs_blocks.find(in->dev);
    if (it != g_rgs_blocks.end() && it->second.live > 0 && --it->second.live == 0) {
        int prev = 0;
        const bool moved = hipGetDevice(&prev) == hipSuccess && prev != in->dev && hipSetDevice(in->dev) == hipSuccess;
        (void)hipFree(it->second.base);
        if (moved) (void)hipSetDevice(prev);
        g_rgs_blocks.erase(it);
    }
#endif
    delete in;
}
/* lever FUSE-M (ds41-fuse): the fn4 switch ds41_mix_fin (r4d_ds41_ggml.hip's rad_fn4_*) */
int r4d_mix_fin_get() { return mix_fin_now(); }
void r4d_mix_fin_set(int v) { g_mix_fin.store(v == 1 ? 1 : 0, std::memory_order_relaxed); }
/* Levers FUSE-M / FUSE-R's libr4d counters, every rank's launches summed, cumulative (a recorded pass counts when it is
 * recorded): out[0] mhc_mix launches in the FIN form under ds41_mix_fin, [1] their rows, [2] router_gemm_f32_scored_topk_
 * scatter launches, [3] their rows. Returns 4. */
extern "C" int rad_fuse_stats(long long* out, int n) {
    if (!out || n < 4) return 0;
    out[0] = g_mf_launch.load(std::memory_order_relaxed);
    out[1] = g_mf_rows.load(std::memory_order_relaxed);
    out[2] = g_rgs_launch.load(std::memory_order_relaxed);
    out[3] = g_rgs_rows.load(std::memory_order_relaxed);
    return 4;
}
extern "C" int r4d_rad_embed_lookup_planes(const RadArgs* a, RadStream s) {
    return embed_planes_body(a, s);
}
extern "C" int r4d_rad_rmsnorm_rope_roundtrip(const RadArgs* a, RadStream s) {
    return norm_rope_rt_body(a, s);
}
extern "C" int r4d_rad_rmsnorm_rope_roundtrip_kv_store(const RadArgs* a, RadStream s) {
    return norm_rope_rt_kv_body(a, s);
}
/* tp8 loop P32d: rope_rows_roundtrip's eight-columns-a-thread form (0 off, 1 on, -1 = RADIANCE_DS41_ROPE8) */
extern "C" void r4d_rope8_pin(int v) { g_rope8_pin.store(v < 0 ? -1 : (v ? 1 : 0), std::memory_order_relaxed); }
extern "C" int r4d_rad_rope_rows_roundtrip(const RadArgs* a, RadStream s) {
    return rope_rt_body(a, s);
}
extern "C" int r4d_rad_mhc_collapse_rmsnorm(const RadArgs* a, RadStream s) {
    return collapse_norm_body(a, s);
}
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm(const RadArgs* a, RadStream s) {
    return post_collapse_norm_body(a, s);
}
/* lever M1 (ds41-int9b): mhc_post_collapse_rmsnorm + mhc_mix over its out, one op (pcn_mix_body); its scratch is mhc_mix's
 * partials for M rows */
extern "C" int r4d_rad_mhc_post_collapse_rmsnorm_mix(const RadArgs* a, RadStream s) { return pcn_mix_body(a, s); }
extern "C" int64_t r4d_rad_mhc_post_collapse_rmsnorm_mix_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0), n = r4d_geti_or(a, "n", 0), hc = r4d_geti_or(a, "hc", 0);
    if (M < 1 || M > 65535 || hc != kPmHC || n < kMixCols || n % kMixCols || n > 16384) return 0;
    return pcn_mix_scratch((long)M, (long)n);
}
/* tests only (r4d_selftest's mutation controls, ds41-int9b): 0 the production forms; 1 lever M1's K1 with its slabs in
 * reverse order, 2 its K2 at another virtual-thread count (NOT exact: a check must catch both); the IX forms' hooks
 * are r4d_ds41_sparse.hip's (3 ..). */
extern "C" void r4d_ix_mutate(int which);   /* r4d_ds41_sparse.hip */
extern "C" void r4d_i9b_mutate(int which) {
    g_i9b_mut.store(which, std::memory_order_relaxed);
    r4d_ix_mutate(which);
}
/* tests only (tests/r4d_emu): mhc_post_collapse_rmsnorm as the all-reduce's word-partitioned epilogue, nb blocks */
extern "C" int r4d_ds41_pcn_words(const RadArgs* a, RadStream s, int nb, unsigned* ctr) {
    return pcn_words_body(a, s, nb, ctr);
}
