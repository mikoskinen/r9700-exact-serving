/* rank_barrier.cpp -- see rank_barrier.h for why this is a condvar and not an IPC handshake. */
#include "rank_barrier.h"
#include "fn4_switch.h"
#include "../rad_internal.h"

namespace rad {

void RankBarrier::arrive_and_wait() {
    std::unique_lock<std::mutex> lk(m_);
    if (aborted_) return;

    const uint64_t my_gen = gen_.load(std::memory_order_relaxed);
    if (++waiting_ == n_) {
        /* Last one in resets the counter and flips the generation, so a rank that re-enters at
         * once waits on the new generation instead of racing through. */
        waiting_ = 0;
        gen_.store(my_gen + 1, std::memory_order_release);
        lk.unlock();
        cv_.notify_all();
        return;
    }
    /* tp8 loop D61: spin on the generation first (host_spin, site 1); the flip -- by the last
     * arrival or abort(), both under m_ -- is seen here without the lock. */
    if (const int64_t spin = fn4::host_spin_ns(fn4::kSpinBarrier); spin > 0) {
        lk.unlock();
        const uint64_t t0 = rad_mono_ns();
        uint64_t t = t0;
        for (uint32_t k = 1;; ++k) {
            if (gen_.load(std::memory_order_acquire) != my_gen) {
                fn4::note_spin(fn4::kSpinBarrier, true, rad_mono_ns() - t0);
                return;
            }
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
            if ((k & 63) == 0 && (int64_t)((t = rad_mono_ns()) - t0) >= spin) break;
        }
        fn4::note_spin(fn4::kSpinBarrier, false, t - t0);
        lk.lock();
    }
    cv_.wait(lk, [&] { return aborted_ || gen_.load(std::memory_order_relaxed) != my_gen; });
}

void RankBarrier::abort() {
    {
        std::lock_guard<std::mutex> lk(m_);
        aborted_ = true;
        waiting_ = 0;
        gen_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_all();
}

}  /* namespace rad */
