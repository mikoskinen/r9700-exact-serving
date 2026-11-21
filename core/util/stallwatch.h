/* stallwatch.h -- naming where a phase that has gone quiet is waiting.
 *
 * A hang (a driver call that never returns, a lock inside a runtime library) leaves no trace in
 * the log, often on machines that cannot be reproduced locally (a guest kernel, a passed-through
 * card), so the process reports where it waits.
 *
 * A StallWatch is held across a phase by the thread doing it. When nothing has been logged for the
 * quiet interval it reports what that thread is doing:
 *   - its scheduler state and kernel wait channel;
 *   - the system call it is in, an ioctl with its device and request, and the library that made
 *     the call;
 *   - when the thread waits in user space (a lock, a sleep), its stack as module+offset frames,
 *     which addr2line symbolises against the released binary. A thread inside an ioctl is not
 *     signalled: the call already names where it waits, and a signal could end a driver wait
 *     early;
 *   - the process's other threads that are inside an ioctl, since a lock in a runtime library is
 *     usually held by one of them.
 * It reports again each time the silence doubles, showing whether the thread moved. A phase whose
 * progress is not logged (a loop of steps) marks it with progress(). The watch reads only its own
 * process's /proc entries and wakes every quarter of the quiet interval, at most once a second,
 * while held.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>

namespace rad {

class StallWatch {
public:
    /* `phase` names what is being watched in the report ("startup"). */
    explicit StallWatch(const char* phase, uint64_t quiet_ms = 60000);
    ~StallWatch();
    StallWatch(const StallWatch&) = delete;
    StallWatch& operator=(const StallWatch&) = delete;

    /* Progress the log does not show: the silence restarts as if a line had been logged. */
    void progress();

    struct Impl;

private:
    std::unique_ptr<Impl> p_;
    std::atomic<uint64_t> beat_{0};   /* rad_mono_ns of the last progress() */
};

/* tp8 loop item 7 (RADIANCE_STUCK_EXIT_S): a watchdog that ENDS the process when a loop stops marking
 * progress. A StallWatch only reports; a step loop that waits forever on its card (a queue stuck on a
 * barrier nothing will release) keeps every other thread -- the HTTP server, a pipeline's link
 * heartbeat -- alive, so neither a client nor the other node nor a supervisor ever sees it down.
 *
 * Only beat() counts as progress (log lines from other threads do not). When `limit_ms` passes
 * without one, `on_fire(silent_s)` runs once on the watchdog's own thread (it should tell whoever
 * must go down with the process; it is given at most `fire_ms`), then the process ends with
 * _exit(`code`) -- no destructors: a wedged queue cannot be torn down. disarm() (or destruction)
 * stops the thread; a disarmed watch never fires. */
class StuckExit {
public:
    StuckExit(std::string what, uint64_t limit_ms, std::function<void(double)> on_fire, int code = 70,
              uint64_t fire_ms = 1500);
    ~StuckExit();
    StuckExit(const StuckExit&) = delete;
    StuckExit& operator=(const StuckExit&) = delete;

    void beat() { beat_.store(now_ns(), std::memory_order_relaxed); }
    /* The thread the exit line describes (the loop's own): the calling thread. */
    void hold();
    void disarm();
    pid_t holder() const { return holder_.load(std::memory_order_relaxed); }

    static uint64_t now_ns();

private:
    void run();
    std::string                 what_;
    uint64_t                    limit_ns_ = 0;
    std::function<void(double)> on_fire_;
    int                         code_ = 70;
    uint64_t                    fire_ms_ = 1500;
    std::atomic<uint64_t>       beat_{0};
    std::atomic<pid_t>          holder_{0};
    std::mutex                  mu_;
    std::condition_variable     cv_;
    bool                        stop_ = false;
    std::thread                 th_;
};

/* What thread `tid` of this process is doing, as the report words it: state, wait channel, system
 * call. With `stack`, a thread waiting in user space is also asked for its frames. */
std::string stall_describe_thread(pid_t tid, bool stack);

/* One line of /proc/<pid>/task/<tid>/syscall in words: "running", "not in a system call", or the
 * call with its arguments decoded where they name something (an ioctl's device and request). */
std::string stall_decode_syscall(const std::string& line);

/* `addr` as the module that maps it and its offset from the module's load base, "libfoo.so+0x1a2b":
 * the number addr2line takes for that module. */
std::string stall_module_offset(uintptr_t addr);

}  /* namespace rad */
