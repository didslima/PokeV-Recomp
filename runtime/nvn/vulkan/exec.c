// Executes recorded NVN command lists with Vulkan (port of renderer.m; same structure, same behaviour).
#include <SDL3/SDL_cpuinfo.h>
#include <SDL3/SDL_timer.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <pthread.h>
#include <setjmp.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#ifdef __linux__
#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "../../kernel.h"
#include "shaders.h"
#include "vk.h"
#include "record.h"
#include "vk_fsr2.h"
#include "vk_dlaa.h"
#include "vk_xess.h"
#include "../renderer.h"
#include "../pace.h"
#include "../recolour.h"
#include "../screen_filter.h"
#include "../shader_prep.h"
#include "../../settings.h"

// Counters the draw path bumps (stats, VK_PROF), one set per translating thread: the shadow workers' parts bumped the
// same globals at once, and every bump moved a cache line - with the read-mostly globals next to it - between cores
// (each part ran ~1.6x slower than alone). A worker's set is added into the render thread's after each run (ctr_fold).
typedef struct {
    uint64_t stat_vtx_cached_, stat_vtx_host_, stat_draws_, stat_skipped_, cum_draws_, prof_cb_reused_, prof_cb_copied_, prof_prog_memo_, prof_pipe_memo_, prof_key_draws_, prof_sets_reused_, prof_sets_new_, prof_hook_draws_, prof_hook_known_, prof_scan_calls_, prof_scan_bytes_, prof_scan_skipped_, prof_vc_hits_, prof_vc_bypass_, prof_vc_copied_, prof_vc_queries_, prof_vc_scan_pages_, prof_vc_scan_ns_;
    uint64_t op_counts_[CMD_COUNT];
} __attribute__((aligned(64))) Ctrs;
static Ctrs ctr_main, ctr_w[8];
static __thread Ctrs *ctr_tls;  // (a shadow worker's; NULL: ctr_main)
static inline Ctrs *ctr(void) { Ctrs *c = ctr_tls; return c ? c : &ctr_main; }
static void ctr_fold(int w) {
    uint64_t *a = (uint64_t *)&ctr_main, *b = (uint64_t *)&ctr_w[w];
    for (size_t i = 0; i < sizeof(Ctrs) / 8; i++) a[i] += b[i], b[i] = 0;
}
#define stat_vtx_cached (ctr()->stat_vtx_cached_)
#define stat_vtx_host (ctr()->stat_vtx_host_)
#define stat_draws (ctr()->stat_draws_)
#define stat_skipped (ctr()->stat_skipped_)
#define cum_draws (ctr()->cum_draws_)
#define prof_cb_reused (ctr()->prof_cb_reused_)
#define prof_cb_copied (ctr()->prof_cb_copied_)
#define prof_prog_memo (ctr()->prof_prog_memo_)
#define prof_pipe_memo (ctr()->prof_pipe_memo_)
#define prof_key_draws (ctr()->prof_key_draws_)
#define prof_sets_reused (ctr()->prof_sets_reused_)
#define prof_sets_new (ctr()->prof_sets_new_)
#define prof_hook_draws (ctr()->prof_hook_draws_)
#define prof_hook_known (ctr()->prof_hook_known_)
#define prof_scan_calls (ctr()->prof_scan_calls_)
#define prof_scan_bytes (ctr()->prof_scan_bytes_)
#define prof_scan_skipped (ctr()->prof_scan_skipped_)
#define prof_vc_hits (ctr()->prof_vc_hits_)
#define prof_vc_bypass (ctr()->prof_vc_bypass_)
#define prof_vc_copied (ctr()->prof_vc_copied_)
#define prof_vc_queries (ctr()->prof_vc_queries_)
#define prof_vc_scan_pages (ctr()->prof_vc_scan_pages_)
#define prof_vc_scan_ns (ctr()->prof_vc_scan_ns_)
#define op_counts (ctr()->op_counts_)
// LARECOMP_SKIP_WHY=1: skipped draws / dispatches per source line, with the stats
static uint32_t skip_why[4096];
#define stat_skipped_here() (stat_skipped++, skip_why[__LINE__ & 4095]++)
static uint64_t stat_xfb_sessions;  // vertex stream bytes bound from the vertex cache / from host memory
static uint64_t stat_scene_wait_ns;  // (wait_scene)
static uint64_t stat_passes, stat_dispatches, stat_wait_ns, stat_flushes, stat_feedback_copies;
// (the guest heap for shaders in chunks: with the heap import below)
static uint32_t gc_shift;
static uint32_t gmem_chunk_shift(void);
static VkDescriptorSetLayout gmem_chunk_layout(void);
static void gc_scan_pointers(const uint32_t *w, uint64_t bytes);
static bool gc_lazy;
static int gc_imported;
static uint64_t gc_imported_bytes, stat_gc_misses;
static uint64_t gmem_heap_size(void);
static void gc_frame(void);
static uint64_t stat_upload_bytes;
static _Atomic uint32_t gmem_seq;  // (gmem_info_quick)
static bool cbuf_lean(void);
static bool key_memo_on(void);
static bool set_reuse_on(void);
static uint64_t prof_slot_wait;  // (VK_PROF) sub(): ring slot waits (ns; GPU waits, inside "pre" when begin_pass starts a submission)
// LARECOMP_VK_PROF=1: render-thread time per category in the fps log (exec = whole command lists, draw = draw(),
// sets = descriptor sets incl. cbuf copies, vtx = index scan + vertex copies, submit = vkQueueSubmit2)
static int prof = -1;
static uint64_t prof_exec, prof_draw, prof_sets, prof_vtx, prof_submit, prof_pre, prof_key, prof_dyn, prof_tex, prof_up, prof_gpu;
static VkQueryPool prof_qp;  // the GPU timer's timestamps (3 per ring slot: start, end, window image acquired)
static uint64_t prof_win, prof_span, prof_wframes;  // (profiling) the window image's submissions: GPU ns, frame spans, count
static uint64_t cum_exec, cum_wait, cum_submits;  // (profiling) never reset: hitch log
static uint64_t submit_draws;  // cum_draws at the last submission (backend_execute_batch)
static bool list_fenced;       // the list being run has a FenceSync: its submission ends with it (backend_execute_batch)
static uint32_t prof_end_line[4096];  // passes ended, by the exec.c line that ended them (0: submit / present)
uint64_t prof_views_created;  // textures.c vk_image_view cache misses
uint64_t prof_up_ns = ~0ull;  // textures.c upload checks (~0: profiling off)
#define PROF_T0() uint64_t prof_t0_ = prof ? SDL_GetTicksNS() : 0
#define PROF_ADD(v) do { if (prof) v += SDL_GetTicksNS() - prof_t0_; } while (0)
static _Atomic int stat_compiles;
bool g_debug_frame;
static char stats_text[2048];
void vk_collect_dead(void);

// ================================================================== submissions
// A submission = optional upload command buffer + the main one, signalling the timeline with its number.
// Every call on a command buffer, the descriptor pools and the queue goes through record.c (the recorder thread, or
// at once with it off): this file decides, the recorder calls. submit_seq counts what this thread handed to the
// recorder (what it will have submitted, in this order); rec_submitted() is what the driver has.
#define SUBS VK_SUBS
typedef struct {
    VkCommandBuffer main, upload;       // (allocated once, vk_exec_init: the pool is the recorder's from then on)
    bool main_begun, upload_begun, ts;  // ts: GPU timestamps written (the GPU timer)
    bool ts_skip;                       // (waits on the window image: not counted, see vk_gpu_total_ns)
    bool ts_win;                        // the window image's submission: a 3rd timestamp after its acquire wait
    uint64_t seq;
    // (its descriptor pools: record.c, per slot)
} Sub;
static Sub subs[SUBS];
static int cur_sub;
static void ss_forget(void);
static uint64_t submit_seq;  // last handed to the recorder (= submitted, with the recorder off)
static pthread_mutex_t seq_lock = PTHREAD_MUTEX_INITIALIZER;
static bool in_pass;
static int passes_in_sub;  // render passes in the submission being recorded (LARECOMP_CHUNK_PASSES, begin_pass)

// CPU / GPU overlap (as renderer.m: the Metal backend, metal-overlap). The game's frame loop submits a frame, presents
// it, and at once waits on that frame's fence (every SyncWait is on the newest submission), so it builds frame N+1
// only once the GPU has finished frame N: a frame costs CPU + GPU time in series. Here the wait is taken one fence
// late: SyncWait(fence N) waits for fence N-1's submission and returns, and the next SyncWait waits for N. The CPU
// builds frame N+1 while the GPU draws N; at most two frames of GPU work are queued; a fence already complete is never
// waited for. Everything this backend recycles is keyed on the timeline (sub(), vk_transient, the set pools after the
// slot's wait, readbacks and screenshots wait on their own submission), not on the game's fence, so only the game
// thread's wait changes. What the game may touch in that frame while the GPU still reads it: its per-frame vertex /
// index data in the guest heap (bound in place; an engine built to overlap double-buffers it anyway), uniform data is
// copied at translation (guest_copy), GPU results the CPU reads (texture / buffer copies) are flushed synchronously,
// and the storage buffers it rewrites one frame ahead get copies (snap_storage). The timing counters are read a frame
// before their GPU time is known: in this mode an end counter reports the latest completed frame's GPU span
// (cq_resolve keeps it host side and no longer writes the game's memory late - the game may have reused the slot).
// LARECOMP_GPU_OVERLAP=0: off (the wait on the fence's own submission, as before); =1: on.
#define VK_OVERLAP_DEFAULT 1  // on where the write watch can run (overlap_on: off by itself without page write tracking)
static uint64_t deferred_seq;   // (seq_lock) the fence the game was told is done but whose submission wasn't waited for
static uint64_t stat_deferred;  // SyncWaits returned without waiting (stats)
static uint64_t stat_snap_bytes, stat_snap_copies;  // storage copies made (snap_storage; stats, per period)
// The overlap needs the storage write watch (snap_storage: the OS's page write tracking), the only thing that catches
// the game rewriting its light lists with the same bytes while the GPU reads them. Where the OS gives no tracking
// (Linux over a userptr or dma-buf heap - AMD, the Steam Deck -, a kernel before 6.7 without PAGEMAP_SCAN, a
// container without userfaultfd, a Windows window reserved without write watch) the overlap turns itself off: the
// full fence wait as before. LARECOMP_GPU_OVERLAP=1 forces it on anyway (tests; the hash probe stands in for the
// watch then), =0 turns it off.
static int pages_cpu_written(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now);  // (the page tables below)
static const char *heap_tracking_reason(void);  // (why the heap has no page write tracking, for the log)
static bool overlap_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LARECOMP_GPU_OVERLAP"), *w = getenv("LARECOMP_OVERLAP_WATCH");
        uint32_t g;
        bool tracked = (!w || atoi(w) != 0) && pages_cpu_written(HEAP_BASE, 4096, 0, &g) >= 0;
        int cs = g_settings.gpu_compat[GPU_OVERLAP];  // (F1 > Compatibility)
        if (e) on = atoi(e) != 0;
        else if (cs != GPU_AUTO) on = cs == GPU_ON;
        else if (VK_OVERLAP_DEFAULT && g_vk.props.vendorID == 0x1002) {
            // AMD: off by default - new in v0.1.10 with the dynamic vertex input (device.c), and AMD GPUs lose the device
            // there at the first frames in the world, "read invalid 0xbeefdeadc000" (#39, #54)
            on = 0;
            LOG("[vk] GPU overlap off on this AMD GPU (#39; F1 > Compatibility or LARECOMP_GPU_OVERLAP=1: on)\n");
            return on;
        } else if (VK_OVERLAP_DEFAULT && !tracked) {
            on = 0;
            LOG("[vk] GPU overlap off: no page write tracking on this system (%s); LARECOMP_GPU_OVERLAP=1 forces it on\n",
                w && atoi(w) == 0 ? "LARECOMP_OVERLAP_WATCH=0" : heap_tracking_reason());
            return on;
        } else on = VK_OVERLAP_DEFAULT;
        LOG("[vk] GPU overlap %s (the game's frame fence waited one frame late; LARECOMP_GPU_OVERLAP=%d: %s%s)\n", on ? "on" : "off",
            !on, on ? "off" : "on", on && !tracked ? "; forced without page write tracking: the hash probe instead of the write watch" : "");
    }
    return on;
}
bool vk_overlap_state(void) { return overlap_on(); }  // (device.c vk_device_fault_report)
static bool xm_on(void);
bool vk_xfb_mirror_state(void) { return xm_on(); }  // (device.c vk_device_fault_report)
static _Atomic uint64_t g_last_gpu_frame_ticks;  // (overlap) the latest resolved counter pair's GPU span, in Switch GPU ticks
static const uint64_t *cq_last_out;              // (cq_lock) the counter resolved last and its GPU ticks: a pair may straddle
static uint64_t cq_last_ticks;                   // two submissions
static uint64_t snap_ring_end(void);             // (snap_storage: the copies' ring after the game's heap, for gmem_heap_size)

uint64_t vk_completed_seq(void) {
    uint64_t v = 0;
    VK_CHECK(vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &v));
    return v;
}

// The indirect draws whose parameters the GPU reads in place (draw_indirect), the last IR_N of them - always on, a
// few KB: a GPU that never finishes a submission (the 10 s stop, #70 #81 #82 #86) or loses the device shows the ones
// still in flight with their parameters as translated and as they are now, which is what the GPU reads.
#define IR_N 256  // (a power of two)
static struct { uint64_t seq, pa, vs; uint32_t rec[5]; } ir_ring[IR_N];
static unsigned ir_n;
static void ir_note(uint64_t pa, const uint32_t *p, uint64_t vs) {
    unsigned i = ir_n++ & (IR_N - 1);
    ir_ring[i].seq = submit_seq + 1; ir_ring[i].pa = pa; ir_ring[i].vs = vs;
    memcpy(ir_ring[i].rec, p, sizeof ir_ring[i].rec);
}
static void ir_dump(void) {
    uint64_t done = 0;
    // (not VK_CHECK: the device may be lost - and then the value is UINT64_MAX or garbage: every entry is shown)
    if (vkGetSemaphoreCounterValue(g_vk.dev, g_vk.timeline, &done) != VK_SUCCESS || done > submit_seq) done = 0;
    unsigned lo = ir_n > IR_N ? ir_n - IR_N : 0, n = 0;
    for (unsigned k = lo; k < ir_n; k++) n += ir_ring[k & (IR_N - 1)].seq > done;
    LOG("[vk] indirect draws in flight (GPU done up to submission %llu): %u of the last %u\n", (unsigned long long)done, n, ir_n - lo);
    unsigned skip = n > 48 ? n - 48 : 0;  // (the last 48 of them)
    for (unsigned k = lo; k < ir_n; k++) {
        const typeof(ir_ring[0]) *e = &ir_ring[k & (IR_N - 1)];
        if (e->seq <= done) continue;
        if (skip) { skip--; continue; }
        const uint32_t *q = (const uint32_t *)(uintptr_t)e->pa, *w = e->rec;
        LOG("[vk]   submission %llu vs %llx args %llx: recorded %u %u %u %d %u, now %u %u %u %d %u\n", (unsigned long long)e->seq,
            (unsigned long long)e->vs, (unsigned long long)e->pa, w[0], w[1], w[2], (int)w[3], w[4], q[0], q[1], q[2], (int)q[3], q[4]);
    }
}

// (LARECOMP_VK_PROF) waits on the GPU by cause: the caller names it in wait_why just before waiting
static __thread const char *wait_why;
static struct { const char *why; uint64_t ns, n; } wait_tab[12];
static pthread_mutex_t wait_tab_lock = PTHREAD_MUTEX_INITIALIZER;
static void wait_account(uint64_t ns) {
    const char *why = wait_why ? wait_why : "other";
    wait_why = NULL;
    pthread_mutex_lock(&wait_tab_lock);
    for (int i = 0; i < 12; i++)
        if (!wait_tab[i].why || wait_tab[i].why == why) { wait_tab[i].why = why; wait_tab[i].ns += ns; wait_tab[i].n++; break; }
    pthread_mutex_unlock(&wait_tab_lock);
}

static void cq_resolve_done(uint64_t done);

void vk_wait_seq(uint64_t seq) {
    if (!seq) { wait_why = NULL; return; }
    if (seq > rec_submitted()) {  // not handed to the driver yet: the recorder first (a timeline value can't be waited for sooner)
        pthread_mutex_lock(&seq_lock);
        uint64_t issued = submit_seq;
        pthread_mutex_unlock(&seq_lock);
        if (seq > issued) {  // (not even queued: a caller's mistake - everything queued so far instead, logged)
            static int n;
            if (n++ < 5) LOG("[vk] wait for submission %llu, only %llu queued\n", (unsigned long long)seq, (unsigned long long)issued);
            seq = issued;
            if (!seq) { wait_why = NULL; return; }
        }
        rec_wait_submitted(seq);
    }
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_vk.timeline,
                              .pValues = &seq};
    uint64_t t0 = SDL_GetTicksNS();
    extern void rq_watch(const char *what, uint64_t arg);
    rq_watch("vkWaitSemaphores (GPU submission)", seq);
    VkResult r = vkWaitSemaphores(g_vk.dev, &wi, 10ull * 1000000000ull);
    rq_watch(NULL, 0);
    uint64_t waited = SDL_GetTicksNS() - t0;  // (the game thread's fence waits and the render thread's: atomic sums)
    __atomic_fetch_add(&stat_wait_ns, waited, __ATOMIC_RELAXED);
    __atomic_fetch_add(&cum_wait, waited, __ATOMIC_RELAXED);
    wait_account(waited);
    // Callers recycle command buffers and allocations after this returns; a timeout is not completion.
    if (r == VK_TIMEOUT) {
        ir_dump();
        FATAL("[vk] GPU did not finish submission %llu within 10 s; stopping before reusing in-flight resources\n",
              (unsigned long long)seq);
    } else if (r == VK_SUCCESS) cq_resolve_done(seq);  // (the game reads its timing counters after waiting)
    else {
        extern void vk_devlost_dump(void);
        ir_dump();
        vk_devlost_dump();
        vk_device_fault_report();
        FATAL("vkWaitSemaphores: %d (device lost)\n", r);
    }
}

uint64_t vk_pending_seq(void) { return submit_seq + 1; }

// ---- the game's GPU timing counters (nvnCommandBufferReportCounter, two per frame)
// The game times its GPU work with these and drops to 30 fps when frames look slow. They used to get the host clock
// at translation time, so our command translation counted as GPU time: busy scenes (outdoors, ~400 draws) looked slow
// and the game locked itself at 30 even on fast GPUs, at any resolution. Now each counter also records a GPU
// timestamp at its place in the command stream; once that submission is done (the game waits on its fence before
// reading - on the Switch the GPU writes them too) the GPU time replaces the provisional host value, in the host
// clock's terms (offset calibrated at each resolve: a timestamp can't be later than the moment its submission is seen
// done). LARECOMP_VK_COUNTER=cpu: host clock only; =frozen: one value per presented frame (test); =legacy: the host
// clock in ns instead of GPU ticks, as before (test: reproduces the locked 30).
#define CQ_MAX 64
typedef struct { VkQueryPool pool; uint32_t n; uint64_t seq; uint64_t *out[CQ_MAX]; } CounterQ;
static CounterQ cq[SUBS];
static pthread_mutex_t cq_lock = PTHREAD_MUTEX_INITIALIZER;
static int64_t cq_offset;  // host ns - GPU ns (the smallest seen)
// Counter timestamps are in the Switch GPU's ticks (614.4 MHz): the game converts the difference to ns itself
// (x 625 / 384). Writing ns made every GPU time look 1.63x as long.
static uint64_t ns_to_gpu_ticks(uint64_t ns) { return (uint64_t)((unsigned __int128)ns * 384 / 625); }
static bool cq_calibrated;
static int counter_mode = -1;  // 0 cpu, 1 frozen, 2 gpu
static uint64_t stat_counters_gpu;

// Pokémon Z-A sizes its dynamic resolution from these counters - and the GPU time they show includes our translation's
// passes, so on any PC it sat at its floor (55-65% of 1080p: pixelated, aliased). Unless the setting dynamic_res =
// game, Z-A's counters show no GPU time (as "frozen"): it renders at full resolution, its frame mode being its own
// (rt_za_frame_cfg). F1 > Dynamic resolution; applies at once.
static int vk_counter_mode(void) {
    static bool env;
    if (counter_mode < 0) {
        const char *e = ENV("LARECOMP_VK_COUNTER");
        env = e != NULL;
        counter_mode = e && !strcmp(e, "cpu") ? 0 : e && !strcmp(e, "frozen") ? 1 : e && !strcmp(e, "legacy") ? 3 : 2;
        if (counter_mode == 2 && !g_vk.props.limits.timestampComputeAndGraphics) counter_mode = 0;
    }
    extern bool g_za_frame_hook;
    if (!env && g_za_frame_hook && !g_settings.game_drs) return 1;
    return counter_mode;
}

// ---- frame pacing (runtime/nvn/pace.c): the game's 30 fps mode when frames don't fit 60, through these counters
static void cq_resolve(int i) {  // (cq_lock held) submission slot i is done: its counters get their GPU times
    CounterQ *q = &cq[i];
    if (!q->n) return;
    uint64_t t[CQ_MAX];
    if (vkGetQueryPoolResults(g_vk.dev, q->pool, 0, q->n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod;
        int64_t cand = (int64_t)plat_now_ns() - (int64_t)((double)t[q->n - 1] * per);
        if (!cq_calibrated || cand < cq_offset) cq_offset = cand, cq_calibrated = true;
        static int log = -1;  // LARECOMP_LOG_COUNTER=1: the host (provisional) and GPU times of each counter
        if (log < 0) log = ENV("LARECOMP_LOG_COUNTER") != NULL;
        for (uint32_t k = 0; k < q->n; k++) {
            uint64_t gpu = ns_to_gpu_ticks((uint64_t)((int64_t)((double)t[k] * per) + cq_offset));
            if (log) LOG("[counter] seq %llu #%u at %llx host %.3f gpu %.3f ms\n", (unsigned long long)q->seq, k,
                         (unsigned long long)(uintptr_t)q->out[k], q->out[k][1] * 625.0 / 384 / 1e6, gpu * 625.0 / 384 / 1e6);
            if (overlap_on()) {
                // kept host side: the pair's span becomes the next end counter's reading (the game has read this pair
                // already - its fence wait returned a frame early - and may have reused the slot for the next frame)
                if (cq_last_out && q->out[k] - cq_last_out == 2 && gpu >= cq_last_ticks && gpu - cq_last_ticks < 614400000ull)
                    atomic_store(&g_last_gpu_frame_ticks, gpu - cq_last_ticks);  // (an end counter 16 bytes after its start; under a second)
                cq_last_out = q->out[k], cq_last_ticks = gpu;
            } else {
                q->out[k][1] = pace_end_counter(k ? q->out[k - 1] : NULL, q->out[k], gpu);
            }
        }
        stat_counters_gpu += q->n;
    }
    q->n = 0;
}

static void cq_resolve_done(uint64_t done) {  // every slot whose submission is done, in submission order
    pthread_mutex_lock(&cq_lock);
    for (;;) {
        int best = -1;
        for (int i = 0; i < SUBS; i++)
            if (cq[i].n && cq[i].seq <= done && (best < 0 || cq[i].seq < cq[best].seq)) best = i;
        if (best < 0) break;
        cq_resolve(best);
    }
    pthread_mutex_unlock(&cq_lock);
}

// ---- GPU timer (dynamic resolution, runtime/nvn/output_scale.c; and LARECOMP_VK_PROF's "gpu"): a timestamp at the
// start and the end of each submission's main command buffer, read when its ring slot comes round again (so a few
// submissions late; the controller averages half a second). Submissions overlap a little (the next one's commands
// may start before the previous one's end): their union is counted, the time the GPU was busy.
// The window image's submission (device.c: the game's image drawn into the window - FSR / scaling - and the menu) waits
// for the image, which can be a vsync's wait, so its start timestamp isn't its work's: it has a 3rd timestamp, written
// after the barrier that waits for the image (vk_gpu_timer_window_start), and its time is counted from there - in a
// total of its own (vk_gpu_split_ns): device.c adds it to what dynamic resolution sees unless
// LARECOMP_DYNRES_COUNT_PRESENT=0. Also per frame: the span from the GPU starting the frame's first game submission
// to the end of its window submission; span - game - window = the GPU idle / doing work no timestamp covers
// (uploads ahead of a submission's main command buffer, waiting for the window image).
static uint64_t gpu_total_ns, gpu_last_end;  // ns counted; (ticks) the latest submission end seen
static uint64_t gpu_win_ns, gpu_span_ns, gpu_frames, span_from;  // window submissions: ns, frame spans' ns, count; (ticks)
static bool gpu_timer_on;

static void gpu_timer_add(const uint64_t *t, bool skip, bool win) {
    double per = g_vk.props.limits.timestampPeriod;
    uint64_t t0 = t[0], t1 = t[1], tw = win ? t[2] : 0;
    if (g_vk.ts_bits < 64) { uint64_t m = (1ull << g_vk.ts_bits) - 1; t0 &= m, t1 &= m, tw &= m; }
    if (t1 < t0 || (t1 - t0) * per > 1e9) { span_from = 0; return; }  // (wrapped, or nonsense)
    if (gpu_last_end > t0 && (gpu_last_end - t0) * per > 1e9) gpu_last_end = 0;  // (the counter wrapped)
    if (span_from > t1 || (span_from && (t1 - span_from) * per > 1e9)) span_from = 0;
    uint64_t prev_end = gpu_last_end;
    uint64_t from = t0 > gpu_last_end ? t0 : gpu_last_end;
    if (t1 > gpu_last_end) gpu_last_end = t1;
    if (win) {  // from after the image was acquired (and the work before it ended), to the end
        if (tw >= t0 && tw <= t1) {
            uint64_t wf = tw > prev_end ? tw : prev_end;
            if (t1 > wf) {
                uint64_t ns = (uint64_t)((double)(t1 - wf) * per);
                __atomic_add_fetch(&gpu_win_ns, ns, __ATOMIC_RELAXED);
                if (prof) prof_win += ns;
            }
        }
        if (span_from && t1 > span_from) {
            uint64_t ns = (uint64_t)((double)(t1 - span_from) * per);
            __atomic_add_fetch(&gpu_span_ns, ns, __ATOMIC_RELAXED);
            __atomic_add_fetch(&gpu_frames, 1, __ATOMIC_RELAXED);
            if (prof) prof_span += ns, prof_wframes++;
        }
        span_from = 0;
        return;
    }
    if (!span_from && !skip) span_from = from;  // the frame's first game submission (from where the GPU got to it)
    if (skip || t1 <= from) return;
    uint64_t ns = (uint64_t)((double)(t1 - from) * per);
    __atomic_add_fetch(&gpu_total_ns, ns, __ATOMIC_RELAXED);
    if (prof) prof_gpu += ns;
}

// GPU time of all counted submissions so far (ns); -1: no GPU timer on this device
int64_t vk_gpu_total_ns(void) { return gpu_timer_on ? (int64_t)__atomic_load_n(&gpu_total_ns, __ATOMIC_RELAXED) : -1; }
// (cumulative) out: game submissions' GPU ns, window submissions' GPU ns, frame spans' ns, frames with a span.
// false: no GPU timer.
bool vk_gpu_split_ns(uint64_t out[4]) {
    if (!gpu_timer_on) return false;
    out[0] = __atomic_load_n(&gpu_total_ns, __ATOMIC_RELAXED);
    out[1] = __atomic_load_n(&gpu_win_ns, __ATOMIC_RELAXED);
    out[2] = __atomic_load_n(&gpu_span_ns, __ATOMIC_RELAXED);
    out[3] = __atomic_load_n(&gpu_frames, __ATOMIC_RELAXED);
    return true;
}
// Frame pacing (pace.c): GPU time so far, the game's submissions and the window image's; -1: no GPU timer.
int64_t pace_gpu_ns(void) {
    uint64_t t[4];
    return vk_gpu_split_ns(t) ? (int64_t)(t[0] + t[1]) : -1;
}
uint64_t vk_cum_wait_ns(void) { return __atomic_load_n(&cum_wait, __ATOMIC_RELAXED); }  // (all vk_wait_seq waits so far)
uint64_t vk_cum_exec_ns(void) { return __atomic_load_n(&cum_exec, __ATOMIC_RELAXED); }  // (all command list translation time so far)

// Shadow workers (sw_*, below): on one, the command buffer is its secondary and the pass is the render thread's
static __thread VkCommandBuffer sw_cb;
static __thread int sw_id = -1;  // (the worker; -1: none)
// Per-thread copies of the draw path's caches and per-draw state: index 0 the render thread's, 1 + sw_id a shadow
// worker's part. The shared state a draw changes (programs and pipelines made, textures' views and uploads, the vertex
// cache, storage snapshots, aliases...) is changed under sw_mx (SWL / SWU: only on a worker; recursive).
#define NCTX 9
static inline int sw_ctx(void) { return sw_cb ? sw_id + 1 : 0; }
// (a spin lock: held for microseconds, taken several times a draw by every part - a mutex that sleeps made each
// contended hand-over cost more than the work)
static _Atomic int sw_mx;
static __thread int sw_mx_depth;
static bool sw_rt_lock;   // (the render thread) async runs in flight: it takes the workers' lock too (sw_async_*)
static bool sw_ainflight;  // (the render thread) jobs of async runs queued and not waited for (sw_async_sync)
static void sw_async_sync(void);
static inline void SWL(void) {
    if (!sw_cb && !sw_rt_lock) return;
    if (sw_mx_depth++) return;  // (held already: recursive)
    if (!atomic_exchange_explicit(&sw_mx, 1, memory_order_acquire)) return;
    for (int k = 0;; k++) {
        if (!atomic_load_explicit(&sw_mx, memory_order_relaxed) && !atomic_exchange_explicit(&sw_mx, 1, memory_order_acquire)) return;
        __builtin_ia32_pause();
        if ((k & 1023) == 1023) SDL_Delay(0);
    }
}
#define SWL_AT(line) SWL()  // (the line: where it is taken, for a count per place when measuring contention)
static inline void SWU(void) {
    if (sw_mx_depth <= 0) return;  // (not taken by this thread)
    if (--sw_mx_depth) return;
    atomic_store_explicit(&sw_mx, 0, memory_order_release);
}
static inline void SWU_ALL(void) { if (sw_mx_depth > 0) { sw_mx_depth = 0; atomic_store_explicit(&sw_mx, 0, memory_order_release); } }
static __thread jmp_buf *sw_jmp;  // (a worker, inside a draw) where a pass end unwinds to: the draw is the render thread's
static bool sw_reexec;  // (render thread) taking over a run from a worker's aborted draw: counters were walked already  // (the worker; -1: none)
static __thread bool sw_walking;  // (render thread) the state-only walk over a run of lists (sw_run): draws skipped (per thread: async runs' workers draw while it walks)
static void sw_slot_reset(int slot);

static Sub *sub(void) {
    Sub *s = &subs[cur_sub];
    if (s->seq != submit_seq + 1) {  // slot reused: its previous submission must be done
        wait_why = "ring slot";
        uint64_t tw = prof ? SDL_GetTicksNS() : 0;
        vk_wait_seq(s->seq);
        if (prof) prof_slot_wait += SDL_GetTicksNS() - tw;
        cq_resolve_done(s->seq);
        if (s->ts) {  // GPU time of that submission
            uint64_t t[3];
            uint32_t nq = s->ts_win ? 3 : 2;
            if (vkGetQueryPoolResults(g_vk.dev, prof_qp, (uint32_t)cur_sub * 3, nq, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                gpu_timer_add(t, s->ts_skip, s->ts_win);
            s->ts = s->ts_skip = s->ts_win = false;
        }
        rec_reset_pools(cur_sub);  // (its sets went with the submission; the recorder resets in order)
        sw_slot_reset(cur_sub);    // (and the shadow workers' secondaries)
        pthread_mutex_lock(&seq_lock);
        s->main_begun = s->upload_begun = false;
        s->seq = submit_seq + 1;
        pthread_mutex_unlock(&seq_lock);
    }
    return s;
}

static void begin_cb(VkCommandBuffer *cb, bool *begun) {
    if (*begun) return;
    rec_BeginCommandBuffer(*cb);
    pthread_mutex_lock(&seq_lock);
    *begun = true;
    pthread_mutex_unlock(&seq_lock);
}

static void gp_mark(VkCommandBuffer cb, const char *fmt, ...);
VkCommandBuffer vk_cmd(void) {
    if (sw_cb) return sw_cb;
    Sub *s = sub();
    if (!s->main_begun) {
        begin_cb(&s->main, &s->main_begun);
        ss_forget();  // (a new command buffer starts with no state)
        if (vk_counter_mode() == 2) {  // this slot's timing counter queries (reset outside any render pass)
            CounterQ *q = &cq[cur_sub];
            if (!q->pool) {
                VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                            .queryCount = CQ_MAX};
                if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &q->pool) != VK_SUCCESS) q->pool = VK_NULL_HANDLE;
            }
            pthread_mutex_lock(&cq_lock);
            q->n = 0;  // (resolved when the slot was freed: sub())
            q->seq = s->seq;
            pthread_mutex_unlock(&cq_lock);
            if (q->pool) rec_CmdResetQueryPool(s->main, q->pool, 0, CQ_MAX);
        }
        gp_mark(s->main, "(submission start; includes GPU idle before it)");
        if (prof_qp) {
            rec_CmdResetQueryPool(s->main, prof_qp, (uint32_t)cur_sub * 3, 3);
            rec_CmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_NONE, prof_qp, (uint32_t)cur_sub * 3);
            s->ts = true;
        }
    }
    return s->main;
}

// (device.c vk_frame_begin) the window image's submission: its work starts here, after the barrier that waits for the
// image (ALL_COMMANDS: after it, so not while the GPU waits for the image)
void vk_gpu_timer_window_start(void) {
    Sub *s = sub();
    if (!s->ts || s->ts_win) return;
    rec_CmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_qp, (uint32_t)cur_sub * 3 + 2);
    s->ts_win = true;
}

VkCommandBuffer vk_upload_cmd(void) {
    Sub *s = sub();
    if (!s->upload_begun) {
        begin_cb(&s->upload, &s->upload_begun);
        // after everything submitted before it: a submission's commands may start while the previous one's still run,
        // and these copies overwrite what those may read (textures being replaced, vertex cache pages)
        vk_full_barrier(s->upload);
    }
    return s->upload;
}

void vk_full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    rec_CmdPipelineBarrier2(cb, &di);
}

static void ring_retire(void);
void vc_flush(void);
static void xm_copy_back(VkCommandBuffer cb);

// ---- LARECOMP_GPUPROF_AT=<frame>: GPU time per pass / dispatch / copy of that frame, logged as "[gpuprof]".
// A timestamp (ALL_COMMANDS: after everything before it) at each mark; a row = the work since the previous mark.
bool g_gpuprof_frame;
static VkQueryPool gp_qp;
static int gp_n;
// LARECOMP_GPU_WRITES=1: the guest memory each GPU operation may write (a compute dispatch's storage
// buffers, a storing dispatch's copy back, transform feedback buffers, buffer copies, counters), the last GW_N of
// them, so a block found overwritten (ind_args_bad / ib_overrun: indirect parameters holding other data, #70 #81
// #82 #86) is logged with the operations that wrote over it in the frames before ("[gpuw]", 6 at most). On by default
// on AMD: #340 #345 #347 (and the RX 550s' #210 #336 #339) - without the overlap there, blocks the GPU reads in place
// hold floats for counts minutes in (seconds on an RX 550), the GPU is on them 7-10 s and the game stops on its
// fence; none of it shows on NVIDIA with AMD's paths forced, so the next such log names the writer. A ring of 16k
// entries, a few hundred notes a frame. =0: off.
#define GW_N 16384  // (a power of two)
static struct { uint64_t lo, hi, frame, prog; const char *what; } gw_ring[GW_N];
static unsigned gw_n;
static int gw_on = -1;
static void gw_note(const char *what, uint64_t lo, uint64_t size, uint64_t prog) {
    if (gw_on < 0) {
        gw_on = ENV("LARECOMP_GPU_WRITES") ? atoi(ENV("LARECOMP_GPU_WRITES")) != 0 : g_vk.props.vendorID == 0x1002;
        if (gw_on) LOG("[vk] GPU write report on: an overwritten indirect block names its writers ([gpuw]; LARECOMP_GPU_WRITES=0: off)\n");
    }
    if (!gw_on || !lo || !size) return;
    extern uint64_t renderer_exec_frame(void);
    unsigned i = gw_n++ & (GW_N - 1);
    gw_ring[i].lo = lo; gw_ring[i].hi = lo + size; gw_ring[i].frame = renderer_exec_frame(); gw_ring[i].prog = prog; gw_ring[i].what = what;
}
static void gw_explain(uint64_t pa) {
    static int n;
    if (gw_on < 0) gw_note("", 0, 0, 0);  // (decides gw_on)
    if (gw_on <= 0 || n >= 6) return;
    n++;
    extern uint64_t renderer_exec_frame(void);
    uint64_t f = renderer_exec_frame(), plo = pa & ~0xFFFull;
    LOG("[gpuw] frame %llu: block %llx overwritten; GPU writes over its page %llx in the last 8 frames:\n", (unsigned long long)f,
        (unsigned long long)pa, (unsigned long long)plo);
    unsigned lo = gw_n > GW_N ? gw_n - GW_N : 0, shown = 0;
    for (unsigned k = lo; k < gw_n && shown < 40; k++) {
        const typeof(gw_ring[0]) *e = &gw_ring[k & (GW_N - 1)];
        if (e->hi <= plo || e->lo >= plo + 0x1000 || e->frame + 8 < f) continue;
        shown++;
        LOG("[gpuw]   frame %llu %s %llx+%llx (program %llx)%s\n", (unsigned long long)e->frame, e->what, (unsigned long long)e->lo,
            (unsigned long long)(e->hi - e->lo), (unsigned long long)e->prog, e->lo <= pa && pa < e->hi ? " <- covers the block" : "");
    }
    if (!shown) LOG("[gpuw]   none: the CPU wrote it\n");
    const uint32_t *w = (const uint32_t *)(uintptr_t)(pa & ~0x3Full);
    LOG("[gpuw]   memory %llx:", (unsigned long long)(uintptr_t)w);
    for (int i = 0; i < 48; i++) LOG(" %u", w[i]);
    LOG("\n");
}

#define GP_MAX 4096  // (marks a frame: a town at World detail 4x has ~550 passes and ~70 submissions)
static char gp_label[GP_MAX][112];
static uint32_t gp_draws[GP_MAX];
// LARECOMP_VK_DEVLOST_TRACE=1: the last 64 GPU operations, dumped on device loss (with LARECOMP_SYNC_SUBMIT=1 the
// dump ends with the command list that faulted).
#define DL_N 4096  // (a power of two)
static char dl_ring[DL_N][128];
static uint64_t dl_args[DL_N];
static uint32_t dl_snap[DL_N][5];  // (those parameters when the draw was recorded)  // a GPU indirect draw's parameters (guest = host address): read back in the dump
static unsigned dl_n;
static int dl_on = -1;
static void dl_note(const char *fmt, ...) {
    if (dl_on < 0) dl_on = ENV("LARECOMP_VK_DEVLOST_TRACE") ? atoi(ENV("LARECOMP_VK_DEVLOST_TRACE")) : 0;
    if (!dl_on) return;
    va_list ap;
    va_start(ap, fmt);
    dl_args[dl_n & (DL_N - 1)] = 0;
    vsnprintf(dl_ring[dl_n++ & (DL_N - 1)], sizeof dl_ring[0], fmt, ap);
    va_end(ap);
}
void vk_dl_note_upload(void *img, uint32_t w, uint32_t h, uint32_t lv) { dl_note("  upload %p %ux%u lv%u", img, w, h, lv); }
void vk_devlost_dump(void) {
    if (!dl_on) return;
    LOG("[vk] last GPU operations before the device loss (oldest first):\n");
    for (unsigned i = dl_n > 64 ? dl_n - 64 : 0; i < dl_n; i++) {
        const uint32_t *q = (const uint32_t *)(uintptr_t)dl_args[i & (DL_N - 1)];
        if (q) LOG("[vk]   %s (args now: %u %u %u %d %u)\n", dl_ring[i & (DL_N - 1)], q[0], q[1], q[2], (int)q[3], q[4]);
        else LOG("[vk]   %s\n", dl_ring[i & (DL_N - 1)]);
    }
    // VK_NV_device_diagnostic_checkpoints: the last draw / dispatch each pipeline stage reached
    extern uint32_t vk_checkpoints(uint64_t *mark, uint32_t *stage, uint32_t max);
    uint64_t mark[16];
    uint32_t stage[16], n = vk_checkpoints(mark, stage, 16);
    for (uint32_t i = 0; i < n; i++) {
        uint64_t m = mark[i];
        LOG("[vk] checkpoint stage %#x: #%llu %s\n", stage[i], (unsigned long long)m,
            m && m <= dl_n && dl_n - m < DL_N ? dl_ring[(m - 1) & (DL_N - 1)] : "(out of the trace)");
    }
    // the operations from the oldest checkpoint on (in flight when the GPU stopped), with indirect parameters
    uint64_t lo = ~0ull;
    for (uint32_t i = 0; i < n; i++) if (mark[i] && mark[i] < lo) lo = mark[i];
    if (lo != ~0ull && dl_n - lo < DL_N)
        for (uint64_t m = lo; m <= dl_n && m < lo + 48; m++) {
            const uint32_t *q = (const uint32_t *)(uintptr_t)dl_args[(m - 1) & (DL_N - 1)];
            const uint32_t *w = dl_snap[(m - 1) & (DL_N - 1)];
            if (q) LOG("[vk]   #%llu %s (args %llx now: %u %u %u %d %u, recorded: %u %u %u %d %u)\n", (unsigned long long)m,
                       dl_ring[(m - 1) & (DL_N - 1)], (unsigned long long)(uintptr_t)q, q[0], q[1], q[2], (int)q[3], q[4],
                       w[0], w[1], w[2], (int)w[3], w[4]);
            else LOG("[vk]   #%llu %s\n", (unsigned long long)m, dl_ring[(m - 1) & (DL_N - 1)]);
        }
}
static void gp_mark(VkCommandBuffer cb, const char *fmt, ...) {
    if (!g_gpuprof_frame || gp_n >= GP_MAX - 1) return;
    if (!gp_qp) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = GP_MAX};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &gp_qp) != VK_SUCCESS) { g_gpuprof_frame = false; return; }
    }
    if (!gp_n) rec_CmdResetQueryPool(cb, gp_qp, 0, GP_MAX);
    rec_CmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, gp_qp, (uint32_t)gp_n);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(gp_label[gp_n], sizeof gp_label[0], fmt, ap);
    va_end(ap);
    gp_draws[gp_n] = 0;
    gp_n++;
}
// A slow frame the GPU held up (slowframe.c asks): the frame after it is profiled as LARECOMP_GPUPROF_AT does, and only
// its heaviest rows go to the log - which pass / dispatch / shader costs the time, on GPUs we can't test here.
// At most 3 a session, a minute apart; one frame waits for its timestamps (a hitch of one frame). LARECOMP_GPUPROF_AUTO=0: off.
static int gp_auto_want, gp_auto_n;
static bool gp_auto;
static uint64_t gp_auto_last;
void vk_gpuprof_request(void) {
    static int off = -1;
    if (off < 0) off = ENV("LARECOMP_GPUPROF_AUTO") && !atoi(ENV("LARECOMP_GPUPROF_AUTO"));
    uint64_t now = plat_now_ns();
    if (off || gp_auto_n >= 3 || (gp_auto_last && now - gp_auto_last < 60000000000ull)) return;
    gp_auto_last = now, gp_auto_n++;
    __atomic_store_n(&gp_auto_want, 1, __ATOMIC_RELEASE);
}
bool vk_gpuprof_take_request(void) {  // (device.c, at a frame's start)
    if (!__atomic_exchange_n(&gp_auto_want, 0, __ATOMIC_ACQ_REL)) return false;
    gp_auto = true;
    return true;
}
static int gp_cmp_ms(const void *a, const void *b) {
    double x = ((const double *)a)[0], y = ((const double *)b)[0];
    return x < y ? 1 : x > y ? -1 : 0;
}
void renderer_flush(void);
void vk_gpuprof_report(void) {
    if (!gp_n) { gp_auto = false; return; }
    renderer_flush();
    static uint64_t t[GP_MAX];
    if (vkGetQueryPoolResults(g_vk.dev, gp_qp, 0, (uint32_t)gp_n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
        double per = g_vk.props.limits.timestampPeriod / 1e6, total = (t[gp_n - 1] - t[0]) * per;
        if (gp_auto) {  // (the heaviest rows, heaviest first)
            static double rows[GP_MAX][2];
            int n = 0;
            for (int i = 0; i + 1 < gp_n; i++) rows[n][0] = (t[i + 1] - t[i]) * per, rows[n][1] = i, n++;
            qsort(rows, (size_t)n, sizeof rows[0], gp_cmp_ms);
            LOG("[gpuprof] slow frame: GPU %.2f ms in %d rows; the heaviest (LARECOMP_GPUPROF_AUTO=0: off)\n", total, n);
            for (int k = 0; k < n && k < 12; k++) {
                int i = (int)rows[k][1];
                LOG("[gpuprof] %6.3f ms %5.1f%%  %s", rows[k][0], 100 * rows[k][0] / (total > 0 ? total : 1), gp_label[i]);
                if (gp_draws[i]) LOG(", %u draws", gp_draws[i]);
                LOG("\n");
            }
            gp_auto = false;
            gp_n = 0;
            return;
        }
        LOG("[gpuprof] %d marks, %.2f ms first to last (rows: the work after each mark, until the next)\n", gp_n, total);
        for (int i = 0; i + 1 < gp_n; i++) {
            double ms = (t[i + 1] - t[i]) * per;
            LOG("[gpuprof] %6.3f ms %5.1f%%  %s%s", ms, 100 * ms / (total > 0 ? total : 1), gp_label[i], gp_draws[i] ? "" : "\n");
            if (gp_draws[i]) LOG(", %u draws\n", gp_draws[i]);
        }
    }
    gp_auto = false;
    gp_n = 0;
}

// Submit the pending work (if any). wait/signal: binary semaphores for presentation (may be null).
void vk_submit_extra(VkSemaphore wait, VkSemaphore signal, VkCommandBuffer extra) {
    (void)extra;
    if (sw_cb) {
        static int n;
        if (n++ < 10) LOG("[sw] a shadow worker asked for a submission: ignored\n");
        return;
    }
    sw_async_sync();  // (the workers' secondaries, vertex cache ranges and uploads belong to this submission)
    vk_end_pass();
    vk_tex_unpack_finish();  // (the upload copies recorded so far read what it writes)
    Sub *s = sub();
    if (!s->main_begun && !s->upload_begun && !wait && !signal) return;
    if (s->main_begun) xm_copy_back(s->main);  // (transform feedback output, to the heap)
    VkCommandBufferSubmitInfo cbs[2];
    int n = 0;
    vc_flush();
    if (s->upload_begun) {
        vk_full_barrier(s->upload);  // uploads complete before anything in the main buffer reads them
        rec_EndCommandBuffer(s->upload);
        cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->upload};
    }
    vk_cmd();  // (an empty main buffer still carries the semaphores)
    s->ts_skip = wait != VK_NULL_HANDLE;
    if (s->ts_win && !s->ts_skip) s->ts_win = false;  // (not the window image's after all: counted as game work)
    if (s->ts) rec_CmdWriteTimestamp2(s->main, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, prof_qp, (uint32_t)cur_sub * 3 + 1);
    rec_EndCommandBuffer(s->main);
    cbs[n++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->main};
    VkSemaphoreSubmitInfo sig[2] = {{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_vk.timeline,
                                     .value = submit_seq + 1, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
                                    {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = signal,
                                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    VkSemaphoreSubmitInfo wt = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = wait,
                                .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = (uint32_t)n, .pCommandBufferInfos = cbs,
                        .signalSemaphoreInfoCount = signal ? 2 : 1, .pSignalSemaphoreInfos = sig,
                        .waitSemaphoreInfoCount = wait ? 1 : 0, .pWaitSemaphoreInfos = &wt};
    dl_note("submit %llu", (unsigned long long)submit_seq + 1);
    rec_QueueSubmit2(&si, submit_seq + 1);  // (the recorder: in order with the commands; its vkQueueSubmit2 time is in RecStats)
    pthread_mutex_lock(&seq_lock);
    submit_seq++;
    cur_sub = (cur_sub + 1) % SUBS;
    pthread_mutex_unlock(&seq_lock);
    passes_in_sub = 0;
    stat_flushes++;
    cum_submits++;
    submit_draws = cum_draws;
    ring_retire();
    vk_collect_dead();
    if (signal) gc_frame();  // (a present: once a frame)
}

void renderer_submit(void) { vk_submit_extra(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE); }

static void overlap_clear(void) {  // (a full wait was made: everything queued so far is done)
    pthread_mutex_lock(&seq_lock);
    deferred_seq = 0;
    pthread_mutex_unlock(&seq_lock);
}

void renderer_flush(void) {
    renderer_submit();
    if (!wait_why) wait_why = "flush";
    vk_wait_seq(submit_seq);
    overlap_clear();
}

// The latest submission the driver has (rqueue.h): with the recorder, behind renderer_issued_seq until it catches up.
uint64_t renderer_submit_seq(void) { return rec_submitted(); }

// The latest submission this backend queued (rqueue.c: an item's work ends with it; renderer_wait_seq waits for the
// recorder to hand it to the driver, then for the GPU)
uint64_t renderer_issued_seq(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    return s;
}

void renderer_sync(void) {
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    wait_why = "QueueFinish / sync all";
    vk_wait_seq(s);
    overlap_clear();
}

// (overlap) The game's SyncWait on submission seq: done is the latest complete submission, *deferred the one a previous
// call returned early for (0: none). Returns the submission to wait for now (0: none); updates *deferred. A fence
// that is complete is never waited for; a newer fence than the deferred one waits for the deferred one and takes its
// place; the deferred fence itself, or an older one, is waited for. (tools/test_vulkan_overlap.py)
static uint64_t overlap_defer(uint64_t seq, uint64_t done, uint64_t *deferred) {
    uint64_t prev = *deferred;
    if (seq <= done) {  // complete (in order: a deferred one no newer than it is complete too)
        if (seq >= prev) *deferred = 0;
        return 0;
    }
    if (seq > prev) {  // a newer fence: the deferred one is waited for now, this one at the next call
        *deferred = seq;
        return prev > done ? prev : 0;
    }
    if (seq == prev) *deferred = 0;  // the deferred fence again, or an older one: a real wait on it
    return seq;
}

uint64_t renderer_fence_seq(void) {
    pthread_mutex_lock(&seq_lock);
    Sub *s = &subs[cur_sub];
    // A reused slot still has its old begun flags until sub() prepares it. An empty queue fence must refer to
    // the last submission, not a future value that submit's empty-work fast path will never signal.
    bool pending = s->seq == submit_seq + 1 && (s->main_begun || s->upload_begun);
    uint64_t q = submit_seq + (pending ? 1 : 0);
    pthread_mutex_unlock(&seq_lock);
    return q;
}

void renderer_wait_seq(uint64_t seq) {
    if (!seq) return;
    pthread_mutex_lock(&seq_lock);
    uint64_t s = submit_seq;
    pthread_mutex_unlock(&seq_lock);
    wait_why = seq > s ? "SyncWait (fence not submitted yet)" : "SyncWait";
    if (seq > s) seq = s;  // not queued yet (other thread): everything queued so far (vk_wait_seq: the recorder, then the GPU)
    if (overlap_on()) {  // (see deferred_seq) wait for the fence deferred last time, defer this one
        uint64_t done = vk_completed_seq();  // (the timeline's value: a submission the recorder hasn't made yet isn't done)
        pthread_mutex_lock(&seq_lock);
        uint64_t w = overlap_defer(seq, done, &deferred_seq);
        pthread_mutex_unlock(&seq_lock);
        if (!w) {
            __atomic_fetch_add(&stat_deferred, 1, __ATOMIC_RELAXED);  // (the game thread; stats read on the render thread)
            wait_why = NULL;
            cq_resolve_done(done);  // (the game reads its timing counters now: what is complete so far)
            return;
        }
        seq = w;
    }
    vk_wait_seq(seq);
}

// A descriptor set for the pending submission (record.c: the slot's pools, allocated in batches per layout). With
// the recorder on it is a stand-in, to be written and bound through rec_* only, and only for this submission.
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout layout) {
    sub();
    return rec_alloc_set(cur_sub, layout);
}

// ================================================================== transient memory ring
// 64 MB host-visible chunks (device-local when the GPU exposes resizable BAR); a chunk is reused once every
// submission that allocated from it has completed. Without resizable BAR the device-local host-visible memory is a
// small window (256 MB on a GTX 1060): once it's full, later chunks go to system memory (issue #45: it stopped the
// game with "out of device memory").
#define CHUNK_SIZE (64ull << 20)
typedef struct { VkBuffer buf; VmaAllocation mem; uint8_t *ptr; uint64_t size, used, seq; } Chunk;
static Chunk *chunks;
static int nchunks, active_t[NCTX] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
#define active (active_t[sw_ctx()])
static pthread_mutex_t ring_mu = PTHREAD_MUTEX_INITIALIZER;  // (picking / making chunks: shadow workers too)
static bool chunk_held(int i) { for (int k = 0; k < NCTX; k++) if (active_t[k] == i) return true; return false; }
// (per-object motion vectors, objmv_capture / objmv_replay) while the characters' draws wait to be drawn again into the
// motion image, the chunks used since the first of them (seq >= mv_pin) aren't recycled, though the submissions that
// used them finished; mv_unpin: they now wait for the next submission. mv_lost: no memory for another chunk
// (vk_transient) took the pin away, so that frame keeps the camera's motion. (ring_mu: shadow workers capture too)
static uint64_t mv_pin;
static _Atomic bool mv_lost;
static void mv_unpin(void) {
    pthread_mutex_lock(&ring_mu);
    if (mv_pin) {
        uint64_t pending = vk_pending_seq();
        for (int i = 0; i < nchunks; i++) if (chunks[i].seq >= mv_pin) chunks[i].seq = pending;
        mv_pin = 0;
    }
    pthread_mutex_unlock(&ring_mu);
}

static int new_chunk(uint64_t size) {
    chunks = realloc(chunks, (nchunks + 1) * sizeof *chunks);
    Chunk *c = &chunks[nchunks];
    memset(c, 0, sizeof *c);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    static bool sysmem;  // (the device-local window ran out: system memory from then on)
    // Without resizable BAR the device-local host-visible window is ~256 MB, shared with other programs and the driver:
    // two 64 MB chunks took 128 of its 132 MB in use while its budget fell to 81 MB (RTX 3080 Ti), and a GTX 1060 ran
    // out (issue #45). A chunk goes there only while the window's use stays under half its budget; past that, system
    // memory (~1 ms more GPU time a frame on a 3080 Ti at 1x). The first chunks, which most frames use, stay fast.
    // LARECOMP_VK_RING_BAR=1: no budget check.
    bool spill = false;
    if (!sysmem && !getenv("LARECOMP_VK_RING_BAR")) {
        uint32_t type;
        if (vmaFindMemoryTypeIndexForBufferInfo(g_vk.vma, &bi, &ai, &type) == VK_SUCCESS) {
            const VkPhysicalDeviceMemoryProperties *mp;
            vmaGetMemoryProperties(g_vk.vma, &mp);
            uint32_t heap = mp->memoryTypes[type].heapIndex;
            VmaBudget bud[VK_MAX_MEMORY_HEAPS];
            vmaGetHeapBudgets(g_vk.vma, bud);
            spill = (mp->memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                    bud[heap].usage + size > bud[heap].budget / 2;
            static bool logged;
            if (spill && !logged) {
                logged = true;
                LOG("[vk] transient ring: device-local window %llu of %llu MB budget in use: new chunks in system memory\n",
                    (unsigned long long)(bud[heap].usage >> 20), (unsigned long long)(bud[heap].budget >> 20));
            }
        }
    }
    if (sysmem || spill || getenv("LARECOMP_VK_RING_SYSMEM")) ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;  // (test: a GPU without resizable BAR)
    VmaAllocationInfo info;
    VkResult r = vmaCreateBuffer(g_vk.vma, &bi, &ai, &c->buf, &c->mem, &info);
    bool moved = false;
    // (a driver may report the full window as out of host memory too; if memory is really gone, the retry fails as well)
    if ((r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) && ai.usage != VMA_MEMORY_USAGE_AUTO_PREFER_HOST) {
        LOG("[vk] transient ring: device-local memory full after %d chunks, system memory from now on\n", nchunks);
        vk_log_memory("ring chunk failed");
        sysmem = moved = true;
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        r = vmaCreateBuffer(g_vk.vma, &bi, &ai, &c->buf, &c->mem, &info);
    }
    if (r != VK_SUCCESS) {  // (system memory too: vk_transient waits for a chunk to come free instead - issue #45)
        LOG("[vk] transient ring: no memory for chunk %d (%llu MB, error %d)\n", nchunks, (unsigned long long)(size >> 20), r);
        vk_log_memory("ring chunk failed");
        return -1;
    }
    c->ptr = info.pMappedData;
    c->size = size;
    static bool spilled;  // (the first chunk in system memory: logged too)
    if (!nchunks || moved || (spill && !spilled)) {  // where the ring lives: device-local (resizable BAR) or system memory read across PCIe
        if (spill) spilled = true;
        VkMemoryPropertyFlags f;
        vmaGetMemoryTypeProperties(g_vk.vma, info.memoryType, &f);
        LOG("[vk] transient ring: memory type %u (%s%s%s)\n", info.memoryType, f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "device-local " : "system memory ",
            f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? "host-visible " : "", f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "cached" : "");
    }
    return nchunks++;
}

static void ring_retire(void) {}  // chunks are recycled lazily in vk_transient

// (planar reflections) while their draws wait to be drawn again, the chunks used since the first (seq >= ring_pin)
// aren't recycled, though the submissions that used them finished; ring_unpin: they now wait for the next submission.
// ring_lost: no memory for another chunk (vk_transient) took the pin away, so that frame has no reflection. (ring_mu:
// shadow workers capture the G-buffer's draws too, as mv_pin's)
static uint64_t ring_pin;
static bool ring_lost;
static void ring_unpin(void) {
    pthread_mutex_lock(&ring_mu);
    if (ring_pin) {
        uint64_t pending = vk_pending_seq();
        for (int i = 0; i < nchunks; i++) if (chunks[i].seq >= ring_pin) chunks[i].seq = pending;
        ring_pin = 0;
    }
    pthread_mutex_unlock(&ring_mu);
}

VkSpan vk_transient(uint64_t size, uint64_t align) {
    if (align < 16) align = 16;
    uint64_t pending = vk_pending_seq();
    if (active >= 0) {
        Chunk *c = &chunks[active];
        uint64_t off = (c->used + align - 1) & ~(align - 1);
        if (off + size <= c->size) {
            c->used = off + size;
            c->seq = pending;
            stat_upload_bytes += size;
            return (VkSpan){c->buf, off, c->ptr + off};
        }
    }
    pthread_mutex_lock(&ring_mu);
    uint64_t done = vk_completed_seq();
    int pick = -1;
    for (int i = 0; i < nchunks; i++)
        if (!chunk_held(i) && chunks[i].seq <= done && !(ring_pin && chunks[i].seq >= ring_pin) && !(mv_pin && chunks[i].seq >= mv_pin) && chunks[i].size >= size) { pick = i; break; }
    if (pick < 0) pick = new_chunk(size > CHUNK_SIZE ? size : CHUNK_SIZE);
    if (pick < 0) {  // no memory for another chunk (a GTX 1060 with 6 GB: issue #45): wait for the work already submitted,
        // which frees the chunks only it used, and take one of those (the recording in progress keeps its own)
        if (ring_pin) ring_pin = 0, ring_lost = true;
        if (mv_pin) mv_pin = 0, atomic_store_explicit(&mv_lost, true, memory_order_relaxed);
        vk_wait_seq(submit_seq);
        done = vk_completed_seq();
        for (int i = 0; i < nchunks; i++)
            if (!chunk_held(i) && chunks[i].seq <= done && chunks[i].size >= size) { pick = i; break; }
        if (pick < 0) FATAL("the graphics card ran out of memory for uploads (%llu MB needed, %d chunks in use)\n",
                            (unsigned long long)(size >> 20), nchunks);
        static int nlog;
        if (nlog++ < 5) LOG("[vk] transient ring: reused chunk %d after waiting for the GPU\n", pick);
    }
    active = pick;
    chunks[pick].used = size;
    chunks[pick].seq = pending;
    pthread_mutex_unlock(&ring_mu);
    stat_upload_bytes += size;
    return (VkSpan){chunks[pick].buf, 0, chunks[pick].ptr};
}

// GPU -> CPU copies: the ring is write-combined (often device-local over the BAR), where CPU reads crawl (a
// 1280x720 screenshot took 7 s). Readbacks use one host-cached buffer instead. Every user waits for the GPU and
// reads the data before the next readback (render thread), so a single grow-only buffer suffices.
VkSpan vk_readback(uint64_t size) {
    static VkBuffer buf;
    static VmaAllocation mem;
    static uint8_t *ptr;
    static uint64_t cap;
    if (size > cap) {
        if (buf) { vk_wait_seq(submit_seq); vmaDestroyBuffer(g_vk.vma, buf, mem); }
        cap = (size + (16 << 20) - 1) & ~((16ull << 20) - 1);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &buf, &mem, &info));
        ptr = info.pMappedData;
    }
    return (VkSpan){buf, 0, ptr};
}

VkSpan vk_transient_copy(const void *data, uint64_t size, uint64_t align) {
    VkSpan s = vk_transient(size ? size : 16, align);
    if (size) memcpy(s.ptr, data, size);
    return s;
}

// Per-draw constants that usually repeat the previous draw's (scale table, tint, driver constants): reuse that copy
// while it's identical, within one submission. slot: one per kind of data.
enum { DEDUPE_RSCALE_V, DEDUPE_RSCALE_F, DEDUPE_TINT, DEDUPE_SLOTS };  // (the driver constants: cbuf_info)
static VkSpan copy_dedupe(int slot, const void *data, uint64_t size, uint64_t align) {
    static struct { uint64_t seq, size; uint8_t data[0x800]; VkSpan span; } last_t[NCTX][DEDUPE_SLOTS];
    __typeof__(last_t[0][0]) *last = last_t[sw_ctx()];
    uint64_t seq = vk_pending_seq();
    if (size > sizeof last[0].data) return vk_transient_copy(data, size, align);
    if (last[slot].seq == seq && last[slot].size == size && !memcmp(last[slot].data, data, size)) return last[slot].span;
    VkSpan s = vk_transient_copy(data, size, align);
    last[slot].seq = seq; last[slot].size = size; memcpy(last[slot].data, data, size); last[slot].span = s;
    return s;
}

// Guest memory copied once per submission: draws binding the same range share the copy.
#define GCOPY_TAB 4096
static struct { uint64_t addr, size, seq; VkSpan span; uint64_t scanned; } gcopy_t[NCTX][GCOPY_TAB];  // (scanned: heap size then, 0 not)
#define gcopy (gcopy_t[sw_ctx()])
static uint64_t gcopy_gen;  // bumped to drop every shared copy (guest memory written mid-list)
static void guest_copy_flush(void) { gcopy_gen++; }
// scan: (lazy heap chunks, a uniform buffer of a shader with global memory) the range is also scanned for heap
// pointers (gc_scan_pointers), once per shared copy: its contents can't change while the copy is shared (again if the
// heap grew since: pointers past its end were passed over).
static bool gc_scan_needed(void);
static VkSpan guest_copy_ex(uint64_t addr, uint64_t size, bool scan) {
    uint64_t seq = vk_pending_seq() + (gcopy_gen << 40), h = (addr * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (GCOPY_TAB - 1);
    bool hit = gcopy[slot].seq == seq && gcopy[slot].addr == addr && gcopy[slot].size == size;
    uint64_t heap = 0;
    if (scan) {
        heap = gmem_heap_size();
        if ((!hit || gcopy[slot].scanned != heap) && gc_scan_needed()) { SWL_AT(987); gc_scan_pointers((const uint32_t *)(uintptr_t)addr, size); SWU(); }
        else prof_scan_skipped++;
    }
    if (hit) {
        if (scan) gcopy[slot].scanned = heap;
        return gcopy[slot].span;
    }
    VkSpan s = vk_transient_copy((const void *)(uintptr_t)addr, size, 256);
    gcopy[slot].addr = addr; gcopy[slot].size = size; gcopy[slot].seq = seq; gcopy[slot].span = s; gcopy[slot].scanned = heap;
    return s;
}
static VkSpan guest_copy(uint64_t addr, uint64_t size) { return guest_copy_ex(addr, size, false); }

// ================================================================== worker threads (shader / pipeline builds)
// First-use compiles run off the render thread; draws needing them are skipped until ready (as on Metal).
// LARECOMP_SYNC_SHADERS=1 builds on first use, blocking (deterministic tests).
// state: what the build sets when done (a draw waiting on it can run it itself: run_queued_now)
typedef struct Job { void (*fn)(void *); void *arg; _Atomic int *state; struct Job *next; } Job;
static Job *jobs_head, *jobs_tail;
static pthread_mutex_t jobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t jobs_cond = PTHREAD_COND_INITIALIZER;
// when each thread's current job started (0: idle) - wait_built says which one a stalled draw is stuck behind
#define MAX_WORKERS 16
static _Atomic uint64_t worker_t0[MAX_WORKERS], translator_t0;
static _Atomic int nworkers;
static _Atomic uint64_t builds_done;  // (shaders / pipelines built, all session: renderer_shader_builds)
// Shader work for runtime/hle/framepace.c (seconds spent building shaders don't count against 60 fps): the builds
// done so far, top bit set while one is being built
uint64_t renderer_shader_builds(void) {
    uint64_t n = atomic_load(&builds_done);
    if (atomic_load(&translator_t0)) n |= 1ull << 63;
    for (int i = 0; i < atomic_load(&nworkers); i++) if (atomic_load(&worker_t0[i])) n |= 1ull << 63;
    return n;
}

static void *worker(void *a) {
    int me = (int)(intptr_t)a;
    for (;;) {
        pthread_mutex_lock(&jobs_lock);
        while (!jobs_head) pthread_cond_wait(&jobs_cond, &jobs_lock);
        Job *j = jobs_head;
        jobs_head = j->next;
        if (!jobs_head) jobs_tail = NULL;
        pthread_mutex_unlock(&jobs_lock);
        atomic_store(&worker_t0[me], SDL_GetTicksNS());
        j->fn(j->arg);
        atomic_store(&worker_t0[me], 0);
        free(j);
    }
    return NULL;
}

static bool sync_shaders(void) {
    static int v = -1;
    if (v < 0) v = ENV("LARECOMP_SYNC_SHADERS") != NULL;
    return v;
}

// Shader translation: one dedicated thread (the translator isn't re-entrant - translate.c's RSC2 string ring - and
// translating on the game thread cost up to ~90 ms per frame on entering an area). Compiles go on to the workers.
static pthread_mutex_t translate_lock = PTHREAD_MUTEX_INITIALIZER;  // (compute shaders translate on the game thread)
static Job *tjobs_head, *tjobs_tail;
static pthread_mutex_t tjobs_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t tjobs_cond = PTHREAD_COND_INITIALIZER;
static void *translator(void *a) {
    (void)a;
    for (;;) {
        pthread_mutex_lock(&tjobs_lock);
        while (!tjobs_head) pthread_cond_wait(&tjobs_cond, &tjobs_lock);
        Job *j = tjobs_head;
        tjobs_head = j->next;
        if (!tjobs_head) tjobs_tail = NULL;
        pthread_mutex_unlock(&tjobs_lock);
        atomic_store(&translator_t0, SDL_GetTicksNS());
        j->fn(j->arg);
        atomic_store(&translator_t0, 0);
        free(j);
    }
    return NULL;
}
static bool sync_shaders(void);
static void run_translate(void (*fn)(void *), void *arg, _Atomic int *state) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    if (!started) { started = true; pthread_t t; pthread_create(&t, NULL, translator, NULL); }
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->state = state; j->next = NULL;
    pthread_mutex_lock(&tjobs_lock);
    if (tjobs_tail) tjobs_tail->next = j; else tjobs_head = j;
    tjobs_tail = j;
    pthread_cond_signal(&tjobs_cond);
    pthread_mutex_unlock(&tjobs_lock);
}

static void run_job(void (*fn)(void *), void *arg, _Atomic int *state) {
    if (sync_shaders()) { fn(arg); return; }
    static bool started;
    if (!started) {
        started = true;
        // first entry into an area builds dozens of pipelines at once: use most of the CPU (draws wait for them)
        int n = plat_cpu_count();  // (the process's affinity mask: a process held to a few cores, few workers)
        if (n <= 0) n = SDL_GetNumLogicalCPUCores();
        n -= 4;
        if (n < 4) n = 4;
        if (n > MAX_WORKERS) n = MAX_WORKERS;
        nworkers = n;
        for (int i = 0; i < n; i++) { pthread_t t; pthread_create(&t, NULL, worker, (void *)(intptr_t)i); }
    }
    Job *j = malloc(sizeof *j);
    j->fn = fn; j->arg = arg; j->state = state; j->next = NULL;
    pthread_mutex_lock(&jobs_lock);
    if (jobs_tail) jobs_tail->next = j; else jobs_head = j;
    jobs_tail = j;
    pthread_cond_signal(&jobs_cond);
    pthread_mutex_unlock(&jobs_lock);
}

// A draw that has to wait for its build (wait_built) takes it out of the queues - the translator's, then the
// workers' - and runs it on its own thread: queued behind the dozens a new area or effect brings, it waited for all
// of them (over a second, a frozen frame). false: not queued (running, or done).
// The queued job building state, taken out of its queue (take) or moved to its front, for the next free thread: a draw
// skipped for it (wait_scene) gets it in a build or two, not after every build queued before it.
static Job *queued_job(_Atomic int *state, bool take) {
    for (int q = 0; q < 2; q++) {
        pthread_mutex_t *lock = q ? &jobs_lock : &tjobs_lock;
        Job **head = q ? &jobs_head : &tjobs_head, **tail = q ? &jobs_tail : &tjobs_tail, *prev = NULL;
        pthread_mutex_lock(lock);
        for (Job **pj = head; *pj; prev = *pj, pj = &(*pj)->next) {
            Job *j = *pj;
            if (j->state != state) continue;
            if (take || pj != head) {
                *pj = j->next;
                if (*tail == j) *tail = prev;
                if (!take) j->next = *head, *head = j;
            }
            pthread_mutex_unlock(lock);
            return j;
        }
        pthread_mutex_unlock(lock);
    }
    return NULL;
}
static bool run_queued_now(_Atomic int *state) {
    Job *j = queued_job(state, true);
    if (!j) return false;
    j->fn(j->arg);
    free(j);
    return true;
}

// ================================================================== state
typedef struct {
    NvnTexture *color[8];
    NvnTextureView cview[8];
    uint8_t has_cview[8];
    int ncolor;
    NvnTexture *depth;
    NvnTextureView dview;
    uint8_t has_dview;
} RtState;

typedef struct { uint64_t addr, size; } Range;

typedef struct TState {
    RtState rt;
    bool rt_dirty;
    NvnProgram *program;  // &comp, or NULL with no stage bound
    NvnProgram comp;      // the bound stages: BindProgram(program, stages) replaces just those
    uint8_t blend[8][8];
    uint32_t channel_mask;
    uint8_t color_state[4], ds[8], ms[0x18], poly[4];
    float poly_offset[3];
    uint8_t depth_clamp;  // nvnCommandBufferSetDepthClamp
    uint32_t attrib[16];
    int nattrib;
    uint32_t stream[16][2];
    int nstream;
    Range vbuf[16], ubuf[6][18], sbuf[6][16];
    uint64_t tex[6][32], img[6][8];
    NvnDescPool *tpool, *spool;
    float viewport[16][4];
    int32_t scissor[16][4];
    float depth_range[16][2];
    float blend_color[4];
    uint32_t stencil_ref[2], stencil_mask[2], stencil_vmask[2];
    bool prim_restart;
    uint32_t restart_index;
    uint64_t copy_row_stride, copy_image_stride;
    bool raster_discard;  // (SetRasterizerDiscard: state - a pass of Scarlet / Violet sets it before each of its draws)
} TState;
// The translation state. The render thread's is st_main; a shadow worker (sw_*) translates its part of a depth-only
// pass on a copy of it, made at the part's start - st is the calling thread's (st_tls), the render thread's when none.
static TState st_main;
static __thread TState *st_tls;
static inline TState *st_cur(void) { TState *p = st_tls; return p ? p : &st_main; }
#define st (*st_cur())

uint32_t nvn_window_origin(void);
uint32_t nvn_depth_mode(void);
bool renderer_origin_lower_left(void) { return nvn_window_origin() == 0; }

void vk_prof_frame(uint64_t frame, uint64_t present_ns) {
    if (!prof) return;
    static uint64_t last_t, l_exec, l_wait, l_draws, l_subs;
    uint64_t now = SDL_GetTicksNS();
    if (last_t && now - last_t > 50000000) {
        uint64_t dt = now - last_t, ex = cum_exec - l_exec, wt = cum_wait - l_wait;
        LOG("[hitch] frame %llu: %.0f ms = backend exec %.1f (of which GPU waits %.1f) + present %.1f + other %.1f; %llu draws, "
            "%llu submits, compiles %d pending\n", (unsigned long long)frame, dt / 1e6, ex / 1e6, wt / 1e6, present_ns / 1e6,
            ((double)dt - ex - present_ns) / 1e6, (unsigned long long)(cum_draws - l_draws), (unsigned long long)(cum_submits - l_subs),
            atomic_load(&stat_compiles));
    }
    last_t = now; l_exec = cum_exec; l_wait = cum_wait; l_draws = cum_draws; l_subs = cum_submits;
}

// (the slow-frame log, slowframe.h; the game thread) running totals this file keeps anyway
void renderer_slow_counters(uint64_t *submits, uint64_t *gpu_wait_ns, uint64_t *imports) {
    *submits = __atomic_load_n(&cum_submits, __ATOMIC_RELAXED);
    *gpu_wait_ns = __atomic_load_n(&cum_wait, __ATOMIC_RELAXED);
    *imports = (uint64_t)__atomic_load_n(&gc_imported, __ATOMIC_RELAXED);
}

// LARECOMP_LOG_SKIPS=1: the skipped draws and dispatches of each stats interval by reason, and the top offenders
// (shader code addresses and content hashes, as in "[shader] slow build" / shader_variants.txt, and how long they
// have been skipped without a break). "building" for tens of seconds is a build that never finishes.
enum { SK_NO_PROGRAM, SK_NO_STAGES, SK_PROG_BUILDING, SK_PROG_FAILED, SK_PIPE_BUILDING, SK_PIPE_FAILED, SK_CS_BUILDING,
       SK_CS_FAILED, SK_TOPOLOGY, SK_DEBUG_SKIP, SK_HIDDEN, SK_N };
static const char *const skip_names[SK_N] = {
    "no program", "program missing a stage", "program building", "program failed", "pipeline building", "pipeline failed",
    "compute building", "compute failed", "topology", "debug skip list", "hidden on purpose"};
static int log_skips = -1;  // (-1: not read yet: the first skip reads it)
static uint64_t skip_n[SK_N];
#define SKIP_TOP 32
static struct { int why; uint64_t a, b, ha, hb, n, first_ns; } skip_top[SKIP_TOP];
static int skip_ntop;
static void skip_note(int why, bool compute) {
    if (log_skips < 0) log_skips = getenv("LARECOMP_LOG_SKIPS") && atoi(getenv("LARECOMP_LOG_SKIPS")) != 0;
    if (!log_skips) return;
    skip_n[why]++;
    const NvnProgram *p = st.program;
    uint64_t a = p ? p->stages[compute ? 5 : 0].code_addr : 0, b = p && !compute ? p->stages[1].code_addr : 0;
    int j = 0;
    while (j < skip_ntop && !(skip_top[j].why == why && skip_top[j].a == a && skip_top[j].b == b)) j++;
    if (j == skip_ntop) {
        if (skip_ntop < SKIP_TOP) skip_ntop++;
        else {  // (full: the entry skipped least this interval goes)
            j = 0;
            for (int i = 1; i < SKIP_TOP; i++) if (skip_top[i].n < skip_top[j].n) j = i;
        }
        skip_top[j].why = why; skip_top[j].a = a; skip_top[j].b = b; skip_top[j].n = 0;
        skip_top[j].ha = p ? shader_stage_hash(&p->stages[compute ? 5 : 0]) : 0;
        skip_top[j].hb = p && !compute ? shader_stage_hash(&p->stages[1]) : 0;
        skip_top[j].first_ns = SDL_GetTicksNS();
    }
    skip_top[j].n++;
}
#define SKIP(why) do { stat_skipped++; if (log_skips) skip_note(why, false); } while (0)
#define SKIP_CS(why) do { stat_skipped++; if (log_skips) skip_note(why, true); } while (0)
static void skip_log(void) {
    if (log_skips <= 0) return;
    uint64_t total = 0;
    for (int i = 0; i < SK_N; i++) total += skip_n[i];
    if (total) {
        LOG("[skips] %llu:", (unsigned long long)total);
        for (int i = 0; i < SK_N; i++) if (skip_n[i]) LOG(" %s %llu", skip_names[i], (unsigned long long)skip_n[i]);
        uint64_t now = SDL_GetTicksNS();
        bool shown[SKIP_TOP] = {0};
        for (int t = 0; t < 4; t++) {  // the top offenders of this interval
            int best = -1;
            for (int i = 0; i < skip_ntop; i++)
                if (skip_top[i].n && !shown[i] && (best < 0 || skip_top[i].n > skip_top[best].n)) best = i;
            if (best < 0) break;
            shown[best] = true;
            bool cs = skip_top[best].why == SK_CS_BUILDING || skip_top[best].why == SK_CS_FAILED;
            LOG("%s %s: %s %llx (%016llx)", t ? "," : " | top", skip_names[skip_top[best].why], cs ? "cs" : "vs",
                (unsigned long long)skip_top[best].a, (unsigned long long)skip_top[best].ha);
            if (!cs) LOG(" fs %llx (%016llx)", (unsigned long long)skip_top[best].b, (unsigned long long)skip_top[best].hb);
            LOG(" x%llu, for %llu s", (unsigned long long)skip_top[best].n,
                (unsigned long long)((now - skip_top[best].first_ns) / 1000000000ull));
        }
        LOG("\n");
    }
    memset(skip_n, 0, sizeof skip_n);
    int k = 0;  // (an offender not skipped this interval is dropped: its "for N s" starts again if it comes back)
    for (int i = 0; i < skip_ntop; i++)
        if (skip_top[i].n) { skip_top[k] = skip_top[i]; skip_top[k++].n = 0; }
    skip_ntop = k;
}

// Builds in flight (programs, graphics / compute pipelines): one still running after 10 s is logged once, with its
// shaders' code addresses. "[shader] slow build" only comes when a build finishes, so one that never does (its draws
// or dispatches skipped for good, compile 0 in the stats) left no trace.
#define BUILDS_MAX 64
static pthread_mutex_t builds_lock = PTHREAD_MUTEX_INITIALIZER;
static struct { const char *what; uint64_t c0, c1, t0; bool logged; } builds[BUILDS_MAX];
static int build_begin(const char *what, uint64_t c0, uint64_t c1) {
    pthread_mutex_lock(&builds_lock);
    int s = 0;
    while (s < BUILDS_MAX && builds[s].what) s++;
    if (s < BUILDS_MAX) { builds[s].what = what; builds[s].c0 = c0; builds[s].c1 = c1; builds[s].t0 = SDL_GetTicksNS(); builds[s].logged = false; }
    pthread_mutex_unlock(&builds_lock);
    return s < BUILDS_MAX ? s : -1;
}
static void build_end(int s) {
    if (s < 0) return;
    pthread_mutex_lock(&builds_lock);
    builds[s].what = NULL;
    pthread_mutex_unlock(&builds_lock);
}
static void builds_check(void) {
    uint64_t now = SDL_GetTicksNS();
    pthread_mutex_lock(&builds_lock);
    for (int s = 0; s < BUILDS_MAX; s++) {
        if (!builds[s].what || builds[s].logged || now - builds[s].t0 < 10000000000ull) continue;
        builds[s].logged = true;
        LOG("[shader] still building after %llu s: %s %llx %llx\n", (unsigned long long)((now - builds[s].t0) / 1000000000ull),
            builds[s].what, (unsigned long long)builds[s].c0, (unsigned long long)builds[s].c1);
    }
    pthread_mutex_unlock(&builds_lock);
}

const char *renderer_stats(void) {
    skip_log();
    builds_check();
    static int dumps;
    if (dumps++ < 3) {
        LOG("[vk] command counts:");
        for (int i = 0; i < CMD_COUNT; i++) if (op_counts[i]) LOG(" %d:%llu", i, (unsigned long long)op_counts[i]);
        LOG("\n");
    }
    snprintf(stats_text, sizeof stats_text,
             "%llu draws, %llu skipped, %llu passes, %llu dispatches, %llu flushes, wait %.0fms, compile %d, fbcopy %llu, upload %.1fMB, "
             "vertices %.0fMB cached %.0fMB host, xfb sessions %llu, scene waits %.0fms, deferred %llu, snap %lluKB/%llu",
             (unsigned long long)stat_draws, (unsigned long long)stat_skipped, (unsigned long long)stat_passes,
             (unsigned long long)stat_dispatches, (unsigned long long)stat_flushes, (double)stat_wait_ns / 1e6,
             atomic_exchange(&stat_compiles, 0), (unsigned long long)stat_feedback_copies, (double)stat_upload_bytes / 1048576.0,
             (double)stat_vtx_cached / 1048576.0, (double)stat_vtx_host / 1048576.0, (unsigned long long)stat_xfb_sessions,
             (double)stat_scene_wait_ns / 1e6, (unsigned long long)__atomic_exchange_n(&stat_deferred, 0, __ATOMIC_RELAXED),
             (unsigned long long)(stat_snap_bytes >> 10), (unsigned long long)stat_snap_copies);  // (overlap: SyncWaits returned early; storage copies made)
    stat_scene_wait_ns = 0;
    stat_feedback_copies = stat_upload_bytes = stat_snap_bytes = stat_snap_copies = 0;
    stat_vtx_cached = stat_vtx_host = stat_xfb_sessions = 0;
    if (ENV("LARECOMP_SKIP_WHY")) {
        LOG("[vk] skipped by line:");
        for (int i = 0; i < 4096; i++) if (skip_why[i]) LOG(" %d:%u", i, skip_why[i]);
        LOG("\n");
    }
    memset(skip_why, 0, sizeof skip_why);
    {
        extern void vk_alias_stats(void);
        vk_alias_stats();
    }
    if (prof) {
        prof_up = prof_up_ns;
        prof_up_ns = 0;
        RecStats rs;  // (the recorder's counters for this interval: record.c)
        rec_stats_take(&rs);
        prof_submit = rs.submit_ns;
        size_t n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | gpu %.0f cpu exec %.0f draw %.0f (pre %.0f key %.0f dyn %.0f sets %.0f tex %.0f (upload %.0f) vtx %.0f) submit %.0f ms",
                 prof_gpu / 1e6, prof_exec / 1e6, prof_draw / 1e6, prof_pre / 1e6, prof_key / 1e6, prof_dyn / 1e6, prof_sets / 1e6,
                 prof_tex / 1e6, prof_up / 1e6, prof_vtx / 1e6, prof_submit / 1e6);
        extern uint64_t prof_tex_checks, prof_tex_samples, prof_tex_full, prof_tex_full_bytes, prof_tex_dyn;
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | tex checks %llu samples %llu full %llu (%.1f MB, dynamic %llu)",
                 (unsigned long long)prof_tex_checks, (unsigned long long)prof_tex_samples, (unsigned long long)prof_tex_full,
                 prof_tex_full_bytes / 1048576.0, (unsigned long long)prof_tex_dyn);
        n = strlen(stats_text);
        snprintf(stats_text + n, sizeof stats_text - n, " | pass ends:");
        for (int l = 0; l < 4096; l++)
            if (prof_end_line[l] > 20) { n = strlen(stats_text); snprintf(stats_text + n, sizeof stats_text - n, " %d:%u", l, prof_end_line[l]); }
        memset(prof_end_line, 0, sizeof prof_end_line);
        n = strlen(stats_text);
        if (gc_shift) {
            snprintf(stats_text + n, sizeof stats_text - n, " | heap chunks %d imported (%llu MB), %llu misses", gc_imported,
                     (unsigned long long)gc_imported_bytes >> 20, (unsigned long long)stat_gc_misses);
            n = strlen(stats_text);
            if (gc_lazy) {
                snprintf(stats_text + n, sizeof stats_text - n, ", pointer scans %llu (%.2f MB), %llu skipped",
                         (unsigned long long)prof_scan_calls, prof_scan_bytes / 1048576.0, (unsigned long long)prof_scan_skipped);
                n = strlen(stats_text);
            }
            prof_scan_calls = prof_scan_bytes = prof_scan_skipped = 0;
        }
        snprintf(stats_text + n, sizeof stats_text - n, " | views created %llu | counters gpu %llu | pace %s, %llu frames over", (unsigned long long)prof_views_created,
                 (unsigned long long)stat_counters_gpu, pace_is_slow() ? "30" : "60", (unsigned long long)pace_frames_over());
        stat_counters_gpu = 0;
        {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n,  " | vcache %llu hit %llu in place %.1f MB copied, %llu queries %.2f ms, %llu pages written",
                     (unsigned long long)prof_vc_hits, (unsigned long long)prof_vc_bypass, prof_vc_copied / 1048576.0,
                     (unsigned long long)prof_vc_queries, prof_vc_scan_ns / 1e6, (unsigned long long)prof_vc_scan_pages);
            prof_vc_hits = prof_vc_bypass = prof_vc_copied = prof_vc_queries = prof_vc_scan_pages = prof_vc_scan_ns = 0;
            extern uint64_t prof_tex_watch_skips, prof_tex_watch_written, prof_tex_watch_unknown, prof_tex_watch_gpu;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | tex unwritten %llu written %llu unknown %llu (GPU pages %llu)",
                     (unsigned long long)prof_tex_watch_skips, (unsigned long long)prof_tex_watch_written,
                     (unsigned long long)prof_tex_watch_unknown, (unsigned long long)prof_tex_watch_gpu);
            prof_tex_watch_skips = prof_tex_watch_written = prof_tex_watch_unknown = prof_tex_watch_gpu = 0;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | const blocks %s: %llu reused %llu copied", cbuf_lean() ? "lean" : "full",
                     (unsigned long long)prof_cb_reused, (unsigned long long)prof_cb_copied);
            prof_cb_reused = prof_cb_copied = 0;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | hook pass %llu draws, %llu with known textures",
                     (unsigned long long)prof_hook_draws, (unsigned long long)prof_hook_known);
            prof_hook_draws = prof_hook_known = 0;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | key memo %s: program %llu pipeline %llu of %llu", key_memo_on() ? "on" : "off",
                     (unsigned long long)prof_prog_memo, (unsigned long long)prof_pipe_memo, (unsigned long long)prof_key_draws);
            prof_prog_memo = prof_pipe_memo = prof_key_draws = 0;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | sets (reuse %s, batch %s, recycle %s): %llu reused %llu written, %llu allocated"
                     " (%llu new from the pools in %llu calls), pool resets %llu %.2f ms%s",
                     set_reuse_on() ? "on" : "off", rec_set_batch() ? "on" : "off", rec_set_recycle() ? "on" : "off",
                     (unsigned long long)prof_sets_reused, (unsigned long long)prof_sets_new, (unsigned long long)rs.set_allocs,
                     (unsigned long long)rs.sets_fresh, (unsigned long long)rs.set_calls, (unsigned long long)rs.pool_resets,
                     rs.pool_reset_ns / 1e6, rec_on() ? " (on the recorder)" : "");
            prof_sets_reused = prof_sets_new = 0;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | slot wait %.1f ms", prof_slot_wait / 1e6);
            prof_slot_wait = 0;
            n = strlen(stats_text);
            // the recorder thread (record.c): packets queued, the translating thread's time queuing them, the recorder's
            // time making the calls, and the waits on it (drain: ImGui / idle waits; seq: GPU waits that had to wait for a
            // submission to be handed to the driver first; present: the next frame's acquire waiting for the last present;
            // full: the ring full)
            snprintf(stats_text + n, sizeof stats_text - n, " | recorder %s: %llu packets (%.1f MB) enqueue %.1f ms, busy %.1f ms (%llu wakes), waits:"
                     " drain %llux %.1f ms, seq %llux %.1f ms, present %llux %.1f ms, full %llux %.1f ms, pipes %llux %.1f ms, dropped %llu",
                     rec_on() ? "on" : "off", (unsigned long long)rs.packets, rs.bytes / 1048576.0, rs.enqueue_ns / 1e6, rs.busy_ns / 1e6,
                     (unsigned long long)rs.wakeups, (unsigned long long)rs.drains, rs.drain_ns / 1e6, (unsigned long long)rs.seq_waits,
                     rs.seq_wait_ns / 1e6, (unsigned long long)rs.present_waits, rs.present_wait_ns / 1e6, (unsigned long long)rs.full_waits,
                     rs.full_ns / 1e6, (unsigned long long)rs.pipe_waits, rs.pipe_wait_ns / 1e6, (unsigned long long)rs.draws_dropped);
        }
        {   // GPU time per frame by part (see gpu_timer_add), CPU waits per frame by place (device.c g_vk_frame_waits)
            static uint64_t lw[4];
            uint64_t dw[4];
            for (int i = 0; i < 4; i++) {  // ([2], [3]: the recorder's)
                uint64_t v = __atomic_load_n(&g_vk_frame_waits[i], __ATOMIC_RELAXED);
                dw[i] = v - lw[i], lw[i] = v;
            }
            double nf = prof_wframes ? (double)prof_wframes : 1, np = dw[3] ? (double)dw[3] : 1;
            double gap = ((double)prof_span - (double)prof_gpu - (double)prof_win) / nf / 1e6;
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, " | gpu/frame game %.1f window %.1f span %.1f gap %.1f ms (%llu frames)"
                     " | cpu wait/frame slot %.1f acquire %.1f present %.1f ms",
                     prof_gpu / nf / 1e6, prof_win / nf / 1e6, prof_span / nf / 1e6, prof_wframes ? gap : 0.0,
                     (unsigned long long)prof_wframes, dw[0] / np / 1e6, dw[1] / np / 1e6, dw[2] / np / 1e6);
            prof_win = prof_span = prof_wframes = 0;
        }
        pthread_mutex_lock(&wait_tab_lock);
        for (int i = 0; i < 12 && wait_tab[i].why; i++) {
            n = strlen(stats_text);
            snprintf(stats_text + n, sizeof stats_text - n, "%s%s %llux %.0f ms", i ? "," : " | waits:", wait_tab[i].why,
                     (unsigned long long)wait_tab[i].n, wait_tab[i].ns / 1e6);
        }
        memset(wait_tab, 0, sizeof wait_tab);
        pthread_mutex_unlock(&wait_tab_lock);
        prof_views_created = 0;
        prof_tex_checks = prof_tex_samples = prof_tex_full = prof_tex_full_bytes = prof_tex_dyn = 0;
        prof_exec = prof_draw = prof_sets = prof_vtx = prof_submit = prof_pre = prof_key = prof_dyn = prof_tex = prof_up = prof_gpu = 0;
    }
    stat_draws = stat_skipped = stat_passes = stat_dispatches = stat_flushes = stat_wait_ns = 0;
    return stats_text;
}

static void st_defaults(void) {
    memset(&st, 0, sizeof st);
    for (int i = 0; i < 16; i++) {
        st.viewport[i][2] = 1280; st.viewport[i][3] = 720;
        st.scissor[i][2] = 16384; st.scissor[i][3] = 16384;
        st.depth_range[i][1] = 1.0f;
    }
    st.channel_mask = 0xFFFFFFFF;
    st.stencil_mask[0] = st.stencil_mask[1] = 0xFF;
    st.stencil_vmask[0] = st.stencil_vmask[1] = 0xFF;
    st.ds[1] = 1; st.ds[2] = 2;
    st.poly[1] = 1; st.poly[2] = 2;
    for (int i = 0; i < 8; i++) { st.blend[i][1] = 2; st.blend[i][2] = 1; st.blend[i][3] = 2; st.blend[i][4] = 1; st.blend[i][5] = 1; st.blend[i][6] = 1; }
}

static VkPipeline pipe_resolve(void *obj, bool wait);
void vk_exec_init(void) {
    maxwell_disk_cache = true;  // (translate.c: the game's translations kept in cache/maxwell)
    // The straight-line shader form (translate.c glsl_flatten, v0.1.10) off on AMD GPUs: an RX 7600 and an RX 9070 XT
    // (AMD 26.10.1) both lost the device on v0.1.10 at the first frames in the world, "read invalid 0xbeefdeadc000"
    // (#54, #39), where v0.1.9 played; and RADV crashed in the driver with it on the whole-window import (#51).
    // LARECOMP_SHADER_FLAT=1: on, =0: off everywhere
    // (F1 > Compatibility: On / Off over the per-GPU default)
    int fcs = g_settings.gpu_compat[GPU_STRAIGHT];
    maxwell_flat_default = fcs != GPU_AUTO ? fcs == GPU_ON : g_vk.props.vendorID != 0x1002;
    if (!getenv("LARECOMP_SHADER_FLAT") && !maxwell_flat_default)
        LOG("[vk] straight-line shader form off (%s; LARECOMP_SHADER_FLAT=1: on)\n",
            fcs == GPU_OFF ? "F1 > Compatibility" : "this AMD GPU, F1 > Compatibility to turn it on");
    // The gather bias (translate.c gather_bias): AMD's and Intel's Windows drivers round a gather coordinate that sits
    // between two texels unlike Maxwell, as Ryujinx's Vulkan back end (MIT) biases for (gatherBiasPrecision: the
    // sub-texel precision bits on IsIntelWindows / IsAmdWindows) and KarPur for Violet on AMD (its GATHER_BIAS) -
    // suspected for the white specks along object edges on an Intel Arc B580. NVIDIA rounds as Maxwell does, and
    // Mesa's drivers aren't in Ryujinx's list. LARECOMP_GATHER_BIAS=<bits> forces it (0: off).
    {
        const char *e = getenv("LARECOMP_GATHER_BIAS");
        uint32_t bits = 0;
#ifdef _WIN32
        uint32_t vendor = g_vk.props.vendorID;
        if (vendor == 0x8086 || vendor == 0x1002) bits = g_vk.props.limits.subTexelPrecisionBits;
#endif
        if (e) bits = (uint32_t)atoi(e);
        if (bits > 16) bits = 16;
        maxwell_gather_bias_bits = bits;
        if (bits) LOG("[vk] gather coordinates biased for %u sub-texel bits (%s)\n", bits,
                      e ? "LARECOMP_GATHER_BIAS" : "this AMD / Intel GPU; LARECOMP_GATHER_BIAS=0: off");
        else LOG("[vk] gather coordinates as written%s\n", e ? " (LARECOMP_GATHER_BIAS=0)" : "");
    }
    prof = getenv("LARECOMP_VK_PROF") != NULL;
    if (prof) prof_up_ns = 0;
    // the GPU timer (LARECOMP_VK_GPU_TIMER=0: off - dynamic resolution then goes by the frame times)
    const char *gt = getenv("LARECOMP_VK_GPU_TIMER");
    if (g_vk.ts_bits && g_vk.props.limits.timestampComputeAndGraphics && g_vk.props.limits.timestampPeriod > 0 && !(gt && !atoi(gt))) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = SUBS * 3};
        if (vkCreateQueryPool(g_vk.dev, &qi, NULL, &prof_qp) != VK_SUCCESS) prof_qp = VK_NULL_HANDLE;
    }
    gpu_timer_on = prof_qp != VK_NULL_HANDLE;
    LOG("[vk] GPU timer %s (dynamic resolution goes by %s)\n", gpu_timer_on ? "on" : "off", gpu_timer_on ? "GPU time" : "frame times");
    st_defaults();
    // The heap's chunk / lazy decision now, on this thread, before the render thread and the build workers exist:
    // they read gc_shift / gc_lazy directly (a buffer copy can come before any shader was translated)
    gmem_chunk_shift();
    // The command buffers of every slot, from the pool, before the recorder thread exists: from then on the pool and
    // the buffers are the recorder's (begun, recorded, ended and submitted in packet order)
    for (int i = 0; i < SUBS; i++) {
        VkCommandBuffer cbs[2];
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_vk.pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2};
        VK_CHECK(vkAllocateCommandBuffers(g_vk.dev, &ai, cbs));
        subs[i].main = cbs[0];
        subs[i].upload = cbs[1];
    }
    rec_set_pipe_resolver(pipe_resolve);
    rec_init(prof != 0);
}

void renderer_texture_scale(NvnTexture *t, float *sx, float *sy);
void renderer_texture_fraction(NvnTexture *t, float *fx, float *fy);  // (textures.c: the part the contents occupy)
bool renderer_texture_presented(NvnTexture *t);
bool renderer_texture_first_write(NvnTexture *t);
bool renderer_texture_level_fresh(NvnTexture *t);
bool renderer_texture_frame_fresh(NvnTexture *t);
void renderer_texture_mark_presented(NvnTexture *t);
uint64_t renderer_present_count(void);
void renderer_save_texture_tex(TexBackend *b, const char *path);
// The pass's attachments, size and scale (begin_pass). Per thread: an async run's workers draw with their pass's while
// the render thread begins the next ones (sw_async_*: pass_tls is the job's copy).
typedef struct { float sx, sy; VkFormat color[8], depth, stencil; uint32_t ncolor, samples, w, h, layers, movie_bar_w, movie_bar_h; bool has_non_movie_draw; } PassG;
static PassG pass_main = {.sx = 1, .sy = 1, .samples = 1, .layers = 1};
static __thread PassG *pass_tls;
static inline PassG *pass_cur(void) { PassG *p = pass_tls; return p ? p : &pass_main; }
#define pass_sx (pass_cur()->sx)
#define pass_sy (pass_cur()->sy)
#define pass_color (pass_cur()->color)
#define pass_depth (pass_cur()->depth)
#define pass_stencil (pass_cur()->stencil)
#define pass_ncolor (pass_cur()->ncolor)
#define pass_samples (pass_cur()->samples)
#define pass_w (pass_cur()->w)
#define pass_h (pass_cur()->h)
#define pass_layers (pass_cur()->layers)
static struct { float sx, sy, ox, oy; } vp_t[NCTX] = {{1, 1, 0, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}, {1, 1, 0, 0},
                                                       {1, 1, 0, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}, {1, 1, 0, 0}};
#define vp_sx (vp_t[sw_ctx()].sx)
#define vp_sy (vp_t[sw_ctx()].sy)
#define vp_ox (vp_t[sw_ctx()].ox)
#define vp_oy (vp_t[sw_ctx()].oy)
static float tex_sc_t[NCTX][2][64][2];
static float tex_fr_t[NCTX][2][64][2];  // per bound texture slot: the part of the texture the contents occupy (Dynamic)
static uint32_t tex_wh_t[NCTX][2][64][2];
#define tex_sc (tex_sc_t[sw_ctx()])
#define tex_fr (tex_fr_t[sw_ctx()])
#define tex_wh (tex_wh_t[sw_ctx()])

// ================================================================== render passes
static int debug_pass_no;
// (draw_indirect) the draw being made reads its parameters on the GPU: offset in the guest memory window's buffer
static bool g_ind_gpu_t[NCTX];
static VkDeviceSize g_ind_off_t[NCTX];
static uint32_t g_ind_stride_t[NCTX];
#define g_ind_gpu (g_ind_gpu_t[sw_ctx()])
#define g_ind_off (g_ind_off_t[sw_ctx()])
#define g_ind_stride (g_ind_stride_t[sw_ctx()])
// Transform feedback state (Begin/EndTransformFeedback, the bound buffers) and rasterizer discard
static bool xfb_on;
#define raster_discard (st.raster_discard)
static uint64_t xfb_addr[4], xfb_size[4];
static uint32_t raw_layer;  // (debug frame: the layer of a layered pass that renderer_save_texture_tex saves)

// A draw's vertex layout. Without dynamic vertex input it is copied into the PipeKey (each layout its own pipeline);
// with it (g_vk.vids) the key leaves it out and draw_ sets it (vkCmdSetVertexInputEXT) when it changes.
typedef struct {
    uint32_t attr_mask, used_streams;
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32];
    uint32_t stride[17], divisor[17];  // divisor 0: per vertex; else per instance
} VtxLayout;

// Redundant-state filter: what draw() last set in the current command buffer, so unchanged pipeline binds, dynamic
// state and vertex / index buffer binds are not recorded again (a draw used to issue ~20 vkCmd* calls; most repeat the
// previous draw's). Forgotten (ss_forget) whenever something else may have recorded state: a new command buffer, a new
// render pass (present, ImGui, post-processing and copies all run between game passes) and the clear pipeline.
// Descriptor sets are always bound (sharing identical sets within a submission was tried: a third matched, but
// comparing them cost more than allocating and writing them on NVIDIA; with dynamic uniform buffers most sets repeat
// their program's previous one, which stage_set keeps: "set reuse").
// LARECOMP_VK_STATE_FILTER=0: every draw sets everything, as before.
enum { SS_PIPE = 1 << 0, SS_DT = 1 << 1, SS_DW = 1 << 2, SS_DOP = 1 << 3, SS_STEN = 1 << 4, SS_SFACE0 = 1 << 5, SS_SFACE1 = 1 << 6,
       SS_BC = 1 << 7, SS_CULL = 1 << 8, SS_FF = 1 << 9, SS_TOPO = 1 << 10, SS_PRE = 1 << 11, SS_DBE = 1 << 12, SS_DB = 1 << 13,
       SS_VP = 1 << 14, SS_SC = 1 << 15, SS_IB = 1 << 16, SS_VI = 1 << 17 };
typedef struct SsState {
    uint32_t known;  // SS_* bits: the value below is what the command buffer has
    uint32_t vb_known;  // per vertex binding
    const void *pipe;   // the Pipe bound (a building one has no handle yet: WAIT_PIPES)
    VkBool32 dt, dw, sten, pre, dbe;
    VkCompareOp dop;
    uint32_t sface[2][7];  // fail, pass, depth fail, compare, compare mask, write mask, reference
    float bc[4], db[3];
    VkCullModeFlags cull;
    VkFrontFace ff;
    VkPrimitiveTopology topo;
    VkViewport vp;
    VkRect2D sc;
    VkBuffer ib; VkDeviceSize io; VkIndexType it;
    VkBuffer vb[17]; VkDeviceSize vo[17];
    VtxLayout vi;  // (dynamic vertex input)
} SsState;
// (per command buffer: a shadow worker's secondary has its own - sw_chunk)
static SsState ss_main;
static __thread SsState *ss_tls;
static inline SsState *ss_cur(void) { SsState *p = ss_tls; return p ? p : &ss_main; }
// (hot functions) the calling thread's state pointers read once a call: every st / ss / pass_* / per-context access read
// the thread's pointer again (thread-local storage: dependent loads), ~1.2 ms a frame of the render thread at World
// detail 4x. A function opens with TLS_LOCALS() and the macros below, and closes with TLS_LOCALS_END (its st_tls etc.
// don't change during the call: sw_chunk sets and restores them around a part, a call of its own)
#define TLS_LOCALS() TState *const st_l_ = st_cur(); SsState *const ss_l_ = ss_cur(); PassG *const pass_l_ = pass_cur(); \
    const int ctx_l_ = sw_ctx(); (void)st_l_; (void)ss_l_; (void)pass_l_; (void)ctx_l_
#define ss (*ss_cur())
static int ss_on = -1;
static void ss_forget(void) { ss.known = 0; ss.vb_known = 0; }
static bool ss_enabled(void) {
    if (ss_on < 0) { const char *e = ENV("LARECOMP_VK_STATE_FILTER"); ss_on = !e || atoi(e) != 0; }
    return ss_on;
}
// true: the state must be recorded (and is remembered as known)
static inline bool ss_need(uint32_t bit, bool same) {
    if (ss_on && (ss.known & bit) && same) return false;
    ss.known |= bit;
    return true;
}

static void vk_end_pass_at(int line);
void vk_end_pass(void) { vk_end_pass_at(0); }
#define vk_end_pass() vk_end_pass_at(__LINE__)
static void vk_end_pass_at(int line) {
    if (sw_cb) {  // (a shadow worker can't end the render thread's pass: the draw goes back to it - sw_run)
        if (sw_jmp) longjmp(*sw_jmp, 1);
        static int n;
        if (n++ < 10) LOG("[sw] a shadow worker asked to end the pass outside a draw (line %d): ignored\n", line);
        return;
    }
    if (!in_pass) return;
    if (prof && line < 4096) prof_end_line[line]++;
    rec_CmdEndRendering(vk_cmd());
    in_pass = false;
    if (g_debug_frame && !ENV("LARECOMP_DEBUG_NOPNG") && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->backend) {
        char path[256];
        static uint64_t debug_pass_frame = ~0ull;
        static int multi = -1;
        if (multi < 0) multi = ENV("LARECOMP_DEBUG_FRAMES") != NULL;
        if (multi && renderer_present_count() != debug_pass_frame) { debug_pass_frame = renderer_present_count(); debug_pass_no = 0; }
        if (multi) snprintf(path, sizeof path, "build/pass_%llu_%03d.png", (unsigned long long)debug_pass_frame, debug_pass_no++);
        else snprintf(path, sizeof path, "build/pass_%03d.png", debug_pass_no++);
        renderer_save_texture_tex(st.rt.color[0]->backend, path);
        for (int ci = 1; ci < st.rt.ncolor; ci++) {  // (multiple render targets: the others as _c<i>)
            if (!st.rt.color[ci] || !st.rt.color[ci]->backend) continue;
            char cpath[300];
            snprintf(cpath, sizeof cpath, "%.*s_c%d.png", (int)strlen(path) - 4, path, ci);
            renderer_save_texture_tex(st.rt.color[ci]->backend, cpath);
        }
        TexBackend *tb = st.rt.color[0]->backend;
        if (tb->width == 1 && tb->height == 1 && tb->format == VK_FORMAT_R32G32B32A32_SFLOAT) {  // (exposure: the raw value)
            VkSpan out = vk_readback(16);
            VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {1, 1, 1}};
            VkCommandBuffer cb = vk_cmd();
            vk_full_barrier(cb);
            rec_CmdCopyImageToBuffer(cb, tb->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
            renderer_flush();
            const float *f = out.ptr;
            LOG("[vk] %s 1x1 = %g %g %g %g\n", path, f[0], f[1], f[2], f[3]);
        }
        uint32_t lys[3] = {1, pass_layers / 2, pass_layers - 1};  // layered: also layers 1, middle, last
        for (int k = 0; k < 3 && pass_layers > 1; k++) {
            uint32_t ly = lys[k];
            if (k && ly <= lys[k - 1]) continue;
            char lpath[300];
            snprintf(lpath, sizeof lpath, "%.*s_l%u.png", (int)strlen(path) - 4, path, ly);
            raw_layer = ly;
            renderer_save_texture_tex(st.rt.color[0]->backend, lpath);
            raw_layer = 0;
        }
    }
}

static uint32_t level_extent(uint32_t v, uint32_t lv) { v >>= lv; return v ? v : 1; }
// The part of a level extent `e` a scalable image's contents occupy (Dynamic resolution: drawn at scale `sc` into an
// image allocated at scale `al`, the top-left part); the whole extent otherwise.
static int content_extent(int e, bool scalable, float sc, float al) {
    if (!scalable || sc == al || al <= 0) return e;
    int c = (int)lroundf(e * sc / al);
    return c < 1 ? 1 : c > e ? e : c;
}

static bool heap_whole_userptr(void);
static void gtao_apply_pending(void);
static void water_apply_pending(void);
static void gtao_forget(NvnTexture *t);  // (vk_exec_texture_destroyed: the AO and water passes' saved targets)
static bool pass_secondary;  // (sw_run) the next pass's contents are secondary command buffers
static bool sw_enabled(void);
static bool pass_mixed;  // (maintenance7, shadow workers on) the pass takes both: runs need no restart
static void begin_pass_(void);
static void planar_replay_pending(void);
static void planar_light_pending(void);
static void begin_pass(void) {
    if (sw_cb) return;  // (a shadow worker draws inside the render thread's pass)
    if (!sw_rt_lock) { begin_pass_(); return; }
    SWL();  // (async runs in flight: the views, textures and tables it uses are the workers' too)
    begin_pass_();
    SWU();
}
static void begin_pass_(void) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    if (in_pass && !st.rt_dirty) return;
    vk_end_pass();
    planar_replay_pending();  // (the G-buffer pass just ended: its draws again, mirrored - before a submission can come)
    planar_light_pending();   // (the lighting pass just ended: its draw again, on them)
    gtao_apply_pending();  // (the lighting pass just ended: its image times the ambient occlusion)
    water_apply_pending();  // (the water drawn: its reflections)
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each pass its own submission)
    else {
        // Submit the frame in chunks (as Metal, renderer.m begin_pass): the game puts a whole frame in one command
        // list and waits for its fence early in the next frame, so with one submission per list the GPU sat idle
        // while the list was translated, and the game's wait covered translation + GPU time - with the render thread
        // the translation left the game thread but came back as that wait. LARECOMP_CHUNK_PASSES=<n>: a submission
        // every n render passes (0 = one per list). Default 8 with the render thread (RTX 3080 Ti + i9, Field_11C:
        // gpu wait 0.6 -> 0.25 ms, game thread busy 3.4-3.5 ms vs 4.9 without the render thread); without it,
        // off (chunks cost that thread 0.1-0.3 ms there). Also off by default while the whole heap is one userptr
        // import (heap_whole_userptr): amdgpu checks all of its pages again on every submission, ~5 ms each for the
        // 3.2 GB heap, so ten chunks a frame cost ~50 ms (issue #92: RX 6800 XT, 12 fps at the title screen, where
        // 0c26936 - one submission per list - got 29-41).
        static int chunk = -1;
        static bool chunk_env;
        if (chunk < 0) {
            extern bool rq_on(void);
            const char *e = getenv("LARECOMP_CHUNK_PASSES");
            chunk_env = e != NULL;
            // (32, not Z-A's 8: Scarlet / Violet's town frames have ~550 passes, most of them transform feedback
            // sessions' - 70 submissions a frame, each with its vertex cache write-tracking queries and a ring slot
            // the GPU often still held: Mesagoza at World detail 4x 31.6 -> 36.5 fps with 32; 96 and 0 the same)
            chunk = e ? atoi(e) : rq_on() ? 32 : 0;
        }
        int c = chunk;
        if (c > 0 && !chunk_env && heap_whole_userptr()) {  // (decided at boot: asked here, not cached before it)
            static bool logged;
            if (!logged) { logged = true; LOG("[vk] submission chunks off: the guest heap is one userptr import (AMD)\n"); }
            c = 0;
        }
        if (c > 0 && ++passes_in_sub >= c && !sw_ainflight) renderer_submit();  // (async runs in flight: after them)
    }
    st.rt_dirty = false;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);  // passes, copies and dispatches see each other's results
    {
        NvnTexture *t0n = st.rt.ncolor ? st.rt.color[0] : NULL;
        dl_note("pass %d color %ux%u fmt %#x depth %p", st.rt.ncolor, t0n ? t0n->width : 0, t0n ? t0n->height : 0, t0n ? t0n->format : 0,
                (void *)st.rt.depth);
    }
    if (g_gpuprof_frame) {
        NvnTexture *t0 = st.rt.ncolor ? st.rt.color[0] : NULL;
        // (color: the game's size for the target; drawn: what is rendered - the render scale applied, as the pass's area)
        const TexBackend *tb = t0 ? t0->backend : NULL;
        float dsx = tb && tb->scalable ? tb->sx : 1, dsy = tb && tb->scalable ? tb->sy : 1;
        gp_mark(cb, "pass: %d color %ux%u (drawn %.0fx%.0f) fmt %#x, depth %s", st.rt.ncolor, t0 ? t0->width : 0,
                t0 ? t0->height : 0, t0 ? t0->width * dsx : 0, t0 ? t0->height * dsy : 0, t0 ? t0->format : 0,
                st.rt.depth ? "yes" : "no");
    }
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    uint32_t w = 0, minw = UINT32_MAX, minh = UINT32_MAX;  // w: first attachment seen
    uint32_t layers = UINT32_MAX;  // layered rendering: the fewest layers of any attachment
    pass_ncolor = (uint32_t)st.rt.ncolor;
    pass_samples = 1;
    for (int i = 0; i < st.rt.ncolor; i++) {
        NvnTexture *t = st.rt.color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        pass_color[i] = VK_FORMAT_UNDEFINED;
        if (!t) continue;
        renderer_mark_gpu_owned(t);
        uint32_t nly;
        VkImageView v = vk_render_view(t, &st.rt.cview[i], st.rt.has_cview[i], &nly);
        TexBackend *b = t->backend;
        if (!v) continue;
        if (nly < layers) layers = nly;
        ca[i].imageView = v;
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        bool fresh = (int)renderer_texture_level_fresh(t) | (int)renderer_texture_frame_fresh(t);  // both: each marks its state
        if (renderer_texture_first_write(t) || fresh) {
            ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            ca[i].clearValue.color.float32[3] = fresh ? 0.0f : 1.0f;
        }
        pass_color[i] = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4) && vk_format(t->format) == b->format
                      ? vk_format(st.rt.cview[i].format) : b->format;
        pass_color[i] = vk_view_format(b, pass_color[i]);  // (g_vk.lean_usage: the format the view really has)
        uint32_t lv = st.rt.has_cview[i] && (st.rt.cview[i].setmask & 1) ? st.rt.cview[i].base_level : 0;
        uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
        if (!w) { renderer_texture_scale(t, &pass_sx, &pass_sy); w = ew; }
        if (ew < minw) minw = ew;
        if (eh < minh) minh = eh;
        pass_samples = b->samples;
    }
    // Bound color slots without a texture / view stay as VK_NULL_HANDLE attachments (legal), but Intel's Windows
    // driver for UHD 630 (Vulkan 1.3.215) crashes in vkCmdBeginRendering reading near a null pointer on the title
    // screen's first clear (issues #8, #46). Empty slots at the end are left out (pipelines and clears follow
    // pass_ncolor); one in the middle, or a pass with no attachment at all, is logged once for the next report.
    while (pass_ncolor && !ca[pass_ncolor - 1].imageView) pass_ncolor--;
    static bool logged_hole;
    for (uint32_t i = 0; i < pass_ncolor && !logged_hole; i++)
        if (!ca[i].imageView) {
            logged_hole = true;
            LOG("[vk] pass with an empty color slot %u of %u (texture %s)\n", i, pass_ncolor, st.rt.color[i] ? "without a view" : "none");
        }
    pass_depth = pass_stencil = VK_FORMAT_UNDEFINED;
    bool has_d = false, has_s = false;
    if (st.rt.depth) {
        renderer_mark_gpu_owned(st.rt.depth);
        uint32_t nly;
        VkImageView v = vk_render_view(st.rt.depth, &st.rt.dview, st.rt.has_dview, &nly);
        TexBackend *b = st.rt.depth->backend;
        if (v && renderer_format_is_depth(st.rt.depth->format)) {
            if (nly < layers) layers = nly;
            VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v,
                                           .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                           .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            if (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da = a; has_d = true; pass_depth = b->format; }
            if (b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa = a; has_s = true; pass_stencil = b->format; }
            uint32_t lv = st.rt.has_dview && (st.rt.dview.setmask & 1) ? st.rt.dview.base_level : 0;
            uint32_t ew = level_extent(b->width, lv), eh = level_extent(b->height, lv);
            if (!w) { renderer_texture_scale(st.rt.depth, &pass_sx, &pass_sy); w = ew; }
            if (ew < minw) minw = ew;
            if (eh < minh) minh = eh;
            if (!pass_ncolor) pass_samples = b->samples;
        }
    }
    if (!w) {
        pass_sx = pass_sy = 1; minw = minh = 16;
        static bool logged_none;
        if (!logged_none) logged_none = true, LOG("[vk] pass with no attachments (%d color slots bound)\n", st.rt.ncolor);
    }
    pass_cur()->movie_bar_w = 0;
    pass_cur()->movie_bar_h = 0;
    pass_cur()->has_non_movie_draw = false;
    // A pass with depth / stencil but no color attachment gets a throwaway one on Intel (LARECOMP_VK_PLACEHOLDER_COLOR=0 /
    // 1 overrides): the UHD 630's Windows driver (Vulkan 1.3.215) crashes in the vkCmdBeginRendering after such a pass,
    // reading near a null pointer (issues #8, #46). Pipelines and clears follow pass_ncolor / pass_color as usual.
    static int placeholder = -1;
    if (placeholder < 0) {
        const char *e = getenv("LARECOMP_VK_PLACEHOLDER_COLOR");
        placeholder = e ? atoi(e) != 0 : g_vk.props.vendorID == 0x8086;
    }
    bool placed = false;
    // The same driver on an Iris Xe (Vulkan 1.3.212) crashed there on an empty slot before a bound one ("empty color
    // slot 0 of 2", #88): on Intel such a hole gets a throwaway attachment too (R8, which the pipelines then take for
    // that slot: whatever the shader writes there is dropped), and so does a pass with no attachment at all
    if (placeholder && (layers == UINT32_MAX || layers == 1)) {  // (the placeholder has one layer: not in a layered pass)
        for (uint32_t i = 0; i < pass_ncolor; i++) {
            if (ca[i].imageView) continue;
            VkImageView v = vk_dummy_color_view(minw, minh, pass_samples);
            if (!v) break;
            ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v,
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE};
            pass_color[i] = VK_FORMAT_R8_UNORM;
            placed = true;
        }
    }
    if (placeholder && !pass_ncolor) {
        VkImageView v = vk_dummy_color_view(minw, minh, pass_samples);
        if (v) {
            ca[0] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v,
                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE};
            pass_color[0] = VK_FORMAT_R8_UNORM;
            pass_ncolor = 1;
            placed = true;
        }
    }
    pass_w = minw; pass_h = minh;
    pass_layers = layers == UINT32_MAX ? 1 : layers;
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {minw, minh}}, .layerCount = pass_layers,
                          .flags = pass_secondary ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT | (g_vk.maint7 ? VK_RENDERING_CONTENTS_INLINE_BIT_KHR : 0) : 0,
                          .colorAttachmentCount = pass_ncolor, .pColorAttachments = pass_ncolor ? ca : NULL,
                          .pDepthAttachment = has_d ? &da : NULL, .pStencilAttachment = has_s ? &sa : NULL};
    // LARECOMP_VK_LOG_PASSES=<n> (default 300 with =1): each of the first n passes before it begins - an Intel crash
    // inside vkCmdBeginRendering (issues #8, #46) ends the log on the pass the driver choked on
    static long log_passes = -1;
    if (log_passes < 0) {
        log_passes = getenv("LARECOMP_VK_LOG_PASSES") ? atol(getenv("LARECOMP_VK_LOG_PASSES")) : 0;
        if (log_passes == 1) log_passes = 300;
    }
    if (log_passes > 0) {
        log_passes--;
        LOG("[vk] pass %ux%u: %u color", minw, minh, pass_ncolor);
        for (uint32_t i = 0; i < pass_ncolor; i++) {
            TexBackend *b = !placed && st.rt.color[i] ? st.rt.color[i]->backend : NULL;
            LOG(" [%u: fmt %d view %s %s", i, pass_color[i], ca[i].imageView ? "yes" : "none", placed ? "placeholder"
                : ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? "clear" : "load");
            // (the image under it: its own format, size, levels / layers, samples; the view's first level and layer)
            if (b) LOG(", image fmt %d %ux%u lv %u ly %u s %u, view lv %u ly %u", b->format, b->width, b->height, b->levels,
                       b->layers, b->samples, st.rt.has_cview[i] && (st.rt.cview[i].setmask & 1) ? st.rt.cview[i].base_level : 0,
                       st.rt.has_cview[i] && (st.rt.cview[i].setmask & 2) ? st.rt.cview[i].min_layer : 0);
            LOG("]");
        }
        if (has_d || has_s)
            LOG(", depth %s fmt %d, stencil %s fmt %d (same view %s, layout %d, load op %d / %d)", has_d ? "yes" : "no",
                pass_depth, has_s ? "yes" : "no", pass_stencil, has_d && has_s && da.imageView == sa.imageView ? "yes" : "no",
                has_d ? da.imageLayout : sa.imageLayout, has_d ? da.loadOp : -1, has_s ? sa.loadOp : -1);
        else LOG(", no depth / stencil");
        LOG(", samples %u\n", pass_samples);
    }
    if (!pass_secondary && g_vk.maint7 && sw_enabled())
        ri.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT | VK_RENDERING_CONTENTS_INLINE_BIT_KHR;
    pass_mixed = (ri.flags & VK_RENDERING_CONTENTS_INLINE_BIT_KHR) != 0;
    rec_CmdBeginRendering(cb, &ri);
    ss_forget();  // (whatever ran between the passes may have set its own state)
    in_pass = true;
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    stat_passes++;
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

static void pass_size(int *w, int *h) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    *w = *h = 0;
    for (int i = 0; i < st.rt.ncolor && !*w; i++)
        if (st.rt.color[i]) { *w = (int)st.rt.color[i]->width; *h = (int)st.rt.color[i]->height; }
    if (!*w && st.rt.depth) { *w = (int)st.rt.depth->width; *h = (int)st.rt.depth->height; }
    *w = (int)lroundf(*w * pass_sx);
    *h = (int)lroundf(*h * pass_sy);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

static VkRect2D scissor_rect(void) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    int w, h;
    pass_size(&w, &h);
    if (!w) return (VkRect2D){{0, 0}, {pass_w, pass_h}};
    int x = st.scissor[0][0], y = st.scissor[0][1], sw = st.scissor[0][2], sh = st.scissor[0][3];
    if (vp_sx != 1 || vp_sy != 1 || vp_ox != 0 || vp_oy != 0) {
        double x0 = floor(x * (double)vp_sx + vp_ox), y0 = floor(y * (double)vp_sy + vp_oy);
        double x1 = ceil(((double)x + sw) * vp_sx + vp_ox), y1 = ceil(((double)y + sh) * vp_sy + vp_oy);
        if (sw >= 16384) x1 = 1e9;
        if (sh >= 16384) y1 = 1e9;
        if (x1 > 1e9) x1 = 1e9;
        if (y1 > 1e9) y1 = 1e9;
        x = (int)x0; y = (int)y0; sw = (int)(x1 - x0); sh = (int)(y1 - y0);
    }
    if (x < 0) { sw += x; x = 0; }
    if (y < 0) { sh += y; y = 0; }
    if (w > (int)pass_w) w = (int)pass_w;
    if (h > (int)pass_h) h = (int)pass_h;
    if (x >= w || y >= h) { x = y = 0; sw = sh = 0; }
    if (x + sw > w) sw = w - x;
    if (y + sh > h) sh = h - y;
    if (sw < 0) sw = 0;
    if (sh < 0) sh = 0;
    return (VkRect2D){{x, y}, {(uint32_t)sw, (uint32_t)sh}};
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

static void apply_scissor(void) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    VkRect2D r = scissor_rect();
    if (!ss_need(SS_SC, !memcmp(&r, &ss.sc, sizeof r))) return;
    ss.sc = r;
    rec_CmdSetScissor(vk_cmd(), 0, 1, &r);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// Metal convention (y up in NDC, origin top-left) through a negative-height viewport.
static void set_viewport(float x, float y, float w, float h, float zn, float zf) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    if (w <= 0) w = 1;
    if (h == 0) h = 1;
    VkViewport vp = {x, y + h, w, -h, fminf(fmaxf(zn, 0), 1), fminf(fmaxf(zf, 0), 1)};
    if (!ss_need(SS_VP, !memcmp(&vp, &ss.vp, sizeof vp))) return;
    ss.vp = vp;
    rec_CmdSetViewport(vk_cmd(), 0, 1, &vp);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// ================================================================== clears
static VkPipelineLayout pc_layout(void) {  // push constants only (clears)
    static VkPipelineLayout l;
    if (!l) {
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                         .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    }
    return l;
}

static const VkDynamicState DYN[] = {
    VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
    VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
#define NDYN (sizeof DYN / sizeof DYN[0])
// The dynamic states the pipelines take: without extended dynamic state 1 / 2 (vk.h VkCompat), their states (DYN[7..14],
// DYN[15..16]) go into the pipelines instead.
// (made once, but by whichever builds come first - Pokémon Z-A's startup shader preparation starts dozens on the
// workers at once: each fills its own copy and stores the same values, then the count. The shared count it counted up
// before ran past the array when two threads filled it together - garbage dynamic states, an NVIDIA driver crash in
// vkCreateGraphicsPipelines about 1 start in 6)
static const VkDynamicState *dyn_states(uint32_t *n) {
    static VkDynamicState d[NDYN];
    static _Atomic uint32_t nd;
    uint32_t c = atomic_load(&nd);
    if (!c) {
        VkDynamicState t[NDYN];
        for (uint32_t i = 0; i < NDYN; i++)
            if (!(g_vkc.no_eds1 && i >= 7 && i <= 14) && !(g_vkc.no_eds2 && i >= 15)) t[c++] = DYN[i];
        memcpy(d, t, sizeof t);
        atomic_store(&nd, c);
    }
    *n = c;
    return d;
}

// Fixed-function state that isn't part of a pipeline is set explicitly on every bind (it persists across pipelines).
static void dyn_defaults(VkCommandBuffer cb) {
    rec_CmdSetDepthBias(cb, 0, 0, 0);
    rec_CmdSetDepthBiasEnable(cb, VK_FALSE);
    float bc[4] = {0, 0, 0, 0};
    rec_CmdSetBlendConstants(cb, bc);
    rec_CmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    rec_CmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    rec_CmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    rec_CmdSetCullMode(cb, VK_CULL_MODE_NONE);
    rec_CmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    rec_CmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    rec_CmdSetDepthTestEnable(cb, VK_FALSE);
    rec_CmdSetDepthWriteEnable(cb, VK_FALSE);
    rec_CmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
    rec_CmdSetStencilTestEnable(cb, VK_FALSE);
    rec_CmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);
    rec_CmdSetPrimitiveRestartEnable(cb, VK_FALSE);
}

typedef struct { uint64_t key; VkPipeline pipe; } PipeEnt;
#define CLEAR_TAB 256
static PipeEnt clear_tab[CLEAR_TAB];

// This AMD driver draws nothing from a pipeline with fixed vertex input while dynamic vertex input is enabled.
// These triangles are generated from gl_VertexIndex and have no attributes, so the dynamic state is an empty layout.
// Without it the movie and the upscaler present uninitialised memory (red noise) while the audio plays.
static void fullscreen_vertex_dyn(VkDynamicState *dyn, uint32_t *n, uint32_t cap) {
    if (g_vk.vids && *n < cap) dyn[(*n)++] = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT;
}
static void fullscreen_vertex_bind(VkCommandBuffer cb) {
    if (!g_vk.vids) return;
    VkVertexInputBindingDescription2EXT b = {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT};
    VkVertexInputAttributeDescription2EXT a = {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT};
    rec_CmdSetVertexInputEXT(cb, 0, &b, 0, &a);
    ss_forget();  // the next game draw must set its own layout again
}

static VkPipeline clear_pipeline(int index, int kind, uint32_t mask, bool depth, bool stencil) {
    uint64_t key = 0xcbf29ce484222325ull;
#define MIX(v) do { uint64_t v_ = (uint64_t)(v); for (int b_ = 0; b_ < 8; b_++) { key ^= (v_ >> (b_ * 8)) & 0xFF; key *= 0x100000001B3ull; } } while (0)
    MIX(index); MIX(kind); MIX(mask); MIX(depth); MIX(stencil); MIX(pass_ncolor); MIX(pass_depth); MIX(pass_stencil); MIX(pass_samples);
    for (uint32_t i = 0; i < pass_ncolor; i++) MIX(pass_color[i]);
    uint32_t slot = (uint32_t)key & (CLEAR_TAB - 1);
    for (int p = 0; p < CLEAR_TAB; p++, slot = (slot + 1) & (CLEAR_TAB - 1)) {
        if (clear_tab[slot].key == key) return clear_tab[slot].pipe;
        if (!clear_tab[slot].key) break;
    }
    char fsname[32];
    if (index >= 0) snprintf(fsname, sizeof fsname, "clear.frag.%c%d", kind == 1 ? 'i' : kind == 2 ? 'u' : 'f', index);
    else snprintf(fsname, sizeof fsname, "clear.frag.none");
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("clear.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fsname), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)pass_samples};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {0};
    for (uint32_t i = 0; i < pass_ncolor; i++)
        if ((int)i == index) cba[i].colorWriteMask = mask & 0xF;
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = pass_ncolor, .pAttachments = cba};
    if (g_vkc.no_eds1) {  // (what clear() sets as dynamic state otherwise)
        ds.depthTestEnable = ds.depthWriteEnable = depth, ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
        ds.stencilTestEnable = stencil;
        ds.front = ds.back = (VkStencilOpState){VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                                                VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 0};
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    }
    uint32_t ndyn;
    const VkDynamicState *dyns = dyn_states(&ndyn);  // (Vulkan 1.2 without extended dynamic state: fewer)
    VkDynamicState dyn[NDYN + 1];
    memcpy(dyn, dyns, ndyn * sizeof *dyn);
    fullscreen_vertex_dyn(dyn, &ndyn, NDYN + 1);
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = ndyn, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo rinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = pass_ncolor,
                                           .pColorAttachmentFormats = pass_color, .depthAttachmentFormat = pass_depth,
                                           .stencilAttachmentFormat = pass_stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rinfo, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = pc_layout()};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p) != VK_SUCCESS) LOG("[vk] clear pipeline failed\n");
    clear_tab[slot].key = key;
    clear_tab[slot].pipe = p;
    return p;
}

static void clear(int index, int kind, const void *color, uint32_t mask, bool depth, float dval, bool stencil, uint32_t sval,
                  uint32_t smask) {
    begin_pass();
    pass_cur()->movie_bar_w = pass_cur()->movie_bar_h = 0;
    pass_cur()->has_non_movie_draw = false;
    VkCommandBuffer cb = vk_cmd();
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    VkRect2D rect = scissor_rect();
    if (!rect.extent.width || !rect.extent.height) return;
    if (index >= 0 && (index >= (int)pass_ncolor || pass_color[index] == VK_FORMAT_UNDEFINED)) return;
    bool simple = (index < 0 || (mask & 0xF) == 0xF) && (!stencil || (smask & 0xFF) == 0xFF);
    if (simple) {
        VkClearAttachment a[2];
        uint32_t n = 0;
        if (index >= 0) {
            a[n] = (VkClearAttachment){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)index};
            memcpy(a[n].clearValue.color.float32, color, 16);
            n++;
        }
        VkImageAspectFlags dsa = (depth && pass_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (stencil && pass_stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        if (dsa) a[n++] = (VkClearAttachment){.aspectMask = dsa, .clearValue.depthStencil = {dval, sval}};
        VkClearRect cr = {rect, 0, pass_layers};
        if (n) rec_CmdClearAttachments(cb, n, a, 1, &cr);
        return;
    }
    VkPipeline p = clear_pipeline(index, kind, mask, depth, stencil);
    if (!p) return;
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    ss_forget();  // (its own pipeline and state: the next draw sets everything again)
    dyn_defaults(cb);
    if (depth) { rec_CmdSetDepthTestEnable(cb, VK_TRUE); rec_CmdSetDepthWriteEnable(cb, VK_TRUE); }
    if (stencil) {
        rec_CmdSetStencilTestEnable(cb, VK_TRUE);
        rec_CmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE,
                          VK_COMPARE_OP_ALWAYS);
        rec_CmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, smask);
        rec_CmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, sval);
    }
    int w, h;
    pass_size(&w, &h);
    VkViewport vp = {0, 0, (float)(w ? w : (int)pass_w), (float)(h ? h : (int)pass_h), 0, 1};
    rec_CmdSetViewport(cb, 0, 1, &vp);
    rec_CmdSetScissor(cb, 0, 1, &rect);
    float pc[8] = {0};
    if (color) memcpy(pc, color, 16);
    pc[4] = dval;
    rec_CmdPushConstants(cb, pc_layout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    fullscreen_vertex_bind(cb);
    rec_CmdDraw(cb, 3, 1, 0, 0);
}

// ================================================================== programs
#define MAX_BIND 96
typedef struct {
    ShaderInfo *vi, *fi;
    uint64_t code[2], cmax[2];  // translation inputs (copied: translated on the translator thread)
    ShaderKey vkey, fkey;
    VkShaderModule vs, fs;
    VkShaderModule fs_late;  // (lazy chunks) fs without early fragment tests, for alpha to coverage pipelines; or NULL
    VkBindingInfo binds[MAX_BIND];
    int nbinds;
    VkDescriptorSetLayout set_layout[3];  // [2] (compute: [1]): the heap chunk set (gmem_chunk_layout)
    VkPipelineLayout layout;
    bool vs_uses_vid;
    _Atomic int state;  // 0 building, 1 ready, 2 failed
    uint64_t prep_vh, prep_fh;  // (prepared programs: their code hashes, for LARECOMP_PREP_CHECK)
    uint64_t sess_vh, sess_fh, sess_ck;  // (sess_progs: its code hashes and content key, entered once translated)
    ShaderFp fp[2];             // (stale-program detector, shader_prep.h; compute: [0]; touched by the replay only)
    struct PipeMemo *pmemo;     // (render thread) draw_'s pipeline key memo for this program
    bool reuse;                 // set reuse (stage_set): some uniform buffers are dynamic, sets are kept while their contents repeat
    struct SetCache *scache[9][2]; // stage_set's last set per set index, per context (exec.c NCTX: the render thread's and
                                   // each shadow worker part's - from its own pools)
    _Atomic uint64_t code_h[2];  // its stages' code hashes (shader_stage_hash), at its first draw: issue #22's workaround
} Prog;
static void sess_put_masked(Prog *p);

// ---- descriptor set reuse (graphics programs; port of vk-draw-cpu-2 d50bbce)
// Every draw allocated and wrote two descriptor sets (~20 descriptors) although most repeat the previous draw's of
// the same program: the textures and samplers are the same, and the uniform buffers are copies in the same transient
// ring buffer at new offsets. With the uniform buffers that move per draw declared dynamic (their offsets go to
// vkCmdBindDescriptorSets), the set's contents (buffers, ranges, views, samplers) repeat, so stage_set keeps the last
// set of each program and set index and binds it again while what it would write is identical. A kept set belongs to
// the pending submission (its pools are reset with the slot) and to the current view generation; buffers bound in
// sets are destroyed deferred (after the submissions using them), or at a reboot (ds_gen).
// LARECOMP_VK_NO_SET_REUSE=1: no dynamic buffers, a new set per draw, as before.
static int set_reuse = -1;
static uint32_t ds_gen;  // bumped when a buffer bound in sets may be destroyed at once (reboot)
static bool set_reuse_on(void) {
    if (set_reuse < 0) set_reuse = getenv("LARECOMP_VK_NO_SET_REUSE") == NULL;
    return set_reuse;
}
static bool is_ubo(VkDescriptorType t) { return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; }
// Up to the device's limit for the whole pipeline layout (8 or more; NVIDIA 15, AMD 16), the uniform buffers whose
// copies move most: the game's constant buffers (bindings 2-16), the scale table (18), the driver constants (0), the
// tint (19), then the compiler constants (1: one copy per submission). (Build workers: before the program is ready.)
static void mark_dynamic(Prog *p) {
    if (!set_reuse_on()) return;
    uint32_t lim = g_vk.props.limits.maxDescriptorSetUniformBuffersDynamic, n = 0;
    if (lim > 32) lim = 32;
    static const int8_t tier_of[20] = {2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 3};
    for (int tier = 0; tier <= 4; tier++)
        for (int i = 0; i < p->nbinds && n < lim; i++) {
            VkBindingInfo *b = &p->binds[i];
            if (b->set > 1 || b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b->binding >= 20 || tier_of[b->binding] != tier) continue;
            b->type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            n++;
        }
    p->reuse = true;
    static _Atomic int logged, logged_left;  // (the limit once; programs left with static buffers: the first 4)
    int left = 0;
    for (int i = 0; i < p->nbinds; i++) left += p->binds[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && p->binds[i].set <= 1;
    if (!atomic_exchange(&logged, 1) || (left && atomic_fetch_add(&logged_left, 1) < 4))
        LOG("[vk] set reuse: %u dynamic uniform buffers allowed; a program with %u dynamic, %d left static\n", lim, n, left);
}

static VkPipelineLayout make_layout(Prog *p, int nsets, VkShaderStageFlags s0, VkShaderStageFlags s1) {
    p->set_layout[0] = vk_set_layout(p->binds, p->nbinds, 0, s0);
    if (nsets > 1) p->set_layout[1] = vk_set_layout(p->binds, p->nbinds, 1, s1);
    bool gm = gc_shift && ((p->vi && p->vi->uses_gmem) || (p->fi && p->fi->uses_gmem));
    if (gm) p->set_layout[nsets++] = gmem_chunk_layout();
    // (lazy chunks, vertex / fragment: the storage buffers' chunks as push constants, shader.h GMEM_DIRECT_MAX)
    VkPushConstantRange gd = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 * GMEM_DIRECT_MAX};
    bool direct = gm && gc_lazy && s1 == VK_SHADER_STAGE_FRAGMENT_BIT;
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = (uint32_t)nsets,
                                     .pSetLayouts = p->set_layout, .pushConstantRangeCount = direct ? 1 : 0,
                                     .pPushConstantRanges = direct ? &gd : NULL};
    VkPipelineLayout l;
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l));
    return l;
}

static void save_glsl(const char *src, uint64_t addr, const char *suffix) {
    if (!ENV("LARECOMP_SAVE_MSL")) return;
    mkdir("build/glsl", 0755);
    char path[256];
    snprintf(path, sizeof path, "build/glsl/%llx_%s.glsl", (unsigned long long)addr, suffix);
    FILE *f = fopen(path, "w");
    if (f) { fputs(src, f); fclose(f); }
}

void slow_note_build(const char *what, uint64_t ns);  // (slowframe.h: the slow-frame log's counts; every build,
void slow_note_cache_hit(void);                       // and the pipelines the driver's cache had)
ShaderInfo *slow_translate(const uint8_t *code, size_t max_size, int stage, const ShaderKey *key);
#define maxwell_translate slow_translate  // (the translations below: counted and timed for the slow-frame log)
#define SLOW_FEEDBACK(info) VkPipelineCreationFeedback fb_ = {0}; VkPipelineCreationFeedbackCreateInfo fbi_ = \
    {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO, .pNext = (info).pNext, .pPipelineCreationFeedback = &fb_}; \
    (info).pNext = &fbi_
#define SLOW_FEEDBACK_HIT() do { if (fb_.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) slow_note_cache_hit(); } while (0)

// A build the driver takes over a second for is logged with its shaders: draws and dispatches are skipped until it
// is done, which looked like a rendering bug (the chain chomps' compute with the heap in chunks: never ready).
static void slow_build(uint64_t t0, const char *what, uint64_t code0, uint64_t code1) {
    slow_note_build(what, SDL_GetTicksNS() - t0);
    uint64_t ms = (SDL_GetTicksNS() - t0) / 1000000;
    if (ms < 1000) return;
    static int n;
    if (n++ < 40) LOG("[shader] slow build: %s %llx %llx took %llu ms\n", what, (unsigned long long)code0, (unsigned long long)code1, (unsigned long long)ms);
}

// A fragment shader the translator gave early fragment tests (lazy chunks, translate.c glsl_early_tests): removes the
// OpExecutionMode EarlyFragmentTests (in place; true if there was one). Alpha to coverage drops samples after the
// shader, and with early tests their depth / stencil would already be written: those pipelines take this module.
static bool spirv_strip_early_tests(uint32_t *w, size_t *nwords) {
    for (size_t i = 5; i < *nwords;) {
        uint32_t n = w[i] >> 16, op = w[i] & 0xFFFF;
        if (!n || i + n > *nwords) return false;
        if (op == 54) return false;  // OpFunction: past the execution modes
        if (op == 16 && n == 3 && w[i + 2] == 9) {  // OpExecutionMode %main EarlyFragmentTests
            memmove(w + i, w + i + n, (*nwords - i - n) * sizeof *w);
            *nwords -= n;
            return true;
        }
        i += n;
    }
    return false;
}

static void compile_program(void *arg) {
    Prog *p = arg;
    uint64_t t0 = SDL_GetTicksNS();
    int bs = build_begin("modules vs/fs", p->code[0] ? p->code[0] : p->prep_vh, p->code[1] ? p->code[1] : p->prep_fh);
    size_t nv = 0, nf = 0;
    uint32_t *sv = vk_glsl_to_spirv(p->vi->msl, STAGE_VERTEX, "vs", &nv);
    uint32_t *sf = sv ? vk_glsl_to_spirv(p->fi->msl, STAGE_FRAGMENT, "fs", &nf) : NULL;
    if (!sv || !sf) { free(sv); free(sf); atomic_store(&p->state, 2); build_end(bs); return; }
    p->nbinds = vk_reflect(sv, nv, p->binds, MAX_BIND);
    p->nbinds += vk_reflect(sf, nf, p->binds + p->nbinds, MAX_BIND - p->nbinds);
    p->vs = vk_module(sv, nv);
    p->fs = vk_module(sf, nf);
    if (spirv_strip_early_tests(sf, &nf)) p->fs_late = vk_module(sf, nf);
    free(sv);
    free(sf);
    mark_dynamic(p);
    p->layout = make_layout(p, 2, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT);
    atomic_fetch_add(&stat_compiles, 1);
    if (!p->vs || !p->fs) LOG("[shader] vs/fs %llx %llx: shader module creation failed\n", (unsigned long long)p->code[0], (unsigned long long)p->code[1]);
    atomic_fetch_add(&builds_done, 1);
    atomic_store(&p->state, p->vs && p->fs ? 1 : 2);
    build_end(bs);
    slow_build(t0, "modules vs/fs", p->code[0], p->code[1]);
}

// The guest memory window that shader global memory, the vertex cache and the texture upload checks cover: the heap
// (SetHeapSize), or for a game on the SDK's virtual address memory (Pokémon Legends Z-A: no heap) its largest physical
// memory region (svcMapPhysicalMemory, at most HEAP_MAX). A game with both gets the larger: Pokémon Legends Arceus sets
// a heap but keeps its graphics memory in a 3.1 GB physical pool (0x7b77b00000) - on the heap window its characters'
// skinning (transform feedback) and storage buffers were outside it, and characters and text weren't drawn. Picked
// once, at first use: shaders bake the base in.
typedef struct { uint64_t base, max; bool phys; } GWin;
static GWin gwin_ = {HEAP_BASE, HEAP_MAX, false};
// (a physical-memory window) the storage copies' ring (snap_storage): 16 MB right past the game's pool, inside the
// window - shaders reach storage by guest address through it. Offset in the window, 0: none (no room, or something
// mapped there). The kernel tells the game of 16 MB less memory for it (kernel.c snap_reserve)
#define GWIN_RING (16ull << 20)
static uint64_t gwin_ring;
static atomic_bool gwin_done;
static void gwin_pick(void) {
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&lock);
    if (!atomic_load(&gwin_done)) {
        uint64_t a = 0, sz = kernel_phys_largest(&a);
        const char *force = ENV("LARECOMP_GWIN");  // (debug) <base hex>:<size hex>
        if (force) {
            unsigned long long fb = 0, fs = 0;
            sscanf(force, "%llx:%llx", &fb, &fs);
            gwin_ = (GWin){fb, fs, true};
            plat_commit(fb, fs);  // (the host import needs every page committed)
            LOG("[vk] guest memory window forced: %llx +%llx\n", fb, fs);
            atomic_store(&gwin_done, true);
        } else if (kernel_heap_size() && kernel_heap_size() >= sz) atomic_store(&gwin_done, true);
        else if (sz >= (64ull << 20)) {
            gwin_ = (GWin){a, sz < HEAP_MAX ? sz : HEAP_MAX, true};
            extern bool kernel_range_free(uint64_t addr, uint64_t size);
            extern void kernel_guard_range(uint64_t addr, uint64_t size);
            const char *e = getenv("LARECOMP_OVERLAP_SNAP");
            uint64_t ro = (sz + 0xFFFF) & ~0xFFFFull;  // (minImportedHostPointerAlignment: 64 KB at most)
            if ((!e || atoi(e) != 0) && ro + GWIN_RING <= HEAP_MAX && kernel_range_free(a + ro, GWIN_RING)) {
                plat_commit(a + ro, GWIN_RING);  // (Windows: backed before the import pins it)
                kernel_guard_range(a + ro, GWIN_RING);
                gwin_ring = ro;
                gwin_.max = ro + GWIN_RING;
            } else LOG("[vk] no room for the storage copies' ring past the game's pool (%llx +%llx)\n", (unsigned long long)a,
                       (unsigned long long)sz);
            LOG("[vk] guest memory window: physical memory %llx +%llx (heap %llu MB)\n", (unsigned long long)a,
                (unsigned long long)gwin_.max, (unsigned long long)kernel_heap_size() >> 20);
            atomic_store(&gwin_done, true);
        }
    }
    pthread_mutex_unlock(&lock);
}
static inline const GWin *gwin(void) {
    if (__builtin_expect(!atomic_load_explicit(&gwin_done, memory_order_acquire), 0)) gwin_pick();
    return &gwin_;
}
#define gwin_base (gwin()->base)
#define gwin_max (gwin()->max)
// bytes of the window the game has set up: the heap size, or the whole physical region
static uint64_t gwin_want(void) { return gwin()->phys ? gwin_.max : kernel_heap_size(); }

// (debug) LARECOMP_GLSL_OVERRIDE=1: a file glsl_override/<code address>.{vert,frag} (the names LARECOMP_DUMP_GLSL
// writes under build/glsl) replaces that stage's translation - a hand-edited copy, to bisect a shader bug in the game
// without a new build (Scarlet / Violet's blue hair: which term of the lighting shader glows). It must keep the
// translation's resources (bindings, inputs, outputs); a file that doesn't compile fails the program as a bad
// translation would. The translation cache keeps the original. The files are the player's own, never shipped.
static void glsl_override(ShaderInfo *si, uint64_t addr, const char *ext) {
    static int on = -1;
    if (on < 0) on = ENV("LARECOMP_GLSL_OVERRIDE") && atoi(ENV("LARECOMP_GLSL_OVERRIDE")) != 0;
    if (!on || !si || !si->msl || !addr) return;
    char path[96];
    snprintf(path, sizeof path, "glsl_override/%llx.%s", (unsigned long long)addr, ext);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = n > 0 && n < (64l << 20) ? malloc((size_t)n + 1) : NULL;
    if (src && fread(src, 1, (size_t)n, f) == (size_t)n) {
        src[n] = 0;
        si->msl = src;  // (the translation's text stays allocated: a cached ShaderInfo may share it)
        LOG("[shader] %s replaces the translation (LARECOMP_GLSL_OVERRIDE)\n", path);
    } else {
        free(src);
        LOG("[shader] %s: unreadable, translation kept\n", path);
    }
    fclose(f);
}

static void gmem_window_get(uint64_t *lo, uint64_t *len);  // (the guest-heap import window, below)
// Translates both stages (false: failed, logged). Holds the translator for the whole program.
static bool translate_program_now(Prog *p) {
    uint64_t wl, wn;
    gmem_window_get(&wl, &wn);
    pthread_mutex_lock(&translate_lock);  // (the globals too: the render thread translates as well - run_queued_now)
    maxwell_gmem_base = gwin_base + wl, maxwell_gmem_size = wn;
    maxwell_gmem_chunk_shift = gmem_chunk_shift();
    p->vi = maxwell_translate((const uint8_t *)(uintptr_t)p->code[0], p->cmax[0], STAGE_VERTEX, &p->vkey);
    if (p->vi->msl) {
        p->fkey.vs_out_mask = p->vi->out_mask;
        p->fi = p->code[1] ? maxwell_translate((const uint8_t *)(uintptr_t)p->code[1], p->cmax[1], STAGE_FRAGMENT, &p->fkey)
                           : maxwell_empty_fragment();
    }
    glsl_override(p->vi, p->code[0], "vert");
    if (p->code[1]) glsl_override(p->fi, p->code[1], "frag");
    if (ENV("LARECOMP_DUMP_GLSL")) {  // (debug) build/glsl/<code address>.{vert,frag}[.asm]: source and disassembly
        mkdir("build/glsl", 0755);
        for (int sg = 0; sg < 2; sg++) {
            ShaderInfo *si = sg ? p->fi : p->vi;
            if (!p->code[sg]) continue;
            char path[96];
            snprintf(path, sizeof path, "build/glsl/%llx.%s", (unsigned long long)p->code[sg], sg ? "frag" : "vert");
            FILE *f = fopen(path, "wb");
            if (f) { if (si && si->msl) fputs(si->msl, f); fclose(f); }
            strcat(path, ".asm");
            if ((f = fopen(path, "wb"))) {
                maxwell_disassemble((const uint8_t *)(uintptr_t)p->code[sg], p->cmax[sg], sg ? STAGE_FRAGMENT : STAGE_VERTEX, f);
                fclose(f);
            }
        }
    }
    pthread_mutex_unlock(&translate_lock);
    if (!p->vi->msl) {
        LOG("[shader] VS translate failed: %s (code %llx)\n", p->vi->error, (unsigned long long)p->code[0]);
        atomic_store(&p->state, 2);
        return false;
    }
    if (!p->fi->msl) {
        LOG("[shader] FS translate failed: %s\n", p->fi->error);
        atomic_store(&p->state, 2);
        return false;
    }
    p->vs_uses_vid = strstr(p->vi->msl, "gl_VertexIndex") != NULL;
    sess_put_masked(p);
    return true;
}

static void translate_program(void *arg) {
    Prog *p = arg;
    if (!translate_program_now(p)) return;
    save_glsl(p->vi->msl, p->code[0], "vs");
    save_glsl(p->fi->msl, p->code[1], "fs");
    run_job(compile_program, p, &p->state);
}

// The code at a shader's address, in brief: its length (to the end of its constants) and four of its words. Programs
// are looked up by code address, and nothing checked that the code there was still the one they were built from - a
// game can load other shaders where old ones were (upstream LAS Recomp's stale-program detector, c93564d, logs it).
// The fingerprint is part of the lookup: other code, its own program (the same code again finds its translation in
// cache/maxwell); "[shader] code changed" says when it happens.
static uint64_t code_fp(const NvnShaderStage *s) {
    uint32_t len = s->const_offset ? s->const_offset + s->const_size : 0;
    if (!s->code_addr || len < 4) return 0;
    const uint32_t *c = (const uint32_t *)(uintptr_t)s->code_addr;
    uint32_t n = len / 4, w[5] = {len, c[n > 0x20 ? 0x20 : 0], c[n / 3], c[2 * (n / 3)], c[n - 1]};
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < 5; i++) { h ^= w[i]; h *= 0x100000001B3ull; }
    return h;
}
// A new program's code: logged when the address held other code before (render thread)
static void note_code(const NvnShaderStage *s, uint64_t fp, const char *stage) {
    static struct { uint64_t addr, fp; } seen[8192];
    if (!s->code_addr) return;
    uint32_t i = (uint32_t)(s->code_addr >> 4) & 8191;
    for (int probe = 0; seen[i].addr && seen[i].addr != s->code_addr; i = (i + 1) & 8191)
        if (++probe == 64) return;  // (full here: not noted)
    static int n;
    if (seen[i].addr && seen[i].fp != fp && n++ < 40)
        LOG("[shader] code changed at %llx (%s): a new program\n", (unsigned long long)s->code_addr, stage);
    seen[i].addr = s->code_addr, seen[i].fp = fp;
}

// ================================================================== prepared objects (shader_prep.h)
// Built before the game from the shipped variant list; found by content (code hashes + keys) at first use, as code
// addresses differ from run to run. Kept for the whole process (reboots included). prep_lock guards the tables.
static pthread_mutex_t prep_lock = PTHREAD_MUTEX_INITIALIZER;
typedef struct { uint64_t key; void *obj; } PrepEnt;
static PrepEnt *prep_progs, *prep_pipes, *prep_cs;  // program / pipeline / compute content key -> Prog / Pipe / CsProg
static size_t prep_progs_cap, prep_pipes_cap, prep_cs_cap, prep_progs_n, prep_pipes_n, prep_cs_n;
static _Atomic int prep_pending;

static void *prep_get(PrepEnt *t, size_t cap, uint64_t key) {
    if (!cap) return NULL;
    if (!key) key = 1;
    for (size_t j = key & (cap - 1); t[j].key; j = (j + 1) & (cap - 1))
        if (t[j].key == key) return t[j].obj;
    return NULL;
}
static void prep_put(PrepEnt **t, size_t *cap, size_t *n, uint64_t key, void *obj) {
    if (!key) key = 1;
    if (*n * 2 >= *cap) {
        size_t nc = *cap ? *cap * 2 : 1024;
        PrepEnt *nt = calloc(nc, sizeof *nt);
        for (size_t i = 0; i < *cap; i++)
            if ((*t)[i].key) {
                size_t j = (*t)[i].key & (nc - 1);
                while (nt[j].key) j = (j + 1) & (nc - 1);
                nt[j] = (*t)[i];
            }
        free(*t);
        *t = nt, *cap = nc;
    }
    size_t j = key & (*cap - 1);
    while ((*t)[j].key && (*t)[j].key != key) j = (j + 1) & (*cap - 1);
    if (!(*t)[j].key) (*n)++;
    (*t)[j] = (PrepEnt){key, obj};
}
#define PREP_GET(tab, key) ({ pthread_mutex_lock(&prep_lock); void *o_ = prep_get(tab, tab##_cap, key); pthread_mutex_unlock(&prep_lock); o_; })
#define PREP_PUT(tab, key, obj) do { pthread_mutex_lock(&prep_lock); prep_put(&tab, &tab##_cap, &tab##_n, key, obj); pthread_mutex_unlock(&prep_lock); } while (0)

static uint64_t prog_content_key(uint64_t vh, uint64_t fh, const ShaderKey *vkey, const ShaderKey *fkey) {
    uint64_t h = 0xcbf29ce484222325ull;
    const uint64_t c[2] = {vh, fh};
    const uint8_t *b = (const uint8_t *)c;
    for (size_t i = 0; i < sizeof c; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    b = (const uint8_t *)vkey;
    for (size_t i = 0; i < sizeof *vkey; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    for (int i = 0; i < 8; i++) { h ^= fkey->rt_kind[i]; h *= 0x100000001B3ull; }
    // the fragment textures bound as stencil / integer views, as in get_program_'s own key: a program prepared before
    // the game is translated with none (the records don't carry them), and it was handed to every draw of its shaders -
    // Scarlet / Violet's lighting reads the scene's stencil through slot 28 (a view in mode STENCIL) and got the
    // prepared program's float sampler on it: the depth's bits where the stencil's were asked, stripes on everything lit
    for (int k = 0; k < 4; k++) { h ^= (fkey->stencil_slots >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
    for (int k = 0; k < 4; k++) { h ^= (fkey->sint_slots >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
    return h;
}

// The caches of programs and pipelines by key (get_program_, draw_): open addressing, doubled at half full. They were
// fixed at 4096 / 8192 entries and never emptied: a long session filled them (Pokémon Z-A, ~29 minutes in, a new
// area's shaders) and the next new key probed forever - the render thread spun, the game froze. (Render thread only.)
typedef struct { uint64_t key; void *val; } KvEnt;
typedef struct { KvEnt *t; uint32_t cap, n; } KvTab;
static KvEnt *kv_find(KvTab *k, uint64_t key) {  // key's entry, or the empty one it would take (key != 0)
    if (!k->cap) {  // (LARECOMP_VK_KV_START=<power of 2> (test): a small start, the growth exercised in any run)
        static uint32_t start;
        if (!start) start = ENV("LARECOMP_VK_KV_START") ? (uint32_t)atoi(ENV("LARECOMP_VK_KV_START")) : 4096;
        if (start < 16 || (start & (start - 1))) start = 4096;  // (the probe masks with cap - 1)
        k->cap = start, k->t = calloc(k->cap, sizeof *k->t);
    }
    uint32_t i = (uint32_t)(key ^ (key >> 31)) & (k->cap - 1);
    while (k->t[i].key && k->t[i].key != key) i = (i + 1) & (k->cap - 1);
    return &k->t[i];
}
static void kv_set(KvTab *k, KvEnt *e, uint64_t key, void *val) {  // e: kv_find's for key, no kv_find of k since
    if (!e->key) {
        if ((k->n + 1) * 2 > k->cap) {
            KvEnt *old = k->t;
            uint32_t oc = k->cap;
            k->cap *= 2, k->t = calloc(k->cap, sizeof *k->t);
            for (uint32_t i = 0; i < oc; i++) if (old[i].key) *kv_find(k, old[i].key) = old[i];
            free(old);
            e = kv_find(k, key);
        }
        k->n++;
    }
    e->key = key, e->val = val;
}
static void kv_clear(KvTab *k) { if (k->t) memset(k->t, 0, k->cap * sizeof *k->t); k->n = 0; }
static KvTab prog_kv;

static void prog_fp_check(Prog *p, NvnProgram *np) {  // (the lookups below are by code address: shader_prep.h)
    shader_fp_check(&p->fp[0], &np->stages[0], "vs");
    shader_fp_check(&p->fp[1], &np->stages[1], "fs");
}

// Key memo: draws mostly repeat the program and pipeline state of a recent draw. draw_ compares the inputs of the
// program lookup with the previous lookup's, and those of the pipeline lookup with the last ones of that program
// (plain compares of everything the keys are made from), and reuses the answer instead of hashing the program key
// byte by byte and filling + hashing the ~700-byte PipeKey. The answers are the objects the tables give. key_gen:
// bumped by vk_exec_reset (tables cleared). LARECOMP_VK_NO_KEY_MEMO=1: off.
static int key_memo = -1;
static uint32_t key_gen = 1;
static bool key_memo_on(void) {
    if (key_memo < 0) key_memo = getenv("LARECOMP_VK_NO_KEY_MEMO") == NULL;
    return key_memo;
}

static Prog *get_program_(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey);
static Prog *get_program(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) {
    static struct { uint64_t code[2], fp[2]; ShaderKey vkey; uint8_t rt[8]; uint32_t gen; Prog *prog; } m_t[NCTX][64];
    // (64 a context, by code address: the shadow workers' parts switch programs across draws of different objects)
#define m (m_t[sw_ctx()][(np->stages[0].code_addr >> 6 ^ np->stages[1].code_addr >> 4 ^ np->stages[1].code_addr >> 13) & 63])
    bool on = key_memo_on();
    // (the code fingerprints too: other code at the same address is another program - code_fp)
    uint64_t fp0 = code_fp(&np->stages[0]), fp1 = code_fp(&np->stages[1]);
    if (on && m.prog && m.gen == key_gen && (np->stage_mask & 3) == 3 && m.code[0] == np->stages[0].code_addr &&
        m.code[1] == np->stages[1].code_addr && m.fp[0] == fp0 && m.fp[1] == fp1 && !memcmp(&m.vkey, vkey, sizeof *vkey) &&
        !memcmp(m.rt, fkey->rt_kind, 8)) {
        prof_prog_memo++;
        return m.prog;
    }
    SWL_AT(2551);  // (shadow workers: the program table)
    Prog *p = get_program_(np, vkey, fkey);
    SWU();
    if (on && p) {
        m.code[0] = np->stages[0].code_addr; m.code[1] = np->stages[1].code_addr; m.fp[0] = fp0; m.fp[1] = fp1;
        m.vkey = *vkey; memcpy(m.rt, fkey->rt_kind, 8); m.gen = key_gen; m.prog = p;
    }
    return p;
#undef m
}

// Programs and compute programs built this run, found again by content (as the prepared ones are) before one is
// translated and compiled anew. A battle in Scarlet / Violet ("Battle", the move-select camera) put ~60 of the field's
// programs through a new build at once, and while they built their draws were skipped (the scene wait budget,
// wait_scene, gone in the first few): for a frame or two the battlefield's ground, the trainer and what hides the far
// land were missing - the far land seen through where the ground was, then the ground unlit (black). Once each battle.
//   - The same shaders at another code address are the same program: the game loads a battle's shaders into new
//     memory each time (sess_progs by the content key).
//   - A key differing only in slots the shaders don't read is the same program: the stencil / integer slot masks
//     (stencil_slots) are taken from all 32 bound texture slots, and at that moment the lighting's stencil view
//     (slot 28) is bound while the ground's shaders draw - they never read it, and the translation (translate.c, by
//     the slots each texture of a shader is read from) comes out the same. Once a program is translated, it is also
//     found by its key with only the slots its shaders read (sess_put_masked / sess_get_masked); a shader reading a
//     texture through a handle it loads (bindless) counts as reading every slot. The programs prepared before the
//     game (shader_prep.h) are entered so too: their records carry no stencil / integer slots, so a draw with the
//     lighting's view bound found none of them - with a warm cache the first battle after starting still went black.
// LARECOMP_VK_SESSION_PROGRAMS=0: off (built again, as before).
static PrepEnt *sess_progs, *sess_cs, *sess_slots;  // content key -> Prog / CsProg; stage code hash -> its slots + 1<<32
static size_t sess_progs_cap, sess_cs_cap, sess_slots_cap, sess_progs_n, sess_cs_n, sess_slots_n;
static bool sess_on(void) {
    static _Atomic int on = -1;
    int v = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        v = !ENV("LARECOMP_VK_SESSION_PROGRAMS") || atoi(ENV("LARECOMP_VK_SESSION_PROGRAMS")) != 0;
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v != 0;
}
static uint32_t sess_used_slots(const ShaderInfo *si) {  // the texture handle slots a translated shader reads
    uint32_t m = 0;
    for (int i = 0; si && i < si->ntex; i++) {
        if (si->tex[i].bindless) return ~0u;
        int hs = ((int)si->tex[i].handle_offset - 0x20) / 8;
        if (hs >= 0 && hs < 32) m |= 1u << hs;
    }
    return m;
}
static uint64_t sess_masked_key(uint64_t vh, uint64_t fh, uint32_t vm, uint32_t fm, const ShaderKey *vkey, const ShaderKey *fkey) {
    ShaderKey v = *vkey, f = *fkey;
    v.stencil_slots &= vm, v.sint_slots &= vm;
    f.stencil_slots &= fm, f.sint_slots &= fm;
    f.vs_out_mask = 0;  // (the translation sets it on its own copy: lookups have none)
    return prog_content_key(vh, fh, &v, &f) ^ 0x5e55a5e55a5e55a5ull;
}
static bool sess_slots_of(uint64_t h, uint32_t *m) {
    uintptr_t v = (uintptr_t)PREP_GET(sess_slots, h);
    if (!v) return false;
    *m = (uint32_t)v;
    return true;
}
// (translator thread) a program just translated: found from now on by its content and by the slots its shaders read.
// Not before: its translation reads the game's code at its own address, which may be freed by then (another battle).
static void sess_put_masked(Prog *p) {
    if (!sess_on() || !p->sess_vh || !p->sess_fh) return;
    if (p->sess_ck) PREP_PUT(sess_progs, p->sess_ck, p);  // (a prepared one is in prep_progs by it already)
    uint32_t vm = sess_used_slots(p->vi), fm = sess_used_slots(p->fi);
    PREP_PUT(sess_slots, p->sess_vh, (void *)(uintptr_t)((uint64_t)vm | 1ull << 32));
    PREP_PUT(sess_slots, p->sess_fh, (void *)(uintptr_t)((uint64_t)fm | 1ull << 32));
    PREP_PUT(sess_progs, sess_masked_key(p->sess_vh, p->sess_fh, vm, fm, &p->vkey, &p->fkey), p);
}
static Prog *sess_get_masked(uint64_t vh, uint64_t fh, const ShaderKey *vkey, const ShaderKey *fkey) {
    uint32_t vm, fm;
    if (!sess_slots_of(vh, &vm) || !sess_slots_of(fh, &fm)) return NULL;
    return PREP_GET(sess_progs, sess_masked_key(vh, fh, vm, fm, vkey, fkey));
}
static Prog *get_program_(NvnProgram *np, ShaderKey *vkey, ShaderKey *fkey) {
    if (!(np->stage_mask & 1)) return NULL;  // (no fragment stage: depth only - maxwell_empty_fragment)
    uint64_t h = 0xcbf29ce484222325ull, fp[2], sess_ck = 0;
    for (int i = 0; i < 2; i++) {
        uint64_t a = np->stages[i].code_addr;
        fp[i] = code_fp(&np->stages[i]);
        for (int k = 0; k < 8; k++) { h ^= (a >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
        for (int k = 0; k < 8; k++) { h ^= (fp[i] >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
    }
    const uint8_t *vb = (const uint8_t *)vkey;
    for (size_t i = 0; i < sizeof *vkey; i++) { h ^= vb[i]; h *= 0x100000001B3ull; }
    for (int i = 0; i < 8; i++) { h ^= fkey->rt_kind[i]; h *= 0x100000001B3ull; }
    for (int k = 0; k < 4; k++) { h ^= (fkey->stencil_slots >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
    for (int k = 0; k < 4; k++) { h ^= (fkey->sint_slots >> (k * 8)) & 0xFF; h *= 0x100000001B3ull; }
    if (!h) h = 1;
    KvEnt *slot = kv_find(&prog_kv, h);
    if (slot->key) {
        prog_fp_check(slot->val, np);
        return slot->val;
    }
    {   // prepared before the game (the same code and keys)
        uint64_t vh = shader_stage_hash(&np->stages[0]), fh = shader_stage_hash(&np->stages[1]);
        uint64_t ck = prog_content_key(vh, fh, vkey, fkey);
        Prog *pp = vh && fh && prep_progs_n ? PREP_GET(prep_progs, ck) : NULL;
        if (!pp && vh && fh && sess_on()) {  // (built earlier: elsewhere, or with another unread slot)
            pp = PREP_GET(sess_progs, ck);
            if (!pp) pp = sess_get_masked(vh, fh, vkey, fkey);
        }
        if (pp) {
            kv_set(&prog_kv, slot, h, pp);
            prog_fp_check(pp, np);
            return pp;
        }
        shader_first_visit("program", vh, fh, ck);
        sess_ck = vh && fh ? ck : 0;
    }
    Prog *p = calloc(1, sizeof *p);
    prog_fp_check(p, np);  // (takes the fingerprint)
    kv_set(&prog_kv, slot, h, p);
    if (sess_ck && sess_on())  // (entered in sess_progs once translated: sess_put_masked)
        p->sess_ck = sess_ck, p->sess_vh = shader_stage_hash(&np->stages[0]), p->sess_fh = shader_stage_hash(&np->stages[1]);
    note_code(&np->stages[0], fp[0], "vs");
    note_code(&np->stages[1], fp[1], "fs");
    for (int i = 0; i < 2; i++) {
        p->code[i] = np->stages[i].code_addr;
        p->cmax[i] = np->stages[i].const_offset ? np->stages[i].const_offset : 0x10000;
    }
    p->vkey = *vkey;
    p->fkey = *fkey;
    run_translate(translate_program, p, &p->state);
    return p;
}

// ================================================================== vertex formats / fixed function tables
static int attr_kind(uint32_t fmt) {
    switch (fmt) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x1B: case 0x20: case 0x23:
    case 0x27: case 0x2C: case 0x2F: case 0x3E:
        return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x1C: case 0x21: case 0x24:
    case 0x28: case 0x2D: case 0x30:
        return 1;
    default: return 0;
    }
}

static VkFormat vertex_format(uint32_t f) {
    switch (f) {
    case 0x01: return VK_FORMAT_R8_UNORM;
    case 0x02: return VK_FORMAT_R8_SNORM;
    case 0x03: return VK_FORMAT_R8_UINT;
    case 0x04: return VK_FORMAT_R8_SINT;
    case 0x05: return VK_FORMAT_R16_SFLOAT;
    case 0x06: return VK_FORMAT_R16_UNORM;
    case 0x07: return VK_FORMAT_R16_SNORM;
    case 0x08: return VK_FORMAT_R16_UINT;
    case 0x09: return VK_FORMAT_R16_SINT;
    case 0x0A: return VK_FORMAT_R32_SFLOAT;
    case 0x0B: return VK_FORMAT_R32_UINT;
    case 0x0C: return VK_FORMAT_R32_SINT;
    case 0x0D: return VK_FORMAT_R8G8_UNORM;
    case 0x0E: return VK_FORMAT_R8G8_SNORM;
    case 0x0F: return VK_FORMAT_R8G8_UINT;
    case 0x10: return VK_FORMAT_R8G8_SINT;
    case 0x11: return VK_FORMAT_R16G16_SFLOAT;
    case 0x12: return VK_FORMAT_R16G16_UNORM;
    case 0x13: return VK_FORMAT_R16G16_SNORM;
    case 0x14: return VK_FORMAT_R16G16_UINT;
    case 0x15: return VK_FORMAT_R16G16_SINT;
    case 0x16: return VK_FORMAT_R32G32_SFLOAT;
    case 0x17: return VK_FORMAT_R32G32_UINT;
    case 0x18: return VK_FORMAT_R32G32_SINT;
    case 0x19: return VK_FORMAT_R8G8B8_UNORM;
    case 0x1A: return VK_FORMAT_R8G8B8_SNORM;
    case 0x1B: return VK_FORMAT_R8G8B8_UINT;
    case 0x1C: return VK_FORMAT_R8G8B8_SINT;
    case 0x1D: return VK_FORMAT_R16G16B16_SFLOAT;
    case 0x1E: return VK_FORMAT_R16G16B16_UNORM;
    case 0x1F: return VK_FORMAT_R16G16B16_SNORM;
    case 0x20: return VK_FORMAT_R16G16B16_UINT;
    case 0x21: return VK_FORMAT_R16G16B16_SINT;
    case 0x22: return VK_FORMAT_R32G32B32_SFLOAT;
    case 0x23: return VK_FORMAT_R32G32B32_UINT;
    case 0x24: return VK_FORMAT_R32G32B32_SINT;
    case 0x25: return VK_FORMAT_R8G8B8A8_UNORM;
    case 0x26: return VK_FORMAT_R8G8B8A8_SNORM;
    case 0x27: return VK_FORMAT_R8G8B8A8_UINT;
    case 0x28: return VK_FORMAT_R8G8B8A8_SINT;
    case 0x29: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case 0x2A: return VK_FORMAT_R16G16B16A16_UNORM;
    case 0x2B: return VK_FORMAT_R16G16B16A16_SNORM;
    case 0x2C: return VK_FORMAT_R16G16B16A16_UINT;
    case 0x2D: return VK_FORMAT_R16G16B16A16_SINT;
    case 0x2E: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case 0x2F: return VK_FORMAT_R32G32B32A32_UINT;
    case 0x30: return VK_FORMAT_R32G32B32A32_SINT;
    case 0x3D: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case 0x69: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case 0x3F: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case 0x40: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case 0x51: return VK_FORMAT_R8_UINT;
    default: {
        static uint8_t told[256];
        if (!told[f & 0xFF]++) LOG("[vk] vertex format %#x unknown: read as RGBA32F\n", f);
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    }
}

// 3-component 8/16-bit vertex formats aren't universally supported: read 4 components instead.
static VkFormat vertex_format_supported(VkFormat f) {
    static uint8_t known[256];  // 1 ok, 2 not
    uint32_t i = f < 256 ? f : 0;
    if (!known[i]) {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &p);
        known[i] = (p.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) ? 1 : 2;
    }
    if (known[i] == 1) return f;
    { static uint8_t told[256]; if (!told[i]++) LOG("[vk] vertex format %d unsupported: read as 4 components\n", f); }
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

static VkBlendFactor blend_factor(uint8_t f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: {
        static uint8_t warned[256];
        if (!warned[f]++) LOG("[vk] unknown blend factor %#x\n", f);
        return VK_BLEND_FACTOR_ONE;
    }
    }
}

static VkBlendOp blend_op(uint8_t e) {
    switch (e) {
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

static VkCompareOp compare_fn(uint8_t f) {
    switch (f) {
    case 1: return VK_COMPARE_OP_NEVER;
    case 2: return VK_COMPARE_OP_LESS;
    case 3: return VK_COMPARE_OP_EQUAL;
    case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 5: return VK_COMPARE_OP_GREATER;
    case 6: return VK_COMPARE_OP_NOT_EQUAL;
    case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

static VkStencilOp stencil_op(uint8_t o) {
    switch (o) {
    case 2: return VK_STENCIL_OP_ZERO;
    case 3: return VK_STENCIL_OP_REPLACE;
    case 4: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case 5: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case 6: return VK_STENCIL_OP_INVERT;
    case 7: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case 8: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default: return VK_STENCIL_OP_KEEP;
    }
}

static int rt_kind(uint32_t f) {
    switch (f) {
    case 0x03: case 0x08: case 0x0B: case 0x0F: case 0x14: case 0x17: case 0x27: case 0x2C: case 0x2F: case 0x3E: return 2;
    case 0x04: case 0x09: case 0x0C: case 0x10: case 0x15: case 0x18: case 0x28: case 0x2D: case 0x30: return 1;
    default: return 0;
    }
}

// ================================================================== graphics pipelines
// The fixed-function state a draw sets as extended dynamic state (fixed_state): without the extension, part of the
// pipeline key instead (vk.h VkCompat.no_eds1).
typedef struct {
    uint8_t dt, dw, dop, sten, cull, ff, topo, equal_tol;
    uint8_t sop[2][4];  // front / back: fail, pass, depth fail, compare op
} FixedState;

// Everything a pipeline bakes in, captured by value so it can be built on a worker thread.
#define ZERO_BINDING 16
typedef struct {
    Prog *prog;
    VkFormat color[8], depth, stencil;
    uint32_t ncolor, samples;
    uint32_t attr_fmt[32], attr_off[32], attr_bind[32];
    uint32_t attr_mask;
    uint32_t stride[17], divisor[17], used_streams;
    uint8_t blend[8][8];
    uint8_t blend_on, a2c, topo_class, poly_fill, depth_clamp;
    uint8_t restart;  // (no extended dynamic state 2 only: primitive restart, a strip topology)
    uint32_t write_mask[8];
    FixedState fx;    // (no extended dynamic state only: what draw_ sets as dynamic state otherwise)
} PipeKey;

typedef struct { PipeKey k; uint64_t hash; VkPipeline pipe; _Atomic int state; } Pipe;  // state: 0 building, 1 ready, 2 failed
// (planar reflections, below the water reflections) a captured draw's call: kind 0 draw, 1 indexed, 2 GPU-indirect
typedef struct { int kind; bool indexed; VkBuffer buf; VkDeviceSize off; uint32_t a, b, c, e; int32_t d; } PlDrawCall;
static bool planar_on(void);
static void planar_capture(Prog *pr, Pipe *pp, uint32_t streams, const VkDescriptorSet sets[2], uint32_t ndyn, const uint32_t *dyn,
                           uint32_t nv_dyn, bool gmem, const PlDrawCall *dc);
static void planar_light(Prog *pr, Pipe *pp, const VkDescriptorImageInfo *fii, VkDescriptorSet vset, const uint32_t *dyn,
                         uint32_t nv_dyn, const float rs0[4], const float *tint, uint32_t streams, const PlDrawCall *dc);
// (record.c, at the bind) the pipeline of a Pipe: NULL while it builds (wait: sleeps until it is done) or failed
static VkPipeline pipe_resolve(void *obj, bool wait) {
    Pipe *pp = obj;
    int st_;
    while ((st_ = atomic_load(&pp->state)) == 0) {
        if (!wait) return VK_NULL_HANDLE;
        struct timespec ts = {0, 200000};
        nanosleep(&ts, NULL);
    }
    return st_ == 1 ? pp->pipe : VK_NULL_HANDLE;
}
// (key memo, see get_program) everything pipe_key_fill reads besides the program: the pass's formats and the PipeIn.
// One per program (draws often alternate between a few programs): the last pipeline that program drew with.
struct PipeMemo {
    uint32_t gen, ncolor, samples, channel_mask;
    int nattrib, nstream;
    VkFormat color[8], depth, stencil;
    uint32_t attrib[16], stream[16][2], rt_fmt[8];
    uint8_t blend[8][8], blend_on, a2c, cls, poly_fill, depth_clamp, restart;  // (the last two: draw_'s own key parts)
    Pipe *pp;
    VtxLayout vl;  // the layout pipe_key_fill made from these inputs (dynamic vertex input leaves it out of pp's key)
};
static KvTab pipe_kv;  // (pipelines by key hash: kv_find above)

// 8 bytes a step (PipeKey is ~700 bytes, hashed per draw); in-memory tables only - hmix values are compared with
// stored constants (fragment-shader hashes) and must not change.
static uint64_t hash_words(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (; n >= 8; n -= 8, b += 8) { uint64_t w; memcpy(&w, b, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; n; n--, b++) { h = (h ^ *b) * 0x100000001B3ull; }
    return h ^ (h >> 32);
}

static uint64_t hmix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001B3ull; }
    return h;
}

static VkPrimitiveTopology topo_class_base(uint8_t c) {
    return c == 0 ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST : c == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
}

// What a pipeline is built from, besides its program and the pass's attachment formats: the draw's state (draw_) or a
// variant record (renderer_prepare)
typedef struct {
    const uint32_t *attrib; int nattrib;
    const uint32_t (*stream)[2]; int nstream;
    const uint32_t *rt_fmt;         // color target texture formats (0: none)
    uint32_t channel_mask;
    uint8_t blend_on;               // color state blend enables
    const uint8_t (*blend)[8];
    uint8_t a2c, topo_class, poly_fill;
} PipeIn;

// vl (optional): the vertex layout, also when the key leaves it out (dynamic vertex input)
static void pipe_key_fill(PipeKey *k, Prog *pr, const PipeIn *in, const VkFormat color[8], uint32_t ncolor, VkFormat depth,
                          VkFormat stencil, uint32_t samples, VtxLayout *vl) {
    memset(k, 0, sizeof *k);
    k->prog = pr;
    k->ncolor = ncolor;
    // (only the pass's own slots: begin_pass leaves the slots past them as the last pass with more targets set them -
    // a depth-only pass's key carried the previous pass's color formats, so prepared pipelines never matched it)
    for (uint32_t i = 0; i < ncolor && i < 8; i++) k->color[i] = color[i];
    k->depth = depth;
    k->stencil = stencil;
    k->samples = samples;
    VtxLayout l_, *l = vl ? vl : &l_;
    memset(l, 0, sizeof *l);
    for (int i = 0; i < 32; i++) {
        if (!(pr->vi->attr_mask & (1u << i))) continue;
        l->attr_mask |= 1u << i;
        uint32_t a = i < in->nattrib ? in->attrib[i] : 0x2E;
        uint32_t fmt = a & 0xFF, stream = (a >> 8) & 0x1F, off = a >> 13;
        if (i >= in->nattrib || fmt == 0 || stream >= 16) {  // unbound attribute: (0, 0, 0, 1) from a constant stream
            l->attr_fmt[i] = VK_FORMAT_R32G32B32A32_SFLOAT;
            l->attr_bind[i] = ZERO_BINDING;
            l->used_streams |= 1u << ZERO_BINDING;
            continue;
        }
        l->attr_fmt[i] = vertex_format_supported(vertex_format(fmt));
        l->attr_off[i] = off;
        l->attr_bind[i] = stream;
        l->used_streams |= 1u << stream;
    }
    for (int s = 0; s < 16; s++) {
        if (!(l->used_streams & (1u << s))) continue;
        l->stride[s] = s < in->nstream ? in->stream[s][0] : 0;
        l->divisor[s] = s < in->nstream && l->stride[s] ? in->stream[s][1] : 0;
        if (l->divisor[s] > 1 && !g_vk.divisor) l->divisor[s] = 1;
    }
    k->attr_mask = l->attr_mask;  // (the program's: the same with either path)
    if (!g_vk.vids) {             // the layout is part of the pipeline
        memcpy(k->attr_fmt, l->attr_fmt, sizeof k->attr_fmt);
        memcpy(k->attr_off, l->attr_off, sizeof k->attr_off);
        memcpy(k->attr_bind, l->attr_bind, sizeof k->attr_bind);
        memcpy(k->stride, l->stride, sizeof k->stride);
        memcpy(k->divisor, l->divisor, sizeof k->divisor);
        k->used_streams = l->used_streams;
    }
    for (uint32_t i = 0; i < k->ncolor && i < 8; i++) {
        if (!in->rt_fmt[i] || k->color[i] == VK_FORMAT_UNDEFINED) continue;
        // Targets the fragment shader doesn't output keep their contents on Metal, but are undefined in Vulkan
        // (NVIDIA writes black): mask them. Foliage depth pre-passes left black silhouettes (user 2026-10-08).
        k->write_mask[i] = ((pr->fi->out_mask >> (i * 4)) & 0xF) ? (in->channel_mask >> (i * 4)) & 0xF : 0;
        if (((in->blend_on >> i) & 1) && !rt_kind(in->rt_fmt[i])) {
            k->blend_on |= (uint8_t)(1u << i);
            memcpy(k->blend[i], in->blend[i], 8);
        }
    }
    const char *a2c_env = ENV("LARECOMP_A2C");
    k->a2c = a2c_env ? (atoi(a2c_env) != 0) : ((k->samples > 1) ? in->a2c : 0);
    k->topo_class = in->topo_class;
    k->poly_fill = in->poly_fill;
}

// A pipeline's create info from its key (the structures point into each other and into the key: filled in place)
typedef struct {
    VkPipelineShaderStageCreateInfo stages[2];
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req32_fs;
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkVertexInputBindingDivisorDescriptionEXT vd[17];
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvi;
    VkPipelineVertexInputStateCreateInfo vi;
    VkPipelineInputAssemblyStateCreateInfo ia;
    VkPipelineViewportStateCreateInfo vs;
    VkPipelineRasterizationStateCreateInfo rs;
    VkPipelineMultisampleStateCreateInfo ms;
    VkPipelineDepthStencilStateCreateInfo ds;
    VkPipelineColorBlendAttachmentState cba[8];
    VkPipelineColorBlendStateCreateInfo cb;
    VkDynamicState dyn[NDYN + 1];  // (+ the vertex layout with dynamic vertex input: draw_ sets it)
    VkPipelineDynamicStateCreateInfo dy;
    VkPipelineRenderingCreateInfo ri;
    VkGraphicsPipelineCreateInfo pi;
} PipeCI;

static bool fs_subgroup32(void) {
    static int ok = -1;
    if (ok < 0) {
        const char *e = ENV("LARECOMP_VK_FS_SUBGROUP");
        ok = 0;
        if (g_vkc.api >= VK_API_VERSION_1_3 && !(e && !strcmp(e, "0"))) {
            VkPhysicalDeviceSubgroupSizeControlProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
            VkPhysicalDeviceProperties2 pp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
            vkGetPhysicalDeviceProperties2(g_vk.phys, &pp);
            ok = sgp.minSubgroupSize <= 32 && sgp.maxSubgroupSize >= 32 && (sgp.requiredSubgroupSizeStages & VK_SHADER_STAGE_FRAGMENT_BIT);
            LOG("[vk] fragment subgroup sizes %u..%u: %s\n", sgp.minSubgroupSize, sgp.maxSubgroupSize, ok ? "fragment shaders run in 32 lanes (Intel/AMD warp alignment)" : "left to the driver");
        }
    }
    return ok;
}

static void pipe_ci_fill(PipeCI *c, const PipeKey *k) {
    memset(c, 0, sizeof *c);
    bool use_a2c = (k->samples > 1) ? (k->a2c != 0) : false;
    const char *a2c_e = ENV("LARECOMP_A2C");
    if (a2c_e) use_a2c = atoi(a2c_e) != 0;
    c->stages[0] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = k->prog->vs, .pName = "main"};
    c->stages[1] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = use_a2c && k->prog->fs_late ? k->prog->fs_late : k->prog->fs, .pName = "main"};
    if (fs_subgroup32()) {
        c->req32_fs = (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
            .requiredSubgroupSize = 32
        };
        c->stages[1].pNext = &c->req32_fs;
    }
    uint32_t na = 0, nb = 0, nd = 0;
    for (int i = 0; i < 32; i++)
        if (k->attr_mask & (1u << i)) c->va[na++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr_bind[i], k->attr_fmt[i], k->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(k->used_streams & (1u << s))) continue;
        bool inst = k->divisor[s] != 0;
        c->vb[nb++] = (VkVertexInputBindingDescription){s, k->stride[s], inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        if (inst && k->divisor[s] > 1 && g_vk.divisor) c->vd[nd++] = (VkVertexInputBindingDivisorDescriptionEXT){s, k->divisor[s]};
    }
    c->dvi = (VkPipelineVertexInputDivisorStateCreateInfoEXT){.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT,
                                                          .vertexBindingDivisorCount = nd, .pVertexBindingDivisors = c->vd};
    c->vi = (VkPipelineVertexInputStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .pNext = nd ? &c->dvi : NULL,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = c->vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = c->va};
    c->ia = (VkPipelineInputAssemblyStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = topo_class_base(k->topo_class)};
    if (k->restart)  // (a list topology can't take primitive restart: the strip of its class, the draw sets the same)
        c->ia.topology = k->topo_class == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        c->ia.primitiveRestartEnable = VK_TRUE;
    c->vs = (VkPipelineViewportStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    c->rs = (VkPipelineRasterizationStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = k->poly_fill ? VK_POLYGON_MODE_FILL : VK_POLYGON_MODE_LINE, .lineWidth = 1,
                                                 .depthClampEnable = k->depth_clamp ? VK_TRUE : VK_FALSE,
                                                 .depthBiasEnable = g_vkc.no_eds2};
    c->ms = (VkPipelineMultisampleStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples, .alphaToCoverageEnable = use_a2c ? VK_TRUE : VK_FALSE};
    c->ds = (VkPipelineDepthStencilStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (g_vkc.no_eds1) {
        const FixedState *f = &k->fx;
        if (!k->restart) c->ia.topology = (VkPrimitiveTopology)f->topo;  // (restart: its strip, above)
        c->ds.depthTestEnable = f->dt, c->ds.depthWriteEnable = f->dw, c->ds.depthCompareOp = (VkCompareOp)f->dop;
        c->ds.stencilTestEnable = f->sten;
        for (int s = 0; s < 2; s++) {
            VkStencilOpState *o = s ? &c->ds.back : &c->ds.front;
            o->failOp = (VkStencilOp)f->sop[s][0], o->passOp = (VkStencilOp)f->sop[s][1];
            o->depthFailOp = (VkStencilOp)f->sop[s][2], o->compareOp = (VkCompareOp)f->sop[s][3];
        }
        c->rs.cullMode = f->cull, c->rs.frontFace = (VkFrontFace)f->ff;
    }
    for (uint32_t i = 0; i < k->ncolor; i++) {
        c->cba[i].colorWriteMask = k->write_mask[i];
        if ((k->blend_on >> i) & 1) {
            const uint8_t *b = k->blend[i];
            c->cba[i].blendEnable = VK_TRUE;
            c->cba[i].srcColorBlendFactor = blend_factor(b[1]);
            c->cba[i].dstColorBlendFactor = blend_factor(b[2]);
            c->cba[i].srcAlphaBlendFactor = blend_factor(b[3]);
            c->cba[i].dstAlphaBlendFactor = blend_factor(b[4]);
            c->cba[i].colorBlendOp = blend_op(b[5]);
            c->cba[i].alphaBlendOp = blend_op(b[6]);
        }
    }
    c->cb = (VkPipelineColorBlendStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                  .attachmentCount = k->ncolor, .pAttachments = c->cba};
    uint32_t ndyn;
    const VkDynamicState *dyns = dyn_states(&ndyn);  // (Vulkan 1.2 without extended dynamic state: fewer)
    memcpy(c->dyn, dyns, ndyn * sizeof *dyns);
    if (g_vk.vids) c->dyn[ndyn++] = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT;
    c->dy = (VkPipelineDynamicStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                               .dynamicStateCount = ndyn, .pDynamicStates = c->dyn};
    c->ri = (VkPipelineRenderingCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->ncolor,
                                            .pColorAttachmentFormats = k->color, .depthAttachmentFormat = k->depth,
                                            .stencilAttachmentFormat = k->stencil};
    c->pi = (VkGraphicsPipelineCreateInfo){.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &c->ri, .stageCount = 2,
                                           .pStages = c->stages, .pVertexInputState = g_vk.vids ? NULL : &c->vi,
                                           .pInputAssemblyState = &c->ia, .pViewportState = &c->vs, .pRasterizationState = &c->rs,
                                           .pMultisampleState = &c->ms, .pDepthStencilState = &c->ds, .pColorBlendState = &c->cb,
                                           .pDynamicState = &c->dy, .layout = k->prog->layout};
}

static void build_pipeline(void *arg) {
    Pipe *pp = arg;
    PipeKey *k = &pp->k;
    PipeCI c;
    pipe_ci_fill(&c, k);
    uint64_t t0 = SDL_GetTicksNS();
    int bs = build_begin("pipeline vs/fs", k->prog->code[0] ? k->prog->code[0] : k->prog->prep_vh,
                         k->prog->code[1] ? k->prog->code[1] : k->prog->prep_fh);
    SLOW_FEEDBACK(c.pi);
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &c.pi, NULL, &pp->pipe);
    SLOW_FEEDBACK_HIT();
    build_end(bs);
    if (r != VK_SUCCESS) {
        static int n;
        if (n++ < 10) LOG("[vk] pipeline creation failed: %d\n", r);
    }
    atomic_fetch_add(&stat_compiles, 1);
    atomic_fetch_add(&builds_done, 1);
    atomic_store(&pp->state, r == VK_SUCCESS ? 1 : 2);
    slow_build(t0, "pipeline vs/fs", k->prog->code[0], k->prog->code[1]);
}

// Dynamic vertex input (g_vk.vids): the draw's layout, as build_pipeline would bake it in
static void set_vertex_input(VkCommandBuffer cb, const VtxLayout *l) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    VkVertexInputAttributeDescription2EXT va[32];
    VkVertexInputBindingDescription2EXT vb[17];
    uint32_t na = 0, nb = 0;
    for (int i = 0; i < 32; i++)
        if (l->attr_mask & (1u << i))
            va[na++] = (VkVertexInputAttributeDescription2EXT){.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .location = (uint32_t)i, .binding = l->attr_bind[i], .format = l->attr_fmt[i], .offset = l->attr_off[i]};
    for (uint32_t s = 0; s < 17; s++) {
        if (!(l->used_streams & (1u << s))) continue;
        bool inst = l->divisor[s] != 0;
        vb[nb++] = (VkVertexInputBindingDescription2EXT){.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
            .binding = s, .stride = l->stride[s], .inputRate = inst ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
            .divisor = inst ? l->divisor[s] : 1};
    }
    rec_CmdSetVertexInputEXT(cb, nb, vb, na, va);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// ================================================================== feedback snapshots (renderer.m resolve_feedback)
static NvnTexture *feedback_tex_t[NCTX][4];
static VkImageView feedback_view_t[NCTX][4];
static TexBackend *feedback_img_t[NCTX][4];  // (the snapshot behind each view: a slot in stencil mode takes its own view of it)
static int nfeedback_t[NCTX];
#define feedback_tex (feedback_tex_t[sw_ctx()])
#define feedback_view (feedback_view_t[sw_ctx()])
#define feedback_img (feedback_img_t[sw_ctx()])
#define nfeedback (nfeedback_t[sw_ctx()])
static uint64_t g_cur_fs_hash_t[NCTX];
#define g_cur_fs_hash (g_cur_fs_hash_t[sw_ctx()])
static NvnTexture *snap_tex[8];
static bool snap_ok[8];
static const char *snap_why[8];
static const char *snap_reason = "?";
static void snapshots_invalidate(NvnTexture *t) {
    for (int i = 0; i < 8; i++) if ((!t || snap_tex[i] == t) && snap_ok[i]) { snap_ok[i] = false; snap_why[i] = snap_reason; }
}

typedef struct { NvnTexture *key; TexBackend copy; } Snap;
static Snap snaps[16];
static int nsnaps;

static NvnTexture *shader_texture_object(int stage, ShaderTexture *st_) {
    int slot = ((int)st_->handle_offset - 0x20) / 8;
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    if (!h || !st.tpool || tic >= (uint32_t)st.tpool->count) return NULL;
    return ((NvnTexEntry *)st.tpool->entries)[tic].tex;
}

static bool is_attachment(NvnTexture *t) {
    if (!t) return false;
    if (t == st.rt.depth) return true;
    for (int i = 0; i < st.rt.ncolor; i++) if (t == st.rt.color[i]) return true;
    return false;
}

// A draw that can change its depth/stencil target's stencil: the test on, a write mask, and an operation other than
// KEEP on either face. The snapshots hold the stencil too (a slot in stencil mode reads it, texture_info), so such a
// draw ends them as a depth write does.
static bool draw_writes_stencil(void) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    if (!st.ds[3] || !((st.stencil_mask[0] | st.stencil_mask[1]) & 0xFF)) return false;
    uint32_t ops = st.ds[5] | st.ds[6] << 8 | st.ds[7] << 16;  // (two faces x fail / depth fail / pass, a nibble each)
    for (int k = 0; k < 6; k++) {
        uint32_t o = (ops >> (k * 4)) & 0xF;
        if (o >= 2 && o <= 8) return true;  // (stencil_op: anything else is KEEP)
    }
    return false;
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

static TexBackend *snapshot_image(NvnTexture *t, TexBackend *src) {
    Snap *s = NULL;
    for (int i = 0; i < nsnaps; i++) if (snaps[i].key == t) s = &snaps[i];
    if (!s) {
        if (nsnaps == 16) { static int rr; s = &snaps[rr++ & 15]; } else s = &snaps[nsnaps++];
        s->key = t;
    }
    TexBackend *d = &s->copy;
    if (d->image && (d->width != src->width || d->height != src->height || d->format != src->format)) {
        vk_views_forget(d);  // (resolution change)
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = src->format,
                                .extent = {src->width, src->height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = (VkSampleCountFlagBits)src->samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = src->format; d->width = src->width; d->height = src->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = src->samples; d->aspect = src->aspect; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

// (debug frames) a depth image's depth values as a greyscale PGM, stretched over its own min..max
static void debug_save_depth(TexBackend *b, const char *path) {
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || b->format != VK_FORMAT_D32_SFLOAT_S8_UINT) return;
    uint64_t n = (uint64_t)b->width * b->height;
    vk_end_pass();  // (no copies inside a render pass: debug_save_stencil)
    VkSpan out = vk_readback(n * 4);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                           .imageExtent = {b->width, b->height, 1}};
    rec_CmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    const float *d = out.ptr;
    float lo = 1e30f, hi = -1e30f;
    for (uint64_t i = 0; i < n; i++) { if (d[i] < lo) lo = d[i]; if (d[i] > hi) hi = d[i]; }
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5 %u %u 255\n", b->width, b->height);
    for (uint64_t i = 0; i < n; i++) fputc(hi > lo ? (int)((d[i] - lo) / (hi - lo) * 255.0f) : 0, f);
    fclose(f);
    LOG("[vk] saved %s (%ux%u depth %.6f..%.6f)\n", path, b->width, b->height, lo, hi);
}

// (debug frames) a depth / stencil image's stencil as a greyscale PGM of the raw values (0..255, not stretched): what
// a shader reading it through a STENCIL view gets. Scarlet / Violet's lighting keeps each pixel's shading model in
// bits 0..2 (models 4..7 take their own paths), bits 3..5 a second parameter, bit 6 "skip".
static void debug_save_stencil(TexBackend *b, const char *path) {
    if (!b || !b->image || !(b->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) || b->samples != 1) return;
    uint64_t n = (uint64_t)b->width * b->height;
    // (a copy inside a render pass is invalid: v0.0.20-24 recorded it in the middle of the lighting pass, Intel's driver
    // dropped it, and the file held the previous readback - a pass PNG's opaque black, 00 00 00 ff. The draw begins
    // its pass again, as after renderer_save_texture_tex.)
    vk_end_pass();
    VkSpan out = vk_readback((n + 3) & ~3ull);
    memset(out.ptr, 0x55, (size_t)n);  // (a copy that still doesn't land shows as 0x55, not as an old readback)
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1},
                           .imageExtent = {b->width, b->height, 1}};
    rec_CmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    const uint8_t *d = out.ptr;
    uint32_t hist[8] = {0};
    for (uint64_t i = 0; i < n; i++) hist[d[i] & 7]++;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P5 %u %u 255\n", b->width, b->height);
    fwrite(d, 1, (size_t)n, f);
    fclose(f);
    LOG("[vk] saved %s (%ux%u stencil; bits 0..2: %u %u %u %u %u %u %u %u)\n", path, b->width, b->height, hist[0], hist[1],
        hist[2], hist[3], hist[4], hist[5], hist[6], hist[7]);
}

// (debug) LARECOMP_COPY_TRACE=<frame>: draws' attachments / sampled textures, copies and snapshots of that frame ("[ct]")
static bool ct_on(void) {
    static const char *e = (const char *)1;
    if (e == (const char *)1) e = ENV("LARECOMP_COPY_TRACE");
    if (!e) return false;
    static uint64_t memo_f = ~0ull;
    static bool memo;
    uint64_t f = renderer_present_count();
    if (f == memo_f) return memo;
    memo_f = f;
    memo = false;
    for (const char *p = e; *p;) {
        if ((uint64_t)strtoll(p, (char **)&p, 10) == f) { memo = true; break; }
        if (*p == ',') p++; else break;
    }
    if (memo) LOG("[ct] ===== frame %llu\n", (unsigned long long)f);
    return memo;
}
static void ct_draw(Prog *pr, const char *what) {
    LOG("[ct] %s fs %016llx rt", what, (unsigned long long)g_cur_fs_hash);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) LOG(" c%d=%p/%#x", i, (void *)st.rt.color[i], st.rt.color[i]->format);
    if (st.rt.depth) LOG(" d=%p/%#x%s", (void *)st.rt.depth, st.rt.depth->format, st.ds[0] && st.ds[1] ? "W" : "");
    LOG(" cm %08x vp %.0f %.0f %.0f %.0f sc %d %d %d %d | tex", st.channel_mask, st.viewport[0][0], st.viewport[0][1], st.viewport[0][2],
        st.viewport[0][3], st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3]);
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < (infos[sg] ? infos[sg]->ntex : 0); i++) {
            NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
            if (t) LOG(" %s%p/%#x", sg ? "" : "v:", (void *)t, t->format);
        }
    LOG("\n");
}

// ================================================================== copy elision (depth copies)
// The game copies its main depth buffer whole, every frame, into textures that are then only sampled while the
// source keeps its depth (the water / transparency depth input), or that get their depth cleared right away
// (stencil kept). At 3x internal resolution each such copy is ~0.15 ms of pure bandwidth. Instead of copying, the
// destination becomes an alias: sampling it (and copying from it) reads the source's image. The alias ends before
// anything changes either side:
//   - the source is about to be overwritten whole (both aspects cleared): the two images swap - the destination
//     keeps the contents without a copy, the source is cleared in the destination's old image;
//   - the destination is about to be overwritten in some aspects (its depth cleared): only the other aspects are
//     copied (stencil, a quarter of the bytes);
//   - anything else (a draw writing depth / stencil, a partial clear, a copy or upload into either, the
//     destination bound as an attachment, a texture freed or reallocated): the real copy is made first.
// Only same-size, same-format, single-level, single-sample depth/stencil images, whole-image copies.
// LARECOMP_VK_COPY_OPT=0: always copy (the old behaviour).
typedef struct { NvnTexture *src, *dst; VkImage simg, dimg; } CopyAlias;
extern uint32_t vk_view_gen;
static CopyAlias aliases[8];
static int naliases;
static uint64_t stat_alias_made, stat_alias_renamed, stat_alias_partial, stat_alias_copied;

static int copy_opt(void) {
    static int on = -1;
    if (on < 0) { const char *e = ENV("LARECOMP_VK_COPY_OPT"); on = e && *e ? atoi(e) : 1; }
    return on;
}

// still valid: both textures keep the images the alias was made with (a resolution change reallocates them)
static bool alias_valid(const CopyAlias *a) {
    const TexBackend *s = a->src->backend, *d = a->dst->backend;
    return s && d && s->image == a->simg && d->image == a->dimg;
}

static void alias_drop(int i) {
    vk_tex_vgen_bump(aliases[i].dst->backend);  // (the view memos: the destination's views change)
    aliases[i] = aliases[--naliases];  // (no view is destroyed: vk_view_gen - the reused sets - stays)
}

// the real copy of the aspects asked for, into the destination's own image
static void alias_copy(CopyAlias *a, VkImageAspectFlags aspects, const char *why) {
    TexBackend *s = a->src->backend, *d = a->dst->backend;
    aspects &= d->aspect;
    if (!aspects) return;
    vk_end_pass();
    st.rt_dirty = true;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    gp_mark(cb, "copy texture (deferred, %s%s) %ux%u", why, aspects == d->aspect ? "" : ", stencil only", a->dst->width, a->dst->height);
    VkImageCopy c = {.srcSubresource = {aspects, 0, 0, 1}, .dstSubresource = {aspects, 0, 0, 1}, .extent = {d->width, d->height, 1}};
    rec_CmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    if (aspects == d->aspect) stat_alias_copied++; else stat_alias_partial++;
    if (ct_on()) LOG("[ct] alias copy %p -> %p aspects %#x (%s)\n", (void *)a->src, (void *)a->dst, aspects, why);
}

// t is about to be written; full: the aspects of it the write overwrites whole (0: unknown / partial)
static void alias_before_write(NvnTexture *t, VkImageAspectFlags full, const char *why) {
    if (!naliases || !t) return;
    for (int i = naliases - 1; i >= 0; i--) {
        CopyAlias *a = &aliases[i];
        if (!alias_valid(a)) { alias_drop(i); continue; }
        if (a->dst != t) continue;
        // (LARECOMP_VK_COPY_OPT=2: all aspects even then - for comparing the stencil-only copy's cost)
        alias_copy(a, ((TexBackend *)t->backend)->aspect & (copy_opt() == 2 ? ~0u : ~full), why);
        alias_drop(i);
    }
    NvnTexture *heir = NULL;  // t overwritten whole: one destination can take its image instead of a copy
    if (t->backend && (((TexBackend *)t->backend)->aspect & ~full) == 0)
        for (int i = 0; i < naliases && !heir; i++) if (aliases[i].src == t) heir = aliases[i].dst;
    for (int i = naliases - 1; i >= 0; i--) {
        CopyAlias *a = &aliases[i];
        if (a->src != t || a->dst == heir) continue;
        alias_copy(a, ((TexBackend *)a->dst->backend)->aspect, why);
        alias_drop(i);
    }
    int rename = -1;
    for (int i = 0; i < naliases; i++) if (heir && aliases[i].src == t && aliases[i].dst == heir) rename = i;
    if (rename >= 0) {
        CopyAlias *a = &aliases[rename];
        TexBackend *s = a->src->backend, *d = a->dst->backend;
        vk_end_pass();
        st.rt_dirty = true;
        VkImage im = s->image; s->image = d->image; d->image = im;
        VmaAllocation m = s->mem; s->mem = d->mem; d->mem = m;
        vk_views_forget(s);
        vk_views_forget(d);
        stat_alias_renamed++;
        if (ct_on()) LOG("[ct] alias rename %p -> %p (%s)\n", (void *)a->src, (void *)a->dst, why);
        alias_drop(rename);
    }
}

// the texture whose image holds t's contents (t itself unless it is an alias)
static NvnTexture *alias_read(NvnTexture *t) {
    for (int i = 0; i < naliases; i++)
        if (aliases[i].dst == t) {
            if (alias_valid(&aliases[i])) return aliases[i].src;
            alias_drop(i);
            return t;
        }
    return t;
}

// copy_texture: a whole-image copy that can be an alias instead (true: done)
static bool alias_make(NvnTexture *src, NvnTexture *dst, TexBackend *s, TexBackend *d) {
    if (!copy_opt() || src == dst) return false;
    if (!(s->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) return false;
    if (s->format != d->format || s->width != d->width || s->height != d->height || s->aspect != d->aspect || s->levels != 1 ||
        d->levels != 1 || s->layers != 1 || d->layers != 1 || s->depth != 1 || d->depth != 1 || s->samples != 1 || d->samples != 1 ||
        s->base_type != d->base_type || s->presented || d->presented || !s->image || !d->image || !s->gpu_owned || !d->gpu_owned)
        return false;
    // the destination's previous contents (or alias) are replaced whole; a destination that is itself the source of
    // aliases gets those resolved first
    alias_before_write(dst, d->aspect, "copy dst");
    src = alias_read(src);  // (a copy of an alias: of its source)
    s = src->backend;
    if (src == dst) return true;  // (copied back onto itself)
    if (naliases == (int)(sizeof aliases / sizeof aliases[0])) return false;
    aliases[naliases++] = (CopyAlias){src, dst, s->image, d->image};
    vk_tex_vgen_bump(d);  // (no view is destroyed: vk_view_gen - the reused sets - stays)
    stat_alias_made++;
    return true;
}

// textures.c: t is being freed - an alias source hands its contents over first
void vk_alias_texture_destroyed(NvnTexture *t) {
    if (naliases) alias_before_write(t, 0, "freed");
}

// HDR (vk_hdr_encode): the game's picture before its HUD. Pokémon Z-A tone-maps its frame into an image of its own,
// copies that opaque - edge anti-aliased on the way, so not the same pixels - into the image it presents, and draws the
// HUD over it with blending. After an opaque draw into an image that has been presented, that image is copied
// (hdr_snapshot, two: the game alternates two such images): what differs from the copy when it is shown is the HUD.
static struct { NvnTexture *shown; TexBackend img; } hdr_snaps[2];
static TexBackend *hdr_game;  // (vk_present_draw: this frame's game image, its place in the window, uv crop and flip)
static float hdr_place[9];
static void hdr_snapshot(NvnTexture *t);
static TexBackend *hdr_snap_of(TexBackend *shown) {
    for (int i = 0; i < 2; i++)
        if (hdr_snaps[i].shown && hdr_snaps[i].shown->backend == shown && hdr_snaps[i].img.image) return &hdr_snaps[i].img;
    return NULL;
}
static NvnTexture *mv_depth;  // (mv_note_draw below)
static uint64_t mv_frame[2];
static float mv_vp[2][16];
// ---- temporal anti-aliasing, AMD FSR 2 (fsr2_copy below): this frame's jitter, in the scene's pixels, while FSR 2
// resolved the frame before (a frame it doesn't resolve is never shown jittered: at most the first one)
static float fsr2_jx, fsr2_jy;
static uint64_t fsr2_jit_frame = ~0ull, fsr2_ok_frame = ~0ull;
static NvnTexture *fsr2_pic;  // the picture resolved last (fsr2_copy: its interface's first draw starts the early resolve)
static uint32_t fsr2_scene_w, fsr2_scene_h;  // (the scene depth's size: mv_note_draw)
static bool fsr2_wanted(void) {  // (F1 > Video > Temporal anti-aliasing; LARECOMP_FSR2=0 / 1 overrides it)
    static int env = -2;
    if (env == -2) env = ENV("LARECOMP_FSR2") ? atoi(ENV("LARECOMP_FSR2")) != 0 : -1;
    // (DLAA when it came up at the start; chosen but not up - a change waiting for the restart, no RTX GPU: the game's)
    return env >= 0 ? env : g_settings.taa == TAA_FSR2 || (g_settings.taa == TAA_DLAA && vk_dlaa_available());
}
static void fsr2_frame_jitter(void) {
    extern uint64_t vk_frame_counter(void);
    uint64_t f = vk_frame_counter();
    if (f == fsr2_jit_frame) return;
    fsr2_jit_frame = f;
    fsr2_jx = fsr2_jy = 0;
    if (!fsr2_wanted() || !fsr2_scene_w || fsr2_ok_frame + 1 != f) return;
    static uint32_t index;
    vk_fsr2_jitter(index++, fsr2_scene_w, fsr2_scene_w, &fsr2_jx, &fsr2_jy);
}
// The 3D scene's passes take the jitter: those drawing with the scene's depth (a scalable, non-square depth target
// at its level 0) - not the shadow maps, the depth pyramid's levels, the full-screen passes or the HUD
static bool fsr2_scene_pass(void) {
    NvnTexture *d = st.rt.depth;
    TexBackend *b = d ? d->backend : NULL;
    if (!b || !b->scalable || d->width == d->height) return false;
    return !(st.rt.has_dview && (st.rt.dview.setmask & 1) && st.rt.dview.base_level);
}
uint32_t nvn_window_origin(void);

// textures.c: t is being freed - the executor's state stops naming it. The render targets stay bound across command
// lists, and a freed one was still in st.rt: a later list's pass began on it, which made it a new backend and put the
// freed texture back on the live list (textures.c g_live); the next texture freed then read that entry's garbage -
// a crash in live_remove when the game freed textures walking into a new area of the city.
void vk_exec_texture_destroyed(NvnTexture *t) {
    bool rt = false;
    for (int i = 0; i < 8; i++) if (st.rt.color[i] == t) st.rt.color[i] = NULL, rt = true;
    if (st.rt.depth == t) st.rt.depth = NULL, rt = true;
    if (rt) st.rt_dirty = true;  // (an open pass keeps its views: they are destroyed once the GPU is done)
    for (int c = 0; c < NCTX; c++) for (int i = 0; i < 4; i++) if (feedback_tex_t[c][i] == t) feedback_tex_t[c][i] = NULL;
    snapshots_invalidate(t);
    for (int i = 0; i < 8; i++) if (snap_tex[i] == t) snap_tex[i] = NULL;
    for (int i = 0; i < nsnaps; i++) if (snaps[i].key == t) snaps[i].key = NULL;
    for (int i = 0; i < 2; i++) if (hdr_snaps[i].shown == t) hdr_snaps[i].shown = NULL;
    if (hdr_game && hdr_game == t->backend) hdr_game = NULL;
    if (mv_depth == t) mv_depth = NULL;
    if (fsr2_pic == t) fsr2_pic = NULL;
    gtao_forget(t);
}
void vk_alias_reset(void) { naliases = 0; }  // (reboot: every texture goes)

// ---- FSR 2 (fsr2_frame): the scene camera per frame - the view-projection in vertex constant buffer 5 of the
// G-buffer draws (4+ colour targets) - and their depth target.
static void mv_note_draw_(uint64_t f);
static void mv_note_draw(Prog *pr) {
    extern uint64_t vk_frame_counter(void);
    uint64_t f = vk_frame_counter();
    int n = 0;
    for (int i = 0; i < 8; i++) n += st.rt.color[i] != NULL;
    // (the shaders' constant buffer k is the stage's binding k - 3: cbuf_info. The G-buffer vertex shaders' clip
    // position is cb5[0..3] (rows) x their model matrix, cb10)
    if (mv_frame[0] == f || !st.rt.depth || !st.ubuf[0][2].addr || st.ubuf[0][2].size < 64) return;
    if (n < 4) return;
    SWL_AT(3459);  // (shadow workers: the frame's camera, once)
    mv_note_draw_(f);
    SWU();
}
static void mv_note_draw_(uint64_t f) {
    if (mv_frame[0] == f) return;
    const float *m = (const float *)(uintptr_t)st.ubuf[0][2].addr;
    for (int i = 0; i < 16; i++) if (!isfinite(m[i])) return;
    if (m[12] == 0 && m[13] == 0 && m[14] == 0) return;  // (perspective: w depends on the position)
    float a0 = m[0] * m[5] - m[1] * m[4], a1 = m[0] * m[6] - m[2] * m[4], a2 = m[0] * m[7] - m[3] * m[4], a3 = m[1] * m[6] - m[2] * m[5],
          a4 = m[1] * m[7] - m[3] * m[5], a5 = m[2] * m[7] - m[3] * m[6], b0 = m[8] * m[13] - m[9] * m[12], b1 = m[8] * m[14] - m[10] * m[12],
          b2 = m[8] * m[15] - m[11] * m[12], b3 = m[9] * m[14] - m[10] * m[13], b4 = m[9] * m[15] - m[11] * m[13], b5 = m[10] * m[15] - m[11] * m[14];
    if (fabsf(a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0) < 1e-9f) return;
    mv_frame[1] = mv_frame[0], memcpy(mv_vp[1], mv_vp[0], sizeof mv_vp[0]);
    mv_frame[0] = f, memcpy(mv_vp[0], m, sizeof mv_vp[0]);
    mv_depth = st.rt.depth;
    if (mv_depth->backend) fsr2_scene_w = ((TexBackend *)mv_depth->backend)->width, fsr2_scene_h = ((TexBackend *)mv_depth->backend)->height;
}

// (renderer_stats, once per stats interval) copies made aliases / renamed / partly or fully copied later
void vk_alias_stats(void) {
    static int n;
    if (!stat_alias_made || (n++ > 5 && n % 30)) return;
    LOG("[vk] copy alias: %llu made, %llu renamed, %llu stencil-only, %llu copied later (live %d)\n",
        (unsigned long long)stat_alias_made, (unsigned long long)stat_alias_renamed, (unsigned long long)stat_alias_partial,
        (unsigned long long)stat_alias_copied, naliases);
    stat_alias_made = stat_alias_renamed = stat_alias_partial = stat_alias_copied = 0;
}

// draws: attachments written (or a destination bound at all) end the aliases involved
static void alias_draw_check_(void);
static void alias_draw_check(void) {
    if (!naliases) return;
    SWL_AT(3492);
    alias_draw_check_();
    SWU();
}
static void alias_draw_check_(void) {
    if (!naliases) return;
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i]) alias_before_write(st.rt.color[i], 0, "color target");
    NvnTexture *d = st.rt.depth;
    if (!d) return;
    bool writes = (st.ds[0] && st.ds[1]) || (st.ds[3] && ((st.stencil_mask[0] | st.stencil_mask[1]) & 0xFF));
    bool is_dst = false;
    for (int i = 0; i < naliases; i++) is_dst |= aliases[i].dst == d;
    if (writes || is_dst) alias_before_write(d, 0, writes ? "depth write" : "dst bound");
}

static void resolve_feedback_(Prog *pr);
static void resolve_feedback(Prog *pr) {
    nfeedback = 0;
    bool any = false;  // (an attachment sampled: rare - the snapshots are shared)
    ShaderInfo *in2[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2 && !any; sg++)
        for (int i = 0; i < in2[sg]->ntex && !any; i++) any = is_attachment(shader_texture_object(sg, &in2[sg]->tex[i]));
    if (!any) return;
    SWL_AT(3516);
    resolve_feedback_(pr);
    SWU();
}
static void resolve_feedback_(Prog *pr) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    nfeedback = 0;
    static int off = -1;
    if (off < 0) off = ENV("LARECOMP_NO_FEEDBACK") != NULL;
    if (off) return;
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < infos[sg]->ntex; i++) {
            NvnTexture *t = shader_texture_object(sg, &infos[sg]->tex[i]);
            if (!is_attachment(t)) continue;
            // Drawn into one mip level above the base, reading the texture's other levels: a mip chain being built,
            // no feedback loop - sampled in place. Pokémon Z-A builds its depth pyramid this way (each level from the
            // one above, the whole texture sampled at an explicit level); the snapshot held level 0 only and was
            // reused for the following levels, so most of the pyramid came out empty or stale, and the game's GPU
            // occlusion culling, which reads it, dropped visible objects: tables, statues, trees and stalls vanishing
            // and coming back with the camera.
            if (t == st.rt.depth && st.rt.has_dview && (st.rt.dview.setmask & 1) && st.rt.dview.base_level) continue;
            {
                bool mip = false;
                for (int k = 0; k < st.rt.ncolor; k++)
                    if (t == st.rt.color[k] && st.rt.has_cview[k] && (st.rt.cview[k].setmask & 1) && st.rt.cview[k].base_level) mip = true;
                if (mip) continue;
            }
            bool have = false;
            for (int k = 0; k < nfeedback; k++) have |= feedback_tex[k] == t;
            if (have || nfeedback == 4) continue;
            static int nosnap = -1;
            if (nosnap < 0) nosnap = ENV("LARECOMP_NO_SNAPSHOT_REUSE") != NULL;
            int si = -1;
            for (int k = 0; k < 8; k++) if (snap_tex[k] == t) si = k;
            bool self_write = t == st.rt.depth && ((st.ds[0] && st.ds[1]) || draw_writes_stencil());
            TexBackend *src = vk_tex(t, true);
            Snap *existing = NULL;
            for (int k = 0; k < nsnaps; k++) if (snaps[k].key == t && snaps[k].copy.image) existing = &snaps[k];
            VkImageAspectFlags sa = src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : src->aspect;
            static const VkComponentMapping id = {0};
            if (!nosnap && si >= 0 && (snap_ok[si] || self_write) && existing) {
                feedback_tex[nfeedback] = t;
                feedback_img[nfeedback] = &existing->copy;
                feedback_view[nfeedback++] = vk_image_view(&existing->copy, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
                continue;
            }
            stat_feedback_copies++;
            if (ct_on()) LOG("[ct] snapshot %p %ux%u (was %s, si %d)\n", (void *)t, src->width, src->height, si >= 0 ? snap_why[si] : "new", si);
            vk_end_pass();
            TexBackend *dst = snapshot_image(t, src);
            VkCommandBuffer cb = vk_cmd();
            vk_full_barrier(cb);
            gp_mark(cb, "feedback snapshot %ux%u fmt %d", src->width, src->height, src->format);
            VkImageCopy rg = {.srcSubresource = {src->aspect, 0, 0, 1}, .dstSubresource = {src->aspect, 0, 0, 1},
                              .extent = {src->width, src->height, 1}};
            rec_CmdCopyImage(cb, src->image, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
            if (g_debug_frame && (src->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
                static int nd;
                char path[128];
                snprintf(path, sizeof path, "build/fbdepth_%03d_src.pgm", nd);
                debug_save_depth(src, path);
                snprintf(path, sizeof path, "build/fbdepth_%03d_snap.pgm", nd++);
                debug_save_depth(dst, path);
                cb = vk_cmd();
            }
            feedback_tex[nfeedback] = t;
            feedback_img[nfeedback] = dst;
            feedback_view[nfeedback++] = vk_image_view(dst, VK_IMAGE_VIEW_TYPE_2D, src->format, 0, 1, 0, 1, id, sa);
            if (si < 0) { static int rr; si = rr++ & 7; snap_tex[si] = t; }
            snap_ok[si] = true;
            st.rt_dirty = true;
        }
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// ================================================================== resource binding
static VkDescriptorBufferInfo ubo_span(VkSpan s, uint64_t size) {
    uint64_t max = g_vk.props.limits.maxUniformBufferRange;
    return (VkDescriptorBufferInfo){s.buf, s.off, size > max ? max : size};
}

// Lean constant blocks: the scale table (66 rows, 1 KB) and the driver constants (up to 2 KB) were zero-filled in
// full and compared in full per stage and draw, although only a few rows / entries carry anything. These build and
// compare only those (the scale table's rows 0..ntex and 65; the driver constants' storage buffer entries below the
// size the shader reads) and write the same bytes as before. LARECOMP_VK_NO_CBUF_LEAN=1: the full blocks, as before.
static int cbuf_lean_v = -1;
static bool cbuf_lean(void) {
    if (cbuf_lean_v < 0) cbuf_lean_v = getenv("LARECOMP_VK_NO_CBUF_LEAN") == NULL;
    return cbuf_lean_v;
}
// The scale table of a graphics stage: rows 0..nt and 65 of rs are set, the rows between are zero (not read from rs).
static VkSpan rscale_dedupe(int sg, const float rs[66][4], int nt) {
    static struct { uint64_t seq; int nt; float r[66][4]; VkSpan span; } last_t[NCTX][2];
    __typeof__(last_t[0][0]) *last = last_t[sw_ctx()];
    uint64_t seq = vk_pending_seq();
    size_t used = (size_t)(nt + 1) * 16;
    if (last[sg].seq == seq && last[sg].nt == nt && !memcmp(last[sg].r, rs, used) && !memcmp(last[sg].r[65], rs[65], 16)) {
        prof_cb_reused++;
        return last[sg].span;
    }
    VkSpan s = vk_transient(66 * 16, 256);
    memcpy(s.ptr, rs, used);
    if (nt < 64) memset((uint8_t *)s.ptr + used, 0, (size_t)(64 - nt) * 16);
    memcpy((uint8_t *)s.ptr + 65 * 16, rs[65], 16);
    last[sg].seq = seq; last[sg].nt = nt; memcpy(last[sg].r, rs, used); memcpy(last[sg].r[65], rs[65], 16); last[sg].span = s;
    prof_cb_copied++;
    return s;
}
// ---- storage buffers the game rewrites one frame ahead (overlap; renderer.m snap_storage)
// The game's tiled light / exposure lists: each frame its CPU memcpys the list for the frame after this one into the
// buffer the frame being drawn reads (7281e33500+32640 and 7281e3bd00+2040, alternating by frame), which the Switch's
// fence made safe. With the fence deferred that copy lands while the GPU still reads the list: screen-aligned tiles
// without shadow, a dark saturated picture when the exposure program (72a8da6000) reads it - worse the slower the GPU.
// Other storage data (the skinning program 72a8da0000's outputs, read in place by the exposure program and by draws
// binding them as vertex data from an offset into the pool: BowWow's body) is only ever written by the GPU and must
// stay where the game put it.
// Which is which is decided per memory range, with no wait and no probe of the programs (renderer.m 660bead). A range
// a draw reads as storage is shader-consumed game data (the lists): COPY for good - every binder of it, the compute
// dispatches that read or modify it too, works on a copy taken at translation, refreshed whenever its data changes
// (hash) and otherwise kept, so the GPU's own modifications stay. The CPU's rewrite can't be told from the data:
// standing still it rewrites the same bytes, and the boxes come from the GPU reading the list while it is rewritten
// (an M6 at 4x in the woods), so no hash decides this case - and on the title screen a probe would have settled the
// lists in place for good. A range only compute slots bind gets a one-frame PROBE on its first use in a level (all
// binders on the copy, the GPU can't touch the original) and its next use looks at the original: changed, the CPU
// rewrites it - COPY; unchanged, a GPU output or static (the skinning outputs, which draws then bind as vertex data
// from an offset into the pool) - IN PLACE for good, probed again when the level changes or after 10 s unused. With
// the write watch below (the default where the heap import allows it) a compute-only range starts in place and is
// copied from the first CPU store into it instead - the probe is the fallback; a copied range is never demoted. Each
// range's decision is logged once ("[snap] ...", the small per-object ranges capped). ~100 KB copied and hashed per
// frame in the field.
// Where the copies live: shaders address storage by guest address through the heap import (one buffer, lazy chunks,
// gfast's copy of the bound ranges), which covers only the heap the game set up (gmem_heap_size; Metal has the whole
// range in one buffer and uses its top). So the ring sits right after the game's heap and gmem_heap_size includes it:
// the one-buffer import takes it in, lazy mode imports its chunk when a copy is first bound (gmem_chunk_bind touches
// the effective addresses), gfast copies from it. Windows commits it (a userptr import needs committed pages).
// The ring is two halves: copies fill one; when it is full the submission pending then is noted for it and the other
// half is taken, after waiting for the submission noted when that half was left - so its freed slots are idle. Live
// copies stay where they are across switches (the allocator walks past them; rs_copy_make), so the GPU's own writes
// into a copy survive; a copy unused for 10 s is released. At ~100 KB a frame that wait is long satisfied; a half
// filling within one submission (not a real case) binds the original for that draw.
// LARECOMP_OVERLAP_SNAP=0: off (everything in place, as before: shows the race).
#define SNAP_HALF (8ull << 20)
#define SNAP_RING (2 * SNAP_HALF)
enum { RS_PROBE, RS_COPY, RS_PLACE };
typedef struct { uint64_t addr, size, copy, pristine, hash, frame; uint32_t gen, level, wgen; uint8_t mode; bool cs_bound, draw_bound; } RangeState;  // (wgen: watch_check; pristine: rs_copy_make)
static RangeState rs_tab[512];
static uint64_t rs_store_frame[512];  // (watch: the frame of the previous CPU write seen on a range's own pages)
static int nrs;
static int snap_state = -1;        // -1 not decided (the heap not set up yet), 0 off, 1 on
static uint64_t snap_base;         // heap offset of the ring (on)
static uint64_t snap_off;          // used bytes of the current half
static int snap_half;              // the half being filled
static uint64_t snap_half_seq[2];  // the submission pending when each half was last left (its copies may be in use until then)
static uint32_t snap_gen = 1;      // (bumped at a half switch: older copies are never bound again)
static uint32_t rs_level_gen = 1;
static Range sb_eff_t[NCTX][6][16];        // the current draw's / dispatch's storage table with the copies' addresses (snap_apply)
#define sb_eff (sb_eff_t[sw_ctx()])
static bool sb_eff_on_t[NCTX];
#define sb_eff_on (sb_eff_on_t[sw_ctx()])
static inline const Range *sbufs(int stage) { return sb_eff_on ? sb_eff[stage] : st.sbuf[stage]; }

static uint64_t range_hash(uint64_t addr, uint64_t size) {
    const uint64_t *w = (const uint64_t *)(uintptr_t)addr;
    uint64_t h = 1469598103934665603ull, n = size / 8;
    for (uint64_t i = 0; i < n; i++) h = (h ^ w[i]) * 1099511628211ull;
    return h;
}
static void snap_ring_import_tail(void);  // (with the lazy chunks below)
// Decided once the game's heap is set up (SetHeapSize), which is before anything is drawn: the ring follows the heap.
static uint64_t snap_org(void) { return atomic_load(&gwin_done) && gwin_.phys ? gwin_.base : HEAP_BASE; }  // (snap_base / the ring: offsets from it)
static bool snap_on(void) {
    if (snap_state >= 0) return snap_state;
    // (a game on physical memory - Scarlet / Violet, Z-A: no heap - has its ring past its pool: gwin_pick. Not picked
    // here: the window is picked at its first real use, once the pool is mapped)
    if (!atomic_load(&gwin_done)) return false;  // (not decided yet)
    bool phys = gwin_.phys;
    uint64_t heap = phys ? gwin_ring : kernel_heap_size();
    if (!heap) {
        if (!phys) return false;  // (not decided yet)
        snap_state = 0;
        LOG("[vk] storage snapshots off: no ring in the guest memory window\n");
        return false;
    }
    const char *e = getenv("LARECOMP_OVERLAP_SNAP");
    snap_state = (!e || atoi(e) != 0) && overlap_on();
    if (snap_state) {
        snap_base = (heap + 0xFFFF) & ~0xFFFFull;  // (minImportedHostPointerAlignment: 64 KB at most)
        if (!phys && snap_base + SNAP_RING > HEAP_MAX) {
            snap_state = 0;
            LOG("[vk] storage snapshots off: the game's heap fills the range\n");
        } else {
            plat_commit(snap_org() + snap_base, SNAP_RING);  // (Windows: backed before the import pins it)
            LOG("[vk] storage buffers the CPU rewrites: copies at %llx (%llu MB ring after the game's %s)\n",
                (unsigned long long)(snap_org() + snap_base), (unsigned long long)(SNAP_RING >> 20), phys ? "pool" : "heap");
            snap_ring_import_tail();  // (lazy chunks: one imported short of the ring already is imported again over it)
        }
    }
    return snap_state;
}
static uint64_t snap_ring_end(void) { return !(atomic_load(&gwin_done) && gwin_.phys) && snap_on() ? snap_base + SNAP_RING : 0; }  // (heap offset; 0: no ring)
static void snap_reset(void) {  // (vk_exec_reset: a reboot - the heap is set up again, its pages replaced)
    snap_state = -1;
    snap_base = snap_off = 0;
    snap_half = 0;
    snap_half_seq[0] = snap_half_seq[1] = 0;
    snap_gen++;
    nrs = 0;
    sb_eff_on = false;
}
static void rs_level_tick(void) {  // (a new level: every range is probed again on its next use)
    extern const char *game_level_name(void);
    static char last[64];
    const char *lv = game_level_name();
    if (lv && strncmp(lv, last, sizeof last - 1)) { snprintf(last, sizeof last, "%s", lv); rs_level_gen++; }
}
static void rs_log(RangeState *e, const char *what, bool cs_slot, uint64_t f) {  // (rare: a new range, a decision)
    static int n;
    if (e->size >= 2040 ? n < 100000 : n++ < 200)  // (the small per-object ranges are many: capped)
        LOG("[snap] %llx+%llu %s (%s bind, frame %llu)\n", (unsigned long long)e->addr, (unsigned long long)e->size, what, cs_slot ? "compute" : "draw", (unsigned long long)f);
}
// A copy lives until its entry remakes it (the original changed) or releases it (unused for 10 s): a half switch no
// longer remakes every copy from its original, which lost the GPU's own writes into a copy - the exposure program's
// into the lists - for a frame (renderer.m 1ea4b16). The allocator moves forward through the current half and walks
// past live copies; a slot freed by a remake (the GPU may still read it) is reused only after the half is left and
// taken again, which waits for the submission pending when it was left - the same guarantee as before.
// The offset in the current half where `need` bytes fit from `at` on, skipping live copies (with their pristine
// images); ~0 when none
static uint64_t rs_ring_fit(uint64_t at, uint64_t need) {
    uint64_t half_lo = snap_org() + snap_base + (uint64_t)snap_half * SNAP_HALF;
    for (bool moved = true; moved && at + need <= SNAP_HALF;) {
        moved = false;
        for (int i = 0; i < nrs; i++) {
            const RangeState *o = &rs_tab[i];
            if (!o->copy || o->copy < half_lo || o->copy >= half_lo + SNAP_HALF) continue;
            uint64_t olo = o->copy - half_lo, ohi = (olo + o->size * (o->pristine ? 2 : 1) + 255) & ~255ull;
            if (at < ohi && at + need > olo) { at = ohi; moved = true; }
        }
    }
    return at + need <= SNAP_HALF ? at : ~0ull;
}
// `need` bytes (256-aligned) of the ring past the live copies: their guest address, or 0 when none can be taken now
// (see the half switch above). A copy being remade is skipped too: the merge in rs_copy_make reads it.
static uint64_t ring_take(uint64_t need) {
    uint64_t at = rs_ring_fit((snap_off + 255) & ~255ull, need);
    if (at == ~0ull) {  // this half is full: the other one, once the GPU is done with what it held
        int other = snap_half ^ 1;
        uint64_t seq = snap_half_seq[other], done = vk_completed_seq();
        if (seq > done) {
            if (seq > renderer_issued_seq()) return 0;  // (still being recorded: can't be waited for - in place this once)
            wait_why = "storage snapshot ring";
            vk_wait_seq(seq);
        }
        snap_half_seq[snap_half] = vk_pending_seq();
        snap_half = other;
        snap_off = 0;
        at = rs_ring_fit(0, need);
        if (at == ~0ull) {  // (the half is all live copies: not in practice - 512 entries, most far smaller)
            static int warned;
            if (warned++ < 3) LOG("[snap] copy ring full: %llu bytes in place\n", (unsigned long long)need);
            return 0;
        }
    }
    snap_off = at + need;
    return snap_org() + snap_base + (uint64_t)snap_half * SNAP_HALF + at;
}
// A copy a compute dispatch binds carries the GPU's own writes into it (the exposure program 72a8da6000 keeps its
// adaptation state in the 2040-byte exposure list it reads and writes; the lists' tile data beside it is the CPU's).
// When the CPU changes the original the copy is remade from it - and the GPU's state went back to whatever the
// original held when the copy was first taken: the screen at half brightness for one frame (the path by the Face
// Shrine; two frames three apart, one per list buffer; renderer.m 69a3524). So such a copy keeps a pristine image of
// what it was copied from right behind it, and a remake merges: a byte the GPU changed (copy != pristine) that the
// CPU didn't (original == pristine) keeps the GPU's value; everything else is the original's, the CPU's write winning
// where both changed (its write is the later one, as in place). Draw-only copies have no GPU state and no image.
// A copy of e's range in the ring: its guest address, or 0 when none can be made now
static uint64_t rs_copy_make(RangeState *e, uint64_t h) {
    bool keep = e->cs_bound;
    uint64_t old = e->copy, old_p = e->pristine;
    uint64_t at = ring_take((e->size * (keep ? 2 : 1) + 255) & ~255ull);
    if (!at) return 0;
    uint8_t *dst = (uint8_t *)(uintptr_t)at;
    const uint8_t *src = (const uint8_t *)(uintptr_t)e->addr;
    memcpy(dst, src, e->size);
    if (keep && old && old_p) {
        const uint8_t *oc = (const uint8_t *)(uintptr_t)old, *op = (const uint8_t *)(uintptr_t)old_p;
        uint64_t kept = 0;
        for (uint64_t i = 0; i < e->size; i++) if (oc[i] != op[i] && src[i] == op[i]) { dst[i] = oc[i]; kept++; }
        if (kept) {
            static int n;
            extern uint64_t renderer_exec_frame(void);
            if (n++ < 20 || getenv("LARECOMP_LOG_BINDS"))
                LOG("[snap] %llx+%llu remade: %llu bytes of the GPU's kept (frame %llu)\n", (unsigned long long)e->addr,
                    (unsigned long long)e->size, (unsigned long long)kept, (unsigned long long)renderer_exec_frame());
        }
    }
    e->copy = at;
    if (keep) { e->pristine = at + e->size; memcpy((void *)(uintptr_t)e->pristine, src, e->size); }
    else e->pristine = 0;
    e->hash = h, e->gen = snap_gen;
    stat_snap_bytes += e->size;
    stat_snap_copies++;
    return e->copy;
}
static void rs_release_unused(uint64_t f) {  // (every 600 frames) copies of entries unbound for 10 s go back to the ring
    static uint64_t last;
    if (f - last < 600) return;
    last = f;
    for (int i = 0; i < nrs; i++) if (rs_tab[i].copy && f - rs_tab[i].frame > 600) rs_tab[i].copy = rs_tab[i].pristine = 0;
}
// Write watch (compute-only ranges; renderer.m cf98fff / c3580a8). The CPU's rewrite of a range can't be seen in its
// bytes (the light lists are rewritten with the same bytes while the view stands still, and the race is the GPU reading
// them mid-rewrite), so a range bound only by compute slots starts in place and the OS's page write tracking tells,
// at its first bind of each frame, whether the CPU wrote any of its pages since the last look: then it is copied from
// then on. The tracking is the vertex cache's and the texture upload checks' (plat_written_pages, the shared page
// generations vc_page_gen): Windows GetWriteWatch over the guest window reserved with MEM_WRITE_WATCH, Linux the
// userfaultfd asynchronous write-protect bits read with PAGEMAP_SCAN. Neither changes a page's protection, and
// neither sees the GPU's own writes (DMA sets no dirty bit): a GPU output stays in place. The Metal backend
// write-protects the pages and catches the store in the fault handler instead; on Windows that (VirtualProtect on
// pages imported with VK_EXT_external_memory_host, NVIDIA) left the GPU reading wrong data - the picture at half
// brightness, grass and HUD digits missing - so the Vulkan backend never changes the protection of imported memory.
// Where the OS gives no tracking (Linux over a userptr or dma-buf heap: the tracking is off there, a kernel without
// PAGEMAP_SCAN, a Windows window reserved without write watch) the one-frame hash probe is the fallback, logged.
// Page granularity (4 KB; 16 KB and 64 KB kernels on arm64 Linux as the kernel reports): a store into a neighbour on
// the same page marks the range too (the lists share a page) - copied, the safe side - except a page shared with a
// small parameter block (watch_check), and a single write (an initialisation) doesn't copy (snap_storage).
// LARECOMP_OVERLAP_WATCH=0: off (compute-only ranges get the one-frame hash probe, as before).
static bool heap_is_userptr(void);  // (with the heap import below)
static int pages_cpu_written(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now);  // (the page tables below)
#define WATCH_PAGE 4096ull  // (the page tables' page, VC_PAGE below)
static bool watch_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("LARECOMP_OVERLAP_WATCH");
        uint32_t g;
        on = !e || atoi(e) != 0;
        if (on && pages_cpu_written(HEAP_BASE, 4096, 0, &g) < 0) {
            on = 0;
            LOG("[vk] storage write watch off: no page write tracking%s (the hash probe instead)\n", heap_is_userptr() ? " over a userptr heap import" : "");
        } else LOG("[vk] storage write watch %s (compute-bound ranges copied once the OS's page write tracking sees a CPU write; no page protection)\n",
                   on ? "on" : "off (LARECOMP_OVERLAP_WATCH=0)");
    }
    return on;
}
// Did the CPU write any page of the range since the range last looked (e->wgen)? The look stamps the pages (every
// look resets the OS bits, so the generation table carries what other lookers saw). 1 written, 0 not, -1 no tracking.
// A page dirty bit can't tell which range on the page was stored to. A store into a small parameter block (the
// skinning dispatch's 12-byte slot 5, rewritten every frame) says nothing about the buffers beside it: counted as
// theirs, it copied BowWow's skinning output - a GPU output the CPU reads back every frame (his chain) - so the
// dispatch wrote the copy, the CPU read the frozen original, and his body came apart as a companion indoors (Metal
// 1ea4b16). So a range of 256 bytes or more counts only its dirty pages that no watched range under 256 bytes shares;
// the light / exposure lists (32640 / 2040 bytes, one allocation the CPU memcpys) share pages only with each other
// and keep their coverage. A small range counts every dirty page of its own.
static int watch_check(RangeState *e) {
    uint32_t since = e->wgen, now = e->wgen;
    int w = pages_cpu_written(e->addr, e->size, since, &now);
    e->wgen = now;
    if (w != 1 || e->size < 256) return w;
    for (uint64_t plo = e->addr & ~(WATCH_PAGE - 1); plo < e->addr + e->size; plo += WATCH_PAGE) {
        bool small_mate = false;
        for (int k = 0; k < nrs && !small_mate; k++) {
            const RangeState *o = &rs_tab[k];
            if (o != e && o->size < 256 && o->cs_bound && o->addr < plo + WATCH_PAGE && plo < o->addr + o->size) small_mate = true;
        }
        if (small_mate) continue;
        uint32_t pg_now;  // (this page written since the range's last look? The look above stamped it: no OS bits left)
        if (pages_cpu_written(plo, WATCH_PAGE, since, &pg_now) == 1) return 1;  // (a dirty page of its own)
    }
    return 0;  // (only pages shared with small parameter blocks were written: theirs)
}

// The address a binder of [addr, addr+size) uses: the original or its copy (cs_slot: bound by a compute dispatch)
static uint32_t vc_gen_now(void);  // (the page tables' current generation: below)
static uint64_t snap_storage(uint64_t addr, uint64_t size, bool cs_slot) {
    if (!snap_on() || !addr || !size || addr < snap_org() || addr - snap_org() + size > snap_base) return addr;
    if (size > SNAP_HALF / 2) {  // (the tile lists grow with the render resolution: ~1.2 MB at 7680x4320; a half holds two of these)
        static int logged;
        if (logged++ < 4) LOG("[vk] storage range %llx+%llu too large for a snapshot: read in place\n", (unsigned long long)addr, (unsigned long long)size);
        return addr;
    }
    extern _Atomic int kernel_guard_hit;  // (a physical mapping over the ring: kernel.c phys_map)
    if (gwin_.phys ? kernel_guard_hit != 0 : kernel_heap_size() > snap_base) {  // (the game grew into the ring: off from now on)
        static bool logged;
        if (!logged) { logged = true; LOG("[vk] storage snapshots off: the game's %s\n", gwin_.phys ? "memory mapped over the ring" : "heap grew into the ring"); }
        snap_state = 0;
        return addr;
    }
    if (gwin_.phys) {
        // A game on physical memory (Scarlet / Violet): a draw's storage read the CPU wrote in the last frames is copied
        // at translation, the rest stays in place. The map's text (its labels' blocks and glyph table, read as storage
        // by the glyphs' vertex program) is rewritten every ~20 frames in one copy; with the frame fence deferred the
        // GPU read the next frame's text - letters missing for 2 frames. The pool-flag decisions below are Link's
        // Awakening's (heap) and know only some of these pools; compute slots stay in place, as before here
        if (cs_slot || size > 65536) return addr;
        // (decided once a frame per range: the glyph table is bound by every text draw, the page query each time cost
        // ~7 fps on the map)
        extern uint64_t renderer_exec_frame(void);
        uint64_t fr = renderer_exec_frame();
        static struct { uint64_t addr, size, frame, at; } memo[1024];
        uint64_t hk = (addr * 0x9E3779B97F4A7C15ull) ^ size;
        __typeof__(memo[0]) *m = &memo[(hk >> 40) & 1023];
        if (m->addr == addr && m->size == size && m->frame == fr + 1) return m->at;
        uint32_t now;
        uint64_t at = addr;
        if (pages_cpu_written(addr, size, vc_gen_now(), &now) == 1) {  // (written lately: still unsettled, or now)
            uint64_t c = ring_take((size + 255) & ~255ull);
            if (c) {
                memcpy((void *)(uintptr_t)c, (const void *)(uintptr_t)addr, size);
                stat_snap_bytes += size;
                stat_snap_copies++;
                at = c;
            }
        }
        *m = (__typeof__(*m)){addr, size, fr + 1, at};
        return at;
    }
    extern int nvn_pool_flags_at(uint64_t addr);
    int pf = nvn_pool_flags_at(addr);
    if (pf < 0 || (pf & 1)) return addr;  // (CPU_NO_ACCESS: static data, nothing to race with)
    rs_level_tick();
    extern uint64_t renderer_exec_frame(void);
    uint64_t f = renderer_exec_frame();
    rs_release_unused(f);
    // A draw's storage read from the game's per-frame pools (CPU_UNCACHED | GPU_UNCACHED, 0x12: the 64-byte per-object
    // blocks at changing addresses, hundreds a frame) is copied without an entry: a fresh range every frame filled the
    // table in a few minutes, and the table starting over remade every copy from its original - the exposure list's
    // GPU state with it (one dark frame per list buffer, see rs_copy_make). Its slot is free again at once: reused only
    // after the half is left and taken again (the wait in ring_take), as a remade copy's old slot.
    if (!cs_slot && (pf & 0x13) == 0x12 && size <= 65536) {
        uint64_t at = ring_take((size + 255) & ~255ull);
        if (!at) return addr;
        memcpy((void *)(uintptr_t)at, (const void *)(uintptr_t)addr, size);
        stat_snap_bytes += size;
        stat_snap_copies++;
        return at;
    }
    RangeState *e = NULL;
    for (int i = 0; i < nrs; i++) if (rs_tab[i].addr == addr && rs_tab[i].size == size) { e = &rs_tab[i]; break; }
    uint64_t c;
    if (!e) {
        int k = nrs;
        if (nrs == 512) {  // table full: the entry bound longest ago makes room (never the whole table at once)
            k = 0;
            for (int i = 1; i < nrs; i++) if (rs_tab[i].frame < rs_tab[k].frame) k = i;
            rs_log(&rs_tab[k], "evicted (table full, bound longest ago)", cs_slot, f);
        } else nrs++;
        e = &rs_tab[k];
        memset(e, 0, sizeof *e);
        rs_store_frame[e - rs_tab] = 0;
        e->addr = addr, e->size = size, e->mode = RS_PROBE, e->level = rs_level_gen, e->frame = f;
        e->cs_bound = cs_slot;
        if (!cs_slot) e->mode = RS_COPY, e->draw_bound = true;  // a draw's storage read: copied for good (see above)
        else if (watch_on()) {  // a compute-only range: in place; the first CPU write its pages' tracking shows makes it a copy
            e->mode = RS_PLACE;
            watch_check(e);  // (from now: the look stamps its pages)
            rs_log(e, "new: in place, CPU writes tracked", cs_slot, f);
            return addr;
        }
        // the probe frame (or the first copy): every binder this frame takes this copy (frame == f below); a probe's
        // decision is at the next frame's use, when a CPU rewrite can have happened
        c = rs_copy_make(e, range_hash(addr, size));
        if (!c) {  // (no copy possible now: the entry is dropped - not probed yet - and the range is created again next time)
            rs_log(e, "no copy now (the ring's other half is still in flight): in place this once", cs_slot, f);
            if (k != --nrs) { rs_tab[k] = rs_tab[nrs]; rs_store_frame[k] = rs_store_frame[nrs]; }  // (an evicted slot: the last entry moves in)
            return addr;
        }
        rs_log(e, cs_slot ? "new: probe" : "new: copy (a draw's storage read)", cs_slot, f);
        return c;
    }
    if (cs_slot) e->cs_bound = true; else e->draw_bound = true;
    if (e->frame == f) {  // (bound again this frame)
        if (e->mode == RS_PLACE && cs_slot) return addr;
        if (!cs_slot && e->mode != RS_COPY) {  // (a draw's storage read of a range probed / settled in place this frame: copied for good)
            e->mode = RS_COPY;
            rs_log(e, "-> copy (a draw's storage read)", cs_slot, f);
            c = rs_copy_make(e, range_hash(addr, size));
            return c ? c : addr;
        }
        if (e->copy) return e->copy;
        c = rs_copy_make(e, e->hash);
        return c ? c : addr;
    }
    // probe again on a new level or after 10 s unused - only the hash fallback's in-place ranges: a copied range stays
    // copied (a draw's storage read, a CPU store seen), a watched one just gets its pages armed again
    bool again = !watch_on() && e->mode == RS_PLACE && (e->level != rs_level_gen || f - e->frame > 600);
    e->level = rs_level_gen;
    e->frame = f;
    if (!cs_slot && e->mode != RS_COPY) {  // a draw reads it as storage: copied for good (see above)
        e->mode = RS_COPY;
        rs_log(e, "-> copy (a draw's storage read)", cs_slot, f);
        c = rs_copy_make(e, range_hash(addr, size));
        return c ? c : addr;
    }
    if (e->mode == RS_PLACE && e->cs_bound && watch_on()) {  // (tracked: the OS's dirty bits decide, no hash)
        if (watch_check(e) != 1) return addr;
        // A write once (an initialisation: BowWow's skinning buffers at his spawn) stays in place; writes that keep
        // coming (within 8 frames of the previous one) are the CPU rewriting it: copied from now on (renderer.m 1ea4b16)
        int k = (int)(e - rs_tab);
        if (!rs_store_frame[k] || f - rs_store_frame[k] > 8) { rs_store_frame[k] = f; return addr; }
        e->mode = RS_COPY;
        rs_log(e, "-> copy (CPU writes keep coming on its own pages)", cs_slot, f);
        c = rs_copy_make(e, range_hash(addr, size));
        return c ? c : addr;
    }
    if (e->mode == RS_PLACE && !again && e->cs_bound) return addr;
    uint64_t h = range_hash(addr, size);
    if (again && !e->draw_bound) {  // (a draw-read range stays a copy across levels)
        e->mode = RS_PROBE;
        rs_log(e, "probe again (level / unused)", cs_slot, f);
        c = rs_copy_make(e, h);
        return c ? c : addr;
    }
    switch (e->mode) {
    case RS_PLACE:  // (never a compute slot's: a change can only be the CPU's)
        if (h == e->hash) return addr;
        e->mode = RS_COPY;
        rs_log(e, "in place -> copy (changed later)", cs_slot, f);
        break;
    case RS_PROBE:  // the use after the probe: did the CPU touch the original meanwhile?
        if (h == e->hash) { e->mode = RS_PLACE; e->copy = e->pristine = 0; rs_log(e, "probe -> in place (unchanged)", cs_slot, f); return addr; }
        e->mode = RS_COPY;
        rs_log(e, "probe -> copy (the CPU rewrote it)", cs_slot, f);
        break;
    default:  // RS_COPY
        if (h == e->hash && e->copy) return e->copy;  // unchanged: the copy (maybe GPU-written: its state) is current
        break;
    }
    c = rs_copy_make(e, h);
    return c ? c : addr;
}

// (debug) LARECOMP_LOG_BINDS=1: guest memory the GPU reads in place, per frame (bytes bound; distinct ranges once),
// and the ranges of the previous frame whose contents the game rewrote by the time this frame is translated: with
// the frame fence deferred (overlap) the GPU may still read them - the racing data. kind: 0 vertex, 1 index,
// 2 storage (a draw's read), 3 compute slot.
static void bind_note(int kind, uint64_t addr, uint64_t size) {
    static int on = -1;
    if (on < 0) on = getenv("LARECOMP_LOG_BINDS") != NULL;
    if (!on || addr < HEAP_BASE || addr - HEAP_BASE + size > HEAP_MAX || size < 8) return;
    extern int nvn_pool_flags_at(uint64_t addr);
    enum { REC = 8192 };
    static struct { uint64_t addr, size, hash; uint8_t kind; } rec[REC];
    static int nrec;
    static uint64_t frame, all[4], uniq[4], wr[4];
    static uint32_t n[4];
    extern uint64_t renderer_exec_frame(void);
    uint64_t f = renderer_exec_frame();
    if (f != frame) {
        // the previous frame's ranges, rewritten since?
        uint64_t chg[4] = {0}, chgn[4] = {0};
        int shown = 0;
        for (int i = 0; i < nrec; i++) {
            if (range_hash(rec[i].addr, rec[i].size) == rec[i].hash) continue;
            chg[rec[i].kind] += rec[i].size; chgn[rec[i].kind]++;
            if (frame % 120 == 0 && shown++ < 12)
                LOG("[race] frame %llu %s %llx+%llu pool flags %#x\n", (unsigned long long)frame,
                    rec[i].kind == 0 ? "vertex" : rec[i].kind == 1 ? "index" : rec[i].kind == 2 ? "storage" : "compute",
                    (unsigned long long)rec[i].addr, (unsigned long long)rec[i].size, nvn_pool_flags_at(rec[i].addr));
        }
        if (frame % 120 == 0 || chgn[0] + chgn[1] + chgn[2] + chgn[3])
            LOG("[binds] frame %llu: vertex %llu KB (%llu KB distinct, %u binds), index %llu KB (%llu), storage %llu KB (%llu, %u), compute %llu KB (%llu) | in CPU-writable pools: %llu/%llu/%llu/%llu KB | rewritten since: vertex %llu KB/%llu, index %llu KB/%llu, storage %llu KB/%llu, compute %llu KB/%llu\n",
                (unsigned long long)frame, (unsigned long long)(all[0] >> 10), (unsigned long long)(uniq[0] >> 10), n[0], (unsigned long long)(all[1] >> 10), (unsigned long long)(uniq[1] >> 10),
                (unsigned long long)(all[2] >> 10), (unsigned long long)(uniq[2] >> 10), n[2], (unsigned long long)(all[3] >> 10), (unsigned long long)(uniq[3] >> 10),
                (unsigned long long)(wr[0] >> 10), (unsigned long long)(wr[1] >> 10), (unsigned long long)(wr[2] >> 10), (unsigned long long)(wr[3] >> 10),
                (unsigned long long)(chg[0] >> 10), (unsigned long long)chgn[0], (unsigned long long)(chg[1] >> 10), (unsigned long long)chgn[1],
                (unsigned long long)(chg[2] >> 10), (unsigned long long)chgn[2], (unsigned long long)(chg[3] >> 10), (unsigned long long)chgn[3]);
        nrec = 0; memset(all, 0, sizeof all); memset(uniq, 0, sizeof uniq); memset(n, 0, sizeof n); memset(wr, 0, sizeof wr);
        frame = f;
    }
    all[kind] += size; n[kind]++;
    for (int i = 0; i < nrec; i++) if (rec[i].addr == addr && rec[i].size == size) return;
    if (nrec == REC) return;
    rec[nrec].addr = addr; rec[nrec].size = size; rec[nrec].kind = (uint8_t)kind;
    rec[nrec].hash = range_hash(addr, size);
    nrec++;
    uniq[kind] += size;
    int pf = nvn_pool_flags_at(addr);
    if (pf < 0 || !(pf & 1)) wr[kind] += size;  // (distinct bytes in pools the CPU can write: what a snapshot would copy)
}

// Before a draw (stages 0 and 1: the slots the shader reads addresses of, si->ssbo_read) or a dispatch (stage 5: every
// bound slot): the stage's effective storage table, with the copies' addresses where a range is snapshotted. Every
// reader of the table in the draw / dispatch (drv_dedupe, cbuf_info, gmem_direct_pick, gmem_chunk_bind, gfast_prepare,
// vc_mark_sbufs) takes sbufs(stage); st.sbuf itself keeps the game's bindings (rehashed next frame).
static void snap_apply_(int stage, uint32_t read_mask);
// A draw stage's slots to snapshot: those its shader reads the addresses of - none, on a physical-memory window (Scarlet
// / Violet), when the shader stores to global memory: its stores must reach the game's memory, and a copy took them.
// The characters' faces: vertex programs (7442131000, 7442131600, ...) write the 48-byte records of storage buffer 13
// that the expressions are drawn from; written into the ring, the originals never got them and every face kept its
// neutral pose - eyes open behind closed eyelashes, mouths shut. Compute dispatches already read in place there
// (snap_storage); the copies stay for the draws that only read (the map's text).
static uint32_t snap_draw_mask(const ShaderInfo *si) {
    if (!si) return 0;
    return si->uses_gmem_store && atomic_load(&gwin_done) && gwin_.phys ? 0 : si->ssbo_read;
}
static void snap_apply(int stage, uint32_t read_mask) {
    if (!read_mask) { snap_apply_(stage, 0); return; }  // (per context only)
    // (a shadow worker: the copies' addresses this part had this frame for the same ranges - snap_storage gives every
    // binder of a range in a frame the same copy - without the lock. Its table's search under the lock was most of the
    // parts' wait for it once the vertex cache needed none)
    static struct { uint64_t addr, size, frame, res; } memo_t[NCTX][256];
    static int binds = -1;
    if (binds < 0) binds = getenv("LARECOMP_LOG_BINDS") != NULL;
    extern uint64_t renderer_exec_frame(void);
    bool quick = sw_cb && stage != 5 && !binds && snap_on();
    uint64_t f = quick ? renderer_exec_frame() : 0;
    __typeof__(memo_t[0][0]) *m = memo_t[sw_ctx()];
    if (quick) {
        Range eff[16];
        memcpy(eff, st.sbuf[stage], sizeof eff);
        bool all = true;
        for (int i = 0; i < 16 && all; i++) {
            if (!eff[i].addr || !((read_mask >> i) & 1)) continue;
            uint64_t h = (eff[i].addr * 0x9E3779B97F4A7C15ull) ^ eff[i].size;
            __typeof__(m[0]) *e = &m[(h >> 40) & 255];
            if (e->addr == eff[i].addr && e->size == eff[i].size && e->frame == f) eff[i].addr = e->res;
            else all = false;
        }
        if (all) {
            sb_eff_on = true;
            memcpy(sb_eff[stage], eff, sizeof eff);
            return;
        }
    }
    SWL_AT(4059);
    snap_apply_(stage, read_mask);
    SWU();
    if (quick && sb_eff_on)
        for (int i = 0; i < 16; i++) {
            const Range *g = &st.sbuf[stage][i];
            if (!g->addr || !((read_mask >> i) & 1)) continue;
            uint64_t h = (g->addr * 0x9E3779B97F4A7C15ull) ^ g->size;
            m[(h >> 40) & 255] = (__typeof__(m[0])){g->addr, g->size, f, sb_eff[stage][i].addr};
        }
}
static void snap_apply_(int stage, uint32_t read_mask) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    bool on = snap_on();
    for (int i = 0; i < 16; i++)
        if (st.sbuf[stage][i].addr && ((read_mask >> i) & 1)) bind_note(stage == 5 ? 3 : 2, st.sbuf[stage][i].addr, st.sbuf[stage][i].size);
    if (!on) {
        sb_eff_on = false;
        if (stage == 5)
            for (int i = 0; i < 16; i++) gw_note("compute storage", st.sbuf[5][i].addr, st.sbuf[5][i].size, st.program ? st.program->stages[5].code_addr : 0);
        return;
    }
    sb_eff_on = true;
    memcpy(sb_eff[stage], st.sbuf[stage], sizeof sb_eff[stage]);
    for (int i = 0; i < 16; i++) {
        Range *r = &sb_eff[stage][i];
        if (r->addr && ((read_mask >> i) & 1)) r->addr = snap_storage(r->addr, r->size, stage == 5);
    }
    if (stage == 5)
        for (int i = 0; i < 16; i++) gw_note("compute storage", st.sbuf[5][i].addr, st.sbuf[5][i].size, st.program ? st.program->stages[5].code_addr : 0);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// The driver constants of a graphics stage (cbuf 0, size bytes): zero but for the stage's storage buffer table at
// base (16 bytes an entry: address low, high, size). Compared by the entries within size.
static VkSpan drv_dedupe(int stage, uint64_t size) {
    static struct { uint64_t seq, size; int stage; Range sb[16]; VkSpan span; } last_t[NCTX][2];
    __typeof__(last_t[0][0]) *last = last_t[sw_ctx()];
    int slot = stage == 1;
    uint32_t base = 0x110 + 0x100 * (stage == 1 ? 4 : 0);
    int ne = size > base ? (int)((size - base + 15) / 16) : 0;
    if (ne > 16) ne = 16;
    uint64_t seq = vk_pending_seq();
    const Range *sb = sbufs(stage);  // (overlap: a snapshotted range's copy)
    if (last[slot].seq == seq && last[slot].size == size && last[slot].stage == stage &&
        !memcmp(last[slot].sb, sb, (size_t)ne * sizeof(Range))) {
        prof_cb_reused++;
        return last[slot].span;
    }
    uint32_t drv[0x800 / 4];
    memset(drv, 0, size);
    for (int i = 0; i < ne; i++) {  // (an entry may run past size: drv is 0x800 bytes, only size of them are copied)
        uint32_t *e = &drv[(base + 16 * i) / 4];
        e[0] = (uint32_t)sb[i].addr;
        e[1] = (uint32_t)(sb[i].addr >> 32);
        e[2] = (uint32_t)sb[i].size;
    }
    VkSpan s = vk_transient_copy(drv, size, 256);
    last[slot].seq = seq; last[slot].size = size; last[slot].stage = stage; last[slot].span = s;
    memcpy(last[slot].sb, sb, (size_t)ne * sizeof(Range));
    prof_cb_copied++;
    return s;
}

// Constant buffer k of a stage (renderer.m bind_cbufs), copied into transient memory.
static VkDescriptorBufferInfo cbuf_info(int stage, ShaderInfo *si, int k, uint32_t gx, uint32_t gy, uint32_t gz) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    static const uint8_t zeros[0x10000];
    if (k == 0) {
        // The driver constants are made from the stage's buffer bindings (and a dispatch's group size) alone: a draw
        // binding the same ones as the previous draw of its kind, in the same submission, shares that copy. Comparing
        // those few hundred bytes replaced building the 2 KB table and comparing it whole, twice a draw (7% of the
        // render thread, the one that limits the frame rate on slower CPUs).
        uint64_t size = si->cbuf_size[0] > 0x800 ? 0x800 : (si->cbuf_size[0] ? si->cbuf_size[0] : 256);
        if (stage != 5 && cbuf_lean()) return ubo_span(drv_dedupe(stage, size), size);
        if (size < 0x500 && stage == 5) size = 0x500;
        const Range *sb = sbufs(stage);  // (overlap: a snapshotted range's copy)
        struct { uint64_t size, sb[16][2], ub[16][2]; uint32_t g[4]; } key;
        memset(&key, 0, sizeof key);
        key.size = size;
        for (int i = 0; i < 16; i++) key.sb[i][0] = sb[i].addr, key.sb[i][1] = sb[i].size;
        if (stage == 5) {
            for (int i = 0; i < 16; i++) key.ub[i][0] = st.ubuf[5][i].addr, key.ub[i][1] = st.ubuf[5][i].size;
            key.g[0] = gx, key.g[1] = gy, key.g[2] = gz;
        }
        static struct { uint64_t seq; __typeof__(key) key; VkSpan span; } last_t[NCTX][3];
        __typeof__(last_t[0][0]) *last = last_t[sw_ctx()];
        int slot = stage == 5 ? 2 : stage == 1 ? 1 : 0;
        uint64_t seq = vk_pending_seq();
        if (last[slot].seq == seq && !memcmp(&last[slot].key, &key, sizeof key)) return ubo_span(last[slot].span, size);
        uint32_t drv[0x800 / 4] = {0};
        if (si->uses_gmem) {  // storage buffers outside the guest memory window: their shader accesses read zero
            static int nlog;
            for (int i = 0; i < 16 && nlog < 30; i++) {
                uint64_t a = st.sbuf[stage][i].addr, sz = st.sbuf[stage][i].size;
                if (a && (a < gwin_base || a + sz > gwin_base + gwin_max)) {
                    nlog++;
                    LOG("[vk] storage buffer outside the gmem window: stage %d sb%d %llx+%llx (program %llx)\n", stage, i,
                        (unsigned long long)a, (unsigned long long)sz, (unsigned long long)st.program->stages[stage].code_addr);
                }
            }
        }
        if (stage == 5) {
            drv[0x200 / 4] = gx; drv[0x204 / 4] = gy; drv[0x208 / 4] = gz;
            for (int i = 0; i < 16; i++) {
                uint32_t *u = &drv[(0x210 + 16 * i) / 4];
                u[0] = (uint32_t)st.ubuf[5][i].addr; u[1] = (uint32_t)(st.ubuf[5][i].addr >> 32); u[2] = (uint32_t)st.ubuf[5][i].size;
                uint32_t *e = &drv[(0x310 + 16 * i) / 4];
                e[0] = (uint32_t)sb[i].addr; e[1] = (uint32_t)(sb[i].addr >> 32); e[2] = (uint32_t)sb[i].size;
            }
        } else {
            uint32_t base = 0x110 + 0x100 * (stage == 1 ? 4 : 0);
            for (int i = 0; i < 16; i++) {
                uint32_t *e = &drv[(base + 16 * i) / 4];
                e[0] = (uint32_t)sb[i].addr;
                e[1] = (uint32_t)(sb[i].addr >> 32);
                e[2] = (uint32_t)sb[i].size;
            }
        }
        VkSpan sp = vk_transient_copy(drv, size, 256);
        last[slot].seq = seq, last[slot].key = key, last[slot].span = sp;
        return ubo_span(sp, size);
    }
    if (k == 1) {  // compiler constants stored after the shader code
        NvnShaderStage *sst = &st.program->stages[stage];
        uint64_t size = sst->const_size ? sst->const_size : 16;
        if (size < si->cbuf_size[1]) size = si->cbuf_size[1];
        return ubo_span(guest_copy(sst->code_addr + sst->const_offset, size), size);  // (constant: shared per submission)
    }
    int idx = k - 3;  // cbuf_base
    Range r = (idx >= 0 && idx < 18) ? st.ubuf[stage][idx] : (Range){0, 0};
    uint64_t size = si->cbuf_size[k] ? si->cbuf_size[k] : 256;
    if (!r.addr) return ubo_span(vk_transient_copy(zeros, size, 256), size);
    if ((si->cbuf_dynamic & (1u << k)) && r.size) size = r.size > 0x10000 ? 0x10000 : r.size;
    if (size > r.size && r.size) size = r.size;
    if (size < 16) size = 16;
    // snapshot: later CPU writes must not leak in. Shared by the draws of one submission binding the same range:
    // within a command list nothing writes guest memory (no inline UpdateUniformBuffer: that writes through below
    // and drops the shared copies).
    // (lazy heap chunks) pointers in it import their chunks: scanned once per shared copy (LARECOMP_VK_NO_SCAN_ONCE=1:
    // at every draw, as before)
    static int scan_once = -1;
    if (scan_once < 0) scan_once = getenv("LARECOMP_VK_NO_SCAN_ONCE") == NULL;
    if (gc_lazy && si->uses_gmem && scan_once) return ubo_span(guest_copy_ex(r.addr, size, true), size);
    if (gc_lazy && si->uses_gmem) gc_scan_pointers((const uint32_t *)(uintptr_t)r.addr, size);
    return ubo_span(guest_copy(r.addr, size), size);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// Global memory (shader LDG/STG, maxwell_gmem_base = gwin_base): the guest memory window imported as a GPU buffer
// (VK_EXT_external_memory_host), zero-copy like metal.m's g_guest_buffer, so compute results land in guest
// memory. The import pins its pages, so it covers only the heap the game set up (SetHeapSize), made on first use.
// LARECOMP_VK_NO_GMEM=1: no import, reads return zero.
static VkBuffer gmem_buf;
static VkDeviceMemory gmem_mem;
static uint64_t gmem_size;
// LARECOMP_VK_GMEM_WINDOW=<MB>: import only the top <MB> of the heap instead of all of it. On the Steam Deck the 3.2 GB host-pointer (userptr) import is
// revalidated by amdgpu at every submit: 1.75 ms x 75 submits a frame, 87 % of the render thread (6.7 fps; 21 fps without the import). Z-A's compute
// storage buffers all sit in an 8 MB spot near the heap's end. gmem_lo: heap offset of the import's first byte (0: the whole heap).
static uint64_t gmem_lo, gmem_window;
#define GM_IN(a, sz) (gmem_buf && (a) >= gwin_base + gmem_lo && (a) + (sz) <= gwin_base + gmem_lo + gmem_size)
#define GM_OFF(a) ((a) - gwin_base - gmem_lo)
// The window is anchored at the end of the physical region (gwin_max: constant for the session, so shaders may bake it): [lo, gwin_max).
// Only with a physical-memory window (no heap), else the whole heap as before. The generated shader code computes ga = addr - base, ok = ga < size,
// so translate_program / build_compute set base and size to the window too (the window changed them by lo: text and buildings read the wrong place).
// default: on an integrated GPU (the Steam Deck: the full import cost 87 % of the render thread), 256 MB; LARECOMP_VK_GMEM_WINDOW=0 imports it all
static void gmem_window_init(void) {
    const char *w = ENV("LARECOMP_VK_GMEM_WINDOW");
    gmem_window = w ? strtoull(w, NULL, 10) << 20 : g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 256ull << 20 : 0;
}
// A game with a heap besides the physical window (Pokémon Legends Arceus) spreads its storage buffers lower: its glyph
// instances 450 KB below the top 256 MB, its exposure's compute buffers ~790 MB below the top - read as zero outside
// the window (no dialogue text, every character overexposed). It gets the top 1 GB (unless LARECOMP_VK_GMEM_WINDOW
// says otherwise). Not the whole pool: the import makes the driver keep all of it resident, and the whole 3.2 GB on a
// 7.4 GB machine pushed the rest into swap - vkQueueSubmit2 stalled 20-30 s, twice in half an hour of play.
static void gmem_window_get(uint64_t *lo, uint64_t *len) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;  // (the translator and the render thread - run_queued_now - both ask)
    pthread_once(&once, gmem_window_init);
    uint64_t w = gmem_window;
    if (w && w < (1ull << 30) && !ENV("LARECOMP_VK_GMEM_WINDOW") && kernel_heap_size()) w = 1ull << 30;
    if (!w || !gwin()->phys || gwin_max <= w) { *lo = 0; *len = gwin_max; return; }
    *lo = (gwin_max - w) & ~0xFFFFFull;
    *len = gwin_max - *lo;
}

// Host memory [host, host + size) as a buffer (VK_EXT_external_memory_host; page-aligned). The allocation's result:
// out of memory tells whole_import_probe_oom to go lazy.
static VkResult host_import_ptr(void *host, uint64_t size, VkBuffer *out_buf, VkDeviceMemory *out_mem, bool bda) {
    PFN_vkGetMemoryHostPointerPropertiesEXT props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(g_vk.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (!props || props(g_vk.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &mp) != VK_SUCCESS || !mp.memoryTypeBits)
        return VK_ERROR_INITIALIZATION_FAILED;
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = host};
    VkMemoryAllocateFlagsInfo fl = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &imp, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = bda ? (void *)&fl : (void *)&imp, .allocationSize = size,
                               .memoryTypeIndex = (uint32_t)__builtin_ctz(mp.memoryTypeBits)};
    VkDeviceMemory mem;
    VkResult r = vkAllocateMemory(g_vk.dev, &ai, NULL, &mem);
    if (r != VK_SUCCESS) return r;
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | (bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0)};
    VkBuffer buf;
    r = vkCreateBuffer(g_vk.dev, &bi, NULL, &buf);
    if (r != VK_SUCCESS) { vkFreeMemory(g_vk.dev, mem, NULL); return r; }
    r = vkBindBufferMemory(g_vk.dev, buf, mem, 0);
    if (r != VK_SUCCESS) { vkDestroyBuffer(g_vk.dev, buf, NULL); vkFreeMemory(g_vk.dev, mem, NULL); return r; }
    *out_buf = buf;
    *out_mem = mem;
    return VK_SUCCESS;
}
// Heap bytes [off, off + size) as a buffer (a page-aligned range of the mapped heap).
static bool host_import(uint64_t off, uint64_t size, VkBuffer *out_buf, VkDeviceMemory *out_mem, bool bda) {
    return host_import_ptr((void *)(uintptr_t)(HEAP_BASE + off), size, out_buf, out_mem, bda) == VK_SUCCESS;
}

// One import of the whole heap, tried once at the start where it should fit but not by much (device.c
// whole_import_fits: g_vk.heap_probe): HEAP_MAX of scratch host memory imported and used by one submission, then
// freed. True: it ran out of memory (the import or the submission: issue #205's driver took the import and failed
// the first submission), and the heap is imported in lazy chunks instead - decided here, at device creation (vk_exec_init),
// before any shader is translated or the game starts. Costs: the import pins HEAP_MAX of RAM for a moment (well under a
// second). Any other failure: logged, and the heap is imported as one buffer as before.
// LARECOMP_VK_SIM_HEAP_OOM=1: the probe runs, then reports out of memory (tests the fallback on any GPU).
static bool whole_import_probe_oom(void) {
    const uint64_t size = HEAP_MAX, align = 64 << 10;  // (minImportedHostPointerAlignment: 4 KB here, 64 KB at most)
    const char *sim = getenv("LARECOMP_VK_SIM_HEAP_OOM");
    rec_drain();  // (the queue below is the recorder's: idle and empty while this runs)
    uint64_t t0 = plat_now_ns();
    VkResult r = VK_ERROR_OUT_OF_HOST_MEMORY;
    const char *step = "host memory";
    void *raw = malloc(size + align);
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (raw) {
        step = "import";
        r = host_import_ptr((void *)(((uintptr_t)raw + align - 1) & ~(uintptr_t)(align - 1)), size, &buf, &mem, false);
    }
    if (r == VK_SUCCESS) {  // a submission that uses it: where the driver makes it resident
        step = "submission";
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = g_vk.qfam};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkCommandBuffer cb = VK_NULL_HANDLE;
        r = vkCreateCommandPool(g_vk.dev, &pi, NULL, &pool);
        if (r == VK_SUCCESS) r = vkCreateFence(g_vk.dev, &fi, NULL, &fence);
        if (r == VK_SUCCESS) {
            VkCommandBufferAllocateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                              .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
            r = vkAllocateCommandBuffers(g_vk.dev, &ci, &cb);
        }
        if (r == VK_SUCCESS) {
            VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                           .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vkBeginCommandBuffer(cb, &bi);
            vkCmdFillBuffer(cb, buf, 0, 256, 0);         // (zeros into zeroed memory: the start and the end of it)
            vkCmdFillBuffer(cb, buf, size - 256, 256, 0);
            r = vkEndCommandBuffer(cb);
        }
        if (r == VK_SUCCESS) {
            VkCommandBufferSubmitInfo cbs = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
            VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbs};
            r = vkQueueSubmit2(g_vk.queue, 1, &si, fence);
            if (r == VK_SUCCESS) r = vkWaitForFences(g_vk.dev, 1, &fence, VK_TRUE, 10000000000ull);
        }
    }
    if (r == VK_ERROR_DEVICE_LOST) vk_devlost_dump();
    if (fence) vkDestroyFence(g_vk.dev, fence, NULL);
    if (pool) vkDestroyCommandPool(g_vk.dev, pool, NULL);
    if (buf) vkDestroyBuffer(g_vk.dev, buf, NULL);
    if (mem) vkFreeMemory(g_vk.dev, mem, NULL);
    free(raw);
    bool oom = r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY;
    if (r == VK_SUCCESS && sim && atoi(sim)) { oom = true; r = VK_ERROR_OUT_OF_DEVICE_MEMORY; step = "simulated (LARECOMP_VK_SIM_HEAP_OOM)"; }
    unsigned ms = (unsigned)((plat_now_ns() - t0) / 1000000);
    if (oom)
        LOG("[vk] guest heap: one import of %llu MB ran out of memory (%s, error %d, %u ms): falling back to chunks "
            "imported as the game binds them\n", (unsigned long long)(size >> 20), step, r, ms);
    else if (r != VK_SUCCESS)
        LOG("[vk] guest heap: one import of %llu MB failed to test (%s, error %d, %u ms): one buffer as before\n",
            (unsigned long long)(size >> 20), step, r, ms);
    else
        LOG("[vk] guest heap: one import of %llu MB works here (%u ms)\n", (unsigned long long)(size >> 20), ms);
    return oom;
}

// Guest window bytes [lo, lo + size) for shader global memory (gmem_info; the window: gwin - the heap, or Pokémon Z-A's
// physical memory)
static bool gmem_import(uint64_t lo, uint64_t size) {
    VkBuffer buf;
    VkDeviceMemory mem;
#ifdef _WIN32
    // (test, #39 #72) LARECOMP_VK_GMEM_PREFAULT=1: every page of the window touched before the import - committed
    // pages the game never wrote have no physical page yet, and a driver that maps only the pages present at the
    // import would fault a shader's read of one. An atomic add of 0: a guest thread's store at the same moment stays.
    const char *pf = ENV("LARECOMP_VK_GMEM_PREFAULT");
    if (pf && atoi(pf)) {
        uint64_t t0 = SDL_GetTicksNS();
        for (uint64_t a = gwin_base + lo; a < gwin_base + lo + size; a += 4096) __atomic_fetch_add((volatile uint32_t *)(uintptr_t)a, 0, __ATOMIC_RELAXED);
        LOG("[vk] gmem prefault: %llu MB touched in %llu ms (LARECOMP_VK_GMEM_PREFAULT)\n", (unsigned long long)(size >> 20),
            (unsigned long long)((SDL_GetTicksNS() - t0) / 1000000));
    }
#endif
    VkResult r = host_import_ptr((void *)(uintptr_t)(gwin_base + lo), size, &buf, &mem, g_vk.gmem_addr);
    if (r != VK_SUCCESS) { LOG("[vk] gmem import: %d (%llu MB)\n", r, (unsigned long long)(size >> 20)); return false; }
    if (g_vk.gmem_addr) {  // (#39 #72: a page fault's address minus this start is the guest address base + lo + that)
        VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buf};
        uint64_t va = vkGetBufferDeviceAddress(g_vk.dev, &ai);
        LOG("[vk] guest window import at device address [%#llx, %#llx) = guest [%llx, %llx)\n", (unsigned long long)va,
            (unsigned long long)(va + size), (unsigned long long)(gwin_base + lo), (unsigned long long)(gwin_base + lo + size));
        extern uint64_t g_gmem_va, g_gmem_guest, g_gmem_va_size;
        g_gmem_va = va, g_gmem_guest = gwin_base + lo, g_gmem_va_size = size;
    }
    if (gmem_buf) vk_defer_buffer(gmem_buf, gmem_mem);  // the heap grew: the pending submission may still use the old import
    atomic_fetch_add_explicit(&gmem_seq, 1, memory_order_acq_rel);  // (gmem_info_quick: being replaced)
    gmem_buf = buf;
    gmem_mem = mem;
    gmem_size = size;
    gmem_lo = lo;
    atomic_fetch_add_explicit(&gmem_seq, 1, memory_order_release);
    LOG("[vk] guest heap imported for shader global memory: %llu MB from heap offset %llu MB\n", (unsigned long long)(size >> 20), (unsigned long long)(lo >> 20));
    return true;
}

// Linux, AMD (RADV): the import above is a userptr buffer, and amdgpu checks every page of a userptr buffer again on
// each submission - 3.2 GB of heap, ~45 ms of driver time a frame on a ROG Ally X, Steam Deck, Radeon laptops (issues
// #40, #51). Here the heap is instead allocated by the driver (cached system memory, exported as a dma-buf) and that
// memory is mapped at HEAP_BASE before the game starts, so the GPU sees an ordinary buffer. Called from boot (main.c)
// on the main thread before any guest thread exists; a reboot frees it with gmem_buf (vk_exec_reset) and the guest
// window's reset unmaps it, so the next boot gets fresh zeroed memory. Costs: the whole HEAP_MAX is committed once the
// GPU first uses it, and Linux write tracking can't watch this mapping (the heap is then re-checked like on Beta 4).
// Nothing writes the heap before boot (the start screen doesn't use guest memory): the mapping replaces it unseen.
// The mapping: where the driver has VK_EXT_map_memory_placed (RADV, Mesa 24.1+), it maps the memory at HEAP_BASE
// itself, through its own DRM file as for any vkMapMemory. Else the memory is exported as a dma-buf and that is
// mmapped there - which some systems refuse (issue #92: RX 6800 XT, Mesa 25.0.7, "Permission denied"), and then
// the whole heap was a userptr import again.
// Default on for RADV; LARECOMP_VK_HEAP_DMABUF=0 off, =1 on any Linux driver with dma-buf export or placed maps.
// LARECOMP_VK_HEAP_PLACED=0: the dma-buf mapping even where placed maps exist (tests).
#ifdef __linux__
static bool gmem_dmabuf;  // the heap is the driver's memory mapped at HEAP_BASE
static bool gmem_placed;  // ... by the driver (vkMapMemory2KHR, placed): unmapped through it at a reboot

// The driver maps `mem` (HEAP_MAX, one allocation) at HEAP_BASE. NULL, or why not; HEAP_BASE is ordinary guest
// memory again after a failure.
static const char *heap_map_placed(VkDeviceMemory mem) {
#ifndef VK_EXT_map_memory_placed
    (void)mem;
    return "built without VK_EXT_map_memory_placed";  // (g_vk.map_placed is never set then)
#else
    PFN_vkMapMemory2KHR map2 = (PFN_vkMapMemory2KHR)vkGetDeviceProcAddr(g_vk.dev, "vkMapMemory2KHR");
    PFN_vkUnmapMemory2KHR unmap2 = (PFN_vkUnmapMemory2KHR)vkGetDeviceProcAddr(g_vk.dev, "vkUnmapMemory2KHR");
    if (!map2 || !unmap2) return "no vkMapMemory2KHR";
    // First mapped anywhere, to know the driver can map it at all: a placed map replaces what's at HEAP_BASE before
    // the driver's own mmap can fail
    VkMemoryMapInfoKHR mi = {.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR, .memory = mem, .offset = 0, .size = VK_WHOLE_SIZE};
    VkMemoryUnmapInfoKHR ui = {.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR, .memory = mem};
    void *p = NULL;
    if (map2(g_vk.dev, &mi, &p) != VK_SUCCESS || !p) return "the driver can't map it";
    unmap2(g_vk.dev, &ui);
    VkMemoryMapPlacedInfoEXT pl = {.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT, .pPlacedAddress = (void *)(uintptr_t)HEAP_BASE};
    mi.flags = VK_MEMORY_MAP_PLACED_BIT_EXT;
    mi.pNext = &pl;
    p = NULL;
    VkResult r = map2(g_vk.dev, &mi, &p);
    if (r == VK_SUCCESS && p == (void *)(uintptr_t)HEAP_BASE) return NULL;
    if (r == VK_SUCCESS) unmap2(g_vk.dev, &ui);
    // (the old mapping may be gone: plain zeroed memory again, registered for write tracking like a reboot's)
    if (!plat_reset_guest(HEAP_BASE, HEAP_BASE + HEAP_MAX)) FATAL("cannot restore the guest heap mapping\n");
    return "the driver's placed mapping failed";
#endif
}

// The driver memory heap at a reboot (vk_exec_reset, GPU idle, guest threads stopped): the driver's mapping is left
// as a reserved range (or unmapped), then HEAP_BASE gets fresh zeroed guest memory at once, so no hole is left in the
// guest window until the window's reset.
static void heap_unmap_placed(void) {
    if (!gmem_placed) return;
    PFN_vkUnmapMemory2KHR unmap2 = (PFN_vkUnmapMemory2KHR)vkGetDeviceProcAddr(g_vk.dev, "vkUnmapMemory2KHR");
    VkMemoryUnmapInfoKHR ui = {.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR, .memory = gmem_mem,
                               .flags = 0};
#ifdef VK_EXT_map_memory_placed
    if (g_vk.unmap_reserve) ui.flags = VK_MEMORY_UNMAP_RESERVE_BIT_EXT;
#endif
    if (unmap2) unmap2(g_vk.dev, &ui);
    gmem_placed = false;
}

static bool guest_heap_dmabuf(void) {
    if (gmem_buf || !g_vk.dev || !(g_vk.dmabuf || g_vk.map_placed) || getenv("LARECOMP_VK_NO_GMEM")) return false;
    const char *e = getenv("LARECOMP_VK_HEAP_DMABUF");
    if (e ? !atoi(e) : !strstr(g_vk.props.deviceName, "RADV")) return false;
    const uint64_t size = HEAP_MAX;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT;  // (cached: the game's CPU work runs in it)
    int type = -1;
    uint64_t largest = 0;  // (the largest heap with a cached host-visible type, for the log when none is big enough)
    for (uint32_t i = 0; i < mp.memoryTypeCount && type < 0; i++) {
        if ((mp.memoryTypes[i].propertyFlags & want) != want) continue;
        uint64_t hs = mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size;
        if (hs >= size) type = (int)i;
        else if (hs > largest) largest = hs;
    }
    if (type < 0) {
        if (largest)  // (Steam Deck: a 3072 MB system heap)
            LOG("[vk] heap as driver memory: the driver's cached system memory heap (%llu MB) is smaller than the guest heap "
                "(%llu MB)\n", (unsigned long long)(largest >> 20), (unsigned long long)(size >> 20));
        else LOG("[vk] heap as driver memory: no cached host-visible memory type\n");
        return false;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT};
    const char *pe = getenv("LARECOMP_VK_HEAP_PLACED");
    bool placed = g_vk.map_placed && (!pe || atoi(pe));
    if (!placed && !g_vk.dmabuf) { LOG("[vk] heap as driver memory: no way to map it here\n"); return false; }
    bi.pNext = placed ? NULL : &eb;  // (an exportable buffer only for the dma-buf mapping)
    VkBuffer buf;
    if (vkCreateBuffer(g_vk.dev, &bi, NULL, &buf) != VK_SUCCESS) { LOG("[vk] heap as driver memory: buffer failed\n"); return false; }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_vk.dev, buf, &req);
    if (!(req.memoryTypeBits & (1u << type))) {
        LOG("[vk] heap as driver memory: memory type %d not allowed for the buffer\n", type);
        vkDestroyBuffer(g_vk.dev, buf, NULL);
        return false;
    }
    if (req.size != size) {  // (the mapping must cover exactly the heap: guest memory follows it)
        LOG("[vk] heap as driver memory: the allocation would be %llu bytes, not %llu\n", (unsigned long long)req.size,
            (unsigned long long)size);
        vkDestroyBuffer(g_vk.dev, buf, NULL);
        return false;
    }
    VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = buf};
    VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                     .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = placed ? (void *)&ded : (void *)&ex,
                               .allocationSize = req.size, .memoryTypeIndex = (uint32_t)type};
    VkDeviceMemory mem;
    VkResult r = vkAllocateMemory(g_vk.dev, &ai, NULL, &mem);
    if (r != VK_SUCCESS) { LOG("[vk] heap as driver memory: allocation failed (%d)\n", r); vkDestroyBuffer(g_vk.dev, buf, NULL); return false; }
    const char *why = NULL;
    if (placed) {
        why = vkBindBufferMemory(g_vk.dev, buf, mem, 0) != VK_SUCCESS ? "binding failed" : heap_map_placed(mem);
        if (why) {
            LOG("[vk] heap as driver memory: placed mapping at %llx failed (%s): the heap is shared as before\n",
                (unsigned long long)HEAP_BASE, why);
            vkDestroyBuffer(g_vk.dev, buf, NULL);
            vkFreeMemory(g_vk.dev, mem, NULL);
            return false;
        }
        gmem_placed = true;
        goto mapped;
    }
    PFN_vkGetMemoryFdKHR get_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetMemoryFdKHR");
    VkMemoryGetFdInfoKHR fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = mem,
                               .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    int fd = -1;
    if (!get_fd || get_fd(g_vk.dev, &fi, &fd) != VK_SUCCESS || fd < 0) why = "no dma-buf for it";
    // First mapped anywhere, to know the driver lets it be mapped at all: MAP_FIXED unmaps what's at HEAP_BASE before
    // the driver is asked, so a refusal there would leave the heap unmapped (review of 2c27f3d)
    if (!why) {
        void *probe = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (probe == MAP_FAILED) why = strerror(errno);
        else munmap(probe, size);
        if (probe == MAP_FAILED) LOG("[vk] heap as driver memory: the dma-buf can't be mapped (%s)\n", why);
    }
    if (!why && mmap((void *)(uintptr_t)HEAP_BASE, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) !=
                    (void *)(uintptr_t)HEAP_BASE) {
        why = strerror(errno);
        // (the old mapping is gone: plain zeroed memory again, registered for write tracking like a reboot's)
        if (!plat_reset_guest(HEAP_BASE, HEAP_BASE + size)) FATAL("cannot restore the guest heap mapping\n");
    }
    if (fd >= 0) close(fd);  // (the mapping keeps the dma-buf)
    VkResult br = why ? VK_SUCCESS : vkBindBufferMemory(g_vk.dev, buf, mem, 0);
    if (!why && br != VK_SUCCESS) {
        why = "binding failed";
        if (!plat_reset_guest(HEAP_BASE, HEAP_BASE + size)) FATAL("cannot restore the guest heap mapping\n");
    }
    if (why) {
        LOG("[vk] heap as driver memory: mapping at %llx failed (%s): the heap is shared as before\n",
            (unsigned long long)HEAP_BASE, why);
        vkDestroyBuffer(g_vk.dev, buf, NULL);
        vkFreeMemory(g_vk.dev, mem, NULL);
        return false;
    }
mapped:;
    // the game expects a zeroed heap; the kernel hands out zeroed pages, but 8 spread pages are checked to be sure
    static const uint8_t zero_page[4096];
    for (int i = 0; i < 8; i++)
        if (memcmp((const void *)(uintptr_t)(HEAP_BASE + size / 8 * i), zero_page, sizeof zero_page)) {
            LOG("[vk] heap as driver memory: not zeroed, clearing it\n");
            memset((void *)(uintptr_t)HEAP_BASE, 0, size);
            break;
        }
    gmem_buf = buf;
    gmem_mem = mem;
    gmem_size = size;
    gmem_dmabuf = true;
    LOG("[vk] guest heap as driver memory (%s, memory type %d): %llu MB at %llx - all of it is held in system memory "
        "once the GPU first uses it\n", gmem_placed ? "placed map" : "dma-buf", type, (unsigned long long)(size >> 20), (unsigned long long)HEAP_BASE);
    return true;
}
#endif

// The guest heap's GPU memory on Linux: the driver's own (above) where it can be, else host-pointer imports - the
// whole heap (gmem_import) or 16 MB chunks as draws first bind them (gc_lazy: RADV without the dma-buf heap, Steam
// Deck; GPUs whose storage buffers can't span the heap). On Mesa's drivers every such import is a userptr buffer, and
// page write tracking (platform/linux.c: userfaultfd write protection, PAGEMAP_SCAN) changes the protection of pages
// inside them: each tracking query over vertex or texture memory is an MMU-notifier invalidation of the imports it
// overlaps, so the kernel driver waits for the GPU to finish with them and checks their pages again at the next
// submission (amdgpu; the CPU and GPU stop overlapping - Steam Deck, issue #181: 30 fps on Beta 4, 20 with tracking
// over the whole-heap import). Chunks only make the re-check smaller, not the wait. So with any userptr import,
// tracking is off, as on Beta 4 (the vertex cache and the texture checks go without it). NVIDIA's own driver pins
// the pages instead and keeps it. Decided here, at boot, before any query protected a page. LARECOMP_WRITE_TRACKING=1:
// on anyway.
static bool gmem_userptr;  // the heap is imported by host pointer: userptr (vertex cache default, vcache_lookup)
static bool heap_is_userptr(void) { return gmem_userptr; }  // (the storage write watch: off over a userptr import)
static const char *heap_tracking_reason(void) {  // (overlap_on: why there is no page write tracking here)
    if (gmem_userptr) return "the guest heap is a userptr import, which the OS can't track";
#ifdef __linux__
    if (gmem_dmabuf) return "the guest heap is driver memory (dma-buf), which the OS can't track";
    return "no userfaultfd write tracking or PAGEMAP_SCAN (Linux 6.7+)";
#else
    return "the guest window has no write watch";
#endif
}
// The whole heap as one userptr import (not chunks) on AMD: each submission has amdgpu check all of its pages
static bool heap_whole_userptr(void) { return gmem_userptr && !gc_lazy && g_vk.props.vendorID == 0x1002; }

void backend_guest_heap_prepare(void) {
    gmem_userptr = false;
#ifdef __linux__
    guest_heap_dmabuf();
#endif
    if (!g_vk.dev) return;
    gmem_chunk_shift();  // (the chunk / lazy decision, before the first draw binds anything)
#ifdef __linux__
    // driver memory: write tracking can't watch that mapping (its queries over the heap would fail; PAGEMAP_SCAN
    // refuses a range without userfaultfd write protection), so it's off there too
    if (gmem_dmabuf) { plat_write_tracking_off("the guest heap is driver memory"); return; }
    if (!g_vk.host_import || getenv("LARECOMP_VK_NO_GMEM") || g_vk.props.vendorID == 0x10DE) return;
    gmem_userptr = true;
    plat_write_tracking_off(gc_lazy ? "the guest heap is imported in userptr chunks" : "the guest heap is a userptr import");
#endif
}

static uint64_t gmem_tried;  // heap size the import was last attempted for (gmem_info imports / re-imports)

// (the shadow workers, the render thread with async runs: a storage buffer binding on most draws took the lock for
// it) the import as it is, read without the lock (gmem_seq: odd while gmem_import replaces it), when it covers the
// window already; false: gmem_info under the lock
static bool gmem_info_quick(VkDescriptorBufferInfo *o) {
    uint32_t s1 = atomic_load_explicit(&gmem_seq, memory_order_acquire);
    if (s1 & 1) return false;
    VkBuffer b = gmem_buf;
    uint64_t sz = gmem_size, lo = gmem_lo;
    atomic_thread_fence(memory_order_acquire);
    if (atomic_load_explicit(&gmem_seq, memory_order_relaxed) != s1 || !b) return false;
    uint64_t want = (gwin_want() + 0xFFFF) & ~0xFFFFull;
    if (want > gwin_max) want = gwin_max;
    if (want > lo + sz) return false;
    *o = (VkDescriptorBufferInfo){b, 0, sz};
    return true;
}
static VkDescriptorBufferInfo gmem_info(void) {
    static int off = -1;
    if (off < 0) off = ENV("LARECOMP_VK_NO_GMEM") != NULL || !g_vk.host_import || gc_lazy;  // (lazy: chunks instead)
    uint64_t want = (gwin_want() + 0xFFFF) & ~0xFFFFull;
    if (want > gwin_max) want = gwin_max;
    uint64_t wlo, wlen;
    gmem_window_get(&wlo, &wlen);
    if (!gmem_window) wlo = 0;
    static int nlog;
    if (!nlog++) LOG("[vk] gmem: window %llu MB, import %s\n", (unsigned long long)(want >> 20), off ? "off" : "on");
    if (!off && want > gmem_lo + gmem_size && !gmem_import(wlo, want - wlo)) {
        LOG("[vk] guest heap import failed (%llu MB): shader global memory reads return zero\n", (unsigned long long)(want >> 20));
        off = 1;
    }
    if (gmem_buf) return (VkDescriptorBufferInfo){gmem_buf, 0, gmem_size};
    static uint32_t zero[1024];
    VkSpan s = vk_transient_copy(zero, sizeof zero, 256);
    return (VkDescriptorBufferInfo){s.buf, s.off, sizeof zero};
}

// ---- the guest heap for shaders in chunks (shader.h maxwell_gmem_chunk_shift)
// Chunks for two reasons:
// - a GPU whose storage buffers can't span HEAP_MAX (maxStorageBufferRange: 1 GB on Intel): binding one range past
//   that limit is out of spec, and the UHD 630's driver then read garbage beyond 1 GB: dark lighting, black objects
//   (issue #8). Lazy chunks as below (device.c lazy_heap_predicted); where they aren't possible (or with
//   LARECOMP_VK_GMEM_LAZY=0), windows of the one imported heap, one binding each;
// - lazy import (gmem_lazy): on Linux/AMD the import is a userptr buffer and amdgpu checks every page of such a buffer
//   again on each submission - 3.2 GB of heap, ~45 ms of driver time a frame (issues #40, #51). The dma-buf heap
//   (guest_heap_dmabuf) avoids that, but needs a cached host-visible memory type that can hold HEAP_MAX, and APUs
//   have none (Steam Deck: a 3072 MB system heap; Ryzen 3200G: 3172 MB, so the 3220 MB import is also over its GTT
//   budget: issues #40, #92). The GPU itself reads only a little of the heap in place: 73 MB over a four-area tour
//   (vertex and index data 67 MB, storage buffers 9 MB), 15-28 MB a frame, scattered over the whole 3.25 GB (measured
//   on the Metal backend, which binds the same ranges). So each 16 MB chunk is imported on its own when a draw or
//   dispatch first binds a range in it (23 chunks, 368 MB, over that tour), and only the imported chunks exist for the
//   driver to check: a fraction of the pages, under the GTT budget, and still the game's own memory (nothing to copy
//   back). Vertex and index data bind their chunk directly; shaders (LDG/STG) take the chunk's device address and
//   size from a table (GChunks, rewritten when a chunk is imported) and go through a buffer reference after a bounds
//   check: a chunk not imported has size 0, the access reads 0 / drops the store, and the shader counts it in GMiss
//   with the lowest word index hit: such a chunk (a pointer the game keeps in a uniform buffer, which no bind told us
//   about) is imported for the next frame, and the count is in the stats (after a chunk's first use it should stay 0).
//   Buffer device addresses are enabled only where this mode will be used (device.c lazy_heap_predicted): on RADV
//   they put every allocation on the driver's global BO list.
//   Default on RADV when the dma-buf heap doesn't apply, wherever one storage buffer can't span the heap and where the
//   system memory heap (or its budget) can't hold one import of it (8 GB laptops: issue #205), and where one import
//   was tried at the start and ran out of memory (whole_import_probe_oom; LARECOMP_VK_SIM_HEAP_OOM=1 simulates it);
//   LARECOMP_VK_GMEM_LAZY=1 anywhere (tests), =0 off.
// LARECOMP_VK_GMEM_CHUNK_MB=<n> forces chunks of n MB (a power of two, 1 GB at most), =0 one buffer.
#define GC_MAX GMEM_CHUNKS_MAX
static struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint64_t size, addr;  // (addr: the chunk's device address, lazy mode)
} gc[GC_MAX];
static int gc_n;             // chunks over HEAP_MAX
static uint32_t gc_shift;    // 0: one buffer
static bool gc_lazy;         // imports per chunk (else windows of gmem_buf)
static VkBuffer gc_dummy, gc_miss_buf;
static VmaAllocation gc_dummy_mem, gc_miss_mem;
static uint32_t *gc_miss;    // GMiss, host-visible: per chunk, accesses that found no chunk, then the lowest word index
static VkDescriptorSetLayout gc_layout;
static VkDescriptorSet gc_set;
static uint64_t gc_set_seq;
static uint32_t gc_gen, gc_set_gen;  // (gen: bumped when a chunk is imported; the set is rebuilt per submission and gen)
static int gc_scanned;                // chunks imported for a pointer found in a uniform buffer
static uint8_t gc_remembered[GC_MAX]; // chunks shaders reached without a bind (a miss, a pointer): imported again after a reboot
                                      // and, through cache/gmem_chunks.txt, at the next start (a machine misses a chunk once)
static bool gc_file_loaded;
static uint64_t gc_remembered_heap;   // the heap size the remembered chunks were imported at this boot (0: not yet)

// cache/gmem_chunks.txt: one heap byte offset (hex) per line of a chunk shaders reached without a bind. Offsets, not
// chunk numbers: the chunk size depends on the GPU. Loaded once, into gc_remembered; written whenever one is added.
static void gc_remembered_load(void) {
    if (gc_file_loaded) return;
    gc_file_loaded = true;
    FILE *f = fopen("cache/gmem_chunks.txt", "r");
    if (!f) return;
    char line[64];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long long off = strtoull(line, NULL, 16);
        int k = (int)(off >> gc_shift);
        if (k >= 0 && k < gc_n && !gc_remembered[k]) { gc_remembered[k] = 1; n++; }
    }
    fclose(f);
    if (n) LOG("[vk] guest heap: %d chunks shaders reached without a bind last time (cache/gmem_chunks.txt): imported at the start\n", n);
}
static void gc_remembered_save(void) {
    mkdir("cache", 0755);
    FILE *f = fopen("cache/gmem_chunks.txt.tmp", "w");
    if (!f) return;
    for (int k = 0; k < gc_n; k++)
        if (gc_remembered[k]) fprintf(f, "%llx\n", (unsigned long long)((uint64_t)k << gc_shift));
    fclose(f);
    remove("cache/gmem_chunks.txt");
    rename("cache/gmem_chunks.txt.tmp", "cache/gmem_chunks.txt");
}
static void gc_remember(int k) {
    if (gc_remembered[k]) return;
    gc_remembered[k] = 1;
    gc_remembered_save();
}
static int gc_imported;
static uint64_t gc_imported_bytes, stat_gc_misses;

static bool gmem_lazy(void) {
    static int lazy = -1;
    if (lazy >= 0) return lazy;
    lazy = g_vk.lazy_heap;
    if (!lazy && g_vk.heap_probe && !getenv("LARECOMP_VK_NO_GMEM")) lazy = whole_import_probe_oom();
    if (lazy && (!g_vk.host_import || !g_vk.bda || !g_vk.ssbo_chunks)) {
        LOG("[vk] lazy heap import: not possible here (%s)\n", !g_vk.host_import ? "no host memory import" : !g_vk.bda ? "no buffer device addresses" : "no stores from fragment / vertex shaders");
        lazy = 0;
    }
    return lazy;
}

// Decided once (pthread_once): the render thread's draws, the translation and pipeline workers and the boot
// (backend_guest_heap_prepare) all ask, and a second caller must not see a half-made decision (shift is 0 until the
// fit loop below ends - a shader translated then would use one buffer while draws bind chunks).
static int gc_shift_decided = -1;
static void gmem_chunk_decide(void);
static pthread_once_t gc_decide_once = PTHREAD_ONCE_INIT;
static uint32_t gmem_chunk_shift(void) {
    pthread_once(&gc_decide_once, gmem_chunk_decide);
    return (uint32_t)gc_shift_decided;
}
static void gmem_chunk_decide(void) {
    int shift;
    // Pokémon Z-A has no heap: its window (gwin: physical memory) is imported as one buffer, as before the chunks -
    // they are laid over HEAP_BASE / HEAP_MAX (data root's extracted/: asked here, the boot may come after this)
    extern bool game_files_are_la(const char *game_dir);
    if (!g_game_is_la || !game_files_are_la("extracted")) {
        gc_lazy = maxwell_gmem_lazy = maxwell_gmem_fs_miss = false;
        gc_shift = 0, gc_n = 0, gc_shift_decided = 0;
        LOG("[vk] guest memory: one import of the game's window (not Link's Awakening: no heap chunks)\n");
        return;
    }
    const char *e = getenv("LARECOMP_VK_GMEM_CHUNK_MB");
    uint64_t lim = g_vk.props.limits.maxStorageBufferRange;
    gc_lazy = gmem_lazy();
    shift = 0;
    if (e) { uint64_t b = strtoull(e, NULL, 10) << 20; if (b) shift = 63 - __builtin_clzll(b); }
    else if (gc_lazy) shift = 24;
    else if (HEAP_MAX > lim && lim) shift = 63 - __builtin_clzll(lim);
    if (shift > 30) shift = 30;  // (1 GB at most: NVIDIA computes a block's length in signed 32 bits, a 2 GB range reads as negative)
    // The table holds GC_MAX chunks at most, and a stage's storage buffers (the windows, gfast, binding 17) its limit
    // (Intel: 200): a chunk size that doesn't fit grows until it does (4 MB asked for -> 16 MB). One buffer only where
    // one buffer is within maxStorageBufferRange.
    int asked = shift, n = 0;
    uint32_t per_stage = g_vk.props.limits.maxPerStageDescriptorStorageBuffers, per_set = g_vk.props.limits.maxDescriptorSetStorageBuffers;
    if (shift && gc_lazy && !g_vk.ssbo_chunks) {
        LOG("[vk] guest heap chunks: not possible here (no stores from fragment / vertex shaders): one buffer\n");
        shift = 0;
    }
    while (shift) {
        n = (int)((HEAP_MAX + (1ull << shift) - 1) >> shift);
        bool fits = n <= GC_MAX && (gc_lazy || ((uint32_t)n + 8 <= per_stage && (uint32_t)n + 1 <= per_set));
        if (fits) break;
        if ((1ull << shift) >= HEAP_MAX) { shift = 0; break; }  // (not even one chunk fits: one buffer)
        shift++;
    }
    if (shift != asked && asked)
        LOG("[vk] guest heap chunks of %llu MB don't fit here (%d chunks at most, %u storage buffers a stage, %u a set): %s\n",
            (unsigned long long)(1ull << asked) >> 20, GC_MAX, per_stage, per_set, shift ? "larger chunks" : "one buffer");
    if (!shift && HEAP_MAX > lim && lim) LOG("[vk] warning: one storage buffer over the heap exceeds this GPU's %llu MB range\n", (unsigned long long)lim >> 20);
    if (!shift) gc_lazy = false;
    gc_shift = (uint32_t)shift;
    gc_n = shift ? n : 0;
    maxwell_gmem_lazy = gc_lazy;
    maxwell_gmem_fs_miss = gc_lazy && getenv("LARECOMP_VK_GMEM_FS_MISS") && atoi(getenv("LARECOMP_VK_GMEM_FS_MISS")) != 0;
    if (maxwell_gmem_fs_miss) LOG("[vk] lazy chunks: every fragment shader counts misses, no early tests (LARECOMP_VK_GMEM_FS_MISS)\n");
    if (shift) LOG("[vk] guest heap bound to shaders in %llu MB %s (storage buffers up to %llu MB here)\n",
                   (unsigned long long)(1ull << shift) >> 20, gc_lazy ? "chunks, imported as the game binds them" : "windows", (unsigned long long)lim >> 20);
    gc_shift_decided = shift;
}

// The heap mapped so far, as the imports see it (SetHeapSize; pages beyond it don't exist yet).
static uint64_t gmem_heap_size(void) {
    uint64_t want = (kernel_heap_size() + 0xFFFF) & ~0xFFFFull;
    uint64_t ring = snap_ring_end();  // (overlap: the storage copies' ring right after the heap is the GPU's too)
    if (ring > want) want = ring;
    return want > HEAP_MAX ? HEAP_MAX : want;
}

static bool host_import(uint64_t off, uint64_t size, VkBuffer *buf, VkDeviceMemory *mem, bool bda);

// Lazy mode: chunk k imported (again, if the heap grew into it since). False: not mapped yet / the import failed.
static bool gc_import(int k) {
    uint64_t off = (uint64_t)k << gc_shift, heap = gmem_heap_size();
    if (k < 0 || k >= gc_n || off >= heap) return false;
    uint64_t size = heap - off < (1ull << gc_shift) ? heap - off : 1ull << gc_shift;
    if (gc[k].buf && gc[k].size >= size) return true;
    VkBuffer buf;
    VkDeviceMemory mem;
    if (!host_import(off, size, &buf, &mem, true)) {
        static int warned;
        if (warned++ < 8) LOG("[vk] guest heap chunk %d (%llu MB at +%llu MB): import failed\n", k, (unsigned long long)size >> 20, (unsigned long long)off >> 20);
        return false;
    }
    if (gc[k].buf) {  // (the heap grew into the chunk: the pending submission may still use the old import)
        vk_defer_buffer(gc[k].buf, gc[k].mem);
        gc_imported_bytes -= gc[k].size;
    } else {
        gc_imported++;
    }
    VkBufferDeviceAddressInfo ai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buf};
    gc[k].buf = buf; gc[k].mem = mem; gc[k].size = size;
    gc[k].addr = vkGetBufferDeviceAddress(g_vk.dev, &ai);
    gc_imported_bytes += size;
    gc_gen++;
    return true;
}

// (snap_on, lazy chunks) The chunk holding the end of the game's heap may have been imported before the ring was
// decided, short of it: imported again over the ring (gc_import replaces it; the old import goes once its submissions
// are done). snap_state is set before this runs, so gmem_heap_size's answer includes the ring here.
static void snap_ring_import_tail(void) {
    if (!gc_lazy || !gc_shift || gwin_.phys) return;  // (a physical-memory window: its ring is in the one import)
    for (int k = (int)(snap_base >> gc_shift); k <= (int)((snap_base + SNAP_RING - 1) >> gc_shift) && k < gc_n; k++)
        if (gc[k].buf && gc[k].size < gmem_heap_size() - ((uint64_t)k << gc_shift)) gc_import(k);
}

// Pointers the game keeps in its uniform buffers (the lighting pass's light list, the grass's instance tables): the
// chunk a shader reaches that way has no storage buffer bound over it, so it is imported from the uniform buffer's
// contents - which the per-draw copy reads anyway - before the draw. Without this the first frame in such a chunk
// read 0 (chunk 47 at +752 MB at every scene entry, chunk 50 at the first grass cut: 576 misses over a tour). An
// 8-byte-aligned word pair that happens to look like a heap address imports a chunk for nothing: bounded, and logged.
#define GC_SCAN_MAX 48
static int gc_scan_warned;
static void gc_scan_pointers(const uint32_t *w, uint64_t bytes) {
    uint64_t heap = gmem_heap_size();
    prof_scan_calls++;
    prof_scan_bytes += bytes;
    for (uint64_t i = 0; i + 1 < bytes / 4; i += 2) {
        uint64_t v = (uint64_t)w[i] | (uint64_t)w[i + 1] << 32;
        if (v < HEAP_BASE || v - HEAP_BASE >= heap) continue;
        int k = (int)((v - HEAP_BASE) >> gc_shift);
        if (k >= gc_n || gc[k].buf) continue;
        if (gc_scanned >= GC_SCAN_MAX) {
            if (!gc_scan_warned++) LOG("[vk] guest heap: %d chunks imported for pointers in uniform buffers: no more\n", gc_scanned);
            return;
        }
        if (gc_import(k)) {
            gc_scanned++;
            gc_remember(k);
            LOG("[vk] guest heap chunk %d (+%llu MB): a pointer in a uniform buffer (%llx): imported\n", k,
                (unsigned long long)((uint64_t)k << gc_shift) >> 20, (unsigned long long)v);
        }
    }
}

// Whether a pointer scan can still import anything: some chunk of the heap mapped so far isn't imported (and the
// scan's import limit isn't used up). Once every chunk is in, the scans are skipped (guest_copy_ex).
static bool gc_scan_needed(void) {
    static uint64_t c_heap;
    static uint32_t c_gen;
    static int c_scanned = -1, c_warned;
    static bool c_need = true;
    uint64_t heap = gmem_heap_size();
    if (heap != c_heap || gc_gen != c_gen || gc_scanned != c_scanned || gc_scan_warned != c_warned) {
        c_heap = heap; c_gen = gc_gen; c_scanned = gc_scanned; c_warned = gc_scan_warned;
        c_need = false;
        if (gc_scanned < GC_SCAN_MAX || !gc_scan_warned) {
            int n = (int)((heap + (1ull << gc_shift) - 1) >> gc_shift);
            if (n > gc_n) n = gc_n;
            for (int k = 0; k < n && !c_need; k++) c_need = !gc[k].buf;
        }
    }
    return c_need;
}

// A heap range a draw or dispatch binds: its chunks get imported (lazy mode), and the chunk after it as well - the
// game's storage buffers are bound with sizes the shaders read past (the lighting pass's tile list: the misses sat
// 1-9 MB into the chunk after the bound range), and a size of 0 means "the rest of the buffer".
static void gc_touch(uint64_t off, uint64_t size) {
    if (!gc_lazy || off >= HEAP_MAX) return;
    int k0 = (int)(off >> gc_shift), k1 = (int)((off + (size ? size : 1) - 1) >> gc_shift) + 1;
    if (k1 >= gc_n) k1 = gc_n - 1;
    for (int k = k0; k <= k1; k++) if (!gc[k].buf) gc_import(k);
}

// The chunk set's layout: lazy mode the table (GChunks), GMiss and the draw's chunks (GDir<k>); windows one storage
// binding each (shader.h).
// (once: pipeline layouts are made on the build workers, sets on the render thread)
static pthread_once_t gc_layout_once = PTHREAD_ONCE_INIT;
static void gmem_chunk_layout_make(void);
static VkDescriptorSetLayout gmem_chunk_layout(void) {
    pthread_once(&gc_layout_once, gmem_chunk_layout_make);
    return gc_layout;
}
static void gmem_chunk_layout_make(void) {
    gmem_chunk_shift();  // (gc_lazy / gc_n decided)
    VkShaderStageFlags st_ = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    static VkDescriptorSetLayoutBinding lb[GC_MAX];
    uint32_t nb = 0;
    if (gc_lazy) {
        lb[nb++] = (VkDescriptorSetLayoutBinding){GMEM_CHUNK_BINDING, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, st_, NULL};
        lb[nb++] = (VkDescriptorSetLayoutBinding){GMEM_MISS_BINDING, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, st_, NULL};
        for (int k = 0; k < GMEM_DIRECT_MAX; k++)
            lb[nb++] = (VkDescriptorSetLayoutBinding){GMEM_DIRECT_BINDING + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, st_, NULL};
    } else {
        for (int k = 0; k < gc_n; k++) lb[nb++] = (VkDescriptorSetLayoutBinding){GMEM_CHUNK_BINDING + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, st_, NULL};
    }
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = nb, .pBindings = lb};
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &ci, NULL, &gc_layout));
}

// The chunk set for the pending submission: the table of imported chunks, GMiss and the draw's chunks dk[k] (-1: none)
// as GDir<k> (lazy), or the windows. Lazy sets are kept per submission and chunk generation, keyed by dk: a frame has a
// handful of storage buffer chunk combinations, so a draw binds a set it already has (no allocation, no update).
#define GC_SET_CACHE 32
static uint64_t gc_direct_bytes(int k) {  // (GDir<k>'s range: the chunk, within the storage buffer range limit)
    uint64_t lim = g_vk.props.limits.maxStorageBufferRange;
    return lim && gc[k].size > lim ? lim & ~3ull : gc[k].size;
}
static VkDescriptorSet gmem_chunk_set(const int dk[GMEM_DIRECT_MAX]) {
    if (!gc_dummy) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO, .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &gc_dummy, &gc_dummy_mem, &info));
        memset(info.pMappedData, 0, 4096);
        bi.size = sizeof(uint32_t) * 2 * GC_MAX;  // n[GC_MAX] accesses, w[GC_MAX] lowest word index hit
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        ai.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &gc_miss_buf, &gc_miss_mem, &info));
        gc_miss = info.pMappedData;
        memset(gc_miss, 0, sizeof(uint32_t) * GC_MAX);
        memset(gc_miss + GC_MAX, 0xFF, sizeof(uint32_t) * GC_MAX);
    }
    uint64_t seq = vk_pending_seq();
    // A boot, or a reboot: the chunks shaders reached without a bind last time (this process, or cache/gmem_chunks.txt),
    // before they miss again. Once a boot per heap size (a chunk beyond the heap mapped so far comes when it grows) -
    // not "while nothing is imported": copies and storage buffers import chunks before the first set is built.
    if (gc_lazy && gc_remembered_heap != gmem_heap_size()) {
        gc_remembered_heap = gmem_heap_size();
        gc_remembered_load();
        for (int k = 0; k < gc_n; k++) if (gc_remembered[k] && !gc[k].buf) gc_import(k);
    }
    static struct { VkDescriptorSet set; int dk[GMEM_DIRECT_MAX]; } cache[GC_SET_CACHE];
    static int ncache, next;
    static VkSpan tab_span;  // (the table for this submission and generation, shared by its sets)
    // (windows: also when the heap was imported again since - guest_direct / gfast_prepare re-import as it grows
    // mid-submission, and a set still on the old import read zeros past its end for the rest of the submission)
    static VkBuffer gc_set_heap;
    if (gc_set_seq != seq || gc_set_gen != gc_gen || (!gc_lazy && gc_set_heap != gmem_buf)) { ncache = 0; next = 0; gc_set = VK_NULL_HANDLE; }
    if (!gc_lazy) {
        if (gc_set) return gc_set;
    } else {
        for (int j = 0; j < ncache; j++)
            if (!memcmp(cache[j].dk, dk, sizeof cache[j].dk)) return cache[j].set;
    }
    static VkDescriptorBufferInfo bi[GC_MAX + 2 + GMEM_DIRECT_MAX];
    static VkWriteDescriptorSet w[GC_MAX + 2 + GMEM_DIRECT_MAX];
    VkDescriptorSet set = vk_alloc_set(gmem_chunk_layout());
    uint32_t nw = 0;
    if (gc_lazy) {  // GChunks: uvec4 per chunk = device address lo, hi, words (0: not imported), 0
        if (!ncache) {
            static uint32_t tab[GC_MAX * 4];
            for (int k = 0; k < gc_n; k++) {
                tab[k * 4 + 0] = (uint32_t)gc[k].addr; tab[k * 4 + 1] = (uint32_t)(gc[k].addr >> 32);
                tab[k * 4 + 2] = gc[k].buf ? (uint32_t)(gc[k].size >> 2) : 0; tab[k * 4 + 3] = 0;
            }
            tab_span = vk_transient_copy(tab, sizeof tab, 256);
        }
        bi[0] = (VkDescriptorBufferInfo){tab_span.buf, tab_span.off, sizeof(uint32_t) * GC_MAX * 4};
        bi[1] = (VkDescriptorBufferInfo){gc_miss_buf, 0, sizeof(uint32_t) * 2 * GC_MAX};
        w[nw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = GMEM_CHUNK_BINDING,
                                         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &bi[0]};
        w[nw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = GMEM_MISS_BINDING,
                                         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1]};
        for (int j = 0; j < GMEM_DIRECT_MAX; j++) {  // (GDir<j>: the chunk's own import, whole; unused: the zero buffer)
            int k = dk[j];
            bi[2 + j] = k >= 0 && k < gc_n && gc[k].buf ? (VkDescriptorBufferInfo){gc[k].buf, 0, gc_direct_bytes(k)}
                                                       : (VkDescriptorBufferInfo){gc_dummy, 0, 4096};
            w[nw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = GMEM_DIRECT_BINDING + j,
                                             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[2 + j]};
        }
        int slot = ncache < GC_SET_CACHE ? ncache++ : (next++ % GC_SET_CACHE);
        cache[slot].set = set;
        memcpy(cache[slot].dk, dk, sizeof cache[slot].dk);
    } else {  // windows of the one import; one past it is the dummy (reads 0, as the zero buffer before the import)
        VkDescriptorBufferInfo all = gmem_info();
        for (int k = 0; k < gc_n; k++) {
            uint64_t off = (uint64_t)k << gc_shift;
            if (all.buffer == gmem_buf && gmem_buf && off < gmem_size)
                bi[k] = (VkDescriptorBufferInfo){gmem_buf, off, gmem_size - off < (1ull << gc_shift) ? gmem_size - off : 1ull << gc_shift};
            else bi[k] = (VkDescriptorBufferInfo){gc_dummy, 0, 4096};
            w[nw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = GMEM_CHUNK_BINDING + k,
                                             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[k]};
        }
    }
    rec_UpdateDescriptorSets(nw, w);
    gc_set = set; gc_set_seq = seq; gc_set_gen = gc_gen; gc_set_heap = gmem_buf;
    return set;
}

// Lazy mode, a draw: the imported chunks under its vertex / fragment storage buffers, in slot order (shader.h
// GMEM_DIRECT_MAX): gd[] for the shaders' GDirect push constants (the range each GDir<k> covers), dk[] the chunks for
// the set's GDir<k> bindings. A range over two chunks gives both. Accesses there are storage buffer loads, not the table.
// LARECOMP_VK_GMEM_DIRECT=0: none (every access through the table, as before; no shader change).
static void gmem_direct_pick(uint32_t stages, uint32_t gd[GMEM_DIRECT_MAX * 4], int dk[GMEM_DIRECT_MAX]) {
    static int on = -1;
    if (on < 0) {
        on = !getenv("LARECOMP_VK_GMEM_DIRECT") || atoi(getenv("LARECOMP_VK_GMEM_DIRECT")) != 0;
        if (!on) LOG("[vk] lazy chunks: storage buffer chunks through the table too (LARECOMP_VK_GMEM_DIRECT=0)\n");
    }
    memset(gd, 0, sizeof(uint32_t) * GMEM_DIRECT_MAX * 4);
    for (int j = 0; j < GMEM_DIRECT_MAX; j++) dk[j] = -1;
    int n = 0;
    for (int stage = 0; on && stage < 2; stage++) {
        if (!(stages & (1u << stage))) continue;
        for (int i = 0; i < 16 && n < GMEM_DIRECT_MAX; i++) {
            uint64_t a = sbufs(stage)[i].addr, sz = sbufs(stage)[i].size;  // (overlap: a snapshotted range's copy)
            if (a < HEAP_BASE || a - HEAP_BASE >= HEAP_MAX) continue;
            uint64_t off = a - HEAP_BASE;
            int k0 = (int)(off >> gc_shift), k1 = (int)((off + (sz ? sz : 1) - 1) >> gc_shift);
            for (int k = k0; k <= k1 && k < gc_n && n < GMEM_DIRECT_MAX; k++) {
                if (!gc[k].buf) continue;
                int j = 0;
                while (j < n && dk[j] != k) j++;
                if (j < n) continue;
                dk[n] = k;
                gd[n * 4 + 0] = (uint32_t)(((uint64_t)k << gc_shift) >> 2);
                gd[n * 4 + 1] = (uint32_t)(gc_direct_bytes(k) >> 2);
                gd[n * 4 + 2] = (uint32_t)gc[k].addr;
                gd[n * 4 + 3] = (uint32_t)(gc[k].addr >> 32);
                n++;
            }
        }
    }
}

// Binds the chunk set for a draw / dispatch whose shaders (stages: a mask of 1 << stage) use global memory: their storage
// buffers' chunks first.
static __thread struct { VkDescriptorSet set; uint32_t index; bool direct; uint32_t gd[GMEM_DIRECT_MAX * 4]; } gm_last;  // (gmem_chunk_bind's last: per thread, shadow workers bind too)
static void gmem_chunk_bind(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set_index, uint32_t stages) {
    static int log_sb = -1;  // LARECOMP_LOG_GMEM_SBUF=1: each distinct storage buffer bound to a gmem shader, once
    if (log_sb < 0) log_sb = getenv("LARECOMP_LOG_GMEM_SBUF") != NULL;
    for (int stage = 0; stage < 6; stage++) {
        if (!(stages & (1u << stage))) continue;
        for (int i = 0; i < 16; i++) {
            uint64_t a = sbufs(stage)[i].addr, sz = sbufs(stage)[i].size;  // (overlap: a snapshotted range's copy - its chunk)
            if (a < HEAP_BASE || a - HEAP_BASE >= HEAP_MAX) continue;
            if (log_sb) {
                static uint64_t seen[64]; static int nseen;
                uint64_t key = a ^ (sz << 40) ^ ((uint64_t)(stage * 16 + i) << 56);
                int j = 0; while (j < nseen && seen[j] != key) j++;
                if (j == nseen && nseen < 64) { seen[nseen++] = key; LOG("[vk] gmem sbuf stage %d slot %d: %llx + %llx (chunk %d)\n", stage, i, (unsigned long long)a, (unsigned long long)sz, (int)((a - HEAP_BASE) >> gc_shift)); }
            }
            gc_touch(a - HEAP_BASE, sz);
        }
    }
    uint32_t gd[GMEM_DIRECT_MAX * 4];
    int dk[GMEM_DIRECT_MAX] = {-1, -1, -1, -1};
    bool direct = gc_lazy && bp == VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (direct) gmem_direct_pick(stages, gd, dk);
    VkDescriptorSet set = gmem_chunk_set(dk);
    rec_CmdBindDescriptorSets(cb, bp, layout, set_index, 1, &set, 0, NULL);
    if (direct) rec_CmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof gd, gd);
    gm_last.set = set, gm_last.index = set_index, gm_last.direct = direct;  // (planar reflections replay the draw)
    if (direct) memcpy(gm_last.gd, gd, sizeof gd);
}

// Once a frame (the present's submission): a chunk shaders missed is imported for the next frame; the lowest address
// that missed is logged (a pointer the game keeps in a uniform buffer, which no bind told us about).
static void gc_frame(void) {
    if (!gc_shift || !gc_miss || !gc_lazy) return;
    for (int k = 0; k < gc_n; k++) {
        uint32_t m = gc_miss[k], w = gc_miss[GC_MAX + k];
        if (!m) continue;
        gc_miss[k] = 0;
        gc_miss[GC_MAX + k] = 0xFFFFFFFFu;
        stat_gc_misses += m;
        static int nlog;
        if (nlog++ < 40) LOG("[vk] guest heap chunk %d (+%llu MB): %u shader accesses before its import, lowest %llx: imported for the next frame\n", k,
                             (unsigned long long)((uint64_t)k << gc_shift) >> 20, m, (unsigned long long)HEAP_BASE + ((uint64_t)w << 2));
        gc_touch((uint64_t)k << gc_shift, 1);
        gc_remember(k);
    }
}

// In lazy mode the heap-side offsets of a copy are in chunks: one vkCmdCopyBuffer per chunk piece. other = the GPU side.
static void gmem_copy(VkCommandBuffer cb, VkBuffer other, const VkBufferCopy *cp, int n, bool to_heap) {
    if (!gc_lazy) {
        if (to_heap) rec_CmdCopyBuffer(cb, other, gmem_buf, (uint32_t)n, cp);
        else rec_CmdCopyBuffer(cb, gmem_buf, other, (uint32_t)n, cp);
        return;
    }
    for (int i = 0; i < n; i++) {
        uint64_t h = to_heap ? cp[i].dstOffset : cp[i].srcOffset, o = to_heap ? cp[i].srcOffset : cp[i].dstOffset, left = cp[i].size;
        while (left) {
            int k = (int)(h >> gc_shift);
            uint64_t in = h - ((uint64_t)k << gc_shift), piece = (1ull << gc_shift) - in < left ? (1ull << gc_shift) - in : left;
            if (k < gc_n && (gc[k].buf || gc_import(k)) && in + piece <= gc[k].size) {
                VkBufferCopy c = to_heap ? (VkBufferCopy){o, in, piece} : (VkBufferCopy){in, o, piece};
                if (to_heap) rec_CmdCopyBuffer(cb, other, gc[k].buf, 1, &c);
                else rec_CmdCopyBuffer(cb, gc[k].buf, other, 1, &c);
            }
            h += piece; o += piece; left -= piece;
        }
    }
}

// The heap is there for the GPU over this range (one import, or lazy chunks: imported on demand).
static bool gmem_covers(uint64_t off, uint64_t size) {
    if (gc_lazy) {
        if (off + size > gmem_heap_size()) return false;
        gc_touch(off, size);
        for (int k = (int)(off >> gc_shift); k <= (int)((off + size - 1) >> gc_shift); k++) if (!gc[k].buf) return false;
        return true;
    }
    return gmem_buf && off + size <= gmem_size;
}


// Compute dispatches that use global memory: the imported heap is host memory across PCIe on a discrete GPU, and a
// shader storing to it in a loop crawls (the little chain chomps in Madam MeowMeow's house: ~8 ms a frame on a GTX 1660
// Super and on an RTX 3080 Ti alike, which the game then waits for on its fence - 49-53 fps there on the 1660). Before
// such a dispatch its bound storage buffers (sorted, merged) are copied into a device-local buffer, the address table
// (binding GMAP_BUFFER_INDEX) tells the shader (translate.c, GLSL compute) where each range sits, loads and stores
// inside them use that copy, and a shader that stores gets the ranges copied back to the heap right after it.
// Accesses outside the table use the heap as before. LARECOMP_VK_GMEM_FAST=0: no copy (empty table: everything goes
// to the heap, the old behaviour).
#define GFAST_SIZE (32ull << 20)
static VkBuffer gfast_buf;
static VmaAllocation gfast_mem;
static uint32_t gmap_tab[(GMAP_MAX + 1) * 4];  // this dispatch's table (stage_set copies it into binding 20)
static VkBufferCopy gfast_cp[GMAP_MAX];       // its copies heap -> gfast_buf
static int gfast_n;
// The copy back after a storing dispatch: only the bound buffers' own bytes, gfast_buf -> heap. The merged ranges
// above also span alignment padding and the gaps between buffers - other data, which the CPU (a frame ahead) or
// earlier GPU work may rewrite while the dispatch runs: copied back, the stale bytes clobbered it (Pokémon Z-A: an
// indirect draw's instance count became an old float, 2 billion instances, device lost).
static VkBufferCopy gfast_back[16];
static int gfast_back_n;
// Even the exact bytes clobbered the CPU's writes: the game fills an indirect draw's index count / first index on the
// CPU while the culling dispatch writes its instance count, both in one buffer. A frame ahead, the CPU wrote a new
// mesh's counts there while the GPU worked on its copy; the copy back restored the old mesh's (Pokémon Z-A: ~12% of
// the culled draws used another mesh's index range - sheets and beams across the sky, the player sunk in a grey slab).
// So a twin of the copy, untouched by the dispatch, goes along, and a merge shader writes back only the words that
// differ from it. LARECOMP_VK_GFAST_MERGE=0: the plain copy back.
static VkBuffer gorig_buf;
static VmaAllocation gorig_mem;
static VkPipeline gmerge_pipe;
static VkPipelineLayout gmerge_layout;
static VkDescriptorSetLayout gmerge_set_layout;
static int gmerge_on = -1;
static bool gfast_stores;  // (set before gfast_prepare: the dispatch stores to global memory)

static bool gmerge_init(void) {
    if (gmerge_on < 0) gmerge_on = !ENV("LARECOMP_VK_GFAST_MERGE") || atoi(ENV("LARECOMP_VK_GFAST_MERGE")) != 0;
    if (!gmerge_on || gmerge_pipe) return gmerge_on && gmerge_pipe;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GFAST_SIZE,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &gorig_buf, &gorig_mem, NULL) != VK_SUCCESS) { gmerge_on = 0; return false; }
    VkDescriptorSetLayoutBinding b[3];
    for (int i = 0; i < 3; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b};
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &gmerge_set_layout));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &gmerge_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &pi, NULL, &gmerge_layout));
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = gmerge_layout,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .module = vk_builtin_shader("gfast_merge.comp"), .pName = "main"}};
    VK_CHECK(vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &gmerge_pipe));
    return true;
}

static VkDescriptorBufferInfo gfast_info(void) {
    if (!gfast_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = GFAST_SIZE,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateBuffer(g_vk.vma, &bi, &ai, &gfast_buf, &gfast_mem, NULL));
    }
    return (VkDescriptorBufferInfo){gfast_buf, 0, GFAST_SIZE};
}

// ---- transform feedback mirror. Pokémon Z-A skins its characters with transform feedback - ~140 sessions a frame,
// their buffers in one ~6 MB arena - and its draws then read the results as vertex streams. Written into the guest
// heap's import (host memory), every vertex was a scattered write across PCIe: 3 ms of GPU time a frame (7 with
// mods/MaxLOD). Instead the sessions write a device-local mirror of the arena (XFB_BUFFER_INDEX), the vertex streams
// inside parts written that way read it (guest_direct), and at each session's end the parts written are copied back
// to the heap (one small DMA transfer), for anything else that reads them there. Ranges outside the mirror's window
// (xm_place) go to the heap as before. LARECOMP_VK_XFB_MIRROR=0: off, =1: on. Off by default on AMD GPUs (#22: an RX 9070 XT
// lost the device on an invalid GPU read in exactly these skinning draws, with robust access on and submissions made
// one at a time) - their transform feedback goes to the heap's import as before the mirror.
#define XM_SIZE (512ull << 20)  // (xm_place: 64 MB until 2026-11, then 256; a town's arena can come in two parts 140 MB apart, and after a long
// play over several areas the characters' sessions spread over ~340 MB: up to a third of a frame's ~450 sessions fell outside
// 256 MB, wrote and read the heap in system memory, and each cost the GPU ~15x as much - 30-45 fps in towns on an RTX
// 4060, where a fresh session held 60. 256 MB more GPU memory: that 4060 had 2.8 of its 7.2 GB budget in use)
#define XM_CHUNK 256ull
static VkBuffer xm_buf;
static VmaAllocation xm_mem;
static uint64_t xm_lo;  // guest address of the mirror's first byte (0: not placed yet)
// (xm_place moves, vk_exec_reset) the per-object motion vectors' copy of last frame's arena (objmv_*) is kept at the
// mirror's offsets: only used when no move came after it
static uint32_t xm_moves;
// the frame each chunk was last written by transform feedback (+1; 0: never): only this frame's are read from the
// mirror - the game reuses the arena's memory (other data written by the CPU when areas / outfits change: read from
// the mirror, a stale skinning drew the player black)
static uint32_t xm_written[XM_SIZE / XM_CHUNK];
// ... and the bytes of the chunk written that frame (one interval [lo, hi); two apart: the larger kept, the other then
// read from the heap). A chunk counted as written whole when a session wrote any of it: a stream sharing the chunk
// with another character's output - a few hundred bytes each in the arena - was read from the mirror where this frame
// wrote nothing of it, the bytes of an earlier session, while the heap held what the game wrote there since: black
// triangles hanging off an NPC and following it (a check of each mirror read against the bytes written found ~1 read
// in 100 in a town).
static uint16_t xm_wlo[XM_SIZE / XM_CHUNK], xm_whi[XM_SIZE / XM_CHUNK];
#define XM_BACK_MAX 4096
static VkBufferCopy xm_back[XM_BACK_MAX];  // the written ranges not copied back yet (mirror -> heap)
static int xm_back_n;
static uint64_t xm_back_lo, xm_back_hi;  // (their bounds, mirror offsets: xm_back_reads)
static uint64_t xm_back_map[XM_SIZE / XM_CHUNK / 64];  // (their 256-byte chunks: xm_back_reads)
static bool xm_draw;  // this draw's transform feedback goes to the mirror

static bool xm_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = ENV("LARECOMP_VK_XFB_MIRROR");
        int cs = g_settings.gpu_compat[GPU_XFB_MIRROR];  // (F1 > Compatibility)
        on = e ? atoi(e) != 0 : cs != GPU_AUTO ? cs == GPU_ON : g_vk.props.vendorID != 0x1002;
        if (!on && !e) LOG("[vk] transform feedback mirror off (%s; LARECOMP_VK_XFB_MIRROR=1: on)\n",
                           cs == GPU_OFF ? "F1 > Compatibility" : "this AMD GPU, F1 > Compatibility to turn it on");
    }
    return on && g_vk.host_import;
}

static bool xm_inside(uint64_t a, uint64_t n) { return xm_lo && a >= xm_lo && a + n <= xm_lo + XM_SIZE; }

static bool xm_all_written(uint64_t a, uint64_t n) {
    if (!n || !xm_inside(a, n)) return false;
    uint32_t now = (uint32_t)vk_frame_counter() + 1;
    uint64_t s0 = a - xm_lo, s1 = s0 + n;
    for (uint64_t c = s0 / XM_CHUNK; c <= (s1 - 1) / XM_CHUNK; c++) {
        uint64_t clo = c * XM_CHUNK, l = s0 > clo ? s0 - clo : 0, h = s1 < clo + XM_CHUNK ? s1 - clo : XM_CHUNK;
        if (xm_written[c] != now || l < xm_wlo[c] || h > xm_whi[c]) return false;
    }
    return true;
}

// Does the program write transform feedback buffer b? (its control section's output count for b, as draw_ reads it
// into vkey.xfb_count). A binding stays set from earlier sessions when a later program writes fewer buffers -
// Pokémon Scarlet / Violet leaves buffer 3 bound for its skinning programs, which write 0..2, at an address 100 MB
// from the arena. Counted, that binding sent each of those sessions to the heap (it lies outside the mirror's window)
// while the other sessions over the same arena went to the mirror, and NPCs' faces seen up close were drawn from stale
// skinning: they covered their own eyebrows and eyelashes (after walking up to them twice; LARECOMP_VK_XFB_MIRROR=0
// hid it).
static bool xm_buffer_used(int b) {
    const uint8_t *ctl = st.program ? st.program->stages[0].control : NULL;
    return !ctl || *(const uint32_t *)(ctl + 0x4DC + 0x80 * b) != 0;
}

// The mirror's place. It was put once, around the first session's buffers - but where the game's skinning arena lies
// varies from one start to the next, and the first session can come from somewhere else (one during the load into the
// field, 70-130 MB past the town's arena): then ~90 % of a town's sessions fell outside the window and wrote the heap's
// import across PCIe as before the mirror - 10-19 ms more GPU time a frame (Mesagoza at World detail 4x: 25-29 fps
// against 30-33, one start in two). And on such a start the arena itself comes in two parts ~140 MB apart, more than
// the 64 MB window was. Now the window is 256 MB, each frame's sessions are counted by the 8 MB piece of the address
// space their buffers are in, and when 16 or more of a frame's fell outside, the window moves to where it covers
// the most of them, if that's 16 more (between frames, with nothing waiting to be copied back: only chunks written in the frame being
// drawn are ever read from the mirror, so what the old place held is simply never read again). LARECOMP_VK_XM_MOVE=0:
// placed once, as before; LARECOMP_LOG_XM=1: the counts every 300 frames.
#define XM_PIECES 32
#define XM_PIECE_SHIFT 23
static struct { uint64_t piece, n; } xm_tab[XM_PIECES];
static uint64_t xm_in_n, xm_out_n, xm_frame, xm_moved_at;
static void xm_place(void) {
    static int on = -1;
    if (on < 0) on = !ENV("LARECOMP_VK_XM_MOVE") || atoi(ENV("LARECOMP_VK_XM_MOVE")) != 0;
    if (!on || !xm_on()) return;
    uint64_t f = vk_frame_counter();
    if (f != xm_frame) {  // (the frame before)
        static int dlog = -1;
        if (dlog < 0) dlog = ENV("LARECOMP_LOG_XM") != NULL;
        if (dlog && f % 300 == 0) {
            LOG("[xm] frame %llu: %llu sessions in the window at %llx, %llu outside; by 8 MB piece:", (unsigned long long)f,
                (unsigned long long)xm_in_n, (unsigned long long)xm_lo, (unsigned long long)xm_out_n);
            for (int i = 0; i < XM_PIECES && xm_tab[i].n; i++)
                LOG(" %llx %llu", (unsigned long long)(xm_tab[i].piece << XM_PIECE_SHIFT), (unsigned long long)xm_tab[i].n);
            LOG("\n");
        }
        if (xm_lo && xm_out_n >= 16 && !xm_back_n && f >= xm_moved_at + 30) {
            // the window [start - 8 MB, + XM_SIZE) over the most sessions, start one of the pieces seen
            const uint64_t span = (XM_SIZE >> XM_PIECE_SHIFT) - 1;  // (pieces after the 8 MB below)
            uint64_t best = 0, best_n = 0;
            for (int i = 0; i < XM_PIECES && xm_tab[i].n; i++) {
                uint64_t n = 0;
                for (int j = 0; j < XM_PIECES && xm_tab[j].n; j++)
                    if (xm_tab[j].piece >= xm_tab[i].piece && xm_tab[j].piece < xm_tab[i].piece + span) n += xm_tab[j].n;
                if (n > best_n) best_n = n, best = xm_tab[i].piece;
            }
            if (best_n >= xm_in_n + 16) {  // (16 sessions more in it, or it stays: no moving back and forth)
                uint64_t lo = (best << XM_PIECE_SHIFT) - (1ull << XM_PIECE_SHIFT);
                if (lo < gwin_base) lo = gwin_base;
                static int nlog;
                if (nlog++ < 8)
                    LOG("[vk] transform feedback mirror moved from %llx to %llx (%llu of the last frame's %llu sessions were "
                        "outside it, %llu now in)\n", (unsigned long long)xm_lo, (unsigned long long)lo,
                        (unsigned long long)xm_out_n, (unsigned long long)(xm_out_n + xm_in_n), (unsigned long long)best_n);
                xm_lo = lo;
                memset(xm_written, 0, sizeof xm_written);
                xm_moved_at = f, xm_moves++;
            }
        }
        xm_in_n = xm_out_n = 0;
        memset(xm_tab, 0, sizeof xm_tab);
        xm_frame = f;
    }
    if (!xm_lo) return;  // (the first session places it: xm_prepare_)
    uint64_t at = 0;
    bool out = false;
    for (int b = 0; b < 4; b++) {
        if (!xfb_addr[b] || !xfb_size[b] || !xm_buffer_used(b)) continue;
        if (!at) at = xfb_addr[b];
        if (!xm_inside(xfb_addr[b], xfb_size[b])) { out = true; at = xfb_addr[b]; break; }
    }
    if (!at) return;
    if (out) xm_out_n++; else xm_in_n++;
    uint64_t piece = at >> XM_PIECE_SHIFT;
    int i = 0;
    while (i < XM_PIECES && xm_tab[i].n && xm_tab[i].piece != piece) i++;
    if (i < XM_PIECES) { xm_tab[i].piece = piece; xm_tab[i].n++; }
}

// before a transform feedback draw: can its buffers all go to the mirror? (then they're marked written, and queued
// for the copy back)
static bool xm_prepare_(void);
static bool xm_prepare(void) {
    xm_place();
    if (xm_prepare_()) return true;
    // To the heap: the parts of the mirror its buffers cover no longer hold this frame's output (a later read of them
    // takes the heap, where this session writes)
    if (!xm_lo) return false;
    for (int b = 0; b < 4; b++) {
        if (!xfb_addr[b] || !xfb_size[b] || !xm_buffer_used(b)) continue;
        uint64_t lo = xfb_addr[b] > xm_lo ? xfb_addr[b] : xm_lo, hi = xfb_addr[b] + xfb_size[b];
        if (hi > xm_lo + XM_SIZE) hi = xm_lo + XM_SIZE;
        for (uint64_t c = (lo - xm_lo) / XM_CHUNK; hi > lo && c <= (hi - 1 - xm_lo) / XM_CHUNK; c++) xm_written[c] = 0;
    }
    return false;
}
static bool xm_prepare_(void) {
    if (!xm_on() || !gmem_buf) return false;
    if (!xm_buf) {
        if (g_vk.props.limits.maxStorageBufferRange < XM_SIZE) {  // (bound whole as a storage buffer: XFB_BUFFER_INDEX)
            static int logged;
            if (!logged++) LOG("[vk] transform feedback mirror off: storage buffers reach %u bytes\n", g_vk.props.limits.maxStorageBufferRange);
            return false;
        }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = XM_SIZE,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &xm_buf, &xm_mem, NULL) != VK_SUCCESS) { xm_buf = VK_NULL_HANDLE; return false; }
    }
    for (int b = 0; b < 4; b++) {
        if (!xfb_addr[b] || !xfb_size[b] || !xm_buffer_used(b)) continue;  // (a buffer the program doesn't write: left alone)
        if (!xm_lo) {  // placed around the first session's buffers, half the window each side (then xm_place)
            uint64_t lo = (xfb_addr[b] & ~0xFFFFFull) - XM_SIZE / 2;
            xm_lo = lo < gwin_base ? gwin_base : lo;
            LOG("[vk] transform feedback mirror at %llx (+%llu MB)\n", (unsigned long long)xm_lo, (unsigned long long)(XM_SIZE >> 20));
        }
        if ((xfb_addr[b] & 3) || !xm_inside(xfb_addr[b], xfb_size[b])) return false;
        if (!GM_IN(xfb_addr[b], xfb_size[b])) {  // (also with the whole heap imported: the mirror may reach past its end)
            static int nl;
            if (nl++ < 8) LOG("[gmem] transform feedback buffer %llx is outside the import window\n", (unsigned long long)xfb_addr[b]);
            return false;
        }
    }
    for (int b = 0; b < 4; b++) {
        if (!xfb_addr[b] || !xfb_size[b] || !xm_buffer_used(b)) continue;
        uint64_t o = xfb_addr[b] - xm_lo;
        uint32_t now = (uint32_t)vk_frame_counter() + 1;
        for (uint64_t c = o / XM_CHUNK; c <= (o + xfb_size[b] - 1) / XM_CHUNK; c++) {
            uint64_t clo = c * XM_CHUNK, l = o > clo ? o - clo : 0, h = o + xfb_size[b] < clo + XM_CHUNK ? o + xfb_size[b] - clo : XM_CHUNK;
            if (xm_written[c] != now || (h - l > (uint64_t)(xm_whi[c] - xm_wlo[c]) && (l > xm_whi[c] || h < xm_wlo[c]))) {
                xm_wlo[c] = (uint16_t)l, xm_whi[c] = (uint16_t)h;  // (the first this frame, or a larger one apart)
            } else if (l <= xm_whi[c] && h >= xm_wlo[c]) {  // (touching or overlapping: one interval)
                if (l < xm_wlo[c]) xm_wlo[c] = (uint16_t)l;
                if (h > xm_whi[c]) xm_whi[c] = (uint16_t)h;
            }
            xm_written[c] = now;
        }
        int i = xm_back_n > 8 ? xm_back_n - 8 : 0;  // (the same range again: a session's draws repeat its buffers)
        while (i < xm_back_n && !(xm_back[i].srcOffset == o && xm_back[i].size == xfb_size[b])) i++;
        if (i == xm_back_n) {
            // (full: copied back now, in a pass of its own. Never one range over them all, as before: that carried
            // the mirror's bytes between the ranges - written by no session this frame - over the heap)
            if (xm_back_n == XM_BACK_MAX) { vk_end_pass(); xm_copy_back(vk_cmd()); }
            if (!xm_back_n || o < xm_back_lo) xm_back_lo = o;
            if (!xm_back_n || o + xfb_size[b] > xm_back_hi) xm_back_hi = o + xfb_size[b];
            xm_back[xm_back_n++] = (VkBufferCopy){o, xm_lo - gwin_base - gmem_lo + o, xfb_size[b]};
            for (uint64_t c = o / XM_CHUNK; c <= (o + xfb_size[b] - 1) / XM_CHUNK; c++) xm_back_map[c / 64] |= 1ull << (c % 64);
        }
    }
    return true;
}

// a vertex stream inside the mirror's transform feedback output: read there
static bool xm_lookup(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off) {
    if (!xm_buf || !xm_all_written(a, n)) return false;
    *buf = xm_buf;
    *off = a - xm_lo;
    return true;
}

// the submission's end (outside any pass): what transform feedback wrote, to the heap
static void xm_copy_back(VkCommandBuffer cb) {
    if (xm_back_n) {
        uint64_t w0 = xm_back_lo / XM_CHUNK / 64, w1 = (xm_back_hi - 1) / XM_CHUNK / 64;
        memset(&xm_back_map[w0], 0, (w1 - w0 + 1) * sizeof xm_back_map[0]);
    }
    if (!xm_back_n || !xm_buf || !gmem_buf) { xm_back_n = 0; return; }
    vk_full_barrier(cb);
    rec_CmdCopyBuffer(cb, xm_buf, gmem_buf, (uint32_t)xm_back_n, xm_back);
    vk_full_barrier(cb);
    xm_back_n = 0;
}

// The copy back after a run of sessions, not after each one. Each session's end used to copy its output back right away
// (vertex and storage reads of it outside the mirror: the characters' shading), which ended its render pass and put two
// full barriers around the copy - Scarlet / Violet skins one character a session, ~500 a frame in a town at People
// detail 4x: 500 passes, 6.7 ms of GPU time and ~3 ms of the render thread a frame. The sessions follow one another with
// only state changes between them, so the copy now waits until anything else comes (exec_list: a draw outside a
// session, a dispatch, a copy, a clear...), a session's draw reads a pending range (xm_back_reads), the list is full,
// or the submission ends. LARECOMP_VK_XFB_BATCH=0: after each session, as before.
static bool xm_batch(void) {
    static int on = -1;
    if (on < 0) on = !ENV("LARECOMP_VK_XFB_BATCH") || atoi(ENV("LARECOMP_VK_XFB_BATCH")) != 0;
    return on;
}
static void xm_flush_back(void) {
    if (!xm_back_n) return;
    vk_end_pass();
    xm_copy_back(vk_cmd());
}
// Ops that only set state (no GPU work, nothing read): the pending copy back can wait past them
static bool xm_state_op(uint32_t op) {
    switch (op) {
    case CMD_END: case CMD_JUMP: case CMD_CALL:
    case CMD_BIND_PROGRAM: case CMD_BIND_BLEND: case CMD_BIND_CHANNEL_MASK: case CMD_BIND_COLOR: case CMD_BIND_DEPTH_STENCIL:
    case CMD_BIND_MULTISAMPLE: case CMD_BIND_POLYGON: case CMD_BIND_VERTEX_ATTRIB: case CMD_BIND_VERTEX_STREAM:
    case CMD_BIND_VERTEX_BUFFERS: case CMD_BIND_UNIFORM_BUFFERS: case CMD_BIND_STORAGE_BUFFERS: case CMD_BIND_TEXTURES:
    case CMD_BIND_IMAGES: case CMD_SET_TEXTURE_POOL: case CMD_SET_SAMPLER_POOL:
    case CMD_SET_VIEWPORTS: case CMD_SET_SCISSORS: case CMD_SET_DEPTH_RANGES: case CMD_SET_BLEND_COLOR: case CMD_SET_STENCIL:
    case CMD_SET_PRIMITIVE_RESTART: case CMD_SET_ALPHA_REF: case CMD_SET_DEPTH_BOUNDS: case CMD_SET_POLYGON_OFFSET:
    case CMD_SET_LINE_WIDTH: case CMD_SET_POINT_SIZE: case CMD_SET_SAMPLE_MASK: case CMD_SET_COPY_STRIDES:
    case CMD_BEGIN_XFB: case CMD_END_XFB: case CMD_BIND_XFB_BUFFER: case CMD_BIND_XFB_BUFFERS:
    case CMD_SET_RASTERIZER_DISCARD: case CMD_SET_DEPTH_CLAMP:
    case CMD_BARRIER:  // (a no-op here: every list starts with one; what reads after it copies back first)
        return true;
    default:
        return false;
    }
}
// A session's draw about to read [a, a + n) (vertex streams, storage, uniforms): inside the pending ranges' bounds?
static bool xm_back_reads(uint64_t a, uint64_t n) {
    if (!xm_back_n || !a || !n || !xm_lo) return false;
    uint64_t lo = xm_lo + xm_back_lo, hi = xm_lo + xm_back_hi;
    if (a >= hi || a + n <= lo) return false;
    // (inside the bounds: the ranges' 256-byte chunks - the arena's sessions are scattered, and a skinning draw's own
    // inputs often lie between pending outputs)
    uint64_t s0 = (a > lo ? a : lo) - xm_lo, s1 = (a + n < hi ? a + n : hi) - xm_lo;
    for (uint64_t c = s0 / XM_CHUNK; c <= (s1 - 1) / XM_CHUNK; c++) {
        if (!(c % 64) && c + 64 <= (s1 - 1) / XM_CHUNK && !xm_back_map[c / 64]) { c += 63; continue; }
        if (xm_back_map[c / 64] >> (c % 64) & 1) {  // (a chunk one of them is in: the ranges themselves)
            for (int i = 0; i < xm_back_n; i++)
                if (a < xm_lo + xm_back[i].srcOffset + xm_back[i].size && a + n > xm_lo + xm_back[i].srcOffset) return true;
            return false;
        }
    }
    return false;
}

// ---- gfast switches (tools/test_gfast_switches.py compiles these two from here)
// The bound storage buffers' device-local copies: LARECOMP_VK_GMEM_FAST=1 / =0; unset: on, except on an integrated GPU
// (there the heap import is not across PCIe: the copy and the table search only cost, 234 -> 97 ms on a Steam Deck).
static bool gfast_storage_on(const char *fast, bool integrated) {
    return fast ? atoi(fast) != 0 : !integrated;
}
// Uniform buffers outside the imported window staged into gfast_buf (issue #27): LARECOMP_VK_UBO_STAGE=1 / =0 decides;
// unset, LARECOMP_VK_GMEM_FAST does when it is set (=0 turns the whole fast path off, this included); neither set: on,
// on every GPU (integrated too: there only dispatches with such a uniform buffer pay for it). Values as atoi reads them.
static bool gfast_ubo_stage_on(const char *fast, const char *stage) {
    return stage ? atoi(stage) != 0 : fast ? atoi(fast) != 0 : true;
}
// ---- end of the gfast switches
static void gfast_prepare(VkCommandBuffer cb) {
    static int on = -1, ubo_on = -1;
    if (on < 0) on = gfast_storage_on(ENV("LARECOMP_VK_GMEM_FAST"), g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    if (ubo_on < 0) ubo_on = gfast_ubo_stage_on(ENV("LARECOMP_VK_GMEM_FAST"), ENV("LARECOMP_VK_UBO_STAGE"));
    memset(gmap_tab, 0, sizeof gmap_tab);
    gfast_n = 0;
    gfast_back_n = 0;
    if (!gc_lazy) gmem_info();  // (the heap import, made on first use)
    gfast_info();
    if (!gfast_buf) return;
    bool do_in = on && (gmem_buf || gc_lazy) && !gmem_lo;  // (gmem_lo: a windowed import - its offsets are not the heap's)
    struct { uint64_t lo, hi; } r[16];
    int n = 0;
    if (do_in) {
        for (int i = 0; i < 16; i++) {  // window offsets of the bound storage buffers, 256-aligned
            uint64_t a = sbufs(5)[i].addr, sz = sbufs(5)[i].size;  // (overlap: a snapshotted range's copy is what gets copied)
            if (!a || !sz || a < gwin_base || !gmem_covers(a - gwin_base, sz)) continue;
            uint64_t lo = (a - gwin_base) & ~255ull, hi = (a - gwin_base + sz + 255) & ~255ull, top = gc_lazy ? gmem_heap_size() : gmem_size;
            if (hi > top) hi = top;
            int k = n++;
            while (k > 0 && r[k - 1].lo > lo) { r[k] = r[k - 1]; k--; }  // (insertion sort by start)
            r[k].lo = lo; r[k].hi = hi;
        }
    }
    int m = 0;
    for (int i = 0; i < n; i++) {  // merge overlapping / nearby ranges (a small gap costs less than another table entry)
        if (m && r[i].lo <= r[m - 1].hi + 4096) { if (r[i].hi > r[m - 1].hi) r[m - 1].hi = r[i].hi; }
        else r[m++] = r[i];
    }
    VkBufferCopy cp[16];
    uint64_t off = 0;
    for (int i = 0; i < m; i++) {
        uint64_t len = r[i].hi - r[i].lo;
        if (m > GMAP_MAX || off + len > GFAST_SIZE) return;  // (too much: the heap as before)
        cp[i] = (VkBufferCopy){r[i].lo, off, len};
        uint64_t ga = gwin_base + r[i].lo;
        gmap_tab[i * 4 + 0] = (uint32_t)ga;
        gmap_tab[i * 4 + 1] = (uint32_t)len;
        gmap_tab[i * 4 + 2] = (uint32_t)(off >> 2);
        gmap_tab[i * 4 + 3] = (uint32_t)(ga >> 32);
        off += (len + 255) & ~255ull;
    }
    // Stage uniform buffers that live outside the imported gmem window (e.g. Scarlet & Violet's secondary VAMM pools
    // at 0x744... when gwin only covers 0x7676b00000+, or below gmem_lo on windowed iGPUs): compute shaders read
    // their cb0[33+i] pointers via LDG/GMR, which would otherwise fail the gmem bounds check and read zero.
    // LARECOMP_VK_UBO_STAGE=0: off (gfast_ubo_stage_on).
    uint64_t g_lo = (gc_lazy || gmem_buf) ? gwin_base + (gc_lazy ? 0 : gmem_lo) : 0;
    uint64_t g_hi = (gc_lazy || gmem_buf) ? gwin_base + (gc_lazy ? gwin_max : gmem_lo + gmem_size) : 0;
    int total = m;
    for (int i = 0; ubo_on && i < 16 && total < GMAP_MAX; i++) {
        uint64_t a = st.ubuf[5][i].addr, sz = st.ubuf[5][i].size;
        if (!a || !sz || sz > (1ull << 20)) continue;
        if (g_hi && a >= g_lo && a + sz <= g_hi && a + sz >= a) continue;
        uint64_t lo = a & ~3ull, hi = (a + sz + 3ull) & ~3ull, len = hi - lo;
        if (off + len > GFAST_SIZE) break;
        VkSpan sp = vk_transient_copy((const void *)(uintptr_t)lo, (size_t)len, 256);
        VkBufferCopy hc = {sp.off, off, len};
        rec_CmdCopyBuffer(cb, sp.buf, gfast_buf, 1, &hc);
        gmap_tab[total * 4 + 0] = (uint32_t)lo;
        gmap_tab[total * 4 + 1] = (uint32_t)len;
        gmap_tab[total * 4 + 2] = (uint32_t)(off >> 2);
        gmap_tab[total * 4 + 3] = (uint32_t)(lo >> 32);
        off += (len + 255) & ~255ull;
        total++;
    }
    if (!total) return;
    if (m && ENV("LARECOMP_LOG_GFAST")) {  // (debug) =<shader code address hex>: that shader's first tables
        static int nlog;
        if (st.program->stages[5].code_addr == strtoull(ENV("LARECOMP_LOG_GFAST"), NULL, 16) && nlog++ < 40) {
            LOG("[gfast] cs %llx: %d ranges, %llu KB:", (unsigned long long)st.program->stages[5].code_addr, m, (unsigned long long)(off >> 10));
            for (int i = 0; i < m; i++) LOG(" %llx+%llx", (unsigned long long)(r[i].lo + gwin_base), (unsigned long long)(r[i].hi - r[i].lo));
            LOG("\n");
        }
    }
    if (m) gmem_copy(cb, gfast_buf, cp, m, false);
    vk_full_barrier(cb);
    if (m && gfast_stores && gmerge_init()) {  // (the twin: see gfast_back)
        VkBufferCopy tw[16];
        for (int i = 0; i < m; i++) tw[i] = (VkBufferCopy){cp[i].dstOffset, cp[i].dstOffset, cp[i].size};
        rec_CmdCopyBuffer(cb, gfast_buf, gorig_buf, (uint32_t)m, tw);
        vk_full_barrier(cb);
    }
    gmap_tab[GMAP_MAX * 4] = (uint32_t)total;
    if (m) memcpy(gfast_cp, cp, (size_t)m * sizeof cp[0]);
    gfast_n = total;
    for (int i = 0; m && i < 16; i++) {  // each bound buffer's exact bytes, from wherever its merged range sits
        uint64_t a = st.sbuf[5][i].addr, sz = st.sbuf[5][i].size;
        if (!a || !sz || a < gwin_base || a - gwin_base + sz > gmem_size) continue;
        uint64_t h = a - gwin_base;
        for (int j = 0; j < m; j++)
            if (h >= cp[j].srcOffset && h + sz <= cp[j].srcOffset + cp[j].size) {
                gfast_back[gfast_back_n++] = (VkBufferCopy){cp[j].dstOffset + (h - cp[j].srcOffset), h, sz};
                break;
            }
    }
}

// After a storing dispatch: its ranges back to the heap (before anything else reads them: draws, the CPU after the fence).
static void gfast_finish(VkCommandBuffer cb) {
    if (!gfast_n) return;
    if (!gfast_back_n) { gfast_n = 0; return; }
    for (int i = 0; i < gfast_back_n; i++)
        gw_note("compute copy back", gwin_base + gfast_back[i].dstOffset, gfast_back[i].size, st.program ? st.program->stages[5].code_addr : 0);
    vk_full_barrier(cb);
    bool merged = false;
    if (!gc_lazy && gfast_back_n && gmerge_init()) {  // (chunks: the plain copy, gmem_copy)
        merged = true;
        for (int i = 0; i < gfast_back_n; i++)
            if ((gfast_back[i].srcOffset | gfast_back[i].dstOffset | gfast_back[i].size) & 3 || gfast_back[i].dstOffset / 4 > 0xFFFFFFFFull)
                merged = false;  // (not word-aligned: the plain copy)
    }
    if (merged) {
        VkDescriptorSet set = vk_alloc_set(gmerge_set_layout);
        VkDescriptorBufferInfo bi[3] = {gmem_info(), {gfast_buf, 0, GFAST_SIZE}, {gorig_buf, 0, GFAST_SIZE}};
        VkWriteDescriptorSet w[3];  // (one descriptor a write: record.c)
        for (uint32_t k = 0; k < 3; k++)
            w[k] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = k,
                                          .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[k]};
        rec_UpdateDescriptorSets(3, w);
        rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gmerge_pipe);
        rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, gmerge_layout, 0, 1, &set, 0, NULL);
        for (int i = 0; i < gfast_back_n; i++) {
            static int cpu_wins = -1;  // (gfast_merge.comp: a word both rewrote keeps the CPU's; LARECOMP_VK_GFAST_CPU_WINS=0: the dispatch's)
            if (cpu_wins < 0) cpu_wins = !ENV("LARECOMP_VK_GFAST_CPU_WINS") || atoi(ENV("LARECOMP_VK_GFAST_CPU_WINS")) != 0;
            uint32_t pc[4] = {(uint32_t)(gfast_back[i].srcOffset / 4), (uint32_t)(gfast_back[i].dstOffset / 4), (uint32_t)(gfast_back[i].size / 4),
                              (uint32_t)cpu_wins};
            rec_CmdPushConstants(cb, gmerge_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            rec_CmdDispatch(cb, (pc[2] + 255) / 256, 1, 1);
        }
    } else if (gfast_back_n) {
        // (overlapping bound buffers copy the same bytes twice - identical data, harmless)
        gmem_copy(cb, gfast_buf, gfast_back, gfast_back_n, true);
    }
    vk_full_barrier(cb);
    gfast_n = 0;
}

// ---- vertex / index cache
// Reading vertex / index data in place from the imported heap crosses PCIe for every vertex fetched on a discrete GPU
// (outdoors at 3x internal resolution on an RTX 3080 Ti: ~0.7-1.9 ms of GPU a frame; more on PCIe 3.0), and copying it
// per draw instead moves about as much onto the CPU. So each range bound for drawing gets a device-local copy, refreshed
// only where the CPU wrote since (the OS's page write tracking, plat_written_pages, asked at submission: vc_flush): one
// sequential GPU copy of the written pages in the submission's upload buffer (which completes before its draws) instead
// of fetches across the bus. Overlapping ranges are separate entries; the per-page generation keeps them all current.
// Pages the GPU writes itself (a compute dispatch's copy-back, gfast_finish) are never cached: draws over them read the
// heap in place as before, and so does everything when the OS gives no write tracking. LARECOMP_VK_VCACHE=0: off.
#define VC_PAGE 4096ull
#define VC_TAB 8192                   // entries (open addressing)
#define VC_MAX_RANGE (64ull << 20)    // larger bound ranges: read in place
#define VC_BUDGET (768ull << 20)      // device-local bytes kept; the least recently used go first
// sq: odd while the lock's holder changes the entry (vc_quick reads it without the lock); used_seq is claimed for the
// submission by compare-and-swap (~0: being dropped); vc_npend grows atomically
typedef struct { uint64_t lo, hi, used_seq; uint32_t gen, epoch, gpu_until; VkBuffer buf; VmaAllocation mem; bool gpu; _Atomic uint32_t sq; } VcEnt;
static VcEnt vc_tab[VC_TAB];
static uint32_t vc_pend[VC_TAB];  // entries the pending submission draws from (vc_flush brings them up to date)
static _Atomic int vc_npend;
static inline void vc_wbegin(VcEnt *e) { atomic_fetch_add_explicit(&e->sq, 1, memory_order_acq_rel); }
static inline void vc_wend(VcEnt *e) { atomic_fetch_add_explicit(&e->sq, 1, memory_order_release); }
static inline void vc_claim(VcEnt *e, uint64_t pend) {  // (first use in the submission: vc_flush brings it up to date)
    uint64_t u = __atomic_load_n(&e->used_seq, __ATOMIC_ACQUIRE);
    if (u != pend && u != ~0ull && __atomic_compare_exchange_n(&e->used_seq, &u, pend, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        vc_pend[atomic_fetch_add_explicit(&vc_npend, 1, memory_order_relaxed)] = (uint32_t)(e - vc_tab);
}
static uint32_t *vc_page_gen;  // per heap page: the generation of its last CPU write seen (vc_gen when observed)
static uint32_t *vc_page_gpu;  // per heap page: the frame the GPU last wrote it (+1; 0: never)
// Writes the OS loses. A write tracking query resets the pages' marks, and a store landing on a page while its mark is
// reset (the game's thread rewriting its per-frame data with memcpy as the render thread asks) can go unrecorded: the
// vertex cache kept its copy of the page from before, a frame's dynamic vertices stayed as they were, and Scarlet /
// Violet's sandstorm, fog and the ground near them drew as huge stretched triangles - at random, as the race. A check
// of every clean page's bytes against its copy (a scripted walk in a city) found such pages within seconds, nearly
// all written 1-3 frames before. So a page seen written is unsettled for the next VC_RECENT frames, a lost store may
// still be landing: whoever looks at it in that time (the vertex cache's copy, a texture upload check, the storage
// watch) looks again after it settles - at the first look after the window, its generation then (vc_page_settle) is
// what a look has to be newer than. (Counted from the look, not the clock: a range not drawn in those frames - an
// object out of view for a moment - was copied before the window and served stale after it, a flickering triangle at
// the academy's gate.)
static uint32_t *vc_page_wfr;     // per heap page: the frame its last write was seen (+1; 0: never)
static uint32_t *vc_page_settle;  // per heap page: the generation of the first look after its window (0: not yet)
#define VC_RECENT 8u
extern uint64_t vk_frame_counter(void);
static uint32_t vc_gen = 1, vc_epoch = 1;  // (epoch: bumped when pages turn GPU-written: entries check again)
static uint32_t vc_gen_now(void) { return vc_gen; }
static inline void vc_seen(uint64_t p, uint32_t g) {
    vc_page_gen[p] = g;
    vc_page_wfr[p] = (uint32_t)vk_frame_counter() + 1;
    vc_page_settle[p] = 0;
}
// written since generation `since`, or maybe (a write seen, and `since` from before the page settled)?
static inline bool vc_dirty(uint64_t p, uint32_t since) {
    if (vc_page_gen[p] > since) return true;
    if (!vc_page_wfr[p]) return false;
    if ((uint32_t)vk_frame_counter() + 1 - vc_page_wfr[p] < VC_RECENT) return true;  // (still unsettled)
    if (!vc_page_settle[p]) vc_page_settle[p] = vc_gen;
    return since < vc_page_settle[p];
}
// A page the GPU hasn't written for this many frames is cached again (copied whole first): the game reuses its
// memory, and pages a compute dispatch or transform feedback wrote once stayed read in place for the rest of the
// session - the vertices read across PCIe grew to ~10 GB per 2 s after half an hour (GPU time +25%, 60 fps lost)
#define VC_GPU_HOLD 120u
static bool vc_gpu_expired;  // (an entry turned cacheable again: vc_flush orders its copy after the GPU's writes)
static uint64_t vc_bytes;
#define VC_DEAD 1024
static struct { VkBuffer b; VmaAllocation m; uint64_t seq; } vc_dead[VC_DEAD];
static int vc_ndead;
static int vc_on = -1;
// (profiler: draws served / read in place, bytes refreshed, write tracking queries, written pages seen, query time)

static void vc_free_dead(bool all) {
    uint64_t done = vk_completed_seq();
    int k = 0;
    for (int i = 0; i < vc_ndead; i++) {
        if (!all && vc_dead[i].seq > done) { vc_dead[k++] = vc_dead[i]; continue; }
        vmaDestroyBuffer(g_vk.vma, vc_dead[i].b, vc_dead[i].m);
    }
    vc_ndead = k;
}

static bool vc_drop(VcEnt *e) {  // (freed once the submissions that may read it are done)
    if (vc_ndead == VC_DEAD) { vk_wait_seq(vk_pending_seq() - 1); vc_free_dead(false); }
    if (vc_ndead == VC_DEAD) return false;  // (all read by the pending submission: keep it)
    uint64_t u = __atomic_load_n(&e->used_seq, __ATOMIC_ACQUIRE);  // (not if vc_quick claimed it for the submission meanwhile)
    if (u >= vk_pending_seq() || !__atomic_compare_exchange_n(&e->used_seq, &u, ~0ull, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        return false;
    vc_wbegin(e);
    vc_dead[vc_ndead].b = e->buf; vc_dead[vc_ndead].m = e->mem; vc_dead[vc_ndead++].seq = u + VK_DEAD_MARGIN;
    vc_bytes -= e->hi - e->lo;
    memset(e, 0, offsetof(VcEnt, sq));
    vc_wend(e);
    return true;
}

// The page tables (vc_page_gen / vc_page_gpu) exist whenever the OS gives write tracking: the vertex cache and the
// texture upload checks (vk_pages_changed) share them. Each write tracking query resets the OS bits, so every query
// stamps the pages it saw written with a new generation, and each user compares against the generation it last saw.
static int pt_state = -1;  // 1: tracking, 0: none
static bool pt_init(void) {
    if (pt_state >= 0) return pt_state;
    uint64_t pg = VC_PAGE, one;
    int n = plat_written_pages(gwin_base, VC_PAGE, &one, 1, &pg);  // (is there write tracking?)
    pt_state = n >= 0 && pg == VC_PAGE;
    if (!pt_state) { LOG("[vk] no page write tracking (%d, page %llu)\n", n, (unsigned long long)pg); return false; }
    vc_page_gen = calloc(HEAP_MAX / VC_PAGE, sizeof *vc_page_gen);
    vc_page_gpu = calloc(HEAP_MAX / VC_PAGE, sizeof *vc_page_gpu);
    vc_page_wfr = calloc(HEAP_MAX / VC_PAGE, sizeof *vc_page_wfr);
    vc_page_settle = calloc(HEAP_MAX / VC_PAGE, sizeof *vc_page_settle);
    vc_page_gen[0] = ++vc_gen;  // (the probe took page 0's tracking)
    return true;
}

// Pages outside the window: Pokémon Z-A's textures sit in the alias region (0x74..., its SDK's address regions), not in
// the physical memory window - every check of theirs was "unknown" and sampled 4096 scattered bytes (~5500 a second
// at 50 fps, 12 % of the render thread). The OS tracks writes over the whole guest space, so those pages get
// generations too: a sparse table over [GUEST_LO, GUEST_HI), 2 MB pieces made on first use. (No GPU-written flag:
// shader stores reach the window only, the gmem import.)
#define PG_PIECE 512  // pages a piece
static uint32_t *pg_dir[(GUEST_HI - GUEST_LO) / (PG_PIECE * VC_PAGE)];
static uint32_t *pg_gen(uint64_t page_addr) {
    uint64_t p = (page_addr - GUEST_LO) / VC_PAGE;
    uint32_t **d = &pg_dir[p / PG_PIECE];
    if (!*d) *d = calloc(PG_PIECE, sizeof **d);
    return &(*d)[p % PG_PIECE];
}

static int pages_changed_outside(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (addr < GUEST_LO || addr + size > GUEST_HI) return -1;
    uint64_t lo = addr & ~(VC_PAGE - 1), hi = (addr + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    static uint64_t pages[4096];
    uint64_t pg = VC_PAGE;
    int n = plat_written_pages(lo, hi - lo, pages, 4096, &pg);
    if (n < 0 || pg != VC_PAGE) return -1;
    uint32_t g = ++vc_gen;
    if (n >= 4096) for (uint64_t a = lo; a < hi; a += VC_PAGE) *pg_gen(a) = g;
    else for (int k = 0; k < n; k++) *pg_gen(pages[k]) = g;
    *now = g;
    for (uint64_t a = lo; a < hi; a += VC_PAGE)
        if (*pg_gen(a) > since) return 1;
    return 0;
}

// Texture upload checks (textures.c): may guest memory [addr, addr + size) have changed since generation `since`?
// 0: no CPU write since, 1: written, < 0: unknown - -1 no tracking, -2 pages the GPU writes (gpu_unknown: the upload
// checks; the storage write watch asks about the CPU alone). *now: the generation to pass next time.
static int pages_written(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now, bool gpu_unknown) {
    if (!size || !pt_init()) return -1;
    if (addr < gwin_base || addr - gwin_base + size > gwin_max) {
        static int outside = -1;  // LARECOMP_VK_PAGES_OUTSIDE=0: unknown there, as before (F1 > Compatibility)
        if (outside < 0) outside = gpu_feature_on(GPU_PAGES_OUTSIDE, "LARECOMP_VK_PAGES_OUTSIDE");
        int r = outside ? pages_changed_outside(addr, size, since, now) : -1;
        static int nlog;  // LARECOMP_LOG_PAGES=1 (debug): the first ranges outside the tracked window
        if (r < 0 && ENV("LARECOMP_LOG_PAGES") && nlog++ < 12)
            LOG("[pages] %llx+%llx outside the window %llx+%llx: untracked\n", (unsigned long long)addr,
                (unsigned long long)size, (unsigned long long)gwin_base, (unsigned long long)gwin_max);
        return r;
    }
    uint64_t lo = (addr - gwin_base) & ~(VC_PAGE - 1), hi = (addr - gwin_base + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    static uint64_t pages[4096];
    uint64_t pg = VC_PAGE;
    int n = plat_written_pages(gwin_base + lo, hi - lo, pages, 4096, &pg);
    if (n < 0) return -1;
    uint32_t g = ++vc_gen;
    if (n >= 4096) for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) vc_seen(p, g);
    else for (int k = 0; k < n; k++) vc_seen((pages[k] - gwin_base) / VC_PAGE, g);
    *now = g;
    int changed = 0;
    for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) {
        if (gpu_unknown && vc_page_gpu[p]) return -2;
        if (vc_dirty(p, since)) changed = 1;
    }
    return changed;
}
int vk_pages_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) { return pages_written(addr, size, since, now, true); }
// The storage write watch (watch_check): CPU writes only; the GPU's own writes to the range (DMA) don't count
static int pages_cpu_written(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) { return pages_written(addr, size, since, now, false); }

// Heap pages [lo, hi) (offsets) written by the GPU: never cached from now on.
static void vc_mark_gpu(uint64_t lo, uint64_t hi) {
    if (!pt_init()) return;
    uint32_t now = (uint32_t)vk_frame_counter() + 1;
    // (the entries look again only when a page turns GPU-written - unmarked, or its mark expired: the ~500 transform
    // feedback sessions a frame in a town mark the same buffers every frame, and each bump sent every vertex cache
    // lookup and guest_direct memo the long way, under the shadow workers' lock)
    bool fresh = false;
    for (uint64_t p = lo / VC_PAGE; p < (hi + VC_PAGE - 1) / VC_PAGE && p < gwin_max / VC_PAGE; p++) {
        uint32_t o = vc_page_gpu[p];
        if (!o || now >= o + VC_GPU_HOLD) fresh = true;
        vc_page_gpu[p] = now;
    }
    if (fresh) vc_epoch++;
}

// A storing shader (stage 0 / 1 / 5) may write any of its bound storage buffers.
static void vc_mark_sbufs(int stage) {
    for (int i = 0; i < 16; i++) {
        uint64_t a = sbufs(stage)[i].addr, sz = sbufs(stage)[i].size;  // (overlap: the copy, where the shader wrote)
        if (a >= gwin_base && sz && a - gwin_base < gwin_max) vc_mark_gpu(a - gwin_base, a - gwin_base + sz);
    }
}

static bool vcache_lookup(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (vc_on < 0) {
        const char *e = ENV("LARECOMP_VK_VCACHE");
        vc_on = !e || atoi(e) != 0;
        // integrated GPU with a userptr heap (only with LARECOMP_WRITE_TRACKING=1: tracking is off there otherwise): the
        // heap already is the GPU's memory, no bus to save - the copies would only cost. LARECOMP_VK_VCACHE=1: on.
        if (vc_on && !e && gmem_userptr && g_vk.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            LOG("[vk] vertex cache off: integrated GPU\n");
            vc_on = 0;
        }
        if (vc_on && !pt_init()) {
            LOG("[vk] vertex cache off: no page write tracking\n");
            vc_on = 0;
        }
    }
    if (!vc_on || (!gmem_buf && !gc_lazy) || gmem_lo || size > VC_MAX_RANGE) return false;
    uint64_t lo = (addr - gwin_base) & ~(VC_PAGE - 1), hi = (addr - gwin_base + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    if (!gmem_covers(lo, hi - lo)) return false;
    uint64_t pend = vk_pending_seq();
    uint64_t h = (lo * 0x9E3779B97F4A7C15ull) ^ (hi * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (VC_TAB - 1), free_slot = ~0u, old_slot = ~0u;
    VcEnt *e = NULL;
    for (int probe = 0; probe < 16; probe++, slot = (slot + 1) & (VC_TAB - 1)) {
        VcEnt *c = &vc_tab[slot];
        if (c->buf && c->lo == lo && c->hi == hi) { e = c; break; }
        if (!c->buf && free_slot == ~0u) free_slot = slot;
        if (c->buf && c->used_seq < pend && (old_slot == ~0u || c->used_seq < vc_tab[old_slot].used_seq)) old_slot = slot;
    }
    if (!e) {  // a new range: its own device-local buffer (gen 0: copied whole at the submission)
        if (vc_npend == VC_TAB) return false;
        if (vc_ndead) vc_free_dead(false);
        if (free_slot == ~0u) {
            if (old_slot == ~0u || !vc_drop(&vc_tab[old_slot])) return false;
            free_slot = old_slot;
        }
        while (vc_bytes + (hi - lo) > VC_BUDGET) {  // over budget: the least recently used entry goes
            int lru = -1;
            for (int i = 0; i < VC_TAB; i++)
                if (vc_tab[i].buf && vc_tab[i].used_seq < pend && (lru < 0 || vc_tab[i].used_seq < vc_tab[lru].used_seq)) lru = i;
            if (lru < 0 || !vc_drop(&vc_tab[lru])) return false;
        }
        e = &vc_tab[free_slot];
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = hi - lo,
                                 .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return false;
        vc_wbegin(e);
        e->lo = lo; e->hi = hi; e->mem = nm;
        __atomic_store_n(&e->buf, nb, __ATOMIC_RELEASE);  // (last: vc_quick matches a range only with its buffer)
        vc_wend(e);
        vc_bytes += hi - lo;
    }
    uint32_t now = (uint32_t)vk_frame_counter() + 1;
    if (e->epoch != vc_epoch || (e->gpu && now >= e->gpu_until)) {  // (pages turned GPU-written since, or their marks expired)
        vc_wbegin(e);
        bool was = e->gpu;
        uint32_t last = 0;
        for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) if (vc_page_gpu[p] > last) last = vc_page_gpu[p];
        e->gpu = last && now < last + VC_GPU_HOLD;
        e->gpu_until = last + VC_GPU_HOLD;
        e->epoch = vc_epoch;
        if (was && !e->gpu) { e->gen = 0; vc_gpu_expired = true; }  // (the GPU's writes aren't tracked: copied whole)
        vc_wend(e);
    }
    if (e->gpu) { prof_vc_bypass++; return false; }
    vc_claim(e, pend);
    prof_vc_hits++;
    *buf = e->buf;
    *off = addr - gwin_base - lo;
    return true;
}

// (the shadow workers) vcache_lookup's answer for a range cached already, up to date with the GPU's writes, without the
// workers' lock: its entry read under its sequence count, claimed for the submission by compare-and-swap. The first use
// of a range in each submission took the lock: ~3400 a frame in a town at World detail 4x, two thirds of the parts' wait
// for it. 1: cached (buf, off); 0: the GPU writes its pages - not cached, decided already (vcache_lookup's bypass);
// -1: vcache_lookup under the lock.
static int vc_quick(uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (vc_on <= 0 || !gmem_buf || gc_lazy || gmem_lo || size > VC_MAX_RANGE) return -1;
    uint64_t lo = (addr - gwin_base) & ~(VC_PAGE - 1), hi = (addr - gwin_base + size + VC_PAGE - 1) & ~(VC_PAGE - 1);
    if (hi > gmem_size) return -1;
    uint64_t pend = vk_pending_seq();
    uint64_t h = (lo * 0x9E3779B97F4A7C15ull) ^ (hi * 0xC2B2AE3D27D4EB4Full);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (VC_TAB - 1);
    for (int probe = 0; probe < 16; probe++, slot = (slot + 1) & (VC_TAB - 1)) {
        VcEnt *e = &vc_tab[slot];
        uint32_t s1 = atomic_load_explicit(&e->sq, memory_order_acquire);
        if (s1 & 1) return -1;
        VkBuffer b = __atomic_load_n(&e->buf, __ATOMIC_ACQUIRE);
        if (!b) return -1;  // (not cached: vcache_lookup makes it)
        if (e->lo != lo || e->hi != hi) continue;
        uint32_t now = (uint32_t)vk_frame_counter() + 1;
        if (e->epoch != vc_epoch || (e->gpu && now >= e->gpu_until)) return -1;  // (its GPU marks to look at again)
        if (e->gpu) {  // (GPU-written: read in the mirror or in place)
            atomic_thread_fence(memory_order_acquire);
            if (atomic_load_explicit(&e->sq, memory_order_relaxed) != s1) return -1;
            prof_vc_bypass++;
            return 0;
        }
        vc_claim(e, pend);  // (claimed here: listed here, whatever follows)
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&e->sq, memory_order_relaxed) != s1 || __atomic_load_n(&e->used_seq, __ATOMIC_ACQUIRE) != pend)
            return -1;
        prof_vc_hits++;
        *buf = b;
        *off = addr - gwin_base - lo;
        return 1;
    }
    return -1;
}

static int vc_pend_cmp(const void *a, const void *b) {
    uint64_t x = vc_tab[*(const uint32_t *)a].lo, y = vc_tab[*(const uint32_t *)b].lo;
    return x < y ? -1 : x > y;
}

// At submission (vk_submit_extra, before the upload buffer closes): the pages the CPU wrote in the ranges this
// submission draws from - one write tracking query per run of nearby ranges (a query costs about as much as scanning
// ~64 pages: ranges closer than that share one) - and their copies into the cache, ahead of the draws.
void vc_flush(void) {
    if (!vc_npend) return;
    qsort(vc_pend, (size_t)vc_npend, sizeof vc_pend[0], vc_pend_cmp);
    uint32_t g = ++vc_gen;
    static uint64_t pages[4096];
    uint64_t t0 = SDL_GetTicksNS();
    for (int i = 0; i < vc_npend;) {
        uint64_t lo = vc_tab[vc_pend[i]].lo, hi = vc_tab[vc_pend[i]].hi;
        int j = i + 1;
        for (; j < vc_npend && vc_tab[vc_pend[j]].lo <= hi + 64 * VC_PAGE; j++)
            if (vc_tab[vc_pend[j]].hi > hi) hi = vc_tab[vc_pend[j]].hi;
        uint64_t pg = VC_PAGE;
        int n = plat_written_pages(gwin_base + lo, hi - lo, pages, 4096, &pg);
        prof_vc_queries++;
        if (n < 0 || n >= 4096) for (uint64_t p = lo / VC_PAGE; p < hi / VC_PAGE; p++) vc_seen(p, g);  // (all)
        else for (int k = 0; k < n; k++) vc_seen((pages[k] - gwin_base) / VC_PAGE, g);
        if (n > 0) prof_vc_scan_pages += (uint64_t)n;
        i = j;
    }
    prof_vc_scan_ns += SDL_GetTicksNS() - t0;
    VkCommandBuffer cb = vk_upload_cmd();
    if (vc_gpu_expired) { vk_full_barrier(cb); vc_gpu_expired = false; }  // (after the earlier submissions' writes)
    for (int i = 0; i < vc_npend; i++) {
        VcEnt *e = &vc_tab[vc_pend[i]];
        VkBufferCopy cp[64];
        int nc = 0;
        for (uint64_t p = e->lo / VC_PAGE; p < e->hi / VC_PAGE; p++) {
            if (e->gen && !vc_dirty(p, e->gen)) continue;
            uint64_t o = p * VC_PAGE;
            if (nc && cp[nc - 1].srcOffset + cp[nc - 1].size == o) { cp[nc - 1].size += VC_PAGE; continue; }
            if (nc == 64) { nc = 1; cp[0] = (VkBufferCopy){e->lo, 0, e->hi - e->lo}; break; }  // (scattered: the whole range)
            cp[nc++] = (VkBufferCopy){o, o - e->lo, VC_PAGE};
        }
        if (nc) {
            gmem_copy(cb, e->buf, cp, nc, false);
            for (int k = 0; k < nc; k++) prof_vc_copied += cp[k].size;
        }
        e->gen = g;
    }
    vc_npend = 0;
}

// Vertex / index data from the imported guest heap (renderer.m bind_guest reads g_guest_buffer the same way): the
// device-local cache above, else read in place - no per-draw copy, no index scan for the copy size. The game fences its
// CPU writes as on the Switch. Falls back to copies outside the heap or without the import; LARECOMP_VK_COPY_VERTICES=1
// forces copies.

// index: an index buffer. Never read from the transform feedback mirror: transform feedback writes vertex data only, the mirror has no INDEX_BUFFER usage (the validation
// layer: vkCmdBindIndexBuffer on it, 423 times in 15 s), and index data the game wrote into the arena's memory after a session (chunks count as written by 256-byte
// chunk) was read from stale mirror bytes: invisible walls and empty text boxes.
static bool guest_direct_(int kind, uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off);
static bool guest_direct(int kind, uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    if (!sw_cb && !sw_rt_lock) return guest_direct_(kind, addr, size, buf, off);
    if (!sw_cb) { SWL(); bool r = guest_direct_(kind, addr, size, buf, off); SWU(); return r; }  // (async runs in flight)
    // (a shadow worker: the same range again in this submission - the answer it had, without the lock. An entry the
    // submission drew from isn't dropped during it - vcache_lookup; GPU-written pages change vc_epoch; the mirror's
    // chunks change between frames only)
    static struct { uint64_t addr, size, seq, frame; VkBuffer buf; VkDeviceSize off; uint32_t epoch; uint8_t kind, ok; } memo_t[NCTX][1024];
    uint64_t seq = vk_pending_seq(), frame = vk_frame_counter();
    uint32_t ep = vc_epoch;
    uint64_t hk = (addr * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full) ^ (uint64_t)kind;
    hk ^= hk >> 31;
    __typeof__(memo_t[0][0]) *e = &memo_t[sw_ctx()][(hk >> 20) & 1023];
    if (e->addr == addr && e->size == size && e->kind == kind && e->seq == seq && e->frame == frame && e->epoch == ep) {
        *buf = e->buf; *off = e->off;
        return e->ok;
    }
    // (another part's answer for this range in this submission, read without the lock: seqlock)
    static struct { _Atomic uint32_t sq; uint64_t addr, size, seq, frame; VkBuffer buf; VkDeviceSize off; uint32_t epoch; uint8_t kind, ok; } sh[4096];
    uint32_t si = (uint32_t)(hk >> 36) & 4095;
    uint32_t s1 = atomic_load_explicit(&sh[si].sq, memory_order_acquire);
    if (!(s1 & 1) && sh[si].addr == addr && sh[si].size == size && sh[si].kind == kind && sh[si].seq == seq && sh[si].frame == frame &&
        sh[si].epoch == ep) {
        VkBuffer b = sh[si].buf;
        VkDeviceSize o = sh[si].off;
        bool ok = sh[si].ok;
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&sh[si].sq, memory_order_relaxed) == s1) {
            *buf = b; *off = o;
            *e = (__typeof__(*e)){addr, size, seq, frame, b, o, ep, (uint8_t)kind, ok};
            return ok;
        }
    }
    static int copy = -1;
    if (copy < 0) copy = getenv("LARECOMP_VK_COPY_VERTICES") != NULL;
    static int binds = -1;
    if (binds < 0) binds = getenv("LARECOMP_LOG_BINDS") != NULL;
    bool r;
    int q = -1;  // (guest_direct_'s answer without the lock, where the vertex cache has decided the range already)
    if (!copy && !binds && addr >= gwin_base && GM_IN(addr, size) && !(addr + size > gwin_base + gmem_lo + gmem_size) &&
        (q = vc_quick(addr, size, buf, off)) >= 0) {
        if (q == 1) stat_vtx_cached += size;
        else if (kind == 0 && xm_lookup(addr, size, buf, off)) stat_vtx_cached += size;  // (as guest_direct_ goes on)
        else { stat_vtx_host += size; *buf = gmem_buf; *off = GM_OFF(addr); }
        // (the shared memo's writers hold the lock: this thread's own memo only)
        *e = (__typeof__(*e)){addr, size, seq, frame, *buf, *off, ep, (uint8_t)kind, 1};
        return true;
    }
    SWL_AT(5894);
    r = guest_direct_(kind, addr, size, buf, off);
    atomic_fetch_add_explicit(&sh[si].sq, 1, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    sh[si].addr = addr; sh[si].size = size; sh[si].seq = seq; sh[si].frame = frame; sh[si].buf = r ? *buf : VK_NULL_HANDLE;
    sh[si].off = r ? *off : 0; sh[si].epoch = ep; sh[si].kind = (uint8_t)kind; sh[si].ok = r;
    atomic_fetch_add_explicit(&sh[si].sq, 1, memory_order_release);
    SWU();
    *e = (__typeof__(*e)){addr, size, seq, frame, r ? *buf : VK_NULL_HANDLE, r ? *off : 0, ep, (uint8_t)kind, r};
    return r;
}
static bool guest_direct_(int kind, uint64_t addr, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {  // kind: 0 vertex, 1 index (bind_note)
    static int copy = -1;
    if (copy < 0) copy = getenv("LARECOMP_VK_COPY_VERTICES") != NULL;
    if (copy || addr < gwin_base) return false;
    bind_note(kind, addr, size);
    if (gc_lazy) {
        uint64_t o = addr - HEAP_BASE;
        if (!size || o + size > HEAP_MAX) return false;
        int k = (int)(o >> gc_shift);
        if ((int)((o + size - 1) >> gc_shift) != k || !gmem_covers(o, size)) return false;  // (crossing chunks: copied)
        if (vcache_lookup(addr, size, buf, off)) return true;
        *buf = gc[k].buf;
        *off = o - ((uint64_t)k << gc_shift);
        return true;
    }
    if (addr + size > gwin_base + gmem_lo + gmem_size && gwin_want() != gmem_tried) { gmem_tried = gwin_want(); gmem_info(); }
    if (gmem_lo && xm_lookup(addr, size, buf, off)) { stat_vtx_cached += size; return true; }  // (the mirror needs no import)
    if (!GM_IN(addr, size)) return false;
    if (vcache_lookup(addr, size, buf, off)) { stat_vtx_cached += size; return true; }
    if (kind == 0 && xm_lookup(addr, size, buf, off)) { stat_vtx_cached += size; return true; }
    stat_vtx_host += size;
    *buf = gmem_buf;
    *off = GM_OFF(addr);
    return true;
}

static int g_cs_n;  // (compute program table below)

// In-place reboot (device.c backend_reset; GPU idle): drop what is keyed by guest addresses - programs by code
// address (the old Prog / Pipe objects are leaked, so pipeline keys holding their pointers can't collide), compute
// programs, shared copies - and the heap import: its pages are replaced (re-imported on first use).
void vk_exec_reset(void) {
    kv_clear(&prog_kv);
    kv_clear(&pipe_kv);
    key_gen++;  // (the key memo: the tables' answers are gone)
    ds_gen++;   // (set reuse: the heap import below is destroyed at once)
    g_cs_n = 0;
    gcopy_gen++;
    if (gmem_buf) {
#ifdef __linux__
        bool placed = gmem_placed;
        heap_unmap_placed();
#endif
        vkDestroyBuffer(g_vk.dev, gmem_buf, NULL);
        vkFreeMemory(g_vk.dev, gmem_mem, NULL);
        gmem_buf = VK_NULL_HANDLE;
        gmem_mem = VK_NULL_HANDLE;
#ifdef __linux__
        if (placed && !plat_reset_guest(HEAP_BASE, HEAP_BASE + HEAP_MAX)) FATAL("cannot restore the guest heap mapping\n");
        gmem_dmabuf = false;  // (the next boot decides again)
#endif
    }
    gmem_size = gmem_tried = 0;
    gmem_lo = 0;
    xm_lo = 0, xm_back_n = 0, xm_moves++;  // (the transform feedback mirror: placed again, nothing written yet)
    memset(xm_written, 0, sizeof xm_written);
    for (int k = 0; k < gc_n; k++) {  // (lazy chunks: their pages are replaced too)
        if (gc[k].buf) { vkDestroyBuffer(g_vk.dev, gc[k].buf, NULL); vkFreeMemory(g_vk.dev, gc[k].mem, NULL); }
        memset(&gc[k], 0, sizeof gc[k]);
    }
    gc_imported = 0; gc_scanned = 0; gc_imported_bytes = 0; gc_gen++;  // (gc_remembered stays: the same heap layout next boot)
    gc_remembered_heap = 0;  // (imported again at the next boot's first chunk set)
    if (gc_miss) { memset(gc_miss, 0, sizeof(uint32_t) * GC_MAX); memset(gc_miss + GC_MAX, 0xFF, sizeof(uint32_t) * GC_MAX); }
    for (int i = 0; i < VC_TAB; i++)  // (the vertex cache: copies of the old pages)
        if (vc_tab[i].buf) vmaDestroyBuffer(g_vk.vma, vc_tab[i].buf, vc_tab[i].mem);
    memset(vc_tab, 0, sizeof vc_tab);
    vc_npend = 0;
    vc_free_dead(true);
    vc_bytes = 0;
    if (vc_page_gpu) memset(vc_page_gpu, 0, gwin_max / VC_PAGE * sizeof *vc_page_gpu);
    vc_epoch++;
    snap_reset();  // (overlap: the storage copies' ring follows the heap, decided again once it is set up)
    st_defaults();
}

// Shadow (sampler*Shadow) slots need a depth view whose format can be compared (VUID-vkCmdDrawIndexed-None-06479):
// the colour dummy or a colour texture there is replaced by a 1x1 depth dummy at 1.0 (lit, textures.c). Each cause is
// logged once (LARECOMP_SHADOW_SLOT_LOG=1: every change of cause / handle, up to 200 lines), with the texture's
// address, size and format, so a run shows why the slot had nothing usable.
static void shadow_slot_check(int stage, const ShaderTexture *st_, int i, uint64_t h, uint32_t tic, const NvnTexEntry *e,
                              NvnTexture *t, VkImageView *view) {
    VkImageView dummy = vk_dummy_view(st_->type);
    if (*view && *view != dummy && t && vk_tex_depth_compare(t)) return;
    enum { HANDLE0, NO_ENTRY, FINALIZED, UNREGISTERED, MULTISAMPLE, FEEDBACK_ARRAY, VIEW_FAILED, TYPE_MISMATCH,
           DEPTH_NO_COMPARE, COLOUR, NCAUSE };
    static const char *names[NCAUSE] = {"handle 0 (nothing bound)", "handle past the texture pool", "texture finalized",
                                        "pool slot not registered", "multisample texture", "feedback snapshot as array",
                                        "view creation failed", "type mismatch", "depth format without compare support",
                                        "colour format"};
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    int why;
    if (!h) why = HANDLE0;
    else if (!e) why = NO_ENTRY;
    else if (!t) why = e->dead_format ? FINALIZED : UNREGISTERED;
    else if (*view == dummy || !*view) {
        bool fb = false;
        for (int k = 0; k < nfeedback; k++) fb |= feedback_tex[k] == e->tex;
        why = b && b->samples > 1 ? MULTISAMPLE : fb && st_->type == TEXTYPE_2D_ARRAY ? FEEDBACK_ARRAY : !*view ? VIEW_FAILED : TYPE_MISMATCH;
    } else why = b && (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? DEPTH_NO_COMPARE : COLOUR;
    static int verbose = -1, nverbose;
    static bool logged[NCAUSE];
    static int last_why = -1;
    static uint64_t last_h;
    if (verbose < 0) verbose = getenv("LARECOMP_SHADOW_SLOT_LOG") != NULL;
    bool log = !logged[why] || (verbose && nverbose < 200 && (why != last_why || h != last_h));
    if (log) {
        logged[why] = true;
        if (verbose) nverbose++;
        last_why = why; last_h = h;
        if (t)
            LOG("[vk] shadow slot t%d (stage %d, type %d) f%llu fs %016llx: %s - handle %#llx tic %u, texture %#llx %ux%ux%u "
                "target %u nvn fmt %#x vk fmt %d aspect %#x samples %u: 1x1 depth dummy instead%s\n",
                i, stage, st_->type, (unsigned long long)vk_frame_counter(), (unsigned long long)g_cur_fs_hash, names[why],
                (unsigned long long)h, tic, (unsigned long long)t->addr, t->width, t->height, t->depth, t->target, t->format,
                b ? b->format : 0, b ? b->aspect : 0, b ? b->samples : 0, verbose ? "" : " (once per cause)");
        else
            LOG("[vk] shadow slot t%d (stage %d, type %d) f%llu fs %016llx: %s - handle %#llx tic %u (finalized nvn fmt %#x): "
                "1x1 depth dummy instead%s\n", i, stage, st_->type, (unsigned long long)vk_frame_counter(),
                (unsigned long long)g_cur_fs_hash, names[why], (unsigned long long)h, tic, e && e->dead_format ? e->dead_format - 1 : 0,
                verbose ? "" : " (once per cause)");
    }
    *view = vk_dummy_depth_view(st_->type);
}

static void texture_info(int stage, ShaderInfo *si, int i, VkDescriptorImageInfo *out, float sc[2], uint32_t wh[2], float fr[2]) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    ShaderTexture *st_ = &si->tex[i];
    int slot = ((int)st_->handle_offset - 0x20) / 8;  // 64-bit handles from offset 0x20
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[stage][slot] : 0;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler smp = VK_NULL_HANDLE;
    uint32_t tic = h & 0xFFFFF, tsc = (uint32_t)(h >> 20) & 0xFFF;
    sc[0] = sc[1] = 1;
    wh[0] = wh[1] = 0;
    fr[0] = fr[1] = 1;
    NvnTexEntry *ent = NULL;  // (shadow slots: what the view came from)
    if (h && st.tpool && tic < (uint32_t)st.tpool->count) {
        NvnTexEntry *e = ent = &((NvnTexEntry *)st.tpool->entries)[tic];
        if (e->tex) {
            // Per-frame memo: ~2400 lookups per overworld frame recomputed the view (format / range math, cache
            // probe). The first lookup of a frame still runs vk_view (upload check); a view that may have been
            // destroyed (eviction, image re-created) bumps vk_view_gen. LARECOMP_VK_NO_TEX_MEMO=1 disables.
            extern uint32_t vk_view_gen;
            extern uint64_t vk_frame_counter(void);
            static int nomemo = -1;
            if (nomemo < 0) nomemo = ENV("LARECOMP_VK_NO_TEX_MEMO") != NULL;
            // (valid while the texture's backend and its vgen stay: TexBackend.vgen; none while copy aliases live)
            static struct { NvnTexture *tex; TexBackend *b; uint64_t vh, frame, vg; uint8_t type, has_view; VkImageView view; } memo_t[NCTX][4096];
            __typeof__(memo_t[0][0]) *memo = memo_t[sw_ctx()];
            uint64_t vh = e->has_view ? hash_words(0, &e->view, sizeof e->view) : 0, frame = vk_frame_counter();
            // (mixed: the textures' addresses share their low bits - a shift and xor put them in a few slots)
            uint64_t hk = ((uintptr_t)e->tex * 0x9E3779B97F4A7C15ull) ^ (vh * 0xC2B2AE3D27D4EB4Full) ^ (uint64_t)st_->type;
            hk ^= hk >> 31;
            uint32_t ms = (uint32_t)(hk >> 20) & 4095;  // (a town at World detail 4x draws with several thousand a frame)
            TexBackend *tb = e->tex->backend;
            uint64_t vg = tb ? tb->vgen : 0;
            if (!nomemo && !naliases && memo[ms].tex == e->tex && memo[ms].vh == vh && memo[ms].frame == frame && tb && memo[ms].b == tb &&
                memo[ms].vg == vg && memo[ms].type == st_->type && memo[ms].has_view == e->has_view) {
                view = memo[ms].view;
            } else {
                // (what another part looked up this frame - a texture's first lookup of the frame, with its upload
                // check, made once - or in an earlier frame, when no upload check is due now (vk_tex_quiet: 7 frames in
                // 8 for most textures): the view is the same until vk_view_gen moves. Every thread; read without the
                // shadow workers' lock: seqlock. ~2300 lookups a frame in a town at World detail 4x took the lock.)
                static struct { _Atomic uint32_t seq; NvnTexture *tex; TexBackend *b; uint64_t vh, frame, vg; uint8_t type, has_view; VkImageView view; } sh[32768];
                extern bool vk_tex_quiet(NvnTexture *t);
                bool got = false;
                uint32_t si = (uint32_t)(hk >> 36) & 32767;
                if (!nomemo && !naliases) {
                    uint32_t s1 = atomic_load_explicit(&sh[si].seq, memory_order_acquire);
                    if (!(s1 & 1) && sh[si].tex == e->tex && sh[si].vh == vh && tb && sh[si].b == tb && sh[si].vg == vg &&
                        sh[si].type == st_->type && sh[si].has_view == e->has_view && (sh[si].frame == frame || vk_tex_quiet(e->tex))) {
                        VkImageView v = sh[si].view;
                        atomic_thread_fence(memory_order_acquire);
                        if (atomic_load_explicit(&sh[si].seq, memory_order_relaxed) == s1) view = v, got = true;
                    }
                }
                if (!got) {
                    SWL_AT(6052);
                    view = vk_view(naliases ? alias_read(e->tex) : e->tex, &e->view, e->has_view, false, st_->type);
                    if (!naliases) {
                        atomic_fetch_add_explicit(&sh[si].seq, 1, memory_order_relaxed);
                        atomic_thread_fence(memory_order_release);
                        tb = e->tex->backend, vg = tb ? tb->vgen : 0;  // (vk_view made it, or its views changed)
                        sh[si].tex = e->tex; sh[si].vh = vh; sh[si].frame = frame; sh[si].b = tb; sh[si].vg = vg;
                        sh[si].type = (uint8_t)st_->type; sh[si].has_view = e->has_view; sh[si].view = view;
                        atomic_fetch_add_explicit(&sh[si].seq, 1, memory_order_release);
                    }
                    SWU();
                }
                tb = e->tex->backend;
                memo[ms].tex = naliases ? NULL : e->tex; memo[ms].vh = vh; memo[ms].frame = frame; memo[ms].b = tb; memo[ms].vg = tb ? tb->vgen : 0;
                memo[ms].type = (uint8_t)st_->type; memo[ms].has_view = e->has_view; memo[ms].view = view;
            }
            renderer_texture_scale(e->tex, &sc[0], &sc[1]);
            renderer_texture_fraction(e->tex, &fr[0], &fr[1]);
            wh[0] = e->tex->width; wh[1] = e->tex->height;
            {   // A shadow map drawn bigger (textures.c tex_shadow_scale) still reports the game's size to the shaders'
                // size queries, and Pokemon Z-A spreads its 16 filter taps by (1 / that size) x a radius: the same
                // blur in the map's coordinates at any size. F1 > Shadow sharpness (shadow_sharpness in the ini;
                // LARECOMP_SHADOW_SHARP=<s> overrides it), up to the map's own scale: the depth map's size queries
                // report s times the game's size, so the taps sit s times closer - s times less blur, as many texels
                // wide as at the game's size. Applies at once. Only the queries see it: the reads use normalized
                // coordinates (uvs() takes the fraction, not this scale).
                static int env_sharp = -2;
                if (env_sharp == -2) { const char *v = ENV("LARECOMP_SHADOW_SHARP"); env_sharp = v ? atoi(v) : -1; }
                int sharp = env_sharp >= 0 ? env_sharp : g_settings.shadow_sharp;
                TexBackend *sb = sharp > 1 ? (TexBackend *)e->tex->backend : NULL;
                if (sb && sb->shadow_k > 1 && (sb->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
                    float d = sharp < sb->shadow_k ? (float)sharp : sb->shadow_k;
                    sc[0] /= d; sc[1] /= d;
                } else if (sb && sb->shadow_k > 1) {
                    // The 8-bit object-id map drawn beside it (Z-A's self-shadowing guard: a tap on the same object at
                    // about the same depth counts as lit) is read with texelFetch at trunc(its queried size x the
                    // tap's uv): at the game's size that is the 4x4 block's centre, up to 2 texels from where the depth
                    // tap is - with the taps narrowed (sharpness 3-4) the id no longer belongs to the depth's texel and
                    // the guard fails along a character's inner edges (dark streaks across a shirt, a Pokémon's
                    // belly). Its real size and no scale instead: the id is read at the depth tap's own texel.
                    sc[0] = sc[1] = 1;
                }
            }
            for (int k = 0; k < nfeedback; k++)
                if (e->tex == feedback_tex[k] && (st_->type == TEXTYPE_2D || st_->type == TEXTYPE_2D_ARRAY)) {
                    view = feedback_view[k];
                    // The snapshot's view above is its depth, whatever the slot asked for. A slot in NVN depth/stencil
                    // mode STENCIL (the view's, else the texture's - as vk_view) reads the snapshot's stencil instead:
                    // the copy holds both aspects. Scarlet / Violet's deferred lighting samples the scene's
                    // depth/stencil twice while it is the pass's depth attachment - once as depth, once through a
                    // STENCIL view for each pixel's shading model (bits 0..2) and its "skip" bit (6) - and its
                    // usampler2D got the depth view: an integer sampler on a float format, undefined in Vulkan. The
                    // Arc B580 returned bits of the depth value, which run in a regular lattice along each triangle's
                    // plane - the stripes / per-triangle pixel noise on every lit surface (v0.0.14).
                    TexBackend *fb = feedback_img[k];
                    if (fb && (fb->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (fb->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) &&
                        (e->has_view && (e->view.setmask & 16) ? e->view.ds_mode : e->tex->ds_mode) == 1) {
                        static const VkComponentMapping id = {0};
                        view = vk_image_view(fb, VK_IMAGE_VIEW_TYPE_2D, fb->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_STENCIL_BIT);
                    }
                    if (st_->type == TEXTYPE_2D_ARRAY) view = vk_dummy_view(TEXTYPE_2D_ARRAY);  // (2D snapshot as array: rare)
                }
        } else if (e->dead_format) {
            nvn_dead_slot_sampled(e, tic);
        }
    }
    if (h && st.spool && tsc < (uint32_t)st.spool->count) {
        NvnSamplerDesc *sd = &((NvnSamplerDesc *)st.spool->entries)[tsc];
        smp = st_->stencil || st_->sint ? vk_sampler_nearest(sd) : vk_sampler(sd);  // (integer texels: unfiltered)
    }
    {   // (debug) LARECOMP_CS_DUMMY_TEX=<compute code hex>: that compute shader samples 1x1 black textures
        static uint64_t dummy_cs = ~0ull;
        if (dummy_cs == ~0ull) dummy_cs = ENV("LARECOMP_CS_DUMMY_TEX") ? strtoull(ENV("LARECOMP_CS_DUMMY_TEX"), NULL, 16) : 0;
        if (stage == 5 && dummy_cs && st.program && (dummy_cs == 1 || st.program->stages[5].code_addr == dummy_cs))
            view = VK_NULL_HANDLE;  // (1: every compute shader - code addresses change between runs)
    }
    {   // A compute shader reading a scaled depth target with a mip chain at a non-integer scale (1.5x, Match display,
        // Dynamic, the shares) gets a 1x1 dummy: no occluder, so nothing is culled for being hidden. That is Pokémon
        // Z-A's GPU occlusion culling reading its depth pyramid (the scene depth's own mip levels). Scaled by 1.5 the
        // levels don't keep the ratio (135 -> 202, 33 -> 50, 16 -> 25), the culling read the wrong depths and dropped
        // visible objects - lamp posts, hedges, parts of buildings vanishing as the player walked. Integer scales are
        // exact and keep it. The game's frustum and distance culling still run. LARECOMP_FRACTIONAL_OCCLUSION=1: keep it.
        static int keep = -1;
        if (keep < 0) keep = ENV("LARECOMP_FRACTIONAL_OCCLUSION") && atoi(ENV("LARECOMP_FRACTIONAL_OCCLUSION"));
        TexBackend *pb = stage == 5 && view && !keep && ent && ent->tex ? (TexBackend *)ent->tex->backend : NULL;
        if (pb && pb->scalable && pb->levels > 1 && (pb->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) &&
            (pb->sx != floorf(pb->sx) || pb->sy != floorf(pb->sy))) {
            static int nlog;
            if (nlog++ < 3) LOG("[vk] non-integer scale %.2f: occlusion culling's depth pyramid (%ux%u, %u levels) off for this compute shader\n",
                                pb->sx, ent->tex->width, ent->tex->height, pb->levels);
            view = VK_NULL_HANDLE;
        }
    }
    if (st_->shadow)
        shadow_slot_check(stage, st_, i, h, tic, ent, ent && ent->tex ? (naliases ? alias_read(ent->tex) : ent->tex) : NULL, &view);
    if (!view) view = vk_dummy_view(st_->type);
    if (!smp) smp = vk_default_sampler(!st_->stencil && !st_->sint);
    *out = (VkDescriptorImageInfo){smp, view, VK_IMAGE_LAYOUT_GENERAL};
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// Writes the descriptor set of one stage. gx..gz: compute group counts (driver constants).
// pre_ii: texture descriptors already resolved by the caller (draw: fragment textures, needed earlier for the
// viewport decision), or NULL.
static float g_mirror_w_t[NCTX];  // mirror mode's composite: its render target width (rscale[65].x; 0 otherwise)
#define g_mirror_w (g_mirror_w_t[sw_ctx()])

// (set reuse) what a set holds, one entry per descriptor written, in stage_set's order (fixed per program)
typedef struct { uint32_t binding, type; uint64_t a, b, c; } SetEnt;
typedef struct SetCache { uint64_t seq; uint32_t view_gen, ds_gen; int n; uint32_t next; VkDescriptorSet set; SetEnt e[MAX_BIND]; } SetCache;
// (SET_WAYS sets remembered per program, set index and context, the oldest replaced: a program drawing a few materials in
// turn - a town's walls, roofs, windows - wrote a new set at almost every draw with the last one alone)
#define SET_WAYS 4

// dyn / ndyn: the set's dynamic offsets are appended at dyn[*ndyn] (binding order, as vkCmdBindDescriptorSets takes
// them); NULL for programs without dynamic buffers (compute).
static VkDescriptorSet stage_set(Prog *p, VkDescriptorSetLayout layout, uint32_t set, int stage, ShaderInfo *si, const float *rscale_row0,
                                 const float tint[RECOLOUR_TINT_FLOATS], uint32_t gx, uint32_t gy, uint32_t gz, const VkDescriptorImageInfo *pre_ii,
                                 uint32_t *dyn, uint32_t *ndyn) {
    VkWriteDescriptorSet w[MAX_BIND];
    VkDescriptorBufferInfo bi[MAX_BIND];
    VkDescriptorImageInfo ii[MAX_BIND];
    uint32_t dbind[MAX_BIND], doff[MAX_BIND];
    int n = 0, nd = 0;
    int sg = stage == 1 ? 1 : 0;
    bool lean = cbuf_lean();
    // ([65]: mirror mode, renderer.m; [66..69]: vertex transform feedback buffers) (lean: only rows 0, 1..ntex and 65
    // are set and read - not with transform feedback, whose rows the lean copy leaves out)
    float rs[70][4];
    bool xfb_rows = stage == 0 && xfb_on;
    if (!lean || xfb_rows) memset(rs, 0, sizeof rs);
    rs[0][0] = rs[0][1] = 1; rs[0][2] = rs[0][3] = 0;
    if (rscale_row0) memcpy(rs[0], rscale_row0, 16);
    rs[65][0] = g_mirror_w; rs[65][1] = rs[65][2] = rs[65][3] = 0;
    if (stage == 0 && xfb_on)
        for (int b = 0; b < 4; b++) {
            uint64_t a = xfb_addr[b];
            if (!a || a < gwin_base || a - gwin_base + xfb_size[b] > gwin_max || (a & 3)) continue;
            uint64_t base = xm_draw ? xm_lo : gwin_base + gmem_lo;  // (the transform feedback mirror's offsets, or the import's)
            if (a < base) continue;
            uint32_t u[4] = {(uint32_t)((a - base) >> 2), (uint32_t)(xfb_size[b] >> 2), 1, 0};  // (offset, size in words)
            memcpy(rs[66 + b], u, 16);
        }
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER && b->binding >= 32) {
            int ti = (int)b->binding - 32;
            float sc[2], fr[2]; uint32_t wh[2];
            if (ti < si->ntex && pre_ii && ti < 64) {
                ii[n] = pre_ii[ti]; sc[0] = tex_sc[sg][ti][0]; sc[1] = tex_sc[sg][ti][1]; wh[0] = tex_wh[sg][ti][0]; wh[1] = tex_wh[sg][ti][1];
                fr[0] = tex_fr[sg][ti][0]; fr[1] = tex_fr[sg][ti][1];
            }
            else if (ti < si->ntex) texture_info(stage, si, ti, &ii[n], sc, wh, fr);
            else { ii[n] = (VkDescriptorImageInfo){vk_default_sampler(true), vk_dummy_view(TEXTYPE_2D), VK_IMAGE_LAYOUT_GENERAL}; sc[0] = sc[1] = 1; wh[0] = wh[1] = 0; fr[0] = fr[1] = 1; }
            if (ti < 64) {
                tex_sc[sg][ti][0] = sc[0]; tex_sc[sg][ti][1] = sc[1];
                tex_wh[sg][ti][0] = wh[0]; tex_wh[sg][ti][1] = wh[1];
                tex_fr[sg][ti][0] = fr[0]; tex_fr[sg][ti][1] = fr[1];
            }
            w[n].pImageInfo = &ii[n];
        } else if (is_ubo(b->type) && b->binding < 17) {
            bi[n] = cbuf_info(stage, si, (int)b->binding, gx, gy, gz);
            w[n].pBufferInfo = &bi[n];
        } else if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
            if (b->binding == GFAST_BUFFER_INDEX || (b->binding == XFB_BUFFER_INDEX && xm_draw) || !(sw_cb || sw_rt_lock) ||
                !gmem_info_quick(&bi[n])) {
                SWL_AT(6205);
                bi[n] = b->binding == GFAST_BUFFER_INDEX ? gfast_info()
                      : b->binding == XFB_BUFFER_INDEX && xm_draw ? (VkDescriptorBufferInfo){xm_buf, 0, XM_SIZE} : gmem_info();
                SWU();
            }
            w[n].pBufferInfo = &bi[n];
        } else {
            continue;  // rscale / tint: written below, once the textures (and their scales) are known
        }
        n++;
    }
    for (int i = 0; i < p->nbinds; i++) {
        VkBindingInfo *b = &p->binds[i];
        if (b->set != set || !is_ubo(b->type) || (b->binding != 18 && b->binding != 19 && b->binding != GMAP_BUFFER_INDEX)) continue;
        w[n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = b->binding,
                                      .descriptorCount = 1, .descriptorType = b->type};
        if (b->binding == GMAP_BUFFER_INDEX) {  // (compute) this dispatch's fast-copy table: gfast_prepare
            bi[n] = ubo_span(vk_transient_copy(gmap_tab, sizeof gmap_tab, 256), sizeof gmap_tab);
        } else if (b->binding == 18) {
            for (int t = 0; t < si->ntex && t < 64; t++) {  // scale x, y, content part x, y
                rs[1 + t][0] = tex_sc[sg][t][0]; rs[1 + t][1] = tex_sc[sg][t][1]; rs[1 + t][2] = tex_fr[sg][t][0]; rs[1 + t][3] = tex_fr[sg][t][1];
            }
            if (lean && !xfb_rows) bi[n] = ubo_span(rscale_dedupe(sg, (const float (*)[4])rs, si->ntex < 64 ? si->ntex : 64), 66 * 16);
            else bi[n] = ubo_span(copy_dedupe(sg ? DEDUPE_RSCALE_F : DEDUPE_RSCALE_V, rs, sizeof rs, 256), sizeof rs);
        } else {
            float zero[RECOLOUR_TINT_FLOATS] = {0};
            bi[n] = ubo_span(copy_dedupe(DEDUPE_TINT, tint ? tint : zero, sizeof zero, 256), sizeof zero);
        }
        w[n].pBufferInfo = &bi[n];
        n++;
    }
    // dynamic buffers: the offset moves to the bind (the descriptor keeps buffer and range)
    for (int i = 0; i < n; i++)
        if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
            int j = nd++;  // (insertion by binding: the order vkCmdBindDescriptorSets takes the offsets in)
            while (j > 0 && dbind[j - 1] > w[i].dstBinding) { dbind[j] = dbind[j - 1]; doff[j] = doff[j - 1]; j--; }
            dbind[j] = w[i].dstBinding;
            doff[j] = (uint32_t)bi[i].offset;
            bi[i].offset = 0;
        }
    if (nd) {
        if (!dyn || !ndyn) FATAL("stage_set: dynamic buffers without an offset array\n");
        memcpy(dyn + *ndyn, doff, (size_t)nd * sizeof doff[0]);
        *ndyn += (uint32_t)nd;
    }
    extern uint32_t vk_view_gen;
    SetCache *c = NULL;
    SetEnt e[MAX_BIND];
    SetCache *ways = NULL;
    if (p->reuse && set < 2) {  // the same contents as one of this program's last sets of this index: bind that one again
        struct SetCache **cs = &p->scache[sw_ctx()][set];
        if (!*cs) *cs = calloc(SET_WAYS, sizeof **cs);  // (each context its own: no other thread writes it)
        ways = *cs;
        c = &ways[ways[0].next % SET_WAYS];  // (the one replaced on a miss)
        memset(e, 0, (size_t)n * sizeof e[0]);  // (padding: compared with memcmp)
        for (int i = 0; i < n; i++) {
            e[i].binding = w[i].dstBinding;
            e[i].type = (uint32_t)w[i].descriptorType;
            if (w[i].pImageInfo) {
                e[i].a = (uint64_t)(uintptr_t)ii[i].sampler; e[i].b = (uint64_t)(uintptr_t)ii[i].imageView; e[i].c = (uint64_t)ii[i].imageLayout;
            } else {
                e[i].a = (uint64_t)(uintptr_t)bi[i].buffer; e[i].b = bi[i].offset; e[i].c = bi[i].range;
            }
        }
        uint64_t seq = vk_pending_seq();
        for (int k = 0; k < SET_WAYS; k++) {
            SetCache *q = &ways[k];
            if (q->set && q->seq == seq && q->view_gen == vk_view_gen && q->ds_gen == ds_gen && q->n == n &&
                !memcmp(q->e, e, (size_t)n * sizeof e[0])) {
                prof_sets_reused++;
                return q->set;
            }
        }
        ways[0].next++;
    }
    VkDescriptorSet ds = vk_alloc_set(layout);
    for (int i = 0; i < n; i++) w[i].dstSet = ds;
    if (n) rec_UpdateDescriptorSets((uint32_t)n, w);
    prof_sets_new++;
    if (c) {
        c->set = ds; c->seq = vk_pending_seq(); c->view_gen = vk_view_gen; c->ds_gen = ds_gen; c->n = n;
        memcpy(c->e, e, (size_t)n * sizeof e[0]);
    }
    return ds;
}

// ================================================================== compute
typedef struct { Prog p; ShaderInfo *si; VkPipeline pipe; uint32_t local[3]; uint64_t code; } CsProg;

// Compute programs run in 32-lane subgroups, as Maxwell's warps do. The game's compute shaders use ballot / shuffle / vote on a 32-lane warp (Pokémon Z-A's GPU culling
// counts the visible instances of a batch with ballot + a leader's atomicAdd); AMD drivers run compute in 64-lane waves by default (RADV), where those counts are wrong:
// whole parts of a scene (walls, arches, lamps, their shadows) were culled, and the translated shader could hang the GPU. Vulkan 1.3 requires the subgroup size control;
// where the device cannot run 32 lanes (min > 32) nothing changes. LARECOMP_VK_CS_SUBGROUP=0: the driver's choice.
static bool cs_subgroup32(void) {
    static int ok = -1;
    if (ok < 0) {
        const char *e = ENV("LARECOMP_VK_CS_SUBGROUP");
        ok = 0;
        if (g_vkc.api >= VK_API_VERSION_1_3 && !(e && !strcmp(e, "0"))) {
            VkPhysicalDeviceSubgroupSizeControlProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
            VkPhysicalDeviceProperties2 pp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
            vkGetPhysicalDeviceProperties2(g_vk.phys, &pp);
            ok = sgp.minSubgroupSize <= 32 && sgp.maxSubgroupSize >= 32 && (sgp.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
            LOG("[vk] compute subgroup sizes %u..%u: %s\n", sgp.minSubgroupSize, sgp.maxSubgroupSize, ok ? "compute programs run in 32 lanes" : "left to the driver");
        }
    }
    return ok;
}

static void build_compute(void *arg) {
    CsProg *cp = arg;
    uint64_t t0 = SDL_GetTicksNS();
    int bs = build_begin("pipeline cs", cp->code, 0);
    size_t n = 0;
    uint32_t *spv = vk_glsl_to_spirv(cp->si->msl, STAGE_COMPUTE, "cs", &n);
    if (!spv) { atomic_store(&cp->p.state, 2); build_end(bs); return; }
    cp->p.nbinds = vk_reflect(spv, n, cp->p.binds, MAX_BIND);
    VkShaderModule m = vk_module(spv, n);
    free(spv);
    if (!m) {
        LOG("[compute] cs %llx: shader module creation failed\n", (unsigned long long)cp->code);
        atomic_store(&cp->p.state, 2);
        build_end(bs);
        return;
    }
    cp->p.layout = make_layout(&cp->p, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0);
    VkSpecializationMapEntry me[3] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}};
    VkSpecializationInfo spec = {3, me, 12, cp->local};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req32 = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
                                                                 .requiredSubgroupSize = 32};
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = cs_subgroup32() ? &req32 : NULL,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main",
                                                .pSpecializationInfo = &spec},
                                      .layout = cp->p.layout};
    SLOW_FEEDBACK(ci);
    VkResult r = vkCreateComputePipelines(g_vk.dev, g_vk.pcache, 1, &ci, NULL, &cp->pipe);
    SLOW_FEEDBACK_HIT();
    atomic_fetch_add(&stat_compiles, 1);
    atomic_fetch_add(&builds_done, 1);
    atomic_store(&cp->p.state, r == VK_SUCCESS ? 1 : 2);
    build_end(bs);
    if (r != VK_SUCCESS) LOG("[compute] cs %llx: pipeline failed: %d\n", (unsigned long long)cp->code, r);
    slow_build(t0, "pipeline cs", cp->code, 0);
}

static struct CsEnt { uint64_t addr, fp; CsProg *cp; } g_cs_tab[256];  // compute programs by code address (code_fp)

static uint64_t cs_content_key(uint64_t code_hash, const uint32_t local[3]) {
    uint64_t h = code_hash;
    for (int i = 0; i < 3; i++) { h ^= local[i]; h *= 0x100000001B3ull; }
    return h;
}

// Scarlet / Violet: World detail (hle/npc_detail.c) for the models drawn in batches - trees, bushes, rocks. Those
// pick their LOD and whether they're drawn at all on the GPU, in one compute pass (gpu_instancing_culling): per
// instance, the distance to the camera (cb5[19]) minus its radius against the batch's draw distance and then its LOD
// distances (per-batch records in memory). That distance, divided by the factor, moves every step that much farther:
// the game's own record values stay. The pass is found by its translation (the one line computing that distance, in
// a shader reading cb5[19]); the factor is read when it's translated - from the next start. LARECOMP_WORLD_DETAIL.
static void world_detail_cs(ShaderInfo *si) {
    extern int rt_sv_world_part(const char *part);
    static const char pat[] = "r15 = as_u(((-as_f(r11)) + as_f(r4)));";
    int f = rt_sv_world_part("INSTANCED");
    char *at;
    if (f <= 1 || !si || !si->msl || !strstr(si->msl, "cb5[19][0]") || !(at = strstr(si->msl, pat))) return;
    char rep[96];
    snprintf(rep, sizeof rep, "r15 = as_u(((-as_f(r11)) + as_f(r4)) * %.6ff);", 1.0 / f);
    size_t n = strlen(si->msl), pl = strlen(pat), rl = strlen(rep), head = (size_t)(at - si->msl);
    char *s = malloc(n - pl + rl + 1);
    if (!s) return;
    memcpy(s, si->msl, head);
    memcpy(s + head, rep, rl);
    memcpy(s + head + rl, at + pl, n - head - pl + 1);
    free(si->msl);  // (maxwell_free frees the new one)
    si->msl = s;
    LOG("[world] instanced culling / LOD pass: distances x%d\n", f);
}

static bool translate_compute(CsProg *cp, const uint8_t *code, size_t max, uint64_t addr) {
    uint64_t wl, wn;
    gmem_window_get(&wl, &wn);
    ShaderKey k = {0};
    k.glsl = 1;
    pthread_mutex_lock(&translate_lock);  // (the globals too: the render thread translates as well - run_queued_now)
    maxwell_gmem_base = gwin_base + wl, maxwell_gmem_size = wn;
    maxwell_gmem_chunk_shift = gmem_chunk_shift();
    cp->si = maxwell_translate(code, max, STAGE_COMPUTE, &k);
    world_detail_cs(cp->si);
    if (ENV("LARECOMP_DUMP_GLSL")) {  // (debug) build/glsl/<code address>.comp
        mkdir("build/glsl", 0755);
        char path[96];
        snprintf(path, sizeof path, "build/glsl/%llx.comp", (unsigned long long)addr);
        FILE *f = fopen(path, "wb");
        if (f) { fputs(cp->si->msl ? cp->si->msl : cp->si->error, f); fclose(f); }
    }
    pthread_mutex_unlock(&translate_lock);
    cp->p.vi = cp->si;
    if (!cp->si->msl) {
        LOG("[compute] translate failed: %s (code %llx)\n", cp->si->error, (unsigned long long)addr);
        atomic_store(&cp->p.state, 2);
        return false;
    }
    return true;
}

// The table's entry for this code address: the one it had (other code there now: it gets the new program) or a new one
static void cs_tab_put(int at, uint64_t addr, uint64_t fp, CsProg *cp) {
    if (at < 0 && g_cs_n < 256) at = g_cs_n++;
    if (at >= 0) g_cs_tab[at] = (struct CsEnt){addr, fp, cp};
}

static CsProg *compute_program(NvnProgram *np) {
    NvnShaderStage *sst = &np->stages[5];
    if (!sst->code_addr) return NULL;
    struct CsEnt *tab = g_cs_tab;
    int n = g_cs_n, at = -1;
    uint64_t fp = code_fp(sst);
    for (int i = 0; i < n && at < 0; i++)
        if (tab[i].addr == sst->code_addr) {
            if (tab[i].fp == fp) return tab[i].cp;
            at = i;  // (other code there now: this entry gets the new program)
        }
    uint32_t local[3];
    for (int i = 0; i < 3; i++) {
        local[i] = sst->control ? *(const uint32_t *)(sst->control + 0x748 + 4 * i) : 1;
        if (!local[i]) local[i] = 1;
    }
    if (shader_rec_recording()) {
        ShaderRec rec = {.kind = 'C', .vs = shader_stage_hash(sst), .local = {local[0], local[1], local[2]}};
        shader_rec_note(&rec);
        shader_seen_code(rec.vs, sst);  // (its code for the next start's preparation: shader_prep.h)
    }
    uint64_t ch = shader_stage_hash(sst);
    CsProg *cp = ch && prep_cs_n ? PREP_GET(prep_cs, cs_content_key(ch, local)) : NULL;  // prepared before the game
    if (!cp && ch && sess_on()) cp = PREP_GET(sess_cs, cs_content_key(ch, local));  // (built this run)
    if (cp) {
        cs_tab_put(at, sst->code_addr, fp, cp);
        shader_fp_check(&cp->p.fp[0], sst, "cs");
        return cp;
    }
    shader_first_visit("compute", ch, 0, cs_content_key(ch, local));
    cp = calloc(1, sizeof *cp);
    shader_fp_check(&cp->p.fp[0], sst, "cs");  // (takes the fingerprint)
    cp->code = sst->code_addr;  // (for the build logs)
    note_code(sst, fp, "cs");
    cs_tab_put(at, sst->code_addr, fp, cp);
    if (ch && sess_on()) PREP_PUT(sess_cs, cs_content_key(ch, local), cp);
    memcpy(cp->local, local, sizeof local);
    if (!translate_compute(cp, (const uint8_t *)(uintptr_t)sst->code_addr, sst->const_offset ? sst->const_offset : 0x10000, sst->code_addr))
        return cp;
    save_glsl(cp->si->msl, sst->code_addr, "cs");
    run_job(build_compute, cp, &cp->p.state);
    return cp;
}

// ================================================================== shader preparation (shader_prep.h)
// Each record: translated here (one thread, the translator isn't re-entrant), compiled and built on the worker
// threads (run_job), kept in the prep_* tables for the game.
typedef struct { void (*fn)(void *); void *arg; } PrepJob;
static void prep_job_run(void *a) {
    PrepJob *j = a;
    j->fn(j->arg);
    free(j);
    prep_pending--;
}
static void prep_job(void (*fn)(void *), void *arg) {
    PrepJob *j = malloc(sizeof *j);
    j->fn = fn, j->arg = arg;
    prep_pending++;
    run_job(prep_job_run, j, NULL);
}

static _Atomic int prep_pipe_dups;  // records whose pipeline another record already made
// (Pokémon Z-A: only once the game has set up its physical memory window - the translations bake its base and size
// in, and before that it's Link's Awakening's heap: the start screen's preparation would have made every program with
// the wrong one. Z-A's runs at its start instead, shader_prep.c shader_prep_boot)
bool game_files_are_la(const char *game_dir);  // (runtime/game_addresses.c)
bool renderer_prepare_begin(void) { return g_vk.dev != VK_NULL_HANDLE && (gwin()->phys || game_files_are_la("extracted")); }
int renderer_prepare_pending(void) { return prep_pending; }
static void gpl_probe(void);
static void gpl_probe_note(Pipe *pp);
void renderer_prepare_end(void) {
    vk_pipeline_cache_save();  // (otherwise only kept at exit)
    static bool logged;        // (the counts: once per process, the background preparation runs once)
    if (!logged) {
        logged = true;
        pthread_mutex_lock(&prep_lock);
        size_t n = prep_pipes_n;
        pthread_mutex_unlock(&prep_lock);
        LOG("[vk] prepared %zu pipelines (%d records shared one, vertex input %s)\n", n, atomic_load(&prep_pipe_dups),
            g_vk.vids ? "dynamic" : "in each pipeline");
        if (g_vk.gpl_probe) gpl_probe();
    }
}

typedef struct { Pipe *pp; ShaderRec r; } PrepPipe;
static void prep_pipeline(void *arg) {
    PrepPipe *a = arg;
    Prog *pr = a->pp->k.prog;
    // its program's compile was queued before (the queue is first in, first out): wait for it to finish
    while (atomic_load(&pr->state) == 0) { struct timespec ts = {0, 1000000}; nanosleep(&ts, NULL); }
    if (atomic_load(&pr->state) != 1) { atomic_store(&a->pp->state, 2); free(a); return; }
    const ShaderRec *r = &a->r;
    // the pass's attachment formats as begin_pass makes them for these targets
    VkFormat color[8] = {0}, depth = VK_FORMAT_UNDEFINED, stencil = VK_FORMAT_UNDEFINED;
    uint32_t ncolor = r->ncolor;
    for (uint32_t i = 0; i < ncolor && i < 8; i++) {
        if (!r->color[i]) continue;
        VkFormat b = vk_image_format(r->color[i], NULL);
        color[i] = r->cview[i] && vk_format(r->color[i]) == b ? vk_format(r->cview[i]) : b;
    }
    while (ncolor && !r->color[ncolor - 1]) ncolor--;
    if (r->depth && renderer_format_is_depth(r->depth)) {
        VkImageAspectFlags asp;
        VkFormat b = vk_image_format(r->depth, &asp);
        if (asp & VK_IMAGE_ASPECT_DEPTH_BIT) depth = b;
        if (asp & VK_IMAGE_ASPECT_STENCIL_BIT) stencil = b;
    }
    const PipeIn in = {r->attrib, r->nattrib, r->stream, r->nstream, r->color, r->channel_mask, r->blend_on, r->blend,
                       r->a2c, r->topo, r->poly_fill};
    pipe_key_fill(&a->pp->k, pr, &in, color, ncolor, depth, stencil, r->samples ? r->samples : 1, NULL);
    uint64_t ph = hash_words(0xcbf29ce484222325ull, &a->pp->k, sizeof a->pp->k);
    if (!ph) ph = 1;
    // Another record makes the same pipeline (with dynamic vertex input: every record differing only in its vertex
    // layout): built once. Claimed before the build, so records on other worker threads at the same time see it too
    // (the game skips a draw whose prepared pipeline is still building or failed, as with one it builds itself).
    pthread_mutex_lock(&prep_lock);
    bool dup = prep_get(prep_pipes, prep_pipes_cap, ph) != NULL;
    if (!dup) prep_put(&prep_pipes, &prep_pipes_cap, &prep_pipes_n, ph, a->pp);
    pthread_mutex_unlock(&prep_lock);
    if (dup) { atomic_fetch_add(&prep_pipe_dups, 1); free(a->pp); free(a); return; }
    build_pipeline(a->pp);
    if (g_vk.gpl_probe) gpl_probe_note(a->pp);
    free(a);
}

bool renderer_prepare(const ShaderRec *r, const ShaderBlob *vsb, const ShaderBlob *fsb) {
    if (r->kind == 'C') {
        uint64_t ck = cs_content_key(r->vs, r->local);
        if (PREP_GET(prep_cs, ck)) return true;
        CsProg *cp = calloc(1, sizeof *cp);
        cp->code = r->vs;  // (prepared: its content hash, for the build logs)
        memcpy(cp->local, r->local, sizeof cp->local);
        if (!translate_compute(cp, vsb->code, vsb->const_offset ? vsb->const_offset : 0x10000, 0)) return false;
        PREP_PUT(prep_cs, ck, cp);
        prep_job(build_compute, cp);
        return true;
    }
    // the keys the draw path makes for this state (draw_)
    ShaderKey vkey = {0}, fkey = {0};
    vkey.glsl = fkey.glsl = 1;
    vkey.flip_y = fkey.flip_y = (r->flags & SR_FLIP_Y) != 0;
    vkey.depth_minus_one = (r->flags & SR_DEPTH_MINUS_ONE) != 0;
    vkey.flip_x = fkey.flip_x = (r->flags & SR_FLIP_X) != 0;
    for (int i = 0; i < r->nattrib; i++) vkey.attr_kind[i] = (uint8_t)attr_kind(r->attrib[i] & 0xFF);
    for (int i = 0; i < r->ncolor && i < 8; i++) if (r->color[i]) fkey.rt_kind[i] = (uint8_t)rt_kind(r->color[i]);
    uint64_t ck = prog_content_key(r->vs, r->fs, &vkey, &fkey);
    Prog *p = PREP_GET(prep_progs, ck);
    if (!p) {
        p = calloc(1, sizeof *p);
        p->code[0] = (uintptr_t)vsb->code;
        p->code[1] = (uintptr_t)fsb->code;
        p->cmax[0] = vsb->const_offset ? vsb->const_offset : 0x10000;
        p->cmax[1] = fsb->const_offset ? fsb->const_offset : 0x10000;
        p->vkey = vkey;
        p->fkey = fkey;
        p->prep_vh = r->vs, p->prep_fh = r->fs;
        p->sess_vh = r->vs, p->sess_fh = r->fs;  // (also found by the slots its shaders read: sess_put_masked - a
                                                 // record has no stencil / integer slots, the draws often do)
        bool ok = translate_program_now(p);
        p->code[0] = p->code[1] = 0;  // (the game files' copies are gone after this call)
        PREP_PUT(prep_progs, ck, p);  // (failed ones too: the game then skips them as it would its own)
        if (!ok) return false;
        prep_job(compile_program, p);
    }
    if (atomic_load(&p->state) == 2) return false;
    PrepPipe *a = calloc(1, sizeof *a);
    a->pp = calloc(1, sizeof *a->pp);
    a->pp->k.prog = p;
    a->r = *r;
    prep_job(prep_pipeline, a);
    return true;
}

// ================================================================== graphics pipeline library probe (measurement only)
// LARECOMP_VK_GPL_PROBE=1 (or =<N>): after shader preparation, the first N prepared pipelines (20; in the order the
// list's records built them) are built again three ways and timed, on the preparing thread, one at a time:
//   full:      one vkCreateGraphicsPipelines, as the game builds them;
//   libraries: the four VK_EXT_graphics_pipeline_library parts (vertex input, pre-rasterization with the vertex shader,
//              fragment shader, fragment output) plus a fast link (no link-time optimization);
//   optimized: the same parts linked again with link-time optimization (what would replace a fast-linked pipeline).
// Every build is cold: no pipeline cache, and a specialization constant no shader reads (a new value per build and
// run) makes each one new to the driver's own caches. The results are destroyed; only [gpl] log lines remain.
// Nothing else uses the extension: the device enables it only with this variable set (device.c).
#ifdef VK_EXT_graphics_pipeline_library
#define GPL_PROBE_MAX 256
static Pipe *gpl_sample[GPL_PROBE_MAX];
static int gpl_nsample;
static void gpl_probe_note(Pipe *pp) {
    if (atomic_load(&pp->state) != 1) return;
    pthread_mutex_lock(&prep_lock);
    if (gpl_nsample < GPL_PROBE_MAX) gpl_sample[gpl_nsample++] = pp;
    pthread_mutex_unlock(&prep_lock);
}

static double gpl_ms_since(uint64_t t0) { return (double)(SDL_GetTicksNS() - t0) / 1e6; }

// One pipeline: [0] full, [1] vertex input, [2] pre-rasterization, [3] fragment shader, [4] fragment output, [5] fast
// link, [6] optimized link (ms). false: a build failed (logged).
static bool gpl_probe_one(const PipeKey *k, uint32_t nonce, double t[7]) {
    PipeCI c;
    pipe_ci_fill(&c, k);
    VkSpecializationMapEntry me = {0x7fff, 0, 4};  // (no shader declares this id: it only makes the build unique)
    VkSpecializationInfo spec = {1, &me, 4, &nonce};
    c.stages[0].pSpecializationInfo = c.stages[1].pSpecializationInfo = &spec;
    VkPipeline full = VK_NULL_HANDLE, lib[4] = {0}, fast = VK_NULL_HANDLE, opt = VK_NULL_HANDLE;
    const char *fail = NULL;
    VkResult r;
    uint64_t t0 = SDL_GetTicksNS();
    r = vkCreateGraphicsPipelines(g_vk.dev, VK_NULL_HANDLE, 1, &c.pi, NULL, &full);
    t[0] = gpl_ms_since(t0);
    if (r != VK_SUCCESS) { fail = "full"; goto out; }
    nonce ^= 0x80000000u;  // (the libraries: new to the caches too, not the full pipeline's shaders again)
    static const VkGraphicsPipelineLibraryFlagsEXT part[4] = {
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    static const char *part_name[4] = {"vertex input", "pre-rasterization", "fragment shader", "fragment output"};
    for (int i = 0; i < 4; i++) {
        VkGraphicsPipelineLibraryCreateInfoEXT li = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                     .pNext = &c.ri, .flags = part[i]};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
            .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
            .pDynamicState = &c.dy, .layout = c.pi.layout};
        switch (i) {
        case 0: pi.pVertexInputState = c.pi.pVertexInputState; pi.pInputAssemblyState = &c.ia; break;
        case 1: pi.stageCount = 1; pi.pStages = &c.stages[0]; pi.pViewportState = &c.vs; pi.pRasterizationState = &c.rs; break;
        case 2: pi.stageCount = 1; pi.pStages = &c.stages[1]; pi.pDepthStencilState = &c.ds; pi.pMultisampleState = &c.ms; break;
        case 3: pi.pColorBlendState = &c.cb; pi.pMultisampleState = &c.ms; break;
        }
        t0 = SDL_GetTicksNS();
        r = vkCreateGraphicsPipelines(g_vk.dev, VK_NULL_HANDLE, 1, &pi, NULL, &lib[i]);
        t[1 + i] = gpl_ms_since(t0);
        if (r != VK_SUCCESS) { fail = part_name[i]; goto out; }
    }
    for (int o = 0; o < 2; o++) {
        VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4,
                                             .pLibraries = lib};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
            .flags = o ? VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT : 0, .layout = c.pi.layout};
        t0 = SDL_GetTicksNS();
        r = vkCreateGraphicsPipelines(g_vk.dev, VK_NULL_HANDLE, 1, &pi, NULL, o ? &opt : &fast);
        t[5 + o] = gpl_ms_since(t0);
        if (r != VK_SUCCESS) { fail = o ? "optimized link" : "fast link"; goto out; }
    }
out:
    if (fail) LOG("[gpl] probe: %s build failed (%d)\n", fail, r);
    VkPipeline all[7] = {full, lib[0], lib[1], lib[2], lib[3], fast, opt};
    for (int i = 0; i < 7; i++) if (all[i]) vkDestroyPipeline(g_vk.dev, all[i], NULL);
    return !fail;
}

static void gpl_probe(void) {
    const char *e = getenv("LARECOMP_VK_GPL_PROBE");
    int want = e && atoi(e) > 1 ? atoi(e) : 20;
    if (want > GPL_PROBE_MAX) want = GPL_PROBE_MAX;
    pthread_mutex_lock(&prep_lock);
    int n = gpl_nsample < want ? gpl_nsample : want;
    pthread_mutex_unlock(&prep_lock);
    if (!n) { LOG("[gpl] probe: no prepared pipelines to measure\n"); return; }
    LOG("[gpl] probe: building %d prepared pipelines three ways (cold, one at a time; vertex input %s)\n", n,
        g_vk.vids ? "dynamic" : "in each pipeline");
    double sum[7] = {0}, mx[7] = {0};
    int ok = 0;
    uint32_t nonce = (uint32_t)SDL_GetTicksNS() & 0x7fffffffu;
    uint64_t t0 = SDL_GetTicksNS();
    for (int i = 0; i < n; i++) {
        const PipeKey *k = &gpl_sample[i]->k;
        double t[7] = {0};
        if (!gpl_probe_one(k, nonce + (uint32_t)i, t)) continue;
        ok++;
        for (int j = 0; j < 7; j++) { sum[j] += t[j]; if (t[j] > mx[j]) mx[j] = t[j]; }
        LOG("[gpl] %2d vs/fs %016llx %016llx: full %.1f ms; libraries %.1f + %.1f + %.1f + %.1f ms, fast link %.2f ms; "
            "optimized link %.1f ms\n", i, (unsigned long long)k->prog->prep_vh, (unsigned long long)k->prog->prep_fh,
            t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
    }
    if (!ok) { LOG("[gpl] probe: every build failed\n"); return; }
    double libs = sum[1] + sum[2] + sum[3] + sum[4];
    LOG("[gpl] probe summary, %d pipelines (%.0f ms in all): full %.0f ms (avg %.1f, max %.1f); "
        "libraries %.0f ms (vertex input %.1f, pre-rasterization %.0f, fragment shader %.0f, fragment output %.1f) "
        "+ fast link %.1f ms (avg %.2f, max %.2f); optimized link %.0f ms (avg %.1f, max %.1f)\n",
        ok, gpl_ms_since(t0), sum[0], sum[0] / ok, mx[0], libs, sum[1], sum[2], sum[3], sum[4], sum[5], sum[5] / ok, mx[5],
        sum[6], sum[6] / ok, mx[6]);
}
#else
static void gpl_probe_note(Pipe *pp) { (void)pp; }
static void gpl_probe(void) {}
#endif

static void dispatch_compute(uint32_t gx, uint32_t gy, uint32_t gz) {
    snap_reason = "compute"; snapshots_invalidate(NULL);
    if (!st.program || !gx || !gy || !gz) return;
    CsProg *cp = compute_program(st.program);
    if (!cp || atomic_load(&cp->p.state) != 1) { SKIP_CS(!cp ? SK_NO_STAGES : atomic_load(&cp->p.state) == 2 ? SK_CS_FAILED : SK_CS_BUILDING); return; }
    {   // LARECOMP_SKIP_CS=<code address hex> (debug): skip that compute shader's dispatches, to see what it does
        static uint64_t skip = ~0ull;
        if (skip == ~0ull) { const char *e = getenv("LARECOMP_SKIP_CS"); skip = e ? strtoull(e, NULL, 16) : 0; }
        if (skip && st.program->stages[5].code_addr == skip) { SKIP_CS(SK_DEBUG_SKIP); return; }
    }
    if (cp->si->uses_gmem && ENV("LARECOMP_LOG_CS_SBUF")) {  // (debug) the storage buffers bound to gmem dispatches
        static uint64_t seen_cs[16];
        static int seen_n[16];
        int k = 0;
        while (k < 15 && seen_cs[k] && seen_cs[k] != st.program->stages[5].code_addr) k++;
        seen_cs[k] = st.program->stages[5].code_addr;
        if (seen_n[k]++ < 24) {
            LOG("[cs] %llx %ux%ux%u:", (unsigned long long)st.program->stages[5].code_addr, gx, gy, gz);
            for (int i = 0; i < 16; i++)
                if (st.sbuf[5][i].addr) LOG(" sb%d %llx+%llx", i, (unsigned long long)st.sbuf[5][i].addr, (unsigned long long)st.sbuf[5][i].size);
            LOG("\n");
        }
    }
    snap_apply(5, 0xFFFF);  // (overlap) every bound slot: the probe / copy decision per range, before gfast and the sets read the table
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    gp_mark(cb, "dispatch %ux%ux%u (local %ux%ux%u) cs %llx gmem %d", gx, gy, gz, cp->local[0], cp->local[1], cp->local[2],
            (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    dl_note("dispatch %ux%ux%u cs %llx gmem %d", gx, gy, gz, (unsigned long long)st.program->stages[5].code_addr, cp->si->uses_gmem);
    if (dl_on > 0) vk_checkpoint(cb, dl_n);
    nfeedback = 0;
    if (ct_on()) {
        LOG("[ct] dispatch cs %llx %ux%ux%u | tex", (unsigned long long)st.program->stages[5].code_addr, gx, gy, gz);
        for (int i = 0; i < cp->si->ntex; i++) {
            NvnTexture *t = shader_texture_object(5, &cp->si->tex[i]);
            if (t) LOG(" %s%p/%#x", cp->si->tex[i].is_image ? "I:" : "", (void *)t, t->format);
        }
        LOG("\n");
    }
    if (ENV("LARECOMP_LOG_CS_TEX")) {  // (debug) each compute shader's textures, once
        static uint64_t seen[64];
        static int nseen;
        uint64_t code = st.program->stages[5].code_addr;
        bool have = false;
        for (int i = 0; i < nseen; i++) have |= seen[i] == code;
        if (!have && nseen < 64) {
            seen[nseen++] = code;
            for (int i = 0; i < cp->si->ntex; i++) {
                NvnTexture *t = shader_texture_object(5, &cp->si->tex[i]);
                LOG("[cs-tex] %llx tex%d: %p %ux%u fmt %#x levels %u target %u addr %llx\n", (unsigned long long)code, i, (void *)t,
                    t ? t->width : 0, t ? t->height : 0, t ? t->format : 0, t ? t->levels : 0, t ? t->target : 0,
                    (unsigned long long)(t ? t->addr : 0));
            }
        }
    }
    gfast_stores = cp->si->uses_gmem_store;
    if (cp->si->uses_gmem_map) gfast_prepare(cb);
    VkDescriptorSet ds = stage_set(&cp->p, cp->p.set_layout[0], 0, 5, cp->si, NULL, NULL, gx, gy, gz, NULL, NULL, NULL);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->pipe);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->p.layout, 0, 1, &ds, 0, NULL);
    if (gc_shift && cp->si->uses_gmem) gmem_chunk_bind(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp->p.layout, GMEM_CHUNK_SET(STAGE_COMPUTE), 1u << STAGE_COMPUTE);
    rec_CmdDispatch(cb, gx, gy, gz);
    if (cp->si->uses_gmem_map && cp->si->uses_gmem_store) gfast_finish(cb);
    if (cp->si->uses_gmem_store) vc_mark_sbufs(5);
    stat_dispatches++;
}

// ================================================================== draws
static void debug_draw(Prog *pr, CmdRec *r, bool indexed);

// Quads / fans / polygons as a 32-bit triangle index list.
static uint32_t *expand_indices(uint32_t mode, uint32_t count, const void *indices, int itype, uint32_t first, uint32_t *out_count) {
    uint32_t tris = mode == 7 ? count / 4 * 2 : count >= 3 ? count - 2 : 0;
    uint32_t *out = malloc(tris * 3 * 4 + 4);
    uint32_t n = 0;
#define IDX(i) (indices ? (itype == 0 ? ((const uint8_t *)indices)[i] : itype == 1 ? ((const uint16_t *)indices)[i] : ((const uint32_t *)indices)[i]) : first + (i))
    if (mode == 7) {
        for (uint32_t q = 0; q + 3 < count; q += 4) {
            out[n++] = IDX(q); out[n++] = IDX(q + 1); out[n++] = IDX(q + 2);
            out[n++] = IDX(q); out[n++] = IDX(q + 2); out[n++] = IDX(q + 3);
        }
    } else {
        for (uint32_t i = 1; i + 1 < count; i++) { out[n++] = IDX(0); out[n++] = IDX(i); out[n++] = IDX(i + 1); }
    }
#undef IDX
    *out_count = n;
    return out;
}

static bool topology(uint32_t mode, VkPrimitiveTopology *t, uint8_t *cls) {
    switch (mode) {
    case 0: *t = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; *cls = 0; return true;
    case 1: *t = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; *cls = 1; return true;
    case 3: *t = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; *cls = 1; return true;
    case 4: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return true;
    case 5: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; *cls = 2; return true;
    default: *t = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; *cls = 2; return false;
    }
}

// Texture handle slots of a stage bound in NVN depth/stencil mode STENCIL or with an unsigned integer format
// (ShaderKey.stencil_slots: usamplers), and with a signed integer format (*sint: isamplers). Maxwell's fetches give
// those formats' bits; a float sampler on them is invalid in Vulkan (VUID-vkCmdDraw-format-07753) - NVIDIA returned the
// bits anyway, other drivers needn't.
// Each entry's answer (0 neither, 1 usampler, 2 isampler) is kept per texture pool entry for one submitted command list
// (memoized: every draw went through all 32 slots of its stages, reading each bound texture's NvnTexture and backend
// - two cache misses a slot, 5.5 % of the render thread at 60 fps. LARECOMP_VK_NO_SLOT_MEMO=1: computed each time.)
static int slot_kind(const NvnTexEntry *e) {
    if (!renderer_format_has_stencil(e->tex->format)) return vk_view_int_kind(e->tex, &e->view, e->has_view);
    return (e->has_view && (e->view.setmask & 16) ? e->view.ds_mode : e->tex->ds_mode) == 1;
}
static uint32_t stencil_slots(int stage, uint32_t *sint) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    *sint = 0;
    if (!st.tpool) return 0;
    static int nomemo = -1;
    if (nomemo < 0) nomemo = ENV("LARECOMP_VK_NO_SLOT_MEMO") != NULL;
    // (the pool's change generation: a texture's kind holds while its entry is unchanged - a memo per queue item
    // recomputed every texture's at its first use each frame, 2.5 % of the render thread)
    extern _Atomic uint32_t g_pool_gen;
    uint32_t gen = atomic_load_explicit(&g_pool_gen, memory_order_relaxed);
    static struct { const NvnDescPool *pool; uint32_t tic, gen; int kind; } memo_t[NCTX][16384];
    __typeof__(memo_t[0][0]) *memo = memo_t[sw_ctx()];
    // Only the slots whose binding changed since the stage's draw before (same pool, its entries unchanged): a draw changes
    // one or two textures if any, and going through all 32 each time - an entry and its memo line each - was 5 % of the
    // render thread in a town at World detail 4x (~11 000 calls a frame)
    static struct { uint64_t tex[32]; const NvnDescPool *pool; uint32_t gen, m, sint; } last_t[NCTX][2];
    __typeof__(last_t[0][0]) *last = last_t[sw_ctx()];
    int ls = stage ? 1 : 0;
    __typeof__(last[0]) *L = &last[ls];
    if (nomemo || L->pool != st.tpool || L->gen != gen) {
        memset(L->tex, 0xFF, sizeof L->tex);  // (no handle is all ones: every slot looked at below)
        L->pool = st.tpool, L->gen = gen, L->m = L->sint = 0;
    }
    for (int s = 0; s < 32; s++) {
        uint64_t h = st.tex[stage][s];
        if (h == L->tex[s]) continue;
        L->tex[s] = h;
        uint32_t tic = h & 0xFFFFF;
        int k = 0;
        if (h && tic < (uint32_t)st.tpool->count) {
            __typeof__(memo[0]) *c = &memo[tic & 16383];
            if (!nomemo && c->pool == st.tpool && c->tic == tic && c->gen == gen) k = c->kind;
            else {
                NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
                k = e->tex ? slot_kind(e) : 0;
                *c = (__typeof__(*c)){st.tpool, tic, gen, k};
            }
        }
        L->m = (L->m & ~(1u << s)) | (k == 1 ? 1u << s : 0);
        L->sint = (L->sint & ~(1u << s)) | (k == 2 ? 1u << s : 0);
    }
    *sint = L->sint;
    return L->m;
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}


// (investigation) the programs drawn with transform feedback on: their control sections to build/xfb_<vs code>.bin
static void xfb_note(void) {
    static uint64_t seen[64];
    static int nseen;
    if (!st.program || nseen >= 64) return;
    uint64_t vs = st.program->stages[0].code_addr;
    for (int i = 0; i < nseen; i++) if (seen[i] == vs) return;
    seen[nseen++] = vs;
    LOG("[xfb] program vs %llx fs %llx discard %d buffers %llx+%llx %llx+%llx %llx+%llx %llx+%llx\n", (unsigned long long)vs,
        (unsigned long long)st.program->stages[1].code_addr, raster_discard, (unsigned long long)xfb_addr[0],
        (unsigned long long)xfb_size[0], (unsigned long long)xfb_addr[1], (unsigned long long)xfb_size[1],
        (unsigned long long)xfb_addr[2], (unsigned long long)xfb_size[2], (unsigned long long)xfb_addr[3],
        (unsigned long long)xfb_size[3]);
    if (ENV("LARECOMP_DUMP_XFB") && st.program->stages[0].control) {
        char path[96];
        snprintf(path, sizeof path, "build/xfb_%llx.bin", (unsigned long long)vs);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(st.program->stages[0].control, 1, 0x800, f); fclose(f); }
    }
}

// The draw's state as a variant record (shader_prep.h; as renderer.m: both backends record the same lines)
static void rec_from_state(ShaderRec *r, bool flip_x, uint32_t mode) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    memset(r, 0, sizeof *r);
    r->kind = 'G';
    r->vs = shader_stage_hash(&st.program->stages[0]);
    r->fs = shader_stage_hash(&st.program->stages[1]);
    r->flags = (nvn_window_origin() == 0 ? SR_FLIP_Y : 0) | (nvn_depth_mode() == 0 ? SR_DEPTH_MINUS_ONE : 0) | (flip_x ? SR_FLIP_X : 0);
    r->nattrib = (uint8_t)(st.nattrib < 16 ? st.nattrib : 16);
    memcpy(r->attrib, st.attrib, sizeof r->attrib[0] * r->nattrib);
    r->nstream = (uint8_t)(st.nstream < 16 ? st.nstream : 16);
    memcpy(r->stream, st.stream, sizeof r->stream[0] * r->nstream);
    r->ncolor = (uint8_t)(st.rt.ncolor < 8 ? st.rt.ncolor : 8);
    r->samples = 1;
    for (int i = 0; i < r->ncolor; i++) {
        NvnTexture *t = st.rt.color[i];
        if (!t) continue;
        r->color[i] = t->format;
        if (st.rt.has_cview[i] && (st.rt.cview[i].setmask & 4)) r->cview[i] = st.rt.cview[i].format;
        if (t->samples > 1) r->samples = t->samples;
        r->channel_mask |= st.channel_mask & (0xFu << (i * 4));
        if (((st.color_state[0] >> i) & 1) && !rt_kind(t->format)) {
            r->blend_on |= (uint8_t)(1u << i);
            memcpy(r->blend[i], st.blend[i], 8);
        }
    }
    if (st.rt.depth) {
        r->depth = st.rt.depth->format;
        if (!r->ncolor && st.rt.depth->samples > 1) r->samples = st.rt.depth->samples;
    }
    const char *a2c_r_env = ENV("LARECOMP_A2C");
    r->a2c = a2c_r_env ? (atoi(a2c_r_env) != 0) : ((r->samples > 1) ? (st.ms[0xC] != 0) : 0);
    r->topo = shader_topo_class(mode);
    r->poly_fill = st.poly[2] != 1;
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// LARECOMP_PREP_CHECK=1 (diagnostics): a pipeline the game builds although its shaders were prepared - logs the
// fields its key differs in from the closest prepared pipeline of the same shaders (and the draw's variant record)
static bool prep_check_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("LARECOMP_PREP_CHECK") && atoi(getenv("LARECOMP_PREP_CHECK"));
    return v;
}

// Fields of two pipeline keys that differ: their number, and (out) "field: prepared / draw" for each
static int pk_diff(const PipeKey *a, const PipeKey *b, char *out, size_t n) {
    int d = 0;
    size_t o = 0;
    if (out && n) out[0] = 0;
#define PKD(cond, ...) do { if (cond) { d++; if (out && o < n) o += (size_t)snprintf(out + o, n - o, __VA_ARGS__); } } while (0)
    PKD(a->prog != b->prog, " program %p/%p (ready %d/%d);", (void *)a->prog, (void *)b->prog, a->prog ? (int)a->prog->state : -1,
        b->prog ? (int)b->prog->state : -1);
    PKD(a->ncolor != b->ncolor, " ncolor %u/%u;", a->ncolor, b->ncolor);
    for (int i = 0; i < 8; i++) PKD(a->color[i] != b->color[i], " color%d %d/%d;", i, a->color[i], b->color[i]);
    PKD(a->depth != b->depth, " depth %d/%d;", a->depth, b->depth);
    PKD(a->stencil != b->stencil, " stencil %d/%d;", a->stencil, b->stencil);
    PKD(a->samples != b->samples, " samples %u/%u;", a->samples, b->samples);
    PKD(a->attr_mask != b->attr_mask, " attr_mask %#x/%#x;", a->attr_mask, b->attr_mask);
    for (int i = 0; i < 32; i++) {
        PKD(a->attr_fmt[i] != b->attr_fmt[i], " attr%d fmt %u/%u;", i, a->attr_fmt[i], b->attr_fmt[i]);
        PKD(a->attr_off[i] != b->attr_off[i], " attr%d off %u/%u;", i, a->attr_off[i], b->attr_off[i]);
        PKD(a->attr_bind[i] != b->attr_bind[i], " attr%d bind %u/%u;", i, a->attr_bind[i], b->attr_bind[i]);
    }
    PKD(a->used_streams != b->used_streams, " streams %#x/%#x;", a->used_streams, b->used_streams);
    for (int s = 0; s < 17; s++) {
        PKD(a->stride[s] != b->stride[s], " stride%d %u/%u;", s, a->stride[s], b->stride[s]);
        PKD(a->divisor[s] != b->divisor[s], " divisor%d %u/%u;", s, a->divisor[s], b->divisor[s]);
    }
    for (int i = 0; i < 8; i++) {
        PKD(memcmp(a->blend[i], b->blend[i], 8), " blend%d differs;", i);
        PKD(a->write_mask[i] != b->write_mask[i], " write_mask%d %#x/%#x;", i, a->write_mask[i], b->write_mask[i]);
    }
    PKD(a->blend_on != b->blend_on, " blend_on %#x/%#x;", a->blend_on, b->blend_on);
    PKD(a->a2c != b->a2c, " a2c %u/%u;", a->a2c, b->a2c);
    PKD(a->topo_class != b->topo_class, " topo %u/%u;", a->topo_class, b->topo_class);
    PKD(a->poly_fill != b->poly_fill, " fill %u/%u;", a->poly_fill, b->poly_fill);
    if (!d && memcmp(a, b, sizeof *a)) {  // (no field differs: other bytes - padding)
        const uint8_t *x = (const uint8_t *)a, *y = (const uint8_t *)b;
        for (size_t i = 0; i < sizeof *a; i++) PKD(x[i] != y[i], " byte %zu %02x/%02x;", i, x[i], y[i]);
    }
#undef PKD
    return d;
}

static void prep_check(const PipeKey *k, uint64_t vh, uint64_t fh, const ShaderRec *rec) {
    char line[2048], diff[2048] = "";
    shader_rec_format(rec, line, sizeof line);
    int nprep = 0, best_d = 1 << 30, progs = 0;
    pthread_mutex_lock(&prep_lock);
    for (size_t i = 0; i < prep_progs_cap; i++) {
        Prog *q = prep_progs[i].obj;
        if (prep_progs[i].key && q && q->prep_vh == vh && q->prep_fh == fh) progs++;
    }
    const Pipe *best = NULL;
    for (size_t i = 0; i < prep_pipes_cap; i++) {
        const Pipe *p = prep_pipes[i].obj;
        if (!prep_pipes[i].key || !p || p->k.prog->prep_vh != vh || p->k.prog->prep_fh != fh) continue;
        nprep++;
        int d = pk_diff(&p->k, k, NULL, 0);
        if (d < best_d) best_d = d, best = p;
    }
    if (best) pk_diff(&best->k, k, diff, sizeof diff);
    pthread_mutex_unlock(&prep_lock);
    LOG("[prepcheck] %016llx %016llx: program %s (%d prepared with these shaders), %d prepared pipelines, closest "
        "differs (prepared/draw) in:%s\n[prepcheck]   draw: %s\n", (unsigned long long)vh, (unsigned long long)fh,
        k->prog->prep_vh ? "prepared" : "not prepared", progs, nprep, best ? diff : " (none)", line);
}

static void draw_(CmdRec *r, bool indexed);
static void gtao_before_lighting(Prog *pr);
static void gtao_mark(NvnTexture *target);
static bool gtao_is_lighting(Prog *pr);
#define GTAO_LIGHTING 0xfe18220311c32b2eull

// The draw's depth / stencil test and rasterizer state as Vulkan values (draw_ sets them as dynamic state, or they go
// into the pipeline key: FixedState)
static void fixed_state(FixedState *f, bool flip_x, bool flip_y, bool native, VkPrimitiveTopology topo) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    memset(f, 0, sizeof *f);
    static int nodt = -1;
    if (nodt < 0) nodt = ENV("LARECOMP_NO_DEPTH_TEST") != NULL;
    f->dt = st.ds[0] && !nodt && pass_depth;
    f->dw = f->dt && st.ds[1];
    VkCompareOp dop = f->dt ? compare_fn(st.ds[2]) : VK_COMPARE_OP_ALWAYS;
    // EQUAL against a depth pre-pass written by a different vertex shader (foliage, trees) needs both shaders to land
    // on identical depths. They do since the translator stopped using unpackHalf2x16 (NVIDIA ran the half math after
    // it in fp16, differently per shader: black slashes in foliage). LARECOMP_VK_EQUAL_TOL=1 brings back the old
    // workaround: LESS_OR_EQUAL with a few ulps of bias towards the camera, which also let walls paint over the
    // plants stuck flat on them (beach) - so it is off.
    static int tol = -1;
    if (tol < 0) tol = ENV("LARECOMP_VK_EQUAL_TOL") != NULL;
    f->equal_tol = dop == VK_COMPARE_OP_EQUAL && tol;
    if (f->equal_tol) dop = VK_COMPARE_OP_LESS_OR_EQUAL;
    f->dop = (uint8_t)dop;
    f->sten = st.ds[3] && pass_stencil;
    uint32_t ops = st.ds[5] | st.ds[6] << 8 | st.ds[7] << 16;
    for (int s = 0; s < 2; s++) {
        uint32_t o = s ? ops >> 12 : ops;
        if (!f->sten) { f->sop[s][0] = f->sop[s][1] = f->sop[s][2] = VK_STENCIL_OP_KEEP; f->sop[s][3] = VK_COMPARE_OP_ALWAYS; continue; }
        f->sop[s][0] = (uint8_t)stencil_op(o & 0xF), f->sop[s][1] = (uint8_t)stencil_op((o >> 8) & 0xF);
        f->sop[s][2] = (uint8_t)stencil_op((o >> 4) & 0xF), f->sop[s][3] = (uint8_t)compare_fn(s ? st.ds[4] >> 4 : st.ds[4] & 0xF);
    }
    uint8_t cull = flip_x ? 0 : st.poly[0];  // (mirror mode's composite: one full-screen quad, either winding)
    f->cull = cull == 1 ? VK_CULL_MODE_FRONT_BIT : cull == 2 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    static int flipw = -1;  // LARECOMP_VK_FLIP_WINDING: invert the front-face mapping (A/B against Metal)
    if (flipw < 0) flipw = ENV("LARECOMP_VK_FLIP_WINDING") != NULL;
    bool front_ccw = ((st.poly[1] == 1) != flip_y) != (flipw != 0);
    f->ff = front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    f->topo = (uint8_t)(native ? topo : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// A draw of a handful of vertices - a full-screen pass: post-processing, a menu's backdrop, the composite into the
// window image - whose shader or pipeline is still being built is waited for (a few ms, once) rather than skipped:
// skipping one of those left the whole frame black (Pokémon Z-A: a black flash as a menu opened or closed). Not an
// effect's (blended() / scene_quad(), below): it waits as a scene draw.
// Scene draws wait a little too: a pipeline already in the caches (cache/vk_pipelines.bin, the driver's, the SPIR-V
// one) is built in 1-2 ms, but skipping the draw took the object off the screen for a whole frame - every time a
// scene came into view, and at different places each run, as the builds' timing varies (trees, buildings, tables
// blinking or going black for a frame in a recorded play). Up to 4 ms a draw, 6 ms a frame
// (LARECOMP_VK_SCENE_WAIT_MS, 0: skip at once); past that it's skipped as before - shaders never built before (a
// moment's pop-in the first time instead of a hitch).
// One that isn't built within 2 s isn't waited for again (it was 2 s per frame: a black screen at 0.5 fps on a
// player's PC as a battle began), and the log says what the shader threads were doing. A full-screen draw runs its
// own build if it's still queued (run_queued_now): it waits for that build, not for every one queued before it.
static bool wait_scene(_Atomic int *state) {
    if (atomic_load(state) == 0) queued_job(state, false);  // (its build next: skipped for a frame or two, not until
                                                             // the whole queue is done; 2: failed, nothing queued)
    static double budget_ms = -1;
    if (budget_ms < 0) budget_ms = ENV("LARECOMP_VK_SCENE_WAIT_MS") ? atof(ENV("LARECOMP_VK_SCENE_WAIT_MS")) : 6;
    static uint64_t frame = ~0ull, spent_ns;
    uint64_t f = renderer_present_count();
    if (f != frame) frame = f, spent_ns = 0;
    uint64_t left = (uint64_t)(budget_ms * 1e6) > spent_ns ? (uint64_t)(budget_ms * 1e6) - spent_ns : 0;
    if (left > 4000000ull) left = 4000000ull;
    if (!left) return false;
    uint64_t t0 = SDL_GetTicksNS(), now = t0;
    while (atomic_load(state) == 0 && (now = SDL_GetTicksNS()) - t0 < left) SDL_DelayPrecise(100000);
    spent_ns += now - t0;
    stat_scene_wait_ns += now - t0;
    return atomic_load(state) == 1;
}
// Not a full-screen pass, though a handful of vertices: Z-A's move effects are such quads; waited for like a full-screen
// pass, each new one stopped the frame for its build, one after another (a battle: 200-1200 ms frames, the other
// builds' threads idle). Skipped, an effect is missing for a frame, never a black frame:
// - blended into every colour target: drawn over what is there (not with ONE / ZERO: that blend is a plain write);
// - a quad (4-6 vertices) tested against the scene's depth: in the world, behind what is in front of it (in a recorded
//   battle most of the wait: sparks, decals, masks; the full-screen ones are triangles of 3, or not depth tested).
static bool scene_quad(const CmdRec *r) {
    return r->x[2] > 3 && st.ds[0] && st.rt.depth && compare_fn(st.ds[2]) != VK_COMPARE_OP_ALWAYS;
}
static bool blended(void) {
    int n = 0;
    for (int i = 0; i < st.rt.ncolor; i++) {
        if (!st.rt.color[i]) continue;
        if (!((st.color_state[0] >> i) & 1) || rt_kind(st.rt.color[i]->format) ||
            blend_factor(st.blend[i][2]) == VK_BLEND_FACTOR_ZERO) return false;
        n++;
    }
    return n > 0;
}
static bool wait_built_(_Atomic int *state, const CmdRec *r, bool indexed, bool pipe);
static bool wait_built(_Atomic int *state, const CmdRec *r, bool indexed, bool pipe) {
    if (atomic_load(state) == 1) return true;
    SWL_AT(7070);  // (shadow workers: the waits' tables)
    bool ok = wait_built_(state, r, indexed, pipe);
    SWU();
    return ok;
}
static bool wait_built_(_Atomic int *state, const CmdRec *r, bool indexed, bool pipe) {
    static int wait_all = -1;  // LARECOMP_VK_WAIT_ALL_SMALL=1: every draw of up to 6 vertices waits, as before
    if (wait_all < 0) wait_all = ENV("LARECOMP_VK_WAIT_ALL_SMALL") != NULL;
    if (r->x[2] > 6 || (!wait_all && (blended() || scene_quad(r)))) return wait_scene(state);
    static _Atomic int *gave_up[64];
    static int ngave_up;
    for (int i = 0; i < ngave_up; i++) if (gave_up[i] == state) return false;
    uint64_t t0 = SDL_GetTicksNS();
    while (atomic_load(state) == 0 && SDL_GetTicksNS() - t0 < 2000000000ull)
        if (!run_queued_now(state)) SDL_Delay(1);  // (a program: its translation, then its compile)
    uint64_t now = SDL_GetTicksNS();
    static int n;  // (from 5 ms: the first 20 of any length were all at startup, under 2 ms)
    if (now - t0 >= 5000000ull && n++ < 60) {  // (what the draw is: which draws are worth a wait)
        const NvnTexture *t = st.rt.ncolor ? st.rt.color[0] : NULL;
        LOG("[vk] full-screen draw waited %.1f ms for its %s (%llu vertices x%u, blend %#x, depth test %d, %d targets "
            "%ux%u fmt %#x, viewport %.0fx%.0f, vs %llx fs %llx)\n", (now - t0) / 1e6, pipe ? "pipeline" : "shaders",
            (unsigned long long)r->x[2], (uint32_t)r->x[indexed ? 6 : 4], st.color_state[0], st.ds[0] && st.rt.depth,
            st.rt.ncolor, t ? t->width : 0, t ? t->height : 0, t ? t->format : 0, st.viewport[0][2], st.viewport[0][3],
            (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr);
    }
    if (atomic_load(state) == 0) {
        if (ngave_up < 64) gave_up[ngave_up++] = state;
        uint64_t tt = atomic_load(&translator_t0), longest = 0;
        int busy = 0;
        for (int i = 0; i < nworkers; i++) {
            uint64_t w = atomic_load(&worker_t0[i]);
            if (w) busy++, longest = now - w > longest ? now - w : longest;
        }
        LOG("[vk] full-screen draw (vs %llx fs %llx): its shader / pipeline isn't built after 2 s - not waited for again; "
            "translator %s %.1f s, %d of %d compile threads busy (longest %.1f s)\n",
            st.program ? (unsigned long long)st.program->stages[0].code_addr : 0ull,
            st.program ? (unsigned long long)st.program->stages[1].code_addr : 0ull, tt ? "busy for" : "idle",
            tt ? (now - tt) / 1e9 : 0.0, busy, nworkers, longest / 1e9);
    }
    return atomic_load(state) == 1;
}
// ================================================================== issue #22: Terastallized Water moves (NVIDIA)
// What: a Terastallized Water-type move's effect (Kyogre's Water Spout) hung the GPU on NVIDIA (RTX 5070 Ti, driver
// 620.73.04, Linux: "GPU did not finish submission N within 10 s", kernel Xid 109 CTX SWITCH TIMEOUT) on the effect's
// first frame. The hang needs one draw, "E": a soft-particle billboard (vertex shader 09c1280fee686904, fragment shader
// 55b596e702185e46; a 4-vertex strip, 1-5 instances, into a 960x540 pass). Left out, nothing hangs (H); drawn with the
// vertex shader of the pass's draw before it, "D" (2aadd88fa6502c59: the same fragment shader, the same interface,
// inputs and compiler constants - on the CPU, D's code gives E's own outputs from E's inputs), nothing hangs either
// (K3, K4, and K5: E otherwise exactly as the game draws it - its vkCmdDraw, parameters, scissor and fragment shader -
// 219 of 219 draws of E, no timeout, no Xid). Why E's own compiled vertex shader hangs the GPU is not known.
// So a draw of E is drawn with D's vertex stage (its code, compiler constants and control section) and everything else
// E's: fragment shader, vertex streams, constant buffers, textures, keys, fixed-function state, draw command and
// parameters - nothing else is touched. Its program is the one D draws with (the same code addresses and keys:
// get_program finds D's), and so its pipeline (the same key) is D's: E's pipeline with D's vertex shader module, as in
// the runs that validated it - the pipeline identity is part of the workaround.
// D's vertex stage: a copy of the NvnShaderStage from D's last draw (the pass draws D before E) - plain data: the
// code's and the control section's guest addresses and the constants' place, nothing of the game's NvnProgram kept -
// its code hashed again at every use: the game loads a battle's shaders into new memory each time, and a copy whose
// code is no longer D's is never used. Without a valid D (none drawn yet, its code gone, or no vertex + fragment
// stage) E is drawn as the game asks, with a [vk] issue #22 line: never with another shader. The substitute's
// program not built yet: the draw is skipped, as any draw whose program is building. Identified by code hashes
// (shader_stage_hash: the code up to the end of its constants), not addresses (other addresses each battle), and a
// vertex + fragment stage: other stage bits are let through - st.program is the bound stages' composite (st.comp,
// CMD_BIND_PROGRAM), which keeps e.g. a compute program's stage from an earlier bind (stage_mask 0x23 on graphics
// draws after a dispatch), and a draw is built from its vertex + fragment stages only. (v0.0.30's first build asked
// for exactly vertex + fragment: in game both draws of E fell back on it - D's, in the same pass, most likely too -
// were drawn with E's own shader, and the GPU stopped.)
// The substitute NvnProgram is st.program for the rest of this one draw only: a per-thread copy (shadow workers
// translate draws in parallel), and draw's caller puts the game's program back after every draw.
// LARECOMP_ISSUE22_FIX=0: off - E drawn with its own vertex shader again (to reproduce the hang). The investigation and
// its runs: docs/issue-22-gpu-hang.md.
#define I22FIX_E_VH 0x09c1280fee686904ull
#define I22FIX_E_FH 0x55b596e702185e46ull
#define I22FIX_D_VH 0x2aadd88fa6502c59ull
#define I22FIX_D_FH 0x55b596e702185e46ull
enum { I22FIX_NONE, I22FIX_IS_D, I22FIX_SUB, I22FIX_NO_D, I22FIX_D_GONE, I22FIX_STAGES };
// The decision (tools/test_issue22_fix.py), by the drawn program's stage hashes (vh / fh, 1: not hashable) and stage
// mask: I22FIX_IS_D = D's draw, its vertex stage to keep; I22FIX_SUB = E, drawn with the kept stage (have_d; d_vh: that
// stage's code hashed now); I22FIX_NO_D / _D_GONE / _STAGES = E, drawn as it is (no D kept yet / the kept stage's code
// is no longer D's / no vertex or no fragment stage); I22FIX_NONE = anything else. Stage bits past vertex + fragment
// (a composite's leftover compute, geometry or tessellation stage) don't count: draws build only those two.
static int i22fix_decide(uint64_t vh, uint64_t fh, uint32_t stage_mask, bool have_d, uint64_t d_vh) {
    bool e = vh == I22FIX_E_VH && fh == I22FIX_E_FH, d = vh == I22FIX_D_VH && fh == I22FIX_D_FH;
    if (!e && !d) return I22FIX_NONE;
    if ((stage_mask & 3) != 3) return e ? I22FIX_STAGES : I22FIX_NONE;  // (a vertex and a fragment stage; others ignored)
    if (d) return I22FIX_IS_D;
    if (!have_d) return I22FIX_NO_D;
    if (d_vh != I22FIX_D_VH) return I22FIX_D_GONE;
    return I22FIX_SUB;
}
// The substitute: E's program with D's vertex stage - E's fragment stage, stage mask and the rest as they are.
static void i22fix_make(NvnProgram *out, const NvnProgram *e, const NvnShaderStage *d_vs) {
    *out = *e;
    out->stages[0] = *d_vs;
}
static const char *i22fix_why(int k) {
    return k == I22FIX_NO_D ? "no draw of D (vs 2aadd88fa6502c59) seen yet this session"
         : k == I22FIX_D_GONE ? "the vertex stage kept from D's draw no longer holds D's code"
         : k == I22FIX_STAGES ? "the program has no vertex stage or no fragment stage"
                              : "?";
}
// ---- end of the issue #22 decision

static pthread_once_t i22fix_once = PTHREAD_ONCE_INIT;
static bool i22fix_enabled;
static void i22fix_init(void) {
    const char *e = getenv("LARECOMP_ISSUE22_FIX");
    i22fix_enabled = !e || strcmp(e, "0") != 0;
    if (!i22fix_enabled) LOG("[vk] issue #22 workaround off (LARECOMP_ISSUE22_FIX=0): E drawn with its own vertex shader\n");
}
static bool i22fix_on(void) {
    pthread_once(&i22fix_once, i22fix_init);
    return i22fix_enabled;
}
// A program's stage hashes, once (its first draw; the same values from any thread: release / acquire on [0]).
static void prog_code_hashes(Prog *p, const NvnProgram *np, uint64_t *vh, uint64_t *fh) {
    uint64_t v = atomic_load_explicit(&p->code_h[0], memory_order_acquire);
    if (v) {
        *vh = v;
        *fh = atomic_load_explicit(&p->code_h[1], memory_order_relaxed);
        return;
    }
    v = shader_stage_hash(&np->stages[0]);
    uint64_t f = shader_stage_hash(&np->stages[1]);
    if (!v) v = 1;  // (no hashable code: never E or D - and not hashed again)
    atomic_store_explicit(&p->code_h[1], f, memory_order_relaxed);
    atomic_store_explicit(&p->code_h[0], v, memory_order_release);
    *vh = v, *fh = f;
}
static pthread_mutex_t i22fix_lock = PTHREAD_MUTEX_INITIALIZER;
static NvnShaderStage i22fix_d;  // (i22fix_lock) D's vertex stage, from its last draw: a copy
static bool i22fix_have_d;       // (i22fix_lock)
static uint64_t i22fix_d_frame;  // (i22fix_lock)
static _Atomic uint64_t i22fix_subs, i22fix_kept;  // draws of E with D's vertex shader / as the game asks
static __thread NvnProgram i22fix_prog;  // (this thread's draw of E: st.program until draw's caller puts the game's back)
// (draw_, once the drawn program is found and built) D: its vertex stage kept. E: *prp and st.program become E's
// program with D's vertex stage. Returns -1 (draw on), or the SKIP reason when the substitute's program is not built.
// Anything else, or off: -1 at once, nothing done.
static int i22fix_apply(Prog **prp, const CmdRec *r, bool indexed, ShaderKey *vkey, ShaderKey *fkey) {
    if (!i22fix_on()) return -1;
    uint64_t vh, fh;
    prog_code_hashes(*prp, st.program, &vh, &fh);
    if (vh != I22FIX_E_VH && vh != I22FIX_D_VH) return -1;  // (every other draw: one compare)
    NvnShaderStage d;
    pthread_mutex_lock(&i22fix_lock);
    d = i22fix_d;
    bool have_d = i22fix_have_d;
    uint64_t d_frame = i22fix_d_frame;
    pthread_mutex_unlock(&i22fix_lock);
    bool is_e = vh == I22FIX_E_VH && fh == I22FIX_E_FH;
    uint64_t d_vh = is_e && have_d ? shader_stage_hash(&d) : 0;  // (D's code must still be there)
    int k = i22fix_decide(vh, fh, st.program->stage_mask, have_d, d_vh);
    if (k == I22FIX_NONE) return -1;
    if (k == I22FIX_IS_D) {
        pthread_mutex_lock(&i22fix_lock);
        bool first = !i22fix_have_d;
        i22fix_d = st.program->stages[0];
        i22fix_have_d = true;
        i22fix_d_frame = vk_frame_counter();
        pthread_mutex_unlock(&i22fix_lock);
        if (first)  // (once a session: D's stage is kept again at each of its draws, silently)
            LOG("[vk] issue #22: D's vertex shader (vs %016llx at %llx, fs %016llx, stage mask %#x) kept for draws of E, "
                "frame %llu\n", (unsigned long long)vh, (unsigned long long)st.program->stages[0].code_addr,
                (unsigned long long)fh, st.program->stage_mask, (unsigned long long)vk_frame_counter());
        return -1;
    }
    if (k != I22FIX_SUB) {
        uint64_t n = atomic_fetch_add(&i22fix_kept, 1) + 1;
        if (n <= 20 || n % 1000 == 0)
            LOG("[vk] issue #22: a draw of E (vs %016llx fs %016llx) drawn with its own vertex shader: %s (stage mask "
                "%#x, %llu so far, frame %llu) - the GPU may stop on it (NVIDIA)\n", (unsigned long long)vh,
                (unsigned long long)fh, i22fix_why(k), st.program->stage_mask, (unsigned long long)n,
                (unsigned long long)vk_frame_counter());
        return -1;
    }
    i22fix_make(&i22fix_prog, st.program, &d);
    Prog *dp = get_program(&i22fix_prog, vkey, fkey);
    if (!dp) {  // (not possible with a vertex + fragment program: drawn as the game asks)
        atomic_fetch_add(&i22fix_kept, 1);
        LOG("[vk] issue #22: a draw of E drawn with its own vertex shader: no program for D's vertex shader\n");
        return -1;
    }
    if (!wait_built(&dp->state, r, indexed, false)) return atomic_load(&dp->state) == 2 ? SK_PROG_FAILED : SK_PROG_BUILDING;
    *prp = dp;
    st.program = &i22fix_prog;
    uint64_t n = atomic_fetch_add(&i22fix_subs, 1) + 1;
    if (n == 1 || n == 10 || n == 100 || n == 1000 || n % 10000 == 0)
        LOG("[vk] issue #22 workaround: %llu draw(s) of E (vs %016llx fs %016llx) drawn with D's vertex shader (vs "
            "%016llx at %llx, kept from D's draw at frame %llu), E's fragment shader, inputs, state and draw (%llu drawn "
            "with their own; LARECOMP_ISSUE22_FIX=0: off), frame %llu\n", (unsigned long long)n, (unsigned long long)vh,
            (unsigned long long)fh, (unsigned long long)d_vh, (unsigned long long)d.code_addr, (unsigned long long)d_frame,
            (unsigned long long)atomic_load(&i22fix_kept), (unsigned long long)vk_frame_counter());
    return -1;
}
// ---- end of issue #22
// A draw into a reshaped pass centred to `src_aspect` (pillarboxed on wider targets, letterboxed on taller ones).
static void vp_box_aspect(float src_aspect) {
    if (!st.rt.color[0] || !(src_aspect > 0.0f) || !isfinite(src_aspect)) return;
    uint32_t gw = st.rt.color[0]->width, gh = st.rt.color[0]->height;
    int pw, ph;
    pass_size(&pw, &ph);
    if (pw <= 0 || ph <= 0 || !gw || !gh) return;
    float pass_aspect = (float)pw / (float)ph;
    float box_w, box_h;
    if (pass_aspect > src_aspect) {
        box_h = (float)ph;
        box_w = box_h * src_aspect;
        vp_ox = ((float)pw - box_w) * 0.5f;
        vp_oy = 0.0f;
    } else {
        box_w = (float)pw;
        box_h = box_w / src_aspect;
        vp_ox = 0.0f;
        vp_oy = ((float)ph - box_h) * 0.5f;
    }
    vp_sx = box_w / (float)gw;
    vp_sy = box_h / (float)gh;
}
// Clear only the unused sidebar/letterbox strips around a boxed movie draw so reused targets or prior draws show black.
static void clear_movie_bars(VkCommandBuffer cb) {
    if (!pass_ncolor || pass_color[0] == VK_FORMAT_UNDEFINED) return;
    int pw, ph;
    pass_size(&pw, &ph);
    if (pw <= 0 || ph <= 0) return;
    VkClearRect rects[2];
    uint32_t n = 0;
    if (vp_ox > 0.5f) {
        uint32_t bw = (uint32_t)floorf(vp_ox);
        if (bw > (uint32_t)pw / 2) bw = (uint32_t)pw / 2;
        if (bw > pass_cur()->movie_bar_w) {
            pass_cur()->movie_bar_w = bw;
            rects[n++] = (VkClearRect){{{0, 0}, {bw, (uint32_t)ph}}, 0, pass_layers};
            rects[n++] = (VkClearRect){{{pw - (int32_t)bw, 0}, {bw, (uint32_t)ph}}, 0, pass_layers};
        }
    } else if (vp_oy > 0.5f) {
        uint32_t bh = (uint32_t)floorf(vp_oy);
        if (bh > (uint32_t)ph / 2) bh = (uint32_t)ph / 2;
        if (bh > pass_cur()->movie_bar_h) {
            pass_cur()->movie_bar_h = bh;
            rects[n++] = (VkClearRect){{{0, 0}, {(uint32_t)pw, bh}}, 0, pass_layers};
            rects[n++] = (VkClearRect){{{0, ph - (int32_t)bh}, {(uint32_t)pw, bh}}, 0, pass_layers};
        }
    }
    if (!n) return;
    VkClearAttachment ca = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
                            .clearValue.color.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
    rec_CmdClearAttachments(cb, 1, &ca, n, rects);
}
static bool movie_box_enabled(void) {
    static int on = -1;
    if (on < 0) { const char *e = ENV("LARECOMP_WIDE_MOVIE_BOX"); on = !e || atoi(e) != 0; }
    return on != 0;
}
// Match CPU-fallback video planes (exact, hardware-aligned luma/RGBA, or half-res chroma plane).
static bool movie_cpu_tex_match(uint32_t w, uint32_t h, uint32_t cpu_mw, uint32_t cpu_mh) {
    if (!w || !h || !cpu_mw || !cpu_mh) return false;
    uint32_t aw = (cpu_mw + 255u) & ~255u, ah = (cpu_mh + 15u) & ~15u;
    if (w >= cpu_mw && w <= aw && h >= cpu_mh && h <= ah) return true;
    uint32_t cw = cpu_mw / 2, ch = cpu_mh / 2;
    if (cw && ch) {
        uint32_t acw = (cw + 255u) & ~255u, ach = (ch + 15u) & ~15u;
        if (w >= cw && w <= acw && h >= ch && h <= ach) return true;
    }
    return false;
}
// Box a viewport/rect (vx, vy, vw, vh) to `movie_w / movie_h` when its aspect differs (e.g. direct swapchain present).
static void movie_box_rect(uint32_t movie_w, uint32_t movie_h, float *vx, float *vy, float *vw, float *vh) {
    if (!movie_w || !movie_h || !(*vw > 0.0f) || !(*vh > 0.0f)) return;
    float movie_aspect = (float)movie_w / (float)movie_h;
    float dst_aspect = *vw / *vh;
    if (fabsf(dst_aspect - movie_aspect) <= 0.01f * movie_aspect) return;
    if (dst_aspect > movie_aspect) {
        float bw = *vh * movie_aspect;
        *vx += (*vw - bw) * 0.5f;
        *vw = bw;
    } else {
        float bh = *vw / movie_aspect;
        *vy += (*vh - bh) * 0.5f;
        *vh = bh;
    }
}
// Identify pre-rendered video draws, apply aspect-ratio boxing on reshaped passes, and propagate metadata only for clean
// fullscreen 1:1 passes (never mixing video with gameplay/UI). Returns true when the draw is a movie draw in a reshaped pass.
static bool movie_draw_apply(VkCommandBuffer cb, TexBackend *rt0, uint32_t ncolor, int ntex, TexBackend *const *tbs,
                             const uint32_t (*wh)[2], bool video_draw_ok, bool reshaped) {
    if (!movie_box_enabled() || !ncolor) return false;
    bool found = false, src_boxed = false;
    uint32_t mw = 0, mh = 0;
    for (int i = 0; i < ntex && i < 64; i++) {
        const TexBackend *b = tbs[i];
        if (b && b->is_movie && b->movie_w > 0 && b->movie_h > 0) {
            found = true; mw = b->movie_w; mh = b->movie_h; src_boxed = b->movie_boxed;
            break;
        }
    }
    if (!found && video_draw_ok) {
        extern bool movie_cpu_video_active(uint32_t *w, uint32_t *h);
        uint32_t cpu_mw = 0, cpu_mh = 0;
        if (movie_cpu_video_active(&cpu_mw, &cpu_mh)) {
            for (int i = 0; i < ntex && i < 64; i++) {
                if (movie_cpu_tex_match(wh[i][0], wh[i][1], cpu_mw, cpu_mh)) {
                    found = true; mw = cpu_mw; mh = cpu_mh; src_boxed = false;
                    break;
                }
            }
        }
    }
    if (!found) {
        pass_cur()->has_non_movie_draw = true;
        if (rt0 && rt0->is_movie && !rt0->movie_boxed) {
            rt0->is_movie = false; rt0->movie_boxed = false; rt0->movie_w = rt0->movie_h = 0;
        }
        return false;
    }
    if (reshaped) {
        if (!src_boxed) {
            vp_box_aspect((float)mw / (float)mh);
            clear_movie_bars(cb);
        }
        if (rt0 && (!pass_cur()->has_non_movie_draw || !src_boxed)) {
            rt0->is_movie = true; rt0->movie_boxed = true; rt0->movie_w = mw; rt0->movie_h = mh;
        }
        return true;
    }
    if (rt0 && ncolor == 1 && video_draw_ok && !pass_cur()->has_non_movie_draw) {
        rt0->is_movie = true; rt0->movie_boxed = src_boxed; rt0->movie_w = mw; rt0->movie_h = mh;
    }
    return false;
}
// True only for whole-content base-level/layer copies in copy_texture_gpu (partial/mip/layer copies never propagate movie state).
static bool movie_copy_is_full(uint32_t slv, uint32_t dlv, uint32_t sly, uint32_t dly,
                               int sx0, int sy0, int dx0, int dy0,
                               int sw, int sh, int dw, int dh,
                               int scw, int sch, int dcw, int dch) {
    return slv == 0 && dlv == 0 && sly == 0 && dly == 0 &&
           sx0 == 0 && sy0 == 0 && dx0 == 0 && dy0 == 0 &&
           sw >= scw && sh >= sch && dw >= dcw && dh >= dch;
}
// Validate that a non-alias copy_texture_gpu call will actually execute a color copy/resolve/blit (no early return).
static bool movie_copy_can_execute(VkImageAspectFlags sa, VkImageAspectFlags da,
                                   uint32_t s_samples, uint32_t d_samples,
                                   VkFormat s_fmt, VkFormat d_fmt,
                                   int sw, int sh, int dw, int dh) {
    if (!(sa & VK_IMAGE_ASPECT_COLOR_BIT) || !(da & VK_IMAGE_ASPECT_COLOR_BIT)) return false;
    if ((sa | da) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) return false;
    if (s_samples > 1 && d_samples == 1 && sw == dw && sh == dh && s_fmt == d_fmt) return true;
    return s_samples == d_samples;
}
static void movie_copy_propagate(TexBackend *d, const TexBackend *s) {
    if (d && s && s->is_movie && s->movie_w > 0 && s->movie_h > 0) {
        d->is_movie = true;
        d->movie_boxed = s->movie_boxed;
        d->movie_w = s->movie_w;
        d->movie_h = s->movie_h;
    }
}
// A draw into a reshaped (wider / taller than 16:9) pass at the uniform scale, centred: UI keeps its 16:9 layout.
static void vp_box_16_9(void) {
    int pw, ph;
    pass_size(&pw, &ph);
    float u = fminf(pass_sx, pass_sy);
    vp_sx = vp_sy = u;
    vp_ox = (pw - st.rt.color[0]->width * u) * 0.5f;
    vp_oy = (ph - st.rt.color[0]->height * u) * 0.5f;
}
static void draw(CmdRec *r, bool indexed) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    PROF_T0();
    if (xfb_on && xm_back_n) {  // (xm_batch) a session reading a pending range: its copy back first
        bool reads = false;
        for (int i = 0; i < 16 && !reads; i++) reads = xm_back_reads(st.vbuf[i].addr, st.vbuf[i].size);
        for (int i = 0; i < 16 && !reads; i++) reads = xm_back_reads(sbufs(0)[i].addr, sbufs(0)[i].size);
        for (int i = 0; i < 18 && !reads; i++) reads = xm_back_reads(st.ubuf[0][i].addr, st.ubuf[0][i].size);
        if (reads) xm_flush_back();
    }
    if (xfb_on) {
        xfb_note();
        // the transform feedback buffers are GPU-written: never served from the vertex cache's CPU-side copies (later
        // draws read them as vertex streams - Z-A's characters take their per-instance data from them; a stale copy
        // froze their animation and stretched their meshes)
        for (int b = 0; b < 4; b++) gw_note("transform feedback", xfb_addr[b], xfb_size[b], st.program ? st.program->stages[0].code_addr : 0);
        for (int b = 0; b < 4; b++)
            if (xfb_addr[b] >= gwin_base && xfb_size[b] && xfb_addr[b] - gwin_base < gwin_max)
                vc_mark_gpu(xfb_addr[b] - gwin_base, xfb_addr[b] - gwin_base + xfb_size[b]);
        xm_draw = xm_prepare();
        static int nlog;
        if (nlog++ < 24)
            LOG("[xfb] draw indexed %d mode %llu first %llx count %llu | buffers %llx+%llx %llx+%llx %llx+%llx\n", indexed,
                (unsigned long long)r->x[0], (unsigned long long)r->x[1], (unsigned long long)r->x[2], (unsigned long long)xfb_addr[0],
                (unsigned long long)xfb_size[0], (unsigned long long)xfb_addr[1], (unsigned long long)xfb_size[1],
                (unsigned long long)xfb_addr[2], (unsigned long long)xfb_size[2]);
    }
    NvnProgram *bound = st.program;  // (issue #22: draw_ may draw E with a substitute program - i22fix_apply - the game's
    draw_(r, indexed);               // program back after every draw, whatever draw_ did)
    st.program = bound;
    if (g_vk.hdr && st.rt.color[0] && !st.color_state[0] && st.rt.color[0]->backend &&
        ((TexBackend *)st.rt.color[0]->backend)->presented)
        hdr_snapshot(st.rt.color[0]);  // (the game's picture in the image it shows, before the HUD over it)
    PROF_ADD(prof_draw);
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}
static uint64_t prof_tp_;
static bool fsr2_copy(Prog *pr);
// (per-object motion vectors: objmv_*, after fsr2_copy) a draw call as it was made, to make it again
typedef struct { int kind; bool indexed; VkBuffer buf; VkDeviceSize off; uint32_t a, b, c, e; int32_t d; } MvDrawCall;
static void objmv_capture(Prog *pr, Pipe *pp, uint32_t streams, const VkDescriptorSet sets[2], const uint32_t *dyn,
                          uint32_t nv_dyn, bool gmem, const MvDrawCall *dc);
static void objmv_replay(TexBackend *mvb, VkImageView zv, uint32_t w, uint32_t h);
static void objmv_check(TexBackend *sb, TexBackend *mvb, uint32_t w, uint32_t h, bool cam);
static bool water_is_surface(Prog *pr);
static void water_before(void);
static void water_surface_cube(Prog *pr);
static void draw_(CmdRec *r, bool indexed) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    if (prof) prof_tp_ = SDL_GetTicksNS();
    if (!st.program) { SKIP(SK_NO_PROGRAM); return; }
    ShaderKey vkey = {0}, fkey = {0};
    vkey.glsl = fkey.glsl = 1;
    vkey.flip_y = fkey.flip_y = nvn_window_origin() == 0;
    vkey.depth_minus_one = nvn_depth_mode() == 0;
    for (int i = 0; i < st.nattrib; i++) vkey.attr_kind[i] = (uint8_t)attr_kind(st.attrib[i] & 0xFF);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) fkey.rt_kind[i] = (uint8_t)rt_kind(st.rt.color[i]->format);
    vkey.stencil_slots = stencil_slots(0, &vkey.sint_slots);
    fkey.stencil_slots = stencil_slots(1, &fkey.sint_slots);
    if (xfb_on && st.program->stages[0].control) {  // the program's transform feedback layout (its control section)
        const uint8_t *ctl = st.program->stages[0].control;
        for (int b = 0; b < 4; b++) {
            vkey.xfb_count[b] = (uint8_t)*(const uint32_t *)(ctl + 0x4DC + 0x80 * b);
            vkey.xfb_stride[b] = *(const uint16_t *)(ctl + 0x6DC + 2 * b);
        }
    }
    Prog *pr = get_program(st.program, &vkey, &fkey);
    if (!pr || !wait_built(&pr->state, r, indexed, false)) {
        SKIP(!pr ? SK_NO_STAGES : atomic_load(&pr->state) == 2 ? SK_PROG_FAILED : SK_PROG_BUILDING);
        return;
    }
    // Issue #22: E drawn with D's vertex shader (i22fix_apply) - pr and st.program become the substitute's for the rest
    // of this draw; its program not built yet: skipped, as any draw whose program is building. Off: nothing.
    int i22fix_skip = i22fix_apply(&pr, r, indexed, &vkey, &fkey);
    if (i22fix_skip >= 0) { SKIP(i22fix_skip); return; }
    {   // Stretched-geometry guard: a vertex shader reading attributes past a one-attribute layout (position only) is
        // drawing with another draw's layout - its mesh spreads across the sky (Z-A: 36-byte vertices read as 12).
        // Skipped until the cause is found; LARECOMP_SKIP_BAD_LAYOUT=0 draws them.
        static int on = -1, fault = -1;
        if (on < 0) on = !ENV("LARECOMP_SKIP_BAD_LAYOUT") || atoi(ENV("LARECOMP_SKIP_BAD_LAYOUT")) != 0;
        // (test) LARECOMP_TEST_BAD_LAYOUT=<n>: every n-th draw of a mesh with interleaved vertices (stride >= 24) gets
        // the bug's layout (position only, stride 12) - the stretching on demand, to test this guard
        if (fault < 0) fault = ENV("LARECOMP_TEST_BAD_LAYOUT") ? atoi(ENV("LARECOMP_TEST_BAD_LAYOUT")) : 0;
        static uint64_t nfault;
        if (fault > 0 && st.nattrib > 1 && st.nstream && st.stream[0][0] >= 24 && (pr->vi->attr_mask & ~1u) &&
            (st.attrib[0] & 0xFF) == 0x22 && (st.attrib[0] >> 8 & 0x1F) == 0 && ++nfault % (uint64_t)fault == 0) {
            st.nattrib = 1;
            st.stream[0][0] = 12;
        }
        if (on && st.nattrib <= 1 && (pr->vi->attr_mask & ~1u)) {
            static uint64_t nskip;
            if (nskip++ < 20) LOG("[vtxstate] skipped draw: vs %llx reads attribs %#x, %d bound (vb0 %llx)\n",
                                  (unsigned long long)st.program->stages[0].code_addr, pr->vi->attr_mask, st.nattrib,
                                  (unsigned long long)st.vbuf[0].addr);
            SKIP(SK_HIDDEN);
            return;
        }
    }
    g_mirror_w = 0;
    {   // mirror mode (runtime/hle/mirror.c; as renderer.m): the scene's composite into the window image is mirrored
        extern bool mirror_draw(bool composite);
        extern bool mirror_scene_size(uint32_t sw, uint32_t sh, uint32_t w, uint32_t h);
        bool composite = false;
        if (st.rt.ncolor && st.rt.color[0] && renderer_texture_presented(st.rt.color[0]) && !(st.color_state[0] & 1) &&
            r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 &&
            st.viewport[0][2] >= st.rt.color[0]->width && st.viewport[0][3] >= st.rt.color[0]->height && pr->fi->ntex) {
            NvnTexture *t0 = shader_texture_object(1, &pr->fi->tex[0]);
            composite = t0 && mirror_scene_size(t0->width, t0->height, st.rt.color[0]->width, st.rt.color[0]->height);
            extern void scene_size_log(bool composite, uint32_t sw, uint32_t sh, uint32_t w, uint32_t h);
            if (t0) scene_size_log(composite, t0->width, t0->height, st.rt.color[0]->width, st.rt.color[0]->height);
        }
        if (mirror_draw(composite)) {
            vkey.flip_x = fkey.flip_x = true;
            pr = get_program(st.program, &vkey, &fkey);
            if (!pr || atomic_load(&pr->state) != 1) { SKIP(!pr ? SK_NO_STAGES : atomic_load(&pr->state) == 2 ? SK_PROG_FAILED : SK_PROG_BUILDING); return; }
            g_mirror_w = (float)st.rt.color[0]->width;
        }
    }
    snap_apply(0, snap_draw_mask(pr->vi));  // (overlap) the storage slots the shaders read: a copy where the CPU rewrites the range
    snap_apply(1, snap_draw_mask(pr->fi));
    {   // fragment-shader hash: interior backdrop detection (widescreen), skip lists, item twinkle (renderer.m)
        static uint64_t last_fs_t[NCTX], last_hash_t[NCTX];
        uint64_t *lfs = &last_fs_t[sw_ctx()], *lh = &last_hash_t[sw_ctx()];
#define last_fs (*lfs)
#define last_hash (*lh)
        uint64_t fs = st.program->stages[1].code_addr;
        if (fs != last_fs) {
            // (once a frame per shader: a town at World detail 4x switches fragment shader on most of its ~5600 draws,
            // and hashing 512 bytes each time was 2.5 % of the render thread)
            static struct { uint64_t fs, hash, frame; } memo_t[NCTX][512];
            __typeof__(memo_t[0][0]) *memo = memo_t[sw_ctx()];
            uint64_t f = vk_frame_counter() + 1;
            __typeof__(memo[0]) *e = &memo[(fs >> 6 ^ fs >> 15) & 511];
            last_fs = fs;
            if (fs && e->fs == fs && e->frame == f) last_hash = e->hash;
            else {
                last_hash = fs ? hmix(1469598103934665603ull, (const void *)(uintptr_t)(fs + 0x30), 0x200) : 0;
                *e = (__typeof__(*e)){fs, last_hash, f};
            }
        }
        static uint64_t skip[16];
        static int nskip = -1;
        if (nskip < 0) {
            nskip = 0;
            for (const char *p = ENV("LARECOMP_SKIP_FSHASH"); p && *p && nskip < 16;) {
                skip[nskip++] = strtoull(p, (char **)&p, 16);
                while (*p == ',') p++;
            }
        }
        for (int i = 0; i < nskip; i++) if (last_hash == skip[i]) { SKIP(SK_DEBUG_SKIP); return; }
        extern bool pot_level_active(void);
        if (last_hash == 0xa2177dbb72197bd9ull && pot_level_active()) { SKIP(SK_HIDDEN); return; }
        if (g_debug_frame && getenv("LARECOMP_LOG_FSHASH")) LOG("[fshash] fs=%llx hash %016llx\n", (unsigned long long)fs, (unsigned long long)last_hash);
        g_cur_fs_hash = last_hash;
#undef last_fs
#undef last_hash
        extern void widescreen_backdrop_seen(uint64_t hash);
        extern void widescreen_scene_ubo(const float *u, uint64_t size);  // (the camera's view-projection)
        if (g_cur_fs_hash) widescreen_backdrop_seen(g_cur_fs_hash);  // (no lock: rare writes of one frame number / camera)
        if (st.ubuf[0][5].addr) widescreen_scene_ubo((const float *)(uintptr_t)st.ubuf[0][5].addr, st.ubuf[0][5].size);
    }
    mv_note_draw(pr);
    if (fsr2_wanted() && fsr2_copy(pr)) return;  // (FSR 2 resolved the frame in place of the game's copy)
    if (g_debug_frame) debug_draw(pr, r, indexed);
    if (ct_on()) ct_draw(pr, "draw");
    alias_draw_check();
    resolve_feedback(pr);
    snap_reason = "draw-depthwrite";
    if (st.rt.depth && ((st.ds[0] && st.ds[1]) || draw_writes_stencil())) snapshots_invalidate(st.rt.depth);
    snap_reason = "draw-colorwrite";
    for (int i = 0; i < st.rt.ncolor; i++)
        if (st.rt.color[i] && ((st.channel_mask >> (i * 4)) & 0xF)) snapshots_invalidate(st.rt.color[i]);
    bool lighting = gtao_is_lighting(pr);
    if (lighting) gtao_before_lighting(pr);  // (GTAO from the scene depth, before the lighting reads the G-buffer)
    if (water_is_surface(pr)) water_before(), water_surface_cube(pr);  // (the depth and the lit image before the water, and its cube: water_apply)
    begin_pass();
    if (lighting) gtao_mark(st.rt.ncolor ? st.rt.color[0] : NULL);
    VkCommandBuffer cb = vk_cmd();

    uint32_t mode = (uint32_t)r->x[0];
    VkPrimitiveTopology topo;
    uint8_t cls;
    bool native = topology(mode, &topo, &cls);
    if (!native && mode != 6 && mode != 7 && mode != 9) { SKIP(SK_TOPOLOGY); return; }

    uint64_t prof_tk_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_pre += prof_tk_ - prof_tp_;
    // ---- pipeline
    uint32_t rt_fmt[8] = {0};
    for (int i = 0; i < st.rt.ncolor && i < 8; i++) rt_fmt[i] = st.rt.color[i] ? st.rt.color[i]->format : 0;
    const char *a2c_draw_env = ENV("LARECOMP_A2C");
    uint8_t a2c_in = a2c_draw_env ? (atoi(a2c_draw_env) != 0) : ((pass_samples > 1) ? (st.ms[0xC] != 0) : 0);
    const PipeIn in = {st.attrib, st.nattrib, (const uint32_t (*)[2])st.stream, st.nstream, rt_fmt, st.channel_mask,
                       st.color_state[0], (const uint8_t (*)[8])st.blend, a2c_in, cls, st.poly[2] != 1};
    // the key's parts draw_ adds to pipe_key_fill's: the depth clamp, wherever the game asks for it (shadow maps: the
    // casters past the light's planes; the UI: its quads, some outside the depth range - clipped, the item-count window
    // after using a candy stayed hidden; Pokémon Z-A's sun shadow cascades, #43), primitive restart and the depth /
    // stencil state in the pipeline on Vulkan 1.2 drivers. Colour passes with depth had it off until v0.0.29 (Vulkan's
    // clamp also turns off near clipping - huge dark sheets in Z-A's plaza at night, from the same days as the GPU-culled
    // draws' wrong index ranges that drew such sheets, 4f30030). But Scarlet / Violet draws its terrain in a depth-only
    // pre-pass and then in the G-buffer with an EQUAL test, both clamped: the ground nearer than the near plane (the
    // catch camera, at ground level) got a clamped depth in the pre-pass and was clipped in the G-buffer - the ground
    // under the Poké Ball black, with stripes at its edge. LARECOMP_VK_CLAMP_COLOR=0: as before (not in colour passes
    // with depth, except onto a square depth target); LARECOMP_VK_NO_DEPTH_CLAMP=1: never.
    static int noclamp = -1, clamp_color = -1;
    if (noclamp < 0) {
        const char *nc = ENV("LARECOMP_VK_NO_DEPTH_CLAMP");
        const char *cc = ENV("LARECOMP_VK_CLAMP_COLOR");
        noclamp = nc && atoi(nc) != 0;
        clamp_color = !cc || atoi(cc) != 0;
    }
    bool square_depth = st.rt.depth && st.rt.depth->width == st.rt.depth->height;
    uint8_t dclamp = st.depth_clamp && !noclamp && (!pass_ncolor || !pass_depth || square_depth || clamp_color);
    uint8_t restart = g_vkc.no_eds2 && st.prim_restart && native && (mode == 3 || mode == 5);  // (as SS_PRE below)
    FixedState fx;
    fixed_state(&fx, vkey.flip_x, vkey.flip_y, native, topo);
    bool memo_on = key_memo_on() && !g_vkc.no_eds1;  // (no extended dynamic state: the whole fixed state is in the key)
    struct PipeMemo *pms = memo_on ? __atomic_load_n(&pr->pmemo, __ATOMIC_ACQUIRE) : NULL;
    if (memo_on && !pms) {  // (one per context: the shadow workers' parts draw with the same programs at once)
        SWL_AT(7318);
        if (!(pms = pr->pmemo)) { pms = calloc(NCTX, sizeof *pms); __atomic_store_n(&pr->pmemo, pms, __ATOMIC_RELEASE); }
        SWU();
    }
    struct PipeMemo *pm = memo_on ? &pms[sw_ctx()] : NULL;
    int nat = st.nattrib < 0 ? 0 : st.nattrib > 16 ? 16 : st.nattrib, nst = st.nstream < 0 ? 0 : st.nstream > 16 ? 16 : st.nstream;
    uint32_t npc = pass_ncolor < 8 ? pass_ncolor : 8;
    prof_key_draws++;
    PipeKey k;
    VtxLayout vl;  // the draw's vertex layout (on a memo hit: the memo's, made from the same inputs)
    Pipe *pp;
    if (pm && pm->pp && pm->gen == key_gen && pm->ncolor == pass_ncolor && pm->samples == pass_samples && pm->depth == pass_depth &&
        pm->stencil == pass_stencil && pm->nattrib == st.nattrib && pm->nstream == st.nstream && pm->channel_mask == in.channel_mask &&
        pm->blend_on == in.blend_on && pm->a2c == in.a2c && pm->cls == in.topo_class && pm->poly_fill == in.poly_fill &&
        pm->depth_clamp == dclamp && pm->restart == restart &&
        !memcmp(pm->color, pass_color, npc * sizeof pass_color[0]) && !memcmp(pm->rt_fmt, rt_fmt, sizeof rt_fmt) &&
        !memcmp(pm->attrib, st.attrib, (size_t)nat * 4) && !memcmp(pm->stream, st.stream, (size_t)nst * 8) &&
        !memcmp(pm->blend, st.blend, sizeof pm->blend)) {
        prof_pipe_memo++;
        pp = pm->pp;
        vl = pm->vl;  // (the key leaves the layout out with dynamic vertex input: kept with the memo)
    } else {
        pipe_key_fill(&k, pr, &in, pass_color, pass_ncolor, pass_depth, pass_stencil, pass_samples, &vl);
        k.depth_clamp = dclamp, k.restart = restart;
        if (g_vkc.no_eds1) k.fx = fx;
        uint64_t ph = hash_words(0xcbf29ce484222325ull, &k, sizeof k);
        if (!ph) ph = 1;
        SWL_AT(7345);  // (shadow workers: the pipeline table)
        KvEnt *pslot = kv_find(&pipe_kv, ph);
        pp = pslot->val;
        if (!pp) {
            if (shader_rec_recording()) {
                ShaderRec rec;
                rec_from_state(&rec, vkey.flip_x, mode);
                shader_rec_note(&rec);
                shader_seen_code(rec.vs, &st.program->stages[0]);  // (their code for the next start's preparation)
                shader_seen_code(rec.fs, &st.program->stages[1]);
            }
            pp = prep_pipes_n ? PREP_GET(prep_pipes, ph) : NULL;  // prepared before the game
            if (pp && memcmp(&pp->k, &k, sizeof k)) pp = NULL;
            if (!pp) {
                uint64_t vh = shader_stage_hash(&st.program->stages[0]), fh = shader_stage_hash(&st.program->stages[1]);
                shader_first_visit("pipeline", vh, fh, ph);
                if (prep_check_on()) {
                    ShaderRec rec;
                    rec_from_state(&rec, vkey.flip_x, mode);
                    prep_check(&k, vh, fh, &rec);
                }
                pp = calloc(1, sizeof *pp);
                pp->k = k;
                run_job(build_pipeline, pp, &pp->state);
            }
            kv_set(&pipe_kv, pslot, ph, pp);
        }
        SWU();
        if (pm) {
            pm->gen = key_gen; pm->ncolor = pass_ncolor; pm->samples = pass_samples; pm->depth = pass_depth; pm->stencil = pass_stencil;
            pm->nattrib = st.nattrib; pm->nstream = st.nstream; pm->channel_mask = in.channel_mask; pm->blend_on = in.blend_on;
            pm->a2c = in.a2c; pm->cls = in.topo_class; pm->poly_fill = in.poly_fill;
            pm->depth_clamp = dclamp; pm->restart = restart;
            memcpy(pm->color, pass_color, sizeof pm->color); memcpy(pm->rt_fmt, rt_fmt, sizeof pm->rt_fmt);
            memcpy(pm->attrib, st.attrib, sizeof pm->attrib); memcpy(pm->stream, st.stream, sizeof pm->stream);
            memcpy(pm->blend, st.blend, sizeof pm->blend);
            pm->pp = pp;
            pm->vl = vl;
        }
    }
    {   // a pipeline still building: the draw is skipped (as on Metal) - unless the recorder waits for it at the bind
        // (LARECOMP_VK_RECORDER_WAIT_PIPES=1, record.c: pipe_resolve), or it is about to be ready (wait_built: one in
        // the driver's cache, up to 4 ms); one that failed is always skipped
        int ps = atomic_load(&pp->state);
        if (ps == 2) { SKIP(SK_PIPE_FAILED); return; }
        if (ps != 1 && !rec_wait_pipes() && !wait_built(&pp->state, r, indexed, true)) { SKIP(SK_PIPE_BUILDING); return; }
    }
    ss_enabled();
    if (ss_need(SS_PIPE, ss.pipe == pp)) { ss.pipe = pp; rec_CmdBindPipelineObj(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp); }
    if (g_vk.vids && ss_need(SS_VI, !memcmp(&ss.vi, &vl, sizeof vl))) { ss.vi = vl; set_vertex_input(cb, &vl); }
    uint64_t prof_td_ = prof ? SDL_GetTicksNS() : 0;
    if (prof) prof_key += prof_td_ - prof_tk_;

    // ---- depth / stencil (dynamic; without extended dynamic state, the parts in the pipeline: fixed_state)
    VkBool32 dt = fx.dt, dw = fx.dw;
    if (ss_need(SS_DT, ss.dt == dt)) { ss.dt = dt; rec_CmdSetDepthTestEnable(cb, dt); }
    if (ss_need(SS_DW, ss.dw == dw)) { ss.dw = dw; rec_CmdSetDepthWriteEnable(cb, dw); }
    VkCompareOp dop = (VkCompareOp)fx.dop;
    bool equal_tol = fx.equal_tol;
    if (ss_need(SS_DOP, ss.dop == dop)) { ss.dop = dop; rec_CmdSetDepthCompareOp(cb, dop); }
    bool sten = fx.sten;
    if (ss_need(SS_STEN, ss.sten == (VkBool32)sten)) { ss.sten = sten; rec_CmdSetStencilTestEnable(cb, sten); }
    {  // (stencil off too: the pipelines take the op as dynamic state, so each command buffer must set it before its first draw)
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            uint32_t v[7] = {fx.sop[f][0], fx.sop[f][1], fx.sop[f][2], fx.sop[f][3], st.stencil_vmask[f] & 0xFF,
                             st.stencil_mask[f] & 0xFF, st.stencil_ref[f]};
            if (!sten) memcpy(v, (uint32_t[7]){VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xFF, 0xFF, 0}, sizeof v);
            if (!ss_need(f ? SS_SFACE1 : SS_SFACE0, !memcmp(v, ss.sface[f], sizeof v))) continue;
            memcpy(ss.sface[f], v, sizeof v);
            rec_CmdSetStencilOp(cb, face, (VkStencilOp)v[0], (VkStencilOp)v[1], (VkStencilOp)v[2], (VkCompareOp)v[3]);
            rec_CmdSetStencilCompareMask(cb, face, v[4]);
            rec_CmdSetStencilWriteMask(cb, face, v[5]);
            rec_CmdSetStencilReference(cb, face, v[6]);
        }
    }
    if (ss_need(SS_BC, !memcmp(ss.bc, st.blend_color, sizeof ss.bc))) { memcpy(ss.bc, st.blend_color, sizeof ss.bc); rec_CmdSetBlendConstants(cb, st.blend_color); }

    // ---- rasterizer
    VkCullModeFlags cm = fx.cull;
    if (ss_need(SS_CULL, ss.cull == cm)) { ss.cull = cm; rec_CmdSetCullMode(cb, cm); }
    VkFrontFace ff = (VkFrontFace)fx.ff;
    if (ss_need(SS_FF, ss.ff == ff)) { ss.ff = ff; rec_CmdSetFrontFace(cb, ff); }
    VkPrimitiveTopology tp = (VkPrimitiveTopology)fx.topo;
    if (ss_need(SS_TOPO, ss.topo == tp)) { ss.topo = tp; rec_CmdSetPrimitiveTopology(cb, tp); }
    VkBool32 pre = st.prim_restart && native && (mode == 3 || mode == 5);
    if (ss_need(SS_PRE, ss.pre == pre)) { ss.pre = pre; rec_CmdSetPrimitiveRestartEnable(cb, pre); }
    {
        float f[3] = {0, 0, 0};
        static int off = -1;
        if (off < 0) off = ENV("LARECOMP_NO_DEPTH_BIAS") != NULL;
        bool on = (st.poly[3] & 4) && !off;
        if (on) memcpy(f, st.poly_offset, sizeof f);
        if (equal_tol) { f[1] -= 8; on = true; }  // (see the EQUAL comment above: a few depth ulps nearer)
        if (ss_need(SS_DBE, ss.dbe == (VkBool32)on)) { ss.dbe = on; rec_CmdSetDepthBiasEnable(cb, on); }
        if (ss_need(SS_DB, !memcmp(ss.db, f, sizeof f))) { memcpy(ss.db, f, sizeof f); rec_CmdSetDepthBias(cb, f[1], f[2], f[0]); }
    }

    // ---- textures first (their scales feed the widescreen decision and the rscale table), then the viewport
    float *vp = st.viewport[0];
    float zn = st.depth_range[0][0], zf = st.depth_range[0][1];
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0; vp_oy = 0;
    // The hooks below look for known textures (texpatch ids, TexBackend.tp > 0) among the fragment textures: one pass
    // finds the texture objects and whether any is known; the loops only run when one is (most draws have none).
    // LARECOMP_VK_NO_HOOK_PASS=1: every loop looks its textures up, as before.
    static int hook_pass = -1;
    if (hook_pass < 0) hook_pass = getenv("LARECOMP_VK_NO_HOOK_PASS") == NULL;
    NvnTexture *ftex[64];
    bool fknown = true;
    if (hook_pass) {
        fknown = false;
        for (int i = 0; i < pr->fi->ntex; i++) {
            ftex[i] = shader_texture_object(1, &pr->fi->tex[i]);
            const TexBackend *b = ftex[i] ? (const TexBackend *)ftex[i]->backend : NULL;
            if (b && b->tp > 0) fknown = true;
        }
        prof_hook_draws++;
        if (fknown) prof_hook_known++;
    }
#define FTEX(i) (hook_pass ? ftex[i] : shader_texture_object(1, &pr->fi->tex[i]))
    {   // the Items page's cursor brackets while the cursor is on the companion card (runtime/hle/pets.c)
        extern bool pets_hide_item_cursor(void);
        extern bool texpatch_is_item_cursor(int id);
        if (pets_hide_item_cursor() && fknown)
            for (int i = 0; i < pr->fi->ntex; i++) {
                NvnTexture *t = FTEX(i);
                const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
                if (b && b->tp > 0 && texpatch_is_item_cursor(b->tp - 1)) { SKIP(SK_HIDDEN); return; }
            }
    }
    float tint[RECOLOUR_TINT_FLOATS] = {0};
    int recol[2] = {0, 0};  // Cosmetics recolour textures' classes (runtime/nvn/recolour.c)
    if (pr->fi->uses_tint) {
        for (int i = 0; i < pr->fi->ntex && fknown; i++) {
            NvnTexture *t = FTEX(i);
            const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
            if (b && b->tp > 0) recolour_add(recol, recolour_class_of(b->tp - 1));
        }
        recolour_tint_draw(recol, tint);  // (the highlights below win)
        extern bool pot_tint(const float *m, float tint[8], uint64_t mesh);
        extern bool pot_glow_active(void);
        // (uniform buffer 2 = the model matrix, hardware cbuf 5 - only when the vertex shader reads it: a binding left
        // from an earlier draw is another object's matrix)
        if (((pr->vi->cbuf_mask >> 5) & 1) && st.ubuf[0][2].addr && st.ubuf[0][2].size >= 48 && pot_glow_active())
            pot_tint((const float *)(uintptr_t)st.ubuf[0][2].addr, tint, st.vbuf[0].addr);
        // the gold Escape Bottle (runtime/hle/escape.c): draws sampling the bottle's item icon / model map
        // and the HUD hearts (heart colour, Cosmetics tab)
        extern int texpatch_bottle_kind(int id);
        extern int texpatch_heart_kind(int id);
        extern void texpatch_heart_tint(int kind, float tint[8]);
        int bottle = 0, heart = 0;
        for (int i = 0; i < pr->fi->ntex && fknown; i++) {
            NvnTexture *t = FTEX(i);
            const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
            if (b && b->tp > 0 && !bottle) bottle = texpatch_bottle_kind(b->tp - 1);
            if (b && b->tp > 0 && !heart) heart = texpatch_heart_kind(b->tp - 1);
        }
        if (heart) texpatch_heart_tint(heart, tint);
        if (bottle) {
            extern void escape_bottle_tint(int kind, const float *ui, uint32_t ui_size, float tint[8]);
            int u = bottle == 3 || bottle == 6 ? 3 : 0;  // (fairy / Piece of Power models: their bone matrices - souls.c, triforce.c)
            escape_bottle_tint(bottle, (const float *)(uintptr_t)st.ubuf[0][u].addr, (uint32_t)st.ubuf[0][u].size, tint);
        }
    }
    // fragment textures are bound before deciding the viewport (tex_sc / tex_wh), rscale depends on the decision:
    // build the vertex set now and the fragment set after the decision (stage_set fills tex_sc for stage 1).
    if (prof) prof_dyn += SDL_GetTicksNS() - prof_td_;
    PROF_T0();
    uint32_t dyn[2 * MAX_BIND], ndyn = 0;  // (set reuse: the sets' dynamic offsets, vertex then fragment)
    VkDescriptorSet vset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, NULL, NULL, 0, 0, 0, NULL, dyn, &ndyn);
    uint32_t nv_dyn = ndyn;  // (the vertex set's share of the offsets: objmv_capture and planar_capture swap that set)
    PROF_ADD(prof_sets);
    VkDescriptorImageInfo fii[64];
    uint64_t prof_tx_ = prof ? SDL_GetTicksNS() : 0;
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {
        float sc[2], fr[2]; uint32_t wh[2];
        texture_info(1, pr->fi, i, &fii[i], sc, wh, fr);
        tex_sc[1][i][0] = sc[0]; tex_sc[1][i][1] = sc[1]; tex_wh[1][i][0] = wh[0]; tex_wh[1][i][1] = wh[1];
        tex_fr[1][i][0] = fr[0]; tex_fr[1][i][1] = fr[1];
        // the Hint Sheet (runtime/hle/hint_sheet.c): the dungeon map's icon in place of the first bottle's
        // (tp read after texture_info: its upload check may have identified the texture just now)
        NvnTexture *t = FTEX(i);
        const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
        extern int texpatch_bottle_kind(int id);
        if (b && b->tp > 0 && texpatch_bottle_kind(b->tp - 1) == 1) {
            extern bool hint_sheet_icon_draw(const float *ui, uint32_t ui_size);
            extern NvnTexture *texpatch_hint_icon(void);
            NvnTexture *ht;
            if (hint_sheet_icon_draw((const float *)(uintptr_t)st.ubuf[0][0].addr, (uint32_t)st.ubuf[0][0].size) &&
                (ht = texpatch_hint_icon())) {
                static NvnTextureView none;
                fii[i].imageView = vk_view(ht, &none, false, false, pr->fi->tex[i].type);
            }
        }
    }
#undef FTEX
    if (prof) prof_tex += SDL_GetTicksNS() - prof_tx_;
    bool movie_boxed = false;
    if (st.rt.ncolor && st.rt.color[0]) {
        TexBackend *ftbs[64];
        int ntex = pr->fi->ntex < 64 ? pr->fi->ntex : 64;
        for (int i = 0; i < ntex; i++) {
            NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
            ftbs[i] = t ? (TexBackend *)t->backend : NULL;
        }
        bool depth_test = st.ds[0] && st.rt.depth && compare_fn(st.ds[2]) != VK_COMPARE_OP_ALWAYS;
        bool video_draw_ok = r->x[2] <= 6 && !depth_test && st.viewport[0][0] <= 1.0f && st.viewport[0][1] <= 1.0f &&
                             st.viewport[0][2] >= (float)st.rt.color[0]->width * 0.99f &&
                             fabsf(st.viewport[0][3]) >= (float)st.rt.color[0]->height * 0.99f;
        bool reshaped = fabsf(pass_sx - pass_sy) > 0.01f * pass_sy;
        movie_boxed = movie_draw_apply(cb, (TexBackend *)st.rt.color[0]->backend, st.rt.ncolor, ntex, ftbs, tex_wh[1],
                                       video_draw_ok, reshaped);
    }
    if (!movie_boxed && fabsf(pass_sx - pass_sy) > 0.01f * pass_sy && st.rt.ncolor && renderer_texture_presented(st.rt.color[0])) {
        // Draws into the swapchain image (see renderer.m): composites fill the window, UI goes in a centred 16:9 box.
        bool reads_scene = false;
        for (int i = 0; i < pr->fi->ntex && i < 64; i++)
            if ((tex_sc[1][i][0] != 1 || tex_sc[1][i][1] != 1) && tex_wh[1][i][0] == st.rt.color[0]->width &&
                tex_wh[1][i][1] == st.rt.color[0]->height)
                reads_scene = true;
        bool fullscreen_quad = r->x[2] <= 6 && st.viewport[0][0] == 0 && st.viewport[0][1] == 0 &&
                               st.viewport[0][2] >= st.rt.color[0]->width && st.viewport[0][3] >= st.rt.color[0]->height;
        extern bool widescreen_ui_pane_partial(const float *u, uint64_t size);  // (a smaller layout pane: boxed UI)
        bool fullscreen_fill = pr->fi->ntex == 0 && fullscreen_quad &&
                               !widescreen_ui_pane_partial((const float *)(uintptr_t)st.ubuf[0][0].addr, st.ubuf[0][0].size);
        if (fullscreen_quad && !reads_scene && pr->fi->ntex == 1 && (tex_sc[1][0][0] != 1 || tex_sc[1][0][1] != 1)) reads_scene = true;
        static uint64_t wide_frame = ~0ull;
        extern uint64_t renderer_exec_frame(void);  // (the frame being translated)
        if (reads_scene) wide_frame = renderer_exec_frame();
        bool fade_fill = false;
        if (fullscreen_fill && st.ubuf[1][0].addr && st.ubuf[1][0].size >= 12) {
            const float *c = (const float *)(uintptr_t)st.ubuf[1][0].addr;
            bool black = c[0] < 0.01f && c[1] < 0.01f && c[2] < 0.01f, white = c[0] > 0.99f && c[1] > 0.99f && c[2] > 0.99f;
            fade_fill = black || white;
        }
        bool composite = reads_scene || fade_fill || (fullscreen_fill && wide_frame == renderer_exec_frame());
        if (!composite) wide_frame = ~0ull;
        if (!composite) vp_box_16_9();
    } else if (!movie_boxed && fabsf(pass_sx - pass_sy) > 0.01f * pass_sy && st.rt.ncolor && st.ubuf[0][0].addr) {
        // HUD layout panes and font text drawn into a reshaped intermediate composite target: centred 16:9 box.
        extern bool widescreen_ui_boxed_draw(const float *u, uint64_t size, int ntex,
                                             const float (*sc_)[2], const uint32_t (*wh_)[2]);
        const float *u = (const float *)(uintptr_t)st.ubuf[0][0].addr;
        int ntex = pr->fi->ntex < 64 ? pr->fi->ntex : 64;
        bool boxed = widescreen_ui_boxed_draw(u, st.ubuf[0][0].size, ntex, tex_sc[1], tex_wh[1]);
        if (boxed) vp_box_16_9();
        if (g_debug_frame && (st.ubuf[0][0].size == 128 || st.ubuf[0][0].size == 560)) {
            LOG("[draw]   ui %s %s: proj %g %g %g %g / %g %g %g %g", st.ubuf[0][0].size == 560 ? "pane" : "text",
                boxed ? "boxed" : "not boxed", u[0], u[3], u[5], u[7], u[12], u[13], u[14], u[15]);
            if (st.ubuf[0][0].size == 560) LOG(", rect %g %g %g %g", u[132], u[133], u[134], u[135]);
            LOG("\n");
        }
    }
    {   // camera-attached effects sized for 16:9 are widened with the view (runtime/hle/widescreen.c)
        extern float widescreen_effect_stretch(uint64_t fs_hash);
        float k = widescreen_effect_stretch(g_cur_fs_hash);
        float vx = vp[0] * vp_sx + vp_ox, vw = vp[2] * vp_sx;
        if (k > 1.0f) { vx -= vw * (k - 1) / 2; vw *= k; }
        float vy = vp[1] * vp_sy + vp_oy;
        fsr2_frame_jitter();
        if ((fsr2_jx != 0 || fsr2_jy != 0) && fsr2_scene_pass())  // (FSR 2: the frame's sub-pixel offset)
            vx += fsr2_jx * vw / (float)fsr2_scene_w, vy += fsr2_jy * fabsf(vp[3] * vp_sy) / (float)fsr2_scene_h;
        set_viewport(vx, vy, vw, vp[3] * vp_sy, zn, zf);
        if (g_debug_frame) {
            VkRect2D sr = scissor_rect();
            LOG("[draw]   vp %.2f %.2f %.2f %.2f (stretch %.3f, fs hash %016llx) scissor %d %d %u %u | guest sc %d %d %d %d | pass %ux%u s %.4f %.4f\n",
                vx, vp[1] * vp_sy + vp_oy, vw, vp[3] * vp_sy, k, (unsigned long long)g_cur_fs_hash, sr.offset.x, sr.offset.y,
                sr.extent.width, sr.extent.height, st.scissor[0][0], st.scissor[0][1], st.scissor[0][2], st.scissor[0][3], pass_w, pass_h,
                vp_sx, vp_sy);
        }
    }
    apply_scissor();
    float rs0[4] = {vp_sx, vp_sy, vp_ox, vp_oy};
    uint64_t prof_t1_ = prof ? SDL_GetTicksNS() : 0;
    VkDescriptorSet fset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, tint, 0, 0, 0, fii, dyn, &ndyn);
    if (prof) prof_sets += SDL_GetTicksNS() - prof_t1_;
    VkDescriptorSet sets[2] = {vset, fset};
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 0, 2, sets, ndyn, ndyn ? dyn : NULL);
    if (gc_shift && (pr->vi->uses_gmem || pr->fi->uses_gmem))
        gmem_chunk_bind(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, GMEM_CHUNK_SET(STAGE_FRAGMENT),
                        (pr->vi->uses_gmem ? 1u << STAGE_VERTEX : 0) | (pr->fi->uses_gmem ? 1u << STAGE_FRAGMENT : 0));
    if (pr->vi->uses_gmem_store) vc_mark_sbufs(0);
    if (pr->fi->uses_gmem_store) vc_mark_sbufs(1);

    uint64_t prof_t2_ = prof ? SDL_GetTicksNS() : 0;
    // ---- indices (copied; the largest index bounds the vertex data copied below)
    uint32_t first = 0, count = (uint32_t)r->x[2], inst, base_inst, max_index = 0;
    int32_t base_vertex = 0;
    VkSpan ib = {0};
    VkIndexType itype_vk = VK_INDEX_TYPE_UINT32;
    uint32_t icount = 0;
    bool use_index = indexed || !native;
    bool ind_ok = !indexed && native;  // (g_ind_gpu: the parameters can be the GPU's - first index etc. as in Vulkan)
    const void *scan_src = NULL;  // deferred index scan (direct index buffer)
    uint32_t scan_isz = 0, scan_restart = 0;
    if (!indexed) {
        first = (uint32_t)r->x[1]; base_inst = (uint32_t)r->x[3]; inst = (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 1;
        max_index = first + count;
        if (!native) {
            uint32_t *e = expand_indices(mode, count, NULL, 0, first, &icount);
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
    } else {
        uint32_t itype = (uint32_t)r->x[1];
        uint64_t iaddr = r->x[3];
        base_vertex = (int32_t)r->x[4];
        base_inst = (uint32_t)r->x[5];
        inst = (uint32_t)r->x[6] ? (uint32_t)r->x[6] : 1;
        const void *src = (const void *)(uintptr_t)iaddr;
        uint32_t restart = st.prim_restart ? st.restart_index : 0xFFFFFFFFu;
        if (native && itype != 0) {
            uint32_t isz = itype == 1 ? 2 : 4;
            itype_vk = itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            icount = count;
            ind_ok = true;
            if (iaddr % isz == 0 && guest_direct(1, iaddr, (uint64_t)count * isz, &ib.buf, &ib.off)) {
                scan_src = src; scan_isz = isz; scan_restart = restart;  // max_index only if a stream gets copied
            } else {
                ib = vk_transient_copy(src, (uint64_t)count * isz + 4, 4);
                if (isz == 2) { const uint16_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != (uint16_t)restart) max_index = s[i]; }
                else { const uint32_t *s = src; for (uint32_t i = 0; i < count; i++) if (s[i] > max_index && s[i] != restart) max_index = s[i]; }
            }
        } else if (native) {  // 8-bit indices: widen to 16
            ib = vk_transient((uint64_t)count * 2 + 4, 4);
            uint16_t *w = ib.ptr;
            const uint8_t *s = src;
            for (uint32_t i = 0; i < count; i++) { w[i] = s[i] == 0xFF && st.prim_restart ? 0xFFFF : s[i]; if (s[i] > max_index && w[i] != 0xFFFF) max_index = s[i]; }
            itype_vk = VK_INDEX_TYPE_UINT16;
            icount = count;
        } else {
            uint32_t *e = expand_indices(mode, count, src, (int)itype, 0, &icount);
            for (uint32_t i = 0; i < icount; i++) if (e[i] > max_index) max_index = e[i];
            ib = vk_transient_copy(e, (uint64_t)icount * 4 + 4, 4);
            free(e);
        }
        if (!scan_src) {
            max_index += 1;
            if (base_vertex > 0) max_index += (uint32_t)base_vertex;
        }
    }
    if (use_index && ss_need(SS_IB, ss.ib == ib.buf && ss.io == ib.off && ss.it == itype_vk)) {
        ss.ib = ib.buf; ss.io = ib.off; ss.it = itype_vk;
        rec_CmdBindIndexBuffer(cb, ib.buf, ib.off, itype_vk);
    }

    // ---- vertex streams: the range this draw can read, copied once per submission
    for (int s = 0; s < 17; s++) {
        if (!(vl.used_streams & (1u << s))) continue;
        VkSpan sp;
        if (s == ZERO_BINDING) {
            static const float zero[4] = {0, 0, 0, 1};
            sp = vk_transient_copy(zero, 16, 16);
        } else {
            Range v = st.vbuf[s];
            VkBuffer dbuf;
            VkDeviceSize doff;
            if (v.addr && v.size && guest_direct(0, v.addr, v.size, &dbuf, &doff)) {
                if (ss_on && (ss.vb_known >> s & 1) && ss.vb[s] == dbuf && ss.vo[s] == doff) continue;
                ss.vb[s] = dbuf; ss.vo[s] = doff; ss.vb_known |= 1u << s;
                rec_CmdBindVertexBuffers(cb, (uint32_t)s, 1, &dbuf, &doff);
                continue;
            }
            if (scan_src) {  // copying after all: the index scan bounds the copy
                if (scan_isz == 2) { const uint16_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != (uint16_t)scan_restart) max_index = q[i]; }
                else { const uint32_t *q = scan_src; for (uint32_t i = 0; i < icount; i++) if (q[i] > max_index && q[i] != scan_restart) max_index = q[i]; }
                max_index += 1;
                if (base_vertex > 0) max_index += (uint32_t)base_vertex;
                scan_src = NULL;
            }
            uint64_t stride = vl.stride[s], need;
            if (!stride) need = 256;
            else if (vl.divisor[s] && g_ind_gpu) need = v.size ? v.size : 1u << 20;  // (instances only the GPU knows: all)
            // (Vulkan applies the first instance outside the divisor: instance i reads element base_inst + i / divisor -
            // the copy runs from the binding's start, so up to base_inst + inst covers it, the runs below included; it
            // was (base_inst + inst) / divisor: short when both are over 1 - found by gabrielsem16-bot, #39)
            else if (vl.divisor[s]) {
                need = ((uint64_t)base_inst + inst) * stride + 256;
                // (#39: how often the old size fell short - the AMD battle faults stopped with this fix in test-amd-3;
                // LARECOMP_VK_OLD_INSTANCE_COPY=1 brings the old size back, for that A/B)
                static int old_copy = -1;
                if (old_copy < 0) old_copy = ENV("LARECOMP_VK_OLD_INSTANCE_COPY") && atoi(ENV("LARECOMP_VK_OLD_INSTANCE_COPY"));
                uint64_t old = ((uint64_t)base_inst + inst + vl.divisor[s] - 1) / vl.divisor[s] * stride + 256;
                uint64_t reads = ((uint64_t)base_inst + (inst ? inst - 1 : 0) / vl.divisor[s] + 1) * stride;
                if (old < reads) {
                    static int nlog;
                    if (nlog++ < 20)
                        LOG("[vk] instance stream %d: the old copy (%llu bytes) was short of the %llu read - first instance %u, "
                            "%u instances, divisor %u, stride %llu (frame %llu)%s\n", s, (unsigned long long)old,
                            (unsigned long long)reads, base_inst, inst, vl.divisor[s], (unsigned long long)stride,
                            (unsigned long long)renderer_present_count(), old_copy ? " - kept short (test)" : "");
                }
                if (old_copy) need = old;
            }
            else need = (uint64_t)max_index * stride + 256;
            if (v.size && need > v.size) need = v.size;
            if (!v.addr) { static const uint8_t zero[256]; sp = vk_transient_copy(zero, 256, 16); }
            else sp = guest_copy(v.addr, need ? need : 16);
        }
        VkDeviceSize off = sp.off;
        if (ss_on && (ss.vb_known >> s & 1) && ss.vb[s] == sp.buf && ss.vo[s] == off) continue;
        ss.vb[s] = sp.buf; ss.vo[s] = off; ss.vb_known |= 1u << s;
        rec_CmdBindVertexBuffers(cb, (uint32_t)s, 1, &sp.buf, &off);
    }

    if (prof) prof_vtx += SDL_GetTicksNS() - prof_t2_;
    // ---- draw
    // Instanced draws with item spots among their instances (bushsanity: bushes are drawn in batches, each instance's
    // model matrix in vertex storage buffer 0, indexed by instance id): drawn as runs of instances, gold runs tinted
    // (as renderer.m; gl_InstanceIndex includes the base instance, so a run [b + k, b + k + n) draws exactly those).
    // Only for a vertex shader that reads storage buffer 0 (its binding stays from the last draw that set it).
    enum { MAX_RUNS = 64 };
    uint32_t run_at[MAX_RUNS], run_n[MAX_RUNS];
    bool run_gold[MAX_RUNS];
    int nruns = 0;
    if (pr->fi->uses_tint && (pr->vi->ssbo_read & 1) && st.sbuf[0][0].addr) {
        extern bool pot_glow_active(void);
        extern bool pot_instance_gold(const float *m);
        uint64_t recs = st.sbuf[0][0].size / 64;
        if (pot_glow_active() && (uint64_t)base_inst + inst <= recs) {
            const float *m = (const float *)(uintptr_t)st.sbuf[0][0].addr;
            for (uint32_t i = 0; i < inst; i++) {
                bool g = pot_instance_gold(m + (size_t)(base_inst + i) * 16);
                if (nruns && run_gold[nruns - 1] == g) { run_n[nruns - 1]++; continue; }
                if (nruns == MAX_RUNS) { nruns = 0; break; }  // (too fragmented: draw as is)
                run_at[nruns] = i; run_n[nruns] = 1; run_gold[nruns] = g; nruns++;
            }
            if (nruns == 1 && !run_gold[0]) nruns = 0;
        }
    }
    if (!nruns) { run_at[0] = 0; run_n[0] = inst; nruns = 1; run_gold[0] = false; }
    if (nruns == 1 && !raster_discard) {  // (the characters' draws, kept to be drawn again into the motion vectors)
        MvDrawCall dc = {.kind = g_ind_gpu && ind_ok ? 2 : use_index ? 1 : 0, .indexed = indexed};
        if (dc.kind == 2) dc.buf = gmem_buf, dc.off = g_ind_off, dc.a = g_ind_stride;
        else if (dc.kind == 1) dc.a = icount, dc.b = run_n[0], dc.d = base_vertex, dc.e = base_inst + run_at[0];
        else dc.a = count, dc.b = run_n[0], dc.c = first, dc.e = base_inst + run_at[0];
        objmv_capture(pr, pp, vl.used_streams, sets, dyn, nv_dyn, gc_shift && (pr->vi->uses_gmem || pr->fi->uses_gmem), &dc);
    }
    PlDrawCall pl_dc = {.kind = g_ind_gpu && ind_ok ? 2 : use_index ? 1 : 0, .indexed = indexed};  // (planar reflections)
    if (pl_dc.kind == 2) pl_dc.buf = gmem_buf, pl_dc.off = g_ind_off, pl_dc.a = g_ind_stride;
    else if (pl_dc.kind == 1) pl_dc.a = icount, pl_dc.b = run_n[0], pl_dc.d = base_vertex, pl_dc.e = base_inst + run_at[0];
    else pl_dc.a = count, pl_dc.b = run_n[0], pl_dc.c = first, pl_dc.e = base_inst + run_at[0];
    if (nruns == 1 && !raster_discard && planar_on())  // (the G-buffer's draws, kept to be drawn again mirrored)
        planar_capture(pr, pp, vl.used_streams, sets, ndyn, dyn, nv_dyn, gc_shift && (pr->vi->uses_gmem || pr->fi->uses_gmem), &pl_dc);
    for (int ri = 0; ri < nruns; ri++) {
        if (nruns > 1 || run_gold[0]) {  // this run's tint: a fragment set of its own
            float rtint[RECOLOUR_TINT_FLOATS] = {0};
            extern void pot_gold_tint(float tint[8]);
            if (run_gold[ri]) pot_gold_tint(rtint);
            else recolour_tint_draw(recol, rtint);
            uint32_t rdyn[MAX_BIND], nrdyn = 0;
            VkDescriptorSet rset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, rtint, 0, 0, 0, fii, rdyn, &nrdyn);
            rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pr->layout, 1, 1, &rset, nrdyn, nrdyn ? rdyn : NULL);
        }
        if (raster_discard) {  // (rasterizer discard: the vertex stage runs - transform feedback - and nothing is drawn)
            VkRect2D none = {{0, 0}, {0, 0}};
            rec_CmdSetScissor(cb, 0, 1, &none);
            ss.known &= ~SS_SC;
        }
        if (g_ind_gpu && ind_ok && nruns == 1) {  // (indirect: instance count etc. written by the GPU - culling)
            rec_CmdDrawIndirect(cb, gmem_buf, g_ind_off, 1, g_ind_stride, indexed);
        } else if (!use_index) rec_CmdDraw(cb, count, run_n[ri], first, base_inst + run_at[ri]);
        else rec_CmdDrawIndexed(cb, icount, run_n[ri], 0, base_vertex, base_inst + run_at[ri]);
    }
    if (lighting && nruns == 1 && planar_on())  // (and the lighting, onto the mirrored G-buffer)
        planar_light(pr, pp, fii, vset, dyn, nv_dyn, rs0, tint, vl.used_streams, &pl_dc);
    dl_note("draw vs %llx fs %llx%s xfb %d", (unsigned long long)st.program->stages[0].code_addr,
            (unsigned long long)st.program->stages[1].code_addr, g_ind_gpu && ind_ok ? " indirect" : "", xfb_on);
    if (g_ind_gpu && ind_ok && dl_on > 0) {
        dl_args[(dl_n - 1) & (DL_N - 1)] = gwin_base + gmem_lo + g_ind_off;
        memcpy(dl_snap[(dl_n - 1) & (DL_N - 1)], (const void *)(uintptr_t)(gwin_base + gmem_lo + g_ind_off), 20);
    }
    if (dl_on > 0) vk_checkpoint(cb, dl_n);
    stat_draws++;
    cum_draws++;
    if (g_gpuprof_frame && gp_n) {
        if (!gp_draws[gp_n - 1]) {  // first draw of the row: its fragment shader hash
            size_t l = strlen(gp_label[gp_n - 1]);
            snprintf(gp_label[gp_n - 1] + l, sizeof gp_label[0] - l, ", first fs %016llx", (unsigned long long)g_cur_fs_hash);
        }
        gp_draws[gp_n - 1]++;
    }
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

static void debug_draw(Prog *pr, CmdRec *r, bool indexed) {
    LOG("[draw] %s inst %llu mode %llu count %llu prog vs=%llx fs=%llx | vs cb %#x fs cb %#x | rt %d", indexed ? "e" : "a",
        (unsigned long long)(indexed ? r->x[6] : r->x[4]), (unsigned long long)r->x[0], (unsigned long long)r->x[2],
        (unsigned long long)st.program->stages[0].code_addr, (unsigned long long)st.program->stages[1].code_addr,
        pr->vi->cbuf_mask, pr->fi->cbuf_mask, st.rt.ncolor);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) LOG(" %ux%u/f%#x", st.rt.color[i]->width, st.rt.color[i]->height, st.rt.color[i]->format);
    // (investigation) each colour slot as bound: its index, the view's format when the game named one, the Vulkan
    // format of the image and of the pass, the texture
    LOG(" | slots");
    for (int i = 0; i < st.rt.ncolor && i < 8; i++) {
        if (!st.rt.color[i]) { LOG(" %d:none", i); continue; }
        const TexBackend *cb_ = (const TexBackend *)st.rt.color[i]->backend;
        LOG(" %d:f%#x", i, st.rt.color[i]->format);
        if (st.rt.has_cview[i]) LOG("/view mask %#x f%#x", st.rt.cview[i].setmask, st.rt.cview[i].format);
        LOG("/vk %d pass %d %p", cb_ ? (int)cb_->format : -1, (int)pass_color[i], (void *)st.rt.color[i]);
    }
    if (st.rt.depth) LOG(" d%#x %p", st.rt.depth->format, (void *)st.rt.depth);
    LOG(" | vp %.0f %.0f %.0f %.0f | attrs %#x blend %#x ds %02x%02x%02x poly %02x%02x chmask %08x offset %x %g %g %g\n",
        st.viewport[0][0], st.viewport[0][1], st.viewport[0][2], st.viewport[0][3], pr->vi->attr_mask, st.color_state[0], st.ds[0],
        st.ds[1], st.ds[2], st.poly[0], st.poly[1], st.channel_mask, st.poly[3], st.poly_offset[0], st.poly_offset[1],
        st.poly_offset[2]);
    // the vertex layout - stretched geometry is a mesh read with another layout: attribute <i>:f<format>/s<stream>+<offset>
    // (those the shader reads), stream <s>:<address>+<size>/<stride>
    LOG("[draw]   layout %d attribs:", st.nattrib);
    for (int i = 0; i < st.nattrib && i < 16; i++)
        if (pr->vi->attr_mask >> i & 1) LOG(" %d:f%02x/s%u+%u", i, st.attrib[i] & 0xFF, (st.attrib[i] >> 8) & 0x1F, st.attrib[i] >> 13);
    LOG(" | streams");
    for (int s = 0; s < st.nstream && s < 16; s++)
        if (st.vbuf[s].addr) LOG(" %d:%llx+%llx/%u", s, (unsigned long long)st.vbuf[s].addr, (unsigned long long)st.vbuf[s].size, st.stream[s][0]);
    LOG("\n");
    if (ENV("LARECOMP_DEBUG_DATA")) {  // vertex streams 0-1 and fragment cbufs: the first floats
        for (int s = 0; s < 2; s++) {
            if (!st.vbuf[s].addr) continue;
            const float *v = (const float *)(uintptr_t)st.vbuf[s].addr;
            LOG("[draw]   vbuf%d %llx size %llu:", s, (unsigned long long)st.vbuf[s].addr, (unsigned long long)st.vbuf[s].size);
            for (int k = 0; k < 16; k++) LOG(" %.4g", v[k]);
            LOG("\n");
        }
        for (int c = 0; c < 18; c++) {
            if (!((pr->fi->cbuf_mask >> c) & 1) || !st.ubuf[1][c].addr) continue;
            const float *u = (const float *)(uintptr_t)st.ubuf[1][c].addr;
            LOG("[draw]   fcb%d %llx size %u:", c, (unsigned long long)st.ubuf[1][c].addr, (unsigned)st.ubuf[1][c].size);
            for (int k = 0; k < 96 && k * 4 < (int)st.ubuf[1][c].size; k++) LOG(" %.6g", u[k]);
            LOG("\n");
        }
        for (int c = 0; c < 18; c++) {  // vertex cbufs (bone palettes...) and storage buffer 0
            if (!((pr->vi->cbuf_mask >> c) & 1) || c < 3 || !st.ubuf[0][c - 3].addr) continue;
            const float *u = (const float *)(uintptr_t)st.ubuf[0][c - 3].addr;
            LOG("[draw]   vcb%d %llx size %u:", c, (unsigned long long)st.ubuf[0][c - 3].addr, (unsigned)st.ubuf[0][c - 3].size);
            for (int k = 0; k < 24 && k * 4 < (int)st.ubuf[0][c - 3].size; k++) LOG(" %.5g", u[k]);
            LOG("\n");
        }
        if (st.sbuf[0][0].addr) {
            const uint32_t *w = (const uint32_t *)(uintptr_t)st.sbuf[0][0].addr;
            LOG("[draw]   vsb0 %llx size %llu:", (unsigned long long)st.sbuf[0][0].addr, (unsigned long long)st.sbuf[0][0].size);
            for (int k = 0; k < 16; k++) LOG(" %08x", w[k]);
            LOG("\n");
        }
    }
    for (int i = 0; i < pr->fi->ntex && i < 64; i++) {  // fragment textures: guest size / format, allocation, scale
        NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
        const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
        if (!t) { LOG("[draw]   tex%d: none\n", i); continue; }
        LOG("[draw]   tex%d: %p %ux%u f%#x lv %u -> %ux%u scale %.4f %.4f%s", i, (void *)t, t->width, t->height, t->format,
            b ? b->levels : 0, b ? b->width : 0, b ? b->height : 0, b ? b->sx : 0, b ? b->sy : 0, b && b->scalable ? " scalable" : "");
        {   // the pool entry behind the slot: its view (what the game asked to see of the texture: a depth / stencil
            // texture's stencil comes through a view in mode 1) and the sampler kind the shader was keyed with
            int slot = ((int)pr->fi->tex[i].handle_offset - 0x20) / 8;
            uint64_t h = (slot >= 0 && slot < 32) ? st.tex[1][slot] : 0;
            uint32_t tic = h & 0xFFFFF;
            const NvnTexEntry *e = h && st.tpool && tic < (uint32_t)st.tpool->count ? &((NvnTexEntry *)st.tpool->entries)[tic] : NULL;
            if (e && e->has_view)
                LOG(" | slot %d id %u view mask %#x fmt %#x ds %u target %u swz %u,%u,%u,%u levels %u+%u kind %d", slot, tic,
                    e->view.setmask, e->view.format, e->view.ds_mode, e->view.target, e->view.swizzle[0], e->view.swizzle[1],
                    e->view.swizzle[2], e->view.swizzle[3], e->view.base_level, e->view.num_levels, slot_kind(e));
            else if (e) LOG(" | slot %d id %u no view (texture ds %u) kind %d", slot, tic, t->ds_mode, slot_kind(e));
        }
        LOG(" vk %d\n", b ? (int)b->format : -1);
        // (investigation: Scarlet / Violet's stripes) the inputs of the full-screen passes that write the scene's HDR
        // colour - the deferred lighting and what follows it: each colour texture as its own bytes (alpha kept), the
        // first 96 of a run. build/textures/in_<draw>_tex<i>.png
        if (!indexed && r->x[2] == 3 && pr->fi->ntex >= 6 && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->format == 0x3F && b &&
            pr->fi->tex[i].type == TEXTYPE_2D && b->samples == 1 &&  // (what the saver's 2D float sampler can read)
            (t->format == 0x25 || t->format == 0x38 || t->format == 0x3F || t->format == 0x29 || t->format == 0x1)) {
            static int nsave, ndraw = -1;
            static const void *last_r;
            if (last_r != (const void *)r) { last_r = r; ndraw++; }
            if (nsave < 96) {
                char path[80];
                mkdir("build/textures", 0755);
                snprintf(path, sizeof path, "build/textures/in_%02d_tex%d.png", ndraw, i);
                nsave++;
                renderer_save_texture_tex((TexBackend *)b, path);
            }
        }
        // (investigation: Scarlet / Violet's blue hair) the same passes' stencil inputs, raw: build/textures/stencil_<i>.pgm
        if (!indexed && r->x[2] == 3 && pr->fi->ntex >= 6 && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->format == 0x3F && b &&
            pr->fi->tex[i].stencil) {
            static int nsten;
            if (nsten < 4) {
                char path[80];
                mkdir("build/textures", 0755);
                snprintf(path, sizeof path, "build/textures/stencil_%02d_tex%d.pgm", nsten++, i);
                debug_save_stencil((TexBackend *)b, path);
            }
        }
        // (the same: the lighting's environment cubes - the blue hair changes colour as the camera turns, a reflection's
        // tell) each face of their first level: build/textures/cube_<n>_tex<i>_f<face>.png (HDR values clip at 1)
        if (!indexed && r->x[2] == 3 && pr->fi->ntex >= 6 && st.rt.ncolor && st.rt.color[0] && st.rt.color[0]->format == 0x3F && b &&
            (pr->fi->tex[i].type == TEXTYPE_CUBE || pr->fi->tex[i].type == TEXTYPE_CUBE_ARRAY) && b->layers >= 6 &&
            b->samples == 1 && !(b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) {
            static int ncube;
            if (ncube < 4) {
                mkdir("build/textures", 0755);
                for (uint32_t f = 0; f < 6; f++) {
                    char path[96];
                    snprintf(path, sizeof path, "build/textures/cube_%02d_tex%d_f%u.png", ncube, i, f);
                    raw_layer = f;
                    renderer_save_texture_tex((TexBackend *)b, path);
                }
                raw_layer = 0;
                ncube++;
            }
        }
        if (t->format == 0x1 && t->width >= 256 && t->addr) {  // (R8 atlases, e.g. a glyph cache: guest bytes vs the GPU copy)
            const uint8_t *g = (const uint8_t *)(uintptr_t)t->addr;
            uint64_t n = (uint64_t)t->width * t->height, nz = 0;
            for (uint64_t k = 0; k < n; k++) nz += g[k] != 0;
            static int ndump;
            char path[64];
            snprintf(path, sizeof path, "build/tex_r8_%02d.png", ndump++);
            LOG("[draw]   tex%d guest %llx: %llu of %llu bytes nonzero -> %s\n", i, (unsigned long long)t->addr,
                (unsigned long long)nz, (unsigned long long)n, path);
            if (b) renderer_save_texture_tex((TexBackend *)b, path);
        }
        // (LARECOMP_DEBUG_TEX: the first 64 sampled textures of 256+ pixels and the first 32 sampled render targets, once each)
        static NvnTexture *saved[96];
        static int nsaved, nsaved_rt, dump = -1;
        if (dump < 0) dump = ENV("LARECOMP_DEBUG_TEX") != NULL;
        bool rt = b && b->scalable;
        if (dump && b && t->width >= 256 && t->height >= 64 && (rt ? nsaved_rt < 32 : nsaved < 64) &&
            !(b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) {
            bool seen = false;
            for (int k = 0; k < nsaved + nsaved_rt; k++) seen |= saved[k] == t;
            if (!seen) {
                char path[96];
                snprintf(path, sizeof path, "build/%s_%02d_f%x_%ux%u.png", rt ? "rt" : "tex", rt ? nsaved_rt : nsaved, t->format,
                         t->width, t->height);
                saved[nsaved + nsaved_rt] = t;
                if (rt) nsaved_rt++; else nsaved++;
                LOG("[draw]   tex%d -> %s (guest %llx, target %u)\n", i, path, (unsigned long long)t->addr, t->target);
                renderer_save_texture_tex((TexBackend *)b, path);
            }
        }
    }
}

// An indexed draw with parameters in GPU memory whose indices run past its vertex buffer (Z-A: the index count is
// another mesh's, so the draw reads on into the next allocation - matrices, material floats - and its triangles
// stretch across the screen as sheets and beams). Checked on the parameters as the CPU sees them (the culling
// compute writes only the instance count) and skipped; LARECOMP_IB_GUARD=0 draws them.
static bool ib_overrun(CmdRec *r, const uint32_t *p, uint64_t pa) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    static int on = -1;
    if (on < 0) on = !ENV("LARECOMP_IB_GUARD") || atoi(ENV("LARECOMP_IB_GUARD")) != 0;
    if (!on || !st.nattrib || !(st.attrib[0] & 0xFF)) return false;
    uint32_t s = st.attrib[0] >> 8 & 0x1F, off = st.attrib[0] >> 13;
    if (s >= (uint32_t)st.nstream || !st.vbuf[s].size || st.stream[s][1] || !st.stream[s][0]) return false;
    uint32_t itype = (uint32_t)r->x[1], isz = itype == 0 ? 1 : itype == 1 ? 2 : 4;
    uint64_t count = p[0], first = p[2], ib = r->x[2];
    if (!count || count > (1u << 22) || !ib || ib < GUEST_LO) return false;
    uint32_t restart = st.prim_restart ? st.restart_index : 0xFFFFFFFFu, mx = 0;
    const uint8_t *ip = (const uint8_t *)(uintptr_t)(ib + first * isz);
    // (the largest index, once a frame per index range: a mesh is drawn into each shadow cascade and the scene, and
    // scanning its indices each time was 2.8 % of the render thread in a town at World detail 4x. The game writes
    // a frame's indices before submitting it)
    static struct { uint64_t at, count, frame; uint32_t isz, restart, mx; } memo_t[NCTX][4096];
    __typeof__(memo_t[0][0]) *memo = memo_t[sw_ctx()];
    uint64_t at = (uint64_t)(uintptr_t)ip, frame = vk_frame_counter() + 1;
    __typeof__(memo[0]) *c = &memo[(at >> 4 ^ at >> 17 ^ count * 0x9E37u) & 4095];
    if (c->at == at && c->count == count && c->frame == frame && c->isz == isz && c->restart == restart) mx = c->mx;
    else {
        for (uint64_t i = 0; i < count; i++) {
            uint32_t v = isz == 1 ? ip[i] : isz == 2 ? ((const uint16_t *)ip)[i] : ((const uint32_t *)ip)[i];
            if (v != (restart & (isz == 1 ? 0xFFu : isz == 2 ? 0xFFFFu : 0xFFFFFFFFu)) && v > mx) mx = v;
        }
        *c = (__typeof__(*c)){at, count, frame, isz, restart, mx};
    }
    int64_t last = (int64_t)mx + (int32_t)p[3];
    uint64_t stride = st.stream[s][0];
    if (last < 0 || (uint64_t)last * stride + off + 4 <= st.vbuf[s].size) return false;
    static int nlog;
    if (nlog++ < 60) {
        LOG("[ibguard] frame %llu: skipped draw: args %llx = count %u inst %u first %u base %d binst %u | "
            "ib %llx max index %u -> vertex %lld x stride %llu > vbuf %llx+%llu (%llu vertices)\n",
            (unsigned long long)renderer_present_count(), (unsigned long long)pa, p[0], p[1], p[2], (int)p[3], p[4], (unsigned long long)ib, mx, (long long)last,
            (unsigned long long)stride, (unsigned long long)st.vbuf[s].addr, (unsigned long long)st.vbuf[s].size,
            (unsigned long long)(st.vbuf[s].size / stride));
    }
    gw_explain(pa);
    stat_skipped_here();
    return true;
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}
// Parameters no draw can have, in blocks the GPU reads in place (#70 #81 #82 #86: the GPU stopped for 10 s, or the
// device was lost, where blocks held other data for frames on end - every field near one small number, a base
// instance in the hundreds of thousands): a count over 4M (ib_overrun leaves those alone), or a base instance whose
// first element already lies past a per-instance stream's buffer (the instance attributes read far beyond it).
// Checked as the CPU sees them, as ib_overrun (the base instance is the CPU's; only the instance count is the GPU's),
// and skipped; LARECOMP_IB_GUARD=0 draws them.
static bool ind_args_bad(const uint32_t *p, uint64_t pa, bool elements) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    static int on = -1;
    if (on < 0) on = !ENV("LARECOMP_IB_GUARD") || atoi(ENV("LARECOMP_IB_GUARD")) != 0;
    if (!on) return false;
    uint32_t binst = elements ? p[4] : p[3];
    const char *why = p[0] > (1u << 22) ? "count over 4M" : NULL;
    for (int a = 0; a < st.nattrib && !why; a++) {
        if (!(st.attrib[a] & 0xFF)) continue;
        uint32_t s = st.attrib[a] >> 8 & 0x1F, off = st.attrib[a] >> 13;
        if (s >= (uint32_t)st.nstream || !st.stream[s][1] || !st.stream[s][0] || !st.vbuf[s].size) continue;
        if ((uint64_t)binst * st.stream[s][0] + off + 4 > st.vbuf[s].size) why = "base instance past an instance stream";
    }
    if (!why) return false;
    static int nlog;
    if (nlog++ < 60)
        LOG("[ibguard] frame %llu: skipped draw (%s): args %llx = %u %u %u %d %u\n", (unsigned long long)renderer_present_count(),
            why, (unsigned long long)pa, p[0], p[1], p[2], (int)p[3], elements ? p[4] : 0);
    gw_explain(pa);
    stat_skipped_here();
    return true;
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}
// Indirect draws: the parameters (GL / Vulkan layout: arrays {count, instances, first, base instance}, elements
// {count, instances, first index, base vertex, base instance}) read from guest memory when the list runs, each made a
// direct draw. Arguments: DrawArraysIndirect (mode, indirect), DrawElementsIndirect (mode, index type, index buffer,
// indirect), MultiDrawArraysIndirectCount (mode, indirect, count buffer, max draws, stride), MultiDrawElementsIndirect-
// Count (mode, index type, index buffer, indirect, count buffer, max draws, stride). (Parameters a compute pass of the
// same submission writes are not ready yet here: they come from the previous one.)
static void draw_indirect(CmdRec *r) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    bool elements = r->op == CMD_DRAW_ELEMENTS_INDIRECT || r->op == CMD_MULTI_DRAW_ELEMENTS_INDIRECT_COUNT;
    bool multi = r->op == CMD_MULTI_DRAW_ARRAYS_INDIRECT_COUNT || r->op == CMD_MULTI_DRAW_ELEMENTS_INDIRECT_COUNT;
    int k = elements ? 3 : 1;  // the indirect buffer's argument
    uint64_t ind = r->x[k];
    uint32_t n = 1, stride = elements ? 20 : 16;
    if (multi) {
        uint64_t cnt = r->x[k + 1];
        n = (uint32_t)r->x[k + 2];
        if (cnt && cnt >= GUEST_LO) { uint32_t c = *(const uint32_t *)(uintptr_t)cnt; if (c < n) n = c; }
        if (r->x[k + 3]) stride = (uint32_t)r->x[k + 3];
    }
    static int nlog;
    static uint32_t seen_ops;
    if (nlog < 8 || !(seen_ops & (1u << (r->op & 31)))) {
        nlog++;
        seen_ops |= 1u << (r->op & 31);
        LOG("[vk] indirect draw op %u: %u draw(s) at %llx stride %u (count buffer %llx max %llu)\n", r->op, n, (unsigned long long)ind,
            stride, (unsigned long long)(multi ? r->x[k + 1] : 0), (unsigned long long)(multi ? r->x[k + 2] : 0));
    }
    if (!ind || ind < GUEST_LO || n > 4096) return;
    CmdRec d;
    memset(&d, 0, sizeof d);
    d.op = elements ? CMD_DRAW_ELEMENTS : CMD_DRAW_ARRAYS;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t *p = (const uint32_t *)(uintptr_t)(ind + (uint64_t)i * stride);
        static int nargs;
        if (nargs < 400 && ENV("LARECOMP_LOG_INDIRECT")) {
            nargs++;
            LOG("[vk] indirect %llx: %u %u %u %u %u\n", (unsigned long long)(ind + (uint64_t)i * stride), p[0], p[1], p[2], p[3],
                elements ? p[4] : 0);
        }
        uint64_t pa = ind + (uint64_t)i * stride;
        gmem_info();  // (the import, made on first use)
        bool gpu = GM_IN(pa, 20) && !(pa & 3);
        if (gpu) {
            // The GPU fills in parts of the parameters (the instance count: Pokémon Z-A's culling compute), so the
            // draw reads them there; the CPU side covers the index range from the buffer's start (the first index
            // applies on the GPU) with one instance
            static int gpu_args_log = -1;  // (debug) LARECOMP_LOG_GPU_ARGS=<n>: n parameter sets as the GPU sees them
            if (gpu_args_log < 0) gpu_args_log = ENV("LARECOMP_LOG_GPU_ARGS") ? atoi(ENV("LARECOMP_LOG_GPU_ARGS")) : 0;
            if (gpu_args_log > 0 && renderer_present_count() > 600) {
                gpu_args_log--;
                vk_end_pass();
                VkSpan rb = vk_readback(32);
                VkBufferCopy bc = {GM_OFF(pa), rb.off, 20};
                vk_full_barrier(vk_cmd());
                rec_CmdCopyBuffer(vk_cmd(), gmem_buf, rb.buf, 1, &bc);
                renderer_flush();
                const uint32_t *q = rb.ptr;
                LOG("[vk] gpu args %llx: count %u instances %u first %u base %d base inst %u (cpu: %u %u)\n", (unsigned long long)pa,
                    q[0], q[1], q[2], (int)q[3], q[4], p[0], p[1]);
            }
            if (ind_args_bad(p, pa, elements) || (elements && ib_overrun(r, p, pa))) continue;
            ir_note(pa, p, st.program ? st.program->stages[0].code_addr : 0);
            g_ind_gpu = true; g_ind_off = GM_OFF(pa); g_ind_stride = stride;
            d.x[0] = r->x[0];
            if (elements) {
                d.x[1] = r->x[1]; d.x[2] = p[0] + p[2]; d.x[3] = r->x[2];
                d.x[4] = (uint64_t)(int64_t)(int32_t)p[3]; d.x[5] = p[4]; d.x[6] = 1;
            } else {
                d.x[1] = 0; d.x[2] = p[0] + p[2]; d.x[3] = p[3]; d.x[4] = 1;
            }
            if (d.x[2]) draw(&d, elements);
            g_ind_gpu = false;
            continue;
        }
        if (!p[0] || !p[1]) continue;  // (no vertices or no instances)
        d.x[0] = r->x[0];  // mode
        if (elements) {  // x: mode, index type, count, index buffer, base vertex, base instance, instances
            d.x[1] = r->x[1]; d.x[2] = p[0];
            d.x[3] = r->x[2] + (uint64_t)p[2] * (r->x[1] == 0 ? 1 : r->x[1] == 1 ? 2 : 4);  // first index (NVN index type 0/1/2: 8/16/32-bit)
            d.x[4] = (uint64_t)(int64_t)(int32_t)p[3]; d.x[5] = p[4]; d.x[6] = p[1];
        } else {  // x: mode, first, count, base instance, instances
            d.x[1] = p[2]; d.x[2] = p[0]; d.x[3] = p[3]; d.x[4] = p[1];
        }
        draw(&d, elements);
    }
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// ================================================================== texture copies
typedef struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; } TexRefRec;

static void copy_region_dims(NvnTexture *t, const TexRefRec *ref, const FormatInfo **fi, int32_t rg[6], int *level,
                             uint64_t *row, uint64_t *img) {
    *fi = nvn_format_info(t->format);
    memcpy(rg, ref->region, sizeof ref->region);
    if (!rg[3]) rg[3] = (int32_t)t->width;
    if (!rg[4]) rg[4] = (int32_t)t->height;
    if (!rg[5]) rg[5] = 1;
    *level = ref->has_view && (ref->view.setmask & 1) ? ref->view.base_level : 0;
    uint32_t wb = ((uint32_t)rg[3] + (*fi)->bw - 1) / (*fi)->bw, hb = ((uint32_t)rg[4] + (*fi)->bh - 1) / (*fi)->bh;
    *row = st.copy_row_stride ? st.copy_row_stride : (uint64_t)wb * (*fi)->bpb;
    *img = st.copy_image_stride ? st.copy_image_stride : *row * hb;
}

static void copy_buffer_to_texture(uint64_t src, const TexRefRec *dst) {
    snap_reason = "buf2tex"; snapshots_invalidate(NULL);
    NvnTexture *t = (NvnTexture *)(uintptr_t)dst->tex;
    if (!t || !src) return;
    alias_before_write(t, 0, "upload");
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, dst, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb) return;
    TexBackend *b = vk_tex(t, false);  // prior CPU contents uploaded first
    renderer_mark_gpu_owned(t);
    if (b->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) return;
    if (b->astc_bw) {  // (blocks would need the CPU decode: not seen in the game yet)
        static int nlog;
        if (nlog++ < 5) LOG("[vk] buffer -> ASTC texture copy skipped (%ux%u)\n", t->width, t->height);
        return;
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t hb = ((uint32_t)rg[4] + fi->bh - 1) / fi->bh, wb = ((uint32_t)rg[3] + fi->bw - 1) / fi->bw;
    uint32_t out_bpb = fi->bpb == 3 ? 4 : fi->bpb;
    uint32_t nz = (uint32_t)rg[5];
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    uint64_t out_row = (uint64_t)wb * out_bpb;
    VkSpan s = vk_transient(out_row * hb * nz, 16);
    for (uint32_t z = 0; z < nz; z++)
        for (uint32_t y = 0; y < hb; y++) {
            const uint8_t *sr = (const uint8_t *)(uintptr_t)(src + (uint64_t)z * img + (uint64_t)y * row);
            uint8_t *dr = (uint8_t *)s.ptr + ((uint64_t)z * hb + y) * out_row;
            if (out_bpb == fi->bpb) memcpy(dr, sr, out_row);
            else for (uint32_t x = 0; x < wb; x++) { memcpy(dr + x * 4, sr + x * 3, 3); dr[x * 4 + 3] = 255; }
        }
    VkBufferImageCopy c = {.bufferOffset = s.off, .bufferRowLength = wb * fi->bw, .bufferImageHeight = hb * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    rec_CmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

static void copy_texture_to_buffer(const TexRefRec *src, uint64_t dst) {
    NvnTexture *t = (NvnTexture *)(uintptr_t)src->tex;
    if (!t || !dst || dst < GUEST_LO) return;
    t = alias_read(t);
    vk_end_pass();
    const FormatInfo *fi;
    int32_t rg[6];
    int level;
    uint64_t row, img;
    copy_region_dims(t, src, &fi, rg, &level, &row, &img);
    if (!fi || !fi->bpb || fi->bpb == 3 || row % fi->bpb) return;
    TexBackend *b = vk_tex(t, false);
    float tsx, tsy;
    renderer_texture_scale(t, &tsx, &tsy);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImage srcimg = b->image;
    VmaAllocation small_mem = NULL;
    if ((tsx != 1 || tsy != 1) && level == 0 && !renderer_format_is_depth(t->format)) {
        // the CPU expects guest-sized texels: filter the scaled allocation down first
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = b->format,
                                .extent = {t->width, t->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &srcimg, &small_mem, NULL));
        vk_image_init_layout(cb, srcimg, VK_IMAGE_ASPECT_COLOR_BIT);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)lroundf(t->width * tsx), (int32_t)lroundf(t->height * tsy), 1}},  // (the contents)
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)t->width, (int32_t)t->height, 1}}};
        rec_CmdBlitImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, srcimg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
        vk_full_barrier(cb);
    }
    bool array = b->layers > 1 && b->depth == 1;
    uint32_t nz = (uint32_t)rg[5];
    uint64_t total = img * nz;
    VkSpan out = vk_readback(total);
    VkBufferImageCopy c = {.bufferOffset = out.off, .bufferRowLength = (uint32_t)(row / fi->bpb) * fi->bw,
                           .bufferImageHeight = (uint32_t)(img / row) * fi->bh,
                           .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect,
                                                (uint32_t)level, array ? (uint32_t)rg[2] : 0, array ? nz : 1},
                           .imageOffset = {rg[0], rg[1], array ? 0 : rg[2]},
                           .imageExtent = {(uint32_t)rg[3], (uint32_t)rg[4], array ? 1 : nz}};
    rec_CmdCopyImageToBuffer(cb, srcimg, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    wait_why = "texture -> buffer readback";
    renderer_flush();  // the CPU may read the result right after the fence
    memcpy((void *)(uintptr_t)dst, out.ptr, total);
    if (small_mem) vmaDestroyImage(g_vk.vma, srcimg, small_mem);
}

// Scaled depth copies (the game's depth pyramid) are blits, but a depth format needn't support being a blit destination:
// Intel's drivers don't for D32_SFLOAT_S8_UINT (validation on a UHD 770; a suspect in the UHD 630's crash after a
// depth pass, issues #8, #46). There the depth goes the long way round with transfers only: to a buffer, into an
// unsigned-integer colour image with the same bits per texel, blitted (nearest: bits kept) into a second one at the
// new size, back to the buffer, and into the depth image. The regions are copy_texture's, already cut to the
// contents' extent (Dynamic resolution's sub-rect, issue #191's content_extent) and kept inside their levels.
// The scratch buffer and images are sized by the source and destination *levels* (not the regions, which follow the
// render scale), so a changing scale doesn't regrow them; outgrown ones are destroyed once the submissions that may
// use them are done. All of it (and depth_blit_ok's table) belongs to the translating thread: the render thread, or
// the game's GPU thread with LARECOMP_RENDER_THREAD=0 - copy_texture runs nowhere else.
static bool depth_blit_ok(VkFormat f) {
    static VkFormat known[8];
    static bool ok[8];
    static int n;
    for (int i = 0; i < n; i++) if (known[i] == f) return ok[i];
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &fp);
    bool r = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
    if (n < 8) { known[n] = f; ok[n++] = r; }
    if (!r) LOG("[vk] depth format %d can't be blitted here: scaled depth copies go through a colour image\n", f);
    return r;
}

typedef struct { VkImage img; VmaAllocation mem; uint32_t w, h; } ScratchImage;

static bool scratch_image(ScratchImage *si, uint32_t w, uint32_t h, VkFormat fmt) {
    if (si->img && si->w >= w && si->h >= h) return true;
    if (si->w > w) w = si->w;
    if (si->h > h) h = si->h;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImage img;
    VmaAllocation mem;
    if (vmaCreateImage(g_vk.vma, &ci, &ai, &img, &mem, NULL) != VK_SUCCESS) return false;
    if (si->img) vk_defer_image(si->img, si->mem);  // (no views: never sampled or rendered to)
    vk_image_init_layout(vk_upload_cmd(), img, VK_IMAGE_ASPECT_COLOR_BIT);
    *si = (ScratchImage){img, mem, w, h};
    return true;
}

static void depth_blit_via_color(VkCommandBuffer cb, TexBackend *s, TexBackend *d, uint32_t slv, uint32_t sly, int sx, int sy,
                                 int sw, int sh, uint32_t dlv, uint32_t dly, int dx, int dy, int dw, int dh) {
    if (s->samples != 1 || d->samples != 1) return;  // (buffer copies and blits are single-sample only, as before)
    // bits per texel of the depth aspect in a buffer: 4 (D32 / X8_D24 variants) or 2 (D16)
    uint32_t bpp = s->format == VK_FORMAT_D16_UNORM || s->format == VK_FORMAT_D16_UNORM_S8_UINT ? 2 : 4;
    VkFormat cf = bpp == 2 ? VK_FORMAT_R16_UINT : VK_FORMAT_R32_UINT;
    static VkBuffer buf;
    static VmaAllocation buf_mem;
    static VkDeviceSize buf_size;
    static struct { VkBuffer buf; VmaAllocation mem; uint64_t seq; } old[8];  // outgrown buffers, until their work is done
    static ScratchImage a[2], b[2];  // per colour format (R16 / R32)
    uint64_t done = vk_completed_seq();
    for (int i = 0; i < 8; i++)
        if (old[i].buf && old[i].seq <= done) { vmaDestroyBuffer(g_vk.vma, old[i].buf, old[i].mem); old[i].buf = VK_NULL_HANDLE; }
    uint32_t slw = level_extent(s->width, slv), slh = level_extent(s->height, slv);
    uint32_t dlw = level_extent(d->width, dlv), dlh = level_extent(d->height, dlv);
    VkDeviceSize need = (VkDeviceSize)(slw * slh > dlw * dlh ? slw * slh : dlw * dlh) * bpp;
    if (need > buf_size) {
        if (need < buf_size * 2) need = buf_size * 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return;
        if (buf) {
            int i = 0;
            while (i < 8 && old[i].buf) i++;
            if (i < 8) old[i].buf = buf, old[i].mem = buf_mem, old[i].seq = vk_pending_seq();
            // (else kept, never freed: 8 outgrown in flight at once, with each at least twice the last - not seen)
        }
        buf = nb; buf_mem = nm; buf_size = need;
    }
    int k = bpp == 2 ? 0 : 1;
    if (!scratch_image(&a[k], slw, slh, cf) || !scratch_image(&b[k], dlw, dlh, cf)) return;
    VkBufferImageCopy c = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, slv, sly, 1}, .imageOffset = {sx, sy, 0},
                           .imageExtent = {(uint32_t)sw, (uint32_t)sh, 1}};
    rec_CmdCopyImageToBuffer(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &c);
    vk_full_barrier(cb);
    c.imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageOffset = (VkOffset3D){0, 0, 0};
    rec_CmdCopyBufferToImage(cb, buf, a[k].img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    vk_full_barrier(cb);
    VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {sw, sh, 1}},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {dw, dh, 1}}};
    rec_CmdBlitImage(cb, a[k].img, VK_IMAGE_LAYOUT_GENERAL, b[k].img, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
    vk_full_barrier(cb);
    c.imageExtent = (VkExtent3D){(uint32_t)dw, (uint32_t)dh, 1};
    rec_CmdCopyImageToBuffer(cb, b[k].img, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &c);
    vk_full_barrier(cb);
    c.imageSubresource = (VkImageSubresourceLayers){VK_IMAGE_ASPECT_DEPTH_BIT, dlv, dly, 1};
    c.imageOffset = (VkOffset3D){dx, dy, 0};
    rec_CmdCopyBufferToImage(cb, buf, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
}

static void copy_texture_gpu(const TexRefRec *src, const TexRefRec *dst, uint64_t flags);
// A copy into a pitch-linear texture (a stride: memory the CPU reads as it is) also lands in guest memory, read back
// with the texture's own row pitch. Royale Z-A's profile photo: the frame is copied into a 1920x1080 linear RGBA8
// texture, the CPU maps that memory, crops it to 852x444 and encodes it to BC1 itself (game fn 0x700189a880,
// blocks by 0x700189bfb0): with the copy only in the Vulkan image the CPU read zeros - a black photo, saved black
// (#64). Rare (one copy per photo), so the wait for the GPU is fine. LARECOMP_VK_LINEAR_WRITEBACK=0: the image only.
static void copy_texture(const TexRefRec *src, const TexRefRec *dst, uint64_t flags) {
    copy_texture_gpu(src, dst, flags);
    NvnTexture *dt = (NvnTexture *)(uintptr_t)dst->tex;
    static int on = -1;
    if (on < 0) on = !ENV("LARECOMP_VK_LINEAR_WRITEBACK") || atoi(ENV("LARECOMP_VK_LINEAR_WRITEBACK")) != 0;
    if (!on || !dt || !dt->stride || dt->addr < GUEST_LO || (dst->has_view && (dst->view.setmask & 1) && dst->view.base_level))
        return;
    const FormatInfo *fi = nvn_format_info(dt->format);
    if (!fi || fi->bw != 1 || fi->bh != 1 || !fi->bpb || fi->bpb == 3 || dt->stride < (uint64_t)dt->width * fi->bpb) return;
    TexRefRec whole = *dst;  // (the whole image: the rows the CPU sees, at the texture's pitch)
    memset(whole.region, 0, sizeof whole.region);
    uint64_t row = st.copy_row_stride, img = st.copy_image_stride;
    st.copy_row_stride = dt->stride, st.copy_image_stride = dt->stride * dt->height;
    copy_texture_to_buffer(&whole, dt->addr);
    st.copy_row_stride = row, st.copy_image_stride = img;
}

static void copy_texture_gpu(const TexRefRec *src, const TexRefRec *dst, uint64_t flags) {
    snap_reason = "tex2tex"; snapshots_invalidate(NULL);
    NvnTexture *st_ = (NvnTexture *)(uintptr_t)src->tex, *dt = (NvnTexture *)(uintptr_t)dst->tex;
    if (!st_ || !dt) return;
    vk_end_pass();
    if (dl_on == 2) renderer_flush();  // (LARECOMP_VK_DEVLOST_TRACE=2: each copy its own submission, waited on)
    gp_mark(vk_cmd(), "copy texture %ux%u -> %ux%u", st_->width, st_->height, dt->width, dt->height);
    dl_note("copy texture %ux%u f%#x -> %ux%u f%#x (%p -> %p)", st_->width, st_->height, st_->format, dt->width, dt->height,
            dt->format, (void *)st_, (void *)dt);
    TexBackend *s = vk_tex(st_, false), *d = vk_tex(dt, true);
    renderer_mark_gpu_owned(dt);
    if (renderer_texture_presented(dt) && st_->width == dt->width && st_->height == dt->height) renderer_texture_mark_presented(st_);
    renderer_texture_first_write(dt);
    renderer_texture_level_fresh(dt);
    const int32_t *sr = src->region, *dr = dst->region;
    float ssx, ssy, dsx, dsy;
    renderer_texture_scale(st_, &ssx, &ssy);
    renderer_texture_scale(dt, &dsx, &dsy);
    int sx0 = (int)lroundf(sr[0] * ssx), sy0 = (int)lroundf(sr[1] * ssy);
    int dx0 = (int)lroundf(dr[0] * dsx), dy0 = (int)lroundf(dr[1] * dsy);
    int sw = sr[3] ? (int)lroundf(sr[3] * ssx) : (int)lroundf(st_->width * ssx), sh = sr[4] ? (int)lroundf(sr[4] * ssy) : (int)lroundf(st_->height * ssy);
    int dw = dr[3] ? (int)lroundf(dr[3] * dsx) : (int)lroundf(dt->width * dsx), dh = dr[4] ? (int)lroundf(dr[4] * dsy) : (int)lroundf(dt->height * dsy);
    uint32_t slv = src->has_view && (src->view.setmask & 1) ? src->view.base_level : 0;
    uint32_t dlv = dst->has_view && (dst->view.setmask & 1) ? dst->view.base_level : 0;
    uint32_t sly = src->has_view && (src->view.setmask & 2) ? src->view.min_layer : 0;
    uint32_t dly = dst->has_view && (dst->view.setmask & 2) ? dst->view.min_layer : 0;
    if (slv >= s->levels || dlv >= d->levels || sly >= s->layers || dly >= d->layers) return;
    bool full_movie_copy = false;
    {   // whole-image regions are the contents' extent in the level, and every region stays inside its level: scaled
        // allocations (internal resolution below 1x) round guest regions past the edge, and out-of-bounds copies /
        // blits are undefined in Vulkan (garbage that changed every frame on NVIDIA: flicker in the depth pyramid's
        // users). Dynamic resolution draws into the top-left part of the allocation: the rest holds older frames, and
        // a whole-image copy of it into a texture of another scale (the save slot's screenshot, 192x108) took those
        // along (the frame small in a corner, stale frames around it)
        int slw = (int)level_extent(s->width, slv), slh = (int)level_extent(s->height, slv);
        int dlw = (int)level_extent(d->width, dlv), dlh = (int)level_extent(d->height, dlv);
        int scw = content_extent(slw, s->scalable, s->sx, s->ax), sch = content_extent(slh, s->scalable, s->sy, s->ay);
        int dcw = content_extent(dlw, d->scalable, d->sx, d->ax), dch = content_extent(dlh, d->scalable, d->sy, d->ay);
        bool same_part = slw == dlw && slh == dlh && scw == dcw && sch == dch;  // (contents in the same part of equal images)
        if (!sr[3]) sw = scw - sx0;
        if (!sr[4]) sh = sch - sy0;
        if (!dr[3]) dw = dcw - dx0;
        if (!dr[4]) dh = dch - dy0;
        if (same_part && sx0 == 0 && sy0 == 0 && dx0 == 0 && dy0 == 0 && sw == scw && sh == sch && dw == dcw && dh == dch)
            sw = dw = slw, sh = dh = slh;  // (the same part of images of one size: the whole image, which may alias)
        if (sx0 + sw > slw) sw = slw - sx0;
        if (sy0 + sh > slh) sh = slh - sy0;
        if (dx0 + dw > dlw) dw = dlw - dx0;
        if (dy0 + dh > dlh) dh = dlh - dy0;
        if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return;
        full_movie_copy = movie_copy_is_full(slv, dlv, sly, dly, sx0, sy0, dx0, dy0, sw, sh, dw, dh, scw, sch, dcw, dch);
    }
    if (g_debug_frame || ct_on())
        LOG("[copytex] %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] -> %p %ux%u f%#x lv%u (%d,%d %dx%d) [%ux%u] phys (%d,%d %dx%d) -> (%d,%d %dx%d) flags %llx\n",
            (void *)st_, st_->width, st_->height, st_->format, slv, sr[0], sr[1], sr[3], sr[4], s->width, s->height, (void *)dt, dt->width,
            dt->height, dt->format, dlv, dr[0], dr[1], dr[3], dr[4], d->width, d->height, sx0, sy0, sw, sh, dx0, dy0, dw, dh,
            (unsigned long long)flags);
    const TexBackend *alias_src = full_movie_copy ? (naliases ? (const TexBackend *)alias_read(st_)->backend : s) : NULL;
    if (copy_opt() && sx0 == 0 && sy0 == 0 && dx0 == 0 && dy0 == 0 && sw == (int)s->width && sh == (int)s->height && dw == (int)d->width &&
        dh == (int)d->height && slv == 0 && dlv == 0 && sly == 0 && dly == 0 && alias_make(st_, dt, s, d)) {
        if (full_movie_copy && (s->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && (d->aspect & VK_IMAGE_ASPECT_COLOR_BIT) &&
            !((s->aspect | d->aspect) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)))
            movie_copy_propagate(d, alias_src);
        if (ct_on()) LOG("[ct] alias made %p -> %p\n", (void *)st_, (void *)dt);
        return;
    }
    alias_before_write(dt, 0, "copy dst");
    if (naliases) s = alias_read(st_)->backend;
    if (full_movie_copy && movie_copy_can_execute(s->aspect, d->aspect, s->samples, d->samples, s->format, d->format, sw, sh, dw, dh))
        movie_copy_propagate(d, s);
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageAspectFlags sa = s->aspect, da = d->aspect;
    if (s->samples > 1 && d->samples == 1 && sw == dw && sh == dh && s->format == d->format && !(sa & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        VkImageResolve rv = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                             .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        rec_CmdResolveImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rv);
        return;
    }
    if (s->samples != d->samples) return;
    if (sw == dw && sh == dh && s->format == d->format) {
        VkImageCopy c = {.srcSubresource = {sa, slv, sly, 1}, .srcOffset = {sx0, sy0, 0}, .dstSubresource = {da, dlv, dly, 1},
                         .dstOffset = {dx0, dy0, 0}, .extent = {(uint32_t)sw, (uint32_t)sh, 1}};
        rec_CmdCopyImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        return;
    }
    // scaled / converting copy: the game builds its depth pyramid this way (nearest), colour copies filter if asked
    bool depth = (sa | da) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (depth && s->format != d->format) return;
    static int via_color = -1;  // LARECOMP_VK_DEPTH_BLIT_VIA_COLOR=1: always (tests on GPUs that can blit depth)
    if (via_color < 0) via_color = getenv("LARECOMP_VK_DEPTH_BLIT_VIA_COLOR") ? atoi(getenv("LARECOMP_VK_DEPTH_BLIT_VIA_COLOR")) : 0;
    if (depth && (sa & da & VK_IMAGE_ASPECT_DEPTH_BIT) && (via_color || !depth_blit_ok(d->format) || !depth_blit_ok(s->format))) {
        dl_note("  depth blit via colour %p lv%u (%d,%d %dx%d) -> %p lv%u (%d,%d %dx%d)", (void *)s->image, slv, sx0, sy0, sw, sh,
                (void *)d->image, dlv, dx0, dy0, dw, dh);
        depth_blit_via_color(cb, s, d, slv, sly, sx0, sy0, sw, sh, dlv, dly, dx0, dy0, dw, dh);
        return;
    }
    VkImageBlit bl = {.srcSubresource = {depth ? (sa & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : sa) : sa, slv, sly, 1},
                      .srcOffsets = {{sx0, sy0, 0}, {sx0 + sw, sy0 + sh, 1}},
                      .dstSubresource = {depth ? (da & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : da) : da, dlv, dly, 1},
                      .dstOffsets = {{dx0, dy0, 0}, {dx0 + dw, dy0 + dh, 1}}};
    bool linear = (flags & 1) && !depth && !rt_kind(st_->format);
    dl_note("  blit %p lv%u/%u %ux%u (%d,%d %dx%d) -> %p lv%u/%u %ux%u (%d,%d %dx%d) %s", (void *)s->image, slv, s->levels, s->width,
            s->height, sx0, sy0, sw, sh, (void *)d->image, dlv, d->levels, d->width, d->height, dx0, dy0, dw, dh, linear ? "linear" : "nearest");
    rec_CmdBlitImage(cb, s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
}

// ================================================================== present helpers (device.c)
static VkPipelineLayout present_layout;
static VkDescriptorSetLayout present_set_layout;

// ---- decoded video into its texture (textures.c renderer_upload_nv12 / upload_pending, render thread): the NV12
// planes go into two images (R8, R8G8: reused while the size stays), and a fullscreen draw converts them into the
// texture through a UNORM view (the bytes as the RGBA upload wrote them: no sRGB encoding). Recorded into the upload
// command buffer, ahead of the frame that samples the texture.
static bool nv12_plane(TexBackend *p, VkFormat f, uint32_t w, uint32_t h, VkCommandBuffer cb) {
    if (p->image) vk_views_forget(p);  // (a new size: the old image is left to the driver's teardown - one per movie size)
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    *p = (TexBackend){0};
    if (vmaCreateImage(g_vk.vma, &ci, &ai, &p->image, &p->mem, NULL) != VK_SUCCESS) { p->image = VK_NULL_HANDLE; return false; }
    vk_image_init_layout(cb, p->image, VK_IMAGE_ASPECT_COLOR_BIT);
    p->format = f; p->width = w; p->height = h; p->depth = 1; p->levels = p->layers = p->samples = 1;
    p->aspect = VK_IMAGE_ASPECT_COLOR_BIT; p->base_type = VK_IMAGE_VIEW_TYPE_2D; p->gpu_owned = true;
    return true;
}

void vk_convert_nv12(TexBackend *b, const void *nv12, uint32_t w, uint32_t h, uint32_t stride, uint32_t uv_off) {
    static TexBackend py, puv;
    static VkDescriptorSetLayout dsl;
    static VkPipelineLayout layout;
    static VkPipeline pipes[2];  // RGBA8, BGRA8 targets
    VkCommandBuffer cb = vk_upload_cmd();
    bool bgra = b->format == VK_FORMAT_B8G8R8A8_UNORM || b->format == VK_FORMAT_B8G8R8A8_SRGB;
    VkFormat raw = bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    if (!layout) {
        VkDescriptorSetLayoutBinding bd[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
                                              {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL}};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bd};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &dsl));
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &layout));
        LOG("[movie] video frames converted to RGBA on the GPU\n");
    }
    if (!pipes[bgra]) {
        VkPipelineShaderStageCreateInfo stages[2] = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
             .module = vk_builtin_shader("nv12.vert"), .pName = "main"},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
             .module = vk_builtin_shader("nv12.frag"), .pName = "main"}};
        VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                     .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
                                                .scissorCount = 1};
        VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                     .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
        VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
        VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                                   .pAttachments = &cba};
        VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        uint32_t ndyn = 2;
        fullscreen_vertex_dyn(dyn, &ndyn, 3);
        VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = ndyn,
                                               .pDynamicStates = dyn};
        VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                            .pColorAttachmentFormats = &raw};
        VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                           .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
                                           .pViewportState = &vs, .pRasterizationState = &rs, .pMultisampleState = &ms,
                                           .pDepthStencilState = &ds, .pColorBlendState = &cbs, .pDynamicState = &dy, .layout = layout};
        if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &pipes[bgra]) != VK_SUCCESS) {
            LOG("[movie] NV12 pipeline failed\n");
            pipes[bgra] = VK_NULL_HANDLE;
            return;
        }
    }
    if (py.width != w || py.height != h || !py.image)
        if (!nv12_plane(&py, VK_FORMAT_R8_UNORM, w, h, cb) || !nv12_plane(&puv, VK_FORMAT_R8G8_UNORM, w / 2, h / 2, cb)) return;
    VkComponentMapping ident = {0};  // (VK_COMPONENT_SWIZZLE_IDENTITY)
    VkImageView vy = vk_image_view(&py, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8_UNORM, 0, 1, 0, 1, ident, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageView vuv = vk_image_view(&puv, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8_UNORM, 0, 1, 0, 1, ident, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageView vt = vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, raw, 0, 1, 0, 1, ident, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!vy || !vuv || !vt) return;
    VkSpan s = vk_transient_copy(nv12, (uint64_t)uv_off + (uint64_t)stride * (h / 2), 16);
    VkBufferImageCopy c[2] = {
        {.bufferOffset = s.off, .bufferRowLength = stride, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {w, h, 1}},
        {.bufferOffset = s.off + uv_off, .bufferRowLength = stride / 2, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {w / 2, h / 2, 1}}};
    vk_full_barrier(cb);
    rec_CmdCopyBufferToImage(cb, s.buf, py.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c[0]);
    rec_CmdCopyBufferToImage(cb, s.buf, puv.image, VK_IMAGE_LAYOUT_GENERAL, 1, &c[1]);
    vk_full_barrier(cb);
    VkDescriptorSet set = vk_alloc_set(dsl);
    VkDescriptorImageInfo ii[2] = {{vk_default_sampler(true), vy, VK_IMAGE_LAYOUT_GENERAL},
                                   {vk_default_sampler(true), vuv, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[1]}};
    rec_UpdateDescriptorSets(2, wr);
    uint32_t rw = w < b->width ? w : b->width, rh = h < b->height ? h : b->height;
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = vt,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {rw, rh}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    rec_CmdBeginRendering(cb, &ri);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipes[bgra]);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    VkViewport vp = {0, 0, (float)rw, (float)rh, 0, 1};
    VkRect2D sc = {{0, 0}, {rw, rh}};
    rec_CmdSetViewport(cb, 0, 1, &vp);
    rec_CmdSetScissor(cb, 0, 1, &sc);
    fullscreen_vertex_bind(cb);
    rec_CmdDraw(cb, 3, 1, 0, 0);
    rec_CmdEndRendering(cb);
    vk_full_barrier(cb);
}

// keep_alpha: texcopy.frag (the texture's alpha) instead of present.frag (opaque)
static VkPipeline present_pipeline_ex(VkFormat fmt, bool keep_alpha) {
    static struct { VkFormat f; bool a; VkPipeline p; } cache[8];
    for (int i = 0; i < 8; i++) if (cache[i].p && cache[i].f == fmt && cache[i].a == keep_alpha) return cache[i].p;
    if (!present_layout) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &present_set_layout));
        // (vertex: flip + crop, 32 bytes; fragment: the screen filter, 16 more)
        VkPushConstantRange r = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 48};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &present_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &present_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("present.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(keep_alpha ? "texcopy.frag" : "present.frag"), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    uint32_t ndyn = 2;
    fullscreen_vertex_dyn(dyn, &ndyn, 3);
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = ndyn, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = present_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 8; i++) if (!cache[i].p) { cache[i].f = fmt; cache[i].a = keep_alpha; cache[i].p = p; break; }
    return p;
}
static VkPipeline present_pipeline(VkFormat fmt) { return present_pipeline_ex(fmt, false); }

// Scanout shows the bytes the game wrote: sRGB images are read through a UNORM view (no decode), like metal.m.
static VkImageView raw_view(TexBackend *b) {
    VkFormat f = b->format == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM
               : b->format == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : b->format;
    static const VkComponentMapping id = {0};
    return vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, f, 0, 1, raw_layer, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// ---- post passes (Video tab "Anti-aliasing" / "Upscaler"): SMAA 1x at the game image's size, then AMD FSR 1 (EASU
// upscale to the image's size in the window, RCAS sharpen) when it is smaller than its place in the window, as
// metal.m does with MetalFX, or a box downscale when it is larger. Each pass: a fullscreen triangle into an RGBA8
// image (gamma-encoded, like the input); up to three inputs (bindings 0-2), sampled linear / clamp.
static VkPipelineLayout post_layout;
static VkDescriptorSetLayout post_set_layout;

// kind: 0 a plain post pass; 1 SMAA edges also writing stencil 1 where they draw, 2 SMAA weights only where the
// stencil is 1 (g_vk.smaa_stencil; dsfmt: the stencil format); 3 multiplies the target's colour by the shader's output
// (GTAO onto the game's lit image). cfmt: the colour target's format.
static VkPipeline post_pipeline_kind(const char *fs, int kind, VkFormat dsfmt, VkFormat cfmt) {
    static struct { const char *fs; int kind; VkFormat f, cf; VkPipeline p; } cache[32];
    for (int i = 0; i < 32; i++)
        if (cache[i].p && cache[i].fs == fs && cache[i].kind == kind && cache[i].f == dsfmt && cache[i].cf == cfmt) return cache[i].p;
    if (!post_layout) {
        VkDescriptorSetLayoutBinding b[6];
        for (uint32_t i = 0; i < 6; i++)
            b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 6, .pBindings = b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &post_set_layout));
        VkPushConstantRange r = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128};  // (32 bytes but for GTAO's camera: 128)
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &post_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &post_layout));
    }
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vk_builtin_shader("post.vert"), .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = vk_builtin_shader(fs), .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    if (kind == 3)  // (colour = dst x src; alpha kept)
        cba = (VkPipelineColorBlendAttachmentState){.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR,
                                                    .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, .colorBlendOp = VK_BLEND_OP_ADD,
                                                    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                    .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xF};
    else if (kind == 5)  // (colour = mix(dst, src, src alpha); alpha kept: the water's reflections)
        cba = (VkPipelineColorBlendAttachmentState){.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                                                    .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
                                                    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                    .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xF};
    else if (kind == 4)  // (colour = dst + src; alpha kept: the indirect light onto the lit image)
        cba = (VkPipelineColorBlendAttachmentState){.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                                                    .dstColorBlendFactor = VK_BLEND_FACTOR_ONE, .colorBlendOp = VK_BLEND_OP_ADD,
                                                    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                    .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xF};
    else if (kind == 6)  // (colour = src + dst x src1, dual-source; alpha kept: the water's depth, caustics and reflections)
        cba = (VkPipelineColorBlendAttachmentState){.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
                                                    .dstColorBlendFactor = VK_BLEND_FACTOR_SRC1_COLOR, .colorBlendOp = VK_BLEND_OP_ADD,
                                                    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                                                    .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                              .pAttachments = &cba};
    VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    uint32_t ndyn = 2;
    fullscreen_vertex_dyn(dyn, &ndyn, 3);
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = ndyn, .pDynamicStates = dyn};
    VkFormat fmt = cfmt ? cfmt : VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (kind == 1 || kind == 2) {
        ri.stencilAttachmentFormat = dsfmt;
        VkStencilOpState so = {.failOp = VK_STENCIL_OP_KEEP, .passOp = kind == 1 ? VK_STENCIL_OP_REPLACE : VK_STENCIL_OP_KEEP,
                               .depthFailOp = VK_STENCIL_OP_KEEP, .compareOp = kind == 1 ? VK_COMPARE_OP_ALWAYS : VK_COMPARE_OP_EQUAL,
                               .compareMask = 0xFF, .writeMask = kind == 1 ? 0xFF : 0, .reference = 1};
        ds.stencilTestEnable = VK_TRUE;
        ds.front = ds.back = so;
    }
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                       .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                       .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                       .pColorBlendState = &cb, .pDynamicState = &dy, .layout = post_layout};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &p));
    for (int i = 0; i < 32; i++)
        if (!cache[i].p) { cache[i].fs = fs; cache[i].kind = kind; cache[i].f = dsfmt; cache[i].cf = cfmt; cache[i].p = p; break; }
    return p;
}

// Post image `i` at w x h (RGBA8, kept until the size changes). 0, 1: SMAA edges / weights, 2: SMAA output,
// 3, 4: FSR EASU / RCAS (also the sharpening after the box downscale), 5: box downscale, 6: the sharpening at the
// image's own size, 7, 8: GTAO raw / filtered. (f: its format - 9, 10, the indirect light: RGBA16F)
static TexBackend *post_image_f(int i, uint32_t w, uint32_t h, VkFormat f) {
    static TexBackend img[11];
    TexBackend *d = &img[i];
    if (d->image && (d->width != w || d->height != h || d->format != f)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = f, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}
static TexBackend *post_image(int i, uint32_t w, uint32_t h) { return post_image_f(i, w, h, VK_FORMAT_R8G8B8A8_UNORM); }

static VkImageView post_view(TexBackend *d) {
    static const VkComponentMapping id = {0};
    return vk_image_view(d, VK_IMAGE_VIEW_TYPE_2D, d->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
}

// ================================================================== temporal anti-aliasing: AMD FSR 2 (vk_fsr2.h)
// Pokémon Z-A tone-maps its frame into an image of its own and copies that, edge anti-aliased, into the image it
// shows; the HUD is drawn over it. With FSR 2 that copy is replaced: the scene was drawn jittered (fsr2_frame_jitter),
// its camera motion vectors and reactive mask are made from the depth and stencil (fsr2_mv.frag, with the view-
// projections of this frame and the last: mv_note_draw), FSR 2 resolves the picture with its history, and the result
// is copied into the shown image. Only at the scene's own size for now (no upscaling), not with Dynamic resolution.

// Its images (kept until the size changes): 0 motion vectors (RG16F), 1 reactive mask (R8), 2 FSR 2's output (RGBA8)
// (3: DLAA's output as sRGB, dlaa_raw_in)
static TexBackend *fsr2_image(int i, VkFormat f, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    static TexBackend img[4];
    TexBackend *d = &img[i];
    if (d->image && (d->width != w || d->height != h)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage | VK_IMAGE_USAGE_SAMPLED_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL) != VK_SUCCESS) { d->image = VK_NULL_HANDLE; return NULL; }
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

// A command buffer for FSR 2 to record into itself (a secondary one the recorder executes in its place). Four in turn:
// the fourth back has been executed by the GPU (waited for when not), so at most three FSR 2 frames are in flight -
// its backend keeps four frames of descriptor sets.
static VkCommandBuffer fsr2_cmd(void) {
    static VkCommandPool pool[4];
    static VkCommandBuffer buf[4];
    static uint64_t seq[4];
    static int k;
    k = (k + 1) & 3;
    if (!pool[k]) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                      .queueFamilyIndex = g_vk.qfam};
        VK_CHECK(vkCreateCommandPool(g_vk.dev, &pi, NULL, &pool[k]));
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool[k],
                                          .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY, .commandBufferCount = 1};
        VK_CHECK(vkAllocateCommandBuffers(g_vk.dev, &ai, &buf[k]));
    } else {
        if (seq[k] > vk_completed_seq()) vk_wait_seq(seq[k]);
        VK_CHECK(vkResetCommandPool(g_vk.dev, pool[k], 0));
    }
    seq[k] = vk_pending_seq();
    VkCommandBufferInheritanceInfo ii = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, .pInheritanceInfo = &ii};
    VK_CHECK(vkBeginCommandBuffer(buf[k], &bi));
    return buf[k];
}

// The texture the bound fragment shader reads as its texture i (as texture_info finds it), or NULL
static NvnTexture *frag_tex(Prog *pr, int i) {
    ShaderInfo *si = pr->fi;
    if (!si || i >= si->ntex || !st.tpool) return NULL;
    int slot = ((int)si->tex[i].handle_offset - 0x20) / 8;
    uint64_t h = slot >= 0 && slot < 32 ? st.tex[1][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    if (!h || tic >= (uint32_t)st.tpool->count) return NULL;
    return ((NvnTexEntry *)st.tpool->entries)[tic].tex;
}

static void inv4(const float *m, float *o) {  // (a 4x4 inverse; the same layout in and out)
    float a[16];
    a[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    a[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    a[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    a[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    a[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    a[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    a[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    a[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    a[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    a[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    a[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    a[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    a[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    a[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    a[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    a[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * a[0] + m[1] * a[4] + m[2] * a[8] + m[3] * a[12];
    float r = det != 0 ? 1 / det : 0;
    for (int i = 0; i < 16; i++) o[i] = a[i] * r;
}

// The picture the game copies into its shown image: its own interface drawn over it before that copy (the area's name
// after a fast travel, the minimap) went through DLAA with the scene - DLSS moved the letters with the camera's motion
// and blended them with their last places: the name smeared while the camera turned. With DLAA the picture is
// resolved in place when the interface's first draw into it comes (no depth target: not the scene - fsr2_early),
// and that interface is drawn over the result as it is; the game's copy then only brings the picture over (decoded,
// as its own copy does). LARECOMP_DLAA_EARLY=0: resolved at the copy, interface included, as before
// (fsr2_pic: declared with fsr2_ok_frame)
static uint64_t fsr2_early_frame = ~0ull;
static bool fsr2_logged_early;
static bool fsr2_copy_(Prog *pr, NvnTexture *early);
static bool fsr2_early_on(void) {
    static int on = -1;
    if (on < 0) on = !ENV("LARECOMP_DLAA_EARLY") || atoi(ENV("LARECOMP_DLAA_EARLY")) != 0;
    return on && g_settings.taa == TAA_DLAA && vk_dlaa_available();
}
static bool fsr2_copy(Prog *pr) {
    extern uint64_t vk_frame_counter(void);
    uint64_t f = vk_frame_counter();
    NvnTexture *rt = st.rt.color[0];
    {   // (debug) LARECOMP_TAA_TRACE=<frame>: that frame's draws into the picture / shown image, against the resolve
        static long tr = -2;
        if (tr == -2) tr = ENV("LARECOMP_TAA_TRACE") ? atol(ENV("LARECOMP_TAA_TRACE")) : -1;
        if (tr >= 0 && (long)f >= tr && (long)f < tr + 2 && rt && pr)
            LOG("[taa] f%llu rt %p%s fmt %#x %ux%u depth %d blend %d fs %llx ntex %d tex0 %p%s; resolved %d early %d mv %d\n",
                (unsigned long long)f, (void *)rt, rt == fsr2_pic ? " (picture)" : "", rt->format, rt->width, rt->height,
                st.rt.depth != NULL, st.color_state[0] & 1, (unsigned long long)(pr->code[1] ? pr->code[1] : pr->prep_fh),
                pr->fi ? pr->fi->ntex : -1, (void *)frag_tex(pr, 0), frag_tex(pr, 0) == fsr2_pic && fsr2_pic ? " (picture)" : "",
                fsr2_ok_frame == f, fsr2_early_frame == f, mv_frame[0] == f);
    }
    if (fsr2_early_frame == f && fsr2_ok_frame == f && rt && rt != fsr2_pic && frag_tex(pr, 0) == fsr2_pic && fsr2_pic) {
        // the copy of a picture resolved early: brought over as it is (decoded: the blit), and the draw skipped
        TexBackend *db = rt->backend, *sb = fsr2_pic->backend;
        if (!db || !sb || st.color_state[0] || !db->presented || sb->width != db->width || sb->height != db->height) return false;
        fsr2_early_frame = ~0ull;
        vk_end_pass();
        renderer_texture_frame_fresh(rt), renderer_texture_level_fresh(rt), renderer_texture_first_write(rt);
        VkCommandBuffer cb = vk_cmd();
        vk_full_barrier(cb);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)sb->width, (int32_t)sb->height, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)db->width, (int32_t)db->height, 1}}};
        rec_CmdBlitImage(cb, sb->image, VK_IMAGE_LAYOUT_GENERAL, db->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
        vk_full_barrier(cb);
        return true;
    }
    // the interface's first draw into last frame's picture (no depth target, blended): the picture resolved now
    if (fsr2_early_on() && fsr2_ok_frame != f && rt && rt == fsr2_pic && !st.rt.depth && (st.color_state[0] & 1) && mv_frame[0] == f) {
        if (fsr2_copy_(pr, rt)) fsr2_early_frame = f;
        return false;  // (the draw goes on, over the resolved picture)
    }
    return fsr2_copy_(pr, NULL);
}
// Called for each draw with its program: true when this draw was the game's copy into its shown image and FSR 2 did
// it instead (the draw is then skipped). early: the picture resolved in place, before its interface (fsr2_copy)
static bool fsr2_copy_(Prog *pr, NvnTexture *early) {
    extern uint64_t vk_frame_counter(void);
    uint64_t f = vk_frame_counter();
    NvnTexture *dst = early ? NULL : st.rt.color[0];
    TexBackend *db = early ? early->backend : dst ? dst->backend : NULL;  // (early: the sizes are the picture's own)
    // (once a frame: this frame's scene is resolved already. ZA Recomp #175: a menu's full-screen background drawn
    // unblended into the shown image afterwards, from a mipmapped picture, went through FSR as a second "copy" and
    // showed as a mosaic of mip 0's stale squares. A copy's source has one level)
    if (fsr2_ok_frame == f) return false;
    if (!db || (!early && (st.color_state[0] || !db->presented)) || mv_frame[0] != f || !mv_depth || !mv_depth->backend) return false;
    NvnTexture *src = early ? early : frag_tex(pr, 0);
    TexBackend *sb = src ? src->backend : NULL, *zb = mv_depth->backend;
    if (!sb || !sb->image || src == dst || sb->levels > 1 || sb->width != db->width || sb->height != db->height || sb->width != zb->width ||
        sb->height != zb->height || (sb->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT))
        return false;
    if (sb->scalable && (sb->ax != sb->sx || sb->ay != sb->sy)) return false;  // (Dynamic: the picture in part of the image)
    uint32_t w = sb->width, h = sb->height;
    vk_end_pass();
    TexBackend *mvb = fsr2_image(0, VK_FORMAT_R16G16_SFLOAT, w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    TexBackend *rb = fsr2_image(1, VK_FORMAT_R8_UNORM, w, h, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    TexBackend *ob = fsr2_image(2, VK_FORMAT_R8G8B8A8_UNORM, w, h, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!mvb || !rb || !ob) return false;
    static const VkComponentMapping id = {0};
    VkImageView zv = vk_image_view(zb, VK_IMAGE_VIEW_TYPE_2D, zb->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkImageView sv = vk_image_view(zb, VK_IMAGE_VIEW_TYPE_2D, zb->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_STENCIL_BIT);
    // DLAA gets an sRGB picture's encoded values (through its UNORM twin), and its output is decoded into the shown
    // image (copied as is into an sRGB image, then blitted: the blit decodes, as the game's own copy does). Read as
    // linear light, as before, the darkest tones had a few steps each: when the screen fades to black (fast travel)
    // DLSS's output showed a grid of squares over every flat surface, and its 8-bit linear output kept them. DLSS
    // expects a picture that isn't HDR in its display encoding. (FSR 2: as before.) LARECOMP_DLAA_RAWIN=0: as before
    static int raw_in = -1;
    if (raw_in < 0) raw_in = !ENV("LARECOMP_DLAA_RAWIN") || atoi(ENV("LARECOMP_DLAA_RAWIN")) != 0;
    bool dlaa_raw = raw_in && g_settings.taa == TAA_DLAA && vk_dlaa_available() && sb->format == VK_FORMAT_R8G8B8A8_SRGB;
    TexBackend *eb = dlaa_raw ? fsr2_image(3, VK_FORMAT_R8G8B8A8_SRGB, w, h, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT) : NULL;
    if (dlaa_raw && !eb) dlaa_raw = false;
    VkFormat cfmt = dlaa_raw ? VK_FORMAT_R8G8B8A8_UNORM : sb->format;
    VkImageView cv = vk_image_view(sb, VK_IMAGE_VIEW_TYPE_2D, cfmt, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!zv || !sv || !cv) return false;
    // ---- the motion vectors and the reactive mask
    static VkPipelineLayout layout;
    static VkDescriptorSetLayout dsl;
    static VkPipeline pipe;
    if (!pipe) {
        VkDescriptorSetLayoutBinding bd[3] = {
            {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
            {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL},
            {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL}};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = bd};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &dsl));
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &layout));
        VkPipelineShaderStageCreateInfo stages[2] = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
             .module = vk_builtin_shader("post.vert"), .pName = "main"},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
             .module = vk_builtin_shader("fsr2_mv.frag"), .pName = "main"}};
        VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                     .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
        VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                     .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
        VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState cba[2] = {{.colorWriteMask = 0xF}, {.colorWriteMask = 0xF}};
        VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 2,
                                                   .pAttachments = cba};
        VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        uint32_t ndyn = 2;
        fullscreen_vertex_dyn(dyn, &ndyn, 3);
        VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = ndyn, .pDynamicStates = dyn};
        VkFormat fmts[2] = {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R8_UNORM};
        VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 2,
                                            .pColorAttachmentFormats = fmts};
        VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                           .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                           .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                           .pColorBlendState = &cbs, .pDynamicState = &dy, .layout = layout};
        VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &pipe));
    }
    // camera cuts (another view: a scene change, a cutscene's cut) start FSR 2's history again
    bool reset = mv_frame[1] + 1 != f || fsr2_ok_frame + 1 != f;
    {
        const float *a = mv_vp[0], *b = mv_vp[1];  // (the view axis: w's row of the projection x view)
        float la = sqrtf(a[3] * a[3] + a[7] * a[7] + a[11] * a[11]), lb = sqrtf(b[3] * b[3] + b[7] * b[7] + b[11] * b[11]);
        if (la > 0 && lb > 0 && (a[3] * b[3] + a[7] * b[7] + a[11] * b[11]) / (la * lb) < 0.95f) reset = true;
    }
    static float react[2] = {-1, -1};
    if (react[0] < 0) {
        react[0] = ENV("LARECOMP_FSR2_REACT_CHAR") ? (float)atof(ENV("LARECOMP_FSR2_REACT_CHAR")) : 0.8f;
        react[1] = ENV("LARECOMP_FSR2_REACT_FOLIAGE") ? (float)atof(ENV("LARECOMP_FSR2_REACT_FOLIAGE")) : 0.2f;
    }
    float u[40];
    inv4(mv_vp[0], u);
    memcpy(u + 16, reset ? mv_vp[0] : mv_vp[1], 64);
    u[32] = fsr2_jx, u[33] = fsr2_jy, u[34] = 1.0f / w, u[35] = 1.0f / h;
    u[36] = react[0], u[37] = react[1], u[38] = u[39] = 0;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkDescriptorSet set = vk_alloc_set(dsl);
    VkSampler near = vk_default_sampler(false);
    VkDescriptorImageInfo ii[2] = {{near, zv, VK_IMAGE_LAYOUT_GENERAL}, {near, sv, VK_IMAGE_LAYOUT_GENERAL}};
    VkDescriptorBufferInfo bi = ubo_span(vk_transient_copy(u, sizeof u, 256), sizeof u);
    VkWriteDescriptorSet wr[3] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &bi}};
    rec_UpdateDescriptorSets(3, wr);
    VkRenderingAttachmentInfo ca[2] = {
        {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(mvb), .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE},
        {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(rb), .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE}};
    VkRenderingInfo rinfo = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                             .colorAttachmentCount = 2, .pColorAttachments = ca};
    rec_CmdBeginRendering(cb, &rinfo);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    rec_CmdSetViewport(cb, 0, 1, &vp);
    rec_CmdSetScissor(cb, 0, 1, &sc);
    fullscreen_vertex_bind(cb);
    rec_CmdDraw(cb, 3, 1, 0, 0);
    rec_CmdEndRendering(cb);
    vk_full_barrier(cb);
    objmv_check(sb, mvb, w, h, true);
    objmv_replay(mvb, zv, w, h);  // (DLAA: the characters' own motion over the camera's)
    objmv_check(sb, mvb, w, h, false);
    cb = vk_cmd();  // (the check may have submitted the command buffer)
    // ---- FSR 2
    static uint64_t last_ns;
    uint64_t now = plat_now_ns();
    float dt = last_ns ? (float)((now - last_ns) / 1e6) : 16.7f;
    last_ns = now;
    const float *m = mv_vp[0];
    float p11 = sqrtf(m[1] * m[1] + m[5] * m[5] + m[9] * m[9]);  // (the projection's y scale: the y row's length)
    VkFsr2Frame fr = {.color = sb->image, .depth = zb->image, .mv = mvb->image, .reactive = rb->image, .out = ob->image,
                      .color_view = cv, .depth_view = zv, .mv_view = post_view(mvb), .reactive_view = post_view(rb), .out_view = post_view(ob),
                      .color_format = cfmt, .depth_format = zb->format, .out_format = VK_FORMAT_R8G8B8A8_UNORM,
                      .w = w, .h = h, .ow = w, .oh = h, .jx = fsr2_jx, .jy = fsr2_jy, .dt_ms = fminf(fmaxf(dt, 1), 100),
                      .near_z = m[14] > 0 ? m[14] : 0.1f, .fov_y = p11 > 0 ? 2 * atanf(1 / p11) : 1.0f, .sharpness = 0, .reset = reset};
    VkCommandBuffer sec = fsr2_cmd();
    bool dlaa = g_settings.taa == TAA_DLAA && vk_dlaa_available();  // (NVIDIA DLAA: the same inputs - dlaa.c)
    bool ok = dlaa ? vk_dlaa_record(g_vk.dev, sec, &fr) : vk_fsr2_record(g_vk.phys, g_vk.dev, sec, &fr);
    VK_CHECK(vkEndCommandBuffer(sec));
    if (!ok) return false;
    // (the shown image's first pass this frame is now the HUD's: a target's first pass in a frame clears it -
    // renderer_texture_frame_fresh - which would wipe the copy below. Taken as done here.)
    if (dst) renderer_texture_frame_fresh(dst), renderer_texture_level_fresh(dst), renderer_texture_first_write(dst);
    rec_CmdExecuteCommands(cb, 1, &sec);
    vk_full_barrier(cb);
    VkImageCopy c = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .extent = {w, h, 1}};
    if (early) {  // (back into the picture: encoded values copied as they are, linear ones encoded by a blit)
        if (dlaa_raw) rec_CmdCopyImage(cb, ob->image, VK_IMAGE_LAYOUT_GENERAL, sb->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        else {
            VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                              .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
            rec_CmdBlitImage(cb, ob->image, VK_IMAGE_LAYOUT_GENERAL, sb->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
        }
    } else if (dlaa_raw) {  // (encoded values: decoded on the way into the shown image)
        rec_CmdCopyImage(cb, ob->image, VK_IMAGE_LAYOUT_GENERAL, eb->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        vk_full_barrier(cb);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
        rec_CmdBlitImage(cb, eb->image, VK_IMAGE_LAYOUT_GENERAL, db->image, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_NEAREST);
    } else rec_CmdCopyImage(cb, ob->image, VK_IMAGE_LAYOUT_GENERAL, db->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    vk_full_barrier(cb);
    {   // (debug) LARECOMP_FSR2_DUMP=<frame>: its inputs and output that frame -> build/fsr2_*.png
        static long at = -2;
        if (at == -2) at = ENV("LARECOMP_FSR2_DUMP") ? atol(ENV("LARECOMP_FSR2_DUMP")) : -1;
        if (at >= 0 && f == (uint64_t)at) {
            renderer_save_texture_tex(sb, "build/fsr2_in.png");
            renderer_save_texture_tex(mvb, "build/fsr2_mv.png");
            renderer_save_texture_tex(rb, "build/fsr2_react.png");
            renderer_save_texture_tex(ob, "build/fsr2_out.png");
        }
    }
    static int nlog;
    if (nlog++ < 3 || (early && !fsr2_logged_early))
        LOG("[%s] frame %llu: %ux%u, jitter %.3f %.3f%s; picture format %d, shown image's %d%s\n", dlaa ? "dlaa" : "fsr2",
            (unsigned long long)f, w, h, fsr2_jx, fsr2_jy, reset ? ", reset" : "", (int)sb->format, (int)db->format,
            early ? "; resolved before the interface drawn over it" : "");
    if (early) fsr2_logged_early = true;
    fsr2_ok_frame = f;
    fsr2_pic = src;
    return true;
}

// SMAA's lookup textures (runtime/nvn/smaa_data.c): 0 = area (160 x 560 R8G8), 1 = search (64 x 16 R8), uploaded once.
static VkImageView smaa_lut(int i) {
    static TexBackend t[2];
    TexBackend *d = &t[i];
    if (!d->image) {
        extern const unsigned char *smaa_area_tex(void);
        extern const unsigned char *smaa_search_tex(void);
        uint32_t w = i ? 64 : 160, h = i ? 16 : 560, bpp = i ? 1 : 2;
        VkFormat f = i ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL));
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        VkCommandBuffer cb = vk_cmd();
        vk_image_init_layout(cb, d->image, d->aspect);
        VkSpan s = vk_transient_copy(i ? smaa_search_tex() : smaa_area_tex(), (uint64_t)w * h * bpp, 16);
        VkBufferImageCopy c = {.bufferOffset = s.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        rec_CmdCopyBufferToImage(cb, s.buf, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    }
    return post_view(d);
}

// One post pass (outside the window image's rendering): srcs (bindings 0..n-1) -> dst, fragment shader `fs`.
// clear: start from zero (SMAA's edge pass only writes edge pixels; its weights / output read them).
// post_area_w / _h (> 0): only that top-left part of dst (Dynamic resolution: the contents' part)
static uint32_t post_area_w, post_area_h;
// (g_vk.smaa_stencil) a stencil image at w x h for SMAA's edge mask (kept until the size changes; NULL: none usable)
static TexBackend *smaa_stencil_image(VkCommandBuffer cb, uint32_t w, uint32_t h) {
    static TexBackend img;
    static bool unusable;
    TexBackend *d = &img;
    if (unusable) return NULL;
    if (d->image && (d->width != w || d->height != h)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        static const VkFormat cand[3] = {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
        VkFormat f = VK_FORMAT_UNDEFINED;
        for (int i = 0; i < 3 && !f; i++) {
            VkFormatProperties fp;
            vkGetPhysicalDeviceFormatProperties(g_vk.phys, cand[i], &fp);
            if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) f = cand[i];
        }
        if (!f) { unusable = true; LOG("[vk] SMAA stencil: no stencil format, off\n"); return NULL; }
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL) != VK_SUCCESS) { d->image = VK_NULL_HANDLE; return NULL; }
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1; d->samples = 1;
        d->aspect = f == VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(cb, d->image, d->aspect);
        LOG("[vk] SMAA stencil image %ux%u format %d\n", w, h, f);
    }
    return d;
}

static void post_pass_ex(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                         bool clear, TexBackend *sten, int kind);
static void post_pass(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                      bool clear) {
    post_pass_ex(cb, fs, srcs, n, dst, pc, clear, NULL, 0);
}
// sten / kind (g_vk.smaa_stencil): kind 1 clears sten and writes 1 where the pass draws, kind 2 draws only there
static const float *post_pc_big;  // (GTAO) 32 floats pushed instead of pc's 8
static bool post_mips_last;  // (water) input 3 sampled with its mip levels (the others: level 0 only)
static bool post_nearest_first;  // (SSPR's hash) input 0 can't be filtered: its format is an integer one
static VkSampler post_mip_sampler(void) {
    static VkSampler s;
    if (!s) {
        VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
                                  .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                  .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = VK_LOD_CLAMP_NONE};
        VK_CHECK(vkCreateSampler(g_vk.dev, &si, NULL, &s));
    }
    return s;
}
static void post_pass_ex(VkCommandBuffer cb, const char *fs, const VkImageView *srcs, int n, TexBackend *dst, const float pc[8],
                         bool clear, TexBackend *sten, int kind) {
    if (!sten && kind < 3) kind = 0;
    uint32_t aw = post_area_w && post_area_w < dst->width ? post_area_w : dst->width;
    uint32_t ah = post_area_h && post_area_h < dst->height ? post_area_h : dst->height;
    // (before the rendering begins: created inside it, the NVIDIA driver crashed)
    VkPipeline pipe = post_pipeline_kind(fs, kind, sten ? sten->format : VK_FORMAT_UNDEFINED,
                                         dst->format == VK_FORMAT_R8G8B8A8_UNORM ? VK_FORMAT_UNDEFINED : dst->format);
    vk_full_barrier(cb);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(dst),
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : kind >= 3 ? VK_ATTACHMENT_LOAD_OP_LOAD
                                            : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {aw, ah}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
    VkRenderingAttachmentInfo sa = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (kind == 1 || kind == 2) {  // (the SMAA stencil kinds; 3 has none)
        static const VkComponentMapping id = {0};
        sa.imageView = vk_image_view(sten, VK_IMAGE_VIEW_TYPE_2D, sten->format, 0, 1, 0, 1, id, sten->aspect);
        sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sa.loadOp = kind == 1 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        sa.storeOp = kind == 1 ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ri.pStencilAttachment = &sa;
    }
    rec_CmdBeginRendering(cb, &ri);
    VkDescriptorSet ds = vk_alloc_set(post_set_layout);
    VkDescriptorImageInfo ii[6];  // (up to 6 inputs: the water's reflections take 6)
    VkWriteDescriptorSet w[6];
    for (int i = 0; i < n; i++) {
        ii[i] = (VkDescriptorImageInfo){post_mips_last && i == 3 ? post_mip_sampler() : vk_default_sampler(i != 5 && !(post_nearest_first && i == 0)), srcs[i],
                                        VK_IMAGE_LAYOUT_GENERAL};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      .pImageInfo = &ii[i]};
    }
    rec_UpdateDescriptorSets((uint32_t)n, w);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, post_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)dst->width, (float)dst->height, 0, 1};
    VkRect2D sc = {{0, 0}, {aw, ah}};  // (the viewport stays the whole image: gl_FragCoord is the pixel)
    rec_CmdSetViewport(cb, 0, 1, &vp);
    rec_CmdSetScissor(cb, 0, 1, &sc);
    if (post_pc_big) rec_CmdPushConstants(cb, post_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 128, post_pc_big);
    else rec_CmdPushConstants(cb, post_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32, pc);
    fullscreen_vertex_bind(cb);
    rec_CmdDraw(cb, 3, 1, 0, 0);
    rec_CmdEndRendering(cb);
}

// ================================================================== GTAO (ambient occlusion)
// Pokémon Z-A draws no screen-space ambient occlusion. Its frame is deferred: the G-buffer, then one lighting pass
// (fragment shader GTAO_LIGHTING) that reads the scene depth (its texture 1) and rebuilds each pixel's world position as
// ndc.x * c12 + ndc.y * c13 + depth * c14 + c15 (rows of its constant buffer 2 - "cb5" in its GLSL -, divided by w; row
// 19 is the camera's position, rows 20 / 21 hold the tangents of half the field of view). Just before the first
// lighting draw of a frame: GTAO (shaders.c gtao.frag, after XeGTAO) from that depth into a half-size image, then a
// depth-aware filter; when the lighting pass ends (the next begin_pass), the result - upsampled with depth weights -
// multiplies its target: the lit image, before the sky, transparents, fog, effects and the HUD. Off by default (F1 >
// Ambient occlusion). Test switches: LARECOMP_GTAO=<0-3> (the level), _RADIUS (metres, 1.5), _INTENSITY (the factor,
// 0.85 at 100%), _FLIPY (ndc y sign, -1), _FROM=<frame> (on from that frame: A/B in one run), _DEBUG (1 the AO alone,
// 2 the normals, 3 distance bands).
static struct { bool pending; NvnTexture *target; VkImageView ao, depth; uint64_t frame; float pc[32]; } gtao;

static float gtao_env(const char *name, float def) {
    const char *v = getenv(name);  // (not ENV(): its cache is per call site, and this one serves several names)
    return v ? (float)atof(v) : def;
}

// F1 > Ambient occlusion (ambient_occlusion: 0 off, 1 low, 2 medium, 3 high; LARECOMP_GTAO=<0-3> overrides it)
static int gtao_level(void) {
    static int env = -2;  // (read once: it is asked on every draw, and getenv is slow on Windows)
    if (env == -2) { const char *e = getenv("LARECOMP_GTAO"); env = e ? atoi(e) : -1; }
    int l = env >= 0 ? env : g_settings.gtao;
    return l < 0 ? 0 : l > 3 ? 3 : l;
}
static bool gtao_on(void) { return gtao_level() > 0; }

// Screen-space indirect light (SSGI: ssgi_apply below): one bounce of the game's light from what is on screen, onto the
// lit image after the AO. F1 > Indirect light (LARECOMP_SSGI=<0-3>: off, 2 / 3 / 4 slices) and its strength
// (_STRENGTH, 1 = 100%: one physical bounce), _RADIUS (metres, 4),
// _DEBUG=1 (the bounce light alone).
static struct { VkImageView albedo; } ssgi;
static VkImageView water_probe;  // (gtao_before_lighting: the lighting's reflection cube, for water_apply_pending)
static int ssgi_level(void) {  // (F1 > Indirect light; LARECOMP_SSGI=<0-3> overrides it)
    static int env = -2;  // (read once: asked on every draw)
    if (env == -2) { const char *e = getenv("LARECOMP_SSGI"); env = e ? atoi(e) : -1; }
    int l = env >= 0 ? env : g_settings.ssgi;
    return l < 0 ? 0 : l > 3 ? 3 : l;
}
static bool ssgi_on(void) { return ssgi_level() > 0; }
static bool water_on(void);  // (water reflections: below)

// The lighting shader's texture i: its pool entry (as texture_info resolves it), or NULL
static NvnTexEntry *gtao_tex_entry(ShaderInfo *si, int i) {
    if (!si || i >= si->ntex) return NULL;
    int slot = ((int)si->tex[i].handle_offset - 0x20) / 8;
    uint64_t h = (slot >= 0 && slot < 32) ? st.tex[1][slot] : 0;
    uint32_t tic = h & 0xFFFFF;
    if (!h || !st.tpool || tic >= (uint32_t)st.tpool->count) return NULL;
    NvnTexEntry *e = &((NvnTexEntry *)st.tpool->entries)[tic];
    return e->tex ? e : NULL;
}

// The deferred lighting pass, in any of its variants (the game swaps in others while a Pokémon is sent out, while a
// move's effects play...): GTAO_LIGHTING, or a fragment shader with 10+ textures among whose first four one depth
// texture is read twice - its depth and its stencil, the scene's - and a camera in constant buffer 2. Decided once per
// shader, with the texture the depth is read through (gtao_depth_i: the last one sampled as a float). Pokémon Z-A reads
// it as its textures 0 and 1 (depth both times); Scarlet / Violet as 0 (the stencil, an unsigned sampler: bit 6, the
// pixels the lighting skips) and 2 (the depth, 74421cd200) - the check for 0 and 1 alone never matched there, so F1 >
// Ambient occlusion, Indirect light and Water reflections did nothing. The camera rows are the same in both
// (cb5[12..15] the inverse view-projection, cb5[19] the camera's position).
static int gtao_depth_i = 1;  // (the last lighting variant matched: its depth texture's index)
static bool gtao_is_lighting_(Prog *pr);
static bool gtao_is_lighting(Prog *pr) {
    if (g_cur_fs_hash == GTAO_LIGHTING) return gtao_depth_i = 1, true;
    if (!(gtao_on() || ssgi_on() || water_on()) || !pr || !pr->fi || pr->fi->ntex < 10 || !st.program) return false;
    SWL_AT(9186);
    bool r = gtao_is_lighting_(pr);
    SWU();
    return r;
}
static bool gtao_is_lighting_(Prog *pr) {
    static uint64_t known[64]; static int8_t verdict[64]; static int nk;  // (verdict: the depth's index, -1 = not lighting)
    uint64_t fs = st.program->stages[1].code_addr;
    for (int i = 0; i < nk; i++)
        if (known[i] == fs) return verdict[i] >= 0 ? (gtao_depth_i = verdict[i], true) : false;
    int d = -1;
    if (st.ubuf[1][2].addr && st.ubuf[1][2].size >= 22 * 16) {
        NvnTexEntry *e[4];
        for (int i = 0; i < 4; i++) e[i] = gtao_tex_entry(pr->fi, i);
        for (int i = 0; i < 4; i++) {
            if (!e[i] || !renderer_format_is_depth(e[i]->tex->format) || !e[i]->tex->backend ||
                !((TexBackend *)e[i]->tex->backend)->scalable || pr->fi->tex[i].type != TEXTYPE_2D)
                continue;
            bool twice = false;
            for (int k = 0; k < 4; k++) twice |= k != i && e[k] && e[k]->tex == e[i]->tex;
            // (the float read: of Z-A's two depth reads the second, texture 1, as before; Violet's other read is the
            // stencil's, an unsigned sampler)
            if (twice && !pr->fi->tex[i].stencil) d = i;
        }
    }
    if (nk < 64) { known[nk] = fs; verdict[nk++] = (int8_t)d; }
    if (d >= 0) LOG("[gtao] lighting variant fs %016llx (%d textures, the depth as texture %d)\n", (unsigned long long)g_cur_fs_hash,
                    pr->fi->ntex, d);
    if (d >= 0) gtao_depth_i = d;
    return d >= 0;
}

// Before the frame's first lighting draw (outside any pass): GTAO and its filter into post images 6 / 7.
static void gtao_before_lighting(Prog *pr) {
    extern uint64_t renderer_exec_frame(void);
    uint64_t frame = renderer_exec_frame();
    if (!(gtao_on() || ssgi_on() || water_on()) || gtao.frame == frame + 1 || frame < (uint64_t)gtao_env("LARECOMP_GTAO_FROM", 0)) return;  // (_FROM: on from that frame - A/B tests)
    gtao.frame = frame + 1;
    NvnTexEntry *de = gtao_tex_entry(pr->fi, gtao_depth_i);
    const Range *cam = &st.ubuf[1][2];
    if (!de || !renderer_format_is_depth(de->tex->format) || !cam->addr || cam->size < 22 * 16) {
        static int nlog;
        if (nlog++ < 3) LOG("[gtao] lighting draw without the depth / camera it expects: skipped\n");
        return;
    }
    VkImageView dv = vk_view(de->tex, &de->view, de->has_view, false, pr->fi->tex[gtao_depth_i].type);
    TexBackend *db = de->tex->backend;
    if (!dv || !db) return;
    const float *c = (const float *)(uintptr_t)cam->addr;
    float *pc = gtao.pc;
    memcpy(pc, c + 12 * 4, 16 * sizeof(float));  // m0..m3
    memcpy(pc + 16, c + 19 * 4, 4 * sizeof(float));  // cam
    uint32_t aw = (db->width + 1) / 2, ah = (db->height + 1) / 2;
    pc[20] = gtao_env("LARECOMP_GTAO_RADIUS", 1.5f);
    // F1 > Ambient occlusion strength (ambient_occlusion_strength, percent: 25-100; 100 = 0.85, the look tuned on
    // Lumiose's streets); LARECOMP_GTAO_INTENSITY=<0-2> sets the factor itself
    pc[21] = getenv("LARECOMP_GTAO_INTENSITY") ? gtao_env("LARECOMP_GTAO_INTENSITY", 0.85f) : 0.85f * g_settings.gtao_strength / 100.0f;
    pc[22] = gtao_env("LARECOMP_GTAO_FLIPY", -1.0f);  // (Z-A: ndc y up - screen y down)
    pc[23] = 0;  // (one noise pattern: with no temporal accumulation, a pattern changing per frame flickered)
    pc[24] = (float)aw; pc[25] = (float)ah; pc[26] = (float)db->width; pc[27] = (float)db->height;
    pc[28] = fabsf(c[20 * 4 + 0]); pc[29] = fabsf(c[21 * 4 + 1]);  // tan(half fov) x / y
    pc[30] = gtao_env("LARECOMP_GTAO_DEBUG", 0);
    pc[31] = (float)(gtao_level() + 1);  // (slices: 2 / 3 / 4, twice as many steps each)  // (1: the lit image replaced by the AO alone, to check it)
    if (pc[28] <= 0 || pc[29] <= 0) return;
    gtao.depth = dv;
    if (ssgi_on()) {  // (SSGI: the G-buffer's albedo - its sRGB colour target at the depth's size - for ssgi_apply)
        ssgi.albedo = VK_NULL_HANDLE;
        for (int i = 2; i < pr->fi->ntex; i++) {
            NvnTexEntry *e = gtao_tex_entry(pr->fi, i);
            if (e && e->tex->format == 0x38 && e->tex->width == de->tex->width && e->tex->height == de->tex->height) {
                ssgi.albedo = vk_view(e->tex, &e->view, e->has_view, false, pr->fi->tex[i].type);
                break;
            }
        }
    }
    if (water_on()) {  // (water: the lighting's biggest reflection cube, for where its mirror ray leaves the picture)
        water_probe = VK_NULL_HANDLE;
        uint32_t best = 0;
        for (int i = 2; i < pr->fi->ntex; i++) {
            NvnTexEntry *e = gtao_tex_entry(pr->fi, i);
            if (e && pr->fi->tex[i].type == TEXTYPE_CUBE && e->tex->width > best) {
                VkImageView v = vk_view(e->tex, &e->view, e->has_view, false, TEXTYPE_CUBE);
                if (v) water_probe = v, best = e->tex->width;            }
        }
    }
    if (!gtao_on()) { gtao.ao = VK_NULL_HANDLE; return; }
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    TexBackend *raw = post_image(7, aw, ah), *filt = post_image(8, aw, ah);
    post_pc_big = pc;
    VkImageView in[2] = {dv, dv};
    post_pass(cb, "gtao.frag", in, 1, raw, NULL, false);
    VkImageView bin[2] = {post_view(raw), dv};
    post_pass(cb, "gtao_blur.frag", bin, 2, filt, NULL, false);
    post_pc_big = NULL;
    gtao.ao = pc[30] >= 2 ? post_view(raw) : post_view(filt);  // (debug views 2 / 3: unfiltered)
    gtao.depth = dv;
    static int nlog;
    if (nlog++ < 2) LOG("[gtao] on: depth %ux%u, AO %ux%u, radius %.2f m, camera (%.1f, %.1f, %.1f)\n", db->width, db->height, aw, ah,
                        pc[20], pc[16], pc[17], pc[18]);
}

// The lighting pass has begun drawing into target: its contents are multiplied when it ends
static void gtao_mark(NvnTexture *target) {
    if (!(gtao.ao || (ssgi_on() && ssgi.albedo) || water_on()) || !target || gtao.frame == 0) return;
    gtao.pending = true;
    gtao.target = target;
}

// (begin_pass, outside any pass) the lighting pass has ended: the lit image times the AO
static void ssgi_apply(TexBackend *tb);
static void gtao_apply_pending(void) {
    if (!gtao.pending) return;
    gtao.pending = false;
    TexBackend *tb = gtao.target ? gtao.target->backend : NULL;
    if (!tb || !tb->image) return;
    if (gtao.ao) {
        float *pc = gtao.pc;
        pc[26] = (float)tb->width; pc[27] = (float)tb->height;  // (apply: the lit image's size)
        VkCommandBuffer cb = vk_cmd();
        post_pc_big = pc;
        VkImageView ain[2] = {gtao.ao, gtao.depth};
        post_pass_ex(cb, "gtao_apply.frag", ain, 2, tb, NULL, false, NULL, pc[30] != 0 ? 0 : 3);
        post_pc_big = NULL;
    }
    if (ssgi_on() && ssgi.albedo) ssgi_apply(tb);
    ssgi.albedo = VK_NULL_HANDLE;  // (this frame's G-buffer's: its texture may be gone by the next)
}

// SSGI, when the lighting pass has ended (its image tb is the bounce light's source and its target): the bounce light
// at half size from the depth and the lit image, a depth-aware filter, then x the albedo, added onto the lit image
static void ssgi_apply(TexBackend *tb) {
    float pc[32];
    memcpy(pc, gtao.pc, sizeof pc);  // (the camera: gtao_before_lighting)
    uint32_t w = (uint32_t)pc[26], h = (uint32_t)pc[27], hw = (w + 1) / 2, hh = (h + 1) / 2;
    if (!w || !h || !gtao.depth) return;
    pc[20] = gtao_env("LARECOMP_SSGI_RADIUS", 4.0f);
    pc[21] = getenv("LARECOMP_SSGI_STRENGTH") ? gtao_env("LARECOMP_SSGI_STRENGTH", 1.0f) : g_settings.ssgi_strength / 100.0f;
    pc[24] = (float)hw, pc[25] = (float)hh;
    pc[30] = gtao_env("LARECOMP_SSGI_DEBUG", 0);
    pc[31] = (float)(ssgi_level() + 1);
    static const VkComponentMapping id = {0};
    VkImageView lit = vk_image_view(tb, VK_IMAGE_VIEW_TYPE_2D, tb->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!lit) return;
    VkCommandBuffer cb = vk_cmd();
    TexBackend *raw = post_image_f(9, hw, hh, VK_FORMAT_R16G16B16A16_SFLOAT), *filt = post_image_f(10, hw, hh, VK_FORMAT_R16G16B16A16_SFLOAT);
    post_pc_big = pc;
    VkImageView in[2] = {gtao.depth, lit};
    post_pass(cb, "ssgi.frag", in, 2, raw, NULL, false);
    VkImageView bin[2] = {post_view(raw), gtao.depth};
    post_pass(cb, "ssgi_blur.frag", bin, 2, filt, NULL, false);
    pc[26] = (float)tb->width, pc[27] = (float)tb->height;
    VkImageView ain[3] = {post_view(filt), gtao.depth, ssgi.albedo};
    post_pass_ex(cb, "ssgi_apply.frag", ain, 3, tb, NULL, false, NULL, pc[30] != 0 ? 0 : 4);
    post_pc_big = NULL;
    static int nlog;
    if (nlog++ < 2) LOG("[ssgi] on: %ux%u (bounce light %ux%u), reach %.1f m, strength %.2f\n", w, h, hw, hh, pc[20], pc[21]);
}

// ================================================================== water reflections
// Pokémon Z-A's water (fountains, ponds...) is a see-through surface drawn onto the lit image after the lighting -
// blended, writing its depth, and reading the scene's depth (for its soft edges) - with no reflection of the world
// around it. Before the frame's first such draw the depth and the lit image are copied (water_before); once the
// game moves on to another target (begin_pass), wherever the depth changed is the water's surface, and
// water_ssr.frag blends what it mirrors onto it (water_apply). F1 > Water reflections; LARECOMP_WATER=0 / 1 overrides
// it, _STRENGTH (1), _DEBUG=1 (the reflection alone). The same pass also darkens and tints the water by its depth (the
// depth before it is the floor), lights its floor with caustics and gives it the stylized vivid look: F1 > Water depth /
// Water caustics / Vivid water (LARECOMP_WATER_DEPTH / _CAUSTICS / _VIVID=0 / 1 override them). Those need dual-source
// blending (the game's water x a per-channel transmittance, plus light): without it only the reflections.
static struct { bool pending; uint64_t frame; NvnTexture *target, *depth; TexBackend dsnap, csnap; } water;

static int water_env(const char *name) {  // (-1: not set)
    const char *e = getenv(name);
    return e ? atoi(e) != 0 : -1;
}
static bool water_reflect_on(void) {
    static int env = -2;
    if (env == -2) env = water_env("LARECOMP_WATER");
    return env >= 0 ? env : g_settings.water_ssr;
}
static int water_extras(void) {  // bits: 1 depth, 2 caustics, 4 vivid (0 without dual-source blending)
    static int dual = -1, ed = -2, ec, ev;
    if (dual < 0) {
        VkPhysicalDeviceFeatures f;
        vkGetPhysicalDeviceFeatures(g_vk.phys, &f);
        dual = f.dualSrcBlend != 0;
    }
    if (ed == -2) ed = water_env("LARECOMP_WATER_DEPTH"), ec = water_env("LARECOMP_WATER_CAUSTICS"), ev = water_env("LARECOMP_WATER_VIVID");
    if (!dual) return 0;
    return (ed >= 0 ? ed : g_settings.water_depth) | (ec >= 0 ? ec : g_settings.water_caustics) << 1 |
           (ev >= 0 ? ev : g_settings.water_vivid) << 2;
}
static bool water_on(void) { return water_reflect_on() || water_extras(); }  // (the water pass runs)
// How the reflections are found: 0 the screen search alone, 1 the picture mirrored in the water's plane (SSPR, below the
// planar ones), 2 the scene drawn again from a mirrored camera (planar). F1 > Reflection method; LARECOMP_WATER_MODE
static int water_mode(void) {
    static int env = -2;
    if (env == -2) { const char *e = getenv("LARECOMP_WATER_MODE"); env = e ? atoi(e) : -1; }
    return !water_reflect_on() ? 0 : env >= 0 ? env : g_settings.water_mode;
}

// Scarlet / Violet: the reflection cube off the picture is the one the game's own water draws bind (its biggest: a
// 256 x 256 BC6H sky, clouds and all), not the lighting's (a 128 x 128 probe of the place). Kept per frame by
// water_surface_cube from the water's draws (draw_); none: the lighting's, as in Z-A. LARECOMP_WATER_CUBE=0: the
// lighting's always.
static VkImageView water_cube;
static uint32_t water_cube_w;
static void water_surface_cube(Prog *pr) {
    static int env = -2;
    if (env == -2) env = water_env("LARECOMP_WATER_CUBE");
    if (env == 0) return;
    for (int i = 0; i < pr->fi->ntex; i++) {
        NvnTexEntry *e = gtao_tex_entry(pr->fi, i);
        if (e && pr->fi->tex[i].type == TEXTYPE_CUBE && e->tex->width > water_cube_w) {
            VkImageView v = vk_view(e->tex, &e->view, e->has_view, false, TEXTYPE_CUBE);
            if (v) water_cube = v, water_cube_w = e->tex->width;
        }
    }
}

// A water surface: blended onto the lit image, writing depth, reading a depth texture
// (and in the picture's own scene pass - fsr2_scene_pass: its depth, not square. Scarlet / Violet draws its water again
// every 8th frame for a square view of its own - a reflection probe's -: the water pass ran there too, and its level
// and sky readings - another camera's - threw the mirror's plane to another water for that frame: the far water and
// a waterfall's crest flickered every 8 frames)
static bool water_is_surface(Prog *pr) {
    if (!water_on() || !gtao.target || st.rt.color[0] != gtao.target || !st.color_state[0] || !st.ds[0] || !st.ds[1] ||
        !st.rt.depth || !pr || !pr->fi || !fsr2_scene_pass())
        return false;
    for (int i = 0; i < pr->fi->ntex; i++) {
        NvnTexEntry *e = gtao_tex_entry(pr->fi, i);
        if (e && renderer_format_is_depth(e->tex->format)) return true;
    }
    return false;
}
// d: an image like s (its format and size: made again when they change); copies s's aspect (colour or depth) into it
static bool water_copy(TexBackend *d, TexBackend *s, VkImageAspectFlags aspect) {
    if (d->image && (d->width != s->width || d->height != s->height || d->format != s->format)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = s->format,
                                .extent = {s->width, s->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL) != VK_SUCCESS) { d->image = VK_NULL_HANDLE; return false; }
        d->format = s->format; d->width = s->width; d->height = s->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = s->aspect; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    VkImageCopy c = {.srcSubresource = {aspect, 0, 0, 1}, .dstSubresource = {aspect, 0, 0, 1}, .extent = {s->width, s->height, 1}};
    rec_CmdCopyImage(vk_cmd(), s->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    return true;
}

static void water_before(void) {
    extern uint64_t renderer_exec_frame(void);
    uint64_t frame = renderer_exec_frame();
    if (water.frame == frame + 1) return;  // (the frame's first water only)
    water.frame = frame + 1;
    TexBackend *db = st.rt.depth->backend, *cb = gtao.target->backend;
    if (!db || !cb || !db->image || !cb->image || db->levels < 1) return;
    vk_end_pass();
    vk_full_barrier(vk_cmd());
    if (!water_copy(&water.dsnap, db, VK_IMAGE_ASPECT_DEPTH_BIT) || !water_copy(&water.csnap, cb, VK_IMAGE_ASPECT_COLOR_BIT)) return;
    vk_full_barrier(vk_cmd());
    water.pending = true, water.target = gtao.target, water.depth = st.rt.depth;
}

// Scarlet / Violet: the reflection cube isn't at the brightness of the sky the game draws - one cube for the time of day
// and weather it was made for, and at night 20-30 times brighter than the sky on screen (the water went white where the
// mirror ray leaves the picture). Its brightness is matched to the drawn sky's: after the water pass, water_sky.frag
// compares a grid of the picture's sky pixels (nothing drawn: depth 0) with the cube in their directions into a 1 x 1
// image, read back a few frames later as planar_level_probe's height is; the scale follows that ratio smoothly (no
// sky in the picture: the last one stays). LARECOMP_WATER_SKY_MATCH=0: the cube as it is (Z-A's, where it matches);
// LARECOMP_WATER_SKY_LOG=1 logs every measurement.
static struct { VkBuffer buf; VmaAllocation mem; float *ptr; uint64_t seq[4]; int next; float scale; bool have; } wsky;
static bool water_sky_match_on(void) {
    static int env = -2;
    if (env == -2) env = water_env("LARECOMP_WATER_SKY_MATCH");
    return env != 0;
}
static float water_sky_scale(void) { return water_sky_match_on() && wsky.have ? wsky.scale : 1.0f; }
static void water_sky_probe(VkImageView dnow, VkImageView dpre, VkImageView cpre, VkImageView probe, const float *pc) {
    if (!water_sky_match_on()) return;
    if (!wsky.buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4 * 16, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &wsky.buf, &wsky.mem, &info) != VK_SUCCESS) { wsky.buf = VK_NULL_HANDLE; return; }
        wsky.ptr = info.pMappedData;
    }
    uint64_t done = vk_completed_seq();
    for (int i = 0; i < 4; i++) {
        int s = (wsky.next + i) & 3;
        if (!wsky.seq[s] || wsky.seq[s] > done) continue;
        vmaInvalidateAllocation(g_vk.vma, wsky.mem, (VkDeviceSize)s * 16, 16);
        const float *v = wsky.ptr + s * 4;  // (the sky's summed luminance, the cube's, how many sky pixels)
        static int dlog = -1;
        if (dlog < 0) dlog = getenv("LARECOMP_WATER_SKY_LOG") != NULL;
        if (dlog) LOG("[water] sky %.3f cube %.3f n %.0f\n", v[0], v[1], v[2]);
        // (16 sky pixels at least; and not nearly the whole grid - a frame with hardly anything drawn yet, as while an
        // area loads: water in the picture leaves far less sky than that)
        if (v[2] >= 16 && v[2] < 0.9f * 48 * 27 && v[1] > 1e-6f && v[0] > 0) {
            float r = v[0] / v[1];
            r = r < 1e-3f ? 1e-3f : r > 4.0f ? 4.0f : r;
            wsky.scale = wsky.have ? wsky.scale + (r - wsky.scale) * 0.1f : r;  // (the first at once; then ~10 frames)
            static int nlog;
            if (!wsky.have || (nlog < 4 && fabsf(r - wsky.scale) > 0.5f * wsky.scale))
                nlog++, LOG("[water] reflection cube matched to the drawn sky: x%.4f (%d sky samples)\n", r, (int)v[2]);
            wsky.have = true;
        }
        wsky.seq[s] = 0;
    }
    static TexBackend img;
    if (!img.image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                                .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(g_vk.vma, &ci, &ai, &img.image, &img.mem, NULL) != VK_SUCCESS) { img.image = VK_NULL_HANDLE; return; }
        img.format = VK_FORMAT_R32G32B32A32_SFLOAT; img.width = img.height = 1; img.depth = img.layers = img.levels = 1;
        img.samples = 1; img.aspect = VK_IMAGE_ASPECT_COLOR_BIT; img.base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), img.image, img.aspect);
    }
    float p[32];
    memcpy(p, pc, sizeof p);
    post_pc_big = p;
    VkImageView in[4] = {dpre, dnow, cpre, probe};
    post_mips_last = true;  // (the cube's smaller levels: its sky without the stars' and lamps' single texels)
    post_pass_ex(vk_cmd(), "water_sky.frag", in, 4, &img, NULL, true, NULL, 0);
    post_pc_big = NULL, post_mips_last = false;
    int s = wsky.next;
    wsky.next = (s + 1) & 3;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = (VkDeviceSize)s * 16, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {1, 1, 1}};
    rec_CmdCopyImageToBuffer(cb, img.image, VK_IMAGE_LAYOUT_GENERAL, wsky.buf, 1, &c);
    vk_full_barrier(cb);
    wsky.seq[s] = vk_pending_seq();
}

static bool planar_inputs(VkImageView *lit, VkImageView *depth, float *y);  // (planar water reflections: below)
static bool sspr_run(VkImageView dpre, VkImageView cpre, const float *wpc, VkImageView *out, float *y);  // (and SSPR)
static void water_apply_pending(void) {
    if (!water.pending || st.rt.color[0] == water.target) return;  // (still drawing onto the lit image: later)
    water.pending = false;
    TexBackend *tb = water.target ? water.target->backend : NULL, *db = water.depth ? water.depth->backend : NULL;
    if (!tb || !db || !tb->image || !db->image) return;
    static const VkComponentMapping id = {0};
    VkImageView dnow = vk_image_view(db, VK_IMAGE_VIEW_TYPE_2D, db->format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkImageView dpre = vk_image_view(&water.dsnap, VK_IMAGE_VIEW_TYPE_2D, water.dsnap.format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_DEPTH_BIT);
    VkImageView cpre = post_view(&water.csnap), probe = water_cube ? water_cube : water_probe;
    water_probe = water_cube = VK_NULL_HANDLE, water_cube_w = 0;  // (this frame's: their textures may be gone by the next)
    if (!dnow || !dpre || !cpre || !probe) return;  // (no reflection cube in this frame's lighting: none)
    extern uint64_t renderer_exec_frame(void);
    float pc[32];
    memcpy(pc, gtao.pc, sizeof pc);
    pc[19] = water_sky_scale();  // (cam.w, which the shaders don't read: the cube's brightness to the drawn sky's)
    int extras = water_extras();
    pc[20] = water_reflect_on() ? gtao_env("LARECOMP_WATER_STRENGTH", 1.0f) : 0;
    pc[21] = (float)extras;  // (bits: 1 depth, 2 caustics, 4 vivid)
    pc[23] = (float)(renderer_exec_frame() % 36000) / 60.0f;  // (the ripples' time, in seconds)
    pc[24] = pc[26] = (float)tb->width, pc[25] = pc[27] = (float)tb->height;
    pc[28] = gtao_env("LARECOMP_WATER_ABSORB", 1.0f);     // (depth: how fast the light fades per metre, x)
    pc[29] = gtao_env("LARECOMP_WATER_CAUSTIC_K", 1.0f);  // (caustics' brightness, x)
    pc[30] = gtao_env("LARECOMP_WATER_DEBUG", 0);
    post_pc_big = pc;
    // (inputs 4, 5: the mirror image and its depth - planar_inputs -, or stand-ins the shader doesn't read: bit 8 off)
    VkImageView in[6] = {dnow, dpre, cpre, probe, cpre, dnow};
    if (planar_inputs(&in[4], &in[5], &pc[31])) pc[21] += 8;
    else if (sspr_run(dpre, cpre, pc, &in[4], &pc[31])) pc[21] += 16;
    post_mips_last = true;  // (the cube: its smaller levels for the blur and the cap)
    // (kind 6: dual-source - the game's water x a per-channel transmittance, plus reflection and light; 5: alpha)
    post_pass_ex(vk_cmd(), extras ? "water_ssr2.frag" : "water_ssr.frag", in, 6, tb, NULL, false, NULL,
                 pc[30] != 0 ? 0 : extras ? 6 : 5);
    post_pc_big = NULL, post_mips_last = false;
    static int nlog;
    if (nlog++ < 2) LOG("[water] on: %ux%u, reflection strength %.2f, depth %d, caustics %d, vivid %d\n", tb->width, tb->height,
                        pc[20], extras & 1, extras >> 1 & 1, extras >> 2 & 1);
    extern void planar_level_probe(VkImageView dnow, VkImageView dpre, const float *pc);
    planar_level_probe(dnow, dpre, pc);
    water_sky_probe(dnow, dpre, cpre, probe, pc);
}

// ================================================================== planar water reflections
// The screen-space reflections above only find what the picture shows: not the underside of a bridge over the water,
// not the back of what faces the camera. Z-A's water is level, so its exact mirror image is the scene drawn from a
// camera mirrored in the water's plane. Each G-buffer draw (4+ colour targets, the scene's depth: planar_capture) is
// kept - its state as the state cache had it (ss), its descriptor sets with the vertex set made again around a
// mirrored copy of its camera (constant buffer 5's rows 0-3: mv_note_draw), its call - and when the G-buffer pass ends
// (begin_pass), all of them are drawn again into images like its targets at a fraction of the size, with the faces'
// winding flipped. The mirrored camera's depth row is replaced by one that clips everything below the water (an
// oblique near plane: depth = 1 - k (y - h) / w, reversed and monotonic along each ray), so the pool's floor doesn't
// hide the world above it, and no shader changes. Only within one submission: when one came between the draws and
// the replay, that frame has none. F1 > Reflection method: Mirrored scene (LARECOMP_WATER_MODE=2). The water shader
// reads the lit image at each water pixel's own uv (a point of the plane is where it was), water at another level keeps
// the screen search; the game's light volumes aren't drawn again, so lamps light the mirror less than the scene.
// LARECOMP_PLANAR_Y=<height> (the water's plane), _SCALE (0.5), _DUMP=<frame> (the images -> build/planar_<i>.png).
#define PL_K 0.25f  // (the clip row's slope: k (y - h) must stay below w for what is seen; 4 x the height over w is cut)
typedef struct {
    Pipe *pp;
    VkPipelineLayout layout;
    uint32_t streams;
    __typeof__(ss) s;
    VkDescriptorSet sets[2];
    uint32_t ndyn, dyn[2 * MAX_BIND];
    bool gmem, gdirect;
    VkDescriptorSet gset;
    uint32_t gindex, gd[GMEM_DIRECT_MAX * 4];
    PlDrawCall dc;
    uint64_t ord;  // (the draw's place in the game's order: draw_ord)
} PlDraw;
static uint64_t draw_ord(void);  // (below, with the shadow workers)
typedef struct { uint64_t ord; int i; } PlOrd;  // (sorted instead of the draws: each holds a state copy of KBs)
static int pl_ord_cmp(const void *a, const void *b) {
    uint64_t x = ((const PlOrd *)a)->ord, y = ((const PlOrd *)b)->ord;
    return x < y ? -1 : x > y;
}
static struct {
    PlDraw *d;
    int n, cap, dropped;
    uint64_t frame, done, seq;  // (the frame captured, the one replayed, the submission the draws are in)
    NvnTexture *rt0;            // (the G-buffer's first target)
    VkFormat color[8], depth, stencil;
    uint32_t ncolor, w, h;
    float y;                    // the water's plane
    struct { uint64_t src; uint64_t size; float data[1024]; } cb[8];  // (mirrored copies of the frame's constant buffers 5)
    int ncb;
    TexBackend img[15];         // (0-7 like the G-buffer's targets, 8 its depth, 9-11 like the lighting's targets, 12 that
                                // depth as the unclipped mirror's: R32F; SSPR: 13 the projection hash, 14 its colours)
    float scale;
    NvnTexture *gtex[8], *gdepth;  // (the G-buffer's targets: the lighting reads them as textures)
    float vpm[16], vpm_std[16]; // (the mirrored camera - planar_matrix of the first constant buffer 5 -, and unclipped)
    uint64_t lit_done;          // (the frame whose lighting was drawn again: planar_light)
    float lcb[1024];            // (the lighting's camera, mirrored)
} pl;
static struct { VkBuffer buf; VmaAllocation mem; float *ptr; uint64_t seq[4], seen; int next; bool have; } pl_lv;  // (planar_level_probe)

// For the water pass: this frame's mirror image, its depth (image 12: the game's projection, so the shader's own
// wpos gives the mirrored point) and the plane they were drawn for. False: no reflection was drawn this frame.
static bool planar_inputs(VkImageView *lit, VkImageView *depth, float *y) {
    extern uint64_t renderer_exec_frame(void);
    if (!planar_on() || pl.lit_done != renderer_exec_frame() || !pl.img[9].image || !pl.img[12].image) return false;
    *lit = post_view(&pl.img[9]), *depth = post_view(&pl.img[12]), *y = pl.y;
    return true;
}

static bool planar_on(void) { return water_mode() == 2; }

// The pass's first colour target when it has 4 or more (the G-buffer: its slot 0 is empty), else NULL - a depth-only
// pass before it binds the slots with nothing in them
static NvnTexture *planar_gbuf_target(void) {
    NvnTexture *first = NULL;
    int n = 0;
    for (int i = 0; i < 8; i++) if (st.rt.color[i]) { if (!first) first = st.rt.color[i]; n++; }
    return n >= 4 ? first : NULL;
}

// m: a view-projection (by columns); out: the same seen in the mirror of the plane y = h, its depth row clipping below
// it (clip false: the depth row as the game's - the lighting's depth, planar_light)
static void planar_matrix(const float *m, float h, float *out, bool clip) {
    float r[16];
    for (int i = 0; i < 4; i++) {
        r[0 + i] = m[0 + i];                       // (x R: column 1 negated, column 3 += 2h column 1)
        r[4 + i] = -m[4 + i];
        r[8 + i] = m[8 + i];
        r[12 + i] = m[12 + i] + 2 * h * m[4 + i];
    }
    const float plane[4] = {0, 1, 0, -h};
    memcpy(out, r, sizeof r);
    if (clip)
        for (int c = 0; c < 4; c++) out[c * 4 + 2] = r[c * 4 + 3] - PL_K * plane[c];
}

static pthread_mutex_t pl_mu = PTHREAD_MUTEX_INITIALIZER;
static void planar_capture_(Prog *pr, Pipe *pp, uint32_t streams, const VkDescriptorSet sets[2], uint32_t ndyn, const uint32_t *dyn,
                            uint32_t nv_dyn, bool gmem, const PlDrawCall *dc, NvnTexture *rt, Range c5, uint64_t f);
static void planar_capture(Prog *pr, Pipe *pp, uint32_t streams, const VkDescriptorSet sets[2], uint32_t ndyn, const uint32_t *dyn,
                           uint32_t nv_dyn, bool gmem, const PlDrawCall *dc) {
    NvnTexture *rt = planar_gbuf_target();
    if (!rt || !fsr2_scene_pass()) return;
    Range c5 = st.ubuf[0][2];
    if (!((pr->vi->cbuf_mask >> 5) & 1) || !c5.addr || c5.size < 64) return;  // (no camera of its own to mirror)
    extern uint64_t renderer_exec_frame(void);
    uint64_t f = renderer_exec_frame();
    // (no water worth a mirror lately - planar_level_probe -: back before the lock. Taken for each of the G-buffer's
    // ~1500 draws on 3 workers at once, with nothing to keep, it cost a town about a frame a second)
    static int fixed_y = -1;
    if (fixed_y < 0) fixed_y = getenv("LARECOMP_PLANAR_Y") != NULL;
    if (!fixed_y && (!pl_lv.have || f - pl_lv.seen > 120)) return;
    pthread_mutex_lock(&pl_mu);  // (shadow workers draw the G-buffer too: one capture at a time)
    planar_capture_(pr, pp, streams, sets, ndyn, dyn, nv_dyn, gmem, dc, rt, c5, f);
    pthread_mutex_unlock(&pl_mu);
}
static void planar_capture_(Prog *pr, Pipe *pp, uint32_t streams, const VkDescriptorSet sets[2], uint32_t ndyn, const uint32_t *dyn,
                            uint32_t nv_dyn, bool gmem, const PlDrawCall *dc, NvnTexture *rt, Range c5, uint64_t f) {
    if (pl.frame != f) {  // (the frame's first G-buffer draw)
        ring_unpin();
        ring_lost = false;
        pl.frame = f, pl.n = 0, pl.ncb = 0, pl.seq = submit_seq, pl.rt0 = rt;
        pl.ncolor = pass_ncolor, pl.depth = pass_depth, pl.stencil = pass_stencil, pl.w = pass_w, pl.h = pass_h;
        memcpy(pl.color, pass_color, sizeof pl.color);
        memcpy(pl.gtex, st.rt.color, sizeof pl.gtex), pl.gdepth = st.rt.depth;
        if (getenv("LARECOMP_PLANAR_Y")) pl.y = gtao_env("LARECOMP_PLANAR_Y", 0.0f), pl_lv.have = true;
        pl.scale = gtao_env("LARECOMP_PLANAR_SCALE", 0.5f);
    }
    // (no water height yet; or no water in the picture for 2 s - its draws, or too few of its pixels -: nothing to mirror,
    // and walking the city costs nothing. Coming back to it, the screen's reflections until the height is read again)
    static int fixed_y = -1;
    if (fixed_y < 0) fixed_y = getenv("LARECOMP_PLANAR_Y") != NULL;
    if (!pl_lv.have || (!fixed_y && f - pl_lv.seen > 120)) return;
    if (rt != pl.rt0 || ring_lost) return;  // (another such pass in the frame: the first only)
    pthread_mutex_lock(&ring_mu);
    if (!ring_pin) ring_pin = vk_pending_seq();  // (the copies of what is kept stay until the replay: planar_replay_pending)
    pthread_mutex_unlock(&ring_mu);
    uint64_t size = c5.size > sizeof pl.cb[0].data ? sizeof pl.cb[0].data : c5.size;
    int ci = 0;
    while (ci < pl.ncb && (pl.cb[ci].src != c5.addr || pl.cb[ci].size != size)) ci++;
    if (ci == pl.ncb) {
        if (pl.ncb == 8) return;
        pl.ncb++;
        pl.cb[ci].src = c5.addr, pl.cb[ci].size = size;
        memcpy(pl.cb[ci].data, (const void *)(uintptr_t)c5.addr, size);
        planar_matrix((const float *)(uintptr_t)c5.addr, pl.y, pl.cb[ci].data, true);
        if (ci == 0) {
            memcpy(pl.vpm, pl.cb[0].data, sizeof pl.vpm);
            planar_matrix((const float *)(uintptr_t)c5.addr, pl.y, pl.vpm_std, false);
        }
    }
    // the vertex set again, around the mirrored copy (its range as the original's: the copy is that long)
    st.ubuf[0][2].addr = (uint64_t)(uintptr_t)pl.cb[ci].data, st.ubuf[0][2].size = size;
    uint32_t mdyn[2 * MAX_BIND], mn = 0;
    VkDescriptorSet mset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, NULL, NULL, 0, 0, 0, NULL, mdyn, &mn);
    st.ubuf[0][2] = c5;
    if (mn != nv_dyn || ndyn > 2 * MAX_BIND) return;
    if (pl.n == pl.cap) {
        int cap = pl.cap ? pl.cap * 2 : 1024;
        PlDraw *d = realloc(pl.d, (size_t)cap * sizeof *d);
        if (!d) return;
        pl.d = d, pl.cap = cap;
    }
    PlDraw *d = &pl.d[pl.n++];
    d->pp = pp, d->layout = pr->layout, d->streams = streams, d->s = ss;
    d->sets[0] = mset, d->sets[1] = sets[1];
    d->ndyn = ndyn;
    memcpy(d->dyn, mdyn, mn * sizeof mdyn[0]);
    memcpy(d->dyn + mn, dyn + nv_dyn, (ndyn - nv_dyn) * sizeof dyn[0]);
    d->gmem = gmem;
    if (gmem) d->gset = gm_last.set, d->gindex = gm_last.index, d->gdirect = gm_last.direct, memcpy(d->gd, gm_last.gd, sizeof d->gd);
    d->dc = *dc;
    d->ord = draw_ord();
}

// i: 0-7 a colour target, 8 the depth; made again when the format or size changes
static TexBackend *planar_image(int i, VkFormat f, uint32_t w, uint32_t h) {
    TexBackend *d = &pl.img[i];
    if (d->image && (d->width != w || d->height != h || d->format != f)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        bool ds = i == 8;
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = i == 13 ? VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                                         : VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         (ds ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL) != VK_SUCCESS) { d->image = VK_NULL_HANDLE; return NULL; }
        d->format = f; d->width = w; d->height = h; d->depth = 1; d->layers = 1; d->levels = 1; d->samples = 1;
        d->aspect = !ds ? VK_IMAGE_ASPECT_COLOR_BIT
                  : VK_IMAGE_ASPECT_DEPTH_BIT | (pl.stencil != VK_FORMAT_UNDEFINED ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
        d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    return d;
}

// The water's height for the mirror (LARECOMP_PLANAR_Y overrides it): after the water pass, water_level.frag averages
// the world height of a grid of the picture's water pixels (where the depth changed) into a 1 x 1 image; that is copied
// into one of four host-visible slots and read once its submission is done - a few frames later, which the level
// (still water) allows. No water seen: the last height stays.
void planar_level_probe(VkImageView dnow, VkImageView dpre, const float *pc) {
    extern uint64_t renderer_exec_frame(void);
    if (water_mode() == 0) return;
    if (!pl_lv.buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4 * 16, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &pl_lv.buf, &pl_lv.mem, &info) != VK_SUCCESS) { pl_lv.buf = VK_NULL_HANDLE; return; }
        pl_lv.ptr = info.pMappedData;
    }
    uint64_t done = vk_completed_seq();  // (the oldest slot whose copy the GPU finished: the newest such height)
    for (int i = 0; i < 4; i++) {
        int s = (pl_lv.next + i) & 3;
        if (!pl_lv.seq[s] || pl_lv.seq[s] > done) continue;
        vmaInvalidateAllocation(g_vk.vma, pl_lv.mem, (VkDeviceSize)s * 16, 16);
        const float *v = pl_lv.ptr + s * 4;
        static int dlog = -1;
        if (dlog < 0) dlog = getenv("LARECOMP_PLANAR_LOG") != NULL;  // (each reading: the level, its samples)
        if (dlog) LOG("[planar] level %.4f (%g samples) frame %llu\n", v[0], v[1], (unsigned long long)renderer_exec_frame());
        // (enough water samples. Scarlet / Violet's sea moves with its waves, and the grid's samples with the camera's
        // jitter: the readings go 27.06 - 27.20 from frame to frame, and the mirror, its plane following them, made the
        // far water flicker with Mirrored picture / scene. Followed slowly; a jump of over a metre is other water, taken
        // once it has held for 12 readings running - a single reading of it, from a frame whose samples fell that way,
        // threw the mirror's plane to it for that frame)
        if (v[1] >= 4) {
            static float cand;
            static int cand_n;
            if (!pl_lv.have) pl.y = v[0];
            else if (fabsf(v[0] - pl.y) <= 1.0f) pl.y += (v[0] - pl.y) * 0.05f, cand_n = 0;
            else {
                if (cand_n && fabsf(v[0] - cand) <= 1.0f) cand += (v[0] - cand) * 0.2f, cand_n++;
                else cand = v[0], cand_n = 1;
                if (cand_n >= 12) pl.y = cand, cand_n = 0;
            }
            // (the mirrored scene, drawn again, only for water worth it: 16 of the grid's 1296 samples, ~1% of the
            // picture. A coastal town's fountain or puddles - 4 or 5 - had every frame's G-buffer, ~1500 draws, drawn
            // again for a few pixels: 54 fps there against 59 with the screen search, which they fall back to)
            pl_lv.have = true;
            if (v[1] >= 16) pl_lv.seen = renderer_exec_frame();
        }
        pl_lv.seq[s] = 0;
    }
    static TexBackend img;
    if (!img.image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                                .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(g_vk.vma, &ci, &ai, &img.image, &img.mem, NULL) != VK_SUCCESS) { img.image = VK_NULL_HANDLE; return; }
        img.format = VK_FORMAT_R32G32B32A32_SFLOAT; img.width = img.height = 1; img.depth = img.layers = img.levels = 1;
        img.samples = 1; img.aspect = VK_IMAGE_ASPECT_COLOR_BIT; img.base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), img.image, img.aspect);
    }
    float p[32];
    memcpy(p, pc, sizeof p);
    post_pc_big = p;
    VkImageView in[2] = {dnow, dpre};
    post_pass_ex(vk_cmd(), "water_level.frag", in, 2, &img, NULL, true, NULL, 0);
    post_pc_big = NULL;
    int s = pl_lv.next;
    pl_lv.next = (s + 1) & 3;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkBufferImageCopy c = {.bufferOffset = (VkDeviceSize)s * 16, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {1, 1, 1}};
    rec_CmdCopyImageToBuffer(cb, img.image, VK_IMAGE_LAYOUT_GENERAL, pl_lv.buf, 1, &c);
    vk_full_barrier(cb);
    pl_lv.seq[s] = vk_pending_seq();
}

// One captured draw into the open pass (its images w x h at s times the game's). flip: the mirrored winding. e: what this
// pass has so far, zeroed at its start - the state cache's filtering doesn't know this pass, and replaying everything
// for each draw was nearly all of the mirrored scene's CPU cost (LARECOMP_VK_STATE_FILTER=0: everything, as draw_ does)
typedef struct { const PlDraw *prev; VkBuffer ib; VkDeviceSize io; VkIndexType it; uint32_t vb_known; VkBuffer vb[17]; VkDeviceSize vo[17]; } PlEmit;
static void planar_emit(VkCommandBuffer cb, const PlDraw *d, float s, uint32_t w, uint32_t h, bool flip, PlEmit *e) {
    const __typeof__(ss) *q = &d->s;
    bool f = ss_enabled();
    const PlDraw *pd = f ? e->prev : NULL;
    const __typeof__(ss) *p = pd ? &pd->s : NULL;
#define PL_NEW(x) (!p || p->x != q->x)
#define PL_NEWM(x) (!p || memcmp(&p->x, &q->x, sizeof q->x))
    if (!pd || pd->pp != d->pp) rec_CmdBindPipelineObj(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, d->pp);
    if (g_vk.vids && PL_NEWM(vi)) set_vertex_input(cb, &q->vi);
    if (PL_NEW(dt)) rec_CmdSetDepthTestEnable(cb, q->dt);
    if (PL_NEW(dw)) rec_CmdSetDepthWriteEnable(cb, q->dw);
    if (PL_NEW(dop)) rec_CmdSetDepthCompareOp(cb, q->dop);
    if (PL_NEW(sten)) rec_CmdSetStencilTestEnable(cb, q->sten);
    for (int fc = 0; fc < 2; fc++) {
        if (!PL_NEWM(sface[fc])) continue;
        VkStencilFaceFlags face = fc ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
        const uint32_t *v = q->sface[fc];
        rec_CmdSetStencilOp(cb, face, (VkStencilOp)v[0], (VkStencilOp)v[1], (VkStencilOp)v[2], (VkCompareOp)v[3]);
        rec_CmdSetStencilCompareMask(cb, face, v[4]);
        rec_CmdSetStencilWriteMask(cb, face, v[5]);
        rec_CmdSetStencilReference(cb, face, v[6]);
    }
    if (PL_NEWM(bc)) rec_CmdSetBlendConstants(cb, q->bc);
    if (PL_NEW(cull)) rec_CmdSetCullMode(cb, q->cull);
    if (PL_NEW(ff)) rec_CmdSetFrontFace(cb, !flip ? q->ff : q->ff == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE);
    if (PL_NEW(topo)) rec_CmdSetPrimitiveTopology(cb, q->topo);
    if (PL_NEW(pre)) rec_CmdSetPrimitiveRestartEnable(cb, q->pre);
    if (PL_NEW(dbe)) rec_CmdSetDepthBiasEnable(cb, q->dbe);
    if (PL_NEWM(db)) rec_CmdSetDepthBias(cb, q->db[1], q->db[2], q->db[0]);
    if (PL_NEWM(vp)) {
        VkViewport vp = q->vp;
        vp.x *= s, vp.y *= s, vp.width *= s, vp.height *= s;
        rec_CmdSetViewport(cb, 0, 1, &vp);
    }
    if (PL_NEWM(sc)) {
        int32_t x0 = (int32_t)floorf(q->sc.offset.x * s), y0 = (int32_t)floorf(q->sc.offset.y * s);
        int32_t x1 = (int32_t)ceilf((q->sc.offset.x + (float)q->sc.extent.width) * s), y1 = (int32_t)ceilf((q->sc.offset.y + (float)q->sc.extent.height) * s);
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > (int32_t)w) x1 = (int32_t)w;
        if (y1 > (int32_t)h) y1 = (int32_t)h;
        VkRect2D sc = {{x0, y0}, {(uint32_t)(x1 > x0 ? x1 - x0 : 0), (uint32_t)(y1 > y0 ? y1 - y0 : 0)}};
        rec_CmdSetScissor(cb, 0, 1, &sc);
    }
#undef PL_NEW
#undef PL_NEWM
    bool same_layout = pd && pd->layout == d->layout;
    if (!same_layout || pd->sets[0] != d->sets[0] || pd->sets[1] != d->sets[1] || pd->ndyn != d->ndyn ||
        memcmp(pd->dyn, d->dyn, d->ndyn * sizeof d->dyn[0]))
        rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, d->layout, 0, 2, d->sets, d->ndyn, d->ndyn ? d->dyn : NULL);
    if (d->gmem) {
        bool same = same_layout && pd->gmem && pd->gset == d->gset && pd->gindex == d->gindex;
        if (!same) rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, d->layout, d->gindex, 1, &d->gset, 0, NULL);
        if (d->gdirect && !(same && pd->gdirect && !memcmp(pd->gd, d->gd, sizeof d->gd)))
            rec_CmdPushConstants(cb, d->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof d->gd, d->gd);
    }
    if ((d->dc.kind == 1 || (d->dc.kind == 2 && d->dc.indexed)) && (!f || e->ib != q->ib || e->io != q->io || e->it != q->it)) {
        rec_CmdBindIndexBuffer(cb, q->ib, q->io, q->it);
        e->ib = q->ib, e->io = q->io, e->it = q->it;
    }
    for (int i = 0; i < 17; i++) {
        if (!(d->streams & (1u << i))) continue;
        if (f && (e->vb_known & (1u << i)) && e->vb[i] == q->vb[i] && e->vo[i] == q->vo[i]) continue;
        rec_CmdBindVertexBuffers(cb, (uint32_t)i, 1, &q->vb[i], &q->vo[i]);
        e->vb_known |= 1u << i, e->vb[i] = q->vb[i], e->vo[i] = q->vo[i];
    }
    if (d->dc.kind == 2) rec_CmdDrawIndirect(cb, d->dc.buf, d->dc.off, 1, d->dc.a, d->dc.indexed);
    else if (d->dc.kind == 1) rec_CmdDrawIndexed(cb, d->dc.a, d->dc.b, 0, d->dc.d, d->dc.e);
    else rec_CmdDraw(cb, d->dc.a, d->dc.b, d->dc.c, d->dc.e);
    e->prev = d;
}

static void planar_replay_pending(void) {
    extern uint64_t renderer_exec_frame(void);
    uint64_t f = renderer_exec_frame();
    if (ring_pin && pl.frame != f) ring_unpin(), pl.n = 0;  // (a frame's draws that never got drawn again)
    if (!pl.n || pl.done == pl.frame || pl.frame != f) return;
    if (planar_gbuf_target() == pl.rt0) return;  // (a new pass on the same targets: still the G-buffer)
    int n = pl.n;
    pl.n = 0;
    if (ring_lost) return;
    // (the G-buffer pass spans submissions: its copies are pinned in the ring - ring_pin -, and its descriptor sets
    // live as long as their submission's pool, until SUBS submissions later)
    if (submit_seq - pl.seq >= SUBS - 4) {
        if (pl.dropped++ < 10) LOG("[planar] frame %llu: %llu submissions during the G-buffer pass, no reflection\n",
                                   (unsigned long long)pl.frame, (unsigned long long)(submit_seq - pl.seq));
        ring_unpin();
        return;
    }
    float s = pl.scale > 0.05f && pl.scale <= 1 ? pl.scale : 0.5f;
    uint32_t w = (uint32_t)(pl.w * s), h = (uint32_t)(pl.h * s);
    if (w < 8 || h < 8 || pl.depth == VK_FORMAT_UNDEFINED) { ring_unpin(); return; }
    VkRenderingAttachmentInfo ca[8] = {0}, da = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    for (uint32_t i = 0; i < pl.ncolor; i++) {
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        if (pl.color[i] == VK_FORMAT_UNDEFINED) continue;
        TexBackend *b = planar_image((int)i, pl.color[i], w, h);
        if (!b) { ring_unpin(); return; }
        ca[i].imageView = post_view(b), ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    TexBackend *db = planar_image(8, pl.depth, w, h);
    TexBackend *zd = planar_image(12, VK_FORMAT_R32_SFLOAT, w, h);  // (planar_light_pending; made here, outside any pass)
    if (!db || !zd) { ring_unpin(); return; }
    static const VkComponentMapping id = {0};
    da.imageView = vk_image_view(db, VK_IMAGE_VIEW_TYPE_2D, db->format, 0, 1, 0, 1, id, db->aspect);
    da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;  // (cleared to 0: reversed, far)
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = pl.ncolor, .pColorAttachments = ca, .pDepthAttachment = &da,
                          .pStencilAttachment = pl.stencil != VK_FORMAT_UNDEFINED ? &da : NULL};
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    rec_CmdBeginRendering(cb, &ri);
    PlEmit em = {0};
    // (in the game's order: the shadow workers capture their parts at once, in any order, and the waterfall's
    // overlapping sheets, drawn again in another order each frame, flickered triangle by triangle in the mirror. An
    // index is sorted, not the draws: moving ~1500 of them, each with its state copy, cost a town 18 fps)
    static PlOrd *ord;
    static int ord_cap;
    if (n > ord_cap) {
        PlOrd *o = realloc(ord, (size_t)n * sizeof *o);
        if (o) ord = o, ord_cap = n;
    }
    bool sorted = n <= ord_cap;
    if (sorted) {
        for (int k = 0; k < n; k++) ord[k].ord = pl.d[k].ord, ord[k].i = k;
        qsort(ord, (size_t)n, sizeof *ord, pl_ord_cmp);
    }
    for (int k = 0; k < n; k++) planar_emit(cb, &pl.d[sorted ? ord[k].i : k], s, w, h, true, &em);
    rec_CmdEndRendering(cb);
    ss_forget();
    vk_full_barrier(cb);
    ring_unpin();  // (its copies: until this submission is done)
    pl.done = pl.frame;  // (only now: planar_light lights what this drew, not a dropped frame's old images)
    static int nlog;
    if (nlog++ < 3) LOG("[planar] frame %llu: %d draws mirrored in y = %.3f into %ux%u (%u targets, %llu submissions since the first)\n",
                        (unsigned long long)pl.frame, n, pl.y, w, h, pl.ncolor, (unsigned long long)(submit_seq - pl.seq));
    static long dump = -2;
    if (dump == -2) dump = getenv("LARECOMP_PLANAR_DUMP") ? atol(getenv("LARECOMP_PLANAR_DUMP")) : -1;
    if (dump >= 0 && pl.frame == (uint64_t)dump)
        for (uint32_t i = 0; i < pl.ncolor; i++) {
            if (!pl.img[i].image) continue;
            char path[64];
            snprintf(path, sizeof path, "build/planar_%u.png", i);
            renderer_save_texture_tex(&pl.img[i], path);
        }
}

// The deferred lighting draw again (draw_, right after the game's, with what it bound), onto the mirrored G-buffer:
// its textures that are the G-buffer's targets swapped for planar_replay_pending's images, its camera (fragment
// constant buffer 2: rows 12-15 the depth -> world matrix, row 19 the eye - gtao_before_lighting) for the mirrored
// one's - the depth was drawn with pl.vpm, so its inverse gives back the real point; the eye is under the water, where
// the light the mirror shows leaves from. Into images like the lighting pass's targets (9-11), the mirrored depth
// attached when the pass has the scene's. Its light volumes and later passes aren't drawn again.
static void mul4(const float *a, const float *b, float *out) {  // (by columns: out = a x b)
    for (int c = 0; c < 4; c++)
        for (int r = 0; r < 4; r++)
            out[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
}

// (kept by planar_light until its pass ends: ended right after the draw, the next draw's begin_pass would apply the
// AO and SSGI in the middle of the lighting pass)
static struct { bool pending; PlDraw d; uint32_t ncolor; VkFormat color[3], depth, stencil; uint64_t seq; } pl_l;

static void planar_light(Prog *pr, Pipe *pp, const VkDescriptorImageInfo *fii, VkDescriptorSet vset, const uint32_t *dyn,
                         uint32_t nv_dyn, const float rs0[4], const float *tint, uint32_t streams, const PlDrawCall *dc) {
    extern uint64_t renderer_exec_frame(void);
    if (pl.done != pl.frame || pl.frame != renderer_exec_frame() || pl.lit_done == pl.frame || pl_l.pending || !pl.img[8].image ||
        !pl.img[12].image) return;
    Range c2 = st.ubuf[1][2];
    if (!c2.addr || c2.size < 22 * 16 || pass_ncolor > 3) return;
    int nt = pr->fi->ntex < 64 ? pr->fi->ntex : 64, nrep = 0;
    VkDescriptorImageInfo alt[64];
    memcpy(alt, fii, (size_t)nt * sizeof alt[0]);
    static const VkComponentMapping id = {0};
    // the lighting turns its depth into distances with the game's projection (fog, the shadows' cascades): not the
    // clipping depth the mirror was drawn with: image 12 instead (planar_light_pending)
    for (int i = 0; i < nt; i++) {
        NvnTexture *t = shader_texture_object(1, &pr->fi->tex[i]);
        if (!t) continue;
        if (t == pl.gdepth) {
            // (the lighting reads this one twice, as depth and as stencil - its material bits: the stencil read through
            // the depth image gave a dotted picture)
            NvnTexEntry *e = gtao_tex_entry(pr->fi, i);
            bool sten = e && (e->has_view && (e->view.setmask & 16) ? e->view.ds_mode : e->tex->ds_mode) == 1;
            alt[i].imageView = sten ? vk_image_view(&pl.img[8], VK_IMAGE_VIEW_TYPE_2D, pl.img[8].format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_STENCIL_BIT)
                                    : post_view(&pl.img[12]);
            nrep++;
            continue;
        }
        for (int k = 0; k < 8; k++)
            if (pl.gtex[k] && t == pl.gtex[k] && pl.img[k].image) { alt[i].imageView = post_view(&pl.img[k]); nrep++; break; }
    }
    if (nrep < 3) {
        static int nlog;
        if (nlog++ < 3) LOG("[planar] a lighting draw reading %d of the G-buffer's images: not drawn again\n", nrep);
        return;
    }
    uint64_t size = c2.size > sizeof pl.lcb ? sizeof pl.lcb : c2.size;
    memcpy(pl.lcb, (const void *)(uintptr_t)c2.addr, size);
    // (rows 12-15, G: the game's own (ndc, depth) -> world, in its conventions - its y sign... VP x G is that convention
    // alone, so inverse(mirrored VP) x VP x G is the same reconstruction for the mirrored depth)
    if (mv_frame[0] == pl.frame) {
        float inv[16], t[16], g[16];
        memcpy(g, pl.lcb + 12 * 4, sizeof g);
        inv4(pl.vpm_std, inv);
        mul4(mv_vp[0], g, t);
        mul4(inv, t, pl.lcb + 12 * 4);
    } else inv4(pl.vpm_std, pl.lcb + 12 * 4);
    pl.lcb[19 * 4 + 1] = 2 * pl.y - pl.lcb[19 * 4 + 1];
    st.ubuf[1][2].addr = (uint64_t)(uintptr_t)pl.lcb, st.ubuf[1][2].size = size;
    uint32_t fdyn[2 * MAX_BIND], nf = 0;
    VkDescriptorSet fset = stage_set(pr, pr->set_layout[1], 1, 1, pr->fi, rs0, tint, 0, 0, 0, alt, fdyn, &nf);
    st.ubuf[1][2] = c2;
    if (nv_dyn + nf > 2 * MAX_BIND) return;
    PlDraw *d = &pl_l.d;
    memset(d, 0, sizeof *d);
    d->pp = pp, d->layout = pr->layout, d->streams = streams, d->s = ss, d->dc = *dc;
    d->sets[0] = vset, d->sets[1] = fset;
    d->ndyn = nv_dyn + nf;
    memcpy(d->dyn, dyn, nv_dyn * sizeof dyn[0]);
    memcpy(d->dyn + nv_dyn, fdyn, nf * sizeof fdyn[0]);
    if (gc_shift && (pr->vi->uses_gmem || pr->fi->uses_gmem))
        d->gmem = true, d->gset = gm_last.set, d->gindex = gm_last.index, d->gdirect = gm_last.direct, memcpy(d->gd, gm_last.gd, sizeof d->gd);
    pl_l.ncolor = pass_ncolor, pl_l.depth = pass_depth, pl_l.stencil = pass_stencil, pl_l.seq = submit_seq;
    memcpy(pl_l.color, pass_color, pass_ncolor * sizeof pass_color[0]);
    pl_l.pending = true;
}

static void planar_light_pending(void) {
    extern uint64_t renderer_exec_frame(void);
    if (!pl_l.pending) return;
    pl_l.pending = false;
    // (a submission since its draw: the ring may have taken back what its sets point at. That frame has no reflection)
    if (pl.frame != renderer_exec_frame() || pl_l.seq != submit_seq) return;
    PlDraw *d = &pl_l.d;
    TexBackend *zd = &pl.img[12];
    uint32_t w = pl.img[8].width, h = pl.img[8].height;
    float s = (float)w / (float)pl.w;
    static const VkComponentMapping id = {0};
    VkRenderingAttachmentInfo ca[3] = {0}, da = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    for (uint32_t i = 0; i < pl_l.ncolor; i++) {
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        if (pl_l.color[i] == VK_FORMAT_UNDEFINED) continue;
        TexBackend *b = planar_image(9 + (int)i, pl_l.color[i], w, h);
        if (!b) return;
        ca[i].imageView = post_view(b), ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    bool with_depth = pl_l.depth != VK_FORMAT_UNDEFINED && pl_l.depth == pl.img[8].format;
    if (!with_depth && (pl_l.depth != VK_FORMAT_UNDEFINED || pl_l.stencil != VK_FORMAT_UNDEFINED)) return;  // (another depth: no image like it)
    if (with_depth) {
        da.imageView = vk_image_view(&pl.img[8], VK_IMAGE_VIEW_TYPE_2D, pl.img[8].format, 0, 1, 0, 1, id, pl.img[8].aspect);
        da.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    {   // (planar_depth.frag: back to the world with the clipping camera's inverse, forward with the unclipped one)
        float m[32];
        inv4(pl.vpm, m);
        memcpy(m + 16, pl.vpm_std, sizeof pl.vpm_std);
        post_pc_big = m;
        VkImageView in[1] = {vk_image_view(&pl.img[8], VK_IMAGE_VIEW_TYPE_2D, pl.img[8].format, 0, 1, 0, 1, id, VK_IMAGE_ASPECT_DEPTH_BIT)};
        post_pass_ex(vk_cmd(), "planar_depth.frag", in, 1, zd, NULL, false, NULL, 0);
        post_pc_big = NULL;
    }
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = pl_l.ncolor, .pColorAttachments = ca, .pDepthAttachment = with_depth ? &da : NULL,
                          .pStencilAttachment = with_depth && pl_l.stencil != VK_FORMAT_UNDEFINED ? &da : NULL};
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    rec_CmdBeginRendering(cb, &ri);
    PlEmit em = {0};
    planar_emit(cb, d, s, w, h, false, &em);
    rec_CmdEndRendering(cb);
    ss_forget();
    vk_full_barrier(cb);
    pl.lit_done = pl.frame;
    static long dump = -2;
    if (dump == -2) dump = getenv("LARECOMP_PLANAR_DUMP") ? atol(getenv("LARECOMP_PLANAR_DUMP")) : -1;
    if (dump >= 0 && pl.frame == (uint64_t)dump && pl.img[9].image) renderer_save_texture_tex(&pl.img[9], "build/planar_lit.png");
}

// ================================================================== screen-space planar reflections
// The planar reflection's picture without drawing anything again (Remi Genin, "Screen Space Planar Reflections in Ghost
// Recon Wildlands", 2017): every pixel of the scene above the water's plane, its depth and lit colour from before the
// water, is mirrored in the plane and projected; sspr_project.comp writes where it came from at the half-size pixel it
// lands on (imageAtomicMax on row << 16 | column, so of the pixels that land together the one lowest on the screen wins:
// nearest the water, and the same one each frame). sspr_resolve.frag then fetches those colours, single-pixel cracks
// filled from a neighbour; its alpha is the distance from the camera for the ripples. No temporal fill or stretch at the
// picture's edges as in the article: what's left empty goes to the water shader's screen search, then the cube.
// Sees only what the picture shows, like that search, but exactly and at one cost. One plane: the commonest level
// (planar_level_probe); other water keeps the search. F1 > Reflection method: Mirrored picture (LARECOMP_WATER_MODE=1).
static bool sspr_run(VkImageView dpre, VkImageView cpre, const float *wpc, VkImageView *out, float *y) {
    if (water_mode() != 1 || !pl_lv.have) return false;
    static VkPipeline pipe;
    static VkPipelineLayout layout;
    static VkDescriptorSetLayout dsl;
    if (!pipe) {
        VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
                                             {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &dsl));
        VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 112};
        VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
                                         .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &pi, NULL, &layout));
        VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
                                          .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                    .module = vk_builtin_shader("sspr_project.comp"), .pName = "main"}};
        VK_CHECK(vkCreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipe));
    }
    uint32_t sw = (uint32_t)wpc[26], sh = (uint32_t)wpc[27], w = sw / 2, h = sh / 2;
    TexBackend *hb = planar_image(13, VK_FORMAT_R32_UINT, w, h), *cb2 = planar_image(14, VK_FORMAT_R16G16B16A16_SFLOAT, w, h);
    if (!hb || !cb2) return false;
    // the projection's constants: the mirror as one matrix from the picture's (ndc, depth) to its own clip space
    // (VP x mirror x VP^-1), and the two rows of VP^-1 that give the world height
    float vp[16], t[16], u[28];
    const float mirror[16] = {1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 2 * pl.y, 0, 1};
    inv4(wpc, vp);
    mul4(mirror, wpc, t);
    mul4(vp, t, u);
    for (int c = 0; c < 4; c++) u[16 + c] = wpc[c * 4 + 1], u[20 + c] = wpc[c * 4 + 3];
    // (a point a metre higher: further down the image's rows or up? - the image can be upside down: the hash's order
    // follows)
    float cen[4], up[4];
    for (int r = 0; r < 4; r++) cen[r] = wpc[8 + r] * 0.5f + wpc[12 + r];
    for (int r = 0; r < 4; r++) up[r] = vp[r] * cen[0] / cen[3] + vp[4 + r] * (cen[1] / cen[3] + 1) + vp[8 + r] * cen[2] / cen[3] + vp[12 + r];
    bool flip = up[1] / up[3] * wpc[22] > 0;
    u[24] = wpc[22], u[25] = pl.y, u[26] = flip ? (float)(sh - 1) : -1, u[27] = 0;
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    static const VkClearColorValue zero = {0};
    VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    rec_CmdClearColorImage(cb, hb->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
    vk_full_barrier(cb);
    VkDescriptorSet set = vk_alloc_set(dsl);
    VkDescriptorImageInfo ii[2] = {{vk_default_sampler(false), dpre, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, post_view(hb), VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[2];
    for (uint32_t k = 0; k < 2; k++)
        wr[k] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = k, .descriptorCount = 1,
                                       .descriptorType = k ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .pImageInfo = &ii[k]};
    rec_UpdateDescriptorSets(2, wr);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
    rec_CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof u, u);
    rec_CmdDispatch(cb, (sw + 7) / 8, (sh + 7) / 8, 1);
    float pc[32];
    memcpy(pc, wpc, sizeof pc);
    pc[28] = u[26], pc[31] = pl.y;
    const float *water_pc = post_pc_big;  // (the water pass's, in the middle of setting up)
    post_pc_big = pc, post_nearest_first = true;
    VkImageView in[3] = {post_view(hb), cpre, dpre};
    post_pass_ex(cb, "sspr_resolve.frag", in, 3, cb2, NULL, true, NULL, 0);
    post_pc_big = water_pc, post_nearest_first = false;
    *out = post_view(cb2), *y = pl.y;
    return true;
}

static void gtao_forget(NvnTexture *t) {
    if (gtao.target == t) gtao.target = NULL;
    if (water.target == t || water.depth == t) water.pending = false, water.target = water.depth = NULL;
}

// The view the present samples: the game image, or its anti-aliased / upscaled version. ow x oh: the whole image's
// size in the window (its crop is drawn into the viewport).
static VkImageView post_process(VkCommandBuffer cb, TexBackend *b, float fx, float fy, double ow_d, double oh_d, bool *upscaled) {
    VkImageView v = raw_view(b);
    bool smaa = g_settings.antialias == AA_SMAA;
    uint32_t ow = (uint32_t)lround(ow_d), oh = (uint32_t)lround(oh_d);
    if (ow > 16384) ow = 16384;
    if (oh > 16384) oh = 16384;
    uint32_t cw = (uint32_t)lroundf(b->width * fx), ch = (uint32_t)lroundf(b->height * fy);  // (the contents' size)
    bool xess = g_settings.upscaler == UPSCALE_XESS && (ow > cw + 1 || oh > ch + 1);
    bool fsr = (g_settings.upscaler == UPSCALE_FSR1 || xess) && (ow > cw + 1 || oh > ch + 1);  // (an upscale only)
    if (xess) {
        VkXeSSFrame xf = {.color = b->image, .color_view = v, .color_format = b->format, .w = cw, .h = ch, .ow = ow, .oh = oh};
        vk_xess_record(g_vk.phys, g_vk.dev, cb, &xf, (VkXeSSQuality)(g_settings.xess_quality >= 1 && g_settings.xess_quality <= 5 ? g_settings.xess_quality : XESS_QUALITY_QUALITY));
    }
    *upscaled = fsr;
    // Internal resolution above the window's: the present's one bilinear tap reads at most 2 x 2 source texels per
    // window pixel - all of them at 2x, 1 in 9 at 3x (its centre: no anti-aliasing at all), 4 in 16 at 4x - so the
    // higher resolutions looked worse than 2x. A box filter over each window pixel's whole footprint instead.
    bool down = !fsr && ow && oh && (ow + 1 < cw || oh + 1 < ch);
    *upscaled = fsr || down;  // (the result fills the window: no contents' scale left for the present)
    // F1 Sharpening (FSR 1's RCAS, settings.h SHARPEN_*): RCAS's strength in stops (0: strongest; < 0: none). Auto:
    // AMD's default 0.2 after an FSR upscale only, as before; Low / Medium / High also after the box downscale or on
    // the image at its own size.
    static const float stops[SHARPEN_N] = {0.2f, -1, 1.0f, 0.5f, 0.1f};
    int sm = g_settings.sharpen >= 0 && g_settings.sharpen < SHARPEN_N ? g_settings.sharpen : SHARPEN_AUTO;
    float sharp = sm == SHARPEN_AUTO && !fsr ? -1 : stops[sm];
    if (!smaa && !fsr && !down && sharp < 0) return v;
    extern void vk_frame_suspend(void);
    extern void vk_frame_resume(void);
    vk_frame_suspend();
    if (smaa) {  // SMAA 1x: luma edges -> blending weights -> neighbourhood blending
        float w = (float)b->width, h = (float)b->height;
        float m[8] = {1.0f / w, 1.0f / h, w, h};
        TexBackend *edges = post_image(0, b->width, b->height), *weights = post_image(1, b->width, b->height);
        TexBackend *out = post_image(2, b->width, b->height);
        // Dynamic resolution: only the contents' part, and 64 pixels past it (the edge searches reach 2 x 16 pixels, so
        // the contents' pixels never read past it) - the window step's cost then scales with the factor like the game's
        post_area_w = cw + 64, post_area_h = ch + 64;
        // (g_vk.smaa_stencil, integrated GPUs) the weights only where the edge pass found edges (its shader discards
        // the others), as SMAA's reference integration does with a stencil
        TexBackend *sten = g_vk.smaa_stencil ? smaa_stencil_image(cb, b->width, b->height) : NULL;
        VkImageView in[3] = {v};
        post_pass_ex(cb, "smaa_edge.frag", in, 1, edges, m, true, sten, 1);
        VkImageView wi[3] = {post_view(edges), smaa_lut(0), smaa_lut(1)};
        post_pass_ex(cb, "smaa_weight.frag", wi, 3, weights, m, true, sten, 2);
        VkImageView bi[3] = {v, post_view(weights)};
        post_pass(cb, "smaa_blend.frag", bi, 2, out, m, false);
        post_area_w = post_area_h = 0;
        v = post_view(out);
    }
    if (fsr) {
        float iw = (float)cw, ih = (float)ch;  // (EASU fetches texels of the contents: a larger texture needs no uv scale)
        TexBackend *e = post_image(3, ow, oh), *r = post_image(4, ow, oh);
        float pe[8] = {iw / ow, ih / oh, 0.5f * iw / ow - 0.5f, 0.5f * ih / oh - 0.5f, iw, ih};
        VkImageView ei[3] = {v};
        post_pass(cb, "easu.frag", ei, 1, e, pe, false);
        v = post_view(e);
        if (sharp >= 0) {
            float pr[8] = {exp2f(-sharp), 0, 0, 0, (float)ow, (float)oh};
            VkImageView ri_[3] = {v};
            post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
            v = post_view(r);
        }
    }
    if (down) {
        float pd[8] = {(float)cw / ow, (float)ch / oh, (float)cw, (float)ch};  // (the contents: Dynamic resolution)
        TexBackend *d = post_image(5, ow, oh);
        VkImageView di[3] = {v};
        post_pass(cb, "boxdown.frag", di, 1, d, pd, false);
        v = post_view(d);
    }
    if (sharp >= 0 && !fsr) {  // (after the box downscale: at the window's size; else the contents' part of the image)
        uint32_t sw = down ? ow : cw, sh = down ? oh : ch;
        TexBackend *r = down ? post_image(4, ow, oh) : post_image(6, b->width, b->height);
        float pr[8] = {exp2f(-sharp), 0, 0, 0, (float)sw, (float)sh};  // (RCAS reads clamped to the contents)
        VkImageView ri_[3] = {v};
        post_area_w = sw, post_area_h = sh;
        post_pass(cb, "rcas.frag", ri_, 1, r, pr, false);
        post_area_w = post_area_h = 0;
        v = post_view(r);
    }
    vk_full_barrier(cb);
    vk_frame_resume();
    {
        static uint32_t piw, pih, pow_, poh;
        static int psm = -1, pfs = -1, pdn = -1;
        static float psh = -2;
        if (b->width != piw || b->height != pih || ow != pow_ || oh != poh || smaa != psm || fsr != pfs || down != pdn || sharp != psh)
            LOG("[post] %ux%u%s%s%s -> %ux%u, sharpening %.1f\n", b->width, b->height, smaa ? " smaa" : "", fsr ? " fsr1" : "",
                down ? " box" : "", ow, oh, sharp);
        piw = b->width; pih = b->height; pow_ = ow; poh = oh; psm = smaa; pfs = fsr; pdn = down; psh = sharp;
    }
    return v;
}

// Inside the window image's rendering (device.c vk_frame_begin): the game image into (vx, vy, vw, vh).
void vk_present_draw(VkCommandBuffer cb, TexBackend *b, float vx, float vy, float vw, float vh, const float pv[8]) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    // Direct presentation of an unboxed movie texture: centre to native aspect on the swapchain (already cleared black
    // by vk_frame_begin in device.c).
    if (movie_box_enabled() && b->is_movie && !b->movie_boxed && b->movie_w > 0 && b->movie_h > 0)
        movie_box_rect(b->movie_w, b->movie_h, &vx, &vy, &vw, &vh);
    // Dynamic: the image occupies the top-left part of the texture; FSR maps that part to its whole output, otherwise
    // the present samples the part
    float fx = b->scalable && b->ax > 0 ? b->sx / b->ax : 1, fy = b->scalable && b->ay > 0 ? b->sy / b->ay : 1;
    if (g_vk.hdr) {  // (vk_hdr_encode: where the game image is, to find its HUD)
        hdr_game = b;
        float g[12] = {vx, vy, vw > 0 ? 1 / vw : 0, vh > 0 ? 1 / vh : 0, pv[4] * fx, pv[5] * fy, pv[6] * fx, pv[7] * fy, pv[0]};
        memcpy(hdr_place, g, sizeof hdr_place);
    }
    bool upscaled = false;
    VkImageView src = post_process(cb, b, fx, fy, pv[6] > 0 ? vw / pv[6] : vw, pv[7] > 0 ? vh / pv[7] : vh, &upscaled);
    float pvc[8];
    memcpy(pvc, pv, sizeof pvc);
    if (!upscaled) { pvc[4] *= fx; pvc[5] *= fy; pvc[6] *= fx; pvc[7] *= fy; }
    pv = pvc;
    VkPipeline p = present_pipeline(g_vk.sc_format);
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), src, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    rec_UpdateDescriptorSets(1, &w);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {vx, vy, vw, vh, 0, 1};
    rec_CmdSetViewport(cb, 0, 1, &vp);
    float pc[12];
    memcpy(pc, pv, 32);
    screen_filter_params(pc + 8);
    rec_CmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 48, pc);
    fullscreen_vertex_bind(cb);
    rec_CmdDraw(cb, 3, 1, 0, 0);
    VkViewport full = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    rec_CmdSetViewport(cb, 0, 1, &full);
}

// (hdr_snaps above) t's image now -> its copy: the copy keeps t's format, read raw like t (raw_view)
static void hdr_snapshot(NvnTexture *t) {
    TexBackend *b = t->backend;
    if (!b->image || b->aspect != VK_IMAGE_ASPECT_COLOR_BIT || b->levels != 1 || b->layers != 1 || b->samples > 1) return;
    static int next;
    int k = hdr_snaps[0].shown == t ? 0 : hdr_snaps[1].shown == t ? 1 : !hdr_snaps[0].shown ? 0 : !hdr_snaps[1].shown ? 1 : (next ^= 1);
    hdr_snaps[k].shown = t;
    TexBackend *d = &hdr_snaps[k].img;
    VkFormat f = b->format == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_R8G8B8A8_UNORM : b->format == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_B8G8R8A8_UNORM : b->format;
    if (d->image && (d->width != b->width || d->height != b->height || d->format != f)) {
        vk_views_forget(d);
        vk_defer_image(d->image, d->mem);
        d->image = VK_NULL_HANDLE;
        d->mem = VK_NULL_HANDLE;
    }
    if (!d->image) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                                .extent = {b->width, b->height, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(g_vk.vma, &ci, &ai, &d->image, &d->mem, NULL) != VK_SUCCESS) { d->image = VK_NULL_HANDLE; return; }
        d->format = f; d->width = b->width; d->height = b->height; d->depth = 1; d->layers = 1; d->levels = 1;
        d->samples = 1; d->aspect = VK_IMAGE_ASPECT_COLOR_BIT; d->base_type = VK_IMAGE_VIEW_TYPE_2D;
        vk_image_init_layout(vk_cmd(), d->image, d->aspect);
    }
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageCopy c = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .extent = {b->width, b->height, 1}};
    rec_CmdCopyImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, d->image, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    vk_full_barrier(cb);
}

// HDR output (F1 > Video > HDR; device.c vk_frame_end): the window's SDR image (sdr; base: its copy from before the
// overlay, or none) -> the window's HDR image (dst, g_vk.hdr_format) through hdr.frag. Its push constants: the game
// image's place in the window and its uv crop (vk_present_draw), paper white / 80, peak / paper white, the knee
// (LARECOMP_HDR_KNEE, linear; 0.45), and the flags. Without the game's picture before its HUD (the start screen, no
// snapshot yet) nothing is brightened: all of it at paper white.
void vk_hdr_encode(VkCommandBuffer cb, VkImageView sdr, VkImageView base, VkImageView dst) {
    static VkPipelineLayout layout;
    static VkDescriptorSetLayout dsl;
    static VkPipeline pipe;
    static VkFormat pipe_fmt;
    if (!layout) {
        VkDescriptorSetLayoutBinding bd[4];
        for (uint32_t i = 0; i < 4; i++)
            bd[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = bd};
        VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &dsl));
        VkPushConstantRange r = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64};
        VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                         .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &r};
        VK_CHECK(vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &layout));
    }
    if (!pipe || pipe_fmt != g_vk.hdr_format) {  // (an old one may still be in flight: kept, a few per session at most)
        VkPipelineShaderStageCreateInfo stages[2] = {
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
             .module = vk_builtin_shader("post.vert"), .pName = "main"},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
             .module = vk_builtin_shader("hdr.frag"), .pName = "main"}};
        VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                     .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
        VkPipelineViewportStateCreateInfo vs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
        VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                     .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
        VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
        VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
                                                   .pAttachments = &cba};
        VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        uint32_t ndyn = 2;
        fullscreen_vertex_dyn(dyn, &ndyn, 3);
        VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = ndyn, .pDynamicStates = dyn};
        VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                            .pColorAttachmentFormats = &g_vk.hdr_format};
        VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri, .stageCount = 2,
                                           .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vs,
                                           .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
                                           .pColorBlendState = &cbs, .pDynamicState = &dy, .layout = layout};
        VK_CHECK(vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &pi, NULL, &pipe));
        pipe_fmt = g_vk.hdr_format;
    }
    TexBackend *game = hdr_game;
    hdr_game = NULL;  // (this frame's only: the start screen draws no game image)
    TexBackend *sb = game ? hdr_snap_of(game) : NULL;
    VkImageView shown_v = sb ? raw_view(game) : VK_NULL_HANDLE, scene_v = shown_v ? post_view(sb) : VK_NULL_HANDLE;
    static float knee = -1;
    if (knee < 0) knee = ENV("LARECOMP_HDR_KNEE") ? (float)atof(ENV("LARECOMP_HDR_KNEE")) : 0.45f;
    if (knee < 0 || knee > 0.95f) knee = 0.45f;
    float paper = (float)g_settings.hdr_paper, peak = (float)g_settings.hdr_peak;
    float pc[16];
    memcpy(pc, hdr_place, 8 * sizeof(float));
    pc[8] = paper / 80, pc[9] = peak > paper ? peak / paper : 1, pc[10] = knee, pc[11] = 0;
    pc[12] = g_vk.hdr_space == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT, pc[13] = base != VK_NULL_HANDLE, pc[14] = scene_v != VK_NULL_HANDLE;
    pc[15] = hdr_place[8] > 0.5f;
    VkDescriptorSet set = vk_alloc_set(dsl);
    VkSampler lin = vk_default_sampler(true);
    VkDescriptorImageInfo ii[4] = {{lin, sdr, VK_IMAGE_LAYOUT_GENERAL}, {lin, base ? base : sdr, VK_IMAGE_LAYOUT_GENERAL},
                                   {lin, scene_v ? shown_v : sdr, VK_IMAGE_LAYOUT_GENERAL},
                                   {lin, scene_v ? scene_v : sdr, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[4];
    for (uint32_t i = 0; i < 4; i++)
        wr[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = i, .descriptorCount = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[i]};
    rec_UpdateDescriptorSets(4, wr);
    VkExtent2D ext = g_vk.sc_extent;
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = dst,
                                    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, ext}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    rec_CmdBeginRendering(cb, &ri);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
    VkViewport vp = {0, 0, (float)ext.width, (float)ext.height, 0, 1};
    VkRect2D sc = {{0, 0}, ext};
    rec_CmdSetViewport(cb, 0, 1, &vp);
    rec_CmdSetScissor(cb, 0, 1, &sc);
    rec_CmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64, pc);
    fullscreen_vertex_bind(cb);
    rec_CmdDraw(cb, 3, 1, 0, 0);
    rec_CmdEndRendering(cb);
}

// Screenshot of a game texture (presented crop applied to build/frame_* shots), via the present shader into RGBA8.
// Only the part its contents occupy (Dynamic resolution: the top-left part; the rest holds older frames).
void renderer_save_texture_tex(TexBackend *b, const char *path) {
    if (!b || !b->image || (b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) return;
    extern const float *vk_shot_crop(void);
    float crop[4] = {0, 0, 1, 1};
    if (strstr(path, "frame_")) memcpy(crop, vk_shot_crop(), sizeof crop);
    float fx = (float)content_extent((int)b->width, b->scalable, b->sx, b->ax) / (float)b->width;
    float fy = (float)content_extent((int)b->height, b->scalable, b->sy, b->ay) / (float)b->height;
    crop[0] *= fx; crop[1] *= fy; crop[2] *= fx; crop[3] *= fy;
    uint32_t w = (uint32_t)(b->width * crop[2]), h = (uint32_t)(b->height * crop[3]);
    if (!w || !h) return;
    vk_end_pass();
    // (an sRGB texture sampled decodes to linear: written back through an sRGB target, the PNG holds its own bytes -
    // what a texture dump / HD pack wants. RGBA8 sRGB images are read raw already: raw_view.)
    VkFormat of = vk_format_is_srgb(b->format) && b->format != VK_FORMAT_R8G8B8A8_SRGB && b->format != VK_FORMAT_B8G8R8A8_SRGB
                ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    VkImage img;
    VmaAllocation mem;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = of,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &img, &mem, NULL));
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    vk_image_init_layout(cb, img, VK_IMAGE_ASPECT_COLOR_BIT);
    VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = of, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView tv;
    VK_CHECK(vkCreateImageView(g_vk.dev, &vci, NULL, &tv));
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = tv, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    rec_CmdBeginRendering(cb, &ri);
    VkPipeline p = present_pipeline_ex(of, strstr(path, "textures/") != NULL);  // (dumps keep their alpha)
    VkDescriptorSet ds = vk_alloc_set(present_set_layout);
    VkDescriptorImageInfo ii = {vk_default_sampler(true), raw_view(b), VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    rec_UpdateDescriptorSets(1, &wr);
    rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, present_layout, 0, 1, &ds, 0, NULL);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    rec_CmdSetViewport(cb, 0, 1, &vp);
    rec_CmdSetScissor(cb, 0, 1, &sc);
    float pv[12] = {renderer_origin_lower_left() && !strstr(path, "window") ? 1.0f : 0.0f, 0, 0, 0, crop[0], crop[1], crop[2], crop[3]};
    if (strstr(path, "frame_")) screen_filter_params(pv + 8);  // (shots of the game image: as shown)
    rec_CmdPushConstants(cb, present_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 48, pv);
    fullscreen_vertex_bind(cb);
    rec_CmdDraw(cb, 3, 1, 0, 0);
    rec_CmdEndRendering(cb);
    vk_full_barrier(cb);
    VkSpan out = vk_readback((uint64_t)w * h * 4);
    VkBufferImageCopy c = {.bufferOffset = out.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    rec_CmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    renderer_flush();
    vk_save_png(path, out.ptr, w, h);
    vkDestroyImageView(g_vk.dev, tv, NULL);
    vmaDestroyImage(g_vk.vma, img, mem);
}

static void exec_list(uint64_t handle, int depth);
static void rt_from_rec(const CmdRec *r, RtState *o) {  // (CMD_SET_RENDER_TARGETS)
    const struct { uint64_t tex[8]; NvnTextureView views[8]; uint8_t has_view[8]; uint64_t depth;
                   NvnTextureView dview; uint8_t has_dview; } *rt = (const void *)r->extra;
    memset(o, 0, sizeof *o);
    o->ncolor = (int)r->x[0] > 8 ? 8 : (int)r->x[0];
    for (int i = 0; i < o->ncolor; i++) {
        o->color[i] = (NvnTexture *)(uintptr_t)rt->tex[i];
        o->cview[i] = rt->views[i];
        o->has_cview[i] = rt->has_view[i];
    }
    o->depth = (NvnTexture *)(uintptr_t)rt->depth;
    o->dview = rt->dview;
    o->has_dview = rt->has_dview;
}
// ================================================================== shadow workers
// The passes with many draws - shadow maps, the depth pre-pass, the G-buffer: ~4000 of a Scarlet / Violet town's ~5600
// draws a frame at World detail 4x, over half the render thread's time, the one that limits the frame rate there - are
// runs of called lists of state and draws. A run is split between worker threads: the render thread walks it once
// setting state only (sw_walking), up to the first record a run can't hold or a draw that must be its own
// (sw_walk_risky: feedback, copy aliases, the lighting draw, water), with a copy of st every SW_SNAP_EVERY draws; the
// parts begin at the copies nearest an even split, and each one is translated on its copy into a secondary command
// buffer of its own (sw_cb: vk_cmd returns it, the recorder calls the driver directly for it - record.c
// rec_worker_enter - and its sets come from pools of its own), part 0 on the render thread; it then records the
// secondaries into the pass in order. The state the draws share (programs, pipelines, texture uploads and views, the
// vertex cache, snapshots) is changed under SWL; the per-draw caches are per thread (NCTX, sw_ctx).
// Town at World detail 4x (Ryzen 7 5700X, RTX 3080): 41 -> 46 fps with 4 workers; 6 workers draw no faster and take
// cores from the game. Once the game's GPU thread no longer walked the lists (rqueue.c's list table) it was as busy as
// this one, and 3 workers left its job threads more: 55.7 fps against 54.2 with 4 (5 runs each).
// LARECOMP_VK_SHADOW_THREADS=<n> workers (0: off); by default 3 from 12 CPU threads, 2 from 8, else none, with the
// recorder thread on.
#define SW_MAX 8
#define SW_LISTS 8192
#define SW_MIN_DRAWS 48
// A part: from record p (in list li of h, through list ln - 1), budget draws, on a copy of the state at p
typedef struct {
    TState tst;
    SsState tss;
    const uint64_t *h;
    const CmdRec *stop;
    const CmdRec *abort_r;  // (the worker) the draw it gave back, in list abort_li; its state then is tst
    uint64_t abort_li;
    uint64_t li, ln, p;
    int budget, depth;
    VkCommandBuffer cb;
} SwJob;
static SwJob sw_jobs[SW_MAX];
static int sw_n = -1;  // workers (LARECOMP_VK_SHADOW_THREADS)
static int sw_njobs;
static pthread_mutex_t sw_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t sw_go = PTHREAD_COND_INITIALIZER, sw_done = PTHREAD_COND_INITIALIZER;
static uint64_t sw_gen;
static int sw_left;
static struct { VkCommandPool pool; VkCommandBuffer *cbs; int n, cap, next; } sw_pools[SW_MAX][VK_SUBS];
static uint64_t sw_stat_runs, sw_stat_draws, sw_stat_aborts, sw_stat_rewound;  // (logged every 600 frames: sw_run)
static uint64_t sw_stat_walk_ns, sw_stat_wait_ns, sw_stat_run_ns;
static __thread int sw_budget;  // (a worker) draws left in its part
// A draw's place in the game's order, for what draws captured draws again (planar_replay_pending): on the render
// thread a count; in a run's part, the run's place, the part and the draw's index in it (render thread only writes
// sw_ord_main / sw_ord_base; a part reads the base it was started with)
static uint64_t sw_ord_main, sw_ord_base;
static __thread int sw_part_k = -1, sw_part_n0;
static uint64_t draw_ord(void) {
    if (sw_part_k >= 0) return sw_ord_base + ((uint64_t)sw_part_k << 16) + (uint64_t)(sw_part_n0 - sw_budget);
    return ++sw_ord_main << 24;
}
static __thread const CmdRec *sw_abort_r;  // (a worker) the draw it gave back (exec_list)
static __thread const CmdRec *sw_stop;  // (a worker, and the walk) the run ends at this record (NULL: its lists' ends)
static const CmdRec *sw_tail_stop;  // (sw_tail -> exec_list) where the render thread goes on after the run
// the walk (render thread): where the parts begin
static int sw_walk_parts, sw_walk_count, sw_walk_k, sw_walk_bound[SW_MAX + 1];
static uint64_t sw_walk_li;
// the game's timing counters in a run: query slots taken in the walk, in the game's order, written by the worker
#define SW_CQ 256
static struct { const CmdRec *r; uint32_t idx; } sw_cq[SW_CQ];
static int sw_ncq;

// draws in a list made only of state and draws (following its jumps); -1: something else in it. With stop: the draws
// before the first record that isn't state or a draw, *stop = that record (NULL: the list's end)
static int sw_list_draws_at(uint64_t handle, const CmdRec **stop) {
    if (stop) *stop = NULL;
    int draws = 0;
    uint64_t p = handle;
    for (int guard = 0; guard < 100000; guard++) {
        const CmdRec *r = (const CmdRec *)(uintptr_t)p;
        // (as exec_list: the records' first read misses the caches. Only their headers are read here, ~100 bytes
        // apart: further ahead than exec_list, which spends longer on each)
        __builtin_prefetch((const char *)r + 1024);
        __builtin_prefetch((const char *)r + 2048);
        switch (r->op) {
        case CMD_END: return draws;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS: case CMD_DRAW_ARRAYS_INDIRECT: case CMD_DRAW_ELEMENTS_INDIRECT:
            draws++;
            break;
        case CMD_BARRIER:
        case CMD_BIND_PROGRAM: case CMD_BIND_BLEND: case CMD_BIND_CHANNEL_MASK: case CMD_BIND_COLOR: case CMD_BIND_DEPTH_STENCIL:
        case CMD_BIND_MULTISAMPLE: case CMD_BIND_POLYGON: case CMD_BIND_VERTEX_ATTRIB: case CMD_BIND_VERTEX_STREAM:
        case CMD_BIND_VERTEX_BUFFERS: case CMD_BIND_UNIFORM_BUFFERS: case CMD_BIND_STORAGE_BUFFERS: case CMD_BIND_TEXTURES:
        case CMD_BIND_IMAGES: case CMD_SET_TEXTURE_POOL: case CMD_SET_SAMPLER_POOL:
        case CMD_SET_VIEWPORTS: case CMD_SET_SCISSORS: case CMD_SET_DEPTH_RANGES: case CMD_SET_BLEND_COLOR: case CMD_SET_STENCIL:
        case CMD_SET_PRIMITIVE_RESTART: case CMD_SET_ALPHA_REF: case CMD_SET_DEPTH_BOUNDS: case CMD_SET_POLYGON_OFFSET:
        case CMD_SET_LINE_WIDTH: case CMD_SET_POINT_SIZE: case CMD_SET_SAMPLE_MASK: case CMD_SET_DEPTH_CLAMP:
        case CMD_SET_RASTERIZER_DISCARD:
        case CMD_REPORT_COUNTER:  // (its CPU part in the walk, its GPU timestamp in the worker's secondary: exec_list)
            break;
        case CMD_SET_RENDER_TARGETS: {  // (the pass's own targets set again: nothing changes)
            RtState t;
            rt_from_rec(r, &t);
            if (memcmp(&t, &st.rt, sizeof t)) { if (stop) { *stop = r; return draws; } return -1; }
            break;
        }
        default:
            if (stop) { *stop = r; return draws; }
            return -1;
        }
        p += r->size;
    }
    return -1;
}

static VkCommandBuffer sw_secondary_at(int w, int slot) {
    __typeof__(sw_pools[0][0]) *sp = &sw_pools[w][slot];
    if (!sp->pool) {
        VkCommandPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                      .queueFamilyIndex = g_vk.qfam};
        VK_CHECK(vkCreateCommandPool(g_vk.dev, &ci, NULL, &sp->pool));
    }
    if (sp->next == sp->n) {
        if (sp->n == sp->cap) { sp->cap = sp->cap ? sp->cap * 2 : 8; sp->cbs = realloc(sp->cbs, sp->cap * sizeof *sp->cbs); }
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = sp->pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY, .commandBufferCount = 1};
        VK_CHECK(vkAllocateCommandBuffers(g_vk.dev, &ai, &sp->cbs[sp->n++]));
    }
    return sp->cbs[sp->next++];
}
static VkCommandBuffer sw_secondary(int w) { return sw_secondary_at(w, cur_sub); }
// (sub(): the slot's submission is done) its secondaries are free again
static void sw_slot_reset(int slot) {
    for (int w = 0; w < SW_MAX; w++)
        if (sw_pools[w][slot].pool && sw_pools[w][slot].next) {
            vkResetCommandPool(g_vk.dev, sw_pools[w][slot].pool, 0);
            sw_pools[w][slot].next = 0;
        }
}

// a part, on the thread that runs it (a worker, or the render thread for part 0)
static void sw_chunk(int w, SwJob *j) {
    st_tls = &j->tst;
    ss_tls = &j->tss;
    memset(&j->tss, 0, sizeof j->tss);
    sw_cb = j->cb;
    sw_id = w;
    VkCommandBufferInheritanceRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
                                                  .colorAttachmentCount = pass_ncolor, .pColorAttachmentFormats = pass_color,
                                                  .depthAttachmentFormat = pass_depth, .stencilAttachmentFormat = pass_stencil,
                                                  .rasterizationSamples = (VkSampleCountFlagBits)pass_samples};
    VkCommandBufferInheritanceInfo ii = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, .pNext = &ri};
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
                                   .pInheritanceInfo = &ii};
    VK_CHECK(vkBeginCommandBuffer(j->cb, &bi));
    extern void rec_worker_enter(int w, VkCommandBuffer cb), rec_worker_leave(void);
    rec_worker_enter(w, j->cb);
    sw_budget = j->budget;
    sw_part_k = w, sw_part_n0 = j->budget;
    sw_stop = j->stop;
    j->abort_r = NULL;
    for (uint64_t i = j->li; i < j->ln && sw_budget > 0 && !j->abort_r; i++) {
        sw_abort_r = NULL;
        exec_list(i == j->li ? j->p : j->h[i], j->depth);
        if (sw_abort_r) j->abort_r = sw_abort_r, j->abort_li = i;
    }
    sw_part_k = -1;
    sw_stop = NULL;
    rec_worker_leave();
    VK_CHECK(vkEndCommandBuffer(j->cb));
    sw_cb = VK_NULL_HANDLE;
    sw_id = -1;
    st_tls = NULL;
    ss_tls = NULL;
}

// ---- Async runs (LARECOMP_VK_SW_ASYNC=1, experimental: no faster in Scarlet / Violet's towns, where nearly every run
// is followed at once by a copy or a draw that ends the pass). A run can't make a worker give a draw back (the walk cut
// it before those: sw_walk_risky), so the render thread needn't wait for it: after the walk it queues the parts as jobs,
// queues an ExecuteCommands that the recorder thread makes once they are done (rec_CmdExecuteCommandsWait), and goes on
// - the next pass's walk and run, its clears and its own draws - while the workers translate, making the parts still
// queued itself when it has to wait. Meanwhile it takes the workers' lock where they do (sw_rt_lock: SWL), the
// recorder's ring is shared (rec_queue_shared), the workers draw with their pass's copy of the pass state (pass_tls),
// and anything else - the lit image's pass, a copy, a compute dispatch, transform feedback, a submission - waits for
// the jobs first (sw_async_op_ok, sw_async_sync). Counters (REPORT_COUNTER) in a run: the waiting path.
#define SW_AJ 32
#define SW_AR 16
typedef struct { _Atomic int left, consumed; VkCommandBuffer cbs[SW_MAX]; bool used, aborted; int draws; } SwRunRec;
typedef struct { SwJob j; PassG pg; uint64_t hbuf[1]; int k, slot; SwRunRec *run; _Atomic int done; } SwAJob;
static SwAJob sw_aj[SW_AJ];
static SwRunRec sw_ar[SW_AR];
static uint64_t sw_aj_head, sw_aj_tail;  // (sw_mu) jobs queued, taken
static int sw_ar_next;
static _Atomic int sw_aj_out;  // jobs not done yet
static uint64_t sw_stat_async_runs, sw_stat_async_wait_ns, sw_stat_async_syncs, sw_stat_async_lost;
static int sw_async_mode = -1;
static bool sw_async_on(void) {
    if (sw_async_mode < 0) { const char *e = ENV("LARECOMP_VK_SW_ASYNC"); sw_async_mode = e && atoi(e) != 0; }
    extern bool rec_exec_wait_ok(void);
    return sw_async_mode && rec_exec_wait_ok();
}
static void sw_chunk(int w, SwJob *j);
static void sw_async_job(int w, SwAJob *a) {  // (a worker)
    SwJob *j = &a->j;
    SwRunRec *rr = a->run;
    if (j->budget > 0) {
        j->cb = sw_secondary_at(w, a->slot);
        pass_tls = &a->pg;
        sw_chunk(w, j);
        pass_tls = NULL;
        rr->cbs[a->k] = j->cb;
        if (j->abort_r) rr->aborted = true;
    }
    atomic_fetch_sub_explicit(&rr->left, 1, memory_order_release);
    atomic_store_explicit(&a->done, 1, memory_order_release);
    atomic_fetch_sub_explicit(&sw_aj_out, 1, memory_order_release);
}
// (the render thread) the async runs' jobs done: the state they shared is its own again
static void sw_async_sync(void) {
    if (!sw_ainflight || sw_cb) return;
    uint64_t t0 = SDL_GetTicksNS();
    int held = sw_mx_depth;  // (inside a locked part - a submission from it: let the workers finish)
    if (held) { sw_mx_depth = 0; atomic_store_explicit(&sw_mx, 0, memory_order_release); }
    for (uint32_t k = 0; atomic_load_explicit(&sw_aj_out, memory_order_acquire) > 0; k++) {
        SwAJob *a = NULL;  // (a part nobody took yet: made here, as the render thread's part of a waiting run)
        if (sw_aj_tail != sw_aj_head) {
            pthread_mutex_lock(&sw_mu);
            if (sw_aj_tail != sw_aj_head) a = &sw_aj[sw_aj_tail++ % SW_AJ];
            pthread_mutex_unlock(&sw_mu);
        }
        if (a) { sw_async_job(0, a); k = 0; continue; }
        if (k < 20000) __builtin_ia32_pause();
        else SDL_Delay(0);
    }
    if (held) {
        while (atomic_exchange_explicit(&sw_mx, 1, memory_order_acquire)) __builtin_ia32_pause();
        sw_mx_depth = held;
    }
    for (int w = 1; w <= sw_n && w < 8; w++) ctr_fold(w);
    extern void rec_queue_shared(bool on);
    rec_queue_shared(false);
    sw_ainflight = sw_rt_lock = false;
    for (int i = 0; i < SW_AR; i++)
        if (sw_ar[i].used && sw_ar[i].aborted) {  // (never expected: such a pass isn't taken - sw_async_pass_ok)
            sw_ar[i].aborted = false;
            if (sw_stat_async_lost++ < 10) LOG("[sw] an async run's worker gave a draw back: the rest of its part is missing\n");
        }
    sw_stat_async_syncs++;
    sw_stat_async_wait_ns += SDL_GetTicksNS() - t0;
}
// (the render thread) a pass whose runs may go on while it translates what follows: not the lit image's (the water
// and the effects after the lighting draw into it), else as sw_pass_ok
static bool sw_pass_ok(void);
static bool sw_async_ops_ok(void) {  // (also the render thread's own draws and clears in it, while runs are in flight)
    if (xfb_on) return false;
    for (int c = 0; c < st.rt.ncolor; c++) if (st.rt.color[c] && renderer_texture_presented(st.rt.color[c])) return false;
    return !(gtao.target && st.rt.ncolor && st.rt.color[0] == gtao.target);
}
static bool sw_async_pass_ok(void) { return sw_pass_ok() && sw_async_ops_ok(); }
// (the render thread, async runs in flight) this record may be translated now; else the jobs are waited for first
static bool sw_walk_risky(void);
static bool sw_async_op_ok(const CmdRec *r) {
    switch (r->op) {
    case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS: case CMD_DRAW_ARRAYS_INDIRECT: case CMD_DRAW_ELEMENTS_INDIRECT:
        return sw_async_ops_ok() && !sw_walk_risky();
    case CMD_CLEAR_COLOR: case CMD_CLEAR_DEPTH_STENCIL:
        return sw_async_ops_ok();
    case CMD_CLEAR_TEXTURE: case CMD_DISPATCH: case CMD_COPY_BUFFER_TO_BUFFER: case CMD_COPY_BUFFER_TO_TEXTURE:
    case CMD_COPY_TEXTURE_TO_BUFFER: case CMD_COPY_TEXTURE_TO_TEXTURE: case CMD_DOWNSAMPLE: case CMD_UPDATE_UNIFORM_BUFFER:
    case CMD_FENCE_SYNC: case CMD_REPORT_COUNTER: case CMD_MULTI_DRAW_ARRAYS_INDIRECT_COUNT:
    case CMD_MULTI_DRAW_ELEMENTS_INDIRECT_COUNT: case CMD_BEGIN_XFB: case CMD_END_XFB: case CMD_BIND_XFB_BUFFER:
    case CMD_BIND_XFB_BUFFERS: case CMD_DRAW_XFB:
        return false;
    default:
        return true;  // (state, targets, lists, and what this translation ignores)
    }
}

#include "../../platform.h"
static void *sw_thread(void *arg) {
    int w = (int)(intptr_t)arg;
    plat_thread_perf_core(PLAT_PERF_RENDER);
    extern void rq_helper_thread(void);
    rq_helper_thread();  // (vk_frame_counter: the frame being translated, as on the render thread)
    ctr_tls = &ctr_w[w];
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&sw_mu);
        while (sw_gen == seen && sw_aj_tail == sw_aj_head) pthread_cond_wait(&sw_go, &sw_mu);
        if (sw_aj_tail != sw_aj_head) {  // (an async run's part)
            SwAJob *a = &sw_aj[sw_aj_tail++ % SW_AJ];
            pthread_mutex_unlock(&sw_mu);
            sw_async_job(w, a);
            continue;
        }
        seen = sw_gen;
        int njobs = sw_njobs;
        pthread_mutex_unlock(&sw_mu);
        if (w < njobs) sw_chunk(w, &sw_jobs[w]);
        pthread_mutex_lock(&sw_mu);
        if (--sw_left == 0) pthread_cond_signal(&sw_done);
        pthread_mutex_unlock(&sw_mu);
    }
    return NULL;
}

static bool sw_enabled(void) {
    if (sw_n < 0) {
        const char *e = ENV("LARECOMP_VK_SHADOW_THREADS");
        extern bool rec_on(void);
        int cpus = SDL_GetNumLogicalCPUCores();
        sw_n = e ? atoi(e) : !rec_on() ? 0 : cpus >= 12 ? 3 : cpus >= 8 ? 2 : 0;
        if (sw_n > SW_MAX - 1) sw_n = SW_MAX - 1;
        if (sw_n < 0 || g_vkc.render_pass) sw_n = 0;  // (Vulkan 1.2 render passes: no dynamic rendering inheritance here)
        for (int w = 1; w <= sw_n; w++) {
            pthread_t t;
            if (pthread_create(&t, NULL, sw_thread, (void *)(intptr_t)w)) { sw_n = w - 1; break; }
            pthread_detach(t);
        }
        if (sw_n) LOG("[sw] %d shadow worker threads\n", sw_n);
    }
    return sw_n > 0;
}

// (the walk) a draw: the next part begins after it once enough came before
// (the walk) a draw the render thread must make itself: one whose program samples an attachment of the pass (its
// snapshot ends the pass - resolve_feedback), or a copy alias over one of the attachments (alias_draw_check). The run
// ends before it (sw_cut_*); it and what follows go on as usual. (A texture bound to such a draw is only a hint: the
// pass's own shadow map stays bound across its draws - the program's own textures decide, as in resolve_feedback.)
static const CmdRec *sw_cut_r;
static uint64_t sw_cut_li;
static bool sw_cut_risky;  // (sw_cut_r is a draw of the render thread's, not a record a run can't hold)
static bool sw_walk_risky_(void);
static bool sw_walk_risky(void) {
    // (the answer while the texture bindings, the program, the pool and the targets stay, within a frame: it was asked
    // at every draw of the walk and of sw_tail, 7 % of the render thread at World detail 4x - most draws change none
    // of them. Copy aliases: asked each time)
    static uint64_t mtex[2][32], mframe = ~0ull, mcode[2], mub[2];
    static const void *mprog, *mpool, *mdepth, *mc0, *mc1;
    static int mnc;
    static uint32_t mcs, mds;
    static bool mres;
    TState *S = st_cur();
    uint64_t f = vk_frame_counter();
    uint64_t code0 = S->program ? S->program->stages[0].code_addr : 0, code1 = S->program ? S->program->stages[1].code_addr : 0;
    uint32_t ds = (uint32_t)S->ds[0] | (uint32_t)S->ds[1] << 8;  // (water_is_surface: the depth test and write)
    if (!naliases && mframe == f && S->program == mprog && code0 == mcode[0] && code1 == mcode[1] && S->tpool == mpool &&
        S->rt.depth == mdepth && S->rt.ncolor == mnc && (S->rt.ncolor < 1 || S->rt.color[0] == mc0) &&
        (S->rt.ncolor < 2 || S->rt.color[1] == mc1) && S->rt.ncolor <= 2 && S->color_state[0] == mcs && ds == mds &&
        S->ubuf[1][2].addr == mub[0] && S->ubuf[1][2].size == mub[1] && !memcmp(S->tex, mtex, sizeof mtex))
        return mres;
    bool r = sw_walk_risky_();
    mframe = f; mprog = S->program; mpool = S->tpool; mdepth = S->rt.depth; mnc = S->rt.ncolor;
    mc0 = S->rt.ncolor > 0 ? S->rt.color[0] : NULL; mc1 = S->rt.ncolor > 1 ? S->rt.color[1] : NULL;
    mcode[0] = code0; mcode[1] = code1; mcs = S->color_state[0]; mds = ds; mub[0] = S->ubuf[1][2].addr; mub[1] = S->ubuf[1][2].size;
    memcpy(mtex, S->tex, sizeof mtex);
    mres = r;
    return r;
}
static bool sw_walk_risky_(void) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    for (int i = 0; i < naliases; i++) if (is_attachment(aliases[i].src) || is_attachment(aliases[i].dst)) return true;
    // (the temporal anti-aliasing's resolve - fsr2_copy - ends the pass, so the draws that can bring it are the render
    // thread's while this frame isn't resolved yet: the interface's first draw into the picture - DLAA's early resolve,
    // blended, no depth target - and the game's copy into the shown image. On a worker the resolve never came: after a
    // fast travel the map's first ~5 interface draws went into a worker's part and through DLSS with the scene (the
    // interface smeared with the camera's motion, as before the early resolve); the resolve waited for the first one
    // back on the render thread)
    if (fsr2_wanted() && fsr2_ok_frame != vk_frame_counter() && st.rt.ncolor && st.rt.color[0] && !st.rt.depth) {
        NvnTexture *r0 = st.rt.color[0];
        if (r0 == fsr2_pic && (st.color_state[0] & 1) && fsr2_early_on()) return true;
        if (!(st.color_state[0] & 1) && r0->backend && ((TexBackend *)r0->backend)->presented) return true;
    }
    if (!st.tpool || !st.program) return false;
    NvnTexture *att[9];  // (the pass's attachments)
    int na = 0;
    if (st.rt.depth) att[na++] = st.rt.depth;
    for (int c = 0; c < st.rt.ncolor && c < 8; c++) if (st.rt.color[c]) att[na++] = st.rt.color[c];
    bool any = false;
    const NvnTexEntry *ents = (const NvnTexEntry *)st.tpool->entries;
    uint32_t cnt = (uint32_t)st.tpool->count;
    // (whether a pool slot holds one of the pass's attachments, remembered while the pool, the attachments and the
    // frame stay: reading the slots' entries missed the caches on most draws. A slot the game fills again with an
    // attachment within the frame is missed here - its draw then goes back from the worker: sw_run's take-over)
    static struct { uint32_t tic; uint8_t att; } tcache[1024];
    static const void *tc_pool;
    static NvnTexture *tc_att[9];
    static int tc_na = -1;
    static uint64_t tc_frame = ~0ull;
    uint64_t frm = vk_frame_counter();
    if (tc_pool != st.tpool || tc_frame != frm || tc_na != na || memcmp(tc_att, att, (size_t)na * sizeof att[0])) {
        memset(tcache, 0xFF, sizeof tcache);
        tc_pool = st.tpool, tc_frame = frm, tc_na = na;
        memcpy(tc_att, att, (size_t)na * sizeof att[0]);
    }
    for (int sg = 0; sg < 2 && !any; sg++)
        for (int k = 0; k < 32 && !any; k++) {
            uint64_t h = st.tex[sg][k];
            uint32_t tic = h & 0xFFFFF;
            if (!h || tic >= cnt) continue;
            __typeof__(tcache[0]) *e = &tcache[(tic * 2654435761u) >> 22];
            if (e->tic != tic) {
                NvnTexture *t = ents[tic].tex;
                bool a_ = false;
                for (int a = 0; a < na; a++) a_ |= t == att[a];
                e->tic = tic, e->att = a_;
            }
            any = e->att;
        }
    // (the lighting draw - gtao_before_lighting - and water surfaces - water_before - end the pass too. Their cheap
    // conditions first - water: into the lit image, blended, depth tested and written; lighting: a fragment shader
    // of ten textures or more, as last seen this frame - then the program)
    bool water_pre = water_on() && gtao.target && st.rt.ncolor && st.rt.color[0] == gtao.target && st.color_state[0] && st.ds[0] &&
                     st.ds[1] && st.rt.depth;
    uint64_t c1 = st.program->stages[1].code_addr, fr = vk_frame_counter();
    static struct { uint64_t code, frame; int ntex; int8_t light; } fc[512];  // (light: -1 not asked yet)
    __typeof__(fc[0]) *fe = &fc[(c1 >> 4 ^ c1 >> 13) & 511];
    bool known = fe->code == c1 && fe->frame == fr;
    if (known && fe->light > 0) return true;  // (the lighting draw, as this frame's walk found)
    bool light_pre = (gtao_on() || ssgi_on() || water_on()) && (!known || fe->light < 0);
    bool special = water_pre || light_pre;
    if (!any && !special) return false;
    ShaderKey vkey = {0}, fkey = {0};  // (the program, as draw_ finds it)
    vkey.glsl = fkey.glsl = 1;
    vkey.flip_y = fkey.flip_y = nvn_window_origin() == 0;
    vkey.depth_minus_one = nvn_depth_mode() == 0;
    for (int i = 0; i < st.nattrib; i++) vkey.attr_kind[i] = (uint8_t)attr_kind(st.attrib[i] & 0xFF);
    for (int i = 0; i < st.rt.ncolor; i++) if (st.rt.color[i]) fkey.rt_kind[i] = (uint8_t)rt_kind(st.rt.color[i]->format);
    vkey.stencil_slots = stencil_slots(0, &vkey.sint_slots);
    fkey.stencil_slots = stencil_slots(1, &fkey.sint_slots);
    Prog *pr = get_program(st.program, &vkey, &fkey);
    if (!pr || atomic_load(&pr->state) != 1 || !pr->vi || !pr->fi) return true;  // (not known yet: the render thread's)
    if (light_pre) {  // (the lighting's: ten textures or more - only then its hash, 512 bytes - once a frame per shader)
        bool l = false;
        if (pr->fi->ntex >= 10) {
            g_cur_fs_hash = c1 ? hmix(1469598103934665603ull, (const void *)(uintptr_t)(c1 + 0x30), 0x200) : 0;  // (draw_'s, asked first)
            l = gtao_is_lighting(pr);
        }
        *fe = (__typeof__(*fe)){c1, fr, pr->fi->ntex, (int8_t)l};
        if (l) return true;
    }
    if (water_pre && water_is_surface(pr)) return true;
    if (!any) return false;
    ShaderInfo *infos[2] = {pr->vi, pr->fi};
    for (int sg = 0; sg < 2; sg++)
        for (int i = 0; i < infos[sg]->ntex; i++)
            if (is_attachment(shader_texture_object(sg, &infos[sg]->tex[i]))) return true;
    return false;
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}
// (the walk) the state every SW_SNAP_EVERY draws: the parts begin at some of them (sw_run, once the draws are counted)
#define SW_SNAP_EVERY 32
#define SW_SNAPS 512
static struct { TState tst; uint64_t li, p; } sw_snap[SW_SNAPS];
static int sw_nsnap;
static void tstate_copy(TState *d, const TState *s_) {  // (the bound stages may be the state's own: &comp)
    *d = *s_;
    if (s_->program == &s_->comp) d->program = &d->comp;
}
static void sw_walk_draw(const CmdRec *r) {
    sw_walk_count++;
    if (sw_walk_count % SW_SNAP_EVERY == 0 && sw_nsnap < SW_SNAPS) {
        tstate_copy(&sw_snap[sw_nsnap].tst, &st);
        sw_snap[sw_nsnap].li = sw_walk_li;
        sw_snap[sw_nsnap++].p = (uint64_t)(uintptr_t)r + r->size;
    }
}
// (the walk) a record a run can hold (as sw_list_draws_at): state, draws, the pass's own targets again
static bool sw_walk_op_ok(const CmdRec *r) {
    switch (r->op) {
    case CMD_END: case CMD_JUMP:
    case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS: case CMD_DRAW_ARRAYS_INDIRECT: case CMD_DRAW_ELEMENTS_INDIRECT:
    case CMD_BARRIER:
    case CMD_BIND_PROGRAM: case CMD_BIND_BLEND: case CMD_BIND_CHANNEL_MASK: case CMD_BIND_COLOR: case CMD_BIND_DEPTH_STENCIL:
    case CMD_BIND_MULTISAMPLE: case CMD_BIND_POLYGON: case CMD_BIND_VERTEX_ATTRIB: case CMD_BIND_VERTEX_STREAM:
    case CMD_BIND_VERTEX_BUFFERS: case CMD_BIND_UNIFORM_BUFFERS: case CMD_BIND_STORAGE_BUFFERS: case CMD_BIND_TEXTURES:
    case CMD_BIND_IMAGES: case CMD_SET_TEXTURE_POOL: case CMD_SET_SAMPLER_POOL:
    case CMD_SET_VIEWPORTS: case CMD_SET_SCISSORS: case CMD_SET_DEPTH_RANGES: case CMD_SET_BLEND_COLOR: case CMD_SET_STENCIL:
    case CMD_SET_PRIMITIVE_RESTART: case CMD_SET_ALPHA_REF: case CMD_SET_DEPTH_BOUNDS: case CMD_SET_POLYGON_OFFSET:
    case CMD_SET_LINE_WIDTH: case CMD_SET_POINT_SIZE: case CMD_SET_SAMPLE_MASK: case CMD_SET_DEPTH_CLAMP:
    case CMD_SET_RASTERIZER_DISCARD: case CMD_REPORT_COUNTER:
        return true;
    case CMD_SET_RENDER_TARGETS: {
        RtState t;
        rt_from_rec(r, &t);
        return !memcmp(&t, &st.rt, sizeof t);
    }
    default:
        return false;
    }
}

// Passes cut this frame (a draw of the render thread's in them - feedback): no more runs in them this frame (their
// draws one by one on the render thread: a run around each of many such draws cost more than it gave)
#define SW_CUT_PASSES 16
static struct { NvnTexture *d, *c0, *c1; } sw_cut_pass[SW_CUT_PASSES];
static int sw_ncut_pass;
static uint64_t sw_cut_frame = ~0ull;
static bool sw_pass_cut(void) {
    // (maintenance7: the pass takes the runs' secondaries and the render thread's draws alike - a run cut around such a
    // draw costs no pass restart, and one too short is translated here anyway: sw_run's rewind. Not barred then)
    if (g_vk.maint7) return false;
    uint64_t f = vk_frame_counter();
    if (f != sw_cut_frame) { sw_cut_frame = f; sw_ncut_pass = 0; return false; }
    for (int i = 0; i < sw_ncut_pass; i++)
        if (sw_cut_pass[i].d == st.rt.depth && sw_cut_pass[i].c0 == (st.rt.ncolor > 0 ? st.rt.color[0] : NULL) &&
            sw_cut_pass[i].c1 == (st.rt.ncolor > 1 ? st.rt.color[1] : NULL))
            return true;
    return false;
}
static void sw_pass_cut_note(void) {
    if (sw_ncut_pass < SW_CUT_PASSES)
        sw_cut_pass[sw_ncut_pass++] = (__typeof__(sw_cut_pass[0])){st.rt.depth, st.rt.ncolor > 0 ? st.rt.color[0] : NULL,
                                                                     st.rt.ncolor > 1 ? st.rt.color[1] : NULL};
}
static bool sw_pass_ok(void) {  // (the render thread) runs in this pass: no transform feedback session open
    if (sw_cb || sw_walking || sw_reexec || !sw_enabled()) return false;
    if (xfb_on) return false;
    // (not into the window's image: the composites, widescreen boxes, mirror mode and HDR snapshots draw_ makes there
    // are the render thread's)
    for (int c = 0; c < st.rt.ncolor; c++) if (st.rt.color[c] && renderer_texture_presented(st.rt.color[c])) return false;
    if (sw_pass_cut()) return false;
    return st.rt.depth != NULL || st.rt.ncolor;
}

static void exec_list(uint64_t handle, int depth);
// (sw_run) the walk over a run's records: their state, as exec_list sets it, and its draws counted (sw_walk_draw) - a
// loop of its own: exec_list's other work on each record (the shadow workers' and async runs' checks, the transform
// feedback flush, counts) was most of the walk's ~0.7 us a draw. Ends at the run's stop, a record a run can't hold or a
// draw that must be the render thread's (sw_cut_r).
static void walk_list(uint64_t handle, int depth) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    uint64_t p = handle;
    for (int guard = 0; guard < 10000000; guard++) {
        CmdRec *r = (CmdRec *)(uintptr_t)p;
        if (r == sw_stop) return;
        __builtin_prefetch((const char *)r + 1024);
        __builtin_prefetch((const char *)r + 2048);
        void *x = r->extra;
        (void)x;
        switch (r->op) {
        case CMD_END: return;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_BIND_PROGRAM: {  // (program, NVNshaderStageBits): a game may bind its stages from several programs
            NvnProgram *p = (NvnProgram *)(uintptr_t)r->x[0];
            uint32_t m = (uint32_t)r->x[1] & 0x3F;
            if (!m) m = 0x3F;
            if (ENV("LARECOMP_LOG_BINDPROG")) {
                static int n;
                if (n++ < 40) LOG("[bindprog] %p stages %#x (program has %#x)\n", (void *)p, m, p ? p->stage_mask : 0);
            }
            for (int i = 0; i < 6; i++) {
                if (!(m >> i & 1)) continue;
                if (p && (p->stage_mask >> i & 1)) {
                    st.comp.stages[i] = p->stages[i];
                    st.comp.stage_mask |= 1u << i;
                } else {
                    memset(&st.comp.stages[i], 0, sizeof st.comp.stages[i]);
                    st.comp.stage_mask &= ~(1u << i);
                }
            }
            st.program = st.comp.stage_mask ? &st.comp : NULL;
            break;
        }
        case CMD_BIND_BLEND: { uint8_t *b = x; memcpy(st.blend[b[0] & 7], b, 8); break; }
        case CMD_BIND_CHANNEL_MASK: memcpy(&st.channel_mask, x, 4); break;
        case CMD_BIND_COLOR: memcpy(st.color_state, x, 4); break;
        case CMD_BIND_DEPTH_STENCIL: memcpy(st.ds, x, 8); break;
        case CMD_BIND_MULTISAMPLE: memcpy(st.ms, x, 0x18); break;
        case CMD_BIND_POLYGON: memcpy(st.poly, x, 4); break;
        case CMD_BIND_VERTEX_ATTRIB:
            st.nattrib = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.attrib, x, st.nattrib * 4);
            break;
        case CMD_BIND_VERTEX_STREAM:
            st.nstream = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.stream, x, st.nstream * 8);
            break;
        case CMD_BIND_VERTEX_BUFFERS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(&st.vbuf[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_UNIFORM_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 18 && r->x[0] < 6; i++)
                memcpy(&st.ubuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_STORAGE_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 16 && r->x[0] < 6; i++)
                memcpy(&st.sbuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_TEXTURES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 32 && r->x[0] < 6; i++)
                memcpy(&st.tex[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_BIND_IMAGES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 8 && r->x[0] < 6; i++)
                memcpy(&st.img[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_TEXTURE_POOL: st.tpool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_SAMPLER_POOL: st.spool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_VIEWPORTS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.viewport[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_SCISSORS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.scissor[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_DEPTH_RANGES:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.depth_range[r->x[0] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_BLEND_COLOR: memcpy(st.blend_color, x, 16); break;
        case CMD_SET_STENCIL: {
            int face = (int)r->x[0];
            for (int f = 0; f < 2; f++) {
                if (!(face & (1 << f))) continue;
                if (r->x[6] == 0) st.stencil_ref[f] = (uint32_t)r->x[1];
                else if (r->x[6] == 1) st.stencil_mask[f] = (uint32_t)r->x[1];
                else st.stencil_vmask[f] = (uint32_t)r->x[1];
            }
            break;
        }
        case CMD_SET_POLYGON_OFFSET: memcpy(st.poly_offset, r->f, sizeof st.poly_offset); break;
        case CMD_SET_DEPTH_CLAMP: st.depth_clamp = r->x[0] & 1; break;
        case CMD_SET_PRIMITIVE_RESTART: st.prim_restart = r->x[0] != 0; st.restart_index = (uint32_t)r->x[1]; break;
        case CMD_SET_COPY_STRIDES: st.copy_row_stride = r->x[0]; st.copy_image_stride = r->x[1]; break;
        case CMD_SET_RASTERIZER_DISCARD: raster_discard = r->x[0] & 1; break;
        case CMD_BARRIER: case CMD_SET_ALPHA_REF: case CMD_SET_DEPTH_BOUNDS: case CMD_SET_LINE_WIDTH: case CMD_SET_POINT_SIZE:
        case CMD_SET_SAMPLE_MASK:
            break;
        case CMD_SET_RENDER_TARGETS: {  // (the pass's own targets again: nothing changes)
            RtState t;
            rt_from_rec(r, &t);
            if (memcmp(&t, &st.rt, sizeof t)) { sw_cut_r = r; sw_cut_li = sw_walk_li; return; }
            break;
        }
        case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS: case CMD_DRAW_ARRAYS_INDIRECT: case CMD_DRAW_ELEMENTS_INDIRECT:
            if (sw_walk_risky()) { sw_cut_r = r; sw_cut_li = sw_walk_li; sw_cut_risky = true; return; }  // (the run ends before it)
            sw_walk_draw(r);
            break;
        case CMD_REPORT_COUNTER: {  // (its CPU part and query slot: exec_list's, for this record alone)
            const CmdRec *stop0 = sw_stop;
            sw_stop = (const CmdRec *)((const uint8_t *)r + r->size);
            exec_list((uint64_t)(uintptr_t)r, depth);
            sw_stop = stop0;
            break;
        }
        default:  // (a record a run can't hold: the run ends before it)
            sw_cut_r = r;
            sw_cut_li = sw_walk_li;
            return;
        }
        p += r->size;
    }
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// A run: lists h[0..n) from record first (in h[0]'s list), draws in all; translated by the workers, its state walked
// here (st_main ends as after the run)
static void sw_async_dispatch(const uint64_t *h, uint64_t n, int depth, const CmdRec *stop, int parts);
static void sw_run(const uint64_t *h, uint64_t n, uint64_t first, int depth, int draws, const CmdRec *stop) {
    // the pass, its contents secondary command buffers (a pass already begun inline ends: its contents are kept)
    uint64_t trun0 = SDL_GetTicksNS();
    bool want_async = sw_async_on() && sw_async_pass_ok();
    if (!want_async) sw_async_sync();  // (this run waits for its workers: the async ones first)
    xm_flush_back();
    begin_pass();  // (the pass, if not begun yet - with maintenance7 it takes the secondaries along with inline draws)
    sw_walk_count = sw_walk_k = 0;
    sw_nsnap = 0;
    sw_ncq = 0;
    SwJob *j0 = &sw_jobs[0];
    tstate_copy(&j0->tst, &st);
    j0->li = 0;
    j0->p = first;
    uint64_t tw0 = SDL_GetTicksNS();
    sw_walking = true;
    sw_stop = stop;
    sw_cut_r = NULL;
    sw_cut_risky = false;
    static int old_walk = -1;  // (LARECOMP_VK_SW_WALK_EXEC=1: the walk through exec_list, as before)
    if (old_walk < 0) old_walk = getenv("LARECOMP_VK_SW_WALK_EXEC") != NULL;
    for (uint64_t li = 0; li < n && !sw_cut_r; li++) {
        sw_walk_li = li;
        if (old_walk) exec_list(li ? h[li] : first, depth);
        else walk_list(li ? h[li] : first, depth);
    }
    sw_stat_walk_ns += SDL_GetTicksNS() - tw0;
    sw_stop = NULL;
    sw_walking = false;
    if (sw_walk_count < SW_MIN_DRAWS) {  // (too few for a run: the state as it was, and those records here)
        const CmdRec *cut = sw_cut_r;
        uint64_t last = cut ? sw_cut_li : n - 1;
        tstate_copy(&st_main, &j0->tst);
        sw_reexec = true;  // (the walk made the counters' CPU part and took their query slots)
        sw_stop = cut ? cut : stop;
        for (uint64_t li = 0; li <= last && li < n; li++) exec_list(li ? h[li] : first, depth);
        sw_stop = NULL;
        sw_reexec = false;
        sw_cut_r = cut;  // (the caller goes on there)
        sw_stat_rewound++;
        return;
    }
    draws = sw_walk_count;
    if (!pass_mixed) {  // (else one of secondaries only: the pass begun inline ends, its contents kept)
        vk_end_pass();
        pass_secondary = true;
        begin_pass();
        pass_secondary = false;
    }
    int parts = sw_n + 1;  // (async: the render thread makes the parts left when it waits - sw_async_sync)
    if (parts > draws / 16) parts = draws / 16 > 1 ? draws / 16 : 1;
    // the parts' starts: the snapshots nearest an even split
    sw_walk_parts = parts;
    sw_walk_bound[0] = 0;
    int k_ = 1, prev = 0;
    for (int k = 1; k < parts; k++) {
        int idx = (int)(((int64_t)draws * k / parts + SW_SNAP_EVERY / 2) / SW_SNAP_EVERY);  // (snapshot idx - 1: after idx x 32 draws)
        if (idx > sw_nsnap) idx = sw_nsnap;
        if (idx <= prev || idx * SW_SNAP_EVERY >= draws) continue;
        prev = idx;
        SwJob *j = &sw_jobs[k_];
        tstate_copy(&j->tst, &sw_snap[idx - 1].tst);
        j->li = sw_snap[idx - 1].li;
        j->p = sw_snap[idx - 1].p;
        sw_walk_bound[k_++] = idx * SW_SNAP_EVERY;
    }
    parts = sw_walk_parts = k_;
    sw_walk_bound[parts] = draws;
    sw_walk_k = parts - 1;
    if (want_async && !sw_ncq) {
        sw_async_dispatch(h, n, depth, stop, parts);
        goto run_done;
    }
    sw_async_sync();  // (counters in it: the waiting path)
    for (int k = 0; k < parts; k++) {
        SwJob *j = &sw_jobs[k];
        j->h = h;
        j->ln = n;
        j->depth = depth;
        int to = sw_walk_bound[k + 1] < sw_walk_count ? sw_walk_bound[k + 1] : sw_walk_count;  // (a cut: the draws walked)
        j->budget = k <= sw_walk_k && to > sw_walk_bound[k] ? to - sw_walk_bound[k] : 0;
        j->stop = stop;
        j->cb = sw_secondary(k);
    }
    // the workers (part 0 here)
    sw_ord_base = ++sw_ord_main << 24;
    pthread_mutex_lock(&sw_mu);
    sw_njobs = parts;
    sw_left = sw_n;
    sw_gen++;
    pthread_cond_broadcast(&sw_go);
    pthread_mutex_unlock(&sw_mu);
    sw_chunk(0, &sw_jobs[0]);  // (part 0 on the render thread, on its copy: st_main stays at the run's end)
    uint64_t tq0 = SDL_GetTicksNS();
    pthread_mutex_lock(&sw_mu);
    while (sw_left) pthread_cond_wait(&sw_done, &sw_mu);
    pthread_mutex_unlock(&sw_mu);
    sw_stat_wait_ns += SDL_GetTicksNS() - tq0;
    for (int k = 1; k < parts; k++) ctr_fold(k);
    int ab = -1;  // (the first part that gave a draw back: the parts after it are dropped, it takes over from that draw)
    for (int k = 0; k < parts && ab < 0; k++) if (sw_jobs[k].budget && sw_jobs[k].abort_r) ab = k;
    VkCommandBuffer cbs[SW_MAX];
    int ncb = 0;
    for (int k = 0; k < parts && (ab < 0 || k <= ab); k++) if (sw_jobs[k].budget) cbs[ncb++] = sw_jobs[k].cb;
    if (ncb) rec_CmdExecuteCommands(vk_cmd(), (uint32_t)ncb, cbs);
    if (!pass_mixed) vk_end_pass();  // (a pass of secondaries takes nothing else: the next draw begins an inline one)
    ss_forget();  // (the secondaries left the dynamic state undefined)
    if (ab >= 0) {
        SwJob *j = &sw_jobs[ab];
        bool own = j->tst.program == &j->tst.comp;
        st_main = j->tst;
        if (own) st_main.program = &st_main.comp;
        sw_reexec = true;
        sw_stop = sw_cut_r ? sw_cut_r : stop;  // (a cut run: up to the cut, where the caller goes on)
        uint64_t last = sw_cut_r ? sw_cut_li : n - 1;
        for (uint64_t li = j->abort_li; li <= last && li < n; li++) exec_list(li == j->abort_li ? (uint64_t)(uintptr_t)j->abort_r : h[li], depth);
        sw_stop = NULL;
        sw_reexec = false;
        sw_stat_aborts++;
    }
run_done:
    sw_stat_run_ns += SDL_GetTicksNS() - trun0;
    sw_stat_runs++;
    sw_stat_draws += (uint64_t)sw_walk_count;
    if (sw_cut_r && sw_cut_risky) sw_pass_cut_note();  // (a draw of the render thread's ended it: the pass is its own now)
    {
        static uint64_t last;
        uint64_t f = vk_frame_counter();
        if (!last) last = f;
        if (f >= last + 600) {
            double fr = (double)(f - last);
            LOG("[sw] %.1f runs (%.0f draws) a frame to the shadow workers, %.1f too short for one; walk %.2f ms, wait %.2f ms "
                "a frame (runs %.2f ms in all); %.2f runs a frame taken back\n", sw_stat_runs / fr, sw_stat_draws / fr,
                sw_stat_rewound / fr, sw_stat_walk_ns / 1e6 / fr, sw_stat_wait_ns / 1e6 / fr, sw_stat_run_ns / 1e6 / fr, sw_stat_aborts / fr);
            if (sw_stat_async_runs)
                LOG("[sw] async: %.2f runs a frame, %.2f waits a frame, %.2f ms waiting a frame, %llu lost\n", sw_stat_async_runs / fr,
                    sw_stat_async_syncs / fr, sw_stat_async_wait_ns / 1e6 / fr, (unsigned long long)sw_stat_async_lost);
            sw_stat_runs = sw_stat_draws = sw_stat_rewound = sw_stat_walk_ns = sw_stat_wait_ns = sw_stat_run_ns = sw_stat_aborts = 0;
            sw_stat_async_runs = sw_stat_async_syncs = sw_stat_async_wait_ns = 0;
            last = f;
        }
    }
}

// (the render thread) an async run's parts to the workers (the walk made their states: sw_jobs), and its secondaries
// to the pass once they are done (by the recorder)
static void sw_async_dispatch(const uint64_t *h, uint64_t n, int depth, const CmdRec *stop, int parts) {
    SwRunRec *rr = &sw_ar[sw_ar_next];
    sw_ar_next = (sw_ar_next + 1) % SW_AR;
    if (rr->used) {  // (the recorder has executed it long ago: 16 runs back)
        for (uint32_t k = 0; !atomic_load_explicit(&rr->consumed, memory_order_acquire); k++) { if (k < 20000) __builtin_ia32_pause(); else SDL_Delay(0); }
    }
    rr->used = true;
    rr->aborted = false;
    rr->draws = sw_walk_count;
    memset(rr->cbs, 0, sizeof rr->cbs);
    atomic_store_explicit(&rr->consumed, 0, memory_order_relaxed);
    atomic_store_explicit(&rr->left, parts, memory_order_release);
    if (!sw_ainflight) {
        extern void rec_queue_shared(bool on);
        rec_queue_shared(true);
        sw_ainflight = sw_rt_lock = true;
    }
    for (int k = 0; k < parts; k++) {
        SwAJob *a = &sw_aj[sw_aj_head % SW_AJ];
        for (uint32_t q = 0; sw_aj_head >= SW_AJ && !atomic_load_explicit(&a->done, memory_order_acquire); q++) {  // (a free job)
            if (q < 20000) __builtin_ia32_pause(); else SDL_Delay(0);
        }
        SwJob *j = &sw_jobs[k];
        a->j = *j;
        if (j->tst.program == &j->tst.comp) a->j.tst.program = &a->j.tst.comp;
        a->j.h = h;
        if (n == 1) { a->hbuf[0] = h[0]; a->j.h = a->hbuf; }  // (sw_tail's list: its array is static)
        a->j.ln = n;
        a->j.depth = depth;
        int to = sw_walk_bound[k + 1] < sw_walk_count ? sw_walk_bound[k + 1] : sw_walk_count;  // (a cut: the draws walked)
        a->j.budget = k <= sw_walk_k && to > sw_walk_bound[k] ? to - sw_walk_bound[k] : 0;
        a->j.stop = stop;
        a->j.cb = VK_NULL_HANDLE;
        a->pg = pass_main;
        a->k = k;
        a->slot = cur_sub;
        a->run = rr;
        atomic_store_explicit(&a->done, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&sw_aj_out, 1, memory_order_relaxed);
        pthread_mutex_lock(&sw_mu);
        sw_aj_head++;
        pthread_mutex_unlock(&sw_mu);
    }
    pthread_mutex_lock(&sw_mu);
    pthread_cond_broadcast(&sw_go);
    pthread_mutex_unlock(&sw_mu);
    extern void rec_CmdExecuteCommandsWait(VkCommandBuffer cb, _Atomic int *left, const VkCommandBuffer *cbs, uint32_t n, _Atomic int *consumed);
    rec_CmdExecuteCommandsWait(vk_cmd(), &rr->left, rr->cbs, (uint32_t)parts, &rr->consumed);
    if (!pass_mixed) vk_end_pass();  // (a pass of secondaries takes nothing else: the next draw begins an inline one)
    ss_forget();  // (the secondaries leave the dynamic state undefined)
    sw_stat_async_runs++;
}

// (a CALL's lists) the run of lists from h[i] (of n): the lists it took (0: none). Too few draws: *skip = the first
// list a run could start at (the lists scanned up to one that isn't only state and draws can't make one either - every
// later start had scanned them again: quadratic, 12 % of the render thread in a town at World detail 4x)
static uint64_t sw_try(const uint64_t *h, uint64_t i, uint64_t n, int depth, uint64_t *skip) {
    if (!sw_pass_ok()) return 0;
    {   // (a count first, of the records up to SW_MIN_DRAWS draws: lists with a record a run can't hold before as many
        // draws make none - ~300 tries a frame in a town at World detail 4x walked them, copied the state twice and
        // rewound, ~1.5 ms. Those lists aren't tried again from any start before that record: *skip)
        int draws = 0;
        uint64_t li = i;
        for (; li < n && draws < SW_MIN_DRAWS; li++) {
            uint64_t q = h[li];
            for (int g = 0; g < 100000 && draws < SW_MIN_DRAWS; g++) {
                const CmdRec *r = (const CmdRec *)(uintptr_t)q;
                if (r->op == CMD_END) break;
                if (r->op == CMD_JUMP) { q = r->x[0]; continue; }
                if (r->op == CMD_DRAW_ARRAYS || r->op == CMD_DRAW_ELEMENTS || r->op == CMD_DRAW_ARRAYS_INDIRECT ||
                    r->op == CMD_DRAW_ELEMENTS_INDIRECT)
                    draws++;
                else if (!sw_walk_op_ok(r)) {
                    *skip = li + 1;
                    return 0;
                }
                q += r->size;
            }
        }
        if (draws < SW_MIN_DRAWS) { *skip = n; return 0; }
    }
    // (no count scan first - the walk stops at the first record a run can't hold, and translates them here when
    // there were too few draws: the scan read every record once more, 10 % of the render thread)
    uint64_t m = n - i < SW_LISTS ? n - i : SW_LISTS;
    sw_run(h + i, m, h[i], depth + 1, 0, NULL);
    if (sw_cut_r) {  // (the rest of that list here - its draws may make a run again - and the CALL goes on after it)
        const CmdRec *c = sw_cut_r;
        uint64_t li = sw_cut_li;
        sw_cut_r = NULL;
        exec_list((uint64_t)(uintptr_t)c, depth + 1);
        return li + 1;
    }
    return m;
}

// (a draw in a list) the list from record r up to its first record that isn't state or a draw (or its end): true - done,
// the render thread goes on at sw_tail_stop (NULL: the list's end)
static int sw_tail(const CmdRec *r, int depth) {  // 1: done; 0: not in this list; -1: not this draw (the next may)
    if (!sw_pass_ok()) return 0;
    if (sw_walk_risky()) { sw_pass_cut_note(); return 0; }  // (this draw is the render thread's, and so is the pass this frame)
    const CmdRec *stop;
    int draws = sw_list_draws_at((uint64_t)(uintptr_t)r, &stop);
    if (draws < SW_MIN_DRAWS) return 0;
    static uint64_t one[1];
    one[0] = (uint64_t)(uintptr_t)r;
    sw_run(one, 1, one[0], depth, draws, stop);
    sw_tail_stop = sw_cut_r ? sw_cut_r : stop;  // (cut: the render thread goes on at that draw)
    sw_cut_r = NULL;
    return 1;
}

// ================================================================== executor (renderer.m exec_list)
static void exec_list(uint64_t handle, int depth) {
    TLS_LOCALS();
#define st_cur() st_l_
#define ss_cur() ss_l_
#define pass_cur() pass_l_
#define sw_ctx() ctx_l_
    uint64_t p = handle;
    bool sw_no = false;  // (sw_tail said no in this list: not again until its targets change - a scan of the rest each draw)
    for (int guard = 0; guard < 10000000; guard++) {
        CmdRec *r = (CmdRec *)(uintptr_t)p;
        if ((sw_walking || sw_cb || sw_reexec) && r == sw_stop) return;  // (a shadow workers' run ends here)
        if (sw_walking && !sw_walk_op_ok(r)) { sw_cut_r = r; sw_cut_li = sw_walk_li; return; }  // (the run ends before it)
        if (sw_ainflight && !sw_walking && !sw_cb && !sw_async_op_ok(r)) sw_async_sync();
        bool sw_lk = sw_rt_lock && !sw_walking && (r->op == CMD_CLEAR_COLOR || r->op == CMD_CLEAR_DEPTH_STENCIL);
        if (sw_lk) SWL();  // (snapshots, aliases, textures: shared with the async runs' workers)
        // (the records ahead: the game's worker threads wrote them on other cores, and the first read of each one
        // missed the caches - 5 % of the render thread on this line. Records follow one another within a chunk;
        // a prefetch past its end is harmless)
        __builtin_prefetch((const char *)r + 256);
        __builtin_prefetch((const char *)r + 384);
        if (sw_walking) {  // (the walk spends less on each record: further ahead)
            __builtin_prefetch((const char *)r + 1024);
            __builtin_prefetch((const char *)r + 2048);
        }
        void *x = r->extra;
        if (r->op < CMD_COUNT) op_counts[r->op]++;
        if (xm_back_n && !xm_state_op(r->op) && !(xfb_on && (r->op == CMD_DRAW_ARRAYS || r->op == CMD_DRAW_ELEMENTS)))
            xm_flush_back();  // (xm_batch: the sessions' output, to the heap before anything that may read it)
        switch (r->op) {
        case CMD_END: return;
        case CMD_JUMP: p = r->x[0]; continue;
        case CMD_CALL: {
            const uint64_t *h = x;
            uint64_t skip = 0;
            for (uint64_t i = 0; i < r->x[0] && depth < 8; i++) {
                uint64_t done = i >= skip ? sw_try(h, i, r->x[0], depth, &skip) : 0;  // (lists translated by the shadow workers)
                if (done) { i += done - 1; continue; }
                exec_list(h[i], depth + 1);
            }
            break;
        }
        case CMD_BIND_PROGRAM: {  // (program, NVNshaderStageBits): a game may bind its stages from several programs
            NvnProgram *p = (NvnProgram *)(uintptr_t)r->x[0];
            uint32_t m = (uint32_t)r->x[1] & 0x3F;
            if (!m) m = 0x3F;
            if (ENV("LARECOMP_LOG_BINDPROG")) {
                static int n;
                if (n++ < 40) LOG("[bindprog] %p stages %#x (program has %#x)\n", (void *)p, m, p ? p->stage_mask : 0);
            }
            for (int i = 0; i < 6; i++) {
                if (!(m >> i & 1)) continue;
                if (p && (p->stage_mask >> i & 1)) {
                    st.comp.stages[i] = p->stages[i];
                    st.comp.stage_mask |= 1u << i;
                } else {
                    memset(&st.comp.stages[i], 0, sizeof st.comp.stages[i]);
                    st.comp.stage_mask &= ~(1u << i);
                }
            }
            st.program = st.comp.stage_mask ? &st.comp : NULL;
            break;
        }
        case CMD_BIND_BLEND: { uint8_t *b = x; memcpy(st.blend[b[0] & 7], b, 8); break; }
        case CMD_BIND_CHANNEL_MASK: memcpy(&st.channel_mask, x, 4); break;
        case CMD_BIND_COLOR: memcpy(st.color_state, x, 4); break;
        case CMD_BIND_DEPTH_STENCIL: memcpy(st.ds, x, 8); break;
        case CMD_BIND_MULTISAMPLE: memcpy(st.ms, x, 0x18); break;
        case CMD_BIND_POLYGON: memcpy(st.poly, x, 4); break;
        case CMD_BIND_VERTEX_ATTRIB:
            st.nattrib = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.attrib, x, st.nattrib * 4);
            break;
        case CMD_BIND_VERTEX_STREAM:
            st.nstream = (int)r->x[0] > 16 ? 16 : (int)r->x[0];
            memcpy(st.stream, x, st.nstream * 8);
            break;
        case CMD_BIND_VERTEX_BUFFERS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(&st.vbuf[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_UNIFORM_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 18 && r->x[0] < 6; i++)
                memcpy(&st.ubuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_STORAGE_BUFFERS:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 16 && r->x[0] < 6; i++)
                memcpy(&st.sbuf[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_BIND_TEXTURES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 32 && r->x[0] < 6; i++)
                memcpy(&st.tex[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_BIND_IMAGES:
            for (uint64_t i = 0; i < r->x[2] && r->x[1] + i < 8 && r->x[0] < 6; i++)
                memcpy(&st.img[r->x[0]][r->x[1] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_TEXTURE_POOL: st.tpool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_SAMPLER_POOL: st.spool = (NvnDescPool *)(uintptr_t)r->x[0]; break;
        case CMD_SET_RENDER_TARGETS: {
            RtState prev = st.rt;  // the same targets again: keep the pass (a restart = full barrier + reload)
            rt_from_rec(r, &st.rt);
            sw_no = false;
            static int always = -1;
            if (always < 0) always = ENV("LARECOMP_VK_PASS_RESTART") != NULL;
            if (always || memcmp(&prev, &st.rt, sizeof prev)) st.rt_dirty = true;
            break;
        }
        case CMD_SET_VIEWPORTS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.viewport[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_SCISSORS:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.scissor[r->x[0] + i], (uint8_t *)x + i * 16, 16);
            break;
        case CMD_SET_DEPTH_RANGES:
            for (uint64_t i = 0; i < r->x[1] && r->x[0] + i < 16; i++) memcpy(st.depth_range[r->x[0] + i], (uint8_t *)x + i * 8, 8);
            break;
        case CMD_SET_BLEND_COLOR: memcpy(st.blend_color, x, 16); break;
        case CMD_SET_STENCIL: {
            int face = (int)r->x[0];
            for (int f = 0; f < 2; f++) {
                if (!(face & (1 << f))) continue;
                if (r->x[6] == 0) st.stencil_ref[f] = (uint32_t)r->x[1];
                else if (r->x[6] == 1) st.stencil_mask[f] = (uint32_t)r->x[1];
                else st.stencil_vmask[f] = (uint32_t)r->x[1];
            }
            break;
        }
        case CMD_SET_POLYGON_OFFSET: memcpy(st.poly_offset, r->f, sizeof st.poly_offset); break;
        case CMD_SET_DEPTH_CLAMP: st.depth_clamp = r->x[0] & 1; break;
        case CMD_SET_PRIMITIVE_RESTART: st.prim_restart = r->x[0] != 0; st.restart_index = (uint32_t)r->x[1]; break;
        case CMD_SET_COPY_STRIDES: st.copy_row_stride = r->x[0]; st.copy_image_stride = r->x[1]; break;
        case CMD_CLEAR_COLOR: {
            snap_reason = "clear-color"; snapshots_invalidate(NULL);
            int index = (int)r->x[0];
            if (ENV("LARECOMP_LOG_CLEARS") && index < st.rt.ncolor && st.rt.color[index])
                LOG("[cclear] frame %llu idx %d %ux%u f%#x mask %#x\n", (unsigned long long)renderer_present_count(), index,
                    st.rt.color[index]->width, st.rt.color[index]->height, st.rt.color[index]->format, (uint32_t)r->x[2]);
            if (ct_on() && index < st.rt.ncolor && st.rt.color[index])
                LOG("[ct] clear-color %p mask %#x sc %d %d %d %d\n", (void *)st.rt.color[index], (uint32_t)r->x[2], st.scissor[0][0], st.scissor[0][1],
                    st.scissor[0][2], st.scissor[0][3]);
            if (index < st.rt.ncolor && st.rt.color[index]) alias_before_write(st.rt.color[index], 0, "clear");
            if (index < st.rt.ncolor && st.rt.color[index]) clear(index, (int)r->x[5], x, (uint32_t)r->x[2], false, 0, false, 0, 0);
            break;
        }
        case CMD_CLEAR_TEXTURE: {
            // ClearTexture(cmd, texture, view, region, color, mask): a render-target clear of the texture alone (renderer.m)
            const struct { uint64_t tex; NvnTextureView view; uint8_t has_view; int32_t region[6]; float color[4]; } *ct = x;
            NvnTexture *t = (NvnTexture *)(uintptr_t)ct->tex;
            if (!t) break;
            snap_reason = "clear-texture"; snapshots_invalidate(NULL);
            alias_before_write(t, 0, "clear");
            RtState saved = st.rt;
            int32_t sc[4];
            memcpy(sc, st.scissor[0], sizeof sc);
            vk_end_pass();
            memset(&st.rt, 0, sizeof st.rt);
            st.rt.ncolor = 1;
            st.rt.color[0] = t;
            if (ct->has_view) { st.rt.cview[0] = ct->view; st.rt.has_cview[0] = 1; }
            st.rt_dirty = true;
            const int32_t *rg = ct->region;
            if (rg[3] > 0 && rg[4] > 0) { st.scissor[0][0] = rg[0]; st.scissor[0][1] = rg[1]; st.scissor[0][2] = rg[3]; st.scissor[0][3] = rg[4]; }
            else { st.scissor[0][0] = st.scissor[0][1] = 0; st.scissor[0][2] = st.scissor[0][3] = 16384; }
            clear(0, (int)r->x[6], ct->color, (uint32_t)r->x[4] ? (uint32_t)r->x[4] : 0xF, false, 0, false, 0, 0);
            vk_end_pass();
            st.rt = saved;
            memcpy(st.scissor[0], sc, sizeof sc);
            st.rt_dirty = true;
            break;
        }
        case CMD_CLEAR_DEPTH_STENCIL:
            snap_reason = "clear-depth"; snapshots_invalidate(NULL);
            if (ct_on() && st.rt.depth)
                LOG("[ct] clear-depth %p d%d s%d sc %d %d %d %d\n", (void *)st.rt.depth, r->x[0] != 0, r->x[2] != 0, st.scissor[0][0],
                    st.scissor[0][1], st.scissor[0][2], st.scissor[0][3]);
            if (st.rt.depth && naliases) {  // (copy aliases: which aspects of the depth target this clear replaces whole)
                NvnTexture *d = st.rt.depth;
                bool whole = st.scissor[0][0] <= 0 && st.scissor[0][1] <= 0 && (int64_t)st.scissor[0][0] + st.scissor[0][2] >= (int64_t)d->width &&
                             (int64_t)st.scissor[0][1] + st.scissor[0][3] >= (int64_t)d->height &&
                             !(st.rt.has_dview && (((st.rt.dview.setmask & 1) && st.rt.dview.base_level) ||
                                                   ((st.rt.dview.setmask & 2) && st.rt.dview.min_layer)));
                for (int i = 0; i < st.rt.ncolor; i++)  // (the clear covers the smallest attachment only)
                    if (st.rt.color[i] && (st.rt.color[i]->width < d->width || st.rt.color[i]->height < d->height)) whole = false;
                VkImageAspectFlags full = 0;
                if (whole && r->x[0] != 0) full |= VK_IMAGE_ASPECT_DEPTH_BIT;
                if (whole && renderer_format_has_stencil(d->format) && (r->x[2] & 0xFF) == 0xFF) full |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (!renderer_format_has_stencil(d->format)) full |= VK_IMAGE_ASPECT_STENCIL_BIT;
                if (ct_on())
                    LOG("[ct] alias clear %p full %#x (whole %d, x0 %llu x1 %#llx x2 %#llx, dview %d/%#x, ncolor %d)\n", (void *)d, full, whole,
                        (unsigned long long)r->x[0], (unsigned long long)r->x[1], (unsigned long long)r->x[2], st.rt.has_dview,
                        st.rt.dview.setmask, st.rt.ncolor);
                alias_before_write(d, full, "clear");
            }
            if (st.rt.depth) {
                bool has_stencil = renderer_format_has_stencil(st.rt.depth->format);
                clear(-1, 0, NULL, 0, r->x[0] != 0, r->f[0], has_stencil && r->x[2] != 0, (uint32_t)r->x[1], (uint32_t)r->x[2]);
            }
            break;
        case CMD_DISPATCH: dispatch_compute((uint32_t)r->x[0], (uint32_t)r->x[1], (uint32_t)r->x[2]); break;
        case CMD_DRAW_ARRAYS: case CMD_DRAW_ELEMENTS:
            if (sw_walking) {
                if (sw_walk_risky()) { sw_cut_r = r; sw_cut_li = sw_walk_li; sw_cut_risky = true; return; }  // (the run ends before it)
                sw_walk_draw(r);
                break;
            }
            if (sw_cb) {  // (a worker: its part ends at its last draw; a draw that needs the render thread goes back)
                if (sw_budget <= 0) return;
                jmp_buf jb;
                sw_jmp = &jb;
                if (setjmp(jb)) { sw_jmp = NULL; SWU_ALL(); sw_abort_r = r; return; }
                draw(r, r->op == CMD_DRAW_ELEMENTS);
                sw_jmp = NULL;
                if (--sw_budget <= 0) return;  // (its part ends here: what follows is the next part's, or not a run's)
                break;
            } else if (!sw_no) {  // (this list up to its next record that isn't state or a draw: the shadow workers)
                int t = sw_tail(r, depth);
                if (t > 0) { if (!sw_tail_stop) return; p = (uint64_t)(uintptr_t)sw_tail_stop; continue; }
                if (!t) sw_no = true;
            }
            draw(r, r->op == CMD_DRAW_ELEMENTS);
            break;
        case CMD_BEGIN_XFB: xfb_on = true; stat_xfb_sessions++; break;
        case CMD_END_XFB: {
            xfb_on = false;
            // the mirror's output, to the heap right away: shaders read it there as storage (e.g. the characters'
            // shading - left for the submission's end, cutscenes drew them black and scenery vanished)
            if (xm_back_n && !xm_batch()) xm_flush_back();  // (else when something else comes: xm_batch)
            static int nlog = -1;  // (debug) LARECOMP_LOG_XFB_DATA=<n>: the first entries of n sessions' buffers, as written
            if (nlog < 0) nlog = ENV("LARECOMP_LOG_XFB_DATA") ? atoi(ENV("LARECOMP_LOG_XFB_DATA")) : 0;
            if (nlog > 0 && renderer_present_count() > 600 && gmem_buf) {
                nlog--;
                vk_end_pass();
                for (int b = 0; b < 3; b++) {
                    uint64_t a = xfb_addr[b];
                    if (!a || !GM_IN(a, 64)) continue;
                    VkSpan rb = vk_readback(64);
                    VkBufferCopy bc = {GM_OFF(a), rb.off, 64};
                    vk_full_barrier(vk_cmd());
                    rec_CmdCopyBuffer(vk_cmd(), gmem_buf, rb.buf, 1, &bc);
                    renderer_flush();
                    const float *f = rb.ptr;
                    const uint32_t *u = rb.ptr;
                    LOG("[xfb] buffer %d %llx: %g %g %g %g %g %g | %08x %08x %08x %08x\n", b, (unsigned long long)a, f[0], f[1], f[2], f[3], f[4],
                        f[5], u[0], u[1], u[2], u[3]);
                }
            }
            break;
        }
        case CMD_BIND_XFB_BUFFER:
            if (r->x[0] < 4) { xfb_addr[r->x[0]] = r->x[1]; xfb_size[r->x[0]] = r->x[2]; }
            break;
        case CMD_BIND_XFB_BUFFERS: {
            const uint64_t *b = x;
            for (uint64_t i = 0; i < r->x[1] && i < 4 && r->x[0] + i < 4; i++) {
                xfb_addr[r->x[0] + i] = b[2 * i];
                xfb_size[r->x[0] + i] = b[2 * i + 1];
            }
            break;
        }
        case CMD_SET_RASTERIZER_DISCARD: raster_discard = r->x[0] & 1; break;
        case CMD_DRAW_XFB: {
            static int nlog;
            if (nlog++ < 4) LOG("[xfb] DrawTransformFeedback mode %llu control %llx\n", (unsigned long long)r->x[0], (unsigned long long)r->x[1]);
            break;
        }
        case CMD_DRAW_ARRAYS_INDIRECT: case CMD_DRAW_ELEMENTS_INDIRECT:
            if (sw_walking) {
                if (sw_walk_risky()) { sw_cut_r = r; sw_cut_li = sw_walk_li; sw_cut_risky = true; return; }
                sw_walk_draw(r);
                break;
            }
            if (sw_cb) {
                if (sw_budget <= 0) return;
                jmp_buf jb;
                sw_jmp = &jb;
                if (setjmp(jb)) { sw_jmp = NULL; SWU_ALL(); sw_abort_r = r; return; }
                draw_indirect(r);
                sw_jmp = NULL;
                if (--sw_budget <= 0) return;  // (its part ends here: what follows is the next part's, or not a run's)
                break;
            } else if (!sw_no) {
                int t = sw_tail(r, depth);
                if (t > 0) { if (!sw_tail_stop) return; p = (uint64_t)(uintptr_t)sw_tail_stop; continue; }
                if (!t) sw_no = true;
            }
            draw_indirect(r);
            break;
        case CMD_MULTI_DRAW_ARRAYS_INDIRECT_COUNT: case CMD_MULTI_DRAW_ELEMENTS_INDIRECT_COUNT:
            draw_indirect(r);  // (never in a shadow workers' run: sw_list_draws)
            break;
        case CMD_UPDATE_UNIFORM_BUFFER:
            memcpy((void *)(uintptr_t)(r->x[0] + r->x[2]), x, r->x[3]);
            guest_copy_flush();  // (cbuf copies shared within the submission are stale now)
            break;
        case CMD_FENCE_SYNC: {
            static int fence_flush = -1;
            if (fence_flush < 0) fence_flush = ENV("LARECOMP_FENCE_FLUSH") ? 1 : 0;
            if (fence_flush) renderer_flush();
            extern void nvn_sync_set_seq(uint64_t guest, uint64_t seq);
            if (r->x[1]) nvn_sync_set_seq(r->x[1], renderer_fence_seq());
            list_fenced = true;
            break;
        }
        case CMD_REPORT_COUNTER: {
            if (sw_cb || sw_reexec) {  // (a shadow worker, or a take-over: the walk did the CPU part and took the query slot)
                for (int k = 0; k < sw_ncq; k++)
                    if (sw_cq[k].r == r) { rec_CmdWriteTimestamp2(vk_cmd(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, cq[cur_sub].pool, sw_cq[k].idx); break; }
                break;
            }
            // The game times its GPU work with these (two per frame) and drops to 30 fps when that looks slow: the
            // host clock now, replaced by the GPU's time at this point once the submission is done (cq_resolve).
            uint64_t *out = (uint64_t *)(uintptr_t)r->x[1];
            uint32_t type = (uint32_t)r->x[0];
            static uint32_t seen;
            if (type < 32 && !(seen & (1u << type))) { seen |= 1u << type; LOG("[nvn] ReportCounter type %u\n", type); }
            int mode = vk_counter_mode();
            static uint64_t frame_seen = ~0ull, frame_ts;
            extern uint64_t vk_frame_counter(void);
            if (mode == 1 && vk_frame_counter() != frame_seen) { frame_seen = vk_frame_counter(); frame_ts = ns_to_gpu_ticks(plat_now_ns()); }
            if (!out) break;
            gw_note("report counter", (uint64_t)(uintptr_t)out, 16, type);
            out[0] = 0;
            // SAMPLES_PASSED (occlusion queries: Pokémon Z-A tests its objects with point draws and skips those that
            // pass no samples): no occlusion queries here, so a count that always grows - every object visible
            static uint64_t samples;
            if (type == 1) out[0] = samples += 1u << 20;
            out[1] = mode == 1 ? frame_ts : mode == 3 ? SDL_GetTicksNS() : ns_to_gpu_ticks(plat_now_ns());  // (3 legacy: ns, the old bug)
            static uint64_t *prev_out;  // (frame pacing: the pair's start)
            // (overlap) the game reads this pair before its GPU work is done: the end counter reports the latest
            // completed pair's GPU span after this pair's start (the frame before, as a frame-late measurement)
            if (mode == 2 && overlap_on() && prev_out && out - prev_out == 2) out[1] = prev_out[1] + atomic_load(&g_last_gpu_frame_ticks);
            if (mode == 0 || mode == 2) out[1] = pace_end_counter(prev_out, out, out[1]);
            prev_out = out;
            {   // (test) LARECOMP_TEST_COUNTER_DELAY_MS=<ms>: stall the host after a frame's first counter - a slow CPU
                static double delay = -1;
                static uint64_t delayed_frame = ~0ull;
                if (delay < 0) delay = ENV("LARECOMP_TEST_COUNTER_DELAY_MS") ? atof(ENV("LARECOMP_TEST_COUNTER_DELAY_MS")) : 0;
                if (delay > 0 && vk_frame_counter() != delayed_frame) {
                    delayed_frame = vk_frame_counter();
                    SDL_DelayPrecise((Uint64)(delay * 1e6));
                }
            }
            if (mode == 2 && type == 0 && sw_walking) {  // (in a shadow workers' run: the slot now, in order; the worker writes it)
                CounterQ *q = &cq[cur_sub];
                pthread_mutex_lock(&cq_lock);
                if (q->pool && q->n < CQ_MAX && sw_ncq < SW_CQ) {
                    sw_cq[sw_ncq].r = r;
                    sw_cq[sw_ncq++].idx = q->n;
                    q->out[q->n++] = out;
                }
                pthread_mutex_unlock(&cq_lock);
            }
            if (mode == 2 && type == 0 && !sw_walking) {  // (type 0: timestamp)
                VkCommandBuffer cb = vk_cmd();  // (first: may begin the buffer, which resets the slot's queries)
                CounterQ *q = &cq[cur_sub];
                pthread_mutex_lock(&cq_lock);
                if (q->pool && q->n < CQ_MAX) {
                    rec_CmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, q->pool, q->n);
                    q->out[q->n++] = out;
                }
                pthread_mutex_unlock(&cq_lock);
            }
            break;
        }
        case CMD_COPY_TEXTURE_TO_TEXTURE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], r->x[6]); break; }
        case CMD_DOWNSAMPLE: { const TexRefRec *t = x; copy_texture(&t[0], &t[1], 1); break; }
        case CMD_COPY_BUFFER_TO_TEXTURE: copy_buffer_to_texture(r->x[0], x); break;
        case CMD_COPY_TEXTURE_TO_BUFFER: copy_texture_to_buffer(x, r->x[3]); break;
        case CMD_COPY_BUFFER_TO_BUFFER:
            wait_why = "buffer -> buffer copy";
            renderer_flush();
            gw_note("buffer copy (CPU, at translation)", r->x[1], r->x[2], r->x[0]);
            memmove((void *)(uintptr_t)r->x[1], (void *)(uintptr_t)r->x[0], r->x[2]);
            break;
        default: break;
        }
        if (sw_lk) SWU();
        p += r->size;
    }
#undef st_cur
#undef ss_cur
#undef pass_cur
#undef sw_ctx
}

// submit: end the submission after this list (else it goes on into the next list's)
static void execute_one(uint64_t handle, bool submit) {
    extern void plat_sample_focus(void);  // (runtime/platform.h: LARECOMP_SAMPLE_EXEC samples this thread)
    plat_sample_focus();
    uint64_t t0 = SDL_GetTicksNS();
    exec_list(handle, 0);
    uint64_t dt = SDL_GetTicksNS() - t0;
    cum_exec += dt;
    if (prof) prof_exec += dt;
    static int sync_submit = -1;
    if (sync_submit < 0) sync_submit = ENV("LARECOMP_SYNC_SUBMIT") ? 1 : 0;
    static long batch_draws = -1;
    if (batch_draws < 0) batch_draws = ENV("LARECOMP_VK_BATCH_DRAWS") ? atol(ENV("LARECOMP_VK_BATCH_DRAWS")) : 0;
    if (sync_submit) renderer_flush();
    else if (submit || list_fenced || (batch_draws > 0 && cum_draws - submit_draws >= (uint64_t)batch_draws)) renderer_submit();
    else if (xm_back_n && !xm_batch()) xm_flush_back();  // (as a submission's end does: a later list may read it - with xm_batch,
                                                          // its first op that isn't state does it)
    list_fenced = false;
}

void backend_execute(uint64_t handle) { execute_one(handle, true); }

// The game hands its command lists over a frame at a time (Pokémon Z-A: ~100 in one nvnQueueSubmitCommands): they go
// to the GPU in a few submissions instead of one each. ~80 submissions a frame cost ~20 % of the render thread
// (vkQueueSubmit, a new command buffer and its state, the vertex cache's write-tracking query, a ring slot each).
// A submission still ends after a list with a fence in it (a waiter on another thread must not find it pending, as
// before), after LARECOMP_VK_BATCH_DRAWS draws if set (it was 400 by default, so the GPU started on the frame while the
// rest was translated; the render passes' chunks do that now - begin_pass - and each submission's vertex cache queries
// and ring slot cost Scarlet / Violet's ~5700-draw town frames 2.3 fps: 40.2 -> 42.5 without it), and with the call. LARECOMP_VK_SUBMIT_EACH=1: one submission per list, as before.
// (profile-guided builds: make RTEXTRA=-fprofile-instr-generate) LARECOMP_PGO_DUMP_AT=<frame>: the profile written then
// (a test run ends with the process killed, before the instrumentation's own write at exit)
extern int __llvm_profile_write_file(void) __attribute__((weak));
static void pgo_dump_check(void) {
    static long at = -2;
    if (at == -2) at = getenv("LARECOMP_PGO_DUMP_AT") ? atol(getenv("LARECOMP_PGO_DUMP_AT")) : -1;
    if (at < 0 || !__llvm_profile_write_file || (long)vk_frame_counter() < at) return;
    at = -1;
    int r = __llvm_profile_write_file();
    LOG("[pgo] profile written (%d)\n", r);
}
void backend_execute_batch(const uint64_t *handles, int n) {
    pgo_dump_check();
    static int each = -1;
    if (each < 0) each = ENV("LARECOMP_VK_SUBMIT_EACH") != NULL;
    for (int i = 0; i < n; i++) execute_one(handles[i], each || i == n - 1);
    sw_async_sync();
}

// A device fault's address owner (device.c ab_explain): which of this file's buffers (or the guest window import's
// memory) the handle is, NULL when none. Called after the device is lost - only reads the tables.
const char *vk_exec_buffer_kind(uint64_t handle) {
    VkBuffer b = (VkBuffer)(uintptr_t)handle;
    if (!b) return NULL;
    static char tmp[96];
    if (b == gmem_buf || (VkDeviceMemory)(uintptr_t)handle == gmem_mem) {
        snprintf(tmp, sizeof tmp, "guest window import: guest %llx + %llu MB", (unsigned long long)(gwin_base + gmem_lo),
                 (unsigned long long)(gmem_size >> 20));
        return tmp;
    }
    if (b == xm_buf) return "transform feedback mirror";
    if (b == gfast_buf) return "GMEM_FAST compute copy";
    if (b == gorig_buf) return "GMEM_FAST merge twin";
    if (b == gc_dummy) return "guest heap chunk placeholder";
    if (b == gc_miss_buf) return "guest heap miss counters";
    for (int k = 0; k < gc_n; k++)
        if (b == gc[k].buf || (VkDeviceMemory)(uintptr_t)handle == gc[k].mem) { snprintf(tmp, sizeof tmp, "guest heap chunk %d", k); return tmp; }
    for (int i = 0; i < nchunks; i++)
        if (b == chunks[i].buf) { snprintf(tmp, sizeof tmp, "transient ring chunk %d", i); return tmp; }
    for (int i = 0; i < VC_TAB; i++)
        if (b == vc_tab[i].buf) {
            snprintf(tmp, sizeof tmp, "vertex cache: guest %llx + %llu KB", (unsigned long long)(gwin_base + vc_tab[i].lo),
                     (unsigned long long)((vc_tab[i].hi - vc_tab[i].lo) >> 10));
            return tmp;
        }
    return NULL;
}

// The [mem] line's counts (device.c): what grows over a long session shows next to the process size (#70).
void vk_exec_counts(uint32_t *progs, uint32_t *pipes) { *progs = prog_kv.n; *pipes = pipe_kv.n; }

// nvnTextureReadTexels[Strided] (objects.c, on the translating thread): copy_texture_to_buffer with the call's strides
// in place of the command buffer's (nvnCommandBufferSetCopyStrides) - the GPU's copy, waited for, into data.
void backend_texture_read(NvnTexture *t, const void *view, const int32_t region[6], void *data, uint64_t row_stride,
                          uint64_t image_stride) {
    TexRefRec ref = {.tex = (uint64_t)(uintptr_t)t};
    if (view) { ref.has_view = 1; memcpy(&ref.view, view, sizeof ref.view); }
    memcpy(ref.region, region, sizeof ref.region);
    uint64_t row = st.copy_row_stride, img = st.copy_image_stride;
    st.copy_row_stride = row_stride, st.copy_image_stride = image_stride;
    copy_texture_to_buffer(&ref, (uint64_t)(uintptr_t)data);
    st.copy_row_stride = row, st.copy_image_stride = img;
}

// ================================================================== per-object motion vectors (DLAA)
// The temporal anti-aliasing's motion vectors follow the camera (fsr2_mv.frag: the depth and the view-projections of
// this frame and the last), so a character that moves on its own - a head turning, a mouth, an arm - gets the
// camera's motion, and DLSS keeps its history where it was: a trail of the face behind a person eating at a table.
// Scarlet / Violet skin their characters with transform feedback (as Pokémon Z-A, where this comes from: ZA Recomp's
// DLAA object motion): one program (rasterizer discard) writes model-space positions (12-byte stride) into an arena,
// at the same address for the same mesh every frame, and the characters' draws read them as vertex stream 0 (from the
// transform feedback mirror, xm_*) and place them with their constant buffers (camera and object). Each such draw in
// the scene's passes is kept (also on a shadow worker: mo_mu), with a second vertex set made around last frame's
// copies of its constant buffers (kept per arena address); after the camera's motion pass the draws are drawn again
// with a variant of their vertex shader that runs the original twice - with last frame's positions (a copy of the
// mirror's range taken at the end of last frame) and constant buffers, then with this frame's - and writes the
// difference into the motion image where the scene's depth shows the same surface. Only what is skinned this way:
// particles, effects and objects moved without bones keep the camera's motion.
// With DLAA only (FSR 2 reads the same motion image but wasn't tried with it); LARECOMP_OBJMV=0: off.
// LARECOMP_OBJMV_EPS: the depth match (relative, default 1e-4).
typedef struct {
    Prog *pr;
    Pipe *pp;
    uint32_t streams;
    SsState s;
    VkDescriptorSet sets[2];  // this frame's vertex set, last frame's
    uint32_t ndyn, dyn[4 * MAX_BIND];
    uint64_t off, size;       // stream 0's range in the mirror, bytes
    MvDrawCall dc;
} MvDraw;
static struct {
    MvDraw *d;
    int n, cap;
    uint64_t frame, seq;
    uint64_t lo, hi;          // the mirror range this frame's captured draws read (copied for the next frame)
    VkBuffer prev;            // last frame's range of the mirror, [prev_lo, prev_hi) at offset 0
    VmaAllocation prev_mem;
    uint64_t prev_cap, prev_lo, prev_hi;
    uint64_t prev_frame;      // the frame whose range is in prev
    uint32_t prev_moves;      // xm_moves when it was copied (the mirror's window moved since: its offsets are stale)
    int captured, no_prev, replayed, failed, outside;  // (the frame's counts, logged)
    _Atomic int built;
    // (the mirror ranges this frame's draws read - iv -, and last frame's, packed into prev - piv: each at base, sorted
    // by lo. Only those are copied: the span from the lowest to the highest grew as the player went from area to area,
    // the characters' data spread over the whole 256 MB window, and copying all of it each frame took a 4060 from 45 to
    // 35 fps, DLAA's own cost included)
    struct MoIv { uint64_t lo, hi, base; } *iv, *piv, *tiv;
    int niv, iv_cap, npiv, piv_cap, ntiv, tiv_cap;
    uint64_t copied;  // (the last copy's size, logged)
} mo;
typedef struct MoIv MoIv;
static int mo_iv_cmp(const void *a, const void *b) {
    uint64_t x = ((const MoIv *)a)->lo, y = ((const MoIv *)b)->lo;
    return x < y ? -1 : x > y;
}
static const MoIv *mo_iv_find(const MoIv *v, int n, uint64_t off, uint64_t size) {  // (the piece holding [off, off + size))
    int lo = 0, hi = n - 1, k = -1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        if (v[m].lo <= off) k = m, lo = m + 1; else hi = m - 1;
    }
    return k >= 0 && off + size <= v[k].hi ? &v[k] : NULL;
}
static pthread_mutex_t mo_mu = PTHREAD_MUTEX_INITIALIZER;

// last frame's constant buffers, per draw (arena address, program, the how-manyth draw of the pair: objmv_capture)
typedef struct { uint64_t key, size, frame[2]; uint32_t mask, len[2][18], cap[2][18]; uint8_t *data[2][18]; } MvCbuf;
static MvCbuf mo_cb[2048];

static bool objmv_on(void) {
    static _Atomic int on = -1;
    int v = atomic_load_explicit(&on, memory_order_relaxed);
    if (v < 0) {
        const char *e = ENV("LARECOMP_OBJMV");
        v = !e || atoi(e) != 0;
        atomic_store_explicit(&on, v, memory_order_relaxed);
    }
    return v != 0 && g_settings.taa == TAA_DLAA && vk_dlaa_available();
}

// An entry not drawn in the last 2 frames is taken for a new mesh (without that the table fills up after travelling
// to a few areas, and no new character gets its motion: found in Z-A)
static MvCbuf *mo_cbuf(uint64_t key, uint64_t f) {
    uint32_t i = (uint32_t)(key ^ (key >> 21)) & 2047;
    MvCbuf *stale = NULL;
    for (int k = 0; k < 2048; k++, i = (i + 1) & 2047) {
        MvCbuf *e = &mo_cb[i];
        if (e->key == key) return e;
        if (!e->key) { if (!stale) stale = e; break; }
        uint64_t last = e->frame[0] > e->frame[1] ? e->frame[0] : e->frame[1];
        if (!stale && last + 2 < f) stale = e;
    }
    if (!stale) return NULL;
    stale->key = key, stale->size = 0, stale->frame[0] = stale->frame[1] = 0;  // (its buffers are kept, for reuse)
    return stale;
}

static void objmv_capture(Prog *pr, Pipe *pp, uint32_t streams, const VkDescriptorSet sets[2], const uint32_t *dyn,
                          uint32_t nv_dyn, bool gmem, const MvDrawCall *dc) {
    if (gmem || !xm_buf || !objmv_on() || !fsr2_scene_pass()) return;
    if (!st.nstream || ss.vb[0] != xm_buf || !(ss.vb_known & 1) || st.stream[0][0] != 12 || st.stream[0][1]) return;
    if ((st.attrib[0] & 0xFF) != 0x22 || (st.attrib[0] >> 8 & 0x1F) != 0 || (st.attrib[0] >> 13) != 0) return;
    if (!pr->vi || !pr->vi->msl || !(pr->vi->attr_mask & 1) || pp->k.samples != 1) return;
    Range v0 = st.vbuf[0];
    if (!v0.addr || !v0.size) return;
    extern uint64_t vk_frame_counter(void);
    uint64_t f = vk_frame_counter();
    pthread_mutex_lock(&mo_mu);
    if (mo.frame != f) {
        mv_unpin(), mo.n = 0;  // (last frame's draws, if never drawn again)
        mo.frame = f, mo.seq = submit_seq;
        atomic_store_explicit(&mv_lost, false, memory_order_relaxed);
        mo.lo = ~0ull, mo.hi = 0, mo.niv = 0;
        mo.captured = mo.no_prev = 0;
    }
    if (atomic_load_explicit(&mv_lost, memory_order_relaxed)) goto out;
    uint64_t off = ss.vo[0];
    if (off < mo.lo) mo.lo = off;
    if (off + v0.size > mo.hi) mo.hi = off + v0.size;
    if (mo.niv == mo.iv_cap) {
        int cap = mo.iv_cap ? mo.iv_cap * 2 : 256;
        MoIv *v = realloc(mo.iv, (size_t)cap * sizeof *v);
        if (!v) goto out;
        mo.iv = v, mo.iv_cap = cap;
    }
    mo.iv[mo.niv++] = (MoIv){off, off + v0.size, 0};
    // this frame's constant buffers kept for the next; last frame's, if this draw was made then. One entry per draw: the
    // mesh (its arena address), the program and the how-manyth draw of that pair this frame. The same skinned mesh is
    // drawn more than once a frame - a depth pass and the G-buffer, or another program over it with constant buffers
    // of its own (a Tera raid boss's crystal coat: mask ca8 over 420) - and one entry per mesh gave the later draws the
    // first one's constant buffers for last frame: motion of tens of thousands of pixels over the whole boss, which
    // DLSS drew as moving black lines
    // (constant buffer k is the state's slot k - 3, as cbuf_info finds it; 0-2 are the driver's own: not the game's)
    uint32_t mask = pr->vi->cbuf_mask & 0x3FFF8u;
    int cur = (int)(f & 1), prv = !cur;
    MvCbuf *e = NULL;
    for (uint64_t k = 0; k < 8; k++) {
        uint64_t key = v0.addr ^ ((uint64_t)(uintptr_t)pr * 0x9E3779B97F4A7C15ull) ^ (k * 0xC2B2AE3D27D4EB4Full);
        MvCbuf *c = mo_cbuf(key ? key : 1, f);
        if (!c) break;
        if (c->frame[cur] != f) { e = c; break; }  // (the first of the pair's entries not taken yet this frame)
    }
    if (!e) {
        static int nl;
        if (nl++ < 5) LOG("[objmv] frame %llu: no room for another draw (2048 made in the last 2 frames, or 8 of one mesh and program)\n", (unsigned long long)f);
        goto out;
    }
    for (int i = 0; i < 18; i++) {
        if (!(mask >> i & 1)) continue;
        Range r = st.ubuf[0][i - 3];
        uint32_t len = r.addr && r.size ? (uint32_t)(r.size < 65536 ? r.size : 65536) : 0;
        if (len > e->cap[cur][i]) {
            uint8_t *nd = realloc(e->data[cur][i], len);
            if (!nd) goto out;
            e->data[cur][i] = nd, e->cap[cur][i] = len;
        }
        if (len) memcpy(e->data[cur][i], (const void *)(uintptr_t)r.addr, len);
        e->len[cur][i] = len;
    }
    e->frame[cur] = f;  // (only mark valid once every slot copied; a mid-loop OOM must not expose stale f-2 slots)
    mo.captured++;
    bool have_prev = e->frame[prv] + 1 == f && e->size == v0.size && mo.prev_frame + 1 == f && mo.prev_moves == xm_moves;
    e->size = v0.size, e->mask = mask;
    if (!have_prev) { mo.no_prev++; goto out; }
    pthread_mutex_lock(&ring_mu);
    if (!mv_pin) mv_pin = vk_pending_seq(), mo.seq = submit_seq;  // (its copies stay until objmv_replay)
    pthread_mutex_unlock(&ring_mu);
    // the vertex set again around last frame's copies
    Range keep[18];
    memcpy(keep, st.ubuf[0], sizeof keep);
    for (int i = 0; i < 18; i++) {
        if (!(mask >> i & 1) || !e->len[prv][i]) continue;
        st.ubuf[0][i - 3].addr = (uint64_t)(uintptr_t)e->data[prv][i], st.ubuf[0][i - 3].size = e->len[prv][i];
    }
    uint32_t pdyn[2 * MAX_BIND], pn = 0;
    VkDescriptorSet pset = stage_set(pr, pr->set_layout[0], 0, 0, pr->vi, NULL, NULL, 0, 0, 0, NULL, pdyn, &pn);
    memcpy(st.ubuf[0], keep, sizeof keep);
    if (!pset || pn != nv_dyn || nv_dyn + pn > 4 * MAX_BIND) goto out;
    if (mo.n == mo.cap) {
        int cap = mo.cap ? mo.cap * 2 : 512;
        MvDraw *d = realloc(mo.d, (size_t)cap * sizeof *d);
        if (!d) goto out;
        mo.d = d, mo.cap = cap;
    }
    MvDraw *d = &mo.d[mo.n++];
    d->pr = pr, d->pp = pp, d->streams = streams, d->s = ss;
    d->sets[0] = sets[0], d->sets[1] = pset;
    d->ndyn = nv_dyn + pn;
    memcpy(d->dyn, dyn, nv_dyn * sizeof *dyn);
    memcpy(d->dyn + nv_dyn, pdyn, pn * sizeof *pdyn);
    d->off = off, d->size = v0.size;
    d->dc = *dc;
out:
    pthread_mutex_unlock(&mo_mu);
}

// ---- the variant vertex shader (from the translator's GLSL)
static bool mo_ident(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }
typedef struct { char *s; size_t n, cap; } MoStr;
static void mo_put(MoStr *o, const char *p, size_t n) {
    if (o->n + n + 1 > o->cap) { o->cap = (o->n + n + 1) * 2; o->s = realloc(o->s, o->cap); }
    memcpy(o->s + o->n, p, n);
    o->n += n;
    o->s[o->n] = 0;
}
static void mo_puts(MoStr *o, const char *p) { mo_put(o, p, strlen(p)); }

static char *objmv_glsl(const char *src, uint32_t mask) {
    if (strstr(src, "push_constant") || strstr(src, "location = 30)") || strstr(src, "location = 31)")) return NULL;
    const char *m = strstr(src, "\nvoid main()");
    if (!m) return NULL;
    m++;
    MoStr o = {0};
    mo_put(&o, src, (size_t)(m - src));
    // last frame's constant buffers: the same blocks in set 1
    for (const char *l = src; l < m;) {
        const char *e = strchr(l, '\n');
        if (!e || e > m) e = m;
        int b, n, k;
        if (sscanf(l, "layout(std140, set = 0, binding = %d) uniform CB%d { uvec4 cb%*d[%d]; };", &b, &n, &k) == 3 && b == n &&
            b >= 0 && b < 18 && (mask >> b & 1)) {
            char line[160];
            snprintf(line, sizeof line, "layout(std140, set = 1, binding = %d) uniform PCB%d { uvec4 pcb%d[%d]; };\n", b, b, b, k);
            mo_puts(&o, line);
        }
        l = e + (e < m);
    }
    mo_puts(&o, "layout(std430, set = 2, binding = 0) readonly buffer MvPrevPos { float mvpp_[]; };\n"
                "layout(push_constant) uniform MvPc { uint off; float w, h, eps; } mvpc;\n"
                "layout(location = 30) out vec4 mv_prev_;\n"
                "layout(location = 31) out vec4 mv_cur_;\n"
                "vec4 pa0;\n");
    // this frame's: main as it was
    const char *body = m + strlen("void main()");
    mo_puts(&o, "void mv_cur_main()");
    mo_puts(&o, body);
    mo_puts(&o, "\n");
    // last frame's: a0 and the constant buffers renamed
    mo_puts(&o, "void mv_prev_main()");
    const char *p = body;
    bool first_brace = true;
    while (*p) {
        if (*p == '{' && first_brace) {
            first_brace = false;
            mo_puts(&o, "{\n  uint mvi_ = mvpc.off / 4u + uint(gl_VertexIndex) * 3u;\n"
                        "  pa0 = vec4(mvpp_[mvi_], mvpp_[mvi_ + 1u], mvpp_[mvi_ + 2u], 1.0);\n");
            p++;
            continue;
        }
        if (mo_ident(*p) && (p == body || !mo_ident(p[-1]))) {
            const char *q = p;
            while (mo_ident(*q)) q++;
            size_t n = (size_t)(q - p);
            int idx;
            char tail;
            if (n == 2 && p[0] == 'a' && p[1] == '0') mo_puts(&o, "pa0");
            else if (n >= 3 && n <= 4 && p[0] == 'c' && p[1] == 'b' && sscanf(p + 2, "%d%c", &idx, &tail) == 2 && idx >= 0 && idx < 18 &&
                     (mask >> idx & 1) && *q == '[') {
                mo_puts(&o, "p");
                mo_put(&o, p, n);
            } else mo_put(&o, p, n);
            p = q;
            continue;
        }
        mo_put(&o, p, 1);
        p++;
    }
    mo_puts(&o, "\nvoid main() {\n  mv_prev_main();\n  vec4 p_ = gl_Position;\n  mv_cur_main();\n"
                "  mv_prev_ = p_;\n  mv_cur_ = gl_Position;\n}\n");
    return o.s;
}

static const char *objmv_frag_src =
    "#version 450\n"
    "layout(location = 30) in vec4 mp;\n"
    "layout(location = 31) in vec4 mc;\n"
    "layout(set = 2, binding = 1) uniform sampler2D sdepth;\n"
    "layout(push_constant) uniform MvPc { uint off; float w, h, eps; } mvpc;\n"
    "layout(location = 0) out vec2 mv;\n"
    "void main() {\n"
    "  float z = texelFetch(sdepth, ivec2(gl_FragCoord.xy), 0).r;\n"
    "  if (abs(z - gl_FragCoord.z) > max(1e-7, abs(gl_FragCoord.z) * mvpc.eps)) discard;\n"
    "  if (mp.w <= 0.0 || mc.w <= 0.0) discard;\n"
    "  mv = (mp.xy / mp.w - mc.xy / mc.w) * 0.5 * vec2(mvpc.w, -mvpc.h);\n"  // (y: as the camera pass's, fsr2_mv.frag)
    "}\n";

static VkDescriptorSetLayout mo_set_layout;
static pthread_once_t mo_layout_once = PTHREAD_ONCE_INIT;
// The variants are built on threads of their own (a vertex shader: a few ms to compile; a pipeline: tens of ms):
// until a draw's is ready it keeps the camera's motion, as without these. state: 0 building, 1 ready, 2 failed.
typedef struct { Prog *pr; VkShaderModule vs; VkPipelineLayout layout; _Atomic int state; } MoProg;
static MoProg mo_progs[256];
typedef struct { Pipe *pp; MoProg *mp; VkPipeline pipe; _Atomic int state; } MoPipe;
static MoPipe mo_pipes[1024];
static int mo_nprogs, mo_npipes;
static VkShaderModule mo_fs;
static pthread_once_t mo_fs_once = PTHREAD_ONCE_INIT;

static void mo_layout_make(void) {
    VkDescriptorSetLayoutBinding bd[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, NULL},
                                          {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bd};
    VK_CHECK(vkCreateDescriptorSetLayout(g_vk.dev, &li, NULL, &mo_set_layout));
}
static void mo_layout_init(void) { pthread_once(&mo_layout_once, mo_layout_make); }

static void *mo_build_prog(void *arg) {
    MoProg *mp = arg;
    Prog *pr = mp->pr;
    char *src = objmv_glsl(pr->vi->msl, pr->vi->cbuf_mask & 0x3FFF8u);
    if (!src) {
        LOG("[objmv] vertex shader %llx: not one the variant can be made of\n", (unsigned long long)pr->code[0]);
        atomic_store(&mp->state, 2);
        return NULL;
    }
    size_t nw = 0;
    uint32_t *spv = vk_glsl_to_spirv(src, STAGE_VERTEX, "objmv vs", &nw);
    if (ENV("LARECOMP_OBJMV_DUMP")) {
        mkdir("build/glsl", 0755);
        char path[96];
        snprintf(path, sizeof path, "build/glsl/objmv_%llx.vert", (unsigned long long)pr->code[0]);
        FILE *fo = fopen(path, "w");
        if (fo) { fputs(src, fo); fclose(fo); }
    }
    free(src);
    if (!spv) {
        LOG("[objmv] vertex shader %llx: the variant didn't compile\n", (unsigned long long)pr->code[0]);
        atomic_store(&mp->state, 2);
        return NULL;
    }
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nw * 4, .pCode = spv};
    VkResult r = vkCreateShaderModule(g_vk.dev, &mi, NULL, &mp->vs);
    free(spv);
    VkDescriptorSetLayout sl[3] = {pr->set_layout[0], pr->set_layout[0], mo_set_layout};
    VkPushConstantRange pc = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 3, .pSetLayouts = sl,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    if (r != VK_SUCCESS || vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &mp->layout) != VK_SUCCESS) {
        LOG("[objmv] vertex shader %llx: no module or pipeline layout for the variant\n", (unsigned long long)pr->code[0]);
        atomic_store(&mp->state, 2);
        return NULL;
    }
    atomic_fetch_add(&mo.built, 1);
    atomic_store(&mp->state, 1);
    return NULL;
}

static void mo_fs_init(void) {
    size_t nw = 0;
    uint32_t *spv = vk_glsl_to_spirv(objmv_frag_src, STAGE_FRAGMENT, "objmv fs", &nw);
    if (!spv) return;
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nw * 4, .pCode = spv};
    if (vkCreateShaderModule(g_vk.dev, &mi, NULL, &mo_fs) != VK_SUCCESS) mo_fs = VK_NULL_HANDLE;
    free(spv);
}

static void *mo_build_pipe(void *arg) {
    MoPipe *e = arg;
    pthread_once(&mo_fs_once, mo_fs_init);
    PipeCI c;
    pipe_ci_fill(&c, &e->pp->k);
    if (!mo_fs || c.pi.stageCount != 2) { atomic_store(&e->state, 2); return NULL; }
    c.stages[0].module = e->mp->vs, c.stages[0].pName = "main", c.stages[0].pSpecializationInfo = NULL;
    c.stages[1].module = mo_fs, c.stages[1].pName = "main", c.stages[1].pSpecializationInfo = NULL;
    c.stages[1].pNext = NULL, c.stages[1].flags = 0;  // (no required subgroup size: not the game's fragment shader)
    static const VkFormat rg = VK_FORMAT_R16G16_SFLOAT;
    c.ri.colorAttachmentCount = 1, c.ri.pColorAttachmentFormats = &rg;
    c.ri.depthAttachmentFormat = VK_FORMAT_UNDEFINED, c.ri.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;
    c.ds.depthTestEnable = c.ds.depthWriteEnable = c.ds.stencilTestEnable = c.ds.depthBoundsTestEnable = VK_FALSE;
    c.cba[0] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT};
    c.cb.attachmentCount = 1, c.cb.logicOpEnable = VK_FALSE;
    c.ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT, c.ms.alphaToCoverageEnable = VK_FALSE;
    c.pi.layout = e->mp->layout;
    uint64_t t0 = SDL_GetTicksNS();
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &c.pi, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
    static _Atomic int nlog;
    if (atomic_fetch_add(&nlog, 1) < 12)
        LOG("[objmv] pipeline %s in %.1f ms (in the background)\n", pipe ? "built" : "FAILED", (SDL_GetTicksNS() - t0) / 1e6);
    e->pipe = pipe;
    atomic_store(&e->state, pipe ? 1 : 2);
    return NULL;
}

static void mo_spawn(void *(*fn)(void *), void *arg) {
    pthread_t th;
    if (!pthread_create(&th, NULL, fn, arg)) pthread_detach(th);
    else fn(arg);
}

// (render thread: objmv_replay)
static MoProg *objmv_prog(Prog *pr) {
    for (int i = 0; i < mo_nprogs; i++) if (mo_progs[i].pr == pr) return &mo_progs[i];
    if (mo_nprogs == 256) return NULL;
    mo_layout_init();
    MoProg *mp = &mo_progs[mo_nprogs++];
    mp->pr = pr, mp->vs = VK_NULL_HANDLE, mp->layout = VK_NULL_HANDLE;
    atomic_store(&mp->state, 0);
    mo_spawn(mo_build_prog, mp);
    return mp;
}

// VK_NULL_HANDLE until it is built (it is asked for then)
static VkPipeline objmv_pipe(MoProg *mp, Pipe *pp) {
    if (atomic_load(&mp->state) != 1 || atomic_load(&pp->state) != 1) return VK_NULL_HANDLE;
    for (int i = 0; i < mo_npipes; i++)
        if (mo_pipes[i].pp == pp && mo_pipes[i].mp == mp) return atomic_load(&mo_pipes[i].state) == 1 ? mo_pipes[i].pipe : VK_NULL_HANDLE;
    if (mo_npipes == 1024) return VK_NULL_HANDLE;
    MoPipe *e = &mo_pipes[mo_npipes++];
    e->pp = pp, e->mp = mp, e->pipe = VK_NULL_HANDLE;
    atomic_store(&e->state, 0);
    mo_spawn(mo_build_pipe, e);
    return VK_NULL_HANDLE;
}

// After the camera's motion pass (fsr2_copy, outside any pass): the kept draws into mvb (this frame's motion, render
// size w x h) where the scene depth (zv) shows them; then this frame's mirror range copied for the next frame
static void objmv_replay(TexBackend *mvb, VkImageView zv, uint32_t w, uint32_t h) {
    extern uint64_t vk_frame_counter(void);
    uint64_t f = vk_frame_counter();
    if (!objmv_on() || !xm_buf) return;
    pthread_mutex_lock(&mo_mu);
    VkCommandBuffer cb = vk_cmd();
    int n = mo.frame == f ? mo.n : 0;
    mo.n = 0;
    mo.replayed = mo.failed = mo.outside = 0;
    // this frame's ranges, merged (gaps under 64 KB too: fewer copies) and packed (tiv); the copy of them must fit: a
    // larger buffer made now (last frame's is lost then: no motion drawn)
    uint64_t need = 0;
    mo.ntiv = 0;
    if (mo.frame == f && mo.niv && mo.hi <= XM_SIZE) {
        qsort(mo.iv, (size_t)mo.niv, sizeof *mo.iv, mo_iv_cmp);
        if (mo.tiv_cap < mo.niv) {
            MoIv *v = realloc(mo.tiv, (size_t)mo.niv * sizeof *v);
            if (v) mo.tiv = v, mo.tiv_cap = mo.niv;
        }
        if (mo.tiv_cap >= mo.niv)
            for (int i = 0; i < mo.niv; i++) {
                MoIv a = mo.iv[i];
                a.lo &= ~15ull;  // (each piece keeps its 16-byte alignment in the copy)
                if (mo.ntiv && a.lo <= mo.tiv[mo.ntiv - 1].hi + (64u << 10)) {
                    if (a.hi > mo.tiv[mo.ntiv - 1].hi) mo.tiv[mo.ntiv - 1].hi = a.hi;
                } else mo.tiv[mo.ntiv++] = a;
            }
        for (int i = 0; i < mo.ntiv; i++) {
            mo.tiv[i].base = need;
            need += (mo.tiv[i].hi - mo.tiv[i].lo + 255) & ~255ull;
        }
    }
    if (need > mo.prev_cap) {
        if (mo.prev) { vk_wait_seq(submit_seq); vmaDestroyBuffer(g_vk.vma, mo.prev, mo.prev_mem); }
        mo.prev = VK_NULL_HANDLE, mo.prev_cap = 0, mo.prev_frame = 0, n = 0;
        uint64_t cap = (need + (16ull << 20) - 1) & ~((16ull << 20) - 1);
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = cap,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &mo.prev, &mo.prev_mem, NULL) == VK_SUCCESS) mo.prev_cap = cap;
        else mo.prev = VK_NULL_HANDLE;
        LOG("[objmv] last frame's copy of the mirror: %llu MB\n", (unsigned long long)(mo.prev_cap >> 20));
    }
    if (n && submit_seq - mo.seq < SUBS - 4 && mo.prev && mo.prev_frame + 1 == f) {
        static float eps = -1;
        if (eps < 0) eps = ENV("LARECOMP_OBJMV_EPS") ? (float)atof(ENV("LARECOMP_OBJMV_EPS")) : 1e-4f;
        mo_layout_init();
        VkDescriptorSet set2 = vk_alloc_set(mo_set_layout);
        if (set2) {
            VkDescriptorBufferInfo bi = {mo.prev, 0, VK_WHOLE_SIZE};
            VkDescriptorImageInfo ii = {vk_default_sampler(false), zv, VK_IMAGE_LAYOUT_GENERAL};
            VkWriteDescriptorSet wr[2] = {
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set2, .dstBinding = 0, .descriptorCount = 1,
                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi},
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set2, .dstBinding = 1, .descriptorCount = 1,
                 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii}};
            rec_UpdateDescriptorSets(2, wr);
            vk_full_barrier(cb);
            VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = post_view(mvb),
                                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                            .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
            VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                                  .colorAttachmentCount = 1, .pColorAttachments = &ca};
            rec_CmdBeginRendering(cb, &ri);
            for (int k = 0; k < n; k++) {
                MvDraw *d = &mo.d[k];
                const MoIv *pv = mo_iv_find(mo.piv, mo.npiv, d->off, d->size);
                if (!pv) { mo.outside++; continue; }
                MoProg *mp = objmv_prog(d->pr);
                VkPipeline pipe = mp && atomic_load(&mp->state) == 1 ? objmv_pipe(mp, d->pp) : VK_NULL_HANDLE;
                if (!pipe) { mo.failed++; continue; }
                const SsState *q = &d->s;
                rec_CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                if (g_vk.vids) set_vertex_input(cb, &q->vi);
                rec_CmdSetDepthTestEnable(cb, VK_FALSE);
                rec_CmdSetDepthWriteEnable(cb, VK_FALSE);
                rec_CmdSetDepthCompareOp(cb, q->dop);
                rec_CmdSetStencilTestEnable(cb, VK_FALSE);
                for (int fc = 0; fc < 2; fc++) {
                    VkStencilFaceFlags face = fc ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
                    const uint32_t *v = q->sface[fc];
                    rec_CmdSetStencilOp(cb, face, (VkStencilOp)v[0], (VkStencilOp)v[1], (VkStencilOp)v[2], (VkCompareOp)v[3]);
                    rec_CmdSetStencilCompareMask(cb, face, v[4]);
                    rec_CmdSetStencilWriteMask(cb, face, v[5]);
                    rec_CmdSetStencilReference(cb, face, v[6]);
                }
                rec_CmdSetBlendConstants(cb, q->bc);
                rec_CmdSetCullMode(cb, q->cull);
                rec_CmdSetFrontFace(cb, q->ff);
                rec_CmdSetPrimitiveTopology(cb, q->topo);
                rec_CmdSetPrimitiveRestartEnable(cb, q->pre);
                rec_CmdSetDepthBiasEnable(cb, q->dbe);
                rec_CmdSetDepthBias(cb, q->db[1], q->db[2], q->db[0]);
                rec_CmdSetViewport(cb, 0, 1, &q->vp);
                rec_CmdSetScissor(cb, 0, 1, &q->sc);
                VkDescriptorSet sets[3] = {d->sets[0], d->sets[1], set2};
                rec_CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, mp->layout, 0, 3, sets, d->ndyn, d->ndyn ? d->dyn : NULL);
                struct { uint32_t off; float w, h, eps; } pc = {(uint32_t)(pv->base + (d->off - pv->lo)), (float)w, (float)h, eps};
                rec_CmdPushConstants(cb, mp->layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
                if (d->dc.kind == 1 || (d->dc.kind == 2 && d->dc.indexed)) rec_CmdBindIndexBuffer(cb, q->ib, q->io, q->it);
                for (int i = 0; i < 17; i++)
                    if (d->streams & (1u << i)) rec_CmdBindVertexBuffers(cb, (uint32_t)i, 1, &q->vb[i], &q->vo[i]);
                if (d->dc.kind == 2) rec_CmdDrawIndirect(cb, d->dc.buf, d->dc.off, 1, d->dc.a, d->dc.indexed);
                else if (d->dc.kind == 1) rec_CmdDrawIndexed(cb, d->dc.a, d->dc.b, 0, d->dc.d, d->dc.e);
                else rec_CmdDraw(cb, d->dc.a, d->dc.b, d->dc.c, d->dc.e);
                mo.replayed++;
            }
            rec_CmdEndRendering(cb);
            vk_full_barrier(cb);
            ss_forget();
        }
    }
    // this frame's range, for the next frame's motion (after the draws above read last frame's)
    if (mo.prev && need && need <= mo.prev_cap) {
        static VkBufferCopy *rg;
        static int rg_cap;
        if (rg_cap < mo.ntiv) {
            VkBufferCopy *r = realloc(rg, (size_t)mo.ntiv * sizeof *r);
            if (r) rg = r, rg_cap = mo.ntiv;
        }
        if (rg_cap >= mo.ntiv) {
            for (int i = 0; i < mo.ntiv; i++) rg[i] = (VkBufferCopy){mo.tiv[i].lo, mo.tiv[i].base, mo.tiv[i].hi - mo.tiv[i].lo};
            vk_full_barrier(cb);
            rec_CmdCopyBuffer(cb, xm_buf, mo.prev, (uint32_t)mo.ntiv, rg);
            vk_full_barrier(cb);
            mo.prev_frame = f, mo.prev_moves = xm_moves, mo.prev_lo = mo.lo, mo.prev_hi = mo.hi;
            MoIv *t = mo.piv; int tc = mo.piv_cap;  // (this frame's pieces are last frame's for the next)
            mo.piv = mo.tiv, mo.piv_cap = mo.tiv_cap, mo.npiv = mo.ntiv;
            mo.tiv = t, mo.tiv_cap = tc, mo.ntiv = 0;
            mo.copied = need;
        }
    }
    pthread_mutex_unlock(&mo_mu);
    mv_unpin();  // (the copies kept for the draws: until this submission is done)
    static int nlog;
    static uint64_t last_log;
    uint64_t now = SDL_GetTicksNS();
    if (nlog < 3 || now - last_log > 30000000000ull) {
        nlog++, last_log = now;
        LOG("[objmv] frame %llu: %d character draws kept (%d without last frame's), %d drawn again, %d without a variant, "
            "%d outside last frame's copy; mirror %llx-%llx, %.1f MB of it copied in %d pieces; %d variants built\n", (unsigned long long)f, mo.captured,
            mo.no_prev, mo.replayed, mo.failed, mo.outside, (unsigned long long)mo.lo, (unsigned long long)mo.hi,
            mo.copied / 1048576.0, mo.npiv, atomic_load(&mo.built));
    }
}

// (check) LARECOMP_OBJMV_CHECK=<frame>: that frame's and the next one's picture (the scene drawn, before the temporal
// anti-aliasing) and motion image, raw, to build/objmv_chk<0|1>_{color,mv,mvcam}.bin (tools: warp the first by the
// second's motion and compare with the second)
static void objmv_check(TexBackend *sb, TexBackend *mvb, uint32_t w, uint32_t h, bool cam) {
    static long at = -2;
    if (at == -2) at = ENV("LARECOMP_OBJMV_CHECK") ? atol(ENV("LARECOMP_OBJMV_CHECK")) : -1;
    extern uint64_t vk_frame_counter(void);
    uint64_t f = vk_frame_counter();
    if (at < 0 || (f != (uint64_t)at && f != (uint64_t)at + 1)) return;
    int k = (int)(f - (uint64_t)at);
    TexBackend *img[2] = {sb, mvb};
    const char *nm[2] = {"color", cam ? "mvcam" : "mv"};  // (cam: the camera's motion alone, before objmv_replay)
    for (int i = cam ? 1 : 0; i < 2; i++) {
        uint32_t bpp = i ? 4 : (img[i]->format == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4);
        VkSpan rb = vk_readback((uint64_t)w * h * bpp);
        VkBufferImageCopy c = {.bufferOffset = rb.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
        VkCommandBuffer cb = vk_cmd();
        vk_full_barrier(cb);
        rec_CmdCopyImageToBuffer(cb, img[i]->image, VK_IMAGE_LAYOUT_GENERAL, rb.buf, 1, &c);
        renderer_flush();
        mkdir("build", 0755);
        char path[96];
        snprintf(path, sizeof path, "build/objmv_chk%d_%s.bin", k, nm[i]);
        FILE *fo = fopen(path, "wb");
        if (fo) { fwrite(rb.ptr, 1, (size_t)w * h * bpp, fo); fclose(fo); }
    }
    LOG("[objmv] check frame %llu (%d): %ux%u, picture format %d, jitter %.4f %.4f\n", (unsigned long long)f, k, w, h,
        (int)sb->format, fsr2_jx, fsr2_jy);
}
