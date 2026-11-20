// Game heaps (engine limits). The game carves its memory into SDK heaps of fixed sizes: nn::mem::StandardAllocator
// (general), nn::lmem unit heaps (fixed-size blocks: object pools) and frame heaps. A heap that runs out returns
// NULL and the game crashes further on (bushsanity: ~450 actors around Link -> null weak_ptr in f_70010d0f00).
// These wrappers call the SDK's own implementation (loader_find_symbol), log heap creation (LARECOMP_LOG_HEAPS=1)
// and every allocation that fails (once per caller), and can grow heaps (heap_scale) - a recomp has memory to spare.
#include <pthread.h>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"

uint64_t loader_find_symbol(const char *name);
void guest_call(Cpu *c, uint64_t addr);

static uint64_t orig(const char *name, uint64_t *cache) {
    if (!*cache) *cache = loader_find_symbol(name);
    if (!*cache) FATAL("heaps: no SDK implementation of %s\n", name);
    return *cache;
}

static void fail_once(const char *what, uint64_t lr, uint64_t size) {
    static uint64_t seen[64];
    static int n;
    for (int i = 0; i < n; i++) if (seen[i] == lr) return;
    if (n < 64) seen[n++] = lr;
    LOG("[heaps] %s FAILED (size %#llx) from lr %llx\n", what, (unsigned long long)size, (unsigned long long)lr);
}

#define CALL(sym)                                    \
    do {                                             \
        static uint64_t a_;                          \
        uint64_t lr_ = c->x[30];                     \
        guest_call(c, orig(sym, &a_));               \
        c->x[30] = lr_;                              \
    } while (0)

// Unit heaps are fixed pools (e.g. 256 x 0x50-byte objects per loaded area: bushsanity's extra actors ran out at
// ~450 actors around Link). Every unit heap gets UNIT_SCALE times the units in a buffer of ours (the game's own
// buffer stays unused); freed again in DestroyUnitHeap. LARECOMP_UNIT_HEAP_SCALE overrides (1 = the game's sizes).
typedef struct { uint64_t head; void *mem; } UnitBuf;
static UnitBuf g_unit[256];
static pthread_mutex_t g_unit_lock = PTHREAD_MUTEX_INITIALIZER;

// Reboot: our unit-heap buffers (host memory, not in the guest window) go with the game that made them.
RT_ON_RESET(heaps_reset) {
    for (int i = 0; i < 256; i++)
        if (g_unit[i].mem) {
#ifdef _WIN32
            _aligned_free(g_unit[i].mem);
#else
            free(g_unit[i].mem);
#endif
            g_unit[i].mem = NULL;
        }
}

static int unit_scale(void) {
    static int s = -1;
    if (s < 0) { const char *e = ENV("LARECOMP_UNIT_HEAP_SCALE"); s = e ? atoi(e) : 4; if (s < 1) s = 1; }
    return s;
}

// nn::lmem::CreateUnitHeap(void *addr, size_t size, size_t unit, int align, int opt, HeapHead *head)
HLE_FUNC("_ZN2nn4lmem14CreateUnitHeapEPvmmiiPNS0_6detail8HeapHeadE", heap_create_unit) {
    uint64_t lr = c->x[30], size = c->x[1], unit = c->x[2];
    int k = unit_scale();
    void *mem = NULL;
    if (k > 1 && unit) {
        uint64_t big = size * (uint64_t)k;
#ifdef _WIN32
        mem = _aligned_malloc(big, 0x1000);
        if (mem) {
#else
        if (posix_memalign(&mem, 0x1000, big) == 0) {
#endif
            memset(mem, 0, big);
            c->x[0] = (uint64_t)(uintptr_t)mem;
            c->x[1] = big;
        } else {
            mem = NULL;
        }
    }
    uint64_t head = c->x[5];
    CALL("_ZN2nn4lmem14CreateUnitHeapEPvmmiiPNS0_6detail8HeapHeadE");
    if (mem) {
        pthread_mutex_lock(&g_unit_lock);
        for (int i = 0; i < 256; i++)
            if (!g_unit[i].mem) { g_unit[i].head = c->x[0] ? c->x[0] : head; g_unit[i].mem = mem; break; }
        pthread_mutex_unlock(&g_unit_lock);
    }
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] unit heap %#llx bytes, unit %#llx (%llu units, x%d) from lr %llx -> %llx\n", (unsigned long long)size,
            (unsigned long long)unit, unit ? (unsigned long long)(size / unit) : 0ull, mem ? k : 1, (unsigned long long)lr,
            (unsigned long long)c->x[0]);
}

HLE_FUNC("_ZN2nn4lmem15DestroyUnitHeapEPNS0_6detail8HeapHeadE", heap_destroy_unit) {
    uint64_t head = c->x[0];
    CALL("_ZN2nn4lmem15DestroyUnitHeapEPNS0_6detail8HeapHeadE");
    pthread_mutex_lock(&g_unit_lock);
    for (int i = 0; i < 256; i++)
        if (g_unit[i].mem && g_unit[i].head == head) {
#ifdef _WIN32
            _aligned_free(g_unit[i].mem);
#else
            free(g_unit[i].mem);
#endif
            g_unit[i].mem = NULL;
            break;
        }
    pthread_mutex_unlock(&g_unit_lock);
}

HLE_FUNC("_ZN2nn4lmem20AllocateFromUnitHeapEPNS0_6detail8HeapHeadE", heap_alloc_unit) {
    uint64_t lr = c->x[30], head = c->x[0];
    CALL("_ZN2nn4lmem20AllocateFromUnitHeapEPNS0_6detail8HeapHeadE");
    if (!c->x[0]) fail_once("unit heap allocation", lr, head);
}

// nn::lmem::CreateFrameHeap(void *addr, size_t size, int opt)
HLE_FUNC("_ZN2nn4lmem15CreateFrameHeapEPvmi", heap_create_frame) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn4lmem15CreateFrameHeapEPvmi");
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] frame heap %#llx bytes from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn4lmem21AllocateFromFrameHeapEPNS0_6detail8HeapHeadEmi", heap_alloc_frame) {
    uint64_t lr = c->x[30], size = c->x[1];
    CALL("_ZN2nn4lmem21AllocateFromFrameHeapEPNS0_6detail8HeapHeadEmi");
    if (!c->x[0]) fail_once("frame heap allocation", lr, size);
}

// nn::mem::StandardAllocator::Initialize(void *addr, size_t size[, bool])
HLE_FUNC("_ZN2nn3mem17StandardAllocator10InitializeEPvm", heap_std_init) {
    uint64_t lr = c->x[30], size = c->x[2];
    CALL("_ZN2nn3mem17StandardAllocator10InitializeEPvm");
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] standard allocator %#llx bytes from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator10InitializeEPvmb", heap_std_init_b) {
    uint64_t lr = c->x[30], size = c->x[2];
    CALL("_ZN2nn3mem17StandardAllocator10InitializeEPvmb");
    if (ENV("LARECOMP_LOG_HEAPS"))
        LOG("[heaps] standard allocator %#llx bytes (b) from lr %llx\n", (unsigned long long)size, (unsigned long long)lr);
}

// Scarlet / Violet's small-object allocator, bigger with World detail. At boot (0x700167d510) the game gives its
// big allocator (main+0x4740f28: 12 GB of address space, pages mapped as used) all the free memory, then takes 20% of
// what that could hold for a second one (main+0x4740ee0, 746 MB here) that serves every allocation of 64 KB or less
// with 16-byte alignment - the scene objects, their components, the streaming states. Its size is the block asked of
// the big one (lr 0x700167d5a4), kept at main+0x4740f70 and read back for the Initialize right after (with the
// block's ends at +0x98 / +0xa0). World detail streams the field farther, so it holds more of them at once: over a
// fast travel (Cascarrafa -> Colonnade Hollow) its least free went 247 MB at 1x, 175 at 2x, 79 at 3x, and at 4x it ran
// out - an allocation returned NULL and the game crashed further on (f_7000c28198, a null object). So 120 MB more per
// level past 1x (with kernel.c extra_memory's 864 MB the game's is 817 MB, so 1177 at 4x: that fast travel's least free
// 420 MB); the memory comes from the big allocator's, which extra_memory grows to match. Sized at boot: a change of
// World detail applies from the next start.
#define SV_BIG_ALLOC   0x4740f28ull  // main + : the big allocator
#define SV_SMALL_SIZE  0x4740f70ull  // main + : the small one's size
#define SV_SMALL_LR    0x167d5a4ull  // main + : where its block is asked for
static uint64_t sv_small_size(Cpu *c, uint64_t lr, uint64_t size) {
    extern int rt_sv_world_detail_boot(void);
    uint64_t base = g_modules[0].base;
    if (!base || lr != base + SV_SMALL_LR || c->x[0] != base + SV_BIG_ALLOC || size < (256ull << 20)) return size;
    int n = rt_sv_world_detail_boot();
    if (n <= 1) return size;
    uint64_t to = size + (uint64_t)(n > 4 ? 3 : n - 1) * (120ull << 20);
    LOG("[heaps] small-object allocator %llu MB (the game's %llu): World detail x%d\n", (unsigned long long)(to >> 20),
        (unsigned long long)(size >> 20), n);
    return to;
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator8AllocateEm", heap_std_alloc) {
    uint64_t lr = c->x[30], size = c->x[1], to = sv_small_size(c, lr, size);
    c->x[1] = to;
    CALL("_ZN2nn3mem17StandardAllocator8AllocateEm");
    if (c->x[0] && to != size) *(volatile uint64_t *)(uintptr_t)(g_modules[0].base + SV_SMALL_SIZE) = to;
    if (!c->x[0] && size) fail_once("standard allocation", lr, size);
}

// ...and its 350 MB range pool, bigger with World detail. At boot the game makes three range pools (0x7001587a5c:
// 350 / 268 / 50 MB, sizes from its memory settings; object vtable main+0x44dacb0, made by 0x7001587e24: the object
// (0x258 bytes) first, then its memory block from the big allocator, 4 KB aligned, through the game's aligned malloc
// (lr 0x7000b40f10), then one free range {offset 0, size} - +0x128 the pool's total, +0x148 its size, +0x108 the root
// of its free ranges by size (node +0x08 / +0x10 children, +0x20 offset, +0x28 size). Allocation 0x7000c28198: a
// range that fits, or NULL - and its caller then reads through the NULL. At 1x the 350 MB one settles around 160 MB
// used; at 4x, the field streamed farther, around 330 (21 MB free after a couple of fast travels, 0 on the sixth):
// the third fast travel in a row crashed the game. So its block is asked for 100 MB bigger per level past 1x, and the
// pool is grown to it as soon as it's made - only while it is exactly as made (one free range, offset 0, the game's
// size); otherwise it stays the game's 350 MB (the extra memory unused) and that is logged. 650 MB at 4x: 246-340 MB
// free over 27 fast travels (Colonnade Hollow / Asado Desert, Cascarrafa / Cortondo), steady, never short.
#define SV_POOL_SIZE   0x15e00000ull  // the game's size of the pool
#define SV_POOL_VTABLE 0x44dacb0ull   // main + : a range pool's vtable
#define SV_MALLOC_LR   0xb40f10ull    // main + : the game's aligned malloc, after its StandardAllocator call
static uint64_t sv_pool_obj, sv_pool_mem, sv_pool_to;  // (the last 0x258-byte block; the pool's block, its new size)
static atomic_int sv_pool_state;                       // 0 nothing yet, 1 block given (to grow), 2 done
static void sv_pool_grow(void) {
    if (atomic_load(&sv_pool_state) != 1) return;
    uint64_t o = sv_pool_obj;
    volatile uint64_t *f = (volatile uint64_t *)(uintptr_t)o;
    if (!o || f[0] != g_modules[0].base + SV_POOL_VTABLE || f[0x138 / 8] != sv_pool_mem) return;  // (not made yet)
    uint64_t root = f[0x108 / 8];
    if (f[0x148 / 8] != SV_POOL_SIZE || f[0x128 / 8] != SV_POOL_SIZE || root < 0x7000000000ull || root >= 0x8000000000ull)
        return;  // (still being made)
    int one = 1;
    if (!atomic_compare_exchange_strong(&sv_pool_state, &one, 2)) return;
    volatile uint64_t *n = (volatile uint64_t *)(uintptr_t)root;
    if (n[0x20 / 8] != 0 || n[0x28 / 8] != SV_POOL_SIZE || n[1] || n[2]) {
        LOG("[heaps] range pool %llx in use before it could grow: the game's %llu MB\n", (unsigned long long)o,
            (unsigned long long)(SV_POOL_SIZE >> 20));
        return;
    }
    n[0x28 / 8] = sv_pool_to;
    f[0x128 / 8] = sv_pool_to;
    f[0x148 / 8] = sv_pool_to;
    LOG("[heaps] range pool %llu MB (the game's %llu): World detail\n", (unsigned long long)(sv_pool_to >> 20),
        (unsigned long long)(SV_POOL_SIZE >> 20));
}
RT_ON_RESET(sv_pool_reset) {  // (a reboot makes the pools again)
    sv_pool_obj = sv_pool_mem = sv_pool_to = 0;
    atomic_store(&sv_pool_state, 0);
}

HLE_FUNC("_ZN2nn3mem17StandardAllocator8AllocateEmm", heap_std_alloc_a) {
    extern int rt_sv_world_detail_boot(void);
    uint64_t lr = c->x[30], size = c->x[1], to = size, alloc = c->x[0], align = c->x[2];
    bool malloc_lr = g_modules[0].base && lr == g_modules[0].base + SV_MALLOC_LR;
    if (atomic_load(&sv_pool_state) == 1) sv_pool_grow();
    if (malloc_lr && size == SV_POOL_SIZE && c->x[2] == 0x1000 && atomic_load(&sv_pool_state) == 0 && sv_pool_obj) {
        int n = rt_sv_world_detail_boot();
        if (n > 1) to = size + (uint64_t)(n > 4 ? 3 : n - 1) * (100ull << 20);
        c->x[1] = to;
    }
    CALL("_ZN2nn3mem17StandardAllocator8AllocateEmm");
    if (to != size) {
        if (c->x[0]) sv_pool_mem = c->x[0], sv_pool_to = to, atomic_store(&sv_pool_state, 1);
        else {  // (no room: the game's size)
            c->x[0] = alloc, c->x[1] = size, c->x[2] = align;
            CALL("_ZN2nn3mem17StandardAllocator8AllocateEmm");
        }
    }
    if (malloc_lr && size == 0x258 && c->x[0] && atomic_load(&sv_pool_state) == 0) sv_pool_obj = c->x[0];
    if (!c->x[0] && size) fail_once("standard allocation (aligned)", lr, size);
}
