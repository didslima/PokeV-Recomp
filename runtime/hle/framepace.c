// Frame pacing: the game's speed doesn't depend on the frame rate.
// The game advances its world each frame by its frame period (16.67 ms) times the swap interval its frame timer
// (f_700000c910) reads from the system object - normally 1. When a frame takes longer than a vsync (GPU-bound at
// 32:9, a hitch), the next frame starts a vsync later but still advanced only 16.67 ms: slow motion (30 fps = half
// speed). tools/patches.py FRAME_STEPS_SITE routes that read here: we return how many vsyncs really passed since
// the last frame, so the world advances by the time that passed. The game's own waiting is unaffected (it reads
// the interval nowhere else). Shared by the macOS and Windows builds.
// FINDING (2026-10-09): forcing 2 steps did NOT change Link's walking speed - gameplay is a true fixed step per
// frame; this timer feeds something else. Kept for research, off by default. (2026-10-18: the read is the game's
// dynamic resolution's frame budget - tools/patches.py GAME_DYNRES_SITE, now off, so it no longer runs at all.)
//   LARECOMP_FRAMEPACE=1     hand the timer the vsyncs elapsed (default: the game's own value)
//   LARECOMP_FRAME_STEPS_MAX maximum steps per frame (default 4: after a long stall the game doesn't jump ahead)
//   LARECOMP_FORCE_STEPS=<n> (test) always n
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../hle.h"

extern volatile uint64_t g_vsync_count;  // runtime/hle/vi.c
uint64_t g_frame_steps_late;             // (stats) frames that advanced more than one step

uint32_t rt_frame_steps(uint32_t interval) {
    static int on = -1, max_steps, forced;
    if (on < 0) {
        const char *e = ENV("LARECOMP_FRAMEPACE");
        on = e ? atoi(e) != 0 : 0;  // off by default: the world (Link's movement) doesn't use this step (tested)
        e = ENV("LARECOMP_FRAME_STEPS_MAX");
        max_steps = e ? atoi(e) : 4;
        if (max_steps < 1) max_steps = 1;
        e = ENV("LARECOMP_FORCE_STEPS");
        forced = e ? atoi(e) : 0;
    }
    static int ncall;
    if (ENV("LARECOMP_LOG_FRAMEPACE") && ncall++ % 120 == 0) LOG("[framepace] call %d interval %u vsync %llu\n", ncall, interval, (unsigned long long)g_vsync_count);
    if (forced > 0) return (uint32_t)forced;
    static uint64_t last;
    uint64_t now = __atomic_load_n(&g_vsync_count, __ATOMIC_RELAXED);
    uint64_t steps = last ? now - last : 1;
    last = now;
    if (!on) return interval;
    if (steps < interval) steps = interval;  // (the game's own 30 fps mode: it waits that many vsyncs anyway)
    if (steps < 1) steps = 1;
    if (steps > (uint64_t)max_steps) steps = (uint64_t)max_steps;
    if (steps > 1) g_frame_steps_late++;
    return (uint32_t)steps;
}

// Pokémon Z-A's frame mode (tools/patches.py ZA_BUILDS): its configured vsyncs per frame - 1 (1: 30 fps). With
// the fps60 setting (or LARECOMP_FPS=60) 0: it presents every vsync and steps its world 1/60 s a frame. A frame
// that takes longer still steps 1/60 s: below 60 fps the whole game moves in slow motion (a player's log: 35 fps
// = 58% speed; the CPU-side game code couldn't keep up, an RTX 4060 waiting half the time). So the setting backs off
// by itself: presented below FPS60_MIN for FPS60_SLOW_S seconds in a row (after FPS60_WARM_S to settle - shaders
// building right after the switch), the game goes back to 30 fps for the rest of the session and the overlay says so
// (menu_notice). Choosing 60 again in F1 tries again. LARECOMP_FPS=30: as is; LARECOMP_FPS=60 or F1's "60 fps (always)"
// (fps60_lock): 60, no backing off - for players who'd rather have some slow motion than 30 fps.
// Seconds spent building shaders count neither way (#22: an RX 9070 XT with a cold pipeline cache had three 1-2 s
// shader stalls in a battle and went back to 30 for the session, holding 60 otherwise).
#include "../settings.h"
#include "../platform.h"
#define FPS60_MIN 50
#define FPS60_SLOW_S 5
#define FPS60_WARM_S 5
bool g_za_frame_hook;    // (the game is Pokémon Z-A: exec.c leaves its GPU timing counters alone)
bool g_fps60_fell_back;  // (60 fps chosen, but this session went back to 30: runtime/ui/menu.mm says so)
static bool fps60_holds(void) {
    extern uint64_t renderer_present_count(void);
    extern void menu_notice(const char *text, int seconds);
    static int last_setting = -1, warm, slow;
    static uint64_t t0, p0;
    int setting = g_settings.fps60 | g_settings.fps60_lock << 1;
    if (last_setting != setting) {  // (chosen again in F1: a new try)
        last_setting = setting;
        g_fps60_fell_back = false, warm = slow = 0, t0 = 0;
    }
    if (!g_settings.fps60 || g_fps60_fell_back) return false;
    if (g_settings.fps60_lock) return true;
    uint64_t now = plat_now_ns(), pc = renderer_present_count();
    if (!t0) { t0 = now, p0 = pc; return true; }
    if (now - t0 < 1000000000ull) return true;
    double fps = (double)(pc - p0) * 1e9 / (double)(now - t0);
    t0 = now, p0 = pc;
    if (warm < FPS60_WARM_S) { warm++; return true; }
    static double min = -1;  // LARECOMP_FPS60_MIN: another floor (tests: 59 makes a busy scene back off)
    if (min < 0) min = ENV("LARECOMP_FPS60_MIN") ? atof(ENV("LARECOMP_FPS60_MIN")) : FPS60_MIN;
    extern uint64_t renderer_shader_builds(void);  // (the renderer: builds done, top bit = one being built now)
    static uint64_t b0;
    uint64_t b = renderer_shader_builds();
    bool building = (b >> 63) || (b & ~(1ull << 63)) - (b0 & ~(1ull << 63)) >= 2;  // (one now, or a few this second)
    b0 = b;
    if (!building) slow = fps < min ? slow + 1 : 0;
    if (slow < FPS60_SLOW_S) return true;
    g_fps60_fell_back = true;
    LOG("[pace] 60 fps not held (%.0f fps for %d s): back to 30 fps - below 60 the game moves in slow motion\n", fps, slow);
    menu_notice("60 fps turned off: this PC didn't keep 60 fps, and below that the game moves in slow motion. "
                "Back to 30 fps (choose 60 again in F1 to retry).", 8);
    return false;
}
int32_t rt_za_frame_cfg(int32_t cfg) {
    g_za_frame_hook = true;
    const char *e = ENV("LARECOMP_FPS");
    bool on = e ? atoi(e) >= 60 : fps60_holds();
    static int logged = -1;
    if (logged != (int)on) { logged = on; LOG("[pace] game frame mode %d -> %d (%s)\n", cfg, on ? 0 : cfg, on ? "60 fps" : "as is"); }
    return on ? 0 : cfg;
}

// Pokémon Legends: Arceus and Scarlet / Violet run in their own 30 fps mode, forced to 60 by tools/pla_patches.py /
// sv_patches.py, and step by the real frame time below 60 (rt_pla_frame_dt_ns; Scarlet / Violet's clock slots too,
// sv_patches.py _sv_frame_clock): 45-50 fps plays at normal speed, so no backing off to 30 here (v0.0.21 had Z-A's,
// fps60_holds, while the slots still stepped 1/60 s - Arceus never needed it).
bool rt_pla_is_60fps(void) {
    g_za_frame_hook = true;
    const char *e = ENV("LARECOMP_FPS");
    if (e) return atoi(e) >= 60;
    return g_settings.fps60;
}
// Scarlet / Violet: the real frame time also goes into the game's clock slots 0 and 1 (sv_patches.py _sv_frame_clock).
// LARECOMP_SV_SLOT_DT=0: not (the master clock only, as before v0.0.22 - slow motion below 60 fps).
bool rt_sv_slot_dt(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = ENV("LARECOMP_SV_SLOT_DT");
        on = !e || atoi(e) != 0;
        LOG("[pace] Scarlet / Violet clock slots follow the real frame time: %s\n", on ? "on" : "off (LARECOMP_SV_SLOT_DT=0)");
    }
    return on;
}

static uint64_t g_frame_dt_ns;  // (rt_pla_frame_dt_ns: the frame time handed out this frame, smoothed; 0 before the first)
bool rt_sv_slot_dt(void);
// Pokémon Legends: Arceus & Pokémon Scarlet & Violet dynamic frame delta-time:
// Supplies the real wall-clock frame duration in nanoseconds (clamped to [16.67 ms, 50.0 ms] when 60 FPS is on),
// so the game runs at 1.0x normal speed at 60 FPS (no 2x speed-up) AND stays at 1.0x speed if FPS dips below 60.
static volatile uint32_t g_game_frame;  // (game frames begun: rt_sv_lua_frame_time writes once per frame)
uint64_t rt_pla_frame_dt_ns(uint32_t slot, uint64_t default_ns) {
    if (slot == 0) {  // (once a game frame, on the game's thread: Scarlet / Violet's npc_shown.c, npc_detail.c)
        g_game_frame++;
        extern void rt_sv_npc_shown(void);
        rt_sv_npc_shown();
        extern void rt_sv_npc_detail(void);
        rt_sv_npc_detail();
    }
    bool fps60 = rt_pla_is_60fps();
    uint64_t target_ns = fps60 ? 0x00fe502bull : (default_ns ? default_ns : 0x01fca056ull);
    static uint64_t last_ns;
    uint64_t cur_dt_ns = g_frame_dt_ns;
    static int logged_mode = -1;
    if (logged_mode != (int)fps60) {
        logged_mode = (int)fps60;
        LOG("[pace] game frame clock mode: %s (target dt = %.3f ms, dynamic delta-time %s)\n",
            fps60 ? "60 FPS" : "30 FPS", (double)target_ns / 1e6, fps60 ? "enabled" : "fixed");
    }
    if (!fps60) {
        last_ns = 0;
        g_frame_dt_ns = target_ns;
        return target_ns;
    }
    if (slot == 0 || !cur_dt_ns) {
        uint64_t now = plat_now_ns();
        uint64_t real_ns = last_ns ? (now - last_ns) : target_ns;
        last_ns = now;
        const char *dyn = ENV("LARECOMP_DYNAMIC_DT");
        if (dyn && atoi(dyn) == 0) {
            cur_dt_ns = target_ns;
        } else if (real_ns > 100000000ull) {
            // Ignore long loading / shader-compile stalls (> 100 ms) so the player doesn't jump ahead
            cur_dt_ns = target_ns;
        } else {
            if (real_ns < target_ns) real_ns = target_ns;
            if (real_ns > 50000000ull) real_ns = 50000000ull;  // floor at 20 FPS (50 ms) max step
            cur_dt_ns = cur_dt_ns ? (cur_dt_ns * 3 + real_ns) / 4 : real_ns;
        }
    }
    g_frame_dt_ns = cur_dt_ns;
    return cur_dt_ns;
}

// Scarlet / Violet's SetSlotMode (sv_patches.py _sv_slot_mode): the 60 fps mode's duration (16.67 ms, table entry 0)
// becomes the frame time measured this frame by the frame clock patch (rt_pla_frame_dt_ns) - every clock slot steps
// by real time; other modes' durations stay as the table has them. LARECOMP_SV_SLOT_DT=0: the table's value.
//
// The table's 30 fps entry (1) is also read by the game's step factor, slot 1 / (table[mode] - 1) (0x7000b83d1c,
// 0x7000d96428), with mode the game's own frame mode - still 1 (30 fps): _sv_frame_pace forces 60 fps in a register,
// not in the game's copies. The factor is how many 30 fps frames a frame lasted, and the UI's animations (and
// the Pokédex's book going onto the shelf on a new registration) advance by it. _sv_frame_pace sets entry 1 to
// 16.67 ms in the 60 fps mode, so the factor was 1 at 60 fps: every UI animation played at double speed (the
// map opened in 0.6 s instead of 1.2), and the book's insertion was cut short (its sequence counts in time, its
// motion in frames). So entry 1 goes back to 33.3 ms here, once that patch has written it this frame
// (SetSlotMode runs right after it, 0x7002b7a8f4): the factor is 0.5 at 60 fps, as the game's 30 fps timing
// expects. A slot set to mode 1 gets the frame time measured, as one set to mode 0 does (the same as when entry
// 1 was 16.67 ms). LARECOMP_SV_UI_STEP=0: entry 1 left at 16.67 ms (as before: UI animations at double speed).
static bool sv_ui_step(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = ENV("LARECOMP_SV_UI_STEP");
        on = !e || atoi(e) != 0;
        LOG("[pace] Scarlet / Violet UI animations step by the 30 fps frame: %s\n", on ? "on" : "off (LARECOMP_SV_UI_STEP=0)");
    }
    return on;
}
uint64_t rt_sv_slot_duration(uint64_t table_ns) {
    if (!g_frame_dt_ns || !rt_pla_is_60fps()) return table_ns;
    bool ui = sv_ui_step();
    if (ui) *(volatile uint64_t *)(uintptr_t)0x7003c2fbb0ull = 0x01fca056ull;  // (the table's entry 1: 33.3 ms)
    if (!rt_sv_slot_dt()) return table_ns == 0x01fca056ull && ui ? 0x00fe502bull : table_ns;  // (nominal 1/60 s)
    if (table_ns != 0x00fe502bull && !(ui && table_ns == 0x01fca056ull)) return table_ns;
    return g_frame_dt_ns;
}


// Scarlet / Violet's scripts (Lua 5.4) step by a frame time of their own, C075A638F130352C3.S44460EB93267F798 in their
// globals: 1/30, set when they load and never changed. At 60 fps whatever they integrate with it ran at twice the
// speed in real time: the ride's jump took off at 8.51 m/s and lost gravity * 1/30 every frame, so it rose half as
// high and landed in half the time (sv_patches.py _sv_lua_frame_time). Here, from lua_pushnumber (natives hand the
// scripts numbers many times a frame, on the thread running them - their tables are not changing under us), that
// global gets this frame's time once a frame: the frame time measured (rt_pla_frame_dt_ns) at 60 fps, 1/30 at 30.
// Their own fixed-step loops keep working with it (one counts 1/30 s steps out of the time it is handed: a step every
// other frame at 60). LARECOMP_SV_LUA_DT=0: left as the scripts set it.
static const uint8_t *lua_find(uint64_t t, const char *name) {  // (Lua 5.4: t.name, a short string key, own hash part)
    size_t len = strlen(name);
    uint8_t lsizenode = *(const uint8_t *)(uintptr_t)(t + 0xb);
    uint64_t node = *(const uint64_t *)(uintptr_t)(t + 0x18);  // (Node: value 0, value tag 8, key tag 9, key 16; 24 bytes)
    for (uint64_t i = 0; i < (1ull << lsizenode); i++) {
        const uint8_t *n = (const uint8_t *)(uintptr_t)(node + 24 * i);
        if (n[9] != 0x44) continue;  // (a short string, collectable)
        uint64_t ts = *(const uint64_t *)(n + 16);  // (TString: shrlen at 0xb, contents at 0x18)
        if (*(const uint8_t *)(uintptr_t)(ts + 0xb) == len && !memcmp((const void *)(uintptr_t)(ts + 0x18), name, len)) return n;
    }
    return NULL;
}
void rt_sv_lua_frame_time(uint64_t L) {
    static int on = -1;
    if (on < 0) {
        const char *e = ENV("LARECOMP_SV_LUA_DT");
        on = !e || atoi(e) != 0;
        LOG("[pace] Scarlet / Violet scripts step by the frame time: %s\n", on ? "on" : "off (LARECOMP_SV_LUA_DT=0)");
    }
    if (!on) return;
    static struct { uint64_t g; uint32_t frame; } seen[4];  // (per global state: each has its own globals)
    uint64_t g = *(const uint64_t *)(uintptr_t)(L + 0x18);  // (lua_State: l_G)
    uint32_t frame = g_game_frame;
    int k = 0;
    while (k < 4 && seen[k].g && seen[k].g != g) k++;
    if (k == 4) return;
    if (seen[k].g == g && seen[k].frame == frame) return;
    seen[k].g = g, seen[k].frame = frame;
    // (global_State: l_registry at 0x40, a table; its array part's [2] is the globals. Looked up every frame - a
    // table may be resized, never a pointer kept)
    const uint8_t *reg = (const uint8_t *)(uintptr_t)(g + 0x40);
    if ((reg[8] & 0x0f) != 5) return;
    uint64_t r = *(const uint64_t *)reg;
    if (*(const uint32_t *)(uintptr_t)(r + 0xc) < 2) return;  // (alimit)
    const uint8_t *gl = (const uint8_t *)(uintptr_t)(*(const uint64_t *)(uintptr_t)(r + 0x10) + 16);
    if ((gl[8] & 0x0f) != 5) return;
    const uint8_t *mod = lua_find(*(const uint64_t *)gl, "C075A638F130352C3");
    if (!mod || (mod[8] & 0x0f) != 5) return;
    uint8_t *dt = (uint8_t *)lua_find(*(const uint64_t *)mod, "S44460EB93267F798");
    if (!dt || dt[8] != 0x13) return;  // (a float)
    double v = rt_pla_is_60fps() && g_frame_dt_ns ? (double)g_frame_dt_ns / 1e9 : 1.0 / 30;
    static int logged;
    if (!logged) {
        logged = 1;
        LOG("[pace] the scripts' frame time: %.6f s, now %.6f s\n", *(const double *)dt, v);
    }
    memcpy(dt, &v, sizeof v);
}

// Scarlet / Violet's scripts run some objects in fixed 1/30 s steps (lines 278989-279004): the object's time (its
// [21]) gets clock slot 2's duration - this frame's time - and its update gets the whole 1/30 s steps in it. The indoor
// camera's update (lines 279349-279422) approaches its target by f = its [41] (0.4) a step:
// CD9AE7C27B00EB066.SF667EA15F51B1676, cur += (target - cur) * f, once per step. At 60 fps a step came every other
// frame: the camera moved at 30 Hz behind a character moving at 60, and indoors the character shook on screen (at 30
// fps it doesn't). From the native that hands the loop slot 2 (sv_patches.py _sv_lua_clock_slot), before the loop adds
// it: that camera's time is set for one step now, and f to what this frame's time makes of a 1/30 s step's,
// 1 - (1 - f)^(dt * 30) - the same approach in time at any frame rate, 30 fps's own at 30. The game's f is kept per
// camera (taken again when the game sets another). 60 fps only; LARECOMP_SV_LUA_CAMERA=0: as before.
static uint8_t *lua_index(uint64_t t, long i) {  // (Lua 5.4: t[i], array part, else the hash part)
    if (!t) return NULL;
    uint32_t alimit = *(const uint32_t *)(uintptr_t)(t + 0xc);
    if (i >= 1 && (uint64_t)i <= alimit) {
        uint64_t arr = *(const uint64_t *)(uintptr_t)(t + 0x10);
        return arr ? (uint8_t *)(uintptr_t)(arr + 16ull * (uint64_t)(i - 1)) : NULL;
    }
    uint8_t lsizenode = *(const uint8_t *)(uintptr_t)(t + 0xb);
    uint64_t node = *(const uint64_t *)(uintptr_t)(t + 0x18);
    if (!node || lsizenode > 24) return NULL;
    for (uint64_t k = 0; k < (1ull << lsizenode); k++) {
        uint8_t *n = (uint8_t *)(uintptr_t)(node + 24 * k);
        if (n[9] == 0x03 && *(const int64_t *)(n + 16) == i) return n;  // (an integer key)
    }
    return NULL;
}
static const uint8_t *lua_method(uint64_t t, const char *name) {  // (t.name, through the metatables' __index tables)
    for (int depth = 0; t && depth < 8; depth++) {
        const uint8_t *v = lua_find(t, name);
        if (v) return v;
        uint64_t mt = *(const uint64_t *)(uintptr_t)(t + 0x28);
        const uint8_t *ix = mt ? lua_find(mt, "__index") : NULL;
        if (!ix || (ix[8] & 0x0f) != 5) return NULL;
        t = *(const uint64_t *)ix;
    }
    return NULL;
}
static struct { uint64_t t; double f30, set; } g_sv_cams[8];
static int g_sv_ncams;
static pthread_mutex_t g_sv_cam_lock = PTHREAD_MUTEX_INITIALIZER;
RT_ON_RESET(sv_cam_reset) {
    pthread_mutex_lock(&g_sv_cam_lock);
    memset(g_sv_cams, 0, sizeof g_sv_cams);
    g_sv_ncams = 0;
    pthread_mutex_unlock(&g_sv_cam_lock);
}
void rt_sv_lua_clock_slot(uint64_t L, uint64_t slot, uint64_t s) {
    static int on = -1;
    pthread_mutex_lock(&g_sv_cam_lock);
    if (on < 0) {
        const char *e = ENV("LARECOMP_SV_LUA_CAMERA");
        on = !e || atoi(e) != 0;
        LOG("[pace] Scarlet / Violet indoor camera in steps of the frame: %s\n", on ? "on" : "off (LARECOMP_SV_LUA_CAMERA=0)");
    }
    if (!on || slot != 2 || !L || !s) {
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    uint64_t ci = *(const uint64_t *)(uintptr_t)(L + 0x20);  // (lua_State: ci - the native's; previous: its Lua caller)
    uint64_t caller = ci ? *(const uint64_t *)(uintptr_t)(ci + 0x10) : 0;
    uint64_t base = caller ? *(const uint64_t *)(uintptr_t)caller : 0;
    if (!base) {
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    const uint8_t *self = (const uint8_t *)(uintptr_t)(base + 16);  // (its R0)
    if ((self[8] & 0x0f) != 5) {
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    uint64_t t = *(const uint64_t *)self;
    const uint8_t *m = t ? lua_method(t, "FC0150FC2959FFA71") : NULL;  // (the update the loop calls)
    if (!m || (m[8] & 0x3f) != 6) {  // (a Lua closure)
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    uint64_t cl = *(const uint64_t *)m;
    uint64_t p = cl ? *(const uint64_t *)(uintptr_t)(cl + 0x18) : 0;  // (its Proto)
    if (!p || *(const int *)(uintptr_t)(p + 0x2c) != 279349 || *(const int *)(uintptr_t)(p + 0x18) != 243) {
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    uint8_t *acc = lua_index(t, 21), *fv = lua_index(t, 41);
    if (!acc || !fv || acc[8] != 0x13 || fv[8] != 0x13) {  // (floats)
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    int i = 0;
    while (i < g_sv_ncams && g_sv_cams[i].t != t) i++;
    if (i == g_sv_ncams) {
        if (g_sv_ncams == 8) i = 0; else g_sv_ncams++;
        g_sv_cams[i].t = t;
        g_sv_cams[i].set = -1;
    }
    double cur = 0;
    memcpy(&cur, fv, 8);
    if (cur != g_sv_cams[i].set) g_sv_cams[i].f30 = cur;  // (the game's own: new, or set again)
    if (!rt_pla_is_60fps()) {  // (30 fps: the game's own again)
        if (g_sv_cams[i].set >= 0) {
            memcpy(fv, &g_sv_cams[i].f30, 8);
            g_sv_cams[i].set = -1;
        }
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    double dur = (double)*(const uint64_t *)(uintptr_t)(s + 8) / 1e9;
    if (!isfinite(dur) || !(dur > 0 && dur < 1) || !isfinite(g_sv_cams[i].f30) || !(g_sv_cams[i].f30 > 0 && g_sv_cams[i].f30 < 1)) {
        pthread_mutex_unlock(&g_sv_cam_lock);
        return;
    }
    double f = 1.0 - pow(1.0 - g_sv_cams[i].f30, dur * 30.0), time = 1.0 / 30 - dur + 1e-9;  // (one step: time + dur >= 1/30)
    g_sv_cams[i].set = f;
    memcpy(fv, &f, 8);
    memcpy(acc, &time, 8);
    pthread_mutex_unlock(&g_sv_cam_lock);
}

// Scarlet / Violet's sequence players (sv_patches.py _sv_timeline_time) - the catch's camera work, cutscenes, UI
// animations - count in 60ths of a second, and each track is a clip from one whole frame to another (start <= t < end).
// At 30 fps a sample steps 2 and lands on even frames only; at 60 fps it steps 1 and lands anywhere (frame times vary,
// so a player's samples drift off whole frames). A clip one frame long is a moment: on an odd frame it is never seen at
// 30 fps, on an even one it is seen once. At 60 fps the odd ones were seen too - on throwing a Poké Ball, the camera cut
// has one: its frame's camera was the sum of two cameras (~3 km away, looking up: a frame of sky, then of a lake far
// below), and the terrain streaming and the grass, which follow the camera, dropped what was near the real one - the
// ground near the catch missing for a few frames and its tall grass for ~0.6 s. A sample inside such a clip, between
// whole frames, plays it as 30 fps would: an even one at its own frame, once; an odd one, or one played already, at
// its end. The time gained or lost is given back on the player's next sample (a hold never goes behind where it was
// played), so the sequence keeps its length and its sound. (Its end every time skipped the even ones too: a Tera raid
// catch lost its camera's keys, stayed low on the crystal rock and then turned up to the cavern's wall, where 30 fps
// looks up the whole time.) The tracks (manager = player + 0xe0: [+0x50, +0x58) and [+0x18, +0x20), 0x1d0 bytes each)
// point at FlatBuffers tables whose fields at vtable offsets 8 and 0xa are the clip's first frame and its end (as
// 0x7000b845c0 reads them). LARECOMP_SV_TIMELINE_SNAP=0: as before; LARECOMP_LOG_TIMELINE_SNAP=1 logs each one.
static uint32_t fb_u32(uint64_t tbl, uint32_t voff, uint32_t def) {
    if (tbl < 0x10000) return def;
    int32_t soff = 0;
    memcpy(&soff, (const void *)(uintptr_t)tbl, sizeof soff);
    if (soff == 0) return def;
    uint64_t vt = tbl - (int64_t)soff;
    if (vt < 0x10000) return def;
    uint16_t vsize = 0;
    memcpy(&vsize, (const void *)(uintptr_t)vt, sizeof vsize);
    if (vsize < 4 || voff + 2 > vsize) return def;
    uint16_t o = 0;
    memcpy(&o, (const void *)(uintptr_t)(vt + voff), sizeof o);
    if (o < 4) return def;
    uint32_t val = 0;
    memcpy(&val, (const void *)(uintptr_t)(tbl + o), sizeof val);
    return val;
}
static int tl_moment(uint64_t player, float t, float *start, float *end) {  // (the one-frame clip t is in, between whole frames)
    if (!isfinite(t) || t < 0.0f || t == floorf(t) || player < 0x10000) return 0;
    uint64_t mgr = player + 0xe0;
    static const uint32_t lists[2][2] = {{0x50, 0x58}, {0x18, 0x20}};
    for (int l = 0; l < 2; l++) {
        uint64_t b = 0, e = 0;
        memcpy(&b, (const void *)(uintptr_t)(mgr + lists[l][0]), sizeof b);
        memcpy(&e, (const void *)(uintptr_t)(mgr + lists[l][1]), sizeof e);
        if (b < 0x10000 || e < b || (e - b) / 0x1d0 > 4096) continue;
        for (uint64_t tr = b; tr + 0x1d0 <= e; tr += 0x1d0) {
            uint64_t tbl = 0;
            memcpy(&tbl, (const void *)(uintptr_t)tr, sizeof tbl);
            if (tbl < 0x10000) continue;
            uint32_t s0 = fb_u32(tbl, 8, 0), e0 = fb_u32(tbl, 0xa, 0);
            if (e0 == s0 + 1 && (float)s0 <= t && t < (float)e0) { *start = (float)s0, *end = (float)e0; return 1; }
        }
    }
    return 0;
}
enum { SV_TL_SLOTS = 32 };
// (per player - a few run at once, several threads may step them: the time it was last played at, and the time it
// played ahead of its own, < 0 behind)
static struct { uint64_t player; float last, owed; } g_tl_pl[SV_TL_SLOTS];
static int g_tl_next;
static pthread_mutex_t g_tl_lock = PTHREAD_MUTEX_INITIALIZER;
RT_ON_RESET(sv_timeline_reset) {
    pthread_mutex_lock(&g_tl_lock);
    memset(g_tl_pl, 0, sizeof g_tl_pl);
    g_tl_next = 0;
    pthread_mutex_unlock(&g_tl_lock);
}
float rt_sv_timeline_time(uint64_t player, float t, float step) {
    static int on = -1, log = 0;
    pthread_mutex_lock(&g_tl_lock);
    if (on < 0) {
        const char *e = ENV("LARECOMP_SV_TIMELINE_SNAP");
        on = !e || atoi(e) != 0;
        log = ENV("LARECOMP_LOG_TIMELINE_SNAP") != NULL;
        LOG("[pace] Scarlet / Violet one-frame sequence clips played as at 30 fps: %s\n", on ? "on" : "off (LARECOMP_SV_TIMELINE_SNAP=0)");
    }
    if (!on || !rt_pla_is_60fps() || !isfinite(t) || t < 0.0f) {
        pthread_mutex_unlock(&g_tl_lock);
        return t;
    }
    int slot = -1;
    for (int i = 0; i < SV_TL_SLOTS; i++) if (g_tl_pl[i].player == player) { slot = i; break; }
    float last = -1.0f;  // (not played: no clip counts as played)
    if (slot < 0) slot = g_tl_next, g_tl_next = (g_tl_next + 1) % SV_TL_SLOTS;
    else if (t >= g_tl_pl[slot].last - 0.5f && t <= g_tl_pl[slot].last + 4.0f) {
        // (the next step of the same sequence - else restarted, or the player's address reused: nothing kept)
        last = g_tl_pl[slot].last;
        float back = t - g_tl_pl[slot].owed;
        t = g_tl_pl[slot].owed > 0 && back < last ? last : back;
    }
    float played = t, s0, e0;
    if (tl_moment(player, t, &s0, &e0)) {
        if (((uint32_t)s0 & 1) == 0 && last < s0) {  // (on 30 fps's frames, not played yet: at its own frame, once)
            played = s0;
            if (log && t - s0 >= 0.02f) LOG("[pace] sequence player %llx: %.3f inside a one-frame clip, played at its frame %.0f\n", (unsigned long long)player, t, s0);
        } else {  // (between 30 fps's frames - never seen there - or played already: at its end)
            played = e0;
            if (log) LOG("[pace] sequence player %llx: %.3f inside a one-frame clip at %.0f, played at %.0f\n", (unsigned long long)player, t, s0, e0);
        }
    }
    g_tl_pl[slot].player = player, g_tl_pl[slot].last = played, g_tl_pl[slot].owed = played - t;
    t = played;
    pthread_mutex_unlock(&g_tl_lock);
    (void)step;
    return t;
}
