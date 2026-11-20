// Scarlet / Violet: how far town people and Pokémon keep their detailed models (F1 > People detail / Pokémon
// detail, settings npc_detail / pokemon_detail).
//
// Every person has three meshes (<name>.trmsh, _lod1, _lod2: chara/model_*), and the game swaps to the simpler ones
// by distance to the camera: the full model only within 8 m, then 15 m, then 30 m - walking up to a bench, its
// person changes model twice. Pokémon (wild and in towns) do the same at 10 / 20 / 30 m. Those distances are
// trinity_ModelLodSetting "chara_model" and "pokemon" (the field's own: 30-1500 m), loaded with the field into the
// model LOD manager. The LOD pick (f_7000b87474) compares the distance with factor[q][i] (manager + 0x90, all 1.0
// here) times the matching setting's distance i, and when the result differs from the model's cached thresholds it
// takes the new ones: a distance changed in the setting applies to every model on screen on its next pick.
//
// The manager: *(*(*(*(main+0x46d4e00)) + 0x38) + 0x50) (vtable main+0x44a00b8); its settings by name, +0x30: a
// hash map whose node list starts at +0x38 - node +0 next, +0x18 the name's chars, +0x20 its length, +0x58 three
// float distances. Each game frame (the frame clock patch, framepace.c rt_pla_frame_dt_ns) the two nodes' distances
// are set to the game's times their setting, and given back at "Game's"; a node the game wrote again (a new field
// loaded) is the game's own again. Found with a hardware read watch on the factor table (every LOD pick reads it),
// its callers' registers pointing at the settings.
//
// Pokémon Violet 4.0.0 only (runtime/game_identity.h); the manager is checked by its vtable and the node by its name
// before anything is written. LARECOMP_NPC_DETAIL=<n> / LARECOMP_POKEMON_DETAIL=<n> override the settings (1: the
// game's).
#include "../rt.h"
#include "../settings.h"
#include <stdlib.h>
#include <string.h>

#define LOD_GLOBAL     0x46d4e00ull  // main + : -> -> the scene system; +0x38 -> +0x50: the model LOD manager
#define LOD_MGR_VTABLE 0x44a00b8ull  // main + : its vtable
#define LOD_MGR_FACTOR 0x90          // float[4][3]: the factors (quality level x step)
#define LOD_MGR_BYNAME 0x30          // the settings by name (a map)
#define MAP_FIRST      0x38          // its first node
#define NODE_NAME      0x18          // (chars; +0x20 the length)
#define NODE_DIST      0x58          // float[3]

static bool gptr(uint64_t p) { return p >= 0x7000000000ull && p < 0x8000000000ull && !(p & 7); }
#define G64(a) (*(volatile uint64_t *)(uintptr_t)(a))

// One named setting's distances times mul: the game's kept (a node the game wrote again is its own again), logged.
typedef struct { const char *name, *what; uint64_t node; float game[3], wrote[3], logged; } LodScale;

static void scale(uint64_t map, LodScale *t, float mul) {
    size_t len = strlen(t->name);
    uint64_t n = G64(map + MAP_FIRST), found = 0;
    for (int i = 0; i < 64 && gptr(n); i++, n = G64(n)) {
        uint64_t s = G64(n + NODE_NAME);
        if (G64(n + NODE_NAME + 8) == len && s >= 0x7000000000ull && s < 0x7fffffff00ull &&
            !memcmp((const void *)(uintptr_t)s, t->name, len)) {
            found = n;
            break;
        }
    }
    if (!found) return;
    volatile float *d = (volatile float *)(uintptr_t)(found + NODE_DIST);
    if (found != t->node || d[0] != t->wrote[0] || d[1] != t->wrote[1] || d[2] != t->wrote[2]) {  // (the game's own)
        if (!(d[0] > 0 && d[0] < d[1] && d[1] < d[2] && d[2] < 10000)) { t->node = 0; return; }  // (not distances)
        t->node = found;
        for (int i = 0; i < 3; i++) t->game[i] = t->wrote[i] = d[i];
    }
    for (int i = 0; i < 3; i++) {
        float to = t->game[i] * mul;
        if (d[i] != to) d[i] = t->wrote[i] = to;
    }
    if (t->logged != mul) {
        t->logged = mul;
        LOG("[npc] %s detail distances: %g / %g / %g m before World detail's factor (the game's: %g / %g / %g)\n", t->what, t->game[0] * mul,
            t->game[1] * mul, t->game[2] * mul, t->game[0], t->game[1], t->game[2]);
    }
}

static int mul_of(const char *env_name, int *env, int setting) {
    if (*env == -2) { const char *e = getenv(env_name); *env = e ? atoi(e) : -1; }
    int mul = *env >= 0 ? *env : setting;
    return mul < 1 ? 1 : mul > 16 ? 16 : mul;
}

// World detail (F1, settings world_detail): every model's LOD distances times n - the LOD manager's factor table
// (manager + 0x90: 4 quality levels x 3 steps, 1.0 here), which the LOD pick multiplies into every setting's
// distances (the field's 20-1500 m, buildings', rocks'...). The people's and Pokémon's own settings are divided by it,
// so that People detail / Pokémon detail stay what they say. Trees, bushes and rocks drawn in batches (instanced)
// pick their LODs on the GPU instead: exec.c translate_compute (rt_sv_world_detail) divides that pass's distance.
int rt_sv_world_detail(void) {
    static int env = -2;
    return mul_of("LARECOMP_WORLD_DETAIL", &env, g_settings.world_detail);
}
// The level the game's memory was sized for: World detail as it was at start (kernel.c extra_memory, hle/heaps.c's
// small-object allocator and range pool, all set once at boot). Streaming farther than that would run them out - a
// fast travel at 4x on the game's sizes crashed - so a higher level chosen in game applies from the next start.
int rt_sv_world_detail_boot(void) {
    static int boot;
    if (!boot) boot = rt_sv_world_detail();
    return boot;
}
// World detail's factor for one of its parts, at most the boot level - LARECOMP_WORLD_<part>=0 leaves that one at the
// game's (to bisect a visual problem): LOD (models' LOD distances), INSTANCED (trees, bushes, rocks: exec.c), TERRAIN
// (the detailed land's streaming distance), STANDINS (the far land stand-ins and the groups streaming their pieces in:
// clod_thread).
int rt_sv_world_part(const char *part) {
    char name[48];
    snprintf(name, sizeof name, "LARECOMP_WORLD_%s", part);
    const char *e = getenv(name);
    int n = rt_sv_world_detail(), boot = rt_sv_world_detail_boot();
    return e && atoi(e) == 0 ? 1 : n < boot ? n : boot;
}

// ...and the detailed land's distance. trinity_TerrainStreamingSetting (a resident scene, loaded at boot: 150 / 350 /
// 3300 / 150 / 800) is copied into an object the game's global main+0x4763b60 points at (vtable main+0x44a71a0,
// the five floats from +0x58); its fifth, 800 m, is how far the land's detailed pieces are streamed in - past it the
// hills and cliffs are their far stand-ins. Times n (3x: about 20% more draws at the Mesagoza lookout, 60 fps; the
// other four change nothing seen). Written from the first frames, before the field streams in; "Game's" puts 800
// back (for what streams in from then on).
#define TERRAIN_GLOBAL 0x4763b60ull
#define TERRAIN_VTABLE 0x44a71a0ull
#define TERRAIN_DIST   0x68
static void terrain_distance(int world) {
    static uint64_t obj;
    static float game, wrote;
    Module *m = &g_modules[0];
    if (TERRAIN_GLOBAL + 8 > m->data_off + m->data_size) return;
    uint64_t o = G64(m->base + TERRAIN_GLOBAL);
    if (!gptr(o) || G64(o) != m->base + TERRAIN_VTABLE) return;
    volatile float *d = (volatile float *)(uintptr_t)(o + TERRAIN_DIST);
    if (o != obj || *d != wrote) {  // (a new object, or the game wrote it again: the game's)
        if (!(*d > 10 && *d < 100000)) { obj = 0; return; }
        obj = o;
        game = wrote = *d;
    }
    float to = game * (float)world;
    if (*d != to) {
        *d = wrote = to;
        LOG("[npc] world detail: detailed land streamed within %g m (the game's %g)\n", to, game);
    }
}

// ...and the land's merged far stand-ins (trinity_CombineLODEntity: a hill's or cliff's pieces drawn as one simple
// model from 240-500 m on, by entity; vtable main+0x44a2ed8, {d^2, (d - 20)^2} at +0x808: back to the stand-in past d,
// to its pieces within d - 20 - read whenever its state is taken: changed, it shows at once). Times n, together with
// what brings their pieces that far:
// - the field's grid streaming groups (trinity_GridStreamingGroup, vtable main+0x44a4790, {load, unload} at +0x130,
//   read every frame: _s 220 m, _m 660, _l 1870, _xl 3135, _xxl 6110; not people / events / sounds, 95-120 m) load
//   the terrain tiles' scenes (ter_XX_YY_streaming_m...) that hold the pieces, and
// - each scene object of a group keeps itself shown only within its group's own object streaming settings (an
//   object's +0x6f8 points at them: {550, 605, 660, 715} for _m, steps of 55 m, then a 1 - read every frame). Past
//   them a piece hides although its tile is in: with only the stand-in pushed out it hid over nothing - a cliff south
//   of Mesagoza at 640 m left the bare ground under it showing in stripes. An object also copies the settings' last
//   value (715 m) to its +0x72c when it's made (0x7000d5af00: the largest of its streaming states'), and past that
//   it stops updating them at all (0x7000c3e478) - so the ones made before the settings were scaled, the tiles in at
//   the field's first load, stayed out past 715 m (the rocks below the peak east of that desert went missing at 4x).
//   Those are found by their +0x6f8 (a known settings block) and their +0x72c scaled the same.
// All of them times n (at 4x the desert south of Mesagoza keeps every cliff and the marked rock is detailed - before,
// it stayed blurry; that much more drawn costs fps there: 55 at 4x, 58 at 3x, 60 before). A stand-in still never
// switches past its pieces' group load distance less 160 m (the game's own margin: a 1 km terrain tile's pieces lie that far past its
// stand-in's centre), its group known from the name it gives ({hash, name, length, group} at +0x810's first element:
// "ter_04_02_streaming_m", Mesagoza's "c01_both_01_field_streaming_m"...: _s 220 m, _m 660...).
//
// None of these have a list of their own to follow (the scene's tables mix them with everything else), so a
// low-priority thread looks for them in all writable guest memory (kernel_data_regions; read with ReadProcessMemory -
// never faults if the game unmaps it meanwhile): back to back when the field first loads and for 20 s after the
// GlobalHeap grows by 64 MB at once (fast travel, a new area; 4 a second then - streaming farther, the heap keeps
// growing in steps and back-to-back passes took 45% of a core), every 2 s otherwise. A pass writes the groups first,
// then the settings, then the stand-ins (after what they rely on). Every value is written only while it still holds
// what was read or last written; the game's own are kept by address and put back at "Game's". Off:
// LARECOMP_WORLD_STANDINS=0.
#include <windows.h>
int kernel_data_regions(uint64_t *addr, uint64_t *size, int max);
#define CLOD_VTABLE 0x44a2ed8ull
#define CLOD_DIST   0x808
#define GSG_VTABLE  0x44a4790ull
#define GSG_DIST    0x130
typedef struct { uint64_t at, game, wrote; } ClodSeen;
static ClodSeen clod_seen[16384];  // (open addressing by address; stale entries are checked by value)

static ClodSeen *clod_slot(uint64_t at) {
    uint32_t h = (uint32_t)((at >> 4) * 0x9E3779B1u) & 16383;
    for (int i = 0; i < 64; i++, h = (h + 1) & 16383)
        if (clod_seen[h].at == at || !clod_seen[h].at) return &clod_seen[h];
    return &clod_seen[h];  // (full around here: reuse)
}

// The game's pair at `at` (holding `pair` when read; kept: a new one, or one the game wrote again). False: not one
// (`ok` says what one looks like).
static ClodSeen *pair_game(uint64_t at, uint64_t pair, bool (*ok)(const float *)) {
    float d[2];
    memcpy(d, &pair, 8);
    ClodSeen *s = clod_slot(at);
    if (!(s->at == at && pair == s->wrote)) {
        if (!ok(d)) return NULL;
        s->at = at;
        s->game = s->wrote = pair;
    }
    return s;
}
static void pair_write(ClodSeen *s, uint64_t pair, float k) {
    float g[2];
    memcpy(g, &s->game, 8);
    float w[2] = {g[0] * k, g[1] * k};
    uint64_t to;
    memcpy(&to, w, 8);
    if (to != pair && __sync_bool_compare_and_swap((volatile uint64_t *)(uintptr_t)s->at, pair, to)) s->wrote = to;
}
static bool clod_ok(const float *d) { return d[0] > d[1] && d[1] > 100 && d[0] < 4.0e8f; }  // ({d^2, (d - 20)^2})
static bool group_ok(const float *d) { return d[0] >= 200 && d[1] > d[0] && d[1] < 20000; }  // (not people's...)

// A group's object streaming settings at `at` ({a, a + s, a + 2s, a + 3s} in whole metres, then 1, as read into `f`):
// the game's times n. True: some of the field's (shown within 200 m or more).
typedef struct { uint64_t at; float game[4], wrote[4]; } SetSeen;
static SetSeen set_seen[4096];
static int set_n;
static SetSeen *settings_game(uint64_t at, const float *f) {
    SetSeen *s = NULL;
    for (int i = 0; i < set_n && !s; i++)
        if (set_seen[i].at == at) s = &set_seen[i];
    if (!s || memcmp(f, s->wrote, 16)) {  // (new, or the game wrote it again: the game's)
        float st = f[1] - f[0];
        if (!(f[0] > 0 && f[0] == (float)(int)f[0] && st >= 10 && st <= 200 && f[2] - f[1] == st && f[3] - f[2] == st &&
              f[2] >= 200 && f[3] < 20000))
            return NULL;
        if (!s) {
            if (set_n >= 4096) return NULL;
            s = &set_seen[set_n++];
        }
        s->at = at;
        memcpy(s->game, f, 16);
        memcpy(s->wrote, f, 16);
    }
    return s;
}
// The known settings blocks by address (an object's +0x6f8 is one of them): set_seen's index + 1, 0 none.
static uint64_t set_key[4096];
static int set_val[4096];
static void set_index(void) {
    memset(set_key, 0, sizeof set_key);
    for (int i = 0; i < set_n; i++) {
        uint32_t h = (uint32_t)((set_seen[i].at >> 4) * 0x9E3779B1u) & 4095;
        while (set_key[h]) h = (h + 1) & 4095;
        set_key[h] = set_seen[i].at, set_val[h] = i + 1;
    }
}
static int set_find(uint64_t at) {
    if (!set_n) return 0;
    for (uint32_t h = (uint32_t)((at >> 4) * 0x9E3779B1u) & 4095; set_key[h]; h = (h + 1) & 4095)
        if (set_key[h] == at) return set_val[h];
    return 0;
}
// An object's own copy of its settings' last value (`v` read at `at`): the game's (the settings' own) times n.
static bool object_reach(uint64_t at, float v, const SetSeen *s, int n) {
    float g = s->game[3], m = v / g;
    if (!(m >= 1 && m <= 16 && m == (float)(int)m && v == g * m)) return false;  // (another state's: not ours)
    float to = g * (float)n;
    uint32_t from_u, to_u;
    memcpy(&from_u, &v, 4);
    memcpy(&to_u, &to, 4);
    if (to != v) __sync_bool_compare_and_swap((volatile uint32_t *)(uintptr_t)at, from_u, to_u);
    return true;
}
static void settings_write(SetSeen *s, const float *f, float k) {
    float w[4];
    for (int i = 0; i < 4; i++) w[i] = s->game[i] * k;
    if (!memcmp(w, f, 16)) return;
    volatile float *d = (volatile float *)(uintptr_t)s->at;
    if (w[3] >= f[3]) d[3] = w[3], d[2] = w[2], d[1] = w[1], d[0] = w[0];  // (farther: the far end first,
    else d[0] = w[0], d[1] = w[1], d[2] = w[2], d[3] = w[3];               //  nearer: the near end first)
    memcpy(s->wrote, w, 16);
}

// Its pieces' group load distance, from the name it gives (0: none known - the stand-in stays at the game's).
static float pieces_load(uint64_t entity) {
    uint64_t e0 = 0, name = 0, len = 0;
    char nm[64] = {0};
    SIZE_T r = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), (void *)(uintptr_t)(entity + CLOD_DIST + 8), &e0, 8, &r) || !gptr(e0) ||
        !ReadProcessMemory(GetCurrentProcess(), (void *)(uintptr_t)(e0 + 8), &name, 8, &r) ||
        !ReadProcessMemory(GetCurrentProcess(), (void *)(uintptr_t)(e0 + 16), &len, 8, &r) || !len || len >= sizeof nm ||
        !ReadProcessMemory(GetCurrentProcess(), (void *)(uintptr_t)name, nm, len, &r))
        return 0;
    const char *k = strstr(nm, "streaming_");
    if (!k) return 0;
    k += 10;
    return !strcmp(k, "s") ? 220 : !strcmp(k, "m") ? 660 : !strcmp(k, "l") ? 1870 : !strcmp(k, "xl") ? 3135
         : !strcmp(k, "xxl") ? 6110 : 0;
}

static volatile bool clod_hurry;  // (the game thread: a field is loading - look back to back for a while)

static DWORD WINAPI clod_thread(LPVOID unused) {
    (void)unused;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    static uint8_t buf[(1 << 20) + 0x810];
    static uint64_t ra[1024], rs[1024];
    static struct { uint64_t at, pair; float load; } cand[8192], grp[1024];
    static struct { uint64_t at; float f[4]; } set[4096];
    static struct { uint64_t at; float v; int set; } obj[65536];
    uint64_t fast_until = 0;
    int logged = -1, last_n = -1, nlog = 0;
    for (;;) {
        if (clod_hurry) { clod_hurry = false; fast_until = GetTickCount64() + 20000; }
        Sleep(GetTickCount64() < fast_until ? 250 : 2000);
        int world = rt_sv_world_part("STANDINS"), nr = kernel_data_regions(ra, rs, 1024), nc = 0, ng = 0, ns = 0, no = 0, n = 0;
        uint64_t vt = g_modules[0].base + CLOD_VTABLE, vt_gsg = g_modules[0].base + GSG_VTABLE;
        for (int r = 0; r < nr; r++)
            for (uint64_t c = ra[r], end = ra[r] + rs[r]; c < end; c += 1 << 20) {
                SIZE_T want = end - c < sizeof buf ? end - c : sizeof buf, got = 0;
                if (!ReadProcessMemory(GetCurrentProcess(), (void *)(uintptr_t)c, buf, want, &got) || got < 0x810) continue;
                for (size_t i = 0; i < (1 << 20) && i + 0x810 <= got; i += 8) {
                    uint64_t q = *(uint64_t *)(buf + i);
                    int si = (q >> 36) == 7 && !(q & 15) ? set_find(q) : 0;
                    if (si) {  // (an object's +0x6f8: its +0x72c)
                        if (no < 65536) {
                            obj[no].at = c + i + 0x34, obj[no].set = si - 1;
                            memcpy(&obj[no++].v, buf + i + 0x34, 4);
                        }
                    } else if (q == vt) {
                        if (nc < 8192) cand[nc].at = c + i + CLOD_DIST, cand[nc].pair = *(uint64_t *)(buf + i + CLOD_DIST),
                            cand[nc++].load = pieces_load(c + i);
                    } else if (q == vt_gsg) {
                        if (ng < 1024) grp[ng].at = c + i + GSG_DIST, grp[ng++].pair = *(uint64_t *)(buf + i + GSG_DIST);
                    } else if (*(uint32_t *)(buf + i + 16) == 1 && (q >> 32) > (q & 0xffffffffu) &&
                               (q & 0xffffffffu) >= 0x3f800000u && ns < 4096 &&  // (two rising floats, then ... 1)
                               settings_game(c + i, (const float *)(buf + i))) {
                        set[ns].at = c + i;
                        memcpy(set[ns++].f, buf + i, 16);
                    }
                }
            }
        // Groups, then settings, then stand-ins.
        int gn = 0, sn = 0;
        for (int i = 0; i < ng; i++) {
            ClodSeen *s = pair_game(grp[i].at, grp[i].pair, group_ok);
            if (!s) continue;
            pair_write(s, grp[i].pair, (float)world);
            gn++;
        }
        for (int i = 0; i < ns; i++) {
            SetSeen *s = settings_game(set[i].at, set[i].f);
            if (!s) continue;
            settings_write(s, set[i].f, (float)world);
            sn++;
        }
        int on = 0;
        for (int i = 0; i < no; i++) on += object_reach(obj[i].at, obj[i].v, &set_seen[obj[i].set], world);
        set_index();  // (the objects' blocks, for the next pass)
        for (int i = 0; i < nc; i++) {
            ClodSeen *s = cand[i].load > 0 ? pair_game(cand[i].at, cand[i].pair, clod_ok) : NULL;
            if (!s) continue;
            float g[2], cap = cand[i].load * (float)world - 160, k = (float)world * (float)world;
            memcpy(g, &s->game, 8);
            if (g[0] * k > cap * cap) k = cap * cap > g[0] ? cap * cap / g[0] : 1.0f;
            pair_write(s, cand[i].pair, k);
            n += k > 1.0f;
        }
        if ((logged != world || (n != last_n && (n == 0 || last_n <= 0 || n > last_n + last_n / 4 || n < last_n - last_n / 4))) &&
            nlog++ < 40)
            LOG("[npc] world detail: %d of %d far land stand-ins switch farther (up to x%d), with %d field streaming groups "
                "and %d object streaming settings (%d objects)\n", n, nc, world, gn, sn, on);
        logged = world, last_n = n;
    }
    return 0;
}

void rt_sv_npc_detail(void) {
    static int env_people = -2, env_pokemon = -2;
    static bool clod_started;
    if (g_modules[0].base) terrain_distance(rt_sv_world_part("TERRAIN"));
    {   // (a field loading: the GlobalHeap's use jumps - clod_thread looks back to back for a while)
        static uint64_t last_used;
        static int frames;
        uint64_t used = G64(g_modules[0].base + 0x4763918ull + 0x38);
        static bool field;
        if (!field && used > (512ull << 20)) { field = true; clod_hurry = true; }  // (the field's first load)
        if (++frames % 30 == 0) {
            if (last_used && used > last_used + (64ull << 20)) clod_hurry = true;
            last_used = used;
        }
    }
    if (!clod_started && g_modules[0].base && rt_sv_world_part("STANDINS") > 1) {  // (started once anything is to change)
        clod_started = true;
        CreateThread(NULL, 0, clod_thread, NULL, 0, NULL);
    }
    static LodScale people = {"chara_model", "people's", 0, {0}, {0}, -1};
    static LodScale pokemon = {"pokemon", "Pokémon's", 0, {0}, {0}, -1};
    static uint64_t fac_mgr;     // the manager whose factor table was written
    static float fac_game[12];   // its own factors
    static float fac_wrote;      // what was written there last
    Module *m = &g_modules[0];
    if (!m->base || LOD_GLOBAL + 8 > m->data_off + m->data_size) return;
    uint64_t p = G64(m->base + LOD_GLOBAL);
    if (!gptr(p) || !gptr(p = G64(p)) || !gptr(p = G64(p + 0x38)) || !gptr(p = G64(p + 0x50))) return;
    if (G64(p) != m->base + LOD_MGR_VTABLE) return;
    uint64_t map = G64(p + LOD_MGR_BYNAME);
    if (!gptr(map)) return;
    int world = rt_sv_world_part("LOD");
    volatile float *fac = (volatile float *)(uintptr_t)(p + LOD_MGR_FACTOR);
    bool ours = p == fac_mgr;
    for (int i = 0; i < 12 && ours; i++) ours = fac[i] == fac_game[i] * fac_wrote;
    if (!ours) {  // (a new manager, or the game set its factors again: those are the game's)
        for (int i = 0; i < 12; i++) if (!(fac[i] > 0.01f && fac[i] < 100)) return;  // (not factors)
        fac_mgr = p;
        fac_wrote = 1;
        for (int i = 0; i < 12; i++) fac_game[i] = fac[i];
    }
    if (fac_wrote != (float)world) {
        for (int i = 0; i < 12; i++) fac[i] = fac_game[i] * (float)world;
        fac_wrote = (float)world;
        LOG("[npc] world detail: model LOD distances x%d (the game's factors %g / %g / %g)\n", world, fac_game[0],
            fac_game[1], fac_game[2]);
    }
    scale(map, &people, (float)mul_of("LARECOMP_NPC_DETAIL", &env_people, g_settings.npc_detail) / (float)world);
    scale(map, &pokemon, (float)mul_of("LARECOMP_POKEMON_DETAIL", &env_pokemon, g_settings.pokemon_detail) / (float)world);
}
