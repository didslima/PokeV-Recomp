"""Pre-rendered movie aspect-ratio boxing (exec.c vp_box_aspect + textures.c movie metadata). No GPU is needed."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
exec_src = (root / 'runtime/nvn/vulkan/exec.c').read_text()
tex_src = (root / 'runtime/nvn/vulkan/textures.c').read_text()
vk_h_src = (root / 'runtime/nvn/vulkan/vk.h').read_text()
ws_src = (root / 'runtime/hle/widescreen.c').read_text()

vp_start = exec_src.index('// A draw into a reshaped pass centred to `src_aspect`')
vp_end = exec_src.index('// A draw into a reshaped (wider / taller than 16:9) pass at the uniform scale')
vp_aspect_code = exec_src[vp_start:vp_end]

# Verify vk.h, textures.c, and exec.c wiring
assert 'bool movie_boxed;' in vk_h_src
assert 'b->is_movie = true;' in tex_src
assert 'b->is_movie = false;' in tex_src
assert 'b->movie_boxed = false;' in tex_src
assert 'LARECOMP_WIDE_MOVIE_BOX' in exec_src
assert 'movie_cpu_video_active' in exec_src
assert 'full_movie_copy' in exec_src
assert 'movie_box_rect(b->movie_w, b->movie_h' in exec_src

harness = r'''
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { uint32_t width, height; bool is_movie, movie_boxed; uint32_t movie_w, movie_h; } FakeColor;
typedef FakeColor TexBackend;
static struct {
    struct { FakeColor *color[1]; } rt;
} st;

#define ENV(name) getenv(name)

typedef void *VkCommandBuffer;
typedef enum { VK_FORMAT_UNDEFINED = 0, VK_FORMAT_R8G8B8A8_UNORM = 37 } VkFormat;
typedef enum { VK_IMAGE_ASPECT_COLOR_BIT = 1 } VkImageAspectFlags;
typedef struct { struct { int32_t x, y; } offset; struct { uint32_t width, height; } extent; } VkRect2D;
typedef struct { VkRect2D rect; uint32_t baseArrayLayer, layerCount; } VkClearRect;
typedef struct {
    VkImageAspectFlags aspectMask;
    uint32_t colorAttachment;
    struct { float float32[4]; } color;
} VkClearValueHelper;
typedef struct {
    VkImageAspectFlags aspectMask;
    uint32_t colorAttachment;
    struct { struct { float float32[4]; } color; } clearValue;
} VkClearAttachment;

static struct {
    uint32_t ncolor, layers, movie_bar_w, movie_bar_h;
    bool has_non_movie_draw;
    VkFormat color[1];
} g_pass = { .ncolor = 1, .layers = 1, .color = { VK_FORMAT_R8G8B8A8_UNORM } };

static bool g_cpu_active = false;
static uint32_t g_cpu_w = 0, g_cpu_h = 0;
bool movie_cpu_video_active(uint32_t *w, uint32_t *h) {
    if (!g_cpu_active) return false;
    *w = g_cpu_w; *h = g_cpu_h;
    return true;
}
#define pass_cur() (&g_pass)
#define pass_ncolor (g_pass.ncolor)
#define pass_color (g_pass.color)
#define pass_layers (g_pass.layers)

static uint32_t g_clear_calls = 0;
static VkClearRect g_last_rects[2];
static float g_last_clear_rgba[4];
static void rec_CmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a, uint32_t nr, const VkClearRect *r) {
    (void)cb; (void)na;
    g_clear_calls++;
    for (int i = 0; i < 4; i++) g_last_clear_rgba[i] = a[0].clearValue.color.float32[i];
    for (uint32_t i = 0; i < nr && i < 2; i++) g_last_rects[i] = r[i];
}

static int g_pw, g_ph;
static float pass_sx, pass_sy, vp_sx, vp_sy, vp_ox, vp_oy;
static void pass_size(int *w, int *h) { *w = g_pw; *h = g_ph; }

''' + vp_aspect_code + r'''

#define NEAR(a, b) (fabsf((a) - (b)) < 1e-2f)

static void run_case(int pw, int ph, uint32_t gw, uint32_t gh, uint32_t mw, uint32_t mh,
                     float exp_x, float exp_y, float exp_w, float exp_h) {
    FakeColor rt = { .width = gw, .height = gh, .is_movie = true, .movie_w = mw, .movie_h = mh };
    st.rt.color[0] = &rt;
    g_pw = pw; g_ph = ph;
    pass_sx = (float)pw / (float)gw;
    pass_sy = (float)ph / (float)gh;
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0.0f; vp_oy = 0.0f;
    vp_box_aspect((float)mw / (float)mh);
    float rx = 0.0f * vp_sx + vp_ox;
    float ry = 0.0f * vp_sy + vp_oy;
    float rw = (float)gw * vp_sx;
    float rh = (float)gh * vp_sy;
    assert(NEAR(rx, exp_x));
    assert(NEAR(ry, exp_y));
    assert(NEAR(rw, exp_w));
    assert(NEAR(rh, exp_h));
}

int main(void) {
    // 1. 5120x1440 (32:9), 1920x1080 source -> 2560x1440 centered
    run_case(5120, 1440, 1920, 1080, 1920, 1080, 1280.0f, 0.0f, 2560.0f, 1440.0f);
    // 2. 3440x1440 (21:9), 1920x1080 source -> 2560x1440 centered
    run_case(3440, 1440, 1920, 1080, 1920, 1080, 440.0f, 0.0f, 2560.0f, 1440.0f);
    // 3. 1920x1080 (16:9), 1920x1080 source -> unchanged
    run_case(1920, 1080, 1920, 1080, 1920, 1080, 0.0f, 0.0f, 1920.0f, 1080.0f);
    // 4. 1920x1200 (16:10), 1920x1080 source -> 1920x1080 centered vertically
    run_case(1920, 1200, 1920, 1080, 1920, 1080, 0.0f, 60.0f, 1920.0f, 1080.0f);
    // 5. 5120x1440 (32:9), 640x480 (4:3) source -> 1920x1440 centered
    run_case(5120, 1440, 1920, 1080, 640, 480, 1600.0f, 0.0f, 1920.0f, 1440.0f);

    // 6. Invalid / zero source dimensions -> vp_* untouched
    FakeColor rt = { .width = 1920, .height = 1080, .is_movie = true, .movie_w = 0, .movie_h = 0 };
    st.rt.color[0] = &rt;
    g_pw = 5120; g_ph = 1440;
    pass_sx = 5120.0f / 1920.0f; pass_sy = 1440.0f / 1080.0f;
    vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0.0f; vp_oy = 0.0f;
    vp_box_aspect(0.0f);
    vp_box_aspect(-1.0f);
    vp_box_aspect(NAN);
    vp_box_aspect(INFINITY);
    assert(NEAR(vp_sx, pass_sx) && NEAR(vp_sy, pass_sy) && vp_ox == 0.0f && vp_oy == 0.0f);

    // 7. Changing movie dimensions updates centered rectangle
    run_case(5120, 1440, 1920, 1080, 1280, 720, 1280.0f, 0.0f, 2560.0f, 1440.0f);
    run_case(5120, 1440, 1920, 1080, 2560, 1080, 853.333f, 0.0f, 3413.333f, 1440.0f);

    // 8. Kill switch check
    const char *ks = getenv("LARECOMP_WIDE_MOVIE_BOX");
    bool box_on = !ks || atoi(ks) != 0;
    if (ks && atoi(ks) == 0) assert(!box_on);
    else assert(box_on);

    // 9. Multiple movie aspect ratios in the same render pass: 16:9 (bw=1280) then 4:3 (bw=1600)
    //    must re-clear wider bars to opaque black {0,0,0,1}, and a narrower 16:9 draw afterwards skips
    g_pass.movie_bar_w = g_pass.movie_bar_h = 0;
    g_clear_calls = 0;
    run_case(5120, 1440, 1920, 1080, 1920, 1080, 1280.0f, 0.0f, 2560.0f, 1440.0f);
    clear_movie_bars(NULL);
    assert(g_clear_calls == 1 && g_pass.movie_bar_w == 1280);
    assert(g_last_clear_rgba[0] == 0.0f && g_last_clear_rgba[1] == 0.0f && g_last_clear_rgba[2] == 0.0f && g_last_clear_rgba[3] == 1.0f);
    assert(g_last_rects[0].rect.extent.width == 1280 && g_last_rects[1].rect.offset.x == 3840);

    run_case(5120, 1440, 1920, 1080, 640, 480, 1600.0f, 0.0f, 1920.0f, 1440.0f);
    clear_movie_bars(NULL);
    assert(g_clear_calls == 2 && g_pass.movie_bar_w == 1600);
    assert(g_last_rects[0].rect.extent.width == 1600 && g_last_rects[1].rect.offset.x == 3520);

    run_case(5120, 1440, 1920, 1080, 1920, 1080, 1280.0f, 0.0f, 2560.0f, 1440.0f);
    clear_movie_bars(NULL);
    assert(g_clear_calls == 2);

    // 10. CPU fallback dimension matching (exact, 256/16 aligned, half-res chroma plane)
    assert(movie_cpu_tex_match(1920, 1080, 1920, 1080));
    assert(movie_cpu_tex_match(2048, 1088, 1920, 1080));
    assert(movie_cpu_tex_match(960, 540, 1920, 1080));
    assert(movie_cpu_tex_match(1024, 544, 1920, 1080));
    assert(!movie_cpu_tex_match(800, 600, 1920, 1080));
    assert(!movie_cpu_tex_match(1920, 1080, 1280, 720));

    // 11. Direct vk_present_draw rect boxing
    {
        float vx = 0.0f, vy = 0.0f, vw = 5120.0f, vh = 1440.0f;
        movie_box_rect(1920, 1080, &vx, &vy, &vw, &vh);
        assert(NEAR(vx, 1280.0f) && NEAR(vy, 0.0f) && NEAR(vw, 2560.0f) && NEAR(vh, 1440.0f));

        vx = 0.0f; vy = 0.0f; vw = 3440.0f; vh = 1440.0f;
        movie_box_rect(1920, 1080, &vx, &vy, &vw, &vh);
        assert(NEAR(vx, 440.0f) && NEAR(vy, 0.0f) && NEAR(vw, 2560.0f) && NEAR(vh, 1440.0f));

        vx = 0.0f; vy = 0.0f; vw = 1920.0f; vh = 1080.0f;
        movie_box_rect(1920, 1080, &vx, &vy, &vw, &vh);
        assert(NEAR(vx, 0.0f) && NEAR(vy, 0.0f) && NEAR(vw, 1920.0f) && NEAR(vh, 1080.0f));
    }

    // 12. 1:1 intermediate RT -> reshaped composition -> second reshaped pass (no double boxing)
    if (box_on) {
        TexBackend dec = { .width = 1920, .height = 1080, .is_movie = true, .movie_boxed = false, .movie_w = 1920, .movie_h = 1080 };
        TexBackend mid = { .width = 1920, .height = 1080 };
        TexBackend sc  = { .width = 1920, .height = 1080 };
        TexBackend sc2 = { .width = 1920, .height = 1080 };
        TexBackend *srcs[1] = { &dec };
        uint32_t wh[1][2] = { { 1920, 1080 } };

        // Pass 1 (1:1): YUV->RGBA / filter pass propagates is_movie (movie_boxed stays false)
        g_pass.has_non_movie_draw = false;
        g_pass.movie_bar_w = g_pass.movie_bar_h = 0;
        g_pw = 1920; g_ph = 1080;
        pass_sx = 1.0f; pass_sy = 1.0f; vp_sx = 1.0f; vp_sy = 1.0f; vp_ox = 0.0f; vp_oy = 0.0f;
        st.rt.color[0] = &mid;
        bool boxed1 = movie_draw_apply(NULL, &mid, 1, 1, srcs, wh, true, false);
        assert(!boxed1 && mid.is_movie && !mid.movie_boxed && mid.movie_w == 1920 && mid.movie_h == 1080);

        // Pass 2 (reshaped 32:9): sampling mid applies boxing + bars and sets sc.movie_boxed = true
        g_pass.has_non_movie_draw = false;
        g_pass.movie_bar_w = g_pass.movie_bar_h = 0;
        g_clear_calls = 0;
        g_pw = 5120; g_ph = 1440;
        pass_sx = 5120.0f / 1920.0f; pass_sy = 1440.0f / 1080.0f;
        vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0.0f; vp_oy = 0.0f;
        st.rt.color[0] = &sc;
        srcs[0] = &mid;
        bool boxed2 = movie_draw_apply(NULL, &sc, 1, 1, srcs, wh, true, true);
        assert(boxed2 && g_clear_calls == 1 && NEAR(vp_ox, 1280.0f) && sc.is_movie && sc.movie_boxed);

        // Pass 3 (reshaped 32:9 sampling already-boxed sc): skips vp_box_aspect and clear_movie_bars, returns true
        g_pass.has_non_movie_draw = false;
        g_pass.movie_bar_w = g_pass.movie_bar_h = 0;
        g_clear_calls = 0;
        vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0.0f; vp_oy = 0.0f;
        st.rt.color[0] = &sc2;
        srcs[0] = &sc;
        bool boxed3 = movie_draw_apply(NULL, &sc2, 1, 1, srcs, wh, true, true);
        assert(boxed3 && g_clear_calls == 0 && vp_ox == 0.0f && NEAR(vp_sx, pass_sx) && sc2.movie_boxed);

        // 13. Mixing non-movie draw onto unboxed movie RT clears is_movie; mixed pass won't re-mark
        g_pass.has_non_movie_draw = false;
        TexBackend ui_tex = { .width = 512, .height = 512, .is_movie = false };
        TexBackend *ui_srcs[1] = { &ui_tex };
        uint32_t ui_wh[1][2] = { { 512, 512 } };
        assert(!movie_draw_apply(NULL, &mid, 1, 1, ui_srcs, ui_wh, true, false));
        assert(!mid.is_movie && g_pass.has_non_movie_draw);
        srcs[0] = &dec;
        assert(!movie_draw_apply(NULL, &mid, 1, 1, srcs, wh, true, false));
        assert(!mid.is_movie);

        // 14. CPU fallback requires both active decoder AND video_draw_ok
        g_pass.has_non_movie_draw = false;
        g_cpu_active = true; g_cpu_w = 1920; g_cpu_h = 1080;
        TexBackend cpu_tex = { .width = 2048, .height = 1088, .is_movie = false };
        TexBackend *cpu_srcs[1] = { &cpu_tex };
        uint32_t cpu_wh[1][2] = { { 2048, 1088 } };
        TexBackend cpu_rt = { .width = 1920, .height = 1080 };
        st.rt.color[0] = &cpu_rt;
        // video_draw_ok = false (e.g. 3D depth-tested draw or small UI quad) must NOT match CPU fallback
        assert(!movie_draw_apply(NULL, &cpu_rt, 1, 1, cpu_srcs, cpu_wh, false, true));
        assert(!cpu_rt.is_movie);
        // video_draw_ok = true with fresh pass matches aligned CPU texture
        g_pass.has_non_movie_draw = false;
        g_pass.movie_bar_w = g_pass.movie_bar_h = 0;
        vp_sx = pass_sx; vp_sy = pass_sy; vp_ox = 0.0f; vp_oy = 0.0f;
        assert(movie_draw_apply(NULL, &cpu_rt, 1, 1, cpu_srcs, cpu_wh, true, true));
        assert(cpu_rt.is_movie && cpu_rt.movie_boxed && NEAR(vp_ox, 1280.0f));
        g_cpu_active = false;
    }

    // 15. movie_copy_is_full: full copy, partial copy, differing dims, aliases, layers, mip levels
    {
        // Full base-level/layer copy (1920x1080 -> 1920x1080)
        assert(movie_copy_is_full(0, 0, 0, 0, 0, 0, 0, 0, 1920, 1080, 1920, 1080, 1920, 1080, 1920, 1080));
        // Full copy with differing src/dst content dimensions (e.g. 1920x1080 -> 1280x720 or 2560x1440)
        assert(movie_copy_is_full(0, 0, 0, 0, 0, 0, 0, 0, 1920, 1080, 1280, 720, 1920, 1080, 1280, 720));
        assert(movie_copy_is_full(0, 0, 0, 0, 0, 0, 0, 0, 1920, 1080, 2560, 1440, 1920, 1080, 2560, 1440));
        // Partial copies: non-zero offsets or smaller region than content extent
        assert(!movie_copy_is_full(0, 0, 0, 0, 16, 0, 0, 0, 1904, 1080, 1904, 1080, 1920, 1080, 1920, 1080));
        assert(!movie_copy_is_full(0, 0, 0, 0, 0, 16, 0, 0, 1920, 1064, 1920, 1064, 1920, 1080, 1920, 1080));
        assert(!movie_copy_is_full(0, 0, 0, 0, 0, 0, 8, 0, 1920, 1080, 1912, 1080, 1920, 1080, 1920, 1080));
        assert(!movie_copy_is_full(0, 0, 0, 0, 0, 0, 0, 8, 1920, 1080, 1920, 1072, 1920, 1080, 1920, 1080));
        assert(!movie_copy_is_full(0, 0, 0, 0, 0, 0, 0, 0, 1280, 720, 1280, 720, 1920, 1080, 1920, 1080));
        // Destination larger than copied region (e.g. 1920x1080 into padded 2048x1088) must NOT propagate
        assert(!movie_copy_is_full(0, 0, 0, 0, 0, 0, 0, 0, 1920, 1080, 1920, 1080, 1920, 1080, 2048, 1088));
        // Mip levels and array layers != 0 must NOT propagate
        assert(!movie_copy_is_full(1, 0, 0, 0, 0, 0, 0, 0, 960, 540, 960, 540, 960, 540, 960, 540));
        assert(!movie_copy_is_full(0, 1, 0, 0, 0, 0, 0, 0, 960, 540, 960, 540, 960, 540, 960, 540));
        assert(!movie_copy_is_full(0, 0, 1, 0, 0, 0, 0, 0, 1920, 1080, 1920, 1080, 1920, 1080, 1920, 1080));
        assert(!movie_copy_is_full(0, 0, 0, 1, 0, 0, 0, 0, 1920, 1080, 1920, 1080, 1920, 1080, 1920, 1080));
        // Alias simulation: whole-image copy from alias backend propagates movie state without losing movie_boxed
        TexBackend alias_src = { .width = 1920, .height = 1080, .is_movie = true, .movie_boxed = true, .movie_w = 1920, .movie_h = 1080 };
        TexBackend alias_dst = { .width = 1920, .height = 1080 };
        if (movie_copy_is_full(0, 0, 0, 0, 0, 0, 0, 0, 1920, 1080, 1920, 1080, 1920, 1080, 1920, 1080)) {
            alias_dst.is_movie = alias_src.is_movie;
            alias_dst.movie_boxed = alias_src.movie_boxed;
            alias_dst.movie_w = alias_src.movie_w;
            alias_dst.movie_h = alias_src.movie_h;
        }
        assert(alias_dst.is_movie && alias_dst.movie_boxed && alias_dst.movie_w == 1920 && alias_dst.movie_h == 1080);
    }

    printf("PASS: movie aspect-ratio boxing tests\n");
    return 0;
}
'''

cc = os.environ.get('CC', 'clang')
with tempfile.TemporaryDirectory() as d:
    c = Path(d) / 'test_movie_box.c'
    exe = Path(d) / 'test_movie_box'
    c.write_text(harness)
    subprocess.run(shlex.split(cc) + ['-std=c11', '-Wall', '-Wextra', '-Werror', str(c), '-lm', '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    subprocess.run([str(exe)], check=True, env={**os.environ, 'LARECOMP_WIDE_MOVIE_BOX': '0'})
    subprocess.run([str(exe)], check=True, env={**os.environ, 'LARECOMP_WIDE_MOVIE_BOX': '1'})
