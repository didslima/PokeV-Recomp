"""Per-draw constant-buffer capture (exec.c objmv_capture / mo_cbuf). No GPU is needed.

Extracts MvCbuf, mo_cbuf, and the cbuf capture block from runtime/nvn/vulkan/exec.c,
hooks realloc() to inject allocation failures, and verifies:
- Successful allocation and capture of all constant buffers.
- realloc() failure on the first allocation.
- realloc() failure midway through the 18-buffer copy loop.
- Subsequent frame never treating partially updated entries as valid previous-frame data.
- Recovery on a later successful capture.
- Separate entries per (mesh, program, draw_index) up to 8 draws.
"""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'runtime/nvn/vulkan/exec.c').read_text()

cbuf_start = source.index('typedef struct { uint64_t key, size, frame[2];')
cbuf_mid = source.index('\nstatic bool objmv_on(', cbuf_start)
mo_cbuf_start = source.index('static MvCbuf *mo_cbuf(', cbuf_mid)
cbuf_end = source.index('\nstatic void objmv_capture(', mo_cbuf_start)
cbuf_defs = source[cbuf_start:cbuf_mid] + '\n' + source[mo_cbuf_start:cbuf_end]

cap_start = source.index('    uint32_t mask = pr->vi->cbuf_mask & 0x3FFF8u;', cbuf_end)
cap_end = source.index('    if (!have_prev) { mo.no_prev++; goto out; }', cap_start)
cap_block = source[cap_start:cap_end]

harness = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail_after = -1;  // (-1: never fail; 0: fail next realloc; >0: decrement per call)
static void *test_realloc(void *ptr, size_t size) {
    if (g_fail_after == 0) { g_fail_after = -1; return NULL; }
    if (g_fail_after > 0) g_fail_after--;
    return realloc(ptr, size);
}
#define realloc test_realloc

#define LOG(...) ((void)0)

typedef struct { uint64_t addr, size; } Range;
typedef struct { uint32_t cbuf_mask; } VertexInfo;
typedef struct { VertexInfo *vi; } Prog;

static struct {
    Range ubuf[1][18];
} st;

static uint64_t xm_moves = 1;
static struct {
    uint64_t prev_frame, prev_moves;
    uint64_t captured, no_prev;
} mo;

''' + cbuf_defs + r'''

static bool run_capture(uint64_t addr, Prog *pr, uint64_t f, uint64_t v0_size, bool *out_have_prev, MvCbuf **out_entry) {
    Range v0 = { addr, v0_size };
    if (out_have_prev) *out_have_prev = false;
    bool ok = false;
''' + cap_block + r'''
    if (out_have_prev) *out_have_prev = have_prev;
    ok = true;
out:
    if (out_entry) *out_entry = e;
    return ok;
}

int main(void) {
    VertexInfo vi = { .cbuf_mask = (1u << 3) | (1u << 4) | (1u << 5) };  // (slots 3, 4, 5 -> st.ubuf[0][0..2])
    Prog pr1 = { .vi = &vi }, pr2 = { .vi = &vi };

    uint8_t buf0[32], buf1[32], buf2[32];
    st.ubuf[0][0] = (Range){ (uint64_t)(uintptr_t)buf0, sizeof buf0 };
    st.ubuf[0][1] = (Range){ (uint64_t)(uintptr_t)buf1, sizeof buf1 };
    st.ubuf[0][2] = (Range){ (uint64_t)(uintptr_t)buf2, sizeof buf2 };

    // 1. realloc() failure on the very first allocation (frame 1) -> frame 2 must NOT see have_prev
    memset(buf0, 0x11, sizeof buf0);
    memset(buf1, 0x12, sizeof buf1);
    memset(buf2, 0x13, sizeof buf2);
    g_fail_after = 0;
    bool have_prev = true;
    MvCbuf *e = NULL;
    mo.prev_frame = 0; mo.prev_moves = 1;
    assert(!run_capture(0x100000, &pr1, 1, 1024, &have_prev, &e));
    assert(e != NULL && e->frame[1] == 0);

    mo.prev_frame = 1;
    assert(run_capture(0x100000, &pr1, 2, 1024, &have_prev, &e));
    assert(!have_prev);  // (frame 1 failed, so frame 2 has no valid previous frame)
    assert(e->frame[0] == 2);

    // 2. Successful capture on frame 2 -> frame 3 sees valid previous frame
    memset(buf0, 0x21, sizeof buf0);
    memset(buf1, 0x22, sizeof buf1);
    memset(buf2, 0x23, sizeof buf2);
    mo.prev_frame = 2;
    assert(run_capture(0x100000, &pr1, 3, 1024, &have_prev, &e));
    assert(have_prev);
    assert(e->frame[1] == 3);

    // 3. realloc() failure midway through frame 4 (slot 3 & 4 grow and succeed, slot 5 fails realloc):
    //    must NOT mark e->frame[0] = 4, so frame 5 must NOT see frame 4 as valid previous data
    //    and must NOT expose stale slot 5 from frame 2!
    uint8_t big0[64], big1[64], big2[64];
    memset(big0, 0x41, sizeof big0);
    memset(big1, 0x42, sizeof big1);
    memset(big2, 0x43, sizeof big2);
    st.ubuf[0][0] = (Range){ (uint64_t)(uintptr_t)big0, sizeof big0 };
    st.ubuf[0][1] = (Range){ (uint64_t)(uintptr_t)big1, sizeof big1 };
    st.ubuf[0][2] = (Range){ (uint64_t)(uintptr_t)big2, sizeof big2 };

    g_fail_after = 2;  // (slot 3 succeeds, slot 4 succeeds, slot 5 fails)
    mo.prev_frame = 3;
    assert(!run_capture(0x100000, &pr1, 4, 1024, &have_prev, &e));
    assert(e->frame[0] == 2);  // (still 2 from earlier, NOT 4)

    mo.prev_frame = 4;
    assert(run_capture(0x100000, &pr1, 5, 1024, &have_prev, &e));
    assert(!have_prev);  // (partially updated frame 4 rejected)
    assert(e->frame[1] == 5);

    // 4. Recovery on frame 6 after successful frame 5
    mo.prev_frame = 5;
    assert(run_capture(0x100000, &pr1, 6, 1024, &have_prev, &e));
    assert(have_prev);
    assert(e->len[0][3] == sizeof big0 && e->data[0][3][0] == 0x41);
    assert(e->len[0][4] == sizeof big1 && e->data[0][4][0] == 0x42);
    assert(e->len[0][5] == sizeof big2 && e->data[0][5][0] == 0x43);

    // 5. Two programs on the same mesh + 8-draw cap per (mesh, program)
    MvCbuf *ep1 = NULL, *ep2 = NULL;
    mo.prev_frame = 9;
    assert(run_capture(0x200000, &pr1, 10, 512, &have_prev, &ep1));
    assert(run_capture(0x200000, &pr2, 10, 512, &have_prev, &ep2));
    assert(ep1 != NULL && ep2 != NULL && ep1 != ep2);
    for (int k = 1; k < 8; k++) {
        MvCbuf *ek = NULL;
        assert(run_capture(0x200000, &pr1, 10, 512, &have_prev, &ek));
        assert(ek != NULL);
    }
    MvCbuf *e9 = NULL;
    assert(!run_capture(0x200000, &pr1, 10, 512, &have_prev, &e9));
    assert(e9 == NULL);

    printf("PASS: objmv_cbuf realloc failure, recovery, and per-draw key tests\n");
    return 0;
}
'''

cc = os.environ.get('CC', 'clang')
with tempfile.TemporaryDirectory() as d:
    c = Path(d) / 'test_objmv_cbuf.c'
    exe = Path(d) / 'test_objmv_cbuf'
    c.write_text(harness)
    subprocess.run(shlex.split(cc) + ['-std=c11', '-Wall', '-Wextra', '-Werror', str(c), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
