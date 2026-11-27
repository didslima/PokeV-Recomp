#!/usr/bin/env python3
"""Unit & edge-case tests for PR #59's rt_sv_timeline_time() in runtime/hle/framepace.c."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent

TEST_C_HEADER = r'''
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ENV(k) getenv(k)
#define LOG(...) ((void)0)
#define RT_ON_RESET(fn) static void fn(void)

static bool rt_pla_is_60fps(void) {
    const char *e = getenv("TEST_60FPS");
    return !e || atoi(e) != 0;
}
'''

TEST_C_MAIN = r'''
typedef struct {
    uint16_t vsize;
    uint16_t tsize;
    uint16_t off0;
    uint16_t off1;
    uint16_t off2;
    uint16_t off3;
} TestVTable;

typedef struct {
    int32_t soff;
    uint32_t s0;
    uint32_t e0;
} TestTable;

static uint8_t g_player1[0x200];
static uint8_t g_player2[0x200];
static uint8_t g_tracks1[0x1d0 * 2];  // (two tracks: an even clip and an odd one)
static uint8_t g_tracks2[0x1d0];

int main(void) {
    // 1. Valid FlatBuffer one-frame clip [10, 11)
    static struct { TestVTable vt; TestTable tbl; } fb1;
    fb1.vt.vsize = sizeof(TestVTable);
    fb1.vt.tsize = sizeof(TestTable);
    fb1.vt.off2 = 4;  // vtable offset 8 -> field at +4 (s0)
    fb1.vt.off3 = 8;  // vtable offset 0xa -> field at +8 (e0)
    fb1.tbl.soff = (int32_t)((uintptr_t)&fb1.tbl - (uintptr_t)&fb1.vt);
    fb1.tbl.s0 = 10;
    fb1.tbl.e0 = 11;

    uint64_t p1 = (uint64_t)(uintptr_t)g_player1;
    uint64_t p2 = (uint64_t)(uintptr_t)g_player2;
    uint64_t tbl1_ptr = (uint64_t)(uintptr_t)&fb1.tbl;
    memcpy(g_tracks1, &tbl1_ptr, 8);
    uint64_t b1 = (uint64_t)(uintptr_t)g_tracks1, e1 = b1 + 0x1d0;
    memcpy(g_player1 + 0xe0 + 0x50, &b1, 8);
    memcpy(g_player1 + 0xe0 + 0x58, &e1, 8);

    // Odd one-frame clip [11, 12) on the same track list (never seen at 30 fps, whose samples land on even frames)
    static struct { TestVTable vt; TestTable tbl; } fb3;
    fb3.vt = fb1.vt;
    fb3.tbl.soff = (int32_t)((uintptr_t)&fb3.tbl - (uintptr_t)&fb3.vt);
    fb3.tbl.s0 = 11;
    fb3.tbl.e0 = 12;
    uint64_t tbl3_ptr = (uint64_t)(uintptr_t)&fb3.tbl;
    memcpy(g_tracks1 + 0x1d0, &tbl3_ptr, 8);
    e1 = b1 + 0x1d0 * 2;
    memcpy(g_player1 + 0xe0 + 0x58, &e1, 8);

    const char *snap_env = getenv("LARECOMP_SV_TIMELINE_SNAP");
    bool snap_on = !snap_env || atoi(snap_env) != 0;
#define NEAR(a, b) (fabsf((a) - (b)) < 1e-4f)

    // 1a. Even clip [10, 11), first sampled at 10.5: played at its own frame, 10 (as 30 fps does)
    assert(NEAR(rt_sv_timeline_time(p1, 10.5f, 1.0f), snap_on ? 10.0f : 10.5f));
    // ...the half frame held back is given on the next sample: 10.0 + 1.0 + 0.5 = 11.5, inside the odd clip [11, 12):
    // never seen at 30 fps, so at its end, 12
    assert(NEAR(rt_sv_timeline_time(p1, snap_on ? 11.0f : 11.5f, 1.0f), snap_on ? 12.0f : 11.5f));
    // ...the half frame gained is given back: 12.0 + 1.0 - 0.5 = 12.5
    assert(NEAR(rt_sv_timeline_time(p1, snap_on ? 13.0f : 12.5f, 1.0f), 12.5f));

    // 1b. Even clip sampled at its whole frame, then again inside it: played once, the second sample at its end
    assert(NEAR(rt_sv_timeline_time(p1, 9.9f, 0.5f), 9.9f));  // (restarted: back from 12.5)
    assert(NEAR(rt_sv_timeline_time(p1, 10.0f, 0.1f), 10.0f));
    assert(NEAR(rt_sv_timeline_time(p1, 10.5f, 0.5f), snap_on ? 11.0f : 10.5f));

    // 2. Restarted sequence / reused address: a held-back time is not paid to a player that starts again
    assert(NEAR(rt_sv_timeline_time(p1, 0.5f, 0.5f), 0.5f));
    assert(NEAR(rt_sv_timeline_time(p1, 10.5f, 0.5f), snap_on ? 10.0f : 10.5f));
    assert(NEAR(rt_sv_timeline_time(p1, 0.5f, 0.5f), 0.5f));

    // 3. Multiple concurrent players + malformed FlatBuffer (soff == 0, vsize == 2)
    static struct { TestVTable vt; TestTable tbl; } fb_bad;
    fb_bad.vt.vsize = 2;  // too small
    fb_bad.tbl.soff = 0;  // invalid soffset
    uint64_t bad_ptr = (uint64_t)(uintptr_t)&fb_bad.tbl;
    memcpy(g_tracks2, &bad_ptr, 8);
    uint64_t b2 = (uint64_t)(uintptr_t)g_tracks2, e2 = b2 + 0x1d0;
    memcpy(g_player2 + 0xe0 + 0x18, &b2, 8);
    memcpy(g_player2 + 0xe0 + 0x20, &e2, 8);
    assert(NEAR(rt_sv_timeline_time(p2, 5.5f, 0.5f), 5.5f));

    // 4. 30 FPS mode leaves time untouched; NaN / negative untouched
    setenv("TEST_60FPS", "0", 1);
    assert(NEAR(rt_sv_timeline_time(p1, 10.5f, 0.5f), 10.5f));
    setenv("TEST_60FPS", "1", 1);
    assert(rt_sv_timeline_time(p1, -1.0f, 0.5f) == -1.0f);

    printf("PASS: sv timeline one-frame clip tests\n");
    return 0;
}
'''


def main():
    if os.name == "nt":
        return
    cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "No C compiler found"
    fp_text = (ROOT / "runtime" / "hle" / "framepace.c").read_text(encoding="utf-8")
    marker = "static uint32_t fb_u32("
    idx = fp_text.find(marker)
    assert idx != -1
    snippet = fp_text[idx:]

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src = t / "test_sv_timeline.c"
        exe = t / "test_sv_timeline"
        src.write_text(TEST_C_HEADER + "\n" + snippet + "\n" + TEST_C_MAIN, encoding="utf-8")
        cmd = shlex.split(cc) + [
            "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Wno-unused-function",
            str(src), "-lm", "-pthread", "-o", str(exe)
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        subprocess.run([str(exe)], cwd=ROOT, check=True)
        env = os.environ.copy()
        env["LARECOMP_SV_TIMELINE_SNAP"] = "0"
        subprocess.run([str(exe)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
