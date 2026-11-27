#!/usr/bin/env python3
"""Concurrency stress test for runtime/kernel.c's klock() and block_current() spin wait."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent

TEST_C = r'''
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static pthread_mutex_t g_klock = PTHREAD_MUTEX_INITIALIZER;
static int g_rt_unwind = 0;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void kspin_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

static int g_klock_spin;
static pthread_once_t g_klock_spin_once = PTHREAD_ONCE_INIT;
static void init_klock_spin(void) {
    const char *e = getenv("LARECOMP_KLOCK_SPIN");
    int v = e ? atoi(e) : 2000;
    g_klock_spin = v > 0 ? v : 0;
}
static void klock(void) {
    pthread_once(&g_klock_spin_once, init_klock_spin);
    for (int i = 0; i < g_klock_spin; i++) {
        if (pthread_mutex_trylock(&g_klock) == 0) return;
        kspin_relax();
    }
    pthread_mutex_lock(&g_klock);
}

enum { WAIT_NONE = 0, WAIT_ARBITER = 3 };
typedef struct Waiter {
    int wait_kind;
    int wake_result;
    uint64_t wait_since;
    pthread_cond_t cond;
    struct Waiter *next;
} Waiter;

static Waiter *g_list = NULL;

static void list_add(Waiter *w) {
    w->next = g_list;
    g_list = w;
}
static void list_remove(Waiter *w) {
    for (Waiter **p = &g_list; *p; p = &(*p)->next) {
        if (*p == w) { *p = w->next; return; }
    }
}

static long g_kwait_spin_ns;
static pthread_once_t g_kwait_spin_once = PTHREAD_ONCE_INIT;
static void init_kwait_spin(void) {
    const char *e = getenv("LARECOMP_KWAIT_SPIN_US");
    long us = e ? atol(e) : 30;
    g_kwait_spin_ns = us > 0 ? us * 1000L : 0L;
}

static int block_current(Waiter *w, int64_t timeout_ns) {
    w->wait_since = now_ns();
    list_add(w);
    pthread_once(&g_kwait_spin_once, init_kwait_spin);
    long spin_budget = g_kwait_spin_ns;
    if (timeout_ns > 0 && timeout_ns < spin_budget) spin_budget = (long)timeout_ns;
    if (spin_budget > 0 && timeout_ns != 0 && !__atomic_load_n(&g_rt_unwind, __ATOMIC_ACQUIRE)) {
        pthread_mutex_unlock(&g_klock);
        uint64_t until = w->wait_since + (uint64_t)spin_budget;
        while (__atomic_load_n(&w->wait_kind, __ATOMIC_ACQUIRE) != WAIT_NONE &&
               !__atomic_load_n(&g_rt_unwind, __ATOMIC_ACQUIRE) &&
               now_ns() < until) {
            kspin_relax();
        }
        klock();
    }
    while (w->wait_kind != WAIT_NONE) {
        if (__atomic_load_n(&g_rt_unwind, __ATOMIC_ACQUIRE)) {
            __atomic_store_n(&w->wait_kind, WAIT_NONE, __ATOMIC_RELEASE);
            w->wake_result = -2;
            break;
        }
        if (timeout_ns == 0 || (timeout_ns > 0 && now_ns() >= w->wait_since + (uint64_t)timeout_ns)) {
            if (w->wait_kind == WAIT_NONE) break;
            __atomic_store_n(&w->wait_kind, WAIT_NONE, __ATOMIC_RELEASE);
            w->wake_result = -1;
            break;
        }
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 100000;
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&w->cond, &g_klock, &ts);
    }
    list_remove(w);
    return w->wake_result;
}

static volatile int g_done = 0;
static uint64_t g_woken_count = 0;
static uint64_t g_timeout_count = 0;

static void *waiter_fn(void *arg) {
    int id = (int)(intptr_t)arg;
    Waiter w = {0};
    pthread_cond_init(&w.cond, NULL);
    for (int i = 0; i < 4000; i++) {
        klock();
        __atomic_store_n(&w.wait_kind, WAIT_ARBITER, __ATOMIC_RELEASE);
        w.wake_result = 0;
        int64_t to = (id % 3 == 0) ? 5000 : -1;  // mix short 5us timeouts and untimed waits
        int r = block_current(&w, to);
        if (r == 1) __atomic_fetch_add(&g_woken_count, 1, __ATOMIC_RELAXED);
        else if (r == -1) __atomic_fetch_add(&g_timeout_count, 1, __ATOMIC_RELAXED);
        pthread_mutex_unlock(&g_klock);
    }
    pthread_cond_destroy(&w.cond);
    return NULL;
}

static void *waker_fn(void *arg) {
    (void)arg;
    while (!__atomic_load_n(&g_done, __ATOMIC_ACQUIRE)) {
        klock();
        for (Waiter *w = g_list; w; w = w->next) {
            if (w->wait_kind != WAIT_NONE) {
                __atomic_store_n(&w->wait_kind, WAIT_NONE, __ATOMIC_RELEASE);
                w->wake_result = 1;
                pthread_cond_signal(&w->cond);
            }
        }
        pthread_mutex_unlock(&g_klock);
        kspin_relax();
    }
    return NULL;
}

int main(void) {
    pthread_t waiters[6], waker;
    pthread_create(&waker, NULL, waker_fn, NULL);
    for (int i = 0; i < 6; i++)
        pthread_create(&waiters[i], NULL, waiter_fn, (void *)(intptr_t)i);
    for (int i = 0; i < 6; i++)
        pthread_join(waiters[i], NULL);
    __atomic_store_n(&g_done, 1, __ATOMIC_RELEASE);
    pthread_join(waker, NULL);
    assert(g_woken_count + g_timeout_count == 6 * 4000);
    printf("PASS: kernel spin stress (woken=%llu, timeout=%llu)\n",
           (unsigned long long)g_woken_count, (unsigned long long)g_timeout_count);
    return 0;
}
'''


def split_cmd(cmd, posix_mode=None):
    if posix_mode is None:
        posix_mode = os.name != "nt"
    return shlex.split(cmd, posix=posix_mode)


def main():
    assert split_cmd(r"C:\msys64\ucrt64\bin\clang.EXE", posix_mode=False) == [r"C:\msys64\ucrt64\bin\clang.EXE"]
    cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("gcc")
    assert cc, "No C compiler found"
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src = t / "test_kernel_spin.c"
        exe = t / ("test_kernel_spin.exe" if os.name == "nt" else "test_kernel_spin")
        src.write_text(TEST_C, encoding="utf-8")
        cmd = split_cmd(cc) + ["-O2", "-Wall", "-Wextra", "-Werror", str(src), "-pthread", "-o", str(exe)]
        subprocess.run(cmd, cwd=ROOT, check=True)
        for kspin in ["0", "2000"]:
            for wsp in ["0", "30"]:
                env = os.environ.copy()
                env["LARECOMP_KLOCK_SPIN"] = kspin
                env["LARECOMP_KWAIT_SPIN_US"] = wsp
                subprocess.run([str(exe)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
