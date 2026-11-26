#!/usr/bin/env python3
"""Unit & integration tests for runtime/ui/updater.cpp (Phase A update checker)."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent

TEST_CPP = r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

extern "C" {
#include "platform.h"

static std::atomic<bool> g_sim_running{false};
static std::atomic<int> g_kill_count{0};
static std::atomic<int> g_spawn_count{0};

PlatProc plat_spawn(const char *const argv[], const char *log_path, bool append_log) {
    (void)argv; (void)log_path; (void)append_log;
    g_spawn_count++;
    g_sim_running = true;
    return (PlatProc)42;
}
int plat_proc_wait(PlatProc p, bool block) {
    (void)p;
    if (block) {
        while (g_sim_running.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return 1;
    }
    return g_sim_running.load() ? -1 : 1;
}
void plat_proc_kill(PlatProc p) {
    (void)p;
    g_kill_count++;
    g_sim_running = false;
}
}

#include "ui/updater.h"

static void set_env(const char *k, const char *v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1);
    else unsetenv(k);
#endif
}

static UpdaterSnapshot wait_check_done() {
    for (int i = 0; i < 400; i++) {
        UpdaterSnapshot s = updater_get_snapshot();
        if (s.state != UPD_CHK_CHECKING) return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    assert(false && "updater_start_check timed out");
    return updater_get_snapshot();
}

int main() {
    // 1. Version comparison
    assert(updater_tag_rank("beta_2") < updater_tag_rank("v0.0.1"));
    assert(updater_tag_rank("v0.0.30") < updater_tag_rank("v0.1.0"));
    assert(updater_tag_rank("v0.1.0") < updater_tag_rank("v0.1.1"));
    assert(updater_tag_rank("v0.1.1") < updater_tag_rank("v0.2.0"));
    assert(updater_tag_rank("v0.2.0") < updater_tag_rank("v1.0.0"));
    assert(updater_tag_rank("v0.2.0-rc1") <= -2000000LL);
    assert(updater_tag_rank("nightly") <= -2000000LL);
    assert(updater_tag_label("v0.1.0", false) == "0.1.0");
    assert(updater_tag_label("v0.1.0", true) == "Version 0.1.0");
    printf("ok   version comparison\n");

    // 2. Auto-check suppression BEFORE any manual check runs
    set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
    set_env("LARECOMP_UPDATE_JSON", R"({"tag_name":"v0.2.0","draft":false,"prerelease":false})");

    set_env("RECOMPLAUNCHER_DATA", "/tmp/rl-data");
    set_env("LARECOMP_NO_UPDATE_CHECK", nullptr);
    updater_configure("v0.1.0", "/tmp/rl-data", false, false);
    assert(updater_is_suppressed_by_launcher());
    updater_start_check(false);
    assert(updater_get_snapshot().state == UPD_CHK_IDLE && !updater_get_snapshot().has_checked);

    set_env("RECOMPLAUNCHER_DATA", nullptr);
    set_env("LARECOMP_NO_UPDATE_CHECK", "1");
    updater_configure("v0.1.0", "", false, false);
    assert(!updater_is_suppressed_by_launcher());
    updater_start_check(false);
    assert(updater_get_snapshot().state == UPD_CHK_IDLE && !updater_get_snapshot().has_checked);

    set_env("LARECOMP_NO_UPDATE_CHECK", nullptr);
    updater_configure("v0.1.0", "", true, false);
    assert(updater_is_suppressed_by_launcher());
    updater_start_check(false);
    assert(updater_get_snapshot().state == UPD_CHK_IDLE);

    updater_configure("v0.1.0", "", false, true);
    assert(updater_is_suppressed_by_launcher());
    updater_start_check(false);
    assert(updater_get_snapshot().state == UPD_CHK_IDLE);
    printf("ok   auto-check suppression (RECOMPLAUNCHER_DATA, LARECOMP_NO_UPDATE_CHECK, --play, --prepare)\n");

    // 3. JSON parsing, escapes, UTF-16 surrogate pairs & unpaired surrogates
    const char *latest_json = R"({
      "url": "https://api.github.com/repos/didslima/PokeV-Recomp/releases/1",
      "html_url": "https://github.com/didslima/PokeV-Recomp/releases/tag/v0.2.0",
      "tag_name": "v0.2.0",
      "draft": false,
      "prerelease": false,
      "author": { "login": "didslima", "tag_name": "ignored_nested" },
      "assets": [ { "name": "PokeV-Recomp-linux-x86_64.AppImage", "browser_download_url": "https://example.invalid/a" } ],
      "body": "Rocket \uD83D\uDE80 Caf\u00e9 \"q\" \\ bad:\uD83Dx low:\uDE80\r\nend"
    })";
    UpdaterParseResult r = updater_parse_release_json(latest_json);
    assert(r.ok && r.found_stable);
    assert(r.tag_name == "v0.2.0");
    assert(r.html_url == "https://github.com/didslima/PokeV-Recomp/releases/tag/v0.2.0");
    assert(r.body == "Rocket \xF0\x9F\x9A\x80 Caf\xC3\xA9 \"q\" \\ bad:\xEF\xBF\xBDx low:\xEF\xBF\xBD\nend");
    printf("ok   json parsing, utf-16 surrogate pairs & unpaired surrogates\n");

    // 4. Null fields, truncated JSON & stable release filtering
    UpdaterParseResult r_nulls = updater_parse_release_json(
        R"({"tag_name":"v0.2.0","draft":false,"prerelease":false,"body":null,"html_url":null})");
    assert(r_nulls.ok && r_nulls.found_stable && r_nulls.body.empty());
    assert(r_nulls.html_url == "https://github.com/didslima/PokeV-Recomp/releases/tag/v0.2.0");

    assert(!updater_parse_release_json(R"({"tag_name":"v0.2.0")").ok);
    assert(!updater_parse_release_json(R"({"tag_name":"v0.2.0","body":"unterminated)").ok);
    assert(!updater_parse_release_json(R"([{"tag_name":"v0.2.0"})").ok);

    UpdaterParseResult r_draft = updater_parse_release_json(
        R"({"tag_name":"v9.9.9","draft":true,"prerelease":false,"html_url":"https://github.com/x/y","body":"x"})");
    assert(r_draft.ok && !r_draft.found_stable);
    UpdaterParseResult r_pre = updater_parse_release_json(
        R"({"tag_name":"v9.9.9","draft":false,"prerelease":true,"html_url":"https://github.com/x/y","body":"x"})");
    assert(r_pre.ok && !r_pre.found_stable);
    UpdaterParseResult r_arr = updater_parse_release_json(
        R"([{"tag_name":"v0.3.0","draft":false,"prerelease":true},{"tag_name":"v0.2.0","draft":false,"prerelease":false,"html_url":"http://insecure.invalid/x"}])");
    assert(r_arr.ok && r_arr.found_stable && r_arr.tag_name == "v0.2.0");
    assert(r_arr.html_url == "https://github.com/didslima/PokeV-Recomp/releases/tag/v0.2.0");
    printf("ok   null fields, truncated json & stable filtering\n");

    // 5. Async available, up-to-date, and rapid repeated manual checks
    set_env("RECOMPLAUNCHER_DATA", nullptr);
    set_env("LARECOMP_NO_UPDATE_CHECK", nullptr);
    set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
    set_env("LARECOMP_UPDATE_JSON", latest_json);
    updater_configure("v0.1.0", "", false, false);
    for (int i = 0; i < 5; i++) updater_start_check(true);
    UpdaterSnapshot s = wait_check_done();
    assert(s.state == UPD_CHK_AVAILABLE && s.latest_tag == "v0.2.0" && s.has_checked);

    updater_configure("v0.2.0", "", false, false);
    updater_start_check(true);
    s = wait_check_done();
    assert(s.state == UPD_CHK_UP_TO_DATE && s.latest_tag == "v0.2.0");
    printf("ok   async update available, up-to-date & rapid manual checks\n");

    // 6. Offline, rate limit, HTTP error, malformed JSON
    set_env("LARECOMP_UPDATE_HTTP_STATUS", "0");
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_OFFLINE);

    set_env("LARECOMP_UPDATE_HTTP_STATUS", "403");
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_RATE_LIMITED);

    set_env("LARECOMP_UPDATE_HTTP_STATUS", "429");
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_RATE_LIMITED);

    set_env("LARECOMP_UPDATE_HTTP_STATUS", "502");
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_HTTP_ERROR);

    set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
    set_env("LARECOMP_UPDATE_JSON", "{not valid json");
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_HTTP_ERROR);
    printf("ok   offline, rate-limit, http error & malformed json\n");

    // 7. Missing curl, explicit override & Unix PATH fallback
    set_env("LARECOMP_CURL_BIN", "/nonexistent/curl_missing_12345");
    assert(updater_find_curl().empty());
    const char *cur_bin = getenv("LARECOMP_TEST_FAKE_CURL");
    assert(cur_bin && *cur_bin);
    set_env("LARECOMP_CURL_BIN", cur_bin);
    assert(updater_find_curl() == cur_bin);
    set_env("LARECOMP_CURL_BIN", nullptr);
#ifndef _WIN32
    {
        std::string found = updater_find_curl();
        assert(!found.empty() && found.size() >= 5 && found.compare(found.size() - 5, 5, "/curl") == 0);
    }
#endif
    printf("ok   curl missing, explicit override & PATH resolution\n");

    // 8. Shutdown while curl subprocess is actively running + repeated shutdown calls
    set_env("LARECOMP_UPDATE_JSON", nullptr);
    set_env("LARECOMP_UPDATE_HTTP_STATUS", nullptr);
    set_env("LARECOMP_CURL_BIN", cur_bin);
    g_kill_count = 0;
    g_spawn_count = 0;
    updater_configure("v0.1.0", "", false, false);
    updater_start_check(true);
    for (int i = 0; i < 100 && g_spawn_count.load() == 0; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    assert(g_spawn_count.load() == 1);
    updater_shutdown();
    updater_shutdown();  // repeated shutdown must be safe
    assert(g_kill_count.load() == 1);
    printf("ok   shutdown kills running curl, joins thread & repeated shutdown is safe\n");

    printf("PASS: updater phase A checks\n");
    return 0;
}
'''


def main():
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    assert cxx, "No C++ compiler found (set CXX or install clang++/g++)"

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        src = t / "test_updater.cpp"
        exe = t / ("test_updater.exe" if os.name == "nt" else "test_updater")
        fake_curl = t / ("curl.exe" if os.name == "nt" else "curl")
        fake_curl.write_bytes(b"#!/bin/sh\n")
        fake_curl.chmod(0o755)
        src.write_text(TEST_CPP, encoding="utf-8")

        cmd = [
            cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror",
            f"-I{ROOT / 'runtime'}",
            str(src),
            str(ROOT / "runtime" / "ui" / "updater.cpp"),
            "-pthread",
            "-o", str(exe),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        env = os.environ.copy()
        env["LARECOMP_TEST_FAKE_CURL"] = str(fake_curl)
        env["PATH"] = f"{t}{os.pathsep}{env.get('PATH', '')}"
        subprocess.run([str(exe)], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
