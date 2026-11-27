#!/usr/bin/env python3
"""Unit & integration tests for runtime/ui/updater.cpp (Phase A checker + Phase B Linux AppImage updater)."""
from pathlib import Path
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent

TEST_CPP = r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/stat.h>
#include <thread>

#ifndef _WIN32
#include <unistd.h>
#endif

extern "C" {
#include "platform.h"

static std::atomic<bool> g_sim_running{false};
static std::atomic<int> g_kill_count{0};
static std::atomic<int> g_spawn_count{0};
static std::atomic<int> g_sim_rc{1};

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
        return g_sim_rc.load();
    }
    return g_sim_running.load() ? -1 : g_sim_rc.load();
}
void plat_proc_kill(PlatProc p) {
    (void)p;
    g_kill_count++;
    g_sim_running = false;
}
}

#include "ui/updater.h"

namespace fs = std::filesystem;

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

#ifdef __linux__
static UpdaterSnapshot wait_install_done() {
    for (int i = 0; i < 400; i++) {
        UpdaterSnapshot s = updater_get_snapshot();
        if (s.install_state != UPD_INST_DOWNLOADING && s.install_state != UPD_INST_VERIFYING) return s;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    assert(false && "updater_start_install timed out");
    return updater_get_snapshot();
}
#endif

static void write_bytes(const std::string &path, const std::string &data) {
    FILE *f = fopen(path.c_str(), "wb");
    assert(f);
    assert(fwrite(data.data(), 1, data.size(), f) == data.size());
    fclose(f);
}

#ifdef __linux__
static std::string read_bytes(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    assert(f);
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f);
    return out;
}
#endif

int main() {
    const char *tmp_dir_c = getenv("LARECOMP_TEST_TMPDIR");
    const char *expected_hello_sha = getenv("LARECOMP_TEST_HELLO_SHA");
    assert(tmp_dir_c && expected_hello_sha);
    std::string tmp_dir = tmp_dir_c;

    // 1. Version comparison (Phase A + RC ordering)
    assert(updater_tag_rank("beta_2") < updater_tag_rank("v0.0.1"));
    assert(updater_tag_rank("v0.0.30") < updater_tag_rank("v0.1.0"));
    assert(updater_tag_rank("v0.1.0") < updater_tag_rank("v0.1.1-alpha.1"));
    assert(updater_tag_rank("v0.1.1-alpha.1") < updater_tag_rank("v0.1.1-beta.1"));
    assert(updater_tag_rank("v0.1.1-beta.1") < updater_tag_rank("v0.1.1-rc.1"));
    assert(updater_tag_rank("v0.1.1-rc.1") < updater_tag_rank("v0.1.1-rc.2"));
    assert(updater_tag_rank("v0.1.1-rc.2") < updater_tag_rank("v0.1.1-rc.10"));
    assert(updater_tag_rank("v0.1.1-rc.10") < updater_tag_rank("v0.1.1"));
    assert(updater_tag_rank("v0.1.1") < updater_tag_rank("v0.2.0"));
    assert(updater_tag_rank("v0.2.0") < updater_tag_rank("v1.0.0"));
    assert(updater_tag_rank("v0.1.1-rc.0") <= -20000000000LL);
    assert(updater_tag_rank("v0.1.1-rc.abc") <= -20000000000LL);
    assert(updater_tag_rank("v0.1.1-foo") <= -20000000000LL);
    assert(updater_tag_rank("nightly") <= -20000000000LL);
    assert(updater_tag_label("v0.1.0", false) == "0.1.0");
    assert(updater_tag_label("v0.1.1-rc.1", false) == "0.1.1-rc.1");
    assert(updater_tag_label("v0.1.0", true) == "Version 0.1.0");
    printf("ok   version comparison (stable & RC)\n");

    // 2. Auto-check suppression BEFORE any manual check runs (Phase A)
    set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
    set_env("LARECOMP_UPDATE_JSON", R"({"tag_name":"v0.2.0","draft":false,"prerelease":false})");

    set_env("RECOMPLAUNCHER_DATA", "/tmp/rl-data");
    set_env("LARECOMP_NO_UPDATE_CHECK", nullptr);
    updater_configure("v0.1.0", "/tmp/rl-data", false, false);
    assert(updater_is_suppressed_by_launcher());
    assert(updater_get_snapshot().install_kind == UPD_INSTALL_RECOMPLAUNCHER);
    updater_start_check(false);
    assert(updater_get_snapshot().state == UPD_CHK_IDLE && !updater_get_snapshot().has_checked);

    set_env("RECOMPLAUNCHER_DATA", nullptr);
    set_env("LARECOMP_NO_UPDATE_CHECK", "1");
    updater_configure("v0.1.0", "", false, false);
    updater_start_check(false);
    assert(updater_get_snapshot().state == UPD_CHK_IDLE && !updater_get_snapshot().has_checked);
    printf("ok   auto-check & RecompLauncher suppression\n");

    // 3. JSON parsing + asset extraction + UTF-16 surrogates (Phase A + B)
    const char *latest_json = R"({
      "url": "https://api.github.com/repos/didslima/PokeV-Recomp/releases/1",
      "html_url": "https://github.com/didslima/PokeV-Recomp/releases/tag/v0.2.0",
      "tag_name": "v0.2.0",
      "draft": false,
      "prerelease": false,
      "author": { "login": "didslima", "tag_name": "ignored_nested" },
      "assets": [
        { "name": "PokeV-Recomp-windows-x64.zip", "size": 999, "browser_download_url": "https://github.com/x/win.zip" },
        { "name": "PokeV-Recomp-linux-x86_64.AppImage", "size": 11, "browser_download_url": "https://github.com/x/lin.AppImage" },
        { "name": "SHA256SUMS", "size": 120, "browser_download_url": "https://github.com/x/SHA256SUMS" }
      ],
      "body": "Rocket \uD83D\uDE80 Caf\u00e9 \"q\" \\ bad:\uD83Dx low:\uDE80\r\nend"
    })";
    UpdaterParseResult r = updater_parse_release_json(latest_json);
    assert(r.ok && r.found_stable);
    assert(r.tag_name == "v0.2.0");
    assert(r.linux_asset_url == "https://github.com/x/lin.AppImage");
    assert(r.linux_asset_size == 11);
    assert(r.sha256sums_url == "https://github.com/x/SHA256SUMS");
    assert(r.body == "Rocket \xF0\x9F\x9A\x80 Caf\xC3\xA9 \"q\" \\ bad:\xEF\xBF\xBDx low:\xEF\xBF\xBD\nend");
    printf("ok   json parsing, assets & utf-16 surrogate pairs\n");

    // 4. Null fields, truncated JSON, stable filtering (Phase A)
    assert(!updater_parse_release_json(R"({"tag_name":"v0.2.0")").ok);
    assert(!updater_parse_release_json(R"([{"tag_name":"v0.2.0"})").ok);
    UpdaterParseResult r_nulls = updater_parse_release_json(
        R"({"tag_name":"v0.2.0","draft":false,"prerelease":false,"body":null,"html_url":null})");
    assert(r_nulls.ok && r_nulls.found_stable && r_nulls.body.empty());
    UpdaterParseResult r_draft = updater_parse_release_json(
        R"({"tag_name":"v9.9.9","draft":true,"prerelease":false})");
    assert(r_draft.ok && !r_draft.found_stable);
    printf("ok   null fields, truncated json & stable filtering\n");

    // 4b. Stable channel vs opt-in LARECOMP_UPDATE_PRERELEASE=1
    const char *rc_array_json = R"([
      {"tag_name":"v0.1.2-rc.1","draft":true,"prerelease":true,"html_url":"https://github.com/x/d"},
      {"tag_name":"v0.1.1-rc.1","draft":false,"prerelease":true,"html_url":"https://github.com/x/rc1","body":"RC 1",
       "assets":[{"name":"PokeV-Recomp-linux-x86_64.AppImage","size":11,"browser_download_url":"https://github.com/x/rc1.AppImage"},
                 {"name":"SHA256SUMS","size":100,"browser_download_url":"https://github.com/x/SHA256SUMS"}]},
      {"tag_name":"v0.1.0","draft":false,"prerelease":false,"html_url":"https://github.com/x/010","body":"Stable"}
    ])";
    // Default (stable channel): must select v0.1.0 and NOT expose v0.1.1-rc.1
    UpdaterParseResult r_stable_only = updater_parse_release_json_opts(rc_array_json, false);
    assert(r_stable_only.ok && r_stable_only.found_stable && r_stable_only.tag_name == "v0.1.0");
    // Opt-in prerelease channel: must ignore draft v0.1.2-rc.1 and select v0.1.1-rc.1
    UpdaterParseResult r_rc_optin = updater_parse_release_json_opts(rc_array_json, true);
    assert(r_rc_optin.ok && r_rc_optin.found_stable && r_rc_optin.tag_name == "v0.1.1-rc.1");
    assert(r_rc_optin.linux_asset_url == "https://github.com/x/rc1.AppImage");

    set_env("LARECOMP_NO_UPDATE_CHECK", nullptr);
    set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
    set_env("LARECOMP_UPDATE_JSON", rc_array_json);
    set_env("LARECOMP_UPDATE_PRERELEASE", nullptr);
    updater_configure("v0.1.0", "", false, false);
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_UP_TO_DATE);  // regular user stays on v0.1.0

    set_env("LARECOMP_UPDATE_PRERELEASE", "1");
    updater_start_check(true);
    UpdaterSnapshot s_rc = wait_check_done();
    assert(s_rc.state == UPD_CHK_AVAILABLE && s_rc.latest_tag == "v0.1.1-rc.1");
    set_env("LARECOMP_UPDATE_PRERELEASE", nullptr);
    printf("ok   stable channel isolation & LARECOMP_UPDATE_PRERELEASE=1 opt-in\n");

    // 5. Async checker states & errors (Phase A)
    set_env("LARECOMP_NO_UPDATE_CHECK", nullptr);
    set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
    set_env("LARECOMP_UPDATE_JSON", latest_json);
    updater_configure("v0.1.0", "", false, false);
    for (int i = 0; i < 5; i++) updater_start_check(true);
    UpdaterSnapshot s = wait_check_done();
    assert(s.state == UPD_CHK_AVAILABLE && s.latest_tag == "v0.2.0");

    set_env("LARECOMP_UPDATE_HTTP_STATUS", "0");
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_OFFLINE);

    set_env("LARECOMP_UPDATE_HTTP_STATUS", "429");
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_RATE_LIMITED);
    printf("ok   async checker states & error handling\n");

    // 6. SHA256SUMS parsing & file_sha256 (Phase B)
    std::string sums_sample =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  PokeV-Recomp-windows-x64.zip\n" +
        std::string(expected_hello_sha) + " *./PokeV-Recomp-linux-x86_64.AppImage\n";
    assert(updater_parse_sha256sums(sums_sample, "PokeV-Recomp-linux-x86_64.AppImage") == expected_hello_sha);
    assert(updater_parse_sha256sums(sums_sample, "missing.AppImage").empty());
    assert(updater_parse_sha256sums("nothex  PokeV-Recomp-linux-x86_64.AppImage\n", "PokeV-Recomp-linux-x86_64.AppImage").empty());

    std::string hello_file = tmp_dir + "/hello.bin";
    write_bytes(hello_file, "hello world");
    assert(updater_file_sha256(hello_file) == expected_hello_sha);
    printf("ok   sha256sums parsing & file_sha256\n");

#ifdef __linux__
    // 7. Install kind detection, paths with spaces & special chars, read-only dir (Phase B)
    set_env("APPIMAGE", nullptr);
    set_env("LARECOMP_DATA_ROOT", nullptr);
    std::string tgt, reason;
    assert(updater_detect_install_kind(tgt, reason) == UPD_INSTALL_UNSUPPORTED);

    std::string spaced_dir = tmp_dir + "/My AppImage (v1) 'test'";
    fs::create_directories(spaced_dir);
    std::string app_path = spaced_dir + "/PokeV-Recomp.AppImage";
    write_bytes(app_path, "old-appimage");
    chmod(app_path.c_str(), 0751);

    set_env("APPIMAGE", app_path.c_str());
    assert(updater_detect_install_kind(tgt, reason) == UPD_INSTALL_LINUX_APPIMAGE);
    assert(tgt == fs::canonical(app_path).string());

    if (geteuid() != 0) {
        std::string ro_dir = tmp_dir + "/ro_dir";
        fs::create_directories(ro_dir);
        std::string ro_app = ro_dir + "/PokeV.AppImage";
        write_bytes(ro_app, "ro");
        chmod(ro_dir.c_str(), 0555);
        set_env("APPIMAGE", ro_app.c_str());
        assert(updater_detect_install_kind(tgt, reason) == UPD_INSTALL_LINUX_APPIMAGE_RO);
        chmod(ro_dir.c_str(), 0755);
    }

    // 8. Download, SHA-256 verification, staging, corrupted file, low disk space (Phase B)
    std::string data_dir = tmp_dir + "/userdata";
    fs::create_directories(data_dir + "/cache");
    std::string new_app_src = tmp_dir + "/new_src.AppImage";
    write_bytes(new_app_src, "hello world");  // 11 bytes (matches latest_json size=11)
    std::string sums_src = tmp_dir + "/SRC_SHA256SUMS";
    write_bytes(sums_src, std::string(expected_hello_sha) + "  PokeV-Recomp-linux-x86_64.AppImage\n");

    set_env("APPIMAGE", app_path.c_str());
    set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
    set_env("LARECOMP_UPDATE_JSON", latest_json);
    set_env("LARECOMP_UPDATE_APPIMAGE", new_app_src.c_str());
    set_env("LARECOMP_UPDATE_SHA256SUMS", sums_src.c_str());
    set_env("LARECOMP_UPDATE_MIN_FREE_BYTES", nullptr);

    updater_configure("v0.1.0", data_dir.c_str(), false, false);
    updater_start_check(true);
    assert(wait_check_done().state == UPD_CHK_AVAILABLE);

    // 8a. Low disk space failure
    set_env("LARECOMP_UPDATE_MIN_FREE_BYTES", "1024");
    updater_start_install();
    s = wait_install_done();
    assert(s.install_state == UPD_INST_FAILED);
    assert(read_bytes(app_path) == "old-appimage");

    // 8b. Corrupted checksum failure
    set_env("LARECOMP_UPDATE_MIN_FREE_BYTES", nullptr);
    std::string bad_sums = tmp_dir + "/BAD_SHA256SUMS";
    write_bytes(bad_sums, "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff  PokeV-Recomp-linux-x86_64.AppImage\n");
    set_env("LARECOMP_UPDATE_SHA256SUMS", bad_sums.c_str());
    updater_start_install();
    s = wait_install_done();
    assert(s.install_state == UPD_INST_FAILED);
    assert(read_bytes(app_path) == "old-appimage");
    assert(!fs::exists(spaced_dir + "/.PokeV-Recomp.v0.2.0.part"));

    // 8c. Successful download, verify, stage & permission preservation
    set_env("LARECOMP_UPDATE_SHA256SUMS", sums_src.c_str());
    updater_start_install();
    s = wait_install_done();
    assert(s.install_state == UPD_INST_STAGED);
    assert(fs::exists(spaced_dir + "/.PokeV-Recomp.staged"));
    printf("ok   staged download, sha256 verification, low-disk & corrupt rejection\n");

    // 9. Atomic replacement, permissions, backup & rollback (Phase B)
    std::string staged_file = spaced_dir + "/.PokeV-Recomp.staged";
    std::string bak_file = app_path + ".bak";
    std::string apply_err;
    assert(updater_apply_staged_file(app_path, staged_file, bak_file, apply_err));
    assert(read_bytes(app_path) == "hello world");
    assert(!fs::exists(bak_file));
    struct stat ast{};
    assert(stat(app_path.c_str(), &ast) == 0 && (ast.st_mode & 0777) == 0751);

    // Rollback test: missing staged file leaves original intact
    assert(!updater_apply_staged_file(app_path, spaced_dir + "/nonexistent.staged", bak_file, apply_err));
    assert(read_bytes(app_path) == "hello world");
    assert(!fs::exists(bak_file));

    // Startup recovery test: missing target + existing .bak restores target, cleans .part, reads _updated.txt
    write_bytes(bak_file, "recovered-backup");
    unlink(app_path.c_str());
    write_bytes(spaced_dir + "/.PokeV-Recomp.stale.part", "stale");
    write_bytes(data_dir + "/cache/_updated.txt", "v0.2.0\n");
    updater_configure("v0.2.0", data_dir.c_str(), false, false);
    assert(read_bytes(app_path) == "recovered-backup");
    assert(!fs::exists(bak_file));
    assert(!fs::exists(spaced_dir + "/.PokeV-Recomp.stale.part"));
    assert(updater_get_snapshot().updated_from_tag == "v0.2.0");
    printf("ok   atomic swap, 0751 perm preservation, rollback & boot recovery\n");

    // 10. Concurrent updater lockfile rejection (live other PID blocks; dead PID cleaned)
    {
        std::string lock_file = spaced_dir + "/.PokeV-Recomp.update.lock";
        write_bytes(lock_file, std::to_string((long)getppid()) + "\n");
        updater_configure("v0.1.0", data_dir.c_str(), false, false);
        updater_start_check(true);
        wait_check_done();
        updater_start_install();
        s = wait_install_done();
        assert(s.install_state == UPD_INST_FAILED);
        assert(s.install_error.find("already in progress") != std::string::npos);
        unlink(lock_file.c_str());
        printf("ok   concurrent updater lockfile rejection\n");
    }

    // 11. Download cancellation + curl shutdown (Phase A + B)
    set_env("LARECOMP_UPDATE_APPIMAGE", "https://github.com/x/lin.AppImage");
    set_env("LARECOMP_UPDATE_SHA256SUMS", "https://github.com/x/SHA256SUMS");
    set_env("LARECOMP_CURL_BIN", getenv("LARECOMP_TEST_FAKE_CURL"));
    g_kill_count = 0;
    g_spawn_count = 0;
    updater_configure("v0.1.0", data_dir.c_str(), false, false);
    updater_start_check(true);
    wait_check_done();
    updater_start_install();
    for (int i = 0; i < 100 && g_spawn_count.load() == 0; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    updater_cancel_install();
    s = wait_install_done();
    assert(s.install_state == UPD_INST_CANCELLED);
    assert(g_kill_count.load() >= 1);
    updater_shutdown();
    updater_shutdown();
    printf("ok   download cancel & shutdown safety\n");
#endif

    // 12. Markdown parser & prerelease channel flag (UX polish)
    {
        std::string md =
            "# Release v0.1.1\n\n"
            "## What's New\n"
            "- Added **auto-update** with `SHA-256` verification\n"
            "- See [GitHub](https://github.com/didslima/PokeV-Recomp)\n\n"
            "> Saves and mods remain untouched.\n\n"
            "```\n./PokeV-Recomp.AppImage\n```\n";
        auto blocks = updater_parse_markdown(md);
        assert(blocks.size() == 6);
        assert(blocks[0].kind == UPD_MD_HEADING1);
        assert(blocks[0].spans.size() == 1 && blocks[0].spans[0].text == "Release v0.1.1");
        assert(blocks[1].kind == UPD_MD_HEADING2);
        assert(blocks[1].spans.size() == 1 && blocks[1].spans[0].text == "What's New");
        assert(blocks[2].kind == UPD_MD_BULLET);
        assert(blocks[2].spans[1].kind == UPD_SPAN_BOLD && blocks[2].spans[1].text == "auto-update");
        assert(blocks[2].spans[3].kind == UPD_SPAN_CODE && blocks[2].spans[3].text == "SHA-256");
        assert(blocks[3].kind == UPD_MD_BULLET);
        assert(blocks[3].spans[1].kind == UPD_SPAN_LINK);
        assert(blocks[3].spans[1].text == "GitHub");
        assert(blocks[3].spans[1].url == "https://github.com/didslima/PokeV-Recomp");
        assert(blocks[4].kind == UPD_MD_QUOTE);
        assert(blocks[4].spans[0].text == "Saves and mods remain untouched.");
        assert(blocks[5].kind == UPD_MD_CODE);
        assert(blocks[5].spans[0].text == "./PokeV-Recomp.AppImage");

        // Verify prerelease channel flag in snapshot (cross-platform)
        set_env("LARECOMP_NO_UPDATE_CHECK", nullptr);
        set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
        set_env("LARECOMP_UPDATE_JSON",
                R"([{"tag_name":"v0.1.1-rc.1","draft":false,"prerelease":true,"html_url":"https://github.com/x/rc1","body":"RC 1"}])");
        set_env("LARECOMP_UPDATE_PRERELEASE", "1");
        updater_configure("v0.1.0", nullptr, false, false);
        updater_start_check(true);
        UpdaterSnapshot ps = wait_check_done();
        assert(ps.is_prerelease_channel == true);
        set_env("LARECOMP_UPDATE_PRERELEASE", nullptr);
        set_env("LARECOMP_UPDATE_JSON",
                R"({"tag_name":"v0.2.0","draft":false,"prerelease":false,"html_url":"https://github.com/x/v020","body":"Stable"})");
        updater_configure("v0.1.0", nullptr, false, false);
        updater_start_check(true);
        ps = wait_check_done();
        assert(ps.is_prerelease_channel == false);
        set_env("LARECOMP_UPDATE_JSON", nullptr);
        set_env("LARECOMP_UPDATE_HTTP_STATUS", nullptr);
        printf("ok   markdown parser & prerelease channel flag\n");
    }

    printf("PASS: updater phase A + B checks\n");
    return 0;
}
'''


def test_external_helper_e2e():
    """End-to-end test of the external /bin/sh helper: waits for PID, swaps staged -> target, relaunches."""
    if not sys.platform.startswith("linux"):
        return
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp) / "Dir With Spaces & 'Quotes'"
        t.mkdir()
        target = t / "PokeV-Recomp.AppImage"
        staged = t / ".PokeV-Recomp.staged"
        backup = t / "PokeV-Recomp.AppImage.bak"
        lock = t / ".PokeV-Recomp.update.lock"
        note = t / "_updated.txt"
        ran_marker = t / "relaunched.txt"

        # Old AppImage is a dummy script; staged AppImage writes ran_marker when relaunched
        target.write_text("#!/bin/sh\nsleep 0.2\n", encoding="utf-8")
        target.chmod(0o755)
        staged.write_text(f"#!/bin/sh\necho relaunched > \"{ran_marker}\"\n", encoding="utf-8")
        staged.chmod(0o755)
        lock.write_text("999999\n", encoding="utf-8")

        dummy = subprocess.Popen([str(target)])
        helper = (
            'PID="$1"; TGT="$2"; STG="$3"; BAK="$4"; LCK="$5"; NOTE="$6"; TAG="$7"; MAXI="${8:-300}"; '
            'i=0; while kill -0 "$PID" 2>/dev/null && [ $i -lt "$MAXI" ]; do sleep 0.05; i=$((i+1)); done; '
            'if kill -0 "$PID" 2>/dev/null; then rm -f "$STG" "$LCK"; exit 1; fi; '
            'rm -f "$BAK"; '
            'if [ -e "$TGT" ]; then '
            '  if ! ln "$TGT" "$BAK" 2>/dev/null && ! cp -p "$TGT" "$BAK" 2>/dev/null; then '
            '    rm -f "$STG" "$LCK"; exit 1; '
            '  fi; '
            'fi; '
            'if ! mv -f "$STG" "$TGT"; then '
            '  if [ ! -e "$TGT" ] && [ -e "$BAK" ]; then mv -f "$BAK" "$TGT" 2>/dev/null || true; fi; '
            '  rm -f "$BAK" "$STG" "$LCK"; exit 1; '
            'fi; '
            'rm -f "$BAK" "$LCK"; '
            'if [ -n "$NOTE" ]; then mkdir -p "$(dirname "$NOTE")" 2>/dev/null || true; printf "%s" "$TAG" > "$NOTE" 2>/dev/null || true; fi; '
            'unset APPIMAGE APPDIR ARGV0 OWD 2>/dev/null || true; '
            'exec "$TGT" >/dev/null 2>&1'
        )
        p = subprocess.Popen(
            ["/bin/sh", "-c", helper, "pokev-updater",
             str(dummy.pid), str(target), str(staged), str(backup), str(lock), str(note), "v0.2.0"]
        )
        dummy.wait(timeout=5)
        p.wait(timeout=5)
        assert p.returncode == 0
        assert ran_marker.read_text(encoding="utf-8").strip() == "relaunched"
        assert note.read_text(encoding="utf-8") == "v0.2.0"
        assert not backup.exists()
        assert not staged.exists()
        assert not lock.exists()
        print("ok   external helper e2e (spaces, wait PID, swap, relaunch, cleanup)")

        # Timeout test: PID stays alive past MAXI=4 -> abort without replacing target or relaunching
        target.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
        target.chmod(0o755)
        staged.write_text(f"#!/bin/sh\necho should-not-run > \"{ran_marker}\"\n", encoding="utf-8")
        staged.chmod(0o755)
        lock.write_text("999999\n", encoding="utf-8")
        ran_marker.unlink()
        stubborn = subprocess.Popen([str(target)])
        try:
            p2 = subprocess.run(
                ["/bin/sh", "-c", helper, "pokev-updater",
                 str(stubborn.pid), str(target), str(staged), str(backup), str(lock), str(note), "v0.2.0", "4"],
                timeout=5,
            )
            assert p2.returncode == 1
            assert "sleep 5" in target.read_text(encoding="utf-8")
            assert not ran_marker.exists()
            assert not backup.exists()
            assert not staged.exists()
            assert not lock.exists()
        finally:
            stubborn.terminate()
            stubborn.wait(timeout=5)
        print("ok   external helper timeout preserves original AppImage and does not relaunch")


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
        env["LARECOMP_TEST_TMPDIR"] = str(t)
        env["LARECOMP_TEST_HELLO_SHA"] = hashlib.sha256(b"hello world").hexdigest()
        env["PATH"] = f"{t}{os.pathsep}{env.get('PATH', '')}"
        subprocess.run([str(exe)], cwd=ROOT, env=env, check=True)

    test_external_helper_e2e()


if __name__ == "__main__":
    main()
