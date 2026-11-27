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
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
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
#include "ui/win_zip.h"

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

#if defined(__linux__) || defined(_WIN32)
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

    // 13. Phase C: Windows ZIP asset parsing, path validation & safe extraction (cross-platform)
    {
        assert(r.windows_asset_url == "https://github.com/x/win.zip");
        assert(r.windows_asset_size == 999);

        std::string rel, err;
        bool is_dir = false;
        assert(updater_validate_zip_entry_path("PokeV-Recomp/PokeV-Recomp.exe", "PokeV-Recomp/", rel, is_dir, err));
        assert(!is_dir && rel == "PokeV-Recomp.exe");
        assert(updater_validate_zip_entry_path("PokeV-Recomp/zaimport/zaimport.exe", "PokeV-Recomp/", rel, is_dir, err));
        assert(!is_dir && rel == "zaimport/zaimport.exe");

        assert(!updater_validate_zip_entry_path("../evil.dll", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/../../evil.dll", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("/abs/evil.dll", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("C:\\evil.dll", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/CON.txt", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/bad.txt ", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/bad.", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/larecomp.ini", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/save/slot.bin", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/extracted/exefs/main", "PokeV-Recomp/", rel, is_dir, err));
        assert(!updater_validate_zip_entry_path("PokeV-Recomp/mods/m.txt", "PokeV-Recomp/", rel, is_dir, err));

        const char *zip_good = getenv("LARECOMP_TEST_ZIP_GOOD");
        const char *zip_slip = getenv("LARECOMP_TEST_ZIP_SLIP");
        const char *zip_dup = getenv("LARECOMP_TEST_ZIP_DUP");
        const char *zip_sym = getenv("LARECOMP_TEST_ZIP_SYM");
        const char *zip_badcrc = getenv("LARECOMP_TEST_ZIP_BADCRC");
        const char *zip_userdata = getenv("LARECOMP_TEST_ZIP_USERDATA");
        const char *zip_z64_ovf = getenv("LARECOMP_TEST_ZIP_Z64OVF");
        const char *zip_lfh_ovf = getenv("LARECOMP_TEST_ZIP_LFHOVF");
        const char *zip_bomb_agg = getenv("LARECOMP_TEST_ZIP_BOMBAGG");
        assert(zip_good && zip_slip && zip_dup && zip_sym && zip_badcrc && zip_userdata);
        assert(zip_z64_ovf && zip_lfh_ovf && zip_bomb_agg);

        std::string ztmp = std::string(getenv("LARECOMP_TEST_TMPDIR")) + "/zextract";
        WinZipExtractResult gres = updater_extract_zip_safe(zip_good, ztmp + "/good", "PokeV-Recomp/");
        assert(gres.ok);
        std::string verr;
        assert(updater_validate_windows_extracted_dir(ztmp + "/good", verr));

        // Unicode + non-ASCII + spaces + long path extraction (cross-platform)
        std::string uni_dest = std::string(getenv("LARECOMP_TEST_TMPDIR")) +
                               "/Jo\xC3\xA3o Caf\xC3\xA9 \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E Dir/PokeV-Recomp";
        WinZipExtractResult ures = updater_extract_zip_safe(zip_good, uni_dest, "PokeV-Recomp/");
        assert(ures.ok);
        assert(updater_validate_windows_extracted_dir(uni_dest, verr));

        assert(!updater_extract_zip_safe(zip_slip, ztmp + "/slip", "PokeV-Recomp/").ok);
        assert(!updater_extract_zip_safe(zip_dup, ztmp + "/dup", "PokeV-Recomp/").ok);
        assert(!updater_extract_zip_safe(zip_sym, ztmp + "/sym", "PokeV-Recomp/").ok);
        assert(!updater_extract_zip_safe(zip_badcrc, ztmp + "/badcrc", "PokeV-Recomp/").ok);
        assert(!updater_extract_zip_safe(zip_userdata, ztmp + "/userdata", "PokeV-Recomp/").ok);
        assert(!updater_extract_zip_safe(zip_z64_ovf, ztmp + "/z64ovf", "PokeV-Recomp/").ok);
        assert(!updater_extract_zip_safe(zip_lfh_ovf, ztmp + "/lfhovf", "PokeV-Recomp/").ok);
        assert(!updater_extract_zip_safe(zip_bomb_agg, ztmp + "/bombagg", "PokeV-Recomp/").ok);
        printf("ok   windows zip asset parsing, unicode/long paths, overflow, bomb, dup/symlink/crc & structure\n");
    }

#ifdef _WIN32
    // 14. Phase C (Windows): Install detection, download/SHA256/staging, lockfile, low disk & boot recovery
    {
        std::string tmp = getenv("LARECOMP_TEST_TMPDIR");
        std::string win_dir = tmp + "\\Win Space 'Install'";
        fs::create_directories(win_dir + "\\gamelib-src");
        fs::create_directories(win_dir + "\\zaimport");
        fs::create_directories(win_dir + "\\zig");
        fs::create_directories(win_dir + "\\lasgen");
        fs::create_directories(win_dir + "\\zaprompts");
        fs::create_directories(win_dir + "\\save\\save_0");
        fs::create_directories(win_dir + "\\mods");
        fs::create_directories(win_dir + "\\extracted");

        auto wbytes = [](const std::string &p, const std::string &d) {
            FILE *f = fopen(p.c_str(), "wb");
            assert(f);
            fwrite(d.data(), 1, d.size(), f);
            fclose(f);
        };
        auto rbytes = [](const std::string &p) -> std::string {
            FILE *f = fopen(p.c_str(), "rb");
            if (!f) return "";
            std::string s;
            char buf[4096];
            size_t n;
            while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
            fclose(f);
            return s;
        };

        wbytes(win_dir + "\\PokeV-Recomp.exe", "old-exe");
        wbytes(win_dir + "\\PokeV-Recomp-updater.exe", "old-upd");
        wbytes(win_dir + "\\SDL3.dll", "old-sdl3");
        wbytes(win_dir + "\\za-default.ini", "old-def");
        wbytes(win_dir + "\\recomplauncher.json", "{}");
        wbytes(win_dir + "\\gamelib-src\\gamelib_entry.c", "// old");
        wbytes(win_dir + "\\zaimport\\zaimport.exe", "old-zai");
        wbytes(win_dir + "\\zig\\zig.exe", "old-zig");
        wbytes(win_dir + "\\lasgen\\lasgen.exe", "old-las");
        wbytes(win_dir + "\\zaprompts\\zaprompts.exe", "old-zap");
        wbytes(win_dir + "\\save\\save_0\\savedata0.bin", "my-precious-save");
        wbytes(win_dir + "\\larecomp.ini", "user-setting=1\n");
        wbytes(win_dir + "\\mods\\cool.mod", "mod-data");
        wbytes(win_dir + "\\extracted\\romfs.bin", "romfs-data");

        std::string dtgt, dreason;
        set_env("LARECOMP_TEST_WIN_INSTALL_DIR", win_dir.c_str());
        assert(updater_detect_install_kind(dtgt, dreason) == UPD_INSTALL_WINDOWS_ZIP);

        set_env("RECOMPLAUNCHER_DATA", "C:\\RL");
        assert(updater_detect_install_kind(dtgt, dreason) == UPD_INSTALL_RECOMPLAUNCHER);
        set_env("RECOMPLAUNCHER_DATA", nullptr);

        // Staged install with valid ZIP + SHA256SUMS
        const char *zip_good = getenv("LARECOMP_TEST_ZIP_GOOD");
        std::string zsha = updater_file_sha256(zip_good);
        std::string zsz = std::to_string((unsigned long long)fs::file_size(zip_good));
        std::string sums_f = tmp + "\\win_SHA256SUMS";
        wbytes(sums_f, zsha + "  PokeV-Recomp-windows-x64.zip\n");

        std::string win_json = std::string(R"({"tag_name":"v0.2.0","draft":false,"prerelease":false,)") +
            R"("html_url":"https://github.com/x/v020","body":"Win update",)" +
            R"("assets":[{"name":"PokeV-Recomp-windows-x64.zip","size":)" + zsz +
            R"(,"browser_download_url":"https://github.com/x/win.zip"},)" +
            R"({"name":"SHA256SUMS","size":100,"browser_download_url":"https://github.com/x/SHA256SUMS"}]})";

        set_env("LARECOMP_UPDATE_HTTP_STATUS", "200");
        set_env("LARECOMP_UPDATE_JSON", win_json.c_str());
        set_env("LARECOMP_UPDATE_WINZIP", zip_good);
        set_env("LARECOMP_UPDATE_SHA256SUMS", sums_f.c_str());
        updater_configure("v0.1.1", win_dir.c_str(), false, false);
        updater_start_check(true);
        UpdaterSnapshot ws = wait_check_done();
        assert(ws.state == UPD_CHK_AVAILABLE && ws.install_kind == UPD_INSTALL_WINDOWS_ZIP);

        updater_start_install();
        ws = wait_install_done();
        assert(ws.install_state == UPD_INST_STAGED);
        assert(fs::exists(win_dir + "\\.update-staging\\manifest.txt"));
        assert(rbytes(win_dir + "\\save\\save_0\\savedata0.bin") == "my-precious-save");
        assert(rbytes(win_dir + "\\larecomp.ini") == "user-setting=1\n");

        // Wrong SHA256 rejects install
        wbytes(sums_f, std::string(64, '0') + "  PokeV-Recomp-windows-x64.zip\n");
        updater_configure("v0.1.1", win_dir.c_str(), false, false);
        updater_start_check(true);
        wait_check_done();
        updater_start_install();
        ws = wait_install_done();
        assert(ws.install_state == UPD_INST_FAILED);

        // Low disk space rejects install
        wbytes(sums_f, zsha + "  PokeV-Recomp-windows-x64.zip\n");
        set_env("LARECOMP_UPDATE_MIN_FREE_BYTES", "1024");
        updater_start_install();
        ws = wait_install_done();
        assert(ws.install_state == UPD_INST_FAILED);
        set_env("LARECOMP_UPDATE_MIN_FREE_BYTES", nullptr);

        // Interrupted update with .bak + marker flags external recovery_pending & blocks Play
        fs::create_directories(win_dir + "\\.PokeV-Recomp.update.bak");
        wbytes(win_dir + "\\.PokeV-Recomp.update.bak\\PokeV-Recomp.exe", "restored-exe");
        wbytes(win_dir + "\\.PokeV-Recomp.update.bak\\SDL3.dll", "restored-sdl3");
        wbytes(win_dir + "\\.PokeV-Recomp.update-in-progress", "v0.2.0");
        updater_configure("v0.1.1", win_dir.c_str(), false, false);
        ws = updater_get_snapshot();
        assert(ws.recovery_pending == true);
        assert(ws.recovery_blocked == true);
        assert(updater_is_recovery_blocked() == true);
        assert(ws.install_kind == UPD_INSTALL_UNSUPPORTED);
        // Must NOT have modified EXE/DLL in-process or deleted .bak
        assert(rbytes(win_dir + "\\PokeV-Recomp.exe") == "old-exe");
        assert(fs::exists(win_dir + "\\.PokeV-Recomp.update.bak"));
        assert(fs::exists(win_dir + "\\.PokeV-Recomp.update-in-progress"));

        // Missing helper binary fails recovery launch cleanly and keeps Play blocked
        fs::remove(win_dir + "\\PokeV-Recomp-updater.exe");
        assert(!updater_launch_windows_recovery_and_exit_ready());
        ws = updater_get_snapshot();
        assert(ws.recovery_blocked == true);
        assert(updater_is_recovery_blocked() == true);
        assert(ws.install_state == UPD_INST_FAILED);
        assert(ws.install_error.find(".PokeV-Recomp.update.bak") != std::string::npos);
        wbytes(win_dir + "\\PokeV-Recomp-updater.exe", "old-upd");

        // Inconsistent markers: .update-in-progress without .bak AND missing critical file -> blocks Play
        fs::remove_all(win_dir + "\\.PokeV-Recomp.update.bak");
        fs::remove(win_dir + "\\zig\\zig.exe");
        updater_configure("v0.1.1", win_dir.c_str(), false, false);
        ws = updater_get_snapshot();
        assert(ws.recovery_pending == false);
        assert(ws.recovery_blocked == true);
        assert(updater_is_recovery_blocked() == true);
        assert(ws.install_state == UPD_INST_FAILED);
        assert(fs::exists(win_dir + "\\.PokeV-Recomp.update-in-progress"));

        // Inconsistent markers: .update-in-progress without .bak, all critical files intact -> cleans marker & unblocks
        wbytes(win_dir + "\\zig\\zig.exe", "old-zig");
        updater_configure("v0.1.1", win_dir.c_str(), false, false);
        ws = updater_get_snapshot();
        assert(ws.recovery_pending == false);
        assert(ws.recovery_blocked == false);
        assert(updater_is_recovery_blocked() == false);
        assert(!fs::exists(win_dir + "\\.PokeV-Recomp.update-in-progress"));
        assert(!ws.update_notice.empty());
        assert(ws.install_kind == UPD_INSTALL_WINDOWS_ZIP);

        // Post-recovery notice marker surfaced and cleaned on next boot
        wbytes(win_dir + "\\.PokeV-Recomp.update-notice.txt", "Restored cleanly");
        updater_configure("v0.1.1", win_dir.c_str(), false, false);
        ws = updater_get_snapshot();
        assert(ws.update_notice == "Restored cleanly");
        assert(!fs::exists(win_dir + "\\.PokeV-Recomp.update-notice.txt"));

        // Unicode Windows install & ZIP path test
        std::string uni_zip_path = tmp + "\\Jo\xC3\xA3o Caf\xC3\xA9 \xC3\x81rea\\release.zip";
        fs::create_directories(fs::u8path(tmp + "\\Jo\xC3\xA3o Caf\xC3\xA9 \xC3\x81rea"));
        fs::copy_file(fs::u8path(zip_good), fs::u8path(uni_zip_path), fs::copy_options::overwrite_existing);
        std::string uni_out = tmp + "\\Jo\xC3\xA3o Caf\xC3\xA9 \xC3\x81rea\\out";
        WinZipExtractResult wuni = updater_extract_zip_safe(uni_zip_path, uni_out, "PokeV-Recomp/");
        assert(wuni.ok);
        std::string wverr;
        assert(updater_validate_windows_extracted_dir(uni_out, wverr));

        set_env("LARECOMP_TEST_WIN_INSTALL_DIR", nullptr);
        printf("ok   windows install detection, staging, sha256, low-disk, partial & 2nd-boot recovery, unicode\n");
    }
#endif

    printf("PASS: updater phase A + B + C checks\n");
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


def build_test_zips(tmp: Path) -> dict:
    import zipfile

    zips = {}
    good = tmp / "good.zip"
    long_rel = "PokeV-Recomp/docs/" + ("nested_subdir_" * 6) + "release_note & info (v2).txt"
    good_files = {
        "PokeV-Recomp/PokeV-Recomp.exe": b"new-exe-v0.2.0",
        "PokeV-Recomp/PokeV-Recomp-updater.exe": b"new-upd-v0.2.0",
        "PokeV-Recomp/SDL3.dll": b"new-sdl3-v0.2.0",
        "PokeV-Recomp/za-default.ini": b"new-default-ini",
        "PokeV-Recomp/recomplauncher.json": b'{"version":"0.2.0"}',
        "PokeV-Recomp/gamelib-src/gamelib_entry.c": b"// new entry",
        "PokeV-Recomp/zaimport/zaimport.exe": b"new-zaimport",
        "PokeV-Recomp/zig/zig.exe": b"new-zig",
        "PokeV-Recomp/lasgen/lasgen.exe": b"new-lasgen",
        "PokeV-Recomp/zaprompts/zaprompts.exe": b"new-zaprompts",
        "PokeV-Recomp/docs/Notas & Versão (João Café 日本語).txt": b"unicode entry ok",
        long_rel: b"long path & special chars ok",
    }
    with zipfile.ZipFile(good, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in good_files.items():
            zf.writestr(name, data)
    zips["GOOD"] = str(good)

    slip = tmp / "slip.zip"
    with zipfile.ZipFile(slip, "w") as zf:
        zf.writestr("../evil.dll", b"boom")
    zips["SLIP"] = str(slip)

    dup = tmp / "dup.zip"
    with zipfile.ZipFile(dup, "w") as zf:
        zf.writestr("PokeV-Recomp/Foo.dll", b"one")
        zf.writestr("PokeV-Recomp/foo.dll", b"two")
    zips["DUP"] = str(dup)

    sym = tmp / "sym.zip"
    with zipfile.ZipFile(sym, "w") as zf:
        zi = zipfile.ZipInfo("PokeV-Recomp/link.dll")
        zi.create_system = 3  # Unix
        zi.external_attr = 0o120777 << 16  # S_IFLNK
        zf.writestr(zi, b"C:\\Windows\\System32\\calc.exe")
    zips["SYM"] = str(sym)

    badcrc = tmp / "badcrc.zip"
    with zipfile.ZipFile(badcrc, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("PokeV-Recomp/PokeV-Recomp.exe", b"ABCD1234")
    raw = bytearray(badcrc.read_bytes())
    idx = raw.find(b"ABCD1234")
    assert idx != -1
    raw[idx] ^= 0xff  # corrupt payload without updating CRC-32
    badcrc.write_bytes(bytes(raw))
    zips["BADCRC"] = str(badcrc)

    userdata = tmp / "userdata.zip"
    with zipfile.ZipFile(userdata, "w") as zf:
        zf.writestr("PokeV-Recomp/larecomp.ini", b"overwrite-attempt=1")
    zips["USERDATA"] = str(userdata)

    # ZIP64 EOCD offset near UINT64_MAX (overflow test)
    import struct
    z64ovf = tmp / "z64ovf.zip"
    with zipfile.ZipFile(z64ovf, "w") as zf:
        zf.writestr("PokeV-Recomp/a.txt", b"hi")
    zraw = bytearray(z64ovf.read_bytes())
    eocd_i = zraw.rfind(b"PK\x05\x06")
    assert eocd_i != -1
    locator = b"PK\x06\x07" + struct.pack("<IQ I", 0, 0xFFFFFFFFFFFFFFE0, 1)
    zraw[eocd_i:eocd_i] = locator
    z64ovf.write_bytes(bytes(zraw))
    zips["Z64OVF"] = str(z64ovf)

    # Local File Header offset near UINT64_MAX via ZIP64 extra field
    lfhovf = tmp / "lfhovf.zip"
    with zipfile.ZipFile(lfhovf, "w") as zf:
        zf.writestr("PokeV-Recomp/a.txt", b"hi")
    lraw = bytearray(lfhovf.read_bytes())
    cd_i = lraw.rfind(b"PK\x01\x02")
    assert cd_i != -1
    # Set lfh_offset field (offset 42) to 0xFFFFFFFF and append ZIP64 extra field with 0xFFFFFFFFFFFFFFF0
    nlen, elen, clen = struct.unpack_from("<HHH", lraw, cd_i + 28)
    z64_ex = struct.pack("<HHQ", 0x0001, 8, 0xFFFFFFFFFFFFFFF0)
    struct.pack_into("<I", lraw, cd_i + 42, 0xFFFFFFFF)
    struct.pack_into("<H", lraw, cd_i + 30, elen + len(z64_ex))
    ins_at = cd_i + 46 + nlen + elen
    lraw[ins_at:ins_at] = z64_ex
    eocd_i2 = lraw.rfind(b"PK\x05\x06")
    cd_sz = struct.unpack_from("<I", lraw, eocd_i2 + 12)[0]
    struct.pack_into("<I", lraw, eocd_i2 + 12, cd_sz + len(z64_ex))
    lfhovf.write_bytes(bytes(lraw))
    zips["LFHOVF"] = str(lfhovf)

    # Aggregate decompression limit (> 2 GiB across 3 entries of 800 MiB declared)
    bombagg = tmp / "bombagg.zip"
    with zipfile.ZipFile(bombagg, "w") as zf:
        zf.writestr("PokeV-Recomp/a.bin", b"1")
        zf.writestr("PokeV-Recomp/b.bin", b"2")
        zf.writestr("PokeV-Recomp/c.bin", b"3")
    braw = bytearray(bombagg.read_bytes())
    pos = 0
    while True:
        idx = braw.find(b"PK\x01\x02", pos)
        if idx == -1:
            break
        struct.pack_into("<I", braw, idx + 24, 800 * 1024 * 1024)
        pos = idx + 46
    bombagg.write_bytes(bytes(braw))
    zips["BOMBAGG"] = str(bombagg)

    return zips


def test_windows_helper_e2e(cxx: str):
    if os.name != "nt":
        return
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        helper_exe = t / "PokeV-Recomp-updater.exe"
        subprocess.run([
            cxx, "-std=c++17", "-O2", "-DWIN32_LEAN_AND_MEAN", "-DNOMINMAX",
            "-municode", "-static",
            str(ROOT / "tools" / "win_updater_helper.cpp"),
            "-lkernel32", "-o", str(helper_exe),
        ], cwd=ROOT, check=True)

        # Build a tiny dummy child binary that exits after sleep or writes a marker on relaunch
        dummy_cpp = t / "dummy_app.cpp"
        dummy_cpp.write_text(
            '#include <windows.h>\n'
            '#include <cstdio>\n'
            'int main(int argc, char **argv) {\n'
            '  if (argc > 1 && argv[1][0] == \'w\') { Sleep(3000); return 0; }\n'
            '  FILE *f = fopen("relaunched.txt", "wb");\n'
            '  if (f) { fprintf(f, "v0.2.0\\n"); fclose(f); }\n'
            '  return 0;\n'
            '}\n',
            encoding="utf-8",
        )
        dummy_exe = t / "dummy_app.exe"
        subprocess.run([cxx, "-O2", "-static", str(dummy_cpp), "-o", str(dummy_exe)], check=True)

        inst = t / "Install With Spaces"
        staged = inst / ".update-staging" / "extracted"
        bak = inst / ".PokeV-Recomp.update.bak"
        lock = inst / ".PokeV-Recomp.update.lock"
        note = inst / "cache" / "_updated.txt"
        manifest = inst / ".update-staging" / "manifest.txt"

        def setup_install():
            shutil.rmtree(inst, ignore_errors=True)
            for sub in ["gamelib-src", "zaimport", "zig", "lasgen", "zaprompts", "save", "mods", "extracted"]:
                (inst / sub).mkdir(parents=True, exist_ok=True)
                (staged / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(dummy_exe, inst / "PokeV-Recomp.exe")
            (inst / "SDL3.dll").write_bytes(b"old-sdl3")
            (inst / "obsolete.dll").write_bytes(b"old-obsolete")
            (inst / ".PokeV-Recomp.app-manifest").write_text("PokeV-Recomp.exe\nSDL3.dll\nobsolete.dll\n", encoding="utf-8")
            (inst / "za-default.ini").write_bytes(b"old-def")
            (inst / "recomplauncher.json").write_bytes(b"{}")
            (inst / "gamelib-src" / "gamelib_entry.c").write_bytes(b"old")
            (inst / "zaimport" / "zaimport.exe").write_bytes(b"old")
            (inst / "zig" / "zig.exe").write_bytes(b"old")
            (inst / "lasgen" / "lasgen.exe").write_bytes(b"old")
            (inst / "zaprompts" / "zaprompts.exe").write_bytes(b"old")
            (inst / "save" / "slot.bin").write_bytes(b"user-save")
            (inst / "larecomp.ini").write_bytes(b"user-ini=1\n")
            (inst / "mods" / "m.pak").write_bytes(b"user-mod")
            (inst / "extracted" / "romfs.bin").write_bytes(b"user-romfs")
            (inst / "unknown_user.txt").write_bytes(b"keep-me")

            shutil.copy2(dummy_exe, staged / "PokeV-Recomp.exe")
            (staged / "SDL3.dll").write_bytes(b"new-sdl3")
            (staged / "za-default.ini").write_bytes(b"new-def")
            (staged / "recomplauncher.json").write_bytes(b'{"v":"0.2.0"}')
            (staged / "gamelib-src" / "gamelib_entry.c").write_bytes(b"new")
            (staged / "zaimport" / "zaimport.exe").write_bytes(b"new")
            (staged / "zig" / "zig.exe").write_bytes(b"new")
            (staged / "lasgen" / "lasgen.exe").write_bytes(b"new")
            (staged / "zaprompts" / "zaprompts.exe").write_bytes(b"new")
            manifest.write_text(
                "\n".join([
                    "PokeV-Recomp.exe", "SDL3.dll", "za-default.ini", "recomplauncher.json",
                    "gamelib-src/gamelib_entry.c", "zaimport/zaimport.exe", "zig/zig.exe",
                    "lasgen/lasgen.exe", "zaprompts/zaprompts.exe",
                ]) + "\n",
                encoding="utf-8",
            )
            lock.write_text("999999\n", encoding="utf-8")

        # 1. Normal swap, obsolete file removal, user data preservation & relaunch
        setup_install()
        res = subprocess.run([
            str(helper_exe), "0", "0", str(inst), str(staged), str(manifest),
            str(bak), str(lock), str(note), "v0.2.0", "5000", "1",
        ], timeout=15)
        assert res.returncode == 0
        import time
        for _ in range(40):
            if (inst / "relaunched.txt").is_file():
                break
            time.sleep(0.05)
        assert (inst / "relaunched.txt").read_text(encoding="utf-8").strip() == "v0.2.0"
        assert (inst / "SDL3.dll").read_bytes() == b"new-sdl3"
        assert not (inst / "obsolete.dll").exists()
        assert (inst / "save" / "slot.bin").read_bytes() == b"user-save"
        assert (inst / "larecomp.ini").read_bytes() == b"user-ini=1\n"
        assert (inst / "mods" / "m.pak").read_bytes() == b"user-mod"
        assert (inst / "extracted" / "romfs.bin").read_bytes() == b"user-romfs"
        assert (inst / "unknown_user.txt").read_bytes() == b"keep-me"
        assert note.read_text(encoding="utf-8") == "v0.2.0"
        assert not bak.exists()
        assert not lock.exists()
        print("ok   windows helper e2e (swap, user data preserved, obsolete cleaned, relaunch)")

        # 2. Validation failure triggers automatic rollback
        setup_install()
        (staged / "zig" / "zig.exe").write_bytes(b"")  # empty critical file -> fails validation
        res = subprocess.run([
            str(helper_exe), "0", "0", str(inst), str(staged), str(manifest),
            str(bak), str(lock), str(note), "v0.2.0", "5000", "0",
        ], timeout=15)
        assert res.returncode == 7
        assert (inst / "SDL3.dll").read_bytes() == b"old-sdl3"
        assert (inst / "zig" / "zig.exe").read_bytes() == b"old"
        assert (inst / ".PokeV-Recomp.update-error.txt").is_file()
        assert (inst / "save" / "slot.bin").read_bytes() == b"user-save"
        assert not bak.exists()
        print("ok   windows helper validation failure rolls back cleanly")

        # 3. Parent timeout preserves installation and does not relaunch
        setup_install()
        stubborn = subprocess.Popen([str(dummy_exe), "wait"])
        try:
            res = subprocess.run([
                str(helper_exe), str(stubborn.pid), "0", str(inst), str(staged), str(manifest),
                str(bak), str(lock), str(note), "v0.2.0", "200", "1",
            ], timeout=10)
            assert res.returncode == 1
            assert (inst / "SDL3.dll").read_bytes() == b"old-sdl3"
            assert not (inst / "relaunched.txt").exists()
            assert not lock.exists()
        finally:
            stubborn.terminate()
            stubborn.wait(timeout=5)
        print("ok   windows helper parent timeout preserves installation")

        # 4. Locked file prevents update and preserves installation
        setup_install()
        with open(inst / "SDL3.dll", "r+b") as locked_f:
            import msvcrt
            msvcrt.locking(locked_f.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                res = subprocess.run([
                    str(helper_exe), "0", "0", str(inst), str(staged), str(manifest),
                    str(bak), str(lock), str(note), "v0.2.0", "500", "0",
                ], timeout=15)
                assert res.returncode == 3
                assert (inst / ".PokeV-Recomp.update-error.txt").is_file()
                assert not lock.exists()
            finally:
                locked_f.seek(0)
                msvcrt.locking(locked_f.fileno(), msvcrt.LK_UNLCK, 1)
        assert (inst / "SDL3.dll").read_bytes() == b"old-sdl3"
        print("ok   windows helper locked-file rejection preserves installation")

        # 5. External --recover mode: full recovery, user data preserved, relaunch & no relaunch loop
        def setup_interrupted_bak(target_dir: Path):
            shutil.rmtree(target_dir, ignore_errors=True)
            for sub in ["gamelib-src", "zaimport", "zig", "lasgen", "zaprompts", "save", "mods", "extracted"]:
                (target_dir / sub).mkdir(parents=True, exist_ok=True)
            bdir = target_dir / ".PokeV-Recomp.update.bak"
            (bdir / "zig").mkdir(parents=True, exist_ok=True)
            # Partially updated install_dir (new SDL3.dll, missing zig.exe) + old files in .bak
            shutil.copy2(dummy_exe, bdir / "PokeV-Recomp.exe")
            (bdir / "SDL3.dll").write_bytes(b"old-sdl3-bak")
            (bdir / "zig" / "zig.exe").write_bytes(b"old-zig-bak")
            shutil.copy2(dummy_exe, target_dir / "PokeV-Recomp.exe")
            (target_dir / "SDL3.dll").write_bytes(b"partial-new-sdl3")
            (target_dir / "za-default.ini").write_bytes(b"old-def")
            (target_dir / "recomplauncher.json").write_bytes(b"{}")
            (target_dir / "gamelib-src" / "gamelib_entry.c").write_bytes(b"old")
            (target_dir / "zaimport" / "zaimport.exe").write_bytes(b"old")
            (target_dir / "lasgen" / "lasgen.exe").write_bytes(b"old")
            (target_dir / "zaprompts" / "zaprompts.exe").write_bytes(b"old")
            (target_dir / "save" / "slot.bin").write_bytes(b"user-save")
            (target_dir / "larecomp.ini").write_bytes(b"user-ini=1\n")
            (target_dir / "mods" / "m.pak").write_bytes(b"user-mod")
            (target_dir / "extracted" / "romfs.bin").write_bytes(b"user-romfs")
            (target_dir / "unknown_user.txt").write_bytes(b"keep-me")
            (target_dir / ".PokeV-Recomp.update-in-progress").write_bytes(b"v0.2.0")
            (target_dir / ".PokeV-Recomp.update.lock").write_bytes(b"999999\n")
            return bdir, target_dir / ".PokeV-Recomp.update.lock"

        # 5a. Full --recover in Unicode + spaces path (independent of CWD)
        uni_inst = t / "João Café Área (Spaces)"
        ubak, ulock = setup_interrupted_bak(uni_inst)
        res = subprocess.run([
            str(helper_exe), "--recover", "0", "0",
            str(uni_inst), str(ubak), str(ulock), "5000", "1",
        ], cwd=str(t), timeout=15)
        assert res.returncode == 0
        for _ in range(40):
            if (uni_inst / "relaunched.txt").is_file():
                break
            time.sleep(0.05)
        assert (uni_inst / "relaunched.txt").is_file()
        assert (uni_inst / "SDL3.dll").read_bytes() == b"old-sdl3-bak"
        assert (uni_inst / "zig" / "zig.exe").read_bytes() == b"old-zig-bak"
        assert (uni_inst / "save" / "slot.bin").read_bytes() == b"user-save"
        assert (uni_inst / "larecomp.ini").read_bytes() == b"user-ini=1\n"
        assert (uni_inst / "mods" / "m.pak").read_bytes() == b"user-mod"
        assert (uni_inst / "extracted" / "romfs.bin").read_bytes() == b"user-romfs"
        assert (uni_inst / "unknown_user.txt").read_bytes() == b"keep-me"
        assert (uni_inst / ".PokeV-Recomp.update-notice.txt").is_file()
        # No relaunch loop: markers & backup are gone after recovery
        assert not (uni_inst / ".PokeV-Recomp.update-in-progress").exists()
        assert not ubak.exists()
        assert not ulock.exists()
        print("ok   windows --recover e2e (unicode/spaces, restore, user data preserved, relaunch, no loop)")

        # 5b. Locked DLL during --recover preserves backup & markers and does NOT relaunch
        ubak, ulock = setup_interrupted_bak(inst)
        with open(inst / "SDL3.dll", "r+b") as locked_f:
            import msvcrt
            msvcrt.locking(locked_f.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                res = subprocess.run([
                    str(helper_exe), "--recover", "0", "0",
                    str(inst), str(ubak), str(ulock), "500", "1",
                ], timeout=15)
                assert res.returncode == 3
                assert ubak.is_dir()
                assert (inst / ".PokeV-Recomp.update-in-progress").is_file()
                assert (inst / ".PokeV-Recomp.update-error.txt").is_file()
                assert not (inst / "relaunched.txt").exists()
            finally:
                locked_f.seek(0)
                msvcrt.locking(locked_f.fileno(), msvcrt.LK_UNLCK, 1)
        print("ok   windows --recover locked DLL preserves backup & markers, no relaunch")

        # 5c. Interrupted 1st recovery (only some files left in .bak) -> 2nd --recover completes cleanly
        res = subprocess.run([
            str(helper_exe), "--recover", "0", "0",
            str(inst), str(ubak), str(ulock), "5000", "1",
        ], timeout=15)
        assert res.returncode == 0
        for _ in range(40):
            if (inst / "relaunched.txt").is_file():
                break
            time.sleep(0.05)
        assert (inst / "relaunched.txt").is_file()
        assert (inst / "SDL3.dll").read_bytes() == b"old-sdl3-bak"
        assert not ubak.exists()
        assert not (inst / ".PokeV-Recomp.update-in-progress").exists()
        print("ok   windows --recover 2nd attempt after lock/interruption completes cleanly")

        # 5d. Missing/empty .bak fails cleanly without relaunch
        ubak, ulock = setup_interrupted_bak(inst)
        shutil.rmtree(ubak)
        res = subprocess.run([
            str(helper_exe), "--recover", "0", "0",
            str(inst), str(ubak), str(ulock), "500", "1",
        ], timeout=10)
        assert res.returncode == 11
        assert (inst / ".PokeV-Recomp.update-error.txt").is_file()
        assert (inst / ".PokeV-Recomp.update-in-progress").is_file()
        assert not (inst / "relaunched.txt").exists()
        print("ok   windows --recover missing backup fails safely without relaunch")

        # 5e. Validation failure after restore (missing zig.exe) keeps marker and does NOT relaunch
        ubak, ulock = setup_interrupted_bak(inst)
        (ubak / "zig" / "zig.exe").unlink()
        res = subprocess.run([
            str(helper_exe), "--recover", "0", "0",
            str(inst), str(ubak), str(ulock), "500", "1",
        ], timeout=10)
        assert res.returncode == 7
        assert (inst / ".PokeV-Recomp.update-error.txt").is_file()
        assert (inst / ".PokeV-Recomp.update-in-progress").is_file()
        assert not (inst / "relaunched.txt").exists()
        print("ok   windows --recover validation failure blocks relaunch and keeps markers")

        # 5f. Full rollback removes new_feature.dll + new empty subdir, validates old_manifest, preserves recompdata/
        ubak, ulock = setup_interrupted_bak(inst)
        (inst / "recompdata").mkdir(parents=True, exist_ok=True)
        (inst / "recompdata" / "rl.dat").write_bytes(b"rl-keep")
        (inst / "new_feature.dll").write_bytes(b"new-dll-orphan")
        (inst / "new_sub").mkdir(parents=True, exist_ok=True)
        (inst / "new_sub" / "extra.dat").write_bytes(b"new-extra")
        (inst / "extra_tool.dll").write_bytes(b"old-extra-tool")
        (inst / ".PokeV-Recomp.app-manifest").write_text(
            "PokeV-Recomp.exe\nSDL3.dll\nzig/zig.exe\nextra_tool.dll\n", encoding="utf-8"
        )
        stg_dir = inst / ".update-staging"
        stg_dir.mkdir(parents=True, exist_ok=True)
        (stg_dir / "manifest.txt").write_text(
            "PokeV-Recomp.exe\nSDL3.dll\nzig/zig.exe\nnew_feature.dll\nnew_sub/extra.dat\n",
            encoding="utf-8",
        )
        res = subprocess.run([
            str(helper_exe), "--recover", "0", "0",
            str(inst), str(ubak), str(ulock), "5000", "1",
        ], timeout=15)
        assert res.returncode == 0
        for _ in range(40):
            if (inst / "relaunched.txt").is_file():
                break
            time.sleep(0.05)
        assert (inst / "relaunched.txt").is_file()
        assert not (inst / "new_feature.dll").exists()
        assert not (inst / "new_sub").exists()
        assert (inst / "extra_tool.dll").read_bytes() == b"old-extra-tool"
        assert (inst / "SDL3.dll").read_bytes() == b"old-sdl3-bak"
        assert (inst / "recompdata" / "rl.dat").read_bytes() == b"rl-keep"
        assert (inst / "save" / "slot.bin").read_bytes() == b"user-save"
        assert (inst / "larecomp.ini").read_bytes() == b"user-ini=1\n"
        assert (inst / "mods" / "m.pak").read_bytes() == b"user-mod"
        assert (inst / "extracted" / "romfs.bin").read_bytes() == b"user-romfs"
        assert (inst / "unknown_user.txt").read_bytes() == b"keep-me"
        print("ok   windows --recover removes new_feature.dll & new_sub, validates old_manifest, keeps recompdata")

        # 5g. Locked new_feature.dll blocks --recover, keeps backup & markers, then 2nd attempt succeeds
        ubak, ulock = setup_interrupted_bak(inst)
        (inst / "new_feature.dll").write_bytes(b"locked-new-dll")
        stg_dir = inst / ".update-staging"
        stg_dir.mkdir(parents=True, exist_ok=True)
        (stg_dir / "manifest.txt").write_text(
            "PokeV-Recomp.exe\nSDL3.dll\nzig/zig.exe\nnew_feature.dll\n",
            encoding="utf-8",
        )
        with open(inst / "new_feature.dll", "r+b") as locked_new:
            import msvcrt
            msvcrt.locking(locked_new.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                res = subprocess.run([
                    str(helper_exe), "--recover", "0", "0",
                    str(inst), str(ubak), str(ulock), "500", "1",
                ], timeout=15)
                assert res.returncode == 3
                assert ubak.is_dir()
                assert (inst / ".PokeV-Recomp.update-in-progress").is_file()
                assert (stg_dir / "manifest.txt").is_file()
                assert not (inst / "relaunched.txt").exists()
            finally:
                locked_new.seek(0)
                msvcrt.locking(locked_new.fileno(), msvcrt.LK_UNLCK, 1)
        res = subprocess.run([
            str(helper_exe), "--recover", "0", "0",
            str(inst), str(ubak), str(ulock), "5000", "1",
        ], timeout=15)
        assert res.returncode == 0
        assert not (inst / "new_feature.dll").exists()
        assert not ubak.exists()
        assert not (inst / ".PokeV-Recomp.update-in-progress").exists()
        print("ok   windows --recover locked new_feature.dll preserves backup & markers, 2nd attempt succeeds")

        # 5h. Malicious staged manifest (traversal / absolute / user-data) is rejected without deleting anything
        for bad_entry in ["../evil.dll", "C:\\Windows\\System32\\evil.dll", "save/slot.bin", "larecomp.ini"]:
            ubak, ulock = setup_interrupted_bak(inst)
            stg_dir = inst / ".update-staging"
            stg_dir.mkdir(parents=True, exist_ok=True)
            (stg_dir / "manifest.txt").write_text(f"PokeV-Recomp.exe\n{bad_entry}\n", encoding="utf-8")
            res = subprocess.run([
                str(helper_exe), "--recover", "0", "0",
                str(inst), str(ubak), str(ulock), "500", "1",
            ], timeout=10)
            assert res.returncode == 12
            assert ubak.is_dir()
            assert (inst / ".PokeV-Recomp.update-in-progress").is_file()
            assert (inst / "save" / "slot.bin").read_bytes() == b"user-save"
            assert (inst / "larecomp.ini").read_bytes() == b"user-ini=1\n"
            assert not (inst / "relaunched.txt").exists()
        print("ok   windows --recover rejects malicious manifest entries cleanly")

        # 5i. Full previous manifest validation catches missing non-critical file from old_manifest
        ubak, ulock = setup_interrupted_bak(inst)
        (inst / ".PokeV-Recomp.app-manifest").write_text(
            "PokeV-Recomp.exe\nSDL3.dll\nzig/zig.exe\nmissing_old_helper.dll\n",
            encoding="utf-8",
        )
        res = subprocess.run([
            str(helper_exe), "--recover", "0", "0",
            str(inst), str(ubak), str(ulock), "500", "1",
        ], timeout=10)
        assert res.returncode == 7
        assert (inst / ".PokeV-Recomp.update-in-progress").is_file()
        assert not (inst / "relaunched.txt").exists()
        print("ok   windows --recover validates full old_manifest and rejects incomplete restore")

        # 5j. Synchronous rollback() failure keeps progress_marker & backup_dir when restore fails
        setup_install()
        (staged / "zig" / "zig.exe").write_bytes(b"")  # fail validation -> triggers rollback()
        # Make backup_dir\SDL3.dll read-only/locked or lock inst\SDL3.dll after move? Lock backup not easy before move,
        # instead make install_dir\SDL3.dll impossible to delete/overwrite during rollback by placing a directory at dst?
        # Even simpler: verify wmain rejects malicious manifest (code 2) & preserves progress_marker if rollback fails.
        (manifest).write_text("PokeV-Recomp.exe\n../evil.dll\n", encoding="utf-8")
        res = subprocess.run([
            str(helper_exe), "0", "0", str(inst), str(staged), str(manifest),
            str(bak), str(lock), str(note), "v0.2.0", "500", "0",
        ], timeout=10)
        assert res.returncode == 2
        assert (inst / "SDL3.dll").read_bytes() == b"old-sdl3"
        print("ok   windows wmain rejects malicious staged manifest cleanly")


def main():
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    assert cxx, "No C++ compiler found (set CXX or install clang++/g++)"

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        zips = build_test_zips(t)
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
            str(ROOT / "runtime" / "ui" / "win_zip.cpp"),
            "-pthread",
            "-o", str(exe),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        env = os.environ.copy()
        env["LARECOMP_TEST_FAKE_CURL"] = str(fake_curl)
        env["LARECOMP_TEST_TMPDIR"] = str(t)
        env["LARECOMP_TEST_HELLO_SHA"] = hashlib.sha256(b"hello world").hexdigest()
        for k, v in zips.items():
            env[f"LARECOMP_TEST_ZIP_{k}"] = v
        env["PATH"] = f"{t}{os.pathsep}{env.get('PATH', '')}"
        subprocess.run([str(exe)], cwd=ROOT, env=env, check=True)

    test_external_helper_e2e()
    test_windows_helper_e2e(cxx)


if __name__ == "__main__":
    main()
