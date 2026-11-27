// Standalone Windows Update Helper (PokeV-Recomp-updater.exe)
// Zero third-party DLL dependencies: uses only Win32 APIs (kernel32).
// Performs PID-reuse-safe parent wait, lock verification, manifest-driven backup,
// transactional swap, post-install validation, automatic rollback, and relaunch.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cwctype>
#include <string>
#include <unordered_set>
#include <vector>

static std::wstring to_long_win_path(const std::wstring &p) {
    if (p.empty()) return L"";
    std::wstring w = p;
    std::replace(w.begin(), w.end(), L'/', L'\\');
    if (w.size() >= 240 && w.rfind(L"\\\\?\\", 0) != 0) {
        if (w.rfind(L"\\\\", 0) == 0) w = L"\\\\?\\UNC\\" + w.substr(2);
        else if (w.size() >= 2 && w[1] == L':') w = L"\\\\?\\" + w;
    }
    return w;
}

static std::wstring utf8_to_wide(const std::string &u8) {
    if (u8.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, u8.c_str(), -1, nullptr, 0);
    if (n <= 0) return L"";
    std::wstring w(n - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, u8.c_str(), -1, w.data(), n);
    return w;
}

static std::string wide_to_utf8(const std::wstring &w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return "";
    std::string s(n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
    return s;
}

static std::wstring to_lower_w(std::wstring s) {
    for (wchar_t &c : s) c = (wchar_t)towlower(c);
    return s;
}

static std::wstring join_path(const std::wstring &a, const std::wstring &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    wchar_t last = a.back();
    if (last == L'\\' || last == L'/') return a + b;
    return a + L'\\' + b;
}

static std::wstring parent_dir(const std::wstring &p) {
    size_t pos = p.find_last_of(L"\\/");
    if (pos == std::wstring::npos) return L"";
    return p.substr(0, pos);
}

static bool file_exists_nonempty(const std::wstring &p, bool check_nonempty = false) {
    std::wstring lp = to_long_win_path(p);
    WIN32_FILE_ATTRIBUTE_DATA fad{};
    if (!GetFileAttributesExW(lp.c_str(), GetFileExInfoStandard, &fad)) return false;
    if ((fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) return false;
    if (!check_nonempty) return true;
    return fad.nFileSizeHigh > 0 || fad.nFileSizeLow > 0;
}

static bool mkdirs_recursive(const std::wstring &dir) {
    if (dir.empty()) return true;
    std::wstring norm = dir;
    std::replace(norm.begin(), norm.end(), L'/', L'\\');
    while (norm.size() > 3 && norm.back() == L'\\') norm.pop_back();

    std::wstring lp = to_long_win_path(norm);
    DWORD attr = GetFileAttributesW(lp.c_str());
    if (attr != INVALID_FILE_ATTRIBUTES) {
        return (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
    }
    std::wstring par = parent_dir(norm);
    if (!par.empty() && par != norm && par.size() >= 2 && !(par.size() == 2 && par[1] == L':')) {
        if (!mkdirs_recursive(par)) return false;
    }
    if (CreateDirectoryW(lp.c_str(), nullptr)) return true;
    DWORD err = GetLastError();
    if (err == ERROR_ALREADY_EXISTS) {
        attr = GetFileAttributesW(lp.c_str());
        return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) != 0;
    }
    return false;
}

static void remove_tree_recursive(const std::wstring &dir) {
    std::wstring lp = to_long_win_path(dir);
    DWORD attr = GetFileAttributesW(lp.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES) return;
    if ((attr & FILE_ATTRIBUTE_DIRECTORY) == 0) {
        SetFileAttributesW(lp.c_str(), FILE_ATTRIBUTE_NORMAL);
        DeleteFileW(lp.c_str());
        return;
    }
    if ((attr & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        RemoveDirectoryW(lp.c_str());
        return;
    }
    WIN32_FIND_DATAW ffd{};
    std::wstring pat = to_long_win_path(join_path(dir, L"*"));
    HANDLE h = FindFirstFileW(pat.c_str(), &ffd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (wcscmp(ffd.cFileName, L".") == 0 || wcscmp(ffd.cFileName, L"..") == 0) continue;
            std::wstring child = join_path(dir, ffd.cFileName);
            if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
                remove_tree_recursive(child);
            } else {
                std::wstring lchild = to_long_win_path(child);
                SetFileAttributesW(lchild.c_str(), FILE_ATTRIBUTE_NORMAL);
                DeleteFileW(lchild.c_str());
            }
        } while (FindNextFileW(h, &ffd));
        FindClose(h);
    }
    SetFileAttributesW(lp.c_str(), FILE_ATTRIBUTE_NORMAL);
    RemoveDirectoryW(lp.c_str());
}

static bool write_text_file(const std::wstring &path, const std::string &content) {
    std::wstring par = parent_dir(path);
    if (!par.empty()) mkdirs_recursive(par);
    HANDLE h = CreateFileW(to_long_win_path(path).c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    bool ok = true;
    if (!content.empty()) {
        ok = WriteFile(h, content.data(), (DWORD)content.size(), &written, nullptr) &&
             written == (DWORD)content.size();
    }
    CloseHandle(h);
    return ok;
}

static std::vector<std::wstring> read_manifest_lines(const std::wstring &path) {
    std::vector<std::wstring> out;
    HANDLE h = CreateFileW(to_long_win_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return out;
    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0 || sz.QuadPart > 16 * 1024 * 1024) {
        CloseHandle(h);
        return out;
    }
    std::string raw((size_t)sz.QuadPart, '\0');
    DWORD rd = 0;
    if (!ReadFile(h, raw.data(), (DWORD)raw.size(), &rd, nullptr)) {
        CloseHandle(h);
        return out;
    }
    CloseHandle(h);
    raw.resize(rd);

    size_t pos = 0;
    while (pos < raw.size()) {
        size_t nl = raw.find('\n', pos);
        if (nl == std::string::npos) nl = raw.size();
        std::string line = raw.substr(pos, nl - pos);
        pos = nl + 1;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        std::wstring w = utf8_to_wide(line);
        std::replace(w.begin(), w.end(), L'/', L'\\');
        if (!w.empty()) out.push_back(w);
    }
    return out;
}

static bool move_file_mkdir(const std::wstring &src, const std::wstring &dst) {
    std::wstring par = parent_dir(dst);
    if (!par.empty() && !mkdirs_recursive(par)) return false;
    std::wstring lsrc = to_long_win_path(src);
    std::wstring ldst = to_long_win_path(dst);
    SetFileAttributesW(ldst.c_str(), FILE_ATTRIBUTE_NORMAL);
    return MoveFileExW(lsrc.c_str(), ldst.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
}

static bool check_file_unlocked(const std::wstring &path) {
    std::wstring lp = to_long_win_path(path);
    DWORD attr = GetFileAttributesW(lp.c_str());
    if (attr == INVALID_FILE_ATTRIBUTES || (attr & FILE_ATTRIBUTE_DIRECTORY) != 0) return true;
    HANDLE h = CreateFileW(lp.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) return true;
        return false;
    }
    CloseHandle(h);
    return true;
}

static bool wait_for_parent_exit(DWORD pid, uintptr_t handle_val, DWORD timeout_ms) {
    if (handle_val != 0) {
        HANDLE hParent = (HANDLE)handle_val;
        DWORD wr = WaitForSingleObject(hParent, timeout_ms);
        CloseHandle(hParent);
        if (wr != WAIT_OBJECT_0) return false;
    } else if (pid != 0) {
        HANDLE hProc = OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        if (hProc) {
            DWORD wr = WaitForSingleObject(hProc, timeout_ms);
            CloseHandle(hProc);
            if (wr != WAIT_OBJECT_0) return false;
        }
    }
    return true;
}

static bool collect_relative_files(const std::wstring &base_dir, const std::wstring &sub_rel,
                                   std::vector<std::wstring> &out) {
    std::wstring cur = sub_rel.empty() ? base_dir : join_path(base_dir, sub_rel);
    std::wstring pat = to_long_win_path(join_path(cur, L"*"));
    WIN32_FIND_DATAW ffd{};
    HANDLE h = FindFirstFileW(pat.c_str(), &ffd);
    if (h == INVALID_HANDLE_VALUE) return false;
    do {
        if (wcscmp(ffd.cFileName, L".") == 0 || wcscmp(ffd.cFileName, L"..") == 0) continue;
        std::wstring rel = sub_rel.empty() ? std::wstring(ffd.cFileName) : join_path(sub_rel, ffd.cFileName);
        if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) continue;
        if ((ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
            collect_relative_files(base_dir, rel, out);
        } else {
            out.push_back(rel);
        }
    } while (FindNextFileW(h, &ffd));
    FindClose(h);
    return true;
}

static bool validate_critical_install_files(const std::wstring &install_dir, std::string &out_missing) {
    static const wchar_t *const k_critical[] = {
        L"PokeV-Recomp.exe",
        L"za-default.ini",
        L"recomplauncher.json",
        L"gamelib-src\\gamelib_entry.c",
        L"zaimport\\zaimport.exe",
        L"zig\\zig.exe",
        L"lasgen\\lasgen.exe",
    };
    for (const wchar_t *c : k_critical) {
        if (!file_exists_nonempty(join_path(install_dir, c), true)) {
            out_missing = wide_to_utf8(c);
            return false;
        }
    }
    return true;
}

static void relaunch_app(const std::wstring &install_dir) {
    std::wstring exe_path = join_path(install_dir, L"PokeV-Recomp.exe");
    std::wstring cmd = L"\"" + exe_path + L"\"";
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (CreateProcessW(to_long_win_path(exe_path).c_str(), cmd.data(), nullptr, nullptr, FALSE,
                       DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                       nullptr, to_long_win_path(install_dir).c_str(), &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
}

static int run_recover_mode(int argc, wchar_t **argv) {
    // --recover <pid> <handle_val> <install_dir> <backup_dir> <lock_path> [timeout_ms] [relaunch_flag]
    if (argc < 7) return 10;

    DWORD pid = (DWORD)wcstoul(argv[2], nullptr, 10);
    uintptr_t handle_val = (uintptr_t)wcstoull(argv[3], nullptr, 10);
    std::wstring install_dir = argv[4];
    std::wstring backup_dir = argv[5];
    std::wstring lock_path = argv[6];
    DWORD timeout_ms = (argc >= 8 && argv[7][0]) ? (DWORD)wcstoul(argv[7], nullptr, 10) : 15000;
    bool do_relaunch = (argc >= 9) ? (wcscmp(argv[8], L"0") != 0) : true;

    std::wstring err_marker = join_path(install_dir, L".PokeV-Recomp.update-error.txt");
    std::wstring progress_marker = join_path(install_dir, L".PokeV-Recomp.update-in-progress");
    std::wstring notice_marker = join_path(install_dir, L".PokeV-Recomp.update-notice.txt");
    std::wstring staging_root = join_path(install_dir, L".update-staging");

    auto fail_keep_backup = [&](const std::string &msg, int code) {
        write_text_file(err_marker, msg);
        DeleteFileW(to_long_win_path(lock_path).c_str());
        return code;
    };

    // 1. Wait for main process to exit completely (inherited HANDLE prevents PID reuse)
    if (!wait_for_parent_exit(pid, handle_val, timeout_ms)) {
        return fail_keep_backup(
            "Timed out waiting for PokeV-Recomp.exe to exit before recovering backup.", 1);
    }

    // 2. Verify backup directory exists and contains files
    DWORD battr = GetFileAttributesW(to_long_win_path(backup_dir).c_str());
    if (battr == INVALID_FILE_ATTRIBUTES || (battr & FILE_ATTRIBUTE_DIRECTORY) == 0) {
        return fail_keep_backup(
            "Recovery failed: .PokeV-Recomp.update.bak is missing. Please reinstall PokeV-Recomp.", 11);
    }
    std::vector<std::wstring> bak_files;
    collect_relative_files(backup_dir, L"", bak_files);
    if (bak_files.empty()) {
        return fail_keep_backup(
            "Recovery failed: .PokeV-Recomp.update.bak is empty. Please reinstall PokeV-Recomp.", 11);
    }

    // 3. Confirm target executables/DLLs are no longer locked (retry up to 5 seconds)
    DWORD lock_wait_ms = 0;
    bool all_unlocked = false;
    while (lock_wait_ms <= 5000) {
        all_unlocked = true;
        for (const auto &rel : bak_files) {
            if (!check_file_unlocked(join_path(install_dir, rel))) {
                all_unlocked = false;
                break;
            }
        }
        if (all_unlocked) break;
        Sleep(100);
        lock_wait_ms += 100;
    }
    if (!all_unlocked) {
        return fail_keep_backup(
            "Recovery blocked: target files are still locked by another process. Close all instances and retry.", 3);
    }

    // 4. Restore every file from backup_dir -> install_dir (keep remaining backup files if any step fails)
    for (const auto &rel : bak_files) {
        if (!move_file_mkdir(join_path(backup_dir, rel), join_path(install_dir, rel))) {
            return fail_keep_backup(
                "Recovery failed while restoring " + wide_to_utf8(rel) +
                    ". Backup preserved in .PokeV-Recomp.update.bak.",
                5);
        }
    }

    // 5. Validate critical files before clearing markers or relaunching
    std::string missing_crit;
    if (!validate_critical_install_files(install_dir, missing_crit)) {
        return fail_keep_backup(
            "Recovery validation failed (missing " + missing_crit +
                "). Manual recovery or reinstallation is required.",
            7);
    }

    // 6. Commit recovery: clear markers BEFORE relaunching to prevent any relaunch loop
    DeleteFileW(to_long_win_path(err_marker).c_str());
    DeleteFileW(to_long_win_path(progress_marker).c_str());
    remove_tree_recursive(backup_dir);
    remove_tree_recursive(staging_root);
    DeleteFileW(to_long_win_path(lock_path).c_str());
    write_text_file(notice_marker,
                    "Previous update was interrupted and safely restored to the previous version.");

    // 7. Relaunch verified installation
    if (do_relaunch) {
        relaunch_app(install_dir);
    }
    return 0;
}

int wmain(int argc, wchar_t **argv) {
    if (argc >= 2 && wcscmp(argv[1], L"--recover") == 0) {
        return run_recover_mode(argc, argv);
    }
    if (argc < 10) return 10;

    DWORD pid = (DWORD)wcstoul(argv[1], nullptr, 10);
    uintptr_t handle_val = (uintptr_t)wcstoull(argv[2], nullptr, 10);
    std::wstring install_dir = argv[3];
    std::wstring staged_dir = argv[4];
    std::wstring manifest_path = argv[5];
    std::wstring backup_dir = argv[6];
    std::wstring lock_path = argv[7];
    std::wstring note_path = argv[8];
    std::string tag_utf8 = wide_to_utf8(argv[9]);
    DWORD timeout_ms = (argc >= 11 && argv[10][0]) ? (DWORD)wcstoul(argv[10], nullptr, 10) : 15000;
    bool do_relaunch = (argc >= 12) ? (wcscmp(argv[11], L"0") != 0) : true;

    std::wstring err_marker = join_path(install_dir, L".PokeV-Recomp.update-error.txt");
    std::wstring progress_marker = join_path(install_dir, L".PokeV-Recomp.update-in-progress");
    std::wstring app_manifest_path = join_path(install_dir, L".PokeV-Recomp.app-manifest");

    auto fail_clean_only = [&](const std::string &msg, int code) {
        write_text_file(err_marker, msg);
        DeleteFileW(to_long_win_path(lock_path).c_str());
        return code;
    };

    // 1. Wait for parent process to exit (inherited HANDLE prevents PID reuse)
    if (!wait_for_parent_exit(pid, handle_val, timeout_ms)) {
        return fail_clean_only("Timed out waiting for PokeV-Recomp.exe to exit before updating.", 1);
    }

    // 2. Load new manifest + optional old manifest
    std::vector<std::wstring> new_manifest = read_manifest_lines(manifest_path);
    if (new_manifest.empty()) {
        return fail_clean_only("Staged update manifest is missing or empty.", 2);
    }
    std::vector<std::wstring> old_manifest = read_manifest_lines(app_manifest_path);

    std::vector<std::wstring> backup_targets;
    std::unordered_set<std::wstring> seen;
    for (const auto &rel : new_manifest) {
        if (seen.insert(to_lower_w(rel)).second) backup_targets.push_back(rel);
    }
    for (const auto &rel : old_manifest) {
        if (seen.insert(to_lower_w(rel)).second) backup_targets.push_back(rel);
    }

    // 3. Confirm target binaries/DLLs are no longer locked (retry up to 5 seconds)
    DWORD lock_wait_ms = 0;
    bool all_unlocked = false;
    while (lock_wait_ms <= 5000) {
        all_unlocked = true;
        for (const auto &rel : backup_targets) {
            std::wstring full = join_path(install_dir, rel);
            if (!check_file_unlocked(full)) {
                all_unlocked = false;
                break;
            }
        }
        if (all_unlocked) break;
        Sleep(100);
        lock_wait_ms += 100;
    }
    if (!all_unlocked) {
        return fail_clean_only("Target application files are still locked by another process.", 3);
    }

    // 4. Mark update in progress and create backup of existing app files
    remove_tree_recursive(backup_dir);
    if (!mkdirs_recursive(backup_dir)) {
        return fail_clean_only("Failed to create backup directory before updating.", 4);
    }
    write_text_file(progress_marker, tag_utf8);

    std::vector<std::wstring> moved_to_backup;
    std::vector<std::wstring> installed_files;

    auto rollback = [&]() {
        for (const auto &rel : installed_files) {
            std::wstring dst = to_long_win_path(join_path(install_dir, rel));
            SetFileAttributesW(dst.c_str(), FILE_ATTRIBUTE_NORMAL);
            DeleteFileW(dst.c_str());
        }
        bool all_restored = true;
        for (const auto &rel : moved_to_backup) {
            if (!move_file_mkdir(join_path(backup_dir, rel), join_path(install_dir, rel))) {
                all_restored = false;
            }
        }
        DeleteFileW(to_long_win_path(progress_marker).c_str());
        if (all_restored) remove_tree_recursive(backup_dir);
        DeleteFileW(to_long_win_path(lock_path).c_str());
    };

    for (const auto &rel : backup_targets) {
        std::wstring cur = join_path(install_dir, rel);
        if (file_exists_nonempty(cur, false)) {
            if (!move_file_mkdir(cur, join_path(backup_dir, rel))) {
                rollback();
                write_text_file(err_marker, "Failed to back up existing file: " + wide_to_utf8(rel));
                return 5;
            }
            moved_to_backup.push_back(rel);
        }
    }

    // 5. Move staged files into install_dir
    for (const auto &rel : new_manifest) {
        std::wstring src = join_path(staged_dir, rel);
        std::wstring dst = join_path(install_dir, rel);
        if (!move_file_mkdir(src, dst)) {
            rollback();
            write_text_file(err_marker, "Failed to install updated file: " + wide_to_utf8(rel));
            return 6;
        }
        installed_files.push_back(rel);
    }

    // 6. Validate critical files in the new installation
    std::string missing_crit;
    if (!validate_critical_install_files(install_dir, missing_crit)) {
        rollback();
        write_text_file(err_marker,
                        "Updated installation failed validation (missing " +
                        missing_crit + "). Rolled back to previous version.");
        return 7;
    }

    // 7. Commit: write new app manifest, _updated.txt, and clean up backup/staging/lock
    std::string manifest_buf;
    for (const auto &rel : new_manifest) {
        manifest_buf += wide_to_utf8(rel) + "\n";
    }
    write_text_file(app_manifest_path, manifest_buf);
    if (!note_path.empty()) {
        write_text_file(note_path, tag_utf8);
    }
    DeleteFileW(to_long_win_path(progress_marker).c_str());
    DeleteFileW(to_long_win_path(err_marker).c_str());
    remove_tree_recursive(backup_dir);
    remove_tree_recursive(parent_dir(staged_dir));
    DeleteFileW(to_long_win_path(lock_path).c_str());

    // 8. Relaunch updated PokeV-Recomp.exe
    if (do_relaunch) {
        relaunch_app(install_dir);
    }
    return 0;
}
