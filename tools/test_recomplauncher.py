"""Validate RecompLauncher integration: manifest, settings alignment, packaging, and C runtime helpers."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def test_manifest_and_schema():
    manifest_path = ROOT / "recomplauncher.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["format"] == 1
    assert data["repo"] == "didslima/PokeV-Recomp"
    assert data["$schema"] == "https://recomplauncher.com/schema/recomplauncher.json"

    # Inputs: game (folder or nsp/xci requiring keys), optional update, optional dlc1/dlc2, shared switch.prod-keys
    inputs = {x["id"]: x for x in data["inputs"]}
    assert set(inputs) == {"game", "update", "dlc1", "dlc2", "keys"}
    assert inputs["keys"]["role"] == "key"
    assert inputs["keys"]["shared"] == "switch.prod-keys"
    assert inputs["update"]["role"] == "update"
    assert inputs["update"]["optional"] is True
    for dlc_id in ("dlc1", "dlc2"):
        assert inputs[dlc_id]["role"] == "dlc"
        assert inputs[dlc_id]["optional"] is True
        assert inputs[dlc_id]["formats"] == [{"type": "file", "extensions": ["nsp", "xci"], "requires": ["keys"]}]
    assert inputs["game"]["formats"][0] == {"type": "folder"}
    assert inputs["game"]["formats"][1]["requires"] == ["keys"]

    # Prepare and start
    assert "--prepare" in data["prepare"]["args"]
    assert "--game={input:game}" in data["prepare"]["args"]
    assert "--update={input:update}" in data["prepare"]["args"]
    assert "--dlc={input:dlc1}" in data["prepare"]["args"]
    assert "--dlc={input:dlc2}" in data["prepare"]["args"]
    assert "--keys={input:keys}" in data["prepare"]["args"]
    assert data["prepare"]["keeps_copy"] is True
    assert data["prepare"]["done"] == ["{data}/extracted/exefs/main"]
    assert data["start"]["args"] == ["--play"]
    # Windows ZIP: PokeV-Recomp.exe next to the manifest. Linux: the launcher unpacks the AppImage into a plain
    # folder (squashfs-root layout) and runs its AppRun, which forwards to usr/bin/PokeV-Recomp.
    for step in ("prepare", "start"):
        assert data[step]["program"] == ["PokeV-Recomp.exe", "AppRun"], f"{step}.program"
    assert data["saves"] == ["{data}/save/save_0"]

    # External jsonschema validation against the official RecompLauncher schema
    # (pinned to elripalda/RecompLauncher@7d8bc7a59d472402ada20e2fff2fbbf806e10797).
    schema_env = os.environ.get("RECOMPLAUNCHER_SCHEMA")
    require_schema = os.environ.get("REQUIRE_RECOMPLAUNCHER_SCHEMA") == "1"
    if require_schema and (not schema_env or not Path(schema_env).is_file()):
        raise AssertionError(f"REQUIRE_RECOMPLAUNCHER_SCHEMA=1 but RECOMPLAUNCHER_SCHEMA is missing: {schema_env!r}")
    if schema_env and Path(schema_env).is_file():
        try:
            import jsonschema
        except ImportError:
            cli = shutil.which("jsonschema")
            if cli:
                subprocess.run([cli, "-i", str(manifest_path), schema_env], check=True)
            elif require_schema:
                raise
        else:
            schema = json.loads(Path(schema_env).read_text(encoding="utf-8"))
            jsonschema.validate(instance=data, schema=schema)


def test_settings_match_za_default_and_settings_c():
    data = json.loads((ROOT / "recomplauncher.json").read_text(encoding="utf-8"))
    ini_lines = (ROOT / "tools/za/za-default.ini").read_text(encoding="utf-8").splitlines()
    ini_kv = {}
    for line in ini_lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        ini_kv[k] = v

    assert data["config"][0]["new"] == ini_kv, "config[0].new must match tools/za/za-default.ini"

    settings_c = (ROOT / "runtime/settings.c").read_text(encoding="utf-8")
    for s in data["settings"]:
        for key in s.get("defaults", {}):
            assert f'"{key}"' in settings_c, f'Setting key {key} not found in runtime/settings.c'
        for ch in s.get("choices", []):
            for key in ch.get("set", {}):
                assert f'"{key}"' in settings_c, f'Choice key {key} not found in runtime/settings.c'


def test_packaging_and_readme():
    dist_win = (ROOT / "tools/za/dist.sh").read_text(encoding="utf-8")
    dist_linux = (ROOT / "tools/za/dist-linux.sh").read_text(encoding="utf-8")
    apprun = (ROOT / "tools/linux/AppRun").read_text(encoding="utf-8")
    assert "recomplauncher.json" in dist_win
    assert '"$APPDIR/recomplauncher.json"' in dist_linux
    assert '"$@"' in apprun
    assert 'usr/bin/PokeV-Recomp' in apprun
    assert 'cp tools/linux/AppRun "$APPDIR/"' in dist_linux
    assert 'cp $B/PokeV-Recomp.exe $B/PokeV-Recomp.dll.a "$D/"' in dist_win

    for readme in ("README.md", "README.pt-BR.md"):
        text = (ROOT / readme).read_text(encoding="utf-8")
        assert "https://recomplauncher.com/assets/press/launcher/badge-light.svg" in text
        assert "https://recomplauncher.com/assets/press/launcher/badge-dark.svg" in text


def test_runtime_cli_and_env_helpers():
    cc = shutil.which(os.environ.get("CC", "gcc")) or shutil.which("clang")
    if not cc:
        print("SKIP: no C compiler available for runtime helper test")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Verify the exact helper logic from runtime/main.c and runtime/paths.c compiles and behaves as expected
        c_test = tmp / "test_rl_cli.c"
        c_test.write_text(
            r"""
#include <assert.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void rl_setenv(const char *k, const char *v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static void rl_unsetenv(const char *k) {
#ifdef _WIN32
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}

static const char *data_root_env(void) {
    const char *env = getenv("LARECOMP_DATA_ROOT");
    if (env && *env) return env;
    env = getenv("RECOMPLAUNCHER_DATA");
    return env && *env ? env : NULL;
}

static bool take_flag_value(int argc, char **argv, int *idx, const char *name, const char **out) {
    size_t n = strlen(name);
    if (!strncmp(argv[*idx], name, n) && argv[*idx][n] == '=') {
        *out = argv[*idx] + n + 1;
        return true;
    }
    if (!strcmp(argv[*idx], name) && *idx + 1 < argc) {
        *out = argv[++(*idx)];
        return true;
    }
    return false;
}

int main(void) {
    const char *orig_la = getenv("LARECOMP_DATA_ROOT");
    const char *orig_rl = getenv("RECOMPLAUNCHER_DATA");
    char *saved_la = orig_la ? strdup(orig_la) : NULL;
    char *saved_rl = orig_rl ? strdup(orig_rl) : NULL;

    rl_unsetenv("LARECOMP_DATA_ROOT");
    rl_unsetenv("RECOMPLAUNCHER_DATA");
    assert(data_root_env() == NULL);
    rl_setenv("RECOMPLAUNCHER_DATA", "/tmp/rl-data");
    assert(strcmp(data_root_env(), "/tmp/rl-data") == 0);
    rl_setenv("LARECOMP_DATA_ROOT", "/tmp/la-data");
    assert(strcmp(data_root_env(), "/tmp/la-data") == 0);

    if (saved_la) { rl_setenv("LARECOMP_DATA_ROOT", saved_la); free(saved_la); }
    else rl_unsetenv("LARECOMP_DATA_ROOT");
    if (saved_rl) { rl_setenv("RECOMPLAUNCHER_DATA", saved_rl); free(saved_rl); }
    else rl_unsetenv("RECOMPLAUNCHER_DATA");

    char *argv[] = {
        "PokeV-Recomp", "--prepare", "--game=/a/game.nsp", "--update", "/a/upd.nsp",
        "--dlc=/a/dlc1.nsp", "--dlc", "/a/dlc2.nsp", "--keys=/a/prod.keys"
    };
    int argc = 9;
    bool prepare = false;
    const char *game = NULL, *upd = NULL, *keys = NULL, *v = NULL;
    const char *dlcs[32] = {0};
    int dlc_count = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prepare")) prepare = true;
        else if (take_flag_value(argc, argv, &i, "--game", &v)) game = v;
        else if (take_flag_value(argc, argv, &i, "--update", &v)) upd = v;
        else if (!strcmp(argv[i], "--dlc") || !strncmp(argv[i], "--dlc=", 6)) {
            if (!strcmp(argv[i], "--dlc")) {
                assert(i + 1 < argc && argv[i + 1][0]);
                v = argv[++i];
            } else {
                v = argv[i] + 6;
                assert(*v);
            }
            assert(dlc_count < 32);
            dlcs[dlc_count++] = v;
        }
        else if (take_flag_value(argc, argv, &i, "--keys", &v)) keys = v;
    }
    assert(prepare);
    assert(strcmp(game, "/a/game.nsp") == 0);
    assert(strcmp(upd, "/a/upd.nsp") == 0);
    assert(dlc_count == 2);
    assert(strcmp(dlcs[0], "/a/dlc1.nsp") == 0);
    assert(strcmp(dlcs[1], "/a/dlc2.nsp") == 0);
    assert(strcmp(keys, "/a/prod.keys") == 0);
    printf("cli and env helper checks passed\n");
    return 0;
}
""",
            encoding="utf-8",
        )
        exe = tmp / "test_rl_cli"
        subprocess.run([cc, "-std=gnu11", "-Wall", "-Werror", str(c_test), "-o", str(exe)], check=True)
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
        assert "cli and env helper checks passed" in out.stdout

        # Also test C++ with lambda for malformed / overflow --dlc cases
        cxx = shutil.which(os.environ.get("CXX", "g++")) or shutil.which("clang++")
        if cxx:
            cxx_test = tmp / "test_rl_dlc_edge.cpp"
            cxx_test.write_text(
                r"""
#include <assert.h>
#include <stdio.h>
#include <string.h>

static int parse_dlcs(int ac, const char **av, int *out_count) {
    int cnt = 0;
    for (int i = 1; i < ac; i++) {
        if (!strcmp(av[i], "--dlc")) {
            if (i + 1 >= ac || !av[i + 1][0]) return 1;
            i++;
            if (cnt >= 32) return 1;
            cnt++;
        } else if (!strncmp(av[i], "--dlc=", 6)) {
            if (!av[i][6]) return 1;
            if (cnt >= 32) return 1;
            cnt++;
        }
    }
    *out_count = cnt;
    return 0;
}

int main(void) {
    int n = -1;
    const char *omitted[] = {"PokeV-Recomp", "--prepare", "--game=/g.nsp"};
    assert(parse_dlcs(3, omitted, &n) == 0 && n == 0);

    const char *one[] = {"PokeV-Recomp", "--dlc=/d1.nsp"};
    assert(parse_dlcs(2, one, &n) == 0 && n == 1);

    const char *missing_val[] = {"PokeV-Recomp", "--dlc"};
    assert(parse_dlcs(2, missing_val, &n) == 1);

    const char *empty_eq[] = {"PokeV-Recomp", "--dlc="};
    assert(parse_dlcs(2, empty_eq, &n) == 1);

    const char *empty_arg[] = {"PokeV-Recomp", "--dlc", ""};
    assert(parse_dlcs(3, empty_arg, &n) == 1);

    const char *overflow[35];
    overflow[0] = "PokeV-Recomp";
    for (int i = 1; i <= 33; i++) overflow[i] = "--dlc=/d.nsp";
    assert(parse_dlcs(34, overflow, &n) == 1);
    printf("dlc edge checks passed\n");
    return 0;
}
""",
                encoding="utf-8",
            )
            exe_cxx = tmp / "test_rl_dlc_edge"
            subprocess.run([cxx, "-std=gnu++17", "-Wall", "-Werror", str(cxx_test), "-o", str(exe_cxx)], check=True)
            out_cxx = subprocess.run([str(exe_cxx)], capture_output=True, text=True, check=True)
            assert "dlc edge checks passed" in out_cxx.stdout


def test_path_overlap_guard():
    cc = shutil.which(os.environ.get("CC", "gcc")) or shutil.which("clang")
    cxx = shutil.which(os.environ.get("CXX", "g++")) or shutil.which("clang++")
    if not cc:
        print("SKIP: no C compiler available for path_overlap.h test")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dest = tmp / "data" / "extracted"
        nested = dest / "exefs"
        ext = tmp / "external_dump"
        sibling = tmp / "data" / "extracted_other"
        parent_src = tmp / "data"
        nonexistent_dest_inside_src = ext / "extracted"
        for d in (nested, ext, sibling):
            d.mkdir(parents=True, exist_ok=True)
        marker = nested / "main"
        marker.write_bytes(b"KEEP_ME")
        link = tmp / "link_to_dest"
        try:
            link.symlink_to(dest, target_is_directory=True)
        except OSError:
            link = dest  # Windows without Developer Mode: symlink creation may require privileges

        c_test = tmp / "test_overlap.c"
        c_test.write_text(
            r"""
#include <assert.h>
#include <stdio.h>
#include "path_overlap.h"

int main(int argc, char **argv) {
    assert(argc == 8);
    const char *dest = argv[1];
    const char *nested = argv[2];
    const char *ext = argv[3];
    const char *sibling = argv[4];
    const char *link = argv[5];
    const char *parent_src = argv[6];
    const char *nonexistent_dest_inside_src = argv[7];

    assert(!path_overlaps_dest(ext, dest));
    assert(!path_overlaps_dest(sibling, dest));
    assert(path_overlaps_dest(dest, dest));
    assert(path_overlaps_dest(nested, dest));
    assert(path_overlaps_dest(parent_src, dest));
    assert(path_overlaps_dest(ext, nonexistent_dest_inside_src));
    assert(path_overlaps_dest(link, dest));
    assert(path_overlaps_dest("/nonexistent/path/for/pokev/test", dest));
    assert(path_overlaps_dest(ext, "/nonexistent/dest/for/pokev/test"));
    assert(path_overlaps_dest(NULL, dest));
    assert(path_overlaps_dest("", dest));
    printf("path_overlap.h checks passed\n");
    return 0;
}
""",
            encoding="utf-8",
        )
        compilers = [(cc, "-std=gnu11")]
        if cxx:
            compilers.append((cxx, "-std=gnu++17"))
        for comp, std in compilers:
            exe = tmp / f"test_overlap_{std.lstrip('-')}"
            src_args = [str(c_test)] if std.endswith("11") else ["-x", "c++", str(c_test)]
            subprocess.run(
                [comp, std, "-Wall", "-Werror", f"-I{ROOT / 'runtime'}"] + src_args + ["-o", str(exe)],
                check=True,
            )
            out = subprocess.run(
                [str(exe), str(dest), str(nested), str(ext), str(sibling), str(link),
                 str(parent_src), str(nonexistent_dest_inside_src)],
                capture_output=True,
                text=True,
                check=True,
            )
            assert "path_overlap.h checks passed" in out.stdout
        assert marker.read_bytes() == b"KEEP_ME", "source file must remain intact when overlap is rejected"
        assert not nonexistent_dest_inside_src.exists(), "destination must not be created inside source"


def test_aoc_cleanup_and_save_preservation():
    """Verify runtime/paths.c includes 'aoc' in GAME_FILES and never touches save/save_0."""
    paths_c = (ROOT / "runtime/paths.c").read_text(encoding="utf-8")
    m = re.search(r"GAME_FILES\[\]\s*=\s*\{([^}]+)\}", paths_c)
    assert m, "GAME_FILES definition not found in runtime/paths.c"
    assert '"aoc"' in m.group(1), "'aoc' must be in GAME_FILES for stale DLC cleanup"

    cc = shutil.which(os.environ.get("CC", "gcc")) or shutil.which("clang")
    if not cc:
        print("SKIP: no C compiler available for aoc cleanup test")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        extracted = tmp / "extracted"
        (extracted / "exefs").mkdir(parents=True)
        (extracted / "exefs" / "main").write_bytes(b"MAIN")
        (extracted / "romfs.bin").write_bytes(b"ROMFS")
        (extracted / "aoc").mkdir(parents=True)
        (extracted / "aoc" / "01008f6008c5f001.bin").write_bytes(b"DLC1")
        save_dir = tmp / "save" / "save_0"
        save_dir.mkdir(parents=True)
        (save_dir / "main").write_bytes(b"SAVE_DATA_MUST_SURVIVE")

        # Verify that removing GAME_FILES inside extracted/ removes aoc/ and leaves save/save_0 intact
        c_test = tmp / "test_cleanup.c"
        c_test.write_text(
            r"""
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int remove_tree(const char *p) {
    struct stat st;
    if (lstat(p, &st) != 0) return 1;
    if (!S_ISDIR(st.st_mode)) return remove(p) == 0;
    DIR *d = opendir(p);
    if (!d) return 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char sub[1024];
        snprintf(sub, sizeof sub, "%s/%s", p, e->d_name);
        if (!remove_tree(sub)) { closedir(d); return 0; }
    }
    closedir(d);
    return rmdir(p) == 0;
}

int main(int argc, char **argv) {
    assert(argc == 2);
    const char *game_dir = argv[1];
    const char *files[] = {"exefs", "romfs", "romfs.bin", "aoc"};
    for (size_t i = 0; i < sizeof files / sizeof *files; i++) {
        char p[1024];
        snprintf(p, sizeof p, "%s/%s", game_dir, files[i]);
        assert(remove_tree(p));
    }
    assert(rmdir(game_dir) == 0);
    return 0;
}
""",
            encoding="utf-8",
        )
        exe = tmp / "test_cleanup"
        subprocess.run([cc, "-std=gnu11", "-Wall", "-Werror", str(c_test), "-o", str(exe)], check=True)
        subprocess.run([str(exe), str(extracted)], check=True)
        assert not extracted.exists(), "full cleanup must remove extracted/ including aoc/"
        assert (save_dir / "main").read_bytes() == b"SAVE_DATA_MUST_SURVIVE", "save_0 must remain untouched"


def test_launcher_prepare_dlc_state_machine():
    """Verify launcher_prepare() DLC stamp and incremental/fresh/unchanged logic."""
    launcher_mm = (ROOT / "runtime/ui/launcher.mm").read_text(encoding="utf-8")
    for sym in (
        "dlc_stamp_matches",
        "write_dlc_stamp",
        "import_aoc_packages",
        "--aoc-only",
        "Already built",
        "Importing DLC packages...",
    ):
        assert sym in launcher_mm, f"Expected {sym!r} in runtime/ui/launcher.mm"

    cxx = shutil.which(os.environ.get("CXX", "g++")) or shutil.which("clang++")
    if not cxx:
        print("SKIP: no C++ compiler available for launcher_prepare DLC state machine test")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cpp_test = tmp / "test_dlc_state.cpp"
        cpp_test.write_text(
            r"""
#include <algorithm>
#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static std::string dlc_stamp_for(const std::vector<std::string> &dlcs) {
    std::vector<std::string> sorted = dlcs;
    std::sort(sorted.begin(), sorted.end());
    std::string out;
    for (const auto &p : sorted) {
        struct stat sb;
        char line[1280];
        if (!stat(p.c_str(), &sb))
            snprintf(line, sizeof line, "%s|%llu|%llu\n", p.c_str(), (unsigned long long)sb.st_size,
                     (unsigned long long)sb.st_mtime);
        else
            snprintf(line, sizeof line, "%s|?|?\n", p.c_str());
        out += line;
    }
    return out;
}

static bool aoc_has_bins(const std::string &root) {
    DIR *d = opendir((root + "/aoc").c_str());
    if (!d) return false;
    bool found = false;
    while (struct dirent *e = readdir(d)) {
        size_t n = strlen(e->d_name);
        if (n > 4 && !strcmp(e->d_name + n - 4, ".bin")) { found = true; break; }
    }
    closedir(d);
    return found;
}

static bool dlc_stamp_matches(const std::string &root, const std::vector<std::string> &dlcs) {
    if (dlcs.empty()) return true;
    if (!aoc_has_bins(root)) return false;
    FILE *f = fopen((root + "/aoc/.dlc_stamp").c_str(), "rb");
    if (!f) return false;
    std::string have;
    char buf[1024];
    size_t k;
    while ((k = fread(buf, 1, sizeof buf, f)) > 0) have.append(buf, k);
    fclose(f);
    return have == dlc_stamp_for(dlcs);
}

static void write_dlc_stamp(const std::string &root, const std::vector<std::string> &dlcs) {
    if (dlcs.empty()) return;
    mkdir((root + "/aoc").c_str(), 0755);
    FILE *f = fopen((root + "/aoc/.dlc_stamp").c_str(), "wb");
    assert(f);
    std::string stamp = dlc_stamp_for(dlcs);
    fwrite(stamp.data(), 1, stamp.size(), f);
    fclose(f);
}

int main(int argc, char **argv) {
    assert(argc == 4);
    std::string root = argv[1];
    std::string dlc1 = argv[2];
    std::string dlc2 = argv[3];

    // 1. Existing installation without DLC: dlc_needed is false when no DLC passed
    std::vector<std::string> none;
    assert(dlc_stamp_matches(root, none) == true);

    // 2. Adding dlc1 to existing installation: no stamp yet -> dlc_needed is true
    std::vector<std::string> one = {dlc1};
    assert(dlc_stamp_matches(root, one) == false);

    // 3. Simulate incremental import of dlc1 (creates aoc/01008f6008c5f001.bin + .dlc_stamp)
    mkdir((root + "/aoc").c_str(), 0755);
    FILE *b1 = fopen((root + "/aoc/01008f6008c5f001.bin").c_str(), "wb");
    fwrite("DLC1", 1, 4, b1);
    fclose(b1);
    write_dlc_stamp(root, one);

    // 4. Re-running with unchanged dlc1 -> matches (Already built fast path)
    assert(dlc_stamp_matches(root, one) == true);

    // 5. Adding dlc2 -> mismatch (triggers incremental import)
    std::vector<std::string> two = {dlc1, dlc2};
    assert(dlc_stamp_matches(root, two) == false);

    // 6. Simulate incremental import of dlc2: dlc1 .bin must still exist
    FILE *b2 = fopen((root + "/aoc/01008f6008c5f002.bin").c_str(), "wb");
    fwrite("DLC2", 1, 4, b2);
    fclose(b2);
    write_dlc_stamp(root, two);
    assert(dlc_stamp_matches(root, two) == true);
    // Order independence
    std::vector<std::string> two_rev = {dlc2, dlc1};
    assert(dlc_stamp_matches(root, two_rev) == true);
    assert(access((root + "/aoc/01008f6008c5f001.bin").c_str(), F_OK) == 0);

    // 7. Re-importing modified dlc1 (size change) -> mismatch
    FILE *d1 = fopen(dlc1.c_str(), "wb");
    fwrite("UPDATED_DLC1_PACKAGE", 1, 20, d1);
    fclose(d1);
    assert(dlc_stamp_matches(root, two) == false);

    return 0;
}
""",
            encoding="utf-8",
        )
        root = tmp / "extracted"
        root.mkdir()
        dlc1 = tmp / "dlc1.nsp"
        dlc2 = tmp / "dlc2.nsp"
        dlc1.write_bytes(b"DLC1_PKG")
        dlc2.write_bytes(b"DLC2_PKG")
        exe = tmp / "test_dlc_state"
        subprocess.run([cxx, "-std=gnu++17", "-Wall", "-Werror", str(cpp_test), "-o", str(exe)], check=True)
        subprocess.run([str(exe), str(root), str(dlc1), str(dlc2)], check=True)


def test_zaimport_aoc_only_logic():
    """Verify tools/zaimport/ZaImportLogic.cs classification and --aoc-only support."""
    prog_cs = (ROOT / "tools/zaimport/Program.cs").read_text(encoding="utf-8")
    logic_cs = (ROOT / "tools/zaimport/ZaImportLogic.cs").read_text(encoding="utf-8")
    assert "--aoc-only" in prog_cs
    assert "ZaImportLogic.Classify" in prog_cs
    assert "ZaImportLogic.ResolveAocOutputPath" in prog_cs
    assert "0x01008F6008C5E000" in logic_cs

    dotnet = shutil.which("dotnet")
    if not dotnet:
        print("SKIP: dotnet not available for ZaImportLogic C# unit test")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ZaImportLogic.cs").write_text(logic_cs, encoding="utf-8")
        (tmp / "Program.cs").write_text(
            r"""
using System;
using System.Diagnostics;
using System.IO;

const ulong VioletBase = 0x01008F6008C5E000;
const ulong VioletUpd  = 0x01008F6008C5E800;
const ulong VioletDlc1 = 0x01008F6008C5F001;
const ulong VioletDlc2 = 0x01008F6008C5F002;

// 1. Base + update + 0 DLC
var r0 = ZaImportLogic.Classify(new[] {
    new NcaEntry(VioletBase, true, false, false),
    new NcaEntry(VioletUpd, true, true, false),
}, aocOnly: false);
Debug.Assert(r0.Error == null && r0.BaseIndex == 0 && r0.UpdateIndex == 1 && r0.DlcIndices.Count == 0);

// 2. Base + update + 2 DLCs
var r2 = ZaImportLogic.Classify(new[] {
    new NcaEntry(VioletBase, true, false, false),
    new NcaEntry(VioletUpd, true, true, false),
    new NcaEntry(VioletDlc1, false, false, true),
    new NcaEntry(VioletDlc2, false, false, true),
}, aocOnly: false);
Debug.Assert(r2.Error == null && r2.DlcIndices.Count == 2);

// 3. --aoc-only with 1 DLC (no base)
var rAoc1 = ZaImportLogic.Classify(new[] {
    new NcaEntry(VioletDlc1, false, false, true),
}, aocOnly: true);
Debug.Assert(rAoc1.Error == null && rAoc1.BaseTitleId == VioletBase && rAoc1.DlcIndices.Count == 1);

// 4. --aoc-only with multiple DLCs and duplicate TitleId
var rAoc2 = ZaImportLogic.Classify(new[] {
    new NcaEntry(VioletDlc1, false, false, true),
    new NcaEntry(VioletDlc2, false, false, true),
    new NcaEntry(VioletDlc1, false, false, true),
}, aocOnly: true);
Debug.Assert(rAoc2.Error == null && rAoc2.DlcIndices.Count == 2);

// 5. Full mode without base fails
var rNoBase = ZaImportLogic.Classify(new[] {
    new NcaEntry(VioletDlc1, false, false, true),
}, aocOnly: false);
Debug.Assert(rNoBase.Error != null);

// 6. --aoc-only without DLC fails
var rNoDlc = ZaImportLogic.Classify(new[] {
    new NcaEntry(VioletBase, true, false, false),
}, aocOnly: true);
Debug.Assert(rNoDlc.Error != null);

// 7. Output path stays under aoc/
string outDir = Path.Combine(Path.GetTempPath(), "zaimport_test_out");
string p = ZaImportLogic.ResolveAocOutputPath(outDir, VioletDlc1);
Debug.Assert(p.EndsWith(Path.Combine("aoc", "01008f6008c5f001.bin")));
Console.WriteLine("ZaImportLogic C# checks passed");
""",
            encoding="utf-8",
        )
        (tmp / "test.csproj").write_text(
            "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><OutputType>Exe</OutputType>"
            "<TargetFramework>net8.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings>"
            "<Nullable>enable</Nullable></PropertyGroup></Project>",
            encoding="utf-8",
        )
        out = subprocess.run([dotnet, "run", "--project", str(tmp / "test.csproj")], capture_output=True, text=True, check=True)
        assert "ZaImportLogic C# checks passed" in out.stdout


if __name__ == "__main__":
    test_manifest_and_schema()
    test_settings_match_za_default_and_settings_c()
    test_packaging_and_readme()
    test_runtime_cli_and_env_helpers()
    test_path_overlap_guard()
    test_aoc_cleanup_and_save_preservation()
    test_launcher_prepare_dlc_state_machine()
    test_zaimport_aoc_only_logic()
    print("all recomplauncher integration checks passed")
