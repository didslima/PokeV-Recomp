"""Regression tests for Pokémon Scarlet 4.0.0 support and save isolation."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import sv_patches


def test_sv_patches():
    violet = '709bfd6611529864'
    scarlet = '421c5411b487eb4d'
    assert sv_patches.SV_BUILDS.get(violet) == '4.0.0'
    assert sv_patches.SV_BUILDS.get(scarlet) == '4.0.0'
    vp = sv_patches.sv_patches(violet)
    sp = sv_patches.sv_patches(scarlet)
    assert len(vp) == 12, f'expected 12 SV patches, got {len(vp)}'
    assert vp == sp, 'Scarlet and Violet 4.0.0 must share the exact same 12 sv_patches'
    assert sv_patches.sv_patches('0000000000000000') == {}
    print('ok   sv_patches: Violet & Scarlet 4.0.0 share all 12 patch sites')


def test_launcher_save_isolation_wiring():
    launcher = (ROOT / 'runtime/ui/launcher.mm').read_text(encoding='utf-8')
    assert 'save_root_dir(' in launcher
    assert 'save_slot0_dir(' in launcher
    assert 'game_identity_save_root(' in launcher
    assert 'No Pokémon Violet or Scarlet game files' in launcher
    main_c = (ROOT / 'runtime/main.c').read_text(encoding='utf-8')
    assert 'fs_set_save_root(' in main_c
    assert 'game_identity_save_root(' in main_c
    print('ok   launcher.mm & main.c wired to game_identity_save_root')


def test_fs_save_isolation_runtime():
    c_src = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "game_identity.h"

void fs_set_save_root(const char *dir);
extern const char *g_save_dir;

int main(int argc, char **argv) {
    assert(argc == 2);
    assert(chdir(argv[1]) == 0);

    const uint8_t violet_id[8]  = {0x70, 0x9b, 0xfd, 0x66, 0x11, 0x52, 0x98, 0x64};
    const uint8_t scarlet_id[8] = {0x42, 0x1c, 0x54, 0x11, 0xb4, 0x87, 0xeb, 0x4d};
    char buf[128];

    const GameIdentity *vg = game_identity_find(violet_id);
    const GameIdentity *sg = game_identity_find(scarlet_id);
    assert(vg && vg->supported);
    assert(sg && sg->supported);

    assert(!strcmp(game_identity_save_root(vg, buf, sizeof buf), "save"));
    fs_set_save_root(buf);
    assert(!strcmp(g_save_dir, "save"));

    assert(!strcmp(game_identity_save_root(sg, buf, sizeof buf), "save/scarlet"));
    fs_set_save_root(buf);
    assert(!strcmp(g_save_dir, "save/scarlet"));

    // Resetting to NULL restores default "save"
    fs_set_save_root(NULL);
    assert(!strcmp(g_save_dir, "save"));

    // game_identity_for_dir detects both games from <dir>/exefs/main
    mkdir("vdir", 0755); mkdir("vdir/exefs", 0755);
    mkdir("sdir", 0755); mkdir("sdir/exefs", 0755);
    uint8_t nso[0x80] = {'N', 'S', 'O', '0'};
    memcpy(nso + 0x40, violet_id, 8);
    FILE *fv = fopen("vdir/exefs/main", "wb"); assert(fv); fwrite(nso, 1, sizeof nso, fv); fclose(fv);
    memcpy(nso + 0x40, scarlet_id, 8);
    FILE *fs = fopen("sdir/exefs/main", "wb"); assert(fs); fwrite(nso, 1, sizeof nso, fs); fclose(fs);

    const GameIdentity *vdir_g = game_identity_for_dir("vdir");
    const GameIdentity *sdir_g = game_identity_for_dir("sdir");
    assert(vdir_g == vg);
    assert(sdir_g == sg);
    assert(!strcmp(game_identity_save_root(vdir_g, buf, sizeof buf), "save"));
    assert(!strcmp(game_identity_save_root(sdir_g, buf, sizeof buf), "save/scarlet"));
    return 0;
}
"""
    # Compile minimal C test checking game_identity.h + fs_set_save_root logic
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        src = td_path / 'test_iso.c'
        exe = td_path / ('test_iso.exe' if os.name == 'nt' else 'test_iso')
        # Provide minimal fs_set_save_root stub matching runtime/services/fs.c
        stub = """
static char g_default_save_dir[128] = "save";
const char *g_save_dir = g_default_save_dir;
void fs_set_save_root(const char *dir) {
    if (dir && *dir) snprintf(g_default_save_dir, sizeof g_default_save_dir, "%s", dir);
    else snprintf(g_default_save_dir, sizeof g_default_save_dir, "save");
    g_save_dir = g_default_save_dir;
}
"""
        src.write_text('#include <stdio.h>\n' + stub + c_src, encoding='utf-8')
        cc = os.environ.get('CC', 'gcc')
        subprocess.run([cc, '-std=gnu11', '-Wall', '-Werror', f'-I{ROOT / "runtime"}', str(src), '-o', str(exe)],
                       check=True)
        scratch = td_path / 'scratch'
        scratch.mkdir()
        subprocess.run([str(exe), str(scratch)], check=True)
    print('ok   save root isolation & game_identity_for_dir runtime checks')


if __name__ == '__main__':
    test_sv_patches()
    test_launcher_save_isolation_wiring()
    test_fs_save_isolation_runtime()
    print('all scarlet support checks passed')
