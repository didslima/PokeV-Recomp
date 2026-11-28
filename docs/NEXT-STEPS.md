# PokeV-Recomp — Next Steps & Roadmap

Este documento reúne os próximos passos planejados para o **PokeV-Recomp**, incluindo bugs em aberto, novas funcionalidades e o plano técnico para melhorias visuais inspiradas no Nintendo Switch 2.

---

## Concluído Recentemente

- [x] **Suporte a pacotes DLC no RecompLauncher (PR #72, merge `084d2311`, release `v0.2.1`):**
  - Suporte completo a importação, detecção incremental e exibição de pacotes DLC (`aoc/*.bin` + `.dlc_stamp`) no `RecompLauncher` e `zaimport` (`tools/zaimport/ZaImportLogic.cs`, `runtime/ui/launcher.mm`).
  - PR #72 merged em `main` e publicado na release `v0.2.1` (`4100b401`).
  - Build de teste validada pelo usuário; o `RecompLauncher` reconheceu corretamente o DLC.
- [x] **Correção de cinematic Bink em ultrawide (PR #69, merge `651d51e2`):**
  - Detecção de vídeo Bink planar mesmo com `depth_test == true`, centralização 16:9 e barras pretas laterais sólidas na apresentação final (`movie_boxed_present_scissor()` + `movie_boxed_frame`), preservando gameplay normal.
- [x] **Release `v0.2.0-rc.2` (pre-release) e promoção para `v0.2.0` (estável) publicadas:** https://github.com/didslima/PokeV-Recomp/releases/tag/v0.2.0 (`PokeV-Recomp-windows-x64.zip`, `PokeV-Recomp-linux-x86_64.AppImage`, `SHA256SUMS`).

---

## Próxima Prioridade: Investigação de Distorção nas Transições de Cutscenes Renderizadas

- **Relatório completo:** [docs/investigations/CUTSCENE-TRANSITION-DISTORTION.md](investigations/CUTSCENE-TRANSITION-DISTORTION.md) (Resultado: **Outcome A — causa raiz confirmada e correção seletiva validada em runtime**).
- **Causa raiz confirmada em runtime (`frames 1865–1882`, saída da cutscene de cura em Los Platos):**
  - Durante transições/crossfades em cutscenes renderizadas, o jogo captura um snapshot da cena em uma textura GPU (`1920x1080`, `gpu_owned=1`, `tex_sc=(1.00, 1.00)`) e a desenha via `ui2d` fullscreen (`ubo=560`, `vtx=6`, `blend=1`).
  - `widescreen_ui_boxed_draw()` (`runtime/hle/widescreen.c`) aplicava `ui_static_16_9_tex()` a esse snapshot GPU, chamando `vp_box_16_9()` (`runtime/nvn/vulkan/exec.c`) e achatando o snapshot 32:9 dentro de uma caixa 16:9 centralizada sobre a cena 32:9 ao vivo (causando achatamento 2x, costuras verticais e ghosting translúcido no crossfade).
  - Testado seletivamente com `LARECOMP_TRANS_DIAG_BYPASS=1` (ignorando `vp_box_16_9()` apenas quando `!widescreen_ui_pane_partial(...) && t0_b->gpu_owned`): distorção eliminada por completo, preservando a Tela Título 16:9 (`gpu_owned=0`), o HUD 16:9 no gameplay 32:9 e os testes `test_movie_box.py` / `test_widescreen_proj.py`.
- **Próximo passo:** Implementar a correção definitiva em `widescreen_ui_boxed_draw()` / `exec.c` (restringir `ui_static_16_9_tex()` a texturas estáticas não-`gpu_owned`), remover a instrumentação `LARECOMP_TRANS_DIAG*` e adicionar teste unitário.

---

## Visão Geral das Próximas Etapas Posteriores

1. **Melhorias padrão Switch 2** *(detalhado na seção [Project: Switch 2 Visual Enhancements](#project-switch-2-visual-enhancements) abaixo)*
   - Avaliação técnica da arquitetura de renderização atual e plano incremental de melhorias de filtragem de texturas, LOD, resolução/upscaling, sombras/iluminação e HDR.

2. **Investigar frame rate acima de 60 fps**
   - Avaliar viabilidade de 90/120+ fps (ou destravado) sobre os patches atuais de 60 fps (`tools/sv_patches.py`: `_sv_frame_clock`, `_sv_frame_pace`, `_sv_lua_frame_time`, `_sv_lua_clock_slot`, `_sv_timeline_time`), física, animações, lógica de câmera e frame pacing (`runtime/hle/framepace.c`, `runtime/nvn/pace.c`).

3. **Suporte para Pokémon Scarlet** *(implementado em `feature/pokemon-scarlet-support` — aguardando CI build e validação em runtime; ver seção [Investigação e Implementação: Suporte a Pokémon Scarlet](#investigação-e-implementação-suporte-a-pokémon-scarlet) abaixo)*
   - **Status atual:** Extração testada, análise binária concluída (equivalência comprovada com Violet 4.0.0), suporte de identidade/patches/isolamento de saves implementado e testado em suíte automatizada; **ainda pendente** recompilação completa + validação de boot e gameplay em runtime.

4. **Travamento ao tentar conectar na internet**
   - Investigar os serviços HLE de rede/conta/amigos (`runtime/services/acc.c`, `misc.c`, stubs de `nifm`/`nsd`/`ldn`/`ssl`) que causam travamento ou espera infinita quando o jogo tenta iniciar conexão online.

5. **Integração de troca de Pokémon entre Switch real e o recomp**
   - Pesquisar protocolo local/LAN (`ldn` / `lan-play` / troca local ou manipulação compatível de protocolo) para viabilizar trocas entre um console Switch real e o PokeV-Recomp.

6. **Cheats (Master Ball, shiny rate, etc.)**
   - Definir arquitetura segura e opcional (menu F1 ou módulo HLE dedicado, sem afetar saves por padrão) para cheats como Master Ball infinita, modificador de taxa de shiny, entre outros.

---

## Investigação e Implementação: Suporte a Pokémon Scarlet

> **Status:**
> - **Extração (`zaimport`):** Testada e funcional (Base + Update 4.0.0 + DLC `0100a3d008c5d001.bin`).
> - **Análise binária (Scarlet vs Violet 4.0.0):** Concluída — equivalência de endereços comprovada byte a byte.
> - **Identidade, patches, isolamento de saves e testes:** Implementados em `feature/pokemon-scarlet-support`.
> - **Build completo (CI) e validação em runtime (boot + gameplay):** **Pendentes** — não marcar como jogável até validação em runtime.

### Resultados da Análise Binária (`exefs/main` Scarlet `421c5411b487eb4d` vs Violet `709bfd6611529864`)

A descompressão LZ4 e comparação byte a byte dos módulos `exefs/main` (4.0.0) provaram que ambos os jogos são compilados a partir do mesmo binário base:

- **`.text` (`0x343fc90` bytes):** apenas **2 instruções** diferem (`0x700149b9c8`: `mov w1, #1` vs `mov w1, wzr`; `0x700149b9d4`: `mov w0, #0x33` vs `mov w0, #0x32`).
- **`.rodata` (`0xf420a0` bytes):** apenas **5 palavras** diferem (`0x700438208c..0x700438209c`), correspondentes ao GNU Build ID de 20 bytes.
- **`.data` (`0x368868` bytes):** **0 bytes** de diferença.
- **`sdk`, `subsdk0`, `rtld`:** **idênticos byte a byte** (mesmos SHA-256).
- **Todos os 12 sites de `tools/sv_patches.py`:** 100% idênticos (instrução alvo e janela de 132 bytes ao redor).
- **Todos os 13 endereços hardcoded em HLEs** (`npc_detail.c`, `npc_shown.c`, `heaps.c`, `framepace.c`, `libc.c`): 100% idênticos.

### Arquitetura e Isolamento de Saves Implementados

- **`runtime/game_identity.h`:** Registrado `Pokémon Scarlet 4.0.0` (`421c5411b487eb4d`, `supported = true`, `save_subdir = "scarlet"`), preservando Violet `709bfd6611529864` com `save_subdir = ""`. Adicionados `game_identity_save_root()` e `game_identity_for_dir()`.
- **`tools/sv_patches.py`:** Adicionado `'421c5411b487eb4d': '4.0.0'` em `SV_BUILDS`, compartilhando os 12 patches verificados com Violet 4.0.0.
- **Isolamento de saves (`runtime/services/fs.c`, `runtime/main.c`, `runtime/ui/launcher.mm`):**
  - **Violet:** mantém `save/save_0` (retrocompatibilidade total, sem migração destrutiva).
  - **Scarlet:** usa `save/scarlet/save_0` (e backups em `save/scarlet/backup-*`).
  - `LARECOMP_SAVE_DIR` continua sobrescrevendo o diretório de save para scripts de teste.
- **Testes automatizados:** `tools/test_game_identity.py` atualizado para ambos os jogos + novo `tools/test_scarlet_support.py`.

### Próximos Passos para Validação Final

1. Executar build completo via CI (Linux AppImage + Windows `.zip` + macOS) na branch `feature/pokemon-scarlet-support`.
2. Testar recompilação (`lasgen` + `gamebuild`) com dump de Pokémon Scarlet 4.0.0 em instalação limpa.
3. Validar em runtime: boot, tela título, seleção de personagem, overworld, batalhas, Centro Pokémon, save/load em `save/scarlet/save_0`, DLC (`0100a3d008c5d001.bin`), 60 FPS e ultrawide.

---

# Project: Switch 2 Visual Enhancements

## Objective

Investigate and implement visual improvements in PokeV-Recomp inspired by the Nintendo Switch 2 version (Pokémon Scarlet/Violet 4.0.0).

The goal is to reproduce technically feasible visual improvements while preserving compatibility, stability, and performance.

## Important Guidelines

- Do not assume Switch 2 uses higher-resolution textures.
- Do not replace original game assets without evidence that replacement assets provide measurable improvements.
- Prioritize renderer improvements over asset modifications.
- Preserve existing game functionality and compatibility.
- Avoid unnecessary architectural changes.
- Keep all changes compatible with the project's existing platform architecture.

## Investigation Priorities

### 1. Texture Filtering
- Investigate anisotropic filtering support.
- Identify current texture sampling configuration.
- Evaluate improvements for ground surfaces, roads, and distant textures.
- Determine whether filtering can be configured independently.

### 2. Level of Detail (LOD)
- Investigate Pokémon model LOD behavior.
- Identify distance-based model switching.
- Compare current behavior with publicly documented Switch 2 improvements.
- Determine whether higher-quality models can be maintained at greater distances.

### 3. Rendering Resolution and Upscaling
- Investigate internal rendering resolution.
- Identify opportunities for higher-resolution rendering.
- Evaluate compatibility with NVIDIA DLSS, AMD FSR, and other upscaling technologies where technically appropriate.
- Avoid introducing vendor-specific dependencies unnecessarily.

### 4. Shadows and Lighting
- Investigate shadow filtering and resolution.
- Identify limitations in the current rendering pipeline.
- Evaluate improvements without introducing significant performance overhead.

### 5. HDR Support
- Investigate whether the rendering pipeline supports HDR.
- Identify required changes to color space, output formats, and display configuration.
- Treat HDR as an optional feature.

## Workflow

For each priority:

1. Inspect the existing implementation.
2. Identify relevant source files and rendering subsystems.
3. Document current behavior.
4. Research the corresponding Switch 2 behavior using reliable public sources.
5. Evaluate technical feasibility.
6. Propose an implementation plan.
7. Implement only after documenting the expected benefits and risks.
8. Add or update tests where applicable.
9. Benchmark performance before and after changes.

## Performance Requirements

- Maintain stable frame pacing.
- Avoid unnecessary VRAM consumption.
- Consider different GPU capabilities.
- Provide configurable quality settings whenever appropriate.
- Do not sacrifice compatibility for visual improvements.

---

## Technical Assessment of Current Rendering Architecture

> **Status:** Initial architecture inspection completed (no renderer modifications performed yet).

### 1. Texture Filtering
- **Current behavior:**
  - `runtime/nvn/vulkan/textures.c` (`vk_sampler_settings`, `sampler_make`, `sample_bias_ok`) já implementa filtragem anisotrópica 16x e filtragem trilinear quando `g_settings.aniso16` (`runtime/settings.h`) está ativo: samplers mipmapped lineares são promovidos para `VK_SAMPLER_MIPMAP_MODE_LINEAR` com `anisotropyEnable = VK_TRUE` e `maxAnisotropy = 16` (excluindo samplers de sombra/comparação de profundidade).
  - `vk_sampler_settings(os.lines)` aplica `mipLodBias` negativo (até `-2.0`, desativável via `LARECOMP_NO_LOD_BIAS`) apenas quando `ratio < 1` (downscale) em samplers com wrapping repetido (`REPEAT` / `MIRRORED_REPEAT`).
- **Relevant files:** `runtime/nvn/vulkan/textures.c`, `runtime/nvn/vulkan/device.c`, `runtime/settings.h`, `runtime/ui/menu.mm`.
- **Technical limitations & opportunities:**
  - `aniso16` hoje é binário (`Game's` vs `High` 16x); não há níveis intermediários (4x/8x) nem controle independente de `mipLodBias` para melhorar nitidez de solo/estradas à distância em resolução nativa ou upscale.
- **Estimated complexity:** Baixa.
- **Risks:** `mipLodBias` agressivo em texturas não-materiais pode causar shimmering/aliasing se `sample_bias_ok()` não filtrar corretamente; impacto mínimo em GPUs modernas, moderado em GPUs integradas com largura de banda limitada.

### 2. Level of Detail (LOD)
- **Current behavior:**
  - `runtime/hle/npc_detail.c` já intercepta o gerenciador de LOD do Pokémon Violet 4.0.0 (`LOD_GLOBAL = 0x46d4e00`, `LOD_MGR_VTABLE = 0x44a00b8`, `trinity_ModelLodSetting` para `"chara_model"` e `"pokemon"`) e escala as distâncias `float[3]` conforme `g_settings.world_detail`, `g_settings.npc_detail` e `g_settings.pokemon_detail` (`LARECOMP_NPC_DETAIL` / `LARECOMP_POKEMON_DETAIL`).
  - `runtime/hle/npc_shown.c` controla o limite de NPCs visíveis simultaneamente (`g_settings.npc_shown`).
- **Relevant files:** `runtime/hle/npc_detail.c`, `runtime/hle/npc_shown.c`, `runtime/settings.h`, `runtime/ui/menu.mm`.
- **Technical limitations & opportunities:**
  - Os endereços em `npc_detail.c` são específicos de Pokémon Violet 4.0.0 (precisam de validação para Scarlet).
  - Falta avaliar LODs de cenário/folhagem/objetos (`world_detail`), LODs de shadow casters e impacto de CPU/draw calls em áreas abertas de Paldea ao elevar distâncias no estilo Switch 2.
- **Estimated complexity:** Média.
- **Risks:** Aumentar distâncias de LOD sem controle eleva significativamente draw calls na CPU e vértices/VRAM; em áreas densas pode agravar quedas de FPS.

### 3. Rendering Resolution and Upscaling
- **Current behavior:**
  - `runtime/nvn/output_scale.c` gerencia escala interna (`RES_MATCH`, `1..8x`, `RES_DYNAMIC`, `RES_Q75/Q50/Q25`, `RES_X15`).
  - `runtime/nvn/vulkan/exec.c` (`post_process`) aplica pós-processamento na apresentação: SMAA (`AA_SMAA`), AMD FSR 1 (`UPSCALE_FSR1`, EASU + RCAS), Intel XeSS (`UPSCALE_XESS`, `runtime/nvn/vulkan/xess.cpp`, `third_party/xess/`), e box downscale para supersampling (`RES_X15`, `2x..8x`).
  - `runtime/nvn/vulkan/fsr2.cpp` (AMD FSR 2.2.1) e `runtime/nvn/vulkan/dlaa.c` (NVIDIA DLAA) atuam no passe temporal da cena (`fsr2_copy` / `TAA_*`).
- **Relevant files:** `runtime/nvn/output_scale.c`, `runtime/nvn/vulkan/exec.c`, `runtime/nvn/vulkan/fsr2.cpp`, `runtime/nvn/vulkan/xess.cpp`, `runtime/nvn/vulkan/dlaa.c`, `runtime/settings.h`.
- **Technical limitations & opportunities:**
  - `UPSCALE_FSR1` / `UPSCALE_XESS` rodam no estágio de apresentação (`post_process`), não no passe 3D principal com motion vectors/depth (onde apenas `TAA_FSR2` e `TAA_DLAA` operam hoje).
  - Avaliar qualidade/estabilidade de `TAA_FSR2` e `TAA_DLAA` em SV 4.0.0, suporte a presets equilibrados de resolução dinâmica e evitar dependências obrigatórias de um único fabricante de GPU.
- **Estimated complexity:** Média a Alta (se integrar upscaling temporal no passe 3D; Baixa/Média para presets e ajustes de pipeline existentes).
- **Risks:** Motion vectors ou jitter incorretos em passes específicos de SV podem gerar ghosting/artefatos em vegetação e partículas; XeSS/DLAA dependem de bibliotecas externas/hardware específico se não houver fallback limpo.

### 4. Shadows and Lighting
- **Current behavior:**
  - `runtime/nvn/vulkan/textures.c` (`shadow_res` 1–4x, `LARECOMP_SHADOW_SCALE`) permite escalar a resolução dos shadow maps.
  - `runtime/nvn/vulkan/exec.c` implementa `shadow_sharp` (1–4x, estreitamento do filtro de sombra), `gtao` (XeGTAO 0–3), `ssgi` (`ssgi_apply` 0–3) e reflexos de água (`water_ssr`, `water_mode`, `water_depth`, `water_caustics`, `water_vivid`).
- **Relevant files:** `runtime/nvn/vulkan/textures.c`, `runtime/nvn/vulkan/exec.c`, `runtime/nvn/maxwell/translate.c`, `runtime/settings.h`.
- **Technical limitations & opportunities:**
  - Os recursos já existem no runtime, mas precisam de validação específica em SV 4.0.0 (alinhamento de cascatas de sombra, shadow acne/bias, custo de XeGTAO/SSGI no mundo aberto de Paldea e interação com passes HDR `fmt=122`).
- **Estimated complexity:** Média.
- **Risks:** Aumentar `shadow_res` multiplica consumo de VRAM e largura de banda de profundidade; XeGTAO/SSGI em níveis altos podem reduzir FPS em GPUs intermediárias se não forem opcionais/calibrados.

### 5. HDR Support
- **Current behavior:**
  - `runtime/nvn/vulkan/device.c` detecta capacidade HDR do display (`g_vk.hdr_capable`) para `VK_COLOR_SPACE_HDR10_ST2084_EXT` e `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT` (`VK_FORMAT_R16G16B16A16_SFLOAT`), controlado por `g_settings.hdr` (`LARECOMP_NO_HDR`).
  - `vk_hdr_mark_base()` separa a imagem do jogo da UI/overlay para que `vk_hdr_encode()` em `exec.c` mantenha o overlay em `hdr_paper` nits e mapeie highlights até `hdr_peak` nits.
- **Relevant files:** `runtime/nvn/vulkan/device.c`, `runtime/nvn/vulkan/exec.c`, `runtime/settings.h`, `runtime/ui/menu.mm`.
- **Technical limitations & opportunities:**
  - O pipeline de HDR já existe como recurso opcional no Windows; falta validar o comportamento em Linux (Wayland/Gamescope `VK_EXT_swapchain_colorspace`) e a interação entre o render target HDR interno do jogo (`VK_FORMAT_B10G11R11_UFLOAT_PACK32`, `fmt=122`), o tone-mapping (`ee1726d3a6be2536`) e `vk_hdr_encode`.
- **Estimated complexity:** Baixa a Média.
- **Risks:** Tone-mapping duplo ou clipping de highlights se o espaço de cor da saída do jogo (`fmt=43`/`fmt=37`) já tiver sido comprimido antes de `vk_hdr_encode`; suporte a HDR no Linux varia conforme compositor/driver.

---

## Recommended Implementation Order & Milestones

1. **Milestone 1 — Baseline & Profiling (SV 4.0.0):** Medir frame time, VRAM e gargalos atuais em Paldea (`LARECOMP_VK_PROF=1`, `[slow]`) com configurações padrão vs `aniso16` / `shadow_res` / `pokemon_detail`.
2. **Milestone 2 — Texture Filtering (Baixo risco / Alto retorno):** Avaliar perfis independentes de anisotropia (4x/8x/16x) e `mipLodBias` seguro para terreno/estradas em `textures.c`.
3. **Milestone 3 — LOD Tuning (Médio risco):** Validar multiplicadores ideais de `world_detail`, `npc_detail` e `pokemon_detail` em `npc_detail.c` para manter modelos de alta qualidade a maiores distâncias sem estourar frame pacing.
4. **Milestone 4 — Shadows & Lighting Validation (Médio risco):** Calibrar `shadow_res`, `shadow_sharp`, XeGTAO e SSGI especificamente para SV 4.0.0 com presets claros no menu F1.
5. **Milestone 5 — Resolution & Upscaling Evaluation (Médio/Alto risco):** Validar estabilidade de FSR 2 / DLAA / XeSS / resolução dinâmica em SV 4.0.0, garantindo fallbacks portáveis sem travamento de fabricante.
6. **Milestone 6 — HDR Verification (Opcional):** Validar HDR10/scRGB no Windows e viabilidade no Linux sem regressões em SDR.

## Testing & Benchmarking Requirements

- Manter a suíte Linux (`tools/test_*.py`) e adicionar testes unitários para quaisquer novas opções de configuração (`test_settings.py`) ou regras de sampler/escala.
- Para cada milestone implementado: registrar benchmark antes/depois (FPS médio, 1% low, frame time por categoria via `LARECOMP_VK_PROF=1` e uso de VRAM) em cena aberta e em cidade, nos modos 30 fps e 60 fps.
