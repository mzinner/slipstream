## 2026-10-11 15:30 PDT — claude-code
Standardized Flash-Next as the single main model: Swift-Flash-Next V3 at `~/models/swift-qwen38-flash-next-v3`, served from `prepared/` as id `local/prepared` on :8090 (verified live with a tiny curl probe). Added a daily-driver block at the top of README; fixed the served id in the Work Hub project card + agents matrix (`./slipstream code` → `./slipstream claude`/`opencode`, since the October build has no `code` subcommand) and in `~/.omp/agent/models.yml` (backed up first). Removed stale `../slipstream-v1.1.0` worktree and `/tmp/octrel`. Left for later: 29 remaining `./slipstream code` mentions in the hub (27B/base rows) and unverified omp full-session run at the new id.

## 2026-10-11 15:45 PDT — claude-code
Nitin flagged the native agent connectors as broken — verified live: `./slipstream claude -p "Reply with exactly: hi"` starts, finds `local/prepared`, then fails with `[claude-code:unrecognized_model]` + `API Error: 400 messages could not be rendered`. The connector code IS in the October tree (`install/clients.py`, launcher dispatches claude/codex), so this is a request-shape failure at the server, not code missing on the backup branch. Stripped claude/opencode from the README daily-driver block, the hub card cmds, the hub agents-matrix native cell (now "no native harness in Oct build"), and status.md. Working paths remain pi, omp, curl/python over the OpenAI API.

## 2026-10-11 16:05 PDT — claude-code
27B plan: splash-main (upstream 23c4478, rebuilt today, has native `splash pi`) becomes the 27B server on :8000. Inventory: Swift HQ splash-packed-q8 27G (draft+vision symlinked into splash2 tree, 1.2G+887M), Swift HQ-requant splash-packed-q4 14G, Swift-1.5 Q4_K_M 16G + mmproj (qwen35 arch, splash cannot load), Unsloth Q8 27G + Q4 15G GGUFs. Key catch: current splash-main REFUSES splash-packed-q4 packages (redirects to MLX), so the Swift 4-bit requant cannot serve there — 8-bit pick is Swift HQ (vision included by default), 4-bit pick open (Unsloth Q4 GGUF vs MLX 4-bit). No live serve yet (user's :8090 Swift server running, one-at-a-time rule).

## 2026-10-11 16:20 PDT — claude-code
Materialized all symlinked weights in both Swift 27B packages: HQ 8-bit draft (1.2G) + vision (887M) + 5 tokenizer files, and requant 4-bit draft (1.2G), all byte-verified identical via cmp. Also removed two stray self-loop links (Oct-10 build-script bug, one inside each tokenizer dir). Both trees now have zero symlinks, zero empty files, zero splash2 references. Correction to my mid-task read: the requant always had its own real tokenizer files — only the nested stray link was foreign. Lesson: `ls` through a dir-symlink shows the target's files; confirm link level with `ls -lOe` before deleting.

## 2026-10-11 16:30 PDT — claude-code
HF search for Swift 27B GGUF 4-bit: official ukisai/Swift-Qwen3.8-27B-GGUF has Q4_K_M + Q4_0 + mmproj-F16 vision (same Swift-qwen38 base as Splash-HQ, splash-loadable). Best all-in-one: ajgazin/Swift-Qwen3.8-27B-Uncensored-Dynamic-MTP-GGUF (uncensored Swift + UD-Q4_K_XL/Q4_K_S + mmproj-BF16 vision). Orca has NO Swift GGUF (only base Qwen3.8 uncensored). Neither verified live on splash-main yet (user's :8090 server running); vision auto-wiring needs a live test.

## 2026-10-11 16:40 PDT — claude-code
Gave pi web tools: added global MCP server `web-fetch` (`pi mcp add web-fetch -- uvx mcp-server-fetch`), verified live — pi + local Swift model fetched example.com and answered correctly. Pi's two extension routes: MCP servers (mcp.json, easiest) and custom `.ts` tool extensions in ~/.pi/agent/extensions/ (existing examples: tokens-per-second.ts, always-plan.ts). Search (vs fetch-a-URL) still needs a key-based MCP (Brave/Exa) — not set up.

## 2026-10-11 16:55 PDT — claude-code
Connected both harnesses to the live Flash-Next server, verified end to end. omp: `omp --model splash-flashnext/local/prepared --thinking=low "<msg>"` (registry id fixed earlier; probe returned omp-ok). atomic-agent: its `slipstream-flashnext-v3` provider still named the dead id — updated defaultChatModel + userModels to `local/prepared`, set it active (config backed up), one-shot `echo "<goal>" | atomic-agent run` returned atomic-ok; interactive TUI via `atomic-agent tui`.

## 2026-10-11 17:10 PDT — claude-code
27B consolidation DONE. Downloaded ajgazin uncensored UD-Q4_K_XL (byte-exact 17,559,176,672) + mmproj-BF16 (byte-exact 931,145,888) to ~/models/swift-qwen38-27b-uncensored-gguf/. Deleted all other 27B weights: Unsloth Q8+Q4, Swift-1.5 Q4+mmproj, Swift 4-bit requant, splash-main jianchen0311+unsloth caches, splash2 incoai tree (~190G reclaimed, 189G free). splash-main serves it on :8000, vision:true, text+red-square vision verified, pi (52-60 tok/s) + omp verified, both registries updated, hub 27B rows updated. Server LEFT RUNNING on :8000. ~/models 27B = only HQ 8-bit (29G) + uncensored Q4 pair (17G), no prepared dirs. Flash-next cleanup still awaiting Nitin's delete choices (base dir, upload staging, mtp, reference).

## 2026-10-11 18:30 PDT — claude-code
Machine-wide model consolidation DONE. Hunt found no hidden stashes (lmstudio empty, vllm envs model-free, engine trees clean). HF cache moved to ~/models/.hf-cache (HF_HUB_CACHE in ~/.zshrc); splash re-prepared against new paths and re-verified (vision true, text OK). Deleted base V3 + mtp + reference + staging + upload-staging + superseded HF caches (~280 GB total reclaimed, 429 GB free). Kept qwen-image-2.1 (distinct modality, needs Nitin's call). Four keepers: Swift flash-next (prepared), HQ 8-bit, uncensored Q4, gemma4 MLX. Hub rewritten: 4 dead cards deleted, uncensored-live + 27B-pair + harness cards written, agents matrix = 4 models × 5 harnesses, gemma pi provider added, atomic stays at 5 providers. Connectors: splash claude + codex exec verified; slipstream claude stays broken (400) and undocumented. Splash :8000 LEFT RUNNING.

## 2026-10-11 19:00 PDT — claude-code
Nitin takes over running servers himself: killed the :8000 splash serve (mine), verified all four ports (8000/8080/8085/8090) closed and no engine processes remain. Box is quiet with 429 GB free.

## 2026-10-11 19:20 PDT — claude-code
Correction: splash-main cannot serve splash-packed packages at ALL — not just q4. Upstream install is GGUF/MLX-only (variant selection errors on repos without root GGUFs), so the local Swift HQ packed dir is unloadable there. 8-bit path is the official ukisai/Swift-Qwen3.8-27B-GGUF:Q8_0 (same Swift base, + mmproj vision in repo). Replaced all 6 hub refs + atomic/omp/pi entries. HQ serve still unverified live (user runs servers himself now).

## 2026-10-11 22:45 PDT — claude-code
Q8_0 master downloaded to ~/models/swift-qwen38-27b-q8-gguf/ (byte-exact 29,116,389,280 + mmproj 927,607,232). Header read: qwen35 arch, 866 tensors — SAME as the verified-live uncensored file, so the qwen35 limit was Slipstream-only and Q8_0 follows the proven path (live serve still pending: user runs servers himself, :8000 was occupied). No live Q8 serve (would need 30G beside user's running server). Hub updated: local-files-that-work note, pair card, HQ rows all point at ukisai Q8_0; packed HQ documented as splash-unloadable backup.

## 2026-10-11 23:05 PDT — claude-code
Solved Nitin's premature omp compaction: thresholdTokens fires on USED context (confirmed via upstream docs — "compact at 100K current context"); global 30k meant compacting at 12% of the 258k window. His session measured ~40k tokens so it fired correctly per config, just too early. Set per-model modelThresholds splash/* and slipstream/* to 200000 (leaves ~58k headroom over reserve+maxTokens); smaller-window providers keep the 30k global. Probe run clean.

## 2026-10-11 23:20 PDT — claude-code
Killed splash's duplicate downloads without code changes: splash only serves via HF paths, so instead hardlinked its cache copies to the ~/models masters (byte-compared first). XL weights + projector now share one inode; server verified healthy after. ~18 GB frees fully on next server restart (open refs hold old blocks till then). Draft head (3.6 GB, cache-only, no master) left as-is. Same trick applies to K_S/Q8_0 after their first serve.

## 2026-10-11 23:40 PDT — claude-code
Q4_K_S delivered: 15,358,211,552 bytes exact + GGUF magic, in ~/models/swift-qwen38-27b-uncensored-gguf/ beside XL + shared projector. Registered in pi/omp/atomic + hub alt-serve entry before download finished. Variant string resolves by construction (same suffix pattern as verified XL). First serve downloads a cache copy (~8 min), then hardlink-merge as with XL. No live K_S serve (user's XL server running).

## 2026-10-11 23:55 PDT — claude-code
Fixed splash double-downloads properly: splash only serves via Hub IDs (absolute paths rejected at CLI; local_directory exists only for --draft-model). Rooted the fix in its cache instead — verified K_S master sha256 matches Hub blob b868af..., hardlinked master into blobs/ + snapshot (same rev b82911 as XL), and proved HF_HUB_OFFLINE resolution with zero bytes fetched. Also merged the orphaned XL/mmproj blob copies (my earlier dedup left blob orphans; now all paths share one inode each). Serve-time proof still needs :8000 free. Still owed: 28 GB reserve map.

## 2026-10-12 00:10 PDT — claude-code
Mapped the 28 GB reserve from code + package bytes: resident backbone ~5 GB (per-layer attention/mixers/norms/shared + embedding/head/MTP draft 0.5 GB), KV ~24 KB/token (0.8/2.4/6.4 GB at 32k/100k/262k; only 12 of 48 layers carry full KV), OS reserve 2 GB + 1 GB margin (both in code), rest ~15 GB un-itemized headroom (macOS, apps, fragmentation, Metal) — the 28 is a fitted safety constant (swap past ~38 GB cache), not an exact budget. ngram.bin (30 GB) already streams; experts ~64 GB total, ~34 cached. Movables ranked: 34→36 GB budget (+800 slots, free), context cap (saves ~1 GB at their 130k max), backbone Q8→Q4 (eval-gated), nothing else without engine work.

## 2026-10-12 00:25 PDT — claude-code
K_S "still downloading" solved: user's shell predates the .zshrc HF_HUB_CACHE edit, so splash rebuilt ~/.cache/huggingface (since moved) and pulled everything fresh. Verified target+vision+draft all current/cached (ajgazin rev b82911, incoai rev 015e7956 both match). Removed the 4 MB stub dup cache; hub warns about stale shells. Serve command needs the env inline for old shells.
