# Vendored AdaptiveFlow code

Unmodified copies of the scripts and templates of two upstream repositories, used by the staged screening mode
(`vs_staged.py`). Both are released under the GNU GPL v2 (the licence files are in each folder), like this repository.

| Folder | Upstream | Commit | Date |
|---|---|---|---|
| `AFVS/` | https://github.com/QuantumAI4Bio/AdaptiveFlow-VS (branch develop) | `aadf971` | 2026-11-02 |
| `AFLP/` | https://github.com/QuantumAI4Bio/AdaptiveFlow-LP (branch develop) | `b201c28` | 2026-01-19 |

Left out: the docking executables of AFVS (`tools/bin`, 64 MB; the same programs are in `executables/` and are linked
into each run), the Docker files, the documentation and tutorials, and the AWS CloudFormation files.

Nothing in these folders is edited. `vs_staged.py` writes the configuration and the job templates of each run into the
run folder, and it checks that the lines it changes are still where it expects them, so an upstream update that moves
them stops with a clear message. To update: replace the folders with the new commit, change the table above, and run
the tests (`pixi run test-all`).
