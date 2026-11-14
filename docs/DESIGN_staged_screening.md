# Design: `ddsuite af`, a wrapper around AdaptiveFlow for ultra-large screens
Status: the original design, written before the build. The staged screen shipped as a mode of `vs` (v0.3.0), not as a separate `ddsuite af` tool: sections 12 to 14 say what changed and what was built and measured, and where they disagree with sections 1 to 11 they win. The feasibility test that gave the evidence (200 Enamine molecules, QuickVina, SLURM partition `ncpu`) was run in `dev/afvs_feasibility`, which has since been deleted.

## 1. Purpose

`ddsuite vs` docks every molecule in a library and is tested to 100,000 molecules. It cannot reach a billion
(about 21 million CPU-hours for QuickVina at exhaustiveness 16). AdaptiveFlow (AFLP for ligand preparation, AFVS for
screening) handles that scale by screening in stages: dock one or a few representatives of each of about 12 million
tranches, rank the tranches, and dock only the best tranches. This tool gives the lab one guided front end for it.

| | `ddsuite vs` | `ddsuite af` (this design) |
|---|---|---|
| Library size | up to a few million | up to billions |
| Method | dock everything | prescreen tranches, then dock the chosen ones |
| Poses | kept for every molecule | scores only (see 7.3) |
| Library | any SMILES or SDF file | your own SMILES (prepared by AFLP) or Enamine REAL Space |

The two tools stay separate. A natural handover: `af` finds the top hits, `vs` redocks them with poses.

## 2. What has and has not been shown

Shown, on the test library (200 molecules, 910 prepared forms):
- AFLP prepares a SMILES library on SLURM with RDKit and Open Babel only, no ChemAxon. 100 molecules took 1 to 2 minutes
  on 8 CPUs. It made 4.57 forms per molecule (stereoisomers and tautomers, AFLP defaults of up to 5 each).
- AFVS docks the prepared forms with QuickVina on SLURM. 910 of 910 succeeded. Two subjobs on 32 CPUs took 15 and
  16 minutes: 64 CPU-seconds per form.
- Scores agree with `vs`: best AFVS form minus the `vs` score has mean -0.18 kcal/mol; 171 of 199 molecules within
  0.5; AFVS better by more than 0.5 for 25 and worse for 3. The small advantage is expected, since AFVS keeps the best
  of several forms.

Not shown:
- Anything with the real library: access, S3 streaming, the 18-letter tranche IDs, the sparse prescreen set.
- The staged workflow (prescreen, postprocess, primary screen) and the neural-network filter. The code is public in
  AFVS (`afvs_postprocess_atg-prescreen.sh`, `afvs_prepare_atg-primary-screen*.sh`, `afvs_train_ml_classifier.py`).
  The filter needs PyTorch, which is not installed yet.
- Runs longer than 16 minutes, or beyond 2 subjobs.

## 3. Packaging

New repository `Chemical-Biology-STP/adaptiveflow_ddsuite` (name to be agreed), registry name `adaptiveflow`,
alias `af`.

```
adaptiveflow_ddsuite/
  pixi.toml, pixi.lock         one environment: python 3.12, rdkit, openbabel, pandas, pyarrow, boto3, jinja2, jq,
                               selfies, func_timeout (pypi); PyTorch (cpu) added in phase 3 for the ML filter
  af_cli.py                    dispatcher (like vs_cli.py): no arguments = wizard, *.yaml = run file, subcommands
  af_wizard.py, af_runfile.py  guided questions and run-file reader/writer
  af_stages.py                 the stage state machine (section 5)
  af_config.py                 writes all.ctrl, todo files, docking scenario folder, SLURM templates (section 6)
  af_library.py                own-library path: AFLP run, repack for AFVS, counts
  af_report.py                 ranking, top hits, cost report
  structure_prep.py, protein_prep.py, vs_ui.py    copied from `vs` (receptor and box preparation, prompts)
  third_party/AFVS, third_party/AFLP              upstream code at pinned commits, unmodified
  NOTICE.md                    GPL v2 notices
  test/
```

Reasons:
- ddsuite deploys one repository per tool and runs `install` inside it, so the tool must carry what it needs. Vendoring
  the two upstream repositories (about 100 MB, mostly docking executables) at pinned commits keeps installs repeatable.
- Upstream code is not edited. All the changes found in the feasibility test are made by generating files per run
  (section 7), so upstream updates can be taken by changing the pinned commit.
- License: AFVS and AFLP are GPL v2, as is our fork. The wrapper is GPL v2 too.
- The docking executables are the ones bundled in AFVS (`qvina02`, `smina`, Vina, PSOVina and others), run unchanged.

Registry file:

```toml
description = "Ultra-large virtual screening with AdaptiveFlow: prepare a library, prescreen tranches, dock the best"
category = "docking"
owner = "yipy"
aliases = ["af"]
repo = "git@github.com:Chemical-Biology-STP/adaptiveflow_ddsuite.git"
ref = "v0.1.0"
install = ["pixi", "install"]
environment = "default"
command = ["python", "{dir}/af_cli.py"]
```

## 4. What the user sees

```
ddsuite af                  guided questions, then writes run.yaml and starts the run
ddsuite af run.yaml         the same from a run file;  --check validates and prints the plan and cost, runs nothing
ddsuite af status [folder]  stage, work units done, failures, estimated time left
ddsuite af advance [folder] normally called by the pipeline itself; also resumes a run that was interrupted
ddsuite af results [folder] ranking, top N, tranche summary
ddsuite af redock [folder] --top 1000      hand the best molecules to `ddsuite vs` for poses
```

Guided steps (same style as `vs`, with the same plain-language prompts and a review page before anything runs):
1. **Protein and binding site.** Same as `vs`: a complex to clean up, or a ready receptor plus a reference ligand or
   centre. The receptor is converted to PDBQT.
2. **Library.** Own SMILES file, or Enamine REAL Space (needs credentials; see 6.4).
3. **Preparation** (own library only). Maximum stereoisomers and tautomers per molecule, pH. Defaults are chosen
   from the cost estimate (see 8): the AFLP defaults multiply the cost by 4.6, so the wizard asks.
4. **Docking.** Program (qvina02, smina, Vina 1.2, PSOVina), box size, exhaustiveness.
5. **Staging** (REAL Space, or an own library big enough to need it). Representatives per tranche, tranche filter,
   size of the primary screen, neural-network filter on or off.
6. **How to run.** Partition `ncpu`, CPUs per subjob, how many subjobs run at once, time limits.
7. **Review**, with the estimated CPU-hours, wall time, output size and job count. Then submit.

A run file mirrors these answers (`receptor`, `reference_ligand`, `library`, `prep:`, `docking:`, `staging:`, `run:`),
with the same rule as `vs`: everything the wizard asks can be set in the file, and the run stops with a clear message
when a decision is missing.

## 5. Run folder and stages

One folder per screen, never reused silently. All inputs are copied in, so jobs never read from the user's folders.

```
<run>/
  run.yaml                   the settings, as written by the wizard
  state.json                 stage, status of each stage, job ids, counts, timestamps
  input/                     receptor.pdbqt, box, SMILES library or tranche list
  prep/                      AFLP working folder and prepared collections        (own library)
  afvs_prescreen/            an AFVS working copy: tools/, workflow/, output-files/
  afvs_primary/              an AFVS working copy for the primary screen
  logs/                      job logs per stage
  results/                   ranking.csv, top_hits.csv, tranche_summary.csv, cost_report.txt
```

Stages (each ends by submitting the next one with `sbatch --dependency=afterok`, so no process has to stay alive):

| # | Stage | Own library | REAL Space | Runs where |
|---|---|---|---|---|
| 0 | Check inputs, prepare receptor, write configs | yes | yes | login node, seconds |
| 1 | AFLP: prepare the library | yes | no (already prepared) | SLURM array |
| 2 | Repack AFLP output for AFVS | yes | no | one small job |
| 3 | Prescreen: dock representatives per tranche | optional | yes | SLURM array |
| 4 | Postprocess prescreen, select tranches | optional | yes | one small job |
| 5 | Optional: train and apply the neural-network filter | optional | optional | one small job |
| 6 | Primary screen | yes (single stage if no staging) | yes | SLURM array |
| 7 | Ranking and report | yes | yes | one small job |

`af advance` reads `state.json`, checks the finished stage (counts of work units, failures, missing outputs), then
submits the next. A failed check stops the run with the reason in `state.json` and in `af status`; fixing it and running
`af advance` resumes where it stopped. Work units are skipped when their outputs exist, so a re-submit does not repeat
finished work.

The ddsuite rules apply: the job scripts call the environment's Python directly, and paths in them point at the
version folder, which stays for the 14-day grace period after an update.

## 6. Generated configuration

### 6.1 AFLP (own library)
`all.ctrl` is written from the template with: `job_storage_mode=sharedfs`, `slurm_partition=ncpu`, RDKit for
stereoisomers, tautomers and conformers, Open Babel for neutralisation and protonation, ChemAxon off (the
`conformation_program_1=molconvert` default needs a licence), `rdkit_stereoisomer_max_count` and
`rdkit_tautomer_max_count` from the wizard, `targetformats=pdbqt`. The SLURM template is rewritten to drop the account
and module lines and to use the tool's environment.

### 6.2 AFVS
`all.ctrl` is written with the settings of the feasibility test: `batchsystem=slurm`, `job_storage_mode=sharedfs`,
`data_storage_mode=sharedfs` for an own library or `s3` for REAL Space, `program_timeout` from the exhaustiveness (900 s
worked at 16; the default of 90 s is too short), one docking scenario folder with `config.txt` (receptor, box,
exhaustiveness), `dockings_per_subjob` chosen so that a subjob takes 10 to 30 minutes.

### 6.3 SLURM limits
The cluster has a per-user job limit (we saw `QOSMaxJobsPerUserLimit` with unrelated jobs queued). The wizard sets
`slurm_array_job_throttle` low by default (5), explains the trade-off, and the review page shows the number of subjobs.

### 6.4 REAL Space access
Needs registration and AWS credentials (not yet available to us). The tool never asks for keys in its prompts. It
reads them from the standard AWS files (`~/.aws/credentials`) and checks access in stage 0 by listing one prefix, so a
credential problem shows before any job is submitted. Open question: whether compute nodes can reach S3 (stage 0 tests
this from a one-CPU job).

## 7. Upstream issues found, and how the wrapper handles them

### 7.1 Handled by generating files
| Issue | Handling |
|---|---|
| SLURM templates contain an empty `--account=`, a virtualenv, `module load` and conda lines | Templates are rewritten per run |
| `program_timeout=90` kills dockings at exhaustiveness 16 | Set from exhaustiveness |
| AFLP wants three-part collection keys (`AA_AA_A`), AFVS summary code wants two (`AAAA_A`); with three parts the summary step crashed and no scores were saved | Stage 2 repacks to two-part keys |
| Open Babel 3 writes a `COMPND` line that QuickVina rejects; AFLP documents Open Babel 2.x | Stage 2 removes the line |
| `afvs_prepare_folders.py --overwrite` leaves old output files in place | Each run uses a new folder; no `--overwrite` |
| Default tranche regex and prescreen settings assume the 18-letter REAL Space IDs | Written per run; disabled for libraries without real tranches |

### 7.2 To fix in the wrapper, not yet understood
- One of 200 molecules (`enm13448`, a small acid) was prepared by AFLP but is absent from the AFVS results. The
  wrapper must count molecules in and out and report differences. Cause to be found.
- AFVS stores stdout logs for every docking (4.1 MB for 910 forms, about 4.5 KB each). At 12 million prescreen dockings
  that would be about 54 GB. Find a setting to turn this off, or delete the logs after each stage.

### 7.3 Limits to state in the documentation
- AFVS saves scores, not poses. Poses need a redock of the top hits (the `af redock` handover to `vs`).
- Each docked form is a row: with several forms per molecule, the ranking needs best-form-per-molecule logic (the
  wrapper does this in stage 7).
- Python 3.12 gives deprecation warnings from `tarfile.extractall` (it breaks on 3.14). The environment pins 3.12.

## 8. Sizing and the cost estimate

Measured: 64 CPU-seconds per docked form with QuickVina at exhaustiveness 16 on 22 Å box (Enamine molecules); `vs`
measured 75 CPU-seconds per molecule (99,982 molecules) for one form. Forms per molecule come from the AFLP
settings: 4.57 with the defaults. So the cost per molecule is forms × 64 s.

| Plan | Molecules | Forms each | CPU-hours | On 256 CPUs |
|---|---|---|---|---|
| Own library, no staging | 1 million | 1 | 18,000 | 3.0 days |
| Own library, no staging | 1 million | 4.57 | 81,000 | 13 days |
| REAL Space prescreen, 1 per tranche | 12 million | 1 | 213,000 | 35 days |
| Primary screen of 100 million | 100 million | 1 | 1.8 million | 290 days |

The last two rows show that even the staged approach needs far more than one node. The review page must show this
before submission, along with an upper bound on jobs and disk. It must also offer the levers: the tranche filter
(restrict by property before the prescreen), fewer forms per molecule, a lower exhaustiveness, and the neural-network
filter. The AdaptiveFlow authors ran on up to millions of cloud CPUs; on NEMO we are limited by the per-user job limit
and fair share, which the wizard should ask the user to check with their group.

## 9. Testing plan with the test library

1. **Own-library end to end (done by hand, to be automated):** `library_2k.smi` first 200 molecules → AFLP → repack →
   AFVS → ranking; compare with the `vs` pilot. Becomes an automated slow test.
2. **Stage state machine without a cluster:** unit tests with a fake `sbatch` and fake AFVS outputs: advance, resume
   after a failed stage, refusal on missing outputs, molecule count checks.
3. **Config generation:** snapshot tests of the generated `all.ctrl`, SLURM templates and todo files, so an upstream
   bump that changes them is caught.
4. **Tranche-format test library (to try):** AFLP can assign tranche IDs from molecular properties
   (`tranche_assignments=true` with RDKit-based types). If it produces 18-letter IDs, running AFLP on the 100,000
   molecule Enamine sample gives a small library in the real tranche layout. The prescreen, postprocess and primary
   screen could then be tried end to end on it, which is the closest local test of the ATG workflow before library
   access. Not yet tried; it might need the sparse-set script `afvs_gen_sparse.py`.
5. **REAL Space:** once credentials exist: access check, one tranche download, a small prescreen.

## 10. Build plan

| Phase | Content | Needs |
|---|---|---|
| 1 | Repository, environment, own-library path (stages 0, 1, 2, 6, 7), wizard and run file, tests 1 to 3, release v0.1.0 | nothing external |
| 2 | Staged workflow on the tranche-format test library (stages 3, 4), test 4, status and resume | phase 1 |
| 3 | Neural-network filter (stage 5, PyTorch), `af redock` handover to `vs` | phase 2 |
| 4 | REAL Space (S3 mode, access check) | library credentials |

Phase 1 is useful on its own: it gives a SLURM front end for AFLP and AFVS with multiple-form preparation, but for
libraries `vs` already handles, so its main value is as the base for phases 2 to 4.

## 11. Decisions needed

1. Name and registry alias: `adaptiveflow` / `af`? (Alternative: `ultralarge`.)
2. Separate repository as proposed, or a new mode inside `virtual_screening`? Separate keeps `vs` small and its
   environment free of PyTorch and the vendored GPL code; the cost is some duplicated receptor-preparation code.
3. Forms per molecule: AFLP defaults (4.57 forms, 4.6 times the cost) or 1 by default with an opt-in?
4. How much cluster capacity may one screen use (CPUs at once, total CPU-hours)? This sets the defaults and the
   warnings in the review page.
5. Start with phase 1 now, or wait for library access before building anything?

## 12. Update after review (decision 2: a mode inside `virtual_screening`)

The tool becomes a mode of `ddsuite vs`, not a separate tool. Changes to sections 3 and 4:

- No new registry entry and no new alias. The files `af_*.py` and `third_party/` go into the `virtual_screening`
  repository next to `vs_*.py`. `ddsuite vs` stays the single entry point; `vs_cli.py` dispatches `status`, `advance`,
  `results` and `redock` to the new modules.
- The first wizard question after the receptor becomes: "dock every molecule" (the current engine) or "staged screen
  with AdaptiveFlow". The wizard recommends the staged mode above a library size that the cost estimate makes
  impractical for the current engine (to be set from the numbers in section 8).
- Receptor and box preparation, `vs_ui.py` and the run-file reader are shared, not copied.
- The environment gains PyTorch (CPU) only when the neural-network filter is added (phase 3). The size of the
  addition has not been measured; the current environment is 1.1 GB.
- Releases are shared: a change to the staged mode ships with the whole tool, so the test suite (367 tests) must pass
  before every release, and the staged mode needs its own tests added to it.
- Resource limits seen on this cluster (`sacctmgr`): the `ncpu` QOS allows one user at most 3,000 CPUs and 3,000 jobs
  running at once; the partition has 23,296 CPUs in 91 nodes and a 7-day time limit.

## 13. Phase 2 as built (tranche strategy, in `vs_staged.py`)

Phase 2 was built without the REAL Space library: a 3,000-molecule experiment showed that AdaptiveFlow's tranche layout
can be produced from any library. What was built differs from section 5 in these points:

- **Tranches are made by the tool** with RDKit (AFLP's `tranche_assignments` needs ChemAxon and only renames files).
  Default properties mw, logp, hba, hbd; the other 14 of the 18 letters of the ID are always `A`; AFLP's default cutoffs.
- **Layout AFVS needs**: collection key `<TRANCHE18>_<7 digits>` (exactly one underscore), file
  `<folder>/<TRANCHE>/<7 digits>.tar.gz` with the folder `<7 digits>` inside (metatranche addressing). The sparse
  collections of the prescreen are `SFL_<7 digits>` with a `.listing` of `<tranche>_<collection>,<form>,<index>`.
- **Representatives**: exactly `representatives` forms per tranche (default 3, AdaptiveFlow allows 1 to 10), chosen by
  the md5 of the form name, so the choice does not depend on the input order.
- **Selection** is our own Python, not the upstream scripts (`dimension_averaging` has two bugs and needs IPython):
  mean prescreen score per tranche, best tranches first, whole collections until about `primary_size` forms.
- **Stages**: prepare (AFLP) -> prescreen (AFVS on `afvs_prescreen/`) -> select -> primary (AFVS on `afvs/`) -> rank.
  The strategy `all` keeps the phase 1 stages. `advance` and `retry` work for every stage.
- **Estimate** shown before submission: a sample of up to 3,000 molecules gives the tranche counts; for a bigger library
  the number of tranches is extrapolated with the bias-corrected Chao1 estimator, and the prescreen is
  `representatives x tranches`, capped at the number of forms.
- **Output**: `results/tranche_scores.csv`; a `tranche` column; molecules whose collection was not docked have status
  `not_docked`.

Measured on the 3,000-molecule library (see the README for the full figures): 910 of 2,993 forms docked (30%), the true
best 30/100/300 found 22/60/172 against 9/30/91 for a random 30%. A simulation over 500 draws of the representatives
gives 25/73/187 on average, so a single run varies by several molecules of the best 30. Known costs: the prescreen forms
of a selected tranche are docked again in the primary screen; the tranche library holds the compressed forms in memory
(about 3 GB per million forms) and is stored in addition to AFLP's output, so the approach is meant for libraries of up
to a few million molecules until the real REAL Space layout (already in tranches) is used in phase 4.

## 14. Phase 3 as built (poses and the neural-network filter)

- **`vs redock <run> --top N`** docks the best N molecules again in the ordinary full mode, with poses kept. The input is
  the best form of each molecule with its charge state kept (`ionisation: keep`), the protein, box, program,
  exhaustiveness and seed are those of the staged run; the run file is checked by the same reader as any other
  (`redock_input/run.yaml`), so it is also a template for a later run. `--report` joins the new scores with the staged
  ones (`results/redock_comparison.csv`). Staged runs now keep `input/receptor.pdb`.
- **Neural-network filter** (`staged.ml_filter`, tranche strategy only): trained on the prescreen, applied to the forms
  of the selected collections, which are repacked into `dock_ml/` with only the kept forms; the primary screen is
  pointed at that folder. Chosen over upstream's ligand-list input because the repacking reuses the layout already
  proven with AFVS and needs no new AFVS setting.
- **PyTorch was measured and not adopted**: `pytorch-cpu` added 1.3 GB to the 1.5 GB environment. The recipe of
  `ml_classifier.py` is implemented in numpy (`vs_mlfilter.py`). The models are not interchangeable.
- **Safeguards added to the upstream recipe**: 15% of the data held back to measure the model; the filter is not used
  (everything in the pool is docked) with fewer than 200 prescreen scores or an area under the curve below 0.6; unreadable
  SMILES are kept; the training set is capped at 200,000 forms; fresh (spawned) processes for the filter pool.
- **Measured** (3,000 molecules, README for the table): a pool of 1,935 forms filtered at 0.5 docked 1,260 forms and found
  28/86/244 of the best 30/100/300, where the tranche selection alone is expected to find about 24/74/221 for the same
  docking. The model was weak (area under the curve 0.71, 431 training forms), a cutoff of 0.7 did worse than no filter.
  Not measured: a large prescreen, more than one training per setting, the real library.
- **Known limits**: the filter step runs inside the `advance` job (4 processes). Prediction ran at about 14,000
  forms/s per process on short made-up SMILES (on a busy machine), training on 100,000 forms took 94 s. A pool of
  hundreds of millions of forms needs the step spread over a SLURM array (not built). The filter uses the same prescreen the tranches use, so it adds little information beyond them when the
  prescreen is small.
