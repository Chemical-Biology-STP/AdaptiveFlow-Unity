# VF Unity
Streamlined version of AdaptiveFlow combining both AFVS and AFLP

## Quick start

> **Using the lab's Drug Discovery Suite?** Then there is nothing to install or clone. Run `ddsuite vs` from the
> folder with your files (see the suite's README for the one-time `ddsuite setup`). The `pixi run` commands in this
> README are for using this repository on its own.

```bash
ddsuite vs                       # the assistant: asks its questions, then submits the job
ddsuite vs run.yaml              # the same steps with all answers in a run file
ddsuite vs run.yaml --check      # check a run file and show what would be done, run nothing
ddsuite vs --example             # print an annotated example run file
ddsuite vs config.txt            # classic mode: the original VF Unity workflow (see below)
```

### What the assistant does

It docks a library of molecules into a protein with a Vina-family program and ranks them by score. It asks for:

1. **The protein and the binding site.** Either a protein-ligand complex (PDB or mmCIF, e.g. from rcsb.org), which it
   cleans up for you and centres the box on the ligand in it, or a protein PDB file plus a ligand in the site (SDF) or
   the centre coordinates. If the protein has no hydrogens it adds them with PDB2PQR and PROPKA at a pH you choose
   (or ADFR's quicker rule-based guess).
2. **The docking settings.** The program, and whether to re-score with NNScore2.
3. **The molecules.** An SDF file, or a text file with one SMILES per line (`SMILES name`, a header line is fine).
   Any number works. Molecules that cannot be docked are listed with the reason, and you can leave them out.
4. **How to run.** Submit to the cluster (SLURM, partition `ncpu`), run in the terminal, or only save the job script.
   It suggests the number of CPUs and a time limit, and checks with the scheduler that your account may use the
   partition.
5. **A review** of everything, before anything is submitted.

Everything you answer is saved as `run.yaml`, so the run can be repeated, or changed and run again, with
`ddsuite vs run.yaml`.

### Programs

All six were run end to end on a real receptor and ligand (the BACE1 case, with a ligand prepared by RDKit and Meeko
and a receptor prepared by ADFR).

| Program | Notes |
|---|---|
| `qvina` (QuickVina 2.1) | Fast Vina variant; the default |
| `AutodockVina_1.2` (Vina 1.2.3) | Current Vina; the only one that also accepts silicon |
| `vina` (Vina 1.1.2) | The original Vina |
| `qvina-w` (QuickVina-W 1.1) | For large boxes |
| `smina` | Vina fork |
| `PSOVina` (2.0) | Particle-swarm search; about 7 times faster, but it found the crystal pose less often in the redocking test and it ignores the random seed, so repeating a run gives different scores (differences of 0.3 kcal/mol typically, up to 0.9) |

Boron and selenium make every program reject a molecule; silicon is accepted only by Vina 1.2. Metals and metalloids other than Mg, Ca, Mn, Fe and Zn (so most organometallics and metal complexes) have no AutoDock atom type and cannot be docked. The assistant flags such molecules before the job starts, naming the element. NNScore2 re-scoring is the one re-scoring method offered.
Other programs and scoring functions listed in `config.txt` are available through classic mode, but have not been tested
here and several need software or licences that are not part of this repository.

### What is prepared, and how

| Step | Done with |
|---|---|
| Complex clean-up (chains, waters, additives, ions, cofactors) | `structure_prep.py` (gemmi) |
| Protonation | PDB2PQR with PROPKA, or ADFR (`prepare_receptor -A checkhydrogens`) |
| Receptor to PDBQT | ADFRsuite `prepare_receptor` |
| Each molecule | Largest fragment kept, charge state set for a pH (or kept as written), 3D coordinates from RDKit (ETKDG), PDBQT from Meeko |
| Poses back to molecules | Meeko, so every pose has all its hydrogens |

**Charge states.** By default each molecule is set to its charge state at pH 7.4 (a pH you can change), whatever
the file shows: carboxylic, sulfonic and phosphoric acids and tetrazoles become anions, aliphatic amines, amidines and
guanidines cations. This uses Open Babel's pH model plus two extra rules (sulfonic acids and acylsulfonamides, which it
misses). A check that the molecule is otherwise unchanged (same atoms, bonds and stereochemistry) falls back to the
input if it fails. Choose `ionisation: keep` to use the charges exactly as written, for a library that was already
prepared at a given pH, for example from ZINC. Rule-based assignment is approximate: it was right for 48 of 49 test
molecules (partly tuned to those) and 17 of 20 real drugs not used to design it. It gives every aliphatic amine a pKa
of 10, so weak bases such as N-arylpiperazines (sildenafil) are charged when they are mostly neutral, and it misses
acids such as 4-hydroxycoumarins (warfarin) and 4-aminopyridine-type bases. Only the single most likely state is
docked; molecules whose pKa is close to the pH are not docked as both forms. It also over-charges some molecules with
two nearby amines: indinavir is given +2, though its published pKa values (about 3.7 and 6.2) make it mostly neutral at pH 7.4.

Effect on pose accuracy, measured by redocking 10 ligands with QuickVina (exhaustiveness 16, one seed, neutral SMILES as
the input): the top pose was within 2 Å of the crystal pose for 7 of 10 with charge states set and 7 of 10 without.
The setting changed the molecule in 6 of the 10. The top-pose error got smaller in 3 (indinavir 1.3 to 0.8 Å, biotin 1.3
to 0.8 Å, benzamidine 0.40 to 0.35 Å), larger in 2 (huperzine 2.7 to 4.3 Å, methotrexate 5.7 to 6.0 Å, both failures
either way) and stayed the same in 1. Ten complexes cannot show a difference this small. Vina-type scoring has no
electrostatic term, so charges matter mainly through how hydrogen-bond donors and acceptors are assigned; the main
reason for the default is that the molecule drawn is the one present in water, not that poses were found to improve.

Tautomers and stereoisomers are used exactly as in your file; they are not enumerated. A centre without a defined
configuration is docked in one arbitrary form.

### The run folder

Everything for one run is kept in one folder, and your input files are never changed:

| In the run folder | What it is |
|---|---|
| `input/` | Copies of the receptor and the molecules the job used |
| `run.yaml` | The answers, to repeat the run |
| `submit_<job name>.sh`, `<job name>_<id>.log` | The job script and its log |
| `protonation_report.txt` | What PDB2PQR did (when it was used) |
| `results/ranking.csv` | All molecules, best score first (score in kcal/mol, more negative is better) |
| `results/results.csv` | The same in input order, with the SMILES that was docked and any note or failure reason |
| `results/top_poses.sdf`, `results/all_poses.sdf` | Best pose of each molecule, and every pose (all hydrogens, with score properties) |
| `results/ligands/<group>/<number>/poses.sdf` | One folder per molecule |

A molecule that fails (cannot be read, has no 3D geometry, contains an unsupported element) is reported with its reason
and the others are still docked; the job then exits with an error so that it shows as failed in the queue.
Docking scores estimate binding strength; use them to rank and to choose molecules to inspect, and look at the poses.

### Resuming

The job writes `results.csv` after every molecule. If a job is stopped or runs out of time, run `run.yaml` again and
choose to resume: finished molecules are skipped and failed ones are retried. Resuming is refused if the receptor,
the molecules, the box or any setting that changes scores is different. Results are only ever deleted if this tool
made the folder.

### How far to trust the results

Redocking test: 13 crystal structures from the PDB (HIV protease, EGFR, trypsin, streptavidin, estrogen receptor, Abl,
Hsp90, DHFR, CDK2, acetylcholinesterase), each docked back into its own protein through this tool (clean-up,
PDB2PQR or ADFR hydrogens, Meeko, the programs below), with the ligand taken from its SMILES. A pose counts as correct
if its heavy-atom RMSD to the crystal pose is 2 Angstrom or less (symmetry-aware, no re-alignment).

| Program (exhaustiveness) | Top-ranked pose correct | Any of 9 poses correct |
|---|---|---|
| QuickVina (8) | 7 of 13 | 10 of 13 |
| QuickVina (32) | 9 of 13 | 10 of 13 |
| Vina 1.2 (8) | 9 of 13 | 10 of 13 |
| smina (8) | 9 of 13 | 10 of 13 |
| PSOVina (8) | 5 of 13 | 5 of 13 |

- This is why the default exhaustiveness is 16 and not Vina's usual 8. Exhaustiveness 16 was run on five of the
  complexes only, including both that QuickVina missed at 8 (1HSG and 1UYG): it found all five, as 32 did, where 8 found
  three. It takes twice as long as 8.
- Three complexes fail with every program (1HVR, a 46-atom flexible ligand; 4DFR, charged methotrexate; 1GPK,
  acetylcholinesterase). Scoring the crystal pose itself showed 1M17 and 4DFR to be scoring failures (the crystal pose
  scores worse than the wrong pose that is ranked first) and 1HVR a search failure (the crystal pose scores 2.2 kcal/mol
  better than anything found).
- The test measures pose prediction, not whether the scores rank molecules correctly. That needs a set of known
  actives and decoys for the target, which was not run here.
- Scores favour large, flat, aromatic molecules. On 4,999 molecules of the NCI set docked into BACE1, score correlated
  with the number of heavy atoms (r = -0.63), the best scores were polycyclic aromatic hydrocarbons, and the top 100
  had a median of 29 heavy atoms against 15 for the whole set. Look at score together with size.
- QuickVina, Vina 1.2 and smina give identical scores when a run is repeated with the same seed. PSOVina does not
  (scores differed by up to 0.9 kcal/mol, typically 0.3).
- 3 of the 13 structures could not be protonated by PDB2PQR because of missing backbone atoms; the tool says so and
  suggests ADFR's hydrogens, which were used for those three. The choice affects scores: the BACE1 test ligand scored
  -9.6 with PDB2PQR hydrogens and -8.8 with ADFR's.

### Size of a run

One CPU docks one molecule at a time; the job docks as many at once as it has CPUs (the assistant suggests up to 32,
the partition has up to 256 per node). The time limit is estimated from times at one CPU per molecule (QuickVina 38 s,
smina 59 s, PSOVina 6 s at exhaustiveness 8; the default of 16 takes twice as long) and doubled. Only the QuickVina
figure was measured on a drug-like library; the others are scaled from smaller test molecules. A job is limited to 7
days; for larger libraries use more CPUs or resume in a second job.

Measured, 100,000 molecules: a random sample of the Enamine screening collection (99,982 molecules after leaving out
18 that contain boron or silicon), docked with QuickVina at exhaustiveness 16 on one 256-CPU node, took 8 h 25 min:
8 h 16 min of docking, about 12,000 molecules per hour, and 9 min to write the final files. All 99,982 were docked, none
failed. Per molecule, docking took 75 s on average (median 71 s, 5th to 95th percentile 26 to 142 s, longest 413 s), and
94% of the allocated CPU time went into docking. Memory peaked at 40.7 GB (159 MB per CPU). The output was 7.5 GB (75 KB
per molecule: 3.3 GB in `all_poses.sdf`, 3.9 GB in the per-molecule folders, 0.4 GB in `top_poses.sdf`). The receptor
was the small Bace1 test pocket, so this shows speed and reliability, not that the scores rank real actives well.
Resuming a copy of this run with the last 200 molecules removed from the table docked only those 200, and gave a table
with the same scores for every molecule and a byte-identical `all_poses.sdf`; it took 19 min on 32 CPUs, 9 of them for
the final files. Of 2,000 molecules docked twice with the same seed, 1,996 gave identical scores and 4 differed by at
most 0.3 kcal/mol.

Earlier, smaller runs: 4,961 molecules of the NCI set (PSOVina, exhaustiveness 8, 64 CPUs) took 16 minutes, and 12
minutes after molecules that cannot be docked started failing at once. The first version rewrote the whole table after
every molecule, which took 144 s of bookkeeping at 10,000 molecules. The job now reads the library one molecule at a
time, keeps only a few in flight, and appends one line per finished molecule to `results/results.csv`, so a stopped job
loses nothing; stopping one with SIGTERM (what SLURM sends at the time limit) and resuming gave a complete,
duplicate-free table. Each molecule gets its own folder, grouped by thousands.

### Staged screening with AdaptiveFlow (your own library)

The design and what was measured while building it are in `docs/DESIGN_staged_screening.md`.

The first question of the assistant is how the molecules should be docked. **Dock every molecule** is everything
above. **Staged screen** hands the work to AdaptiveFlow: AFLP prepares every molecule (stereoisomers, tautomers,
protonation at a pH, 3D coordinates) and AFVS docks the prepared forms, both as SLURM job arrays, and the best form of
each molecule is ranked. The upstream code is in `third_party/` (GPL v2, pinned commits in `third_party/VERSIONS.md`)
and is not changed; the tool writes the configuration of each run into the run folder.

It keeps scores, not poses (AFVS stores no poses); `vs redock` (below) gets the poses of the best molecules.
Your own SMILES or SDF library is supported, with two strategies and an optional neural-network filter (below). The
ultra-large Enamine REAL Space library is not built yet.

Start it from the assistant, or with a run file (`vs --example` shows the keys):

```yaml
receptor: protein.pdb
reference_ligand: ligand.sdf
ligands: library.smi
output: run_staged
screen: staged
docking: {program: qvina, exhaustiveness: 16}
staged: {max_stereoisomers: 1, max_tautomers: 1}   # forms prepared per molecule; every form is docked
run: {mode: submit, cpus: 500}
```

**Two strategies.** `staged.strategy` is `all` (the default) or `tranche`.

- `all` docks every prepared form. Complete, and the cost grows with the library.
- `tranche` is the ATG strategy of AdaptiveFlow. The molecules are grouped into tranches by their properties
  (molecular weight, logP, hydrogen-bond acceptors and donors by default; `tranche_properties` can name any of the 18
  of the tranche ID). A few forms of every tranche (`representatives`, 1 to 10, default 3) are docked first, the
  prescreen. Each tranche is scored by the mean score of its prescreened forms, and the best tranches are docked in
  full until about `primary_size` molecules are reached (default 20% of the library). Whole collections are taken, so
  the primary screen can be a little larger than asked for. A tranche is a group of similar molecules, so a good
  mean score makes it likely that the tranche holds good molecules; it is a bet, not a guarantee.

```yaml
staged:
  strategy: tranche
  representatives: 3          # forms docked per tranche in the prescreen
  primary_size: 100000        # molecules docked in the primary screen
  tranche_properties: [mw, logp, hba, hbd]
```

The tranches are made by this tool with RDKit (AFLP's own assignment needs ChemAxon and only renames files). The 18
properties are mw, logp, hba, hbd, tpsa, atomcount, bondcount, mr, ringcount, aromaticringcount, formalcharge,
positivechargecount, negativechargecount, fsp3, chiralcentercount, halogencount, sulfurcount and NOcount. A property
that is not chosen counts as one class for all molecules, so the library falls into few tranches that each hold many
molecules; the real REAL Space library has about 5,600 per tranche. A property has up to six classes (A to F). The class limits are the default cutoffs of AFLP; where AFLP has more than six classes (ringcount, NOcount) the upper ones are merged or the steps are made coarser. The assistant samples the library (up to 3,000 molecules) and shows the number of tranches and the size of
the prescreen before anything is submitted; for a library bigger than the sample the number of tranches is an estimate.

The stages follow each other by themselves: each ends by queueing a small `advance` job that waits for the stage's jobs,
checks their output and submits the next stage. The run folder keeps its state in `state.json`.

```
vs status  <run folder>            where the run is
vs results <run folder> [--top N]  the best molecules
vs advance <run folder> [--retry]  continue by hand; --retry after a stage stopped and the cause is fixed
vs redock  <run folder> [--top N]  dock the best molecules again and keep their poses (see below)
```

You get an email if a stage fails and when the run ends. The tables are `results/ranking.csv` (best first) and
`results/results.csv` (input order), with the number of forms docked and the best form of each molecule. A molecule
that AFLP cannot prepare is listed with status `failed`. With the tranche strategy a molecule that was prepared but whose
collection was not chosen for the primary screen has status `not_docked`, the tables have a `tranche` column, and
`results/tranche_scores.csv` lists every tranche with its mean and best prescreen score and whether it was selected.

**Poses for the best molecules.** When a run has finished, `vs redock <run folder> --top 1000` writes the best 1,000
molecules (the best form of each, as AFLP prepared it, with its charge state kept) to `redock_input/` and starts an ordinary
full-mode run on them in `<run folder>/redock/`, with the protein, box, program, exhaustiveness and seed of the staged
run. `--exhaustiveness`, `--num-modes` and `--cpus` change those; `--save` only prepares the job. The poses are in
`redock/results/` (`top_poses.sdf`, `all_poses.sdf`, per-molecule folders). When it has finished,
`vs redock <run folder> --report` writes `results/redock_comparison.csv` (staged score, new score, difference) and prints
a summary. Measured on the top 20 molecules of the 3,000-molecule run below: all 20 docked with 9 poses, the new
score was 0.17 kcal/mol worse on average, 17 of 20 within 0.5 (correlation 0.65 over these 20), and it ran for about 5
minutes on 20 CPUs. The two scores differ because the full mode prepares its own 3D conformers. A staged run started
before this feature does not keep its protein file; the command then uses the file named in the run's plan, if it still exists.

**CPUs.** The assistant suggests 500 CPUs working at the same time (fewer for a small library) and allows as many as
the cluster lets one user have, read from the scheduler (3,000 on `ncpu`). The work is cut into jobs of 32 CPUs of about
25 minutes, so the number is used in steps of 32. Above 1,000 CPUs or 100,000 CPU-hours it asks for confirmation.

**Cost.** Every prepared form is a docking. One stereoisomer and one tautomer (the default) is one form per molecule.
AFLP's own defaults (5 and 5) gave 4.57 forms per molecule on 200 Enamine molecules and make the docking 4.6 times
more expensive. The review page shows the estimate (CPU-hours, wall time, jobs, disk) before anything is submitted.

**Measured** (200 Enamine molecules, QuickVina, exhaustiveness 16, the test pocket, one form each, 32 CPUs per job):
199 were prepared and docked, and the ranked table was ready 9 minutes after the start (about 1 minute of AFLP, the rest
docking and ranking). Their scores against the full mode: mean difference 0.00 kcal/mol, 177 of 199 within 0.5,
identical for 34 (the 3D conformers and charge states are made differently). In a manual run of the same molecules
with AFLP's defaults (4.57 forms each), the best form scored 0.18 kcal/mol better on average. The one molecule that
failed (`CCOC(C)=C(C#N)C(=O)O`) got a nonsensical charge state from Open Babel at pH 7.4; AFLP reported its form as
failed but the molecule as a success, and the tool lists it as failed. AFVS keeps a log of every docking, about 4.5 KB
each (4.5 GB per million forms).

**Neural-network filter** (`staged.ml_filter: true`, with the tranche strategy). After the tranches are chosen, AdaptiveFlow's
filter looks at every form of the selected tranches and drops those that are unlikely to dock well, so the primary screen
docks fewer forms. The network is trained on the prescreen: Morgan fingerprints of the forms (the SMILES AFLP wrote into
each form), and the best 25% of the prescreen scores as the positive class. `primary_size` is then the pool the filter looks
at, so choose it 2 to 3 times larger than without the filter; `ml_cutoff` (default 0.5, from 0.05 to 0.95) is the predicted
probability a form needs to be docked. The recipe is the one of AdaptiveFlow's `ml_classifier.py` (network of
1024-512-256-128-1, Adam, batches of 256, early stopping), written with numpy: PyTorch would add 1.3 GB to the
environment (measured), and the models are not interchangeable with AdaptiveFlow's. 15% of the prescreen is held back to
measure the model. **The filter is not used, and the whole pool is docked, when the prescreen has fewer than 200 scored
forms or the model's area under the ROC curve on the forms held back is below 0.6**; the run says so, in `status` and in
`results/ml_filter.json`, which also gives the area under the curve and the share of positives the cutoff recalls. A form
whose SMILES cannot be read is kept. The probability of every form is in `results/ml_filter.csv`. The filter runs inside
the `advance` job (4 CPUs, 6 GB more memory), in up to 4 processes. Training uses at most 200,000 prescreen forms
(100,000 forms took 94 s). Applying the model ran at about 14,000 forms per second per process on short made-up
SMILES while the machine was busy with other jobs; real drug-like molecules are slower to fingerprint. A pool of a few million
forms takes minutes, but one of hundreds of millions would take days in this single job, so the filter is meant for
pools of up to a few tens of millions of forms until it is spread over an array.

Measured on the 3,000-molecule run (the filter was trained on its 431 prescreen forms; AUC 0.71 on the 64 held back, so a
weak model from little data, one training per setting). The forms docked are the prescreen plus what the primary screen
docks; the control docked all 2,993 forms:

| Strategy | Forms docked | Best 30 found | Best 100 | Best 300 |
|---|---|---|---|---|
| tranches, pool 608 | 910 (30%) | 22 | 60 | 172 |
| tranches, pool 930 | 1,181 (39%) | 23 | 71 | 212 |
| tranches, pool 1,308 | 1,502 (50%) | 26 | 81 | 239 |
| tranches, pool 1,935 | 2,076 (69%) | 30 | 99 | 293 |
| pool 930 + filter at 0.5 | 852 (28%) | 21 | 61 | 180 |
| pool 1,308 + filter at 0.5 | 1,000 (33%) | 24 | 71 | 203 |
| pool 1,935 + filter at 0.5 | 1,260 (42%) | 28 | 86 | 244 |
| pool 1,935 + filter at 0.7 | 620 (21%) | 22 | 58 | 131 |
| pool 1,935 + filter at 0.3 | 1,887 (63%) | 30 | 98 | 289 |

Between the rows without the filter, 1,260 forms would find about 24, 74 and 221, so the best filtered row finds 4, 12 and
23 more of the best 30, 100 and 300 for the same docking. The filter at 0.7 drops too much for this weak model and does
worse than tranches alone at the same cost, and at 0.3 it saves little. The gain comes from the same prescreen data the
tranche selection already uses, and with so little training data the model is weak; a prescreen of millions of forms should
train a better one, which has not been measured here.

A full run through SLURM with the filter on (the same 3,000 molecules, but exhaustiveness 2 to keep it short, pool of 1,900
molecules): AFLP made 2,993 forms, the prescreen docked 431, 113 of 175 tranches were selected (1,926 forms), the model
trained on 174 prescreen forms had an area under the curve of 0.80 on 65 held back and kept 1,458 of the 1,926 (76%), and
AFVS docked exactly those 1,458 forms (checked against the form lists: none missing, none extra). With scores this noisy (exhaustiveness
2) the model separates less sharply, so do not read the 76% as typical.

**Measured, tranche strategy** (3,000 Enamine molecules, QuickVina, exhaustiveness 16, one form each; the control
docked all 2,993 prepared forms). 175 tranches (properties mw, logp, hba, hbd), 431 forms in the prescreen (3 per
tranche), 52 tranches and 608 forms in the primary screen: 910 forms docked, 30% of the library, 7 molecules not
prepared by AFLP. The run took 56 minutes with a free queue, 32 CPUs per stage (preparation 8 minutes, prescreen 18, primary 30
minutes; a library this small fills one job per stage whatever CPU number is asked for), 19 CPU-hours of
docking. It found 22 of the true best 30, 60 of the best 100 and 172 of the best 300 (a random 30% would find 9, 30 and
91). The median score of the primary screen was -8.2 against -7.5 for the library. How much the strategy finds depends on
which forms are drawn as representatives: replaying the selection on the control scores with 500 different draws gave
25 of the best 30 on average (22 to 28 for 90% of the draws), 73 of the best 100 (63 to 81) and 187 of the best 300
(169 to 203). This run's draw was on the low side (a replay of the same representatives gives 22, 61 and 179). The
scores of a form differ a little between preparations of the same molecule (mean difference 0.13 kcal/mol, identical
for 487 of 910 forms) because the 3D conformers are not the same each time. A library of 3,000 molecules in 175
tranches is not the REAL Space library (about 5,600 molecules per tranche); how well the strategy works there has not
been measured.

**Limits of this first version.**
- The ranking keeps every molecule in memory: 1.1 GB per million molecules, 14 s per million (measured on synthetic
  tables); the job that does it asks for that much. Up to a few tens of millions is fine; beyond that it needs a
  different approach.
- A staged run can take days. Its jobs use the copy of this tool that started them, and ddsuite deletes a replaced
  version after 14 days (`keep_old_versions_days`). Do not let a run outlive an update by that long; ask the
  maintainer to keep the old version if it must.
- Only your own library is supported. The Enamine REAL Space library (needs registration and AWS credentials) is not
  built.
- The tranche library is built in one job that keeps the compressed forms in memory (about 3 GB per million forms) and
  writes them again as tranche collections, so the library is stored twice (AFLP's output and the tranche
  collections). It is meant for libraries of up to a few million molecules; the real REAL Space library will need the
  tranche layout it already has.
- The prescreen forms of a selected tranche are docked again in the primary screen (same scores, a little wasted work).
- A stopped stage is not repaired automatically: `advance --retry` submits the failed work unit again as a whole.

Also tested: the stage logic with a stand-in scheduler (including a failed job, a retry and two `advance` at once) in
`test/test_staged.py`, and the real thing on the cluster in `test/test_staged_cluster.py` (`pixi run test-cluster`,
three runs (each strategy, and the filter on a library too small for it), about 1.5 minutes each with a free queue, longer when the queue is busy).

From the tool folder: `pixi run test` runs 638 fast tests (about 20 seconds: ligand preparation, charge states, results
table, run files, SLURM scripts, wizard, command line, the staged screen with a stand-in scheduler). `pixi run test-all` adds 36 tests that dock with the real
programs on a small receptor (about 7 more minutes): every offered program, resume after a stop, a killed worker
process, and the output files. `VS_CLUSTER_TEST_DIR=<shared folder> pixi run test-cluster` submits a real staged run to SLURM. The tests do not judge whether docking scores are scientifically good; they check that
the tool does what it says.

## Prerequisites (standalone use)
The suite installs all of this. On its own, the tool needs [Pixi](https://pixi.sh/) (0.83.0 or newer):
```
git clone git@github.com:Chemical-Biology-STP/AdaptiveFlow-Unity.git
cd AdaptiveFlow-Unity
pixi install --all
pixi run python vs_cli.py
```
`--all` also builds the separate `adfr` environment with ADFRsuite (its conda package bundles Python 2.7, so it cannot
share the main environment). The main environment holds Python 3.12, RDKit, Meeko, Open Babel, PDB2PQR and PROPKA
(see `pixi.toml`).

## File Navigator
* `vs_cli.py`: Entry point of `ddsuite vs`: the assistant, a run file, or classic mode. 
* `vs_wizard.py`: The assistant's steps (protein, binding site, settings, molecules, how to run, review). 
* `vs_dock.py`: The docking engine used by the assistant and by the job script (receptor and ligand preparation, docking, re-scoring, results). 
* `vs_slurm.py`, `vs_runfile.py`, `vs_ui.py`: The SLURM job script, the YAML run file, and terminal helpers. 
* `vs_staged.py`: Staged screening with AdaptiveFlow: configuration of each run, the stages, the `status`/`advance`/`results` commands.
* `third_party/`: Unmodified AdaptiveFlow code (AFVS and AFLP, GPL v2) used by the staged screen; see `third_party/VERSIONS.md`.
* `structure_prep.py`, `protein_prep.py`: Complex clean-up and protonation (from the anchored_docking_workflow repository, see `NOTICE.md`). 
* `vs_classic.py`: Classic mode: builds a run folder and runs `run_vf_unity.py` inside it. 
* `run_vf_unity.py`: Main file that initiates docking calculations. 
* `initiate_calc.py`: File that initiates scoring and docking calculations. 
* `lig_process.py`: Process provided ligand into 3D format compatible with docking program (used by )
* `pose_prediction.py`: File for running pose prediction on processed ligands (used by pose_prediction). 
* `scoring_functions.py`: File for running scoring on already docked ligands (used by initiate_calc.py). 
* `config.txt`: Config file consisting of user-definable parameters for running the calculation. 
* `/ligands/`: Directory created by VF Unity, which will contain all processed ligands in a ready-to-dock format. 
* `/outputs/`: Directory created by VF Unity, which will contain docked ligand files. 



## Classic mode (the original VF Unity workflow, from a config.txt file)
`ddsuite vs config.txt` runs the original workflow below in a new run folder (`--example-config` prints a config file, `-o` chooses the folder). It supports many more programs than the assistant, but only `qvina` and `qvina+nnscore2` were tested here. It docks one molecule given as a SMILES string, and enumerates stereoisomers and tautomers.

## Quick Start (Using a config.txt file)
We will be running QuickVina on a processed protein located in the config directory (`5wiu_test.pdbqt`). We provide a config.txt file which contains all parmeters for this simple run. 
Please edit this file based on your preference: 
```
# The choice of the docking method
# Possible choices:
# AutoDock-Koto, AutodockVina_1.2, AutodockZN, EquiBind, FRED
# FitDock, GalaxyDock3, LigandFit, LightDock, M-Dock
# MCDock, MM-GBSA, PLANTS, PSOVina, RLDock
# SEED, adfr, autodock_cpu, autodock_gpu, AutodockVina_1.1.2
# dock6, flexx, gnina, gnina-scoring
# gold, gwovina, iGemDock, idock, ledock
# molegro, nnscore2, qvina, qvina-w, rDock
# rf-score, rosetta-ligand, smina, smina-scoring, vina
# vina_carb, vina_xb, GlideSP, GlideXP, GlideHTVS
# qvina_gpu, qvina_w_gpu, vina_gpu, vina_gpu_2.0
# The choice of the scoring function
# Possible choices:
# nnscore2, rf-score, smina-scoring, ad4_scoring, vinandro_scoring
# vina_scoring, gnina_scoring, chemplp_scoring, PLP_scoring, PLP95_scoring
# contact_scoring, continuous_scoring, grid_scoring, 
# asp, chemscore, goldscore, plp, mm_gbsa_scoring, Hawkins_gbsa
# Please note: different pose prediction/docking methods can be combined with scoring functions.
# For example: ’qvina+nnscore2’.
# For supported choices/combinations, please see the manuscript or the homepage.

program_choice=qvina+nnscore2

# The x,y&z coordinates of the center of the docking space. The binding space describes the location where a molecule
# is allowed to bind.
center_x=-17.820
center_y=16.140
center_z=-18.643

# The size (in Angstroms) of the docking space in the x,y&z directions.
size_x=20
size_y=20
size_z=20

# How many poses to search for, providing a limit to the maximum number of iterations that a docking program performs in
# search for good poses
# Large exhaustiveness settings lead to increased computational costs.
exhaustiveness=10 


# Molecule (either a string in smiles, selfies or amino-acid sequence) to be used for docking
# If the is_selfies=True, or is_peptide=True, a conversion from selfies->smiles
# and aa-sequence->smiles is performed. 
smi=C1=CC(=CC=C1CSCC2C(C(C(O2)N3C=NC4=C(N=CN=C43)N)O)O)Cl 
is_selfies=False
is_peptide=False


# Location to the prepared receptor file
#  The receptor needs to be in the correct format, supported by the user's selected docking program.
#  Additionally, the file needs to be present in the config directory.
receptor=./config/5wiu_test.pdbqt
```
To execute the program, please run: 
```
python3 run_vf_unity.py
```
We note: 
1. The processed ligands will be located within the newly created ligands directory.
2. The docked output from running QuickVina will be located in the newly created outputs directory.
3. The default behaviour of the program (QuickVina) is to make use of all available CPUs. 
4. A summary csv file `docking_output.csv` is created (for running QuickVina): 
    ```
    Ligand File,Docking Values,Docking Pose
    7.pdbqt,"-9.9,-9.8,-9.1,-9.0,-9.0,-8.8,-8.8,-8.7,-8.6",./outputs/pose_7.pdbqt
    14.pdbqt,"-9.6,-9.1,-9.0,-8.8,-8.7,-8.6,-8.6,-8.4,-8.4",./outputs/pose_14.pdbqt
    6.pdbqt,"-9.8,-9.7,-9.2,-8.9,-8.8,-8.8,-8.7,-8.7,-8.5",./outputs/pose_6.pdbqt
    9.pdbqt,"-9.3,-9.3,-9.3,-9.1,-9.1,-9.0,-9.0,-8.9,-8.8",./outputs/pose_9.pdbqt
    12.pdbqt,"-8.9,-8.9,-8.8,-8.5,-8.5,-8.5,-8.5,-8.1,-8.0",./outputs/pose_12.pdbqt
    4.pdbqt,"-9.6,-9.3,-9.2,-9.2,-9.0,-9.0,-8.8,-8.7,-8.7",./outputs/pose_4.pdbqt
    3.pdbqt,"-9.9,-9.2,-9.0,-8.9,-8.8,-8.7,-8.6,-8.5,-8.4",./outputs/pose_3.pdbqt
    1.pdbqt,"-9.4,-9.4,-9.1,-9.0,-8.9,-8.8,-8.7,-8.7,-8.7",./outputs/pose_1.pdbqt
    5.pdbqt,"-9.8,-9.5,-9.4,-9.3,-9.3,-9.2,-9.1,-9.1,-9.0",./outputs/pose_5.pdbqt
    10.pdbqt,"-9.6,-9.5,-9.4,-9.3,-9.2,-9.2,-9.1,-8.9,-8.8",./outputs/pose_10.pdbqt
    2.pdbqt,"-9.6,-9.3,-8.9,-8.9,-8.8,-8.8,-8.8,-8.6,-8.5",./outputs/pose_2.pdbqt
    15.pdbqt,"-9.8,-9.3,-9.2,-9.2,-9.1,-9.0,-9.0,-8.8,-8.8",./outputs/pose_15.pdbqt
    13.pdbqt,"-9.1,-9.1,-9.0,-8.8,-8.7,-8.7,-8.7,-8.7,-8.7",./outputs/pose_13.pdbqt
    8.pdbqt,"-9.1,-9.1,-9.1,-8.8,-8.7,-8.7,-8.6,-8.6,-8.5",./outputs/pose_8.pdbqt
    11.pdbqt,"-9.7,-9.5,-9.3,-9.2,-9.2,-9.1,-9.0,-8.9,-8.8",./outputs/pose_11.pdbqt
    0.pdbqt,"-9.6,-9.3,-9.3,-9.0,-8.9,-8.9,-8.7,-8.7,-8.5",./outputs/pose_0.pdbqt
    ```
5. Sepperately, a summary csv file is created (`rescoring_output.csv`) is created when running rescoring (nnscore2): 
    ```
    Docked Ligand,Re-scored Value
    ./outputs/pose_11.pdbqt,Kd = 114.94 fM;Kd = 2.29 pM;Kd = 3.22 fM;Kd = 10.3 fM;Kd = 11.92 fM;Kd = 0.87 fM;Kd = 307.09 fM;Kd = 2.18 pM;Kd = 22.69 pM
    ./outputs/pose_10.pdbqt,Kd = 14.22 fM;Kd = 13.13 fM;Kd = 10.96 pM;Kd = 29.48 fM;Kd = 73.28 fM;Kd = 2.35 fM;Kd = 108.96 fM;Kd = 122.82 fM;Kd = 0.35 fM
    ./outputs/pose_9.pdbqt,Kd = 216.4 fM;Kd = 685.98 fM;Kd = 155.95 fM;Kd = 116.24 fM;Kd = 72.66 fM;Kd = 189.04 pM;Kd = 4.74 pM;Kd = 7.01 pM;Kd = 1.04 nM
    ./outputs/pose_5.pdbqt,Kd = 0.14 fM;Kd = 3.58 pM;Kd = 2.75 fM;Kd = 23.24 fM;Kd = 7.79 pM;Kd = 152.06 nM;Kd = 0.12 fM;Kd = 121.77 fM;Kd = 18.27 pM
    ./outputs/pose_12.pdbqt,Kd = 0.16 fM;Kd = 11.43 fM;Kd = 1.76 pM;Kd = 4.74 pM;Kd = 2.15 pM;Kd = 4.79 pM;Kd = 364.24 fM;Kd = 55.44 fM;Kd = 20.77 fM
    ./outputs/pose_7.pdbqt,Kd = 101.4 fM;Kd = 0.12 fM;Kd = 1.14 fM;Kd = 134.77 fM;Kd = 0.55 fM;Kd = 10.53 nM;Kd = 0.0 fM;Kd = 129.84 fM;Kd = 8.86 nM
    ./outputs/pose_13.pdbqt,Kd = 0.34 fM;Kd = 0.13 fM;Kd = 1.74 pM;Kd = 73.35 fM;Kd = 203.96 fM;Kd = 11.28 pM;Kd = 3.75 pM;Kd = 85.11 fM;Kd = 5.17 pM
    ./outputs/pose_0.pdbqt,Kd = 19.81 fM;Kd = 0.01 fM;Kd = 0.06 fM;Kd = 14.96 fM;Kd = 0.1 fM;Kd = 578.15 fM;Kd = 120.14 fM;Kd = 24.31 fM;Kd = 3.01 pM
    ./outputs/pose_4.pdbqt,Kd = 12.63 fM;Kd = 2.81 pM;Kd = 0.0 fM;Kd = 52.42 fM;Kd = 0.96 fM;Kd = 310.52 fM;Kd = 36.57 pM;Kd = 17.94 fM;Kd = 0.13 fM
    ./outputs/pose_6.pdbqt,Kd = 0.91 fM;Kd = 8.34 pM;Kd = 0.14 fM;Kd = 0.19 fM;Kd = 0.02 fM;Kd = 0.0 fM;Kd = 16.4 fM;Kd = 60.84 fM;Kd = 1.34 pM
    ./outputs/pose_8.pdbqt,Kd = 94.53 fM;Kd = 23.83 pM;Kd = 6.22 pM;Kd = 1.34 fM;Kd = 276.47 fM;Kd = 0.63 fM;Kd = 71.78 fM;Kd = 5.64 fM;Kd = 18.55 pM
    ./outputs/pose_1.pdbqt,Kd = 0.13 fM;Kd = 0.09 fM;Kd = 11.47 pM;Kd = 11.69 fM;Kd = 5.36 pM;Kd = 8.35 pM;Kd = 2.44 pM;Kd = 4.19 pM;Kd = 23.34 pM
    ./outputs/pose_3.pdbqt,Kd = 0.0 fM;Kd = 1.65 fM;Kd = 14.99 fM;Kd = 0.24 fM;Kd = 213.73 pM;Kd = 6.34 pM;Kd = 472.1 fM;Kd = 329.75 pM;Kd = 261.08 fM
    ./outputs/pose_15.pdbqt,Kd = 1.06 pM;Kd = 136.55 fM;Kd = 1.82 pM;Kd = 106.86 fM;Kd = 4.26 pM;Kd = 1.55 pM;Kd = 6.23 pM;Kd = 61.89 fM;Kd = 242.37 pM
    ./outputs/pose_2.pdbqt,Kd = 11.5 fM;Kd = 18.0 fM;Kd = 2.34 pM;Kd = 783.94 fM;Kd = 202.75 fM;Kd = 79.9 fM;Kd = 210.65 fM;Kd = 13.13 fM;Kd = 511.8 fM
    ./outputs/pose_14.pdbqt,Kd = 162.12 fM;Kd = 726.22 fM;Kd = 10.38 fM;Kd = 432.15 fM;Kd = 0.06 fM;Kd = 9.24 pM;Kd = 3.07 pM;Kd = 23.07 nM;Kd = 7.97 pM
    ```
    
## Quick Start (Using a Python function call)
```
from run_vf_unity import main 

program_choice   = 'qvina'
scoring_function = 'nnscore2' 
center_x         = -17.820
center_y         = 16.140
center_z         = -18.643
size_x           = 20
size_y           = 20 
size_z           = 20
exhaustiveness   = 10
smi              = 'C1=CC(=CC=C1CSCC2C(C(C(O2)N3C=NC4=C(N=CN=C43)N)O)O)Cl'
is_selfies       = False
is_peptide       = False
receptor         = './config/5wiu_test.pdbqt'

pose_pred_out, re_scored_values = main(program_choice, scoring_function, center_x, center_y, center_z, size_x, size_y, size_z, exhaustiveness, smi, is_selfies, is_peptide, receptor)
```
We note: 
1. The processed ligands will be located within the newly created ligands directory.
2. The docked output from running QuickVina will be located in the newly created outputs directory.
3. The default behaviour of the program (QuickVina) is to make use of all available CPUs. 
4. The output from running the QuickVina calculation will be stored in the dictionary pose_pred_out.
5. The output from running the QuickVina NNScore2.0 will be stored in the dictionary re_scored_values.
6. No output csv files are created in this case. 

## Running calculations for multiple molecules: 
Please prepare a file concisting of a list of smiles. For example, `molecules.txt`: 
```
Index,Smiles
0,CCCCCCCCCC
1,C1=CC(=CC=C1CSCC2C(C(C(O2)N3C=NC4=C(N=CN=C43)N)O)O)Cl
``` 
The molecules can be run simply using out python function call: 
```
import os 
from run_vf_unity import main 

program_choice   = 'qvina'
scoring_function = '' 
center_x         = -17.820
center_y         = 16.140
center_z         = -18.643
size_x           = 20
size_y           = 20 
size_z           = 20
exhaustiveness   = 10
is_selfies       = False
is_peptide       = False
receptor         = './config/5wiu_test.pdbqt'

with open('./molecules.smi', 'r') as f: 
    lines = f.readlines()
lines = lines[1: ]

for item in lines: 
    idx,smi = item.split(',')
    pose_pred_out, re_scored_values = main(program_choice, scoring_function, center_x, center_y, center_z, size_x, size_y, size_z, exhaustiveness, smi, is_selfies, is_peptide, receptor)
    os.system('rm -rf ligands')
    os.system('cp -a outputs outputs_{}'.format(idx))
```
The corresponding index (column 1) of a molecule in the `molecules.txt` file will be used for storing the results in `outputs_*index*`

## Special Considerations
### Using AutoDock-GPU/CPU
Please compile the code using instructions from: [https://github.com/ccsb-scripps/AutoDock-GPU](https://github.com/ccsb-scripps/AutoDock-GPU). 
After successful compilation, within the bin directory, an executable will be made (example name: `autodock_gpu_1wi`). Then, the code is ready to run. 
We provide an example inside `./executables/vf_gpu_example.zip`. Inside the directory, a prepared protein-ligand pair is provided and the code can be run 
using: `./autodock_gpu_1wi --ffile 1stp_protein.maps.fld --lfile ./1stp_ligand.pdbqt`


### Using EquiBind
Please download the code using the instructions from: [https://github.com/HannesStark/EquiBind](https://github.com/HannesStark/EquiBind). 
Please create a conda environment per the instructions of the EquiBind repository. 
Copy-paste all files inside the working directory of VF Unity. 


### Using rDock
Please compile the code using the instructions provided in: [https://rdock.sourceforge.net/installation/](https://rdock.sourceforge.net/installation/). 
We installed rDock using Anaconda (with `conda install -c bioconda rdock`)

### Using MM-GBSA
Please install AmberTools: [https://ambermd.org/GetAmber.php#ambertools](https://ambermd.org/GetAmber.php#ambertools). We managed to perform the download using conda. 
Additionally, please note: the variable `chimera_path` should be updated to the location of the Chimera on your system. Chimera can be downloaded: [https://www.cgl.ucsf.edu/chimera/download.html](https://www.cgl.ucsf.edu/chimera/download.html).


### AutoDockZN
Requirments
Please install the ADFR software suite: [https://ccsb.scripps.edu/adfr/downloads/](https://ccsb.scripps.edu/adfr/downloads/). 
Please install meeko with: `pip install meeko`. 
Instructions for running AutoDockZN can be found on their official website: [https://autodock-vina.readthedocs.io/en/latest/docking_zinc.html](https://autodock-vina.readthedocs.io/en/latest/docking_zinc.html).


### AutoDockZN
Requirments
Please install the OpenEye software suite: [https://docs.eyesopen.com/toolkits/python/quickstart-python/install.html](https://docs.eyesopen.com/toolkits/python/quickstart-python/install.html). 

### SEED
Requirments
Please install AmberTools: [https://ambermd.org/GetAmber.php#ambertools](https://ambermd.org/GetAmber.php#ambertools).

### RF-score
Please paste the executable from [https://github.com/oddt/rfscorevs_binary](https://github.com/oddt/rfscorevs_binary) in the executables directory. 

### qvina_gpu
A qvina_gpu executable (of name 'qvina_gpu') should be placed in directory: /executables
Instructions for compilation are provided in https://github.com/DeltaGroupNJUPT/QuickVina2-GPU

### qvina_w_gpu
A qvina_gpu executable (of name 'qvina_w_gpu') should be placed in directory: /executables
Instructions for compilation are provided in https://github.com/DeltaGroupNJUPT/QVina-W-GPU

### vina_gpu
A qvina_gpu executable (of name 'vina_gpu') should be placed in directory: /executables
Instructions for compilation are provided in https://github.com/DeltaGroupNJUPT/Vina-GPU

### vina_gpu_2.0
A qvina_gpu executable (of name 'vina_gpu_2.0') should be placed in directory: /executables
Instructions for compilation are provided in https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0

### ADFR
An adfr executable (of name adfr) should be placed in directory: /executables directory. An executable needs to be compiled based on a user’s system using instructions described in [https://ccsb.scripps.edu/adfr/downloads/](https://ccsb.scripps.edu/adfr/downloads/).

### HDock
An HDock executable (of name hdock) should be placed in directory: /executables directory. 
An createpl executable (of name createpl) should be placed in directory: /executables directory. 


### FRED
Requirments
Please install OpenEye: [https://docs.eyesopen.com/toolkits/python/quickstart-python/install.html](https://docs.eyesopen.com/toolkits/python/quickstart-python/install.html).
Additionally, a valid OpenEye licence is required. Namely, a file named oe_license.txt needs to be placed in the working directory. 

### GalaxyDock3
Please paste the executable from [https://galaxy.seoklab.org/files/by2hsnvxjf/softwares/galaxydock.html](https://galaxy.seoklab.org/files/by2hsnvxjf/softwares/galaxydock.html) in the executables directory. 
Note a data directory (name 'data') is required for successful runs. 

### LightDock
We suggest downloading LightDock from: [https://github.com/lightdock/lightdock](https://github.com/lightdock/lightdock). For this, within the current working directory (VF-Unity), please run: 
```
git clone https://github.com/lightdock/lightdock.git
virtualenv venv
source venv/bin/activate
cd lightdock
pip install -e .
```

### MpSDockZN
Please add the executable (of name `MpSDock`) in executables directory.
Please install AmberTools: [https://ambermd.org/GetAmber.php#ambertools](https://ambermd.org/GetAmber.php#ambertools).
We managed to perform the download using conda. 
Please note: the variable `chimera_path` should be updated to the location of the Chimera on your system. Chimera can be downloaded: [https://www.cgl.ucsf.edu/chimera/download.html](https://www.cgl.ucsf.edu/chimera/download.html).
A working dock6 download (with a valid licence) is required. The variable `dock6_path` should be updated to location of the Chimera on your system.
A `box.in` input file is required for the program. Please specify the path in variable `box_in_file`.
A `grid.in` input file is required for the program. Please specify the path in variable `grid_in_file`.
A `dock.in` input file is required for the program. Please specify the path in variable `dock_in_file`.

### Running with CovDock
A valid Schrödinger license is required to run CovDock.

### Running with GlideSP/XP/HTVS
A valid Schrödinger license is required to run GlideSP/XP/HTVS.

### Re-scoring with GOLD fitness functions
A valid GOLD license is required to re-score docking results with GOLD fitness functions.  
An executable named `gold_auto` should be placed in the `/executables` directory.

When specifying a GOLD fitness function with which to re-score docking results, a dynamically loadable 
shared object library must be available in the top-level directory of this repository and named correspondingly 
with the parameter defined in the VFU configuration, i.e., one of `plp`, `asp`, `chemscore`, `goldscore`.

### Contributing
If you are interested in contributing to AdaptiveFlow, whether it is to report a bug or to extend AdaptiveFlow with your own code, please see the file [CONTRIBUTING.md](CONTRIBUTING.md) and the file [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).



### License
The project is distributed under the GNU GPL v2.0. Please see the file [LICENSE](LICENSE) for more details. 


### Citation
Gorgulla, Christoph, et al. "AI-Enhanced Adaptive Virtual Screening Platform Enabling Exploration of 69 Billion Molecules Discovers Structurally Validated FSP1 Inhibitors." bioRxiv (2025): 2025-8-14.
