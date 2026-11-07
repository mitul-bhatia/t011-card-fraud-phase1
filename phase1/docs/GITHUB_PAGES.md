# One reading page for both phases

The repository has one homepage: root `index.html`. It links the notebooks, results, reports, presentations and study guides in `phase1/` and `phase2/`.

Reading copies are stored in `phase1/output/notebooks/` and `phase2/output/notebooks/`. Their Project home links return to the relevant section of the root index. The pages include saved code outputs and embedded figures.

To refresh the local pages after experiments:

```bash
python phase1/scripts/run_all.py
python phase2/scripts/run_all.py
python phase2/scripts/prepare_submission.py
python phase2/scripts/build_index.py
```

If publishing through GitHub Pages, preserve the complete relative folder structure and publish the repository root. Publishing, repository visibility and faculty access have not been changed by this local work. The old Phase 1 site ZIP is retained under `phase1/output/archive/` as a historical export.
