# Edit or rebuild the report

`Phase1_Report.tex` is the report source. `IEEEtran.cls` is the official IEEE LaTeX class, with its original license header preserved.

If you only need to read the report, open `../output/Phase1_Report.pdf`.

To rebuild it, run these commands from this `report` folder with a LaTeX installation:

```bash
pdflatex -output-directory=../output Phase1_Report.tex
pdflatex -output-directory=../output Phase1_Report.tex
```

The second run resolves citations and figure references. Keep the `phase1/figures/` folder next to `report/` when compiling. All numbers and charts come from the five notebooks. `../FILE_MAP.md` identifies each producing notebook. If you change the experiment, update the report and slides to match the new run.

The historical local `../output/archive/T011_Report_LaTeX.zip` convenience bundle contains `main.tex`, the IEEE class and the four report figures for Overleaf. ZIP bundles are excluded from Git; the editable source files remain here. If using the repository directly, upload `Phase1_Report.tex`, `IEEEtran.cls` and the four referenced figures to Overleaf, change the graphic path to `./figures/`, and set the report as the main document.

The report is jointly written by all five members. Its contribution section records the notebook leads and Mitul's overall project leadership. For the Phase 1 form, the final PDF is `../output/Phase1_Report.pdf`; keep the `.tex` source in the repository.
