# Simple Workflow Chart

This is a compact workflow chart for the top reconstruction pipeline.

```mermaid
flowchart TD
    A[Automation / Recipe\nfull_chain] --> B([Input ROOT Sample\nttbar.root\ntruth triplets + genjet branches])

    B --> C[Stage 1: dataset_build\nenumerate triplets\ncompute features\nassign is_truth]
    C --> C1([triplets_raw.parquet\ntriplet-level rows\nfeatures + observables + is_truth])

    C1 --> D[Stage 2: dataset_prepare\ndeterministic event split\ntrain / val / test]
    D --> D1([train.parquet])
    D --> D2([val.parquet])
    D --> D3([test.parquet\nheld-out analysis sample])

    D1 --> E[Stage 3: train\nfit classifier\nXGBoost or TabPFN]
    D2 --> E
    E --> E1([trained model\nmodel_xgb.json or model_tabpfn.pkl])

    D3 --> F[Stage 4: infer\nscore all triplets in test set]
    E1 --> F
    F --> F1([inference_test_xgb.parquet\nfull scored triplet sample\nincludes score_xgb])

    F1 --> G[Stage 5: analysis / select_triplets\napply selection strategy\ndefault: greedy_disjoint]
    G --> G1([selected_triplets.parquet\nselected candidate triplets])
    G --> G2([event_selection.parquet\nper-event top candidates\nand summary variables])
    G --> G3([selection_report.json\nstrategy-level summary])

    A --> H([run_manifest.json\nparameters + artifact locations\ncompletion status])
    G3 --> H

    B -. diagnostics .-> X[Branch / sample inspection\ntriplet interpretation checks]

    T[Truth note:\nis_truth is built in Stage 1 and retained through inference,\nso strategy comparisons on the held-out test split are possible.]
    F1 -. truth retained .-> T
    G -. compare strategies .-> T
```

## Notes

- Rectangles represent processing stages.
- Rounded boxes represent stored artifacts.
- Dashed arrows represent auxiliary diagnostics or annotations.
- Analysis runs on the held-out test split, not on training events.