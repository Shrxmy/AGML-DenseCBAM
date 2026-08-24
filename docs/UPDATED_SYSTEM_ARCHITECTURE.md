# Updated AGML-DenseCBAM System Architecture (V2)

The following Mermaid code reflects the final V2 implementation used for the completed five-fold experiment. It can be copied directly into a Mermaid-compatible Markdown editor or the Mermaid Live Editor.

```mermaid
flowchart TB
    %% =========================
    %% DATA GOVERNANCE
    %% =========================
    subgraph DATA["1. Dataset Governance and Five-Fold Preparation"]
        direction LR
        D1["Local source pool<br/>3,425 extracted TMJ images"]
        D2["SHA-256 content audit<br/>duplicate and label-conflict detection"]
        D3["Remove same-label duplicate copies<br/>exclude unresolved ambiguous groups under documented policy"]
        D4["Final experimental pool<br/>3,002 unique, non-conflicting images"]
        D5["Leakage-resistant stratified 5-fold split<br/>train / validation / held-out test"]
        D6["Fold manifests and fingerprints<br/>integrity checked before training"]

        D1 --> D2 --> D3 --> D4 --> D5 --> D6
    end

    %% =========================
    %% INPUT AND CORRUPTION
    %% =========================
    subgraph INPUT["2. Input and Synthetic Artifact Processing"]
        direction LR
        I1{"Experimental condition"}
        I2["Clean condition<br/>unmodified image<br/>artifact label: none"]
        I3["V2 artifact-mix condition<br/>uniformly sampled category"]
        I4["None"]
        I5["Motion blur<br/>horizontal kernel: 5, 7, or 9"]
        I6["Gaussian noise<br/>sigma: 8 to 12"]
        I7["Localized metal streak<br/>1-2 additive blurred streaks"]
        I8["Artifact target<br/>4 classes"]
        I9["Resize to 224 x 224 RGB<br/>DenseNet ImageNet preprocessing"]

        I1 -->|"Clean"| I2 --> I9
        I1 -->|"Artifact mix"| I3
        I3 --> I4 --> I9
        I3 --> I5 --> I9
        I3 --> I6 --> I9
        I3 --> I7 --> I9
        I3 -.-> I8
    end

    D6 --> I1

    %% =========================
    %% BENCHMARK MODEL
    %% =========================
    subgraph BENCH["3A. Reconstructed DenseNet201 Attention Benchmark - C1 and C2"]
        direction TB
        B1["ImageNet-pretrained DenseNet201<br/>full-network fine-tuning"]
        B2["pool3_relu feature map"]
        B3["Connected self-attention"]
        B4["Average pooling and 1x1 projection"]
        B5["conv5_block32_concat feature map"]
        B6["Concatenate pool3 attention features<br/>with final DenseNet features"]
        B7["1x1 fusion convolution<br/>1,024 channels"]
        B8["Global average pooling"]
        B9["Dense 1,024 + L2<br/>Dropout 0.5 + Batch normalization"]
        B10["Dense 128"]
        B11["TMD output<br/>Normal / Subluxation"]
        BL["Categorical cross-entropy<br/>class-weighted TMD loss"]

        B1 --> B2 --> B3 --> B4 --> B6
        B1 --> B5 --> B6
        B6 --> B7 --> B8 --> B9 --> B10 --> B11 --> BL
    end

    %% =========================
    %% PROPOSED MODEL
    %% =========================
    subgraph PROPOSED["3B. Proposed AGML-DenseCBAM V2 - C3 and C4"]
        direction TB
        P1["Shared ImageNet-pretrained DenseNet201 encoder<br/>full-network fine-tuning"]
        P2["Pre-CBAM final DenseNet feature map<br/>conv5_block32_concat"]

        subgraph PRIMARY["Primary TMD Branch"]
            direction TB
            P3["CBAM channel attention<br/>average/max descriptors + shared MLP"]
            P4["CBAM spatial attention<br/>channel average/max maps + 7x7 convolution"]
            P5["Global average pooling"]
            P6["Dense 1,024 + L2<br/>Dropout 0.5 + Batch normalization"]
            P7["Dense 128"]
            P8["TMD output<br/>Normal / Subluxation"]
            PT["Class-weighted TMD loss<br/>weight = 1.0"]

            P3 --> P4 --> P5 --> P6 --> P7 --> P8 --> PT
        end

        subgraph AUX["Auxiliary Synthetic-Artifact Branch"]
            direction TB
            A1["Pre-CBAM feature input<br/>preserves localized artifact evidence"]
            A2["Global average pooling"]
            A3["Global max pooling"]
            A4["Concatenate pooled features"]
            A5["Dense 256 + L2<br/>Dropout 0.3"]
            A6["Artifact output<br/>None / Blur / Noise / Metal streak"]
            AT["Artifact cross-entropy loss<br/>weight = 0.3"]

            A1 --> A2 --> A4
            A1 --> A3 --> A4
            A4 --> A5 --> A6 --> AT
        end

        PJ["Joint multi-task objective<br/>L = 1.0 L_TMD + 0.3 L_artifact"]

        P1 --> P2
        P2 --> P3
        P2 --> A1
        PT --> PJ
        AT --> PJ
    end

    I9 --> B1
    I9 --> P1
    I8 -.->|"auxiliary target"| A6

    %% =========================
    %% TRAINING AND EVALUATION
    %% =========================
    subgraph EVAL["4. Training, Evaluation, and Chapter IV Outputs"]
        direction LR
        E1["Adam optimizer<br/>learning rate 1e-4<br/>batch size 8<br/>maximum 50 epochs"]
        E2["Primary TMD validation-loss monitoring<br/>Early stopping: patience 5<br/>ReduceLROnPlateau: factor 0.1, patience 3"]
        E3["Held-out fold predictions"]
        E4["Accuracy, Precision, Recall,<br/>Specificity, F1 and confusion matrices"]
        E5["Artifact accuracy, macro-F1<br/>and per-class recall"]
        E6["ECE, inference throughput<br/>and latency"]
        E7["Grad-CAM<br/>benchmark fused features / proposed post-CBAM features"]
        E8["Five-fold mean, SD and 95% CI<br/>paired statistical comparisons"]

        E1 --> E2 --> E3
        E3 --> E4 --> E8
        E3 --> E5 --> E8
        E3 --> E6 --> E8
        E3 --> E7
    end

    BL --> E1
    PJ --> E1

    %% =========================
    %% CASE DEFINITIONS
    %% =========================
    C1["C1: Benchmark + Clean"]
    C2["C2: Benchmark + Artifact Mix"]
    C3["C3: Proposed + Clean"]
    C4["C4: Proposed + Artifact Mix"]

    I2 -.-> C1 -.-> B1
    I3 -.-> C2 -.-> B1
    I2 -.-> C3 -.-> P1
    I3 -.-> C4 -.-> P1

    %% =========================
    %% STYLING
    %% =========================
    classDef data fill:#E8F1FB,stroke:#2F5597,color:#111,stroke-width:1.5px;
    classDef process fill:#FFF2CC,stroke:#BF9000,color:#111,stroke-width:1.5px;
    classDef benchmark fill:#E2F0D9,stroke:#548235,color:#111,stroke-width:1.5px;
    classDef proposed fill:#FCE4D6,stroke:#C55A11,color:#111,stroke-width:1.5px;
    classDef output fill:#E4DFEC,stroke:#7030A0,color:#111,stroke-width:1.5px;
    classDef caseNode fill:#F2F2F2,stroke:#595959,color:#111,stroke-width:1.5px;

    class D1,D2,D3,D4,D5,D6 data;
    class I1,I2,I3,I4,I5,I6,I7,I8,I9 process;
    class B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,BL benchmark;
    class P1,P2,P3,P4,P5,P6,P7,P8,PT,A1,A2,A3,A4,A5,A6,AT,PJ proposed;
    class E1,E2,E3,E4,E5,E6,E7,E8 output;
    class C1,C2,C3,C4 caseNode;
```

## Figure caption

**Figure X. Updated system architecture of the proposed AGML-DenseCBAM V2 framework and reconstructed DenseNet201 attention benchmark.** The common five-fold pipeline applies identical data partitions, preprocessing, and synthetic artifact conditions to both architectures. The proposed model uses a shared DenseNet201 encoder, a post-CBAM primary TMD classification branch, and a pre-CBAM auxiliary artifact branch with combined global average and max pooling. The multi-task objective combines the class-weighted TMD loss and synthetic-artifact classification loss using weights of 1.0 and 0.3, respectively.

## Experimental case mapping

| Case | Architecture | Input condition | Outputs used for evaluation |
|---|---|---|---|
| C1 | Reconstructed DenseNet201 attention benchmark | Clean | TMD classification |
| C2 | Reconstructed DenseNet201 attention benchmark | V2 artifact mix | TMD classification |
| C3 | Proposed AGML-DenseCBAM V2 | Clean | TMD classification; artifact output is trivially none |
| C4 | Proposed AGML-DenseCBAM V2 | V2 artifact mix | TMD and four-class synthetic-artifact classification |

## Implementation safeguards to mention in the paper

1. The artifact categories are controlled **synthetic corruptions**, not independently annotated real clinical acquisition artifacts.
2. The local experimental pool contains 3,002 unique, non-conflicting images after exact-content auditing and documented exclusion of unresolved ambiguous groups; the source `data/` directory remains unchanged. State separately whether adviser/domain approval was obtained.
3. Patient identifiers were unavailable, so exact-image leakage is prevented but patient-level independence cannot be guaranteed.
4. Benchmark and proposed models use identical folds, input conditions, optimizer settings, batch size, and training hardware.
5. The artifact branch enters before CBAM because development-only validation showed that post-CBAM global-average features suppressed or erased localized streak evidence.
6. Grad-CAM is qualitative unless independently prepared expert ROI annotations are available.
