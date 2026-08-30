# Clean AGML-DenseCBAM System Architecture

This version is intentionally simplified for a thesis figure. It follows the vertical, stage-based layout of the provided examples while retaining the components that are methodologically necessary. The primary diagram reflects the completed V2 experiment. The exploratory V3 training schedule changes optimization and augmentation, not the model's inference architecture.

## Main system architecture

```mermaid
flowchart TB
    A[("Source TMJ Radiographs<br/>3,425 extracted images")]

    subgraph S1["1. Data Governance"]
        B["SHA-256 content audit"]
        C["Duplicate removal and<br/>conflicting-label handling"]
        D["Final experimental pool<br/>3,002 images"]
    end

    subgraph S2["2. Five-Fold Data Preparation"]
        E["Stratified image-wise<br/>five-fold partitioning"]
        F["Training partition<br/>model fitting only"]
        G["Validation partition<br/>checkpoint selection only"]
        H["Held-out test partition<br/>final evaluation only"]
    end

    subgraph S3["3. Preprocessing and Experimental Conditions"]
        I["Training preprocessing<br/>Resize · augmentation · normalization"]
        J{"Training condition"}
        K["Clean"]
        L["Controlled artifact mix<br/>None · Blur · Noise · Metal streak"]
        M["Deterministic validation preprocessing<br/>same case condition"]
        MA["Deterministic test preprocessing<br/>same case condition"]
    end

    subgraph S4["4. Comparative Models"]
        direction LR

        subgraph BM["Reconstructed Benchmark"]
            B1["DenseNet201"]
            B2["Connected pool3<br/>self-attention"]
            B3["Feature fusion"]
            B4["TMD classifier"]
            B1 --> B2 --> B3 --> B4
        end

        subgraph PM["Proposed AGML-DenseCBAM"]
            P1["Shared DenseNet201"]
            P2["Post-CBAM<br/>TMD branch"]
            P3["Pre-CBAM<br/>artifact branch"]
            P4["Normal / Subluxation"]
            P5["None / Blur / Noise / Metal"]
            P1 --> P2 --> P4
            P1 --> P3 --> P5
        end
    end

    subgraph S5["5. Training and Model Selection"]
        N["Class-weighted optimization"]
        O["Primary validation-loss monitoring"]
        Q["Early stopping and<br/>best-checkpoint selection"]
        R["Training and validation<br/>loss-curve figures"]
        N --> O --> Q --> R
    end

    subgraph S6["6. Evaluation and Explainability"]
        T["Held-out fold predictions"]
        U["Accuracy · Precision · Recall<br/>Specificity · F1 · ECE"]
        V["Artifact metrics<br/>Throughput · Latency"]
        W["Grad-CAM panels"]
        X["Five-fold mean ± SD · 95% CI<br/>Shapiro-Wilk · Paired test · Cohen's dz"]
        T --> U --> X
        T --> V --> X
        T --> W
    end

    Y["C1: Benchmark + Clean"]
    Z["C2: Benchmark + Artifact Mix"]
    AA["C3: Proposed + Clean"]
    AB["C4: Proposed + Artifact Mix"]

    A --> B --> C --> D --> E
    E --> F
    E --> G
    E --> H
    F --> I --> J
    J --> K
    J --> L
    G --> M --> O
    H --> MA --> T

    K --> Y --> B1
    L --> Z --> B1
    K --> AA --> P1
    L --> AB --> P1

    B4 --> N
    P4 --> N
    P5 --> N
    N --> O
    Q --> T
    B3 -.->|"benchmark feature maps"| W
    P2 -.->|"post-CBAM feature maps"| W

    classDef stage fill:#FFFFFF,stroke:#333333,color:#111111,stroke-width:1.4px;
    classDef source fill:#F2F2F2,stroke:#222222,color:#111111,stroke-width:1.5px;
    classDef decision fill:#FFF4CC,stroke:#806000,color:#111111,stroke-width:1.4px;
    classDef output fill:#EAF2F8,stroke:#24536B,color:#111111,stroke-width:1.4px;
    classDef caseNode fill:#F7F7F7,stroke:#666666,color:#111111,stroke-width:1.2px;

    class A source;
    class B,C,D,E,F,G,H,I,K,L,M,MA,B1,B2,B3,B4,P1,P2,P3,P4,P5,N,O,Q,R stage;
    class J decision;
    class T,U,V,W,X output;
    class Y,Z,AA,AB caseNode;
```

## Proposed model detail

This second diagram can be used as a model-architecture figure when the main system diagram is too high-level.

```mermaid
flowchart LR
    A["224 × 224 RGB image"] --> B["DenseNet201 shared encoder"]
    B --> C["Pre-CBAM feature map<br/>conv5_block32_concat"]

    subgraph PRIMARY["Primary TMD Branch"]
        D["CBAM channel attention"] --> E["CBAM spatial attention"]
        E --> F["Global average pooling"]
        F --> G["Dense 1,024 · Dropout · BatchNorm"]
        G --> H["Dense 128"]
        H --> I["Normal / Subluxation"]
    end

    subgraph AUXILIARY["Auxiliary Artifact Branch"]
        J["Global average pooling"] --> L["Concatenate"]
        K["Global max pooling"] --> L
        L --> M["Dense 256 · Dropout"]
        M --> N["None / Blur / Noise / Metal streak"]
    end

    C --> D
    C --> J
    C --> K
    I --> O["TMD loss × 1.0"]
    N --> P["Artifact loss × 0.3"]
    O --> Q["Joint multi-task objective"]
    P --> Q
    E -.-> R["Grad-CAM target"]

    classDef block fill:#FFFFFF,stroke:#333333,color:#111111,stroke-width:1.3px;
    classDef output fill:#EAF2F8,stroke:#24536B,color:#111111,stroke-width:1.3px;
    class A,B,C,D,E,F,G,H,J,K,L,M,O,P,Q,R block;
    class I,N output;
```

## Figure caption

**Figure X. System architecture of the AGML-DenseCBAM research pipeline.** Existing TMJ radiographic images undergo exact-content auditing, conflicting-label handling, and stratified image-wise five-fold preparation. Identical clean and controlled synthetic-artifact conditions are supplied to the reconstructed DenseNet201 attention benchmark and proposed AGML-DenseCBAM model. The proposed model uses a shared DenseNet201 encoder, a post-CBAM primary TMD branch, and a pre-CBAM auxiliary artifact branch. Best checkpoints are selected using primary validation TMD loss. Held-out predictions are evaluated using classification, calibration, artifact, and efficiency metrics, while Grad-CAM provides qualitative model-attention visualizations.

## Experimental case mapping

| Case | Model | Condition | Evaluated outputs |
|---|---|---|---|
| C1 | Reconstructed benchmark | Clean | TMD classification |
| C2 | Reconstructed benchmark | Artifact mix | TMD classification |
| C3 | Proposed AGML-DenseCBAM | Clean | TMD classification; auxiliary target is `none` |
| C4 | Proposed AGML-DenseCBAM | Artifact mix | TMD and four-class synthetic-artifact classification |

## Important interpretation notes

1. The split is image-wise because patient and original-panorama identifiers were unavailable.
2. Exact-image leakage is prevented, but patient-level independence cannot be guaranteed.
3. Artifact categories are controlled synthetic corruptions, not clinically annotated real acquisition artifacts.
4. Grad-CAM is qualitative unless independently prepared expert ROI annotations are supplied.
5. The V3 two-stage training extension does not change the proposed inference graph shown above. It changes training-only augmentation and which DenseNet layers are trainable during each optimization stage.
