// Content for the public "How It's Built" process page (/process).
// Single source of truth — edit prose, metrics, tables, and figures here.
// Numbers verified against ml_pipeline/data/processed/*_v3_results.txt,
// subtype_mapping_v3.json, baseline_results_v3.txt, and the data-prep scripts.

const IMG = "/process";

export type Figure = { src: string; alt: string; caption: string; wide?: boolean };
export type Stat = { value: string; label: string; caption?: string };
export type Table = { headers: string[]; rows: string[][]; caption?: string };

export type Section = {
  id: string;
  kicker: string;
  title: string;
  prose: string[];
  stats?: Stat[];
  table?: Table;
  tables?: Table[];
  figures?: Figure[];
  pipeline?: boolean;
};

// Hero summary pills (reuse .stat-pill styling from the landing page).
export const HERO_PILLS: Stat[] = [
  { value: "0.9556", label: "Stage 1 Macro F1" },
  { value: "0.9715", label: "Stage 2 Accuracy" },
  { value: "0.955", label: "Line hit@1" },
  { value: "0.8542", label: "Skill Macro F1" },
  { value: "14 → 3", label: "Subtypes → Coarse" },
  { value: "2", label: "Roles" },
];

export const SECTION_NAV: { id: string; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "data", label: "Data" },
  { id: "eda", label: "EDA" },
  { id: "preprocessing", label: "Preprocessing" },
  { id: "features", label: "Features" },
  { id: "baselines", label: "Baselines" },
  { id: "transformer", label: "Transformer" },
  { id: "pipeline", label: "Pipeline" },
  { id: "results", label: "Results" },
];

export const SECTIONS: Section[] = [
  {
    id: "overview",
    kicker: "01 · Overview",
    title: "From raw buggy code to a working AI tutor",
    prose: [
      "PyTutor is an adaptive AI Python tutor. Given a snippet of code it classifies the defect, localises the offending line, adapts its feedback to a Student or Professional role, recommends targeted lessons, and closes a spaced-repetition learning loop so recurring mistakes are revisited.",
      "The engine is a hierarchical CodeBERT pipeline that reaches 0.9556 macro F1 on problems it has never seen during training. The sections below follow the build in order — the data and its label schema, the exploratory analysis that shaped it, the leakage-free preprocessing, the three feature views, the classical baselines that set the floor, the transformer that beats them, how the stages compose at inference, and an honest look at where the model generalises and where it does not. Each section builds directly on the one before it.",
    ],
    stats: [
      { value: "2", label: "Model tracks", caption: "bug classifier + skill detector" },
      { value: "14 → 3", label: "Bug schema", caption: "subtypes → coarse classes + no_bug" },
      { value: "3", label: "Neural heads", caption: "Stage 1 · Stage 2 · line localiser" },
      { value: "1.0", label: "AST-gate confidence", caption: "exact line for parse errors" },
    ],
    pipeline: true,
    figures: [
      {
        src: `${IMG}/architecture_system.png`,
        alt: "System architecture diagram of the PyTutor pipeline from code input to feedback",
        caption: "System architecture — code flows through the AI analysis pipeline to role-adaptive feedback and the learning loop.",
      },
    ],
  },
  {
    id: "data",
    kicker: "02 · Data & Analysis",
    title: "Datasets and the label schema",
    prose: [
      "The system has two model tracks, and they use different data. The bug classifier is trained on BuggedPythonLeetCode: correct LeetCode solutions with programmatically injected bugs (the OpenBugger transformers), which give 14 fine-grained bug subtypes. The clean no_bug class is drawn from the original, un-mutated solutions and topped up with the flytech/python-codes-25k corpus for stylistic diversity, giving a 4,000-sample no_bug pool. LeetCodeDataset is not used here.",
      "The 14 subtypes collapse onto 3 coarse bug classes — syntax_error, logic_error and variable_misuse — plus no_bug, giving the Stage 1 model 4 classes. A fifth app-facing label, indentation_error, is not learned by any model: it is caught deterministically by an AST parse gate at inference. Separately, the skill detector is trained on LeetCodeDataset (Easy → novice, Hard → professional), and PyBugHive — real GitHub bug-fix patches — is held out entirely as an external test set. This is the data the next section explores.",
    ],
    tables: [
      {
        caption: "Table 2.1 — Datasets and their role in the study.",
        headers: ["Dataset", "Nature", "Used for"],
        rows: [
          ["BuggedPythonLeetCode", "Injected bugs (14 subtypes)", "Bug classifier — bug classes"],
          ["flytech/python-codes-25k", "Clean Python snippets", "Bug classifier — no_bug pool"],
          ["LeetCodeDataset", "Rated solutions (Easy/Hard)", "Skill detector — novice/professional"],
          ["PyBugHive", "Real GitHub bug-fixes", "External generalisation test only"],
          ["CodeXGLUE · Defectors", "Defect corpora", "Considered in EDA, not adopted"],
        ],
      },
      {
        caption: "Table 2.2 — The 14 injected subtypes mapped onto 3 coarse classes (+ no_bug, + AST-gated indentation).",
        headers: ["Coarse class", "Subtypes", "Count"],
        rows: [
          ["syntax_error", "incorrect_type, missing_argument", "2"],
          ["logic_error", "swapped_comparison_operands, wrong_comparison_target, infinite_while_loop, non_existing_method, off_by_one_index, returning_early, swapped_for_range", "7"],
          ["variable_misuse", "forgotten_variable_update, incorrect_initialization, mutable_default_argument, use_before_definition, variable_name_typo", "5"],
          ["no_bug", "clean originals + flytech", "—"],
          ["indentation_error", "AST parse gate (not a model class)", "—"],
        ],
      },
    ],
    figures: [
      {
        src: `${IMG}/data_subtype_distribution.png`,
        alt: "Bar chart of the 14 raw injected bug subtypes and their counts",
        caption: "The 14 raw injected bug subtypes before mapping — these are what the classifier is ultimately built from.",
      },
    ],
  },
  {
    id: "eda",
    kicker: "03 · Exploratory Data Analysis",
    title: "Analysing the prepared data",
    prose: [
      "With the datasets from Section 2 in hand, exploratory analysis looks at how the labels are distributed once the 14 subtypes are mapped to the 3 coarse classes plus no_bug. The distribution is clearly imbalanced — logic_error and variable_misuse dominate, syntax_error is smallest — which directly motivates the class weighting applied in the next section, and warns that raw accuracy would be a misleading headline metric (hence macro F1 throughout).",
      "A second observation shapes every later modelling choice: the lexical content around a defect — the specific tokens, identifiers and operators — carries strong, learnable signal for the bug type, more so than coarse structural counts. This foreshadows the baseline result that TF-IDF features beat hand-crafted structural features, and the decision to fine-tune a code-aware transformer rather than rely on tabular features.",
    ],
    stats: [
      { value: "logic + variable", label: "Majority classes", caption: "syntax_error is the minority" },
      { value: "Macro F1", label: "Chosen metric", caption: "robust to the imbalance" },
      { value: "Lexical", label: "Dominant signal", caption: "tokens beat structure" },
    ],
    figures: [
      {
        src: `${IMG}/data_label_distribution.png`,
        alt: "Bar chart of the 4-class label distribution across train, validation, and test splits",
        caption: "Coarse label balance across the grouped train / validation / test splits — imbalanced, motivating class weighting.",
      },
      {
        src: `${IMG}/data_line_distribution.png`,
        alt: "Distribution of buggy-line positions used for the line-detection task",
        caption: "Where injected bugs sit — the line-label distribution that the line localiser is trained on.",
      },
    ],
  },
  {
    id: "preprocessing",
    kicker: "04 · Preprocessing",
    title: "Cleaning, mapping, and a leakage-free split",
    prose: [
      "Preprocessing normalises the heterogeneous source labels onto the 4-class schema from Section 2, filters out trivially short snippets, deduplicates, and applies class weighting to counter the imbalance found in Section 3.",
      "The decisive step is the data split. An initial random split let bugged variants of the same underlying problem fall into both train and test — because BuggedPythonLeetCode makes up to 15 near-identical variants per solution — so the model could memorise problems instead of learning bug patterns. This was replaced with a problem-grouped split: a problem id is the hash of its original clean solution, and every variant of a problem is forced into a single split. The split was audited to have zero problems spanning partitions, and one split assignment drives all three model datasets so they are mutually leakage-free by construction. Every model in the following sections is trained and evaluated on this honest split.",
    ],
    stats: [
      { value: "2,328", label: "Unique problems", caption: "grouped by original solution" },
      { value: "1,630 / 349 / 349", label: "Train / Val / Test", caption: "problem-level split" },
      { value: "0", label: "Problems spanning splits", caption: "leakage audit asserted" },
    ],
    table: {
      caption: "Table 4.1 — Grouped split sizes per model dataset.",
      headers: ["Dataset", "Train", "Val", "Test"],
      rows: [
        ["Stage 1 (4-class)", "12,692", "2,679", "2,740"],
        ["Stage 2 (14 subtypes, bugs only)", "9,892", "2,079", "2,140"],
        ["Line detection (bugs only)", "7,257", "1,521", "1,584"],
      ],
    },
    figures: [
      {
        src: `${IMG}/preprocessing_leakage_audit.png`,
        alt: "Leakage audit chart confirming no problem appears in more than one split",
        caption: "Leakage audit of the grouped split — no problem appears in more than one partition.",
      },
    ],
  },
  {
    id: "features",
    kicker: "05 · Feature Extraction",
    title: "Three feature views for three model families",
    prose: [
      "Three different representations are extracted from the same code. The transformer consumes CodeBERT subword tokenisation (max length 256) directly from the raw source — no hand-engineering. The classical baselines instead use a tabular feature set: 40 engineered features were designed in total (AST structural counts, token statistics, behavioural style features, and code-size metrics). For the v3 baseline, 7 length-correlated or shortcut-prone features were removed — code length, line count, average line length, comment ratio, and the assert / raise / try-except counts — leaving 33 structural features, so the model cannot cheat by keying on how long the code is.",
      "The skill detector uses a separate 14-feature behavioural set describing coding style rather than correctness: 11 behavioural features (indentation depth, single-letter variables, type hints, comprehensions, generators, recursion, magic numbers, built-ins, descriptive naming, comment ratio) plus 3 size metrics. The histogram below plots those 11 behavioural features across the three bug classes — the 3 pure size metrics are excluded because they describe length, not behaviour. The analysis figure ranks features by between-class variance to confirm which ones actually separate the classes; this is what selects the most discriminative subset rather than using all 40 blindly.",
    ],
    tables: [
      {
        caption: "Table 5.1 — The three feature representations.",
        headers: ["Model family", "Representation", "Size"],
        rows: [
          ["CodeBERT (Stage 1/2/line)", "Subword tokens on raw code", "max 256 tokens"],
          ["Classical baselines", "Engineered tabular features", "40 → 33 (7 dropped)"],
          ["Skill detector", "Behavioural style features", "14 (11 behaviour + 3 size)"],
        ],
      },
    ],
    figures: [
      {
        src: `${IMG}/features_behaviour.png`,
        alt: "Histograms of 11 behavioural features across the three bug classes",
        caption: "The 11 behavioural features (size metrics excluded), plotted across the three bug classes.",
      },
      {
        src: `${IMG}/features_analysis.png`,
        alt: "Feature analysis ranking features by between-class variance",
        caption: "Feature-extraction analysis — features ranked by between-class variance to select the most discriminative subset.",
      },
    ],
  },
  {
    id: "baselines",
    kicker: "06 · Machine Learning Baselines",
    title: "Classical models — and why the winner changed",
    prose: [
      "Before the transformer, classical models establish a competent floor on the tabular features from Section 5. Two findings stand out. First, lexical beats structural: TF-IDF models reach ~0.89 macro F1 while models built only on structural/AST features plateau at 0.51–0.62, confirming the EDA observation that token content carries the signal.",
      "Second, the honest split changed which model wins. On the earlier leaky split the best baseline was an SVM (TF-IDF) at 0.9278 macro F1. On the grouped split the SVM still records the highest training accuracy (0.9894) — but it overfits: its validation and test scores fall behind. XGBoost (TF-IDF) generalises best on unseen problems (0.8897 test accuracy, 0.8958 macro F1) and becomes the selected baseline. Reading the train-versus-test columns together is the whole point — the highest training number is not the best model.",
    ],
    table: {
      caption: "Table 6.1 — Baseline classifiers on the grouped split. SVM tops training but XGBoost generalises best.",
      headers: ["Model", "Features", "Train", "Val", "Test", "Macro F1"],
      rows: [
        ["XGBoost", "TF-IDF", "0.9524", "0.8879", "0.8897", "0.8958"],
        ["Logistic Regression", "TF-IDF", "0.9737", "0.8831", "0.8874", "0.8913"],
        ["SVM", "TF-IDF", "0.9894", "0.8850", "0.8785", "0.8839"],
        ["Random Forest", "TF-IDF", "0.9417", "0.8312", "0.8206", "0.8278"],
        ["XGBoost", "Structural / AST", "0.8470", "0.6065", "0.6131", "0.6154"],
        ["Logistic Regression", "Structural / AST", "0.5572", "0.5445", "0.5696", "0.5701"],
      ],
    },
    figures: [
      {
        src: `${IMG}/baselines_model_comparison.png`,
        alt: "Bar chart comparing baseline model macro F1 scores",
        caption: "Baseline comparison — TF-IDF variants clearly outperform structural-feature variants.",
      },
    ],
  },
  {
    id: "transformer",
    kicker: "07 · Deep Learning Transformer",
    title: "The hierarchical CodeBERT model, stage by stage",
    prose: [
      "The 0.896 baseline is the bar the transformer must clear. Three encoders were fine-tuned under identical conditions — BERT, RoBERTa and CodeBERT — and CodeBERT won, as expected from its code-aware pre-training. It is then used as three coordinated heads on the grouped split.",
      "Stage 1 predicts the 4 coarse classes and reaches 0.9556 macro F1 (0.9558 accuracy) on unseen problems — up from the 0.896 baseline, and to be read against the leakage-inflated 0.9840 that the naive split produced. Its confusion matrix is cleanest on variable_misuse (F1 0.987) and syntax_error (0.981); no_bug is the softest class (0.907), the natural place for a defect to hide.",
      "Stage 2 refines a detected defect into one of the 14 subtypes and reaches 0.9715 accuracy (0.9553 macro F1). Its logits are masked to Stage 1's predicted coarse group, so a subtype can never contradict the coarse call; coarse-from-fine agreement is 0.986. The weakest subtype is wrong_comparison_target, which has only 17 test samples — a small-sample caveat worth stating aloud in a viva.",
      "A third token-classification head localises the offending line, reaching token F1 0.955 and line hit@1 0.955 (its best epoch was epoch 3, after which it began to overfit). Three confidence gates keep inference honest — a no-bug threshold of 0.60, a subtype threshold of 0.65, and a line threshold of 0.50 — below which the pipeline abstains rather than guessing.",
    ],
    stats: [
      { value: "0.9556", label: "Stage 1 Macro F1", caption: "grouped · leaky was 0.9840" },
      { value: "0.9715", label: "Stage 2 Accuracy", caption: "0.986 coarse-from-fine" },
      { value: "0.955", label: "Line hit@1", caption: "token F1 0.955" },
      { value: "0.60 / 0.65 / 0.50", label: "Gates", caption: "no-bug / subtype / line" },
    ],
    figures: [
      {
        src: `${IMG}/transformer_backbone_comparison.png`,
        alt: "Comparison of BERT, RoBERTa, and CodeBERT macro F1",
        caption: "Backbone selection — CodeBERT's code-aware pre-training edges out BERT and RoBERTa.",
      },
      {
        src: `${IMG}/transformer_stage1_cm.png`,
        alt: "Confusion matrix for the Stage 1 four-class CodeBERT classifier",
        caption: "Stage 1 (4-class) confusion matrix on unseen problems — no_bug is the softest class.",
      },
      {
        src: `${IMG}/transformer_stage1_history.png`,
        alt: "Stage 1 training history over epochs",
        caption: "Stage 1 training history — validation F1 climbing to 0.957 by epoch 10.",
      },
      {
        src: `${IMG}/transformer_stage2_cm.png`,
        alt: "Confusion matrix for the Stage 2 fourteen-subtype classifier",
        caption: "Stage 2 (14 subtypes) confusion matrix — masked to the Stage 1 coarse group.",
      },
      {
        src: `${IMG}/transformer_stage2_history.png`,
        alt: "Stage 2 training history over epochs",
        caption: "Stage 2 training history across epochs.",
      },
      {
        src: `${IMG}/transformer_line_history.png`,
        alt: "Training history of the line-detection token classifier",
        caption: "Line-localiser training history — best line hit@1 at epoch 3.",
      },
    ],
  },
  {
    id: "pipeline",
    kicker: "08 · The Pipeline",
    title: "How the stages compose at inference",
    prose: [
      "At inference the heads from Section 7 run in a fixed order. An AST parse gate runs first and owns indentation and parse-level syntax errors: when Python cannot parse the code the exact line is known and confidence is 1.0, so no model is needed. Otherwise Stage 1 runs behind its no-bug gate, then the masked Stage 2, then the line localiser.",
      "In parallel, the skill detector — a Logistic Regression on the 14 behavioural features, 0.8542 macro F1 — infers Novice or Professional. Its output and the detected bug drive role-adaptive feedback generated by a local Qwen2.5-Coder model via Ollama, with an honest error path when the model is unreachable rather than silent boilerplate. A provider comparison found the local model scored highest on the manual rubric. Finally the recommender selects lessons, and the learning loop records progress and schedules Leitner-style spaced repetition.",
    ],
    pipeline: true,
    figures: [
      {
        src: `${IMG}/pipeline_skill_cm.png`,
        alt: "Confusion matrix for the skill detector",
        caption: "Skill detector — Novice vs Professional confusion matrix (0.8542 macro F1).",
      },
      {
        src: `${IMG}/pipeline_llm_comparison.png`,
        alt: "Chart comparing LLM feedback providers on the rubric",
        caption: "Feedback-provider comparison — the local Qwen2.5-Coder scores highest on the manual rubric.",
      },
    ],
  },
  {
    id: "results",
    kicker: "09 · Results & Generalisation",
    title: "Honest scores, and an honest limitation",
    prose: [
      "On the grouped split the pipeline performs strongly on problems never seen in training: Stage 1 at 0.9556 macro F1, Stage 2 at 0.9715 accuracy, and line localisation at 0.955 hit@1. The end-to-end hierarchical confusion matrix below shows the coarse-to-subtype path holding together across all 14 subtypes.",
      "The external result is reported in full rather than hidden. On PyBugHive — real GitHub bug-fixes the model never trained on — it reaches only 0.048 macro F1, with 94% of genuinely buggy snippets collapsing into no_bug. This quantifies a synthetic-to-real domain shift: a model trained on injected bugs learns the signature of injection, not the full diversity of real defects, and problem-grouping alone cannot close that gap. Stating this is a strength of the evaluation — it bounds exactly where the system can be trusted and points to fine-tuning on real bug-fix data as the clear next step.",
    ],
    stats: [
      { value: "0.9556", label: "In-domain Macro F1", caption: "grouped, unseen problems" },
      { value: "0.048", label: "PyBugHive Macro F1", caption: "external real bugs" },
      { value: "94%", label: "no_bug collapse", caption: "on genuinely buggy code" },
    ],
    figures: [
      {
        src: `${IMG}/results_hierarchical_cm.png`,
        alt: "Confusion matrix of the full hierarchical pipeline on the in-domain test set",
        caption: "End-to-end hierarchical pipeline on the in-domain grouped test set.",
      },
      {
        src: `${IMG}/results_pybughive_cm.png`,
        alt: "Confusion matrix on the external PyBugHive test set showing collapse to no_bug",
        caption: "External PyBugHive test — most real bugs collapse into the no_bug column, exposing the domain shift.",
      },
    ],
  },
];
