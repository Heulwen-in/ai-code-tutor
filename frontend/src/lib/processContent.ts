// Content for the public "How It's Built" process page (/process).
// Single source of truth: edit prose, metrics, tables, and figures here.
// Numbers verified against report Chapter 6 (6.1 architecture, 6.2 implementation),
// the *_v3_results.txt files, and the served pipeline thresholds in bug_classifier.py
// (NO_BUG 0.60, SUBTYPE 0.65, LINE 0.50). Prose intentionally avoids em dashes.

const IMG = "/process";

export type Figure = { src: string; alt: string; caption: string; wide?: boolean; notes?: string[] };
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
  { value: "0.9552", label: "Line hit@1" },
  { value: "0.8542", label: "Skill Macro F1" },
  { value: "14 -> 3", label: "Subtypes : Coarse" },
  { value: "2", label: "Roles" },
];

export const SECTION_NAV: { id: string; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "architecture", label: "Architecture" },
  { id: "pipeline", label: "Pipeline" },
  { id: "data", label: "Data" },
  { id: "eda", label: "EDA" },
  { id: "features", label: "Features" },
  { id: "baselines", label: "Baselines" },
  { id: "transformer", label: "Transformer" },
  { id: "skill", label: "Skill Detector" },
  { id: "feedback", label: "Feedback" },
  { id: "results", label: "Results" },
];

export const SECTIONS: Section[] = [
  {
    id: "overview",
    kicker: "01 : Overview",
    title: "From raw buggy code to a working AI tutor",
    prose: [
      "PyTutor is an adaptive AI Python tutor. Given a snippet of code it classifies the defect, localises the offending line, adapts its feedback to a Student or Professional role, recommends targeted lessons, and closes a spaced-repetition learning loop so recurring mistakes are revisited.",
      "The engine is a hierarchical CodeBERT pipeline that reaches 0.9556 macro F1 on problems it has never seen during training. The sections below are organised as follows: the system and AI-pipeline architecture, how the trained stages compose into one served inference pipeline, the data and its label schema, the exploratory analysis that shaped it, the three feature views, the classical baselines that set the floor, the transformer that beats them, the parallel skill detector, the role-adaptive feedback that turns a prediction into a lesson, and a look at where the model generalises and where it does not.",
    ],
    stats: [
      { value: "2", label: "Model tracks", caption: "bug classifier + skill detector" },
      { value: "14 -> 3", label: "Bug schema", caption: "subtypes to coarse classes + no_bug" },
      { value: "3", label: "Neural heads", caption: "Stage 1, Stage 2, line localiser" },
      { value: "1.0", label: "AST-gate confidence", caption: "exact line for parse errors" },
    ],
  },
  {
    id: "architecture",
    kicker: "02 : System & AI Architecture",
    title: "How the system is put together",
    prose: [
      "PyTutor is not a single predictive model but a closed learning loop: a user submits code, the system decides whether it is defective and, if so, of what kind and where, explains the defect at a depth matched to the user, recommends lessons, invites a revised resubmission, and records the episode for spaced review. Five design drivers govern every decision: role adaptivity (the same defect is explained differently to a student and a professional), pedagogical constraint (feedback to a student withholds the corrected code in favour of hints), transparency (the system exposes its coarse class, subtype, confidence and implicated line rather than an unqualified verdict), privacy (analysis and the feedback model run locally, so code need not leave the machine), and graceful degradation (advanced stages are optional, and a missing model artefact falls back to simpler but still-correct behaviour).",
      "The system is a four-layer architecture in which each layer depends only on the one beneath it, isolating the machine-learning concerns from the web concerns. The presentation layer captures code and role and renders results; the API layer validates every request, authenticates users, and mediates the services; the AI/application layer performs classification, localisation, skill estimation, feedback generation, recommendation and retention logic; and the data layer persists users, analyses and review items and stores the trained model artefacts. The Ollama runtime hosting Qwen2.5-Coder sits deliberately outside the application boundary, reached over local HTTP, which honours the privacy driver and lets the model be swapped without touching the app. The neural models are loaded once at startup, so inference latency excludes model loading.",
      "The AI is the core contribution and is best understood at three levels. At the basic level it is a two-stage engine: an Analysis Engine detects the defect class, localises the line and assesses skill, and its structured output drives an Adaptive Feedback Engine that generates a role- and skill-aware explanation and recommends lessons. At the component level a single CodeBERT backbone is fine-tuned into three heads (a coarse classifier, a masked subtype classifier and a token line-classifier), surrounded by a deterministic AST parser, a lightweight logistic-regression skill detector, and the local language model used only for the final explanation. At the detailed level the stages run from cheap and certain to progressively probabilistic, with confidence gates between them that protect valid code from being coerced into a defect class.",
    ],
    tables: [
      {
        caption: "Table 2.1 : The four architectural layers and their responsibilities.",
        headers: ["Layer", "Responsibility", "Principal technologies"],
        rows: [
          ["Presentation", "Renders the UI, captures code & role, client-side route protection", "Next.js, React, TypeScript, Recharts"],
          ["API", "REST endpoints, request/response validation, auth, service mediation", "FastAPI, Uvicorn, Pydantic, JWT"],
          ["AI / Application", "Classification, localisation, skill, feedback, recommendation, retention", "PyTorch, Transformers, scikit-learn, Ollama client"],
          ["Data & Persistence", "Users, analyses, lesson progress, review items; model artefacts", "SQLAlchemy, SQLite"],
        ],
      },
      {
        caption: "Table 2.2 : The AI components: one CodeBERT backbone specialised into three heads, plus a parser, a skill model and a local LLM.",
        headers: ["Component", "Type", "Task : Output"],
        rows: [
          ["Rule-based parser", "Python AST (deterministic)", "Parse-level defects : indentation_error + exact line"],
          ["Stage 1", "CodeBERT sequence (4 classes)", "Coarse defect : class + confidence"],
          ["Stage 2", "CodeBERT sequence (14, masked)", "Fine subtype : subtype + confidence"],
          ["Line localiser", "CodeBERT token classifier", "Localisation : line number (or none)"],
          ["Skill detector", "Logistic Regression (14 features)", "Register calibration : novice / professional"],
          ["Feedback model", "Qwen2.5-Coder 7B via Ollama", "Explanation : role-adaptive JSON"],
        ],
      },
      {
        caption: "Table 2.3 : The confidence gates that keep inference honest (each is configurable).",
        headers: ["Gate", "Value", "Effect"],
        rows: [
          ["No-bug (Stage 1)", "0.60", "Below this top probability, report no_bug"],
          ["Subtype (Stage 2)", "0.65", "Below this, discard as an unconfirmed false positive"],
          ["Line detection", "0.50", "Below this mean per-line probability, report no line"],
        ],
      },
    ],
    figures: [
      {
        src: `${IMG}/architecture_system.png`,
        alt: "Four-layer system architecture: presentation, API, AI/application and data layers, with the Ollama model outside the boundary",
        caption: "Figure 1 : Four-layer system architecture. Each layer depends only on the one beneath it; the Ollama model sits outside the boundary for privacy.",
      },
      {
        src: `${IMG}/ai_pipeline_basic.png`,
        alt: "AI pipeline basic view: a two-stage Analysis Engine feeding an Adaptive Feedback Engine and the learning loop",
        caption: "Figure 2 : AI pipeline (basic view). The Analysis Engine detects, localises and assesses skill; the Adaptive Feedback Engine explains and recommends; the loop closes with resubmission and review.",
      },
      {
        src: `${IMG}/ai_pipeline_detailed.png`,
        alt: "AI pipeline detailed view: AST gate, Stage 1, masked Stage 2, line localiser, parallel skill estimation, feedback, lessons and retention",
        caption: "Figure 3 : AI pipeline (detailed view). Stages run from cheap-and-certain to probabilistic, with confidence gates between them and skill estimation in parallel.",
      },
    ],
  },
  {
    id: "pipeline",
    kicker: "03 : Inference Pipeline",
    title: "How the trained stages compose at run time",
    prose: [
      "At inference the AST parser and the three neural heads compose into one served pipeline that prefers honesty to coverage. The Python AST parser runs first and owns indentation and parse-level errors, reporting the exact line at confidence 1.0 so no model is needed. Otherwise Stage 1 runs behind gate A, then the masked Stage 2 behind gate B, then the line localiser. Two gates keep it honest: the served coarse label is no_bug if the Stage 1 probability is below 0.60 or the Stage 2 subtype confidence is below 0.65, and a line is shown only if its score clears 0.50. The subtype gate is the subtle one: when Stage 1 fires on genuinely clean code, Stage 2 usually cannot confirm any subtype, so a low subtype confidence is a reliable signal of a false positive, and discarding on it suppresses the error at source. Every advanced stage degrades gracefully, so a missing Stage 2 or line model reduces the system to coarse-only classification without failure.",
      "The assembled result (class, subtype, line, confidence and skill) is the payload consumed by the adaptive feedback stage detailed in Section 10; the models that populate it are developed in the sections that follow.",
    ],
    pipeline: true,
  },
  {
    id: "data",
    kicker: "04 : Data & Analysis",
    title: "Datasets and the label schema",
    prose: [
      "The engine is trained and evaluated on four complementary corpora, because no single public dataset provides labelled Python defects, matched clean code, a skill signal and a real-world validation set at once. BuggedPythonLeetCode is the foundation: a fault-injection tool applies mutating transformations to canonical LeetCode solutions, generating up to fifteen bugged siblings per correct snippet, each carrying the injected fault name and the bug location. The clean no_bug class is built from the un-mutated originals (same provenance and style as the bugged code) topped up with the stylistically different flytech corpus, capped at a 4,000-snippet pool.",
      "The raw corpus exposes fifteen fault types, but one has only two usable samples and is discarded, leaving fourteen subtypes that collapse onto three coarse bug classes (syntax_error, logic_error, variable_misuse) plus no_bug, giving Stage 1 four classes. A fifth app-facing label, indentation_error, is not learned by any model: it is caught deterministically by the AST parse gate at inference, with the exact line at certainty. Separately, LeetCodeDataset (rated Easy or Hard) supplies the skill detector's proxy labels, and PyBugHive (real GitHub bug-fixes) is held out entirely as an external test set.",
      "All three model datasets derive from a single 70/15/15 train / validation / test partition taken over the 2,328 unique problems (1,630 / 349 / 349). Because the pipeline is hierarchical, each stage draws on a progressively narrower slice of that partition: Stage 1 sees all four classes, whereas Stage 2 and the line localiser are restricted to buggy snippets only, which is why their row counts fall in Table 4.3. This is the data the next section explores.",
    ],
    tables: [
      {
        caption: "Table 4.1 : The four corpora and the distinct role each plays.",
        headers: ["Dataset", "Approx. size", "Buggy?", "Role in project"],
        rows: [
          ["BuggedPythonLeetCode", "~28,000 rows", "Yes (injected)", "Primary training set : the bug classes"],
          ["flytech/python-codes-25k", "~25,000 snippets", "No", "Clean-code diversity : the no_bug pool"],
          ["LeetCodeDataset", "~2,900 solutions", "No", "Skill detector : Easy to novice, Hard to professional"],
          ["PyBugHive", "~1,255 issues", "Yes (real)", "External generalisation test : never trained on"],
        ],
      },
      {
        caption: "Table 4.2 : The 14 injected subtypes mapped onto 3 coarse classes (+ no_bug, + AST-gated indentation).",
        headers: ["Coarse class", "Subtypes", "Count"],
        rows: [
          ["syntax_error", "incorrect_type, missing_argument", "2"],
          ["logic_error", "swapped_comparison_operands, wrong_comparison_target, infinite_while_loop, non_existing_method, off_by_one_index, returning_early, swapped_for_range", "7"],
          ["variable_misuse", "forgotten_variable_update, incorrect_initialization, mutable_default_argument, use_before_definition, variable_name_typo", "5"],
          ["no_bug", "clean LeetCode originals + flytech snippets", ":"],
          ["indentation_error", "AST parse gate (not a model class)", ":"],
        ],
      },
      {
        caption: "Table 4.3 : The 70/15/15 split sizes per model dataset.",
        headers: ["Dataset", "Train", "Val", "Test"],
        rows: [
          ["Stage 1 (4-class)", "12,692", "2,679", "2,740"],
          ["Stage 2 (14 subtypes, bugs only)", "9,892", "2,079", "2,140"],
          ["Line localiser (bugs only)", "7,257", "1,521", "1,584"],
        ],
      },
    ],
    figures: [
      {
        src: `${IMG}/data_subtype_distribution.png`,
        alt: "Bar chart of the 14 injected bug subtypes coloured by parent coarse class",
        caption: "The 14 fine-grained subtypes, coloured by parent coarse class. A pronounced long tail exists: the smallest subtypes anticipate the widest confidence intervals in Stage 2.",
      },
    ],
  },
  {
    id: "eda",
    kicker: "05 : Exploratory Data Analysis",
    title: "Analysing the prepared data",
    prose: [
      "Exploratory analysis is not description for its own sake: each finding drives a concrete downstream decision. The four-class distribution is moderately imbalanced (variable_misuse and logic_error are the largest coarse classes, syntax_error the smallest buggy class, no_bug capped at 4,000), which is why macro-averaged F1 is the headline metric throughout, and why training uses inverse-frequency class weighting rather than resampling: the natural distribution is preserved while minority classes are not ignored.",
      "A second observation shapes every later modelling choice: the discriminative signal for these defects is lexical and local (a swapped operator, a misspelt identifier) rather than a matter of aggregate code shape. This foreshadows the baseline result that lexical features beat structural ones, and it warns that code length is confounded with difficulty, which is the reasoning that later removes length features from the baselines so the classifier cannot key on a spurious correlate.",
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
        caption: "Coarse label balance across the train / validation / test splits: imbalanced, motivating class weighting and macro F1.",
      },
      {
        src: `${IMG}/eda_code_length.png`,
        alt: "Code-length distribution across the corpus",
        caption: "Code-length distribution. Because length is confounded with difficulty, length-based features are later removed from the baselines.",
      },
    ],
  },
  {
    id: "features",
    kicker: "06 : Feature Extraction",
    title: "Three feature views for three model families",
    prose: [
      "Three representations are extracted from the same code. The transformer consumes CodeBERT subword tokenisation (max length 256) directly from the raw source, with no hand-engineering. The classical baselines use a tabular feature set: forty interpretable features were engineered in total (AST structural counts, token statistics, behavioural style indicators and size metrics), then seven length-correlated or shortcut-prone features were removed, leaving thirty-three structural features that describe code shape without encoding its provenance.",
      "The skill detector uses a separate fourteen-feature behavioural set describing coding style rather than correctness. The histograms below plot the behavioural features across the three defect classes, and their heavy overlap is the visual explanation of the roughly 0.62 macro F1 ceiling of the structural models: no single structural feature cleanly separates a syntax error from a logic error from a variable misuse. What distinguishes these defects is the specific tokens involved, not the aggregate shape of the code, which is why a term-frequency representation, and ultimately a contextual language model, are required.",
    ],
    tables: [
      {
        caption: "Table 6.1 : The three feature representations.",
        headers: ["Model family", "Representation", "Size"],
        rows: [
          ["CodeBERT (Stage 1/2/line)", "Subword tokens on raw code", "max 256 tokens"],
          ["Classical baselines", "Engineered tabular features", "40 to 33 (7 dropped)"],
          ["Skill detector", "Behavioural style features", "14 (11 behaviour + 3 size)"],
        ],
      },
    ],
    figures: [
      {
        src: `${IMG}/feature_histograms.png`,
        alt: "Histograms of behavioural features across the three defect classes",
        caption: "Behavioural features across the three defect classes. The heavy overlap explains why structural-only models plateau near 0.62 macro F1.",
      },
      {
        src: `${IMG}/features_analysis.png`,
        alt: "Feature analysis ranking features by between-class variance",
        caption: "Feature-extraction analysis: features ranked by between-class variance to select the most discriminative subset.",
      },
    ],
  },
  {
    id: "baselines",
    kicker: "07 : Machine Learning Baselines",
    title: "Classical models, and why the winner changed",
    prose: [
      "A large transformer is an expensive commitment that must be justified against a simpler alternative. Five algorithms were compared across two feature sets. Two findings follow. First, lexical evidence dominates structural evidence: the strongest structural model reaches only about 0.62 macro F1 while the strongest lexical (TF-IDF) model approaches 0.90, confirming that a coarse defect is signalled more by a swapped operator or misspelt identifier than by aggregate counts of functions or loops.",
      "Second, the highest training accuracy does not identify the best model. The SVM records the highest training accuracy of any baseline (0.9894), yet its validation and test scores fall behind, whereas XGBoost generalises best to unseen problems (0.8958 macro F1) and is therefore selected as the baseline. Only evaluation on genuinely unseen problems distinguishes the two, and this 0.8958 baseline is the bar the transformer must clear.",
    ],
    table: {
      caption: "Table 7.1 : Baseline classifiers on the held-out test set. SVM tops training but XGBoost generalises best.",
      headers: ["Model", "Features", "Train", "Val", "Test", "Macro F1"],
      rows: [
        ["XGBoost", "TF-IDF", "0.9524", "0.8879", "0.8897", "0.8958"],
        ["Logistic Regression", "TF-IDF", "0.9737", "0.8831", "0.8874", "0.8913"],
        ["SVM", "TF-IDF", "0.9894", "0.8850", "0.8785", "0.8839"],
        ["Random Forest", "TF-IDF", "0.9417", "0.8312", "0.8206", "0.8278"],
        ["Decision Tree", "TF-IDF", "0.8933", "0.7985", "0.7921", "0.8054"],
        ["XGBoost", "Structural / AST", "0.8470", "0.6065", "0.6131", "0.6154"],
        ["Random Forest", "Structural / AST", "0.7148", "0.5810", "0.5879", "0.5877"],
        ["Logistic Regression", "Structural / AST", "0.5572", "0.5445", "0.5696", "0.5701"],
      ],
    },
    figures: [
      {
        src: `${IMG}/baselines_model_comparison.png`,
        alt: "Bar chart comparing baseline model macro F1 scores",
        caption: "Baseline comparison: TF-IDF variants cluster near 0.88 to 0.90; structural variants near 0.51 to 0.62. The gap is the reference the transformer must clear.",
      },
    ],
  },
  {
    id: "transformer",
    kicker: "08 : Deep Learning Transformer",
    title: "The hierarchical CodeBERT model, stage by stage",
    prose: [],
    stats: [
      { value: "0.9556", label: "Stage 1 Macro F1", caption: "held-out test set" },
      { value: "0.9715", label: "Stage 2 Accuracy", caption: "0.9855 coarse-from-fine" },
      { value: "0.9552", label: "Line hit@1", caption: "token F1 0.9547, best epoch 3" },
      { value: "0.894 / 0.980", label: "no_bug recall", caption: "same-provenance / diverse clean" },
    ],
    tables: [
      {
        caption: "Table 8.1 : Backbone comparison (earlier 3-class task). Scores are close and RoBERTa is nominally top; CodeBERT is chosen on task fit and reuse.",
        headers: ["Encoder", "Macro F1", "Accuracy", "Variable F1", "Best epoch"],
        rows: [
          ["BERT", "0.9797", "0.9778", "0.9768", "13"],
          ["RoBERTa", "0.9807", "0.9792", "0.9799", "6"],
          ["CodeBERT (selected)", "0.9793", "0.9797", "0.9839", "5"],
        ],
      },
      {
        caption: "Table 8.2 : Stage 1 classification report on the unseen-problem test set.",
        headers: ["Class", "Precision", "Recall", "F1", "Support"],
        rows: [
          ["syntax_error", "0.98", "0.98", "0.9806", "411"],
          ["logic_error", "0.96", "0.93", "0.9471", "865"],
          ["variable_misuse", "0.99", "0.99", "0.9873", "864"],
          ["no_bug", "0.89", "0.93", "0.9073", "600"],
          ["Macro / Accuracy", "0.95", "0.96", "0.9556 / acc 0.9558", "2,740"],
        ],
      },
    ],
    figures: [
      {
        src: `${IMG}/transformer_backbone_comparison.png`,
        alt: "Comparison of BERT, RoBERTa, and CodeBERT against the best baseline",
        caption: "Backbone selection: the three encoders are close and all clear the baseline. CodeBERT is chosen on task fit, reuse and predictable failure, not on a higher score.",
        notes: [
          "Three pretrained encoders were fine-tuned under an identical protocol so the choice could rest on evidence: BERT, RoBERTa and CodeBERT. On raw score the three are almost tied, and RoBERTa is in fact nominally ahead (0.9807 macro F1 versus CodeBERT's 0.9793 on the earlier three-class task). CodeBERT was nonetheless selected, on three grounds that outweigh a hair's-breadth difference in score: task fit (it is pretrained on source code and is strongest on variable misuse, the class most dependent on identifier cues), downstream reuse (the same backbone is specialised into all three heads, an economy no general-language model offers), and predictable failure (a code-pretrained model's confidence behaves more sensibly on out-of-distribution input, which the confidence gates rely on).",
        ],
      },
      {
        src: `${IMG}/transformer_stage1_cm.png`,
        alt: "Confusion matrix for the Stage 1 four-class CodeBERT classifier",
        caption: "Stage 1 (4-class) confusion matrix on unseen problems. The dominant off-diagonal mass is between logic_error and no_bug, the confusion the gates handle conservatively.",
        notes: [
          "STAGE 1. The selected backbone becomes three coordinated heads. Stage 1 is the workhorse: a four-class sequence classifier that assigns the coarse defect (syntax_error, logic_error, variable_misuse or no_bug) shown first in the Analyse panel. On unseen problems it reaches 0.9556 macro F1 and 0.9558 accuracy, up from the 0.896 baseline. It is near perfect on variable_misuse (F1 0.987) and syntax_error (0.981); logic_error is the hardest coarse class (0.947) and no_bug the softest (0.907). A diagnostic detail is the no_bug recall split: 0.894 on clean code of the same competitive-programming provenance as the bugged examples, but 0.980 on stylistically diverse clean snippets, showing the model does not simply call code clean because it resembles a LeetCode solution.",
        ],
      },
      {
        src: `${IMG}/transformer_stage1_history.png`,
        alt: "Stage 1 training history over epochs",
        caption: "Stage 1 training history: validation macro F1 plateaus near 0.956.",
      },
      {
        src: `${IMG}/transformer_stage2_cm.png`,
        alt: "Confusion matrix for the Stage 2 fourteen-subtype classifier",
        caption: "Stage 2 (14 subtypes) confusion matrix. Almost all confusion is contained within a coarse group, the visual counterpart of the 0.9855 coarse-from-fine agreement.",
        notes: [
          "STAGE 1 to STAGE 2. Stage 1 answers what kind of defect is present; its coarse decision then conditions Stage 2, which answers which specific defect. Only once Stage 1 has named a class does the second head run, and it is constrained to refine that class rather than to reopen it.",
          "STAGE 2. Stage 2 refines the coarse class into one of fourteen human-readable subtypes (for example, from variable_misuse to variable_name_typo). Its defining choice is hierarchical masking: at inference its logits are restricted to the subtypes belonging to Stage 1's predicted class before the softmax, so a subtype is structurally incapable of contradicting its parent. It reaches 0.9715 accuracy and 0.9553 macro F1, and when its subtype is mapped back up it agrees with Stage 1 in 0.9855 of cases. The score distribution across subtypes is exactly what the label analysis predicted: well-supported, lexically distinctive subtypes such as use_before_definition exceed 0.99 F1, while the two weakest are simply the two smallest (wrong_comparison_target with 14 test samples, infinite_while_loop with 36), a consequence of scarce supervision reported transparently rather than a modelling failure.",
        ],
      },
      {
        src: `${IMG}/transformer_stage2_history.png`,
        alt: "Stage 2 training history over epochs",
        caption: "Stage 2 training history: validation macro F1 peaks before the epoch cap and triggers early stopping.",
      },
      {
        src: `${IMG}/transformer_line_history.png`,
        alt: "Training history of the line-detection token classifier",
        caption: "Line-localiser training history: token F1 keeps rising, but hit@1 peaks at epoch 3, so that checkpoint is deployed.",
        notes: [
          "STAGE 2 to LINE. Naming a defect is useful; pointing to it is more useful still. Independently of the class, a third head localises the offending line so the interface can highlight it.",
          "LINE LOCALISER. The line localiser is reframed from sequence to token classification: each subword token is scored for belonging to a buggy line, and the per-token scores are aggregated per line. Supervision is obtained automatically by diffing each bugged snippet against its correct original. It reaches token F1 0.9547 and, more importantly, line hit@1 0.9552, the metric that matches the interface, which highlights exactly one line. Its hit@1 peaks at epoch 3 and does not improve after, so the third-epoch checkpoint is deployed even though a later epoch reports a higher token F1: the metric the learner actually sees is the one optimised. A line is highlighted only when its score clears 0.50; otherwise none is shown, so localisation abstains rather than guessing.",
        ],
      },
    ],
  },
  {
    id: "skill",
    kicker: "09 : The Skill Detector",
    title: "Estimating novice versus professional style",
    prose: [
      "Running in parallel with the defect pipeline, and never gating any capability, a skill detector estimates whether a submission reads as novice or professional, so the feedback register and lesson difficulty can be pitched appropriately. Because no dataset of labelled novice and professional code exists, it is trained on a transparent proxy: LeetCode problems rated Easy are treated as novice and Hard as professional, with the ambiguous Medium tier excluded to sharpen the contrast, yielding a balanced 1,371 examples (686 novice, 685 professional). The proxy is stated openly as a limitation, which is exactly why the estimate is positioned as a soft calibration signal, never as a judgement of the user.",
      "Each submission is reduced to fourteen interpretable behavioural features that describe coding style rather than correctness: maximum and mean indentation depth, single-letter-variable count, comment ratio, the presence of type hints, comprehensions, generators and recursion, magic-number and built-in counts, a descriptive-naming ratio, and three size measures. A Logistic Regression over these features was selected on evidence, reaching 0.8542 accuracy and macro F1, well above a Decision Tree (0.8025) and a majority-class floor (0.334), with balanced per-class scores (novice F1 0.857, professional F1 0.851).",
      "Because the model is linear its coefficients can be read directly, and they are intuitive: greater code length, more single-letter variables and a higher line count push the estimate toward professional (the terse, dense competitive-programming style), while a higher comment ratio and shallower indentation push it toward novice (the more explanatory, less deeply nested code beginners tend to write). These are defensible stylistic correlates of experience, which is the point of choosing an interpretable model: a reviewer can audit why a submission was judged either way. The roughly 0.85 macro F1 is a deliberately modest, honestly-reported ceiling for a signal derived from a difficulty proxy, and a misjudged submission costs only a slightly mis-pitched explanation, never a withheld feature or a wrong defect verdict.",
    ],
    stats: [
      { value: "0.8542", label: "Skill Macro F1", caption: "LR; DT 0.8025, majority 0.334" },
      { value: "1,371", label: "Proxy examples", caption: "686 novice / 685 professional" },
      { value: "14", label: "Behavioural features", caption: "style, not correctness" },
      { value: "2", label: "Roles", caption: "calibration signal, not a verdict" },
    ],
    table: {
      caption: "Table 9.1 : Skill-detector model selection (Easy-to-novice, Hard-to-professional proxy).",
      headers: ["Model", "Accuracy", "Macro F1"],
      rows: [
        ["Logistic Regression (selected)", "0.8542", "0.8542"],
        ["Decision Tree", "0.8047", "0.8025"],
        ["Majority-class baseline", "0.5015", "0.3340"],
      ],
    },
    figures: [
      {
        src: `${IMG}/eda_features.png`,
        alt: "Behavioural feature usage and code length by problem difficulty",
        caption: "Behavioural feature usage and code length rise monotonically with problem difficulty (Easy to Hard), which is the signal that justifies the difficulty-to-skill proxy.",
      },
      {
        src: `${IMG}/pipeline_skill_cm.png`,
        alt: "Confusion matrix for the skill detector",
        caption: "Skill detector confusion matrix: balanced, near-symmetric errors, with a slight tendency to over-assign novice. Exactly the property a calibration signal should have.",
      },
    ],
  },
  {
    id: "feedback",
    kicker: "10 : Adaptive Feedback",
    title: "Turning a prediction into a lesson",
    prose: [
      "The assembled analysis result (class, subtype, line, confidence and skill) drives feedback generation. Three providers were compared on thirty cases with a manual rubric: a deterministic template, Gemini Flash 2.5 over a free API, and a local Qwen2.5-Coder via Ollama. The local model wins clearly on pedagogical quality (Correctness 3.77, Actionability 3.67, Clarity 3.83 out of 5) while staying perfectly reliable (valid JSON 1.00), whereas the template is reliable but thin and the hosted model is capable but unreliable (valid JSON 0.37).",
      "A single prompt is conditioned on role: for a student it forbids disclosing the corrected code and asks for hints and questions, simplified further when the skill detector reports a novice; for a professional it permits a direct technical diagnosis. If the local model is unreachable the system shows a visible feedback-unavailable state rather than fabricating tutor prose. Finally the recommender maps the class and role to lessons, and the loop records progress and schedules Leitner spaced repetition, closing the learning cycle.",
    ],
    stats: [
      { value: "1.00", label: "Valid-JSON rate", caption: "local model, 30 cases" },
      { value: "3.77 / 5", label: "Correctness", caption: "manual rubric, selected model" },
      { value: "2", label: "Role prompts", caption: "student vs professional" },
    ],
    table: {
      caption: "Table 10.1 : Feedback-provider comparison over 30 cases (rubric scores out of 5).",
      headers: ["Provider", "Valid-JSON", "Correctness", "Actionability", "Clarity"],
      rows: [
        ["Template baseline", "1.00", "2.07", "2.00", "4.00"],
        ["Gemini Flash 2.5 (API)", "0.37", "2.07", "2.13", "2.23"],
        ["Qwen2.5-Coder (local, selected)", "1.00", "3.77", "3.67", "3.83"],
      ],
    },
    figures: [
      {
        src: `${IMG}/pipeline_llm_comparison.png`,
        alt: "Chart comparing LLM feedback providers on the rubric",
        caption: "Feedback-provider comparison: the local Qwen2.5-Coder is alone in the upper region of both reliability and rubric quality.",
      },
    ],
  },
  {
    id: "results",
    kicker: "11 : Results & Generalisation",
    title: "Generalisation to real-world defects",
    prose: [
      "The pipeline was evaluated on an external test set of real GitHub bug-fixes that the model never encountered during training, isolating its capacity to generalise to defects drawn from a different distribution. On this external set the three-class macro F1 falls to 0.0477, with 94% of the genuinely buggy snippets (469 of 499) classified as no_bug.",
      "This outcome reflects a synthetic-to-real domain shift: the classifier was optimised to recognise the specific mutation signatures of injected defects, whereas real-world bugs exhibit a markedly different distribution of surface patterns. Confronted with an unfamiliar defect the model abstains and returns no_bug rather than issuing a confident but incorrect diagnosis, the behaviour the confidence gates are designed to enforce. This result is the direct motivation for the future-work step of fine-tuning the classifier on real bug-fix data.",
    ],
    stats: [
      { value: "0.0477", label: "External Macro F1", caption: "real GitHub bug-fixes" },
      { value: "94%", label: "Classified no_bug", caption: "469 of 499 real bugs" },
      { value: "3", label: "Coarse classes", caption: "external evaluation schema" },
    ],
    figures: [
      {
        src: `${IMG}/results_pybughive_cm.png`,
        alt: "Confusion matrix on the external test set showing predictions concentrated in the no_bug column",
        caption: "External test set: almost the entire mass falls into the no_bug column, bounding the model's external validity.",
      },
    ],
  },
];
