# Tutor Feedback Action Plan

## 1. Best Model Answer

Current evidence from `ml_pipeline/data/processed/transformer_results.txt`:

| Model | Test accuracy | Macro F1 | Training time |
|---|---:|---:|---:|
| SVM TF-IDF | 0.9183 | 0.9278 | see baseline logs |
| BERT | 0.9740 | 0.9778 | 1712.9s |
| RoBERTa | 0.9835 | 0.9841 | 1824.9s |
| CodeBERT | 0.9849 | 0.9855 | 1664.4s |

Best current model: **CodeBERT**.

Reason:
- Highest macro F1: 0.9855.
- Highest test accuracy: 0.9849.
- Best per-class balance among the 3 transformer models.
- Faster than RoBERTa in the recorded run.

Important note:
RoBERTa is very close. The final report should say CodeBERT is selected, but RoBERTa is a strong second-place model.

## 2. Does Tutor Feedback Fit Current Progress?

Yes, mostly.

Already done:
- TF-IDF baseline models exist.
- SVM TF-IDF is the best classical baseline.
- BERT, RoBERTa, and CodeBERT have already been trained and compared.
- Training time is already recorded in `ml_pipeline/logs/transformer_results.json`.
- Indentation detection exists in `backend/app/services/bug_classifier.py`.

Missing or weak:
- Current transformer run is 5 epochs, not 30.
- Current transformer script did not have configurable early stopping.
- Current TF-IDF best model is classical LinearSVC, which is not epoch-based.
- Indentation detection needs a self-labeled validation set and explanation file.
- The report needs clearer explanation of incorrect/misclassified parse cases.

## 3. TF-IDF 15-Epoch Observation

The existing SVM TF-IDF model does not train by epochs. It trains to an optimizer convergence criterion.
So a separate TF-IDF epoch experiment is needed.

Use:

```bash
cd F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code
.\.venv\Scripts\activate
python ml_pipeline\tfidf_epoch_experiment.py --epochs 15 --patience 3
```

This trains:
- SGD-SVM with TF-IDF
- SGD logistic regression with TF-IDF

It records:
- train macro F1 per epoch
- validation macro F1 per epoch
- best epoch
- early stopping point
- test macro F1
- training time

Outputs:
- `ml_pipeline/logs/tfidf_epoch_experiment_results.json`
- `ml_pipeline/data/processed/tfidf_epoch_experiment_results.txt`

## 4. Original Snippet Code 30-Epoch Transformer Experiment

The original snippet code is the `code` column from the processed BuggedPythonLeetCode splits:
- `train.parquet`
- `val.parquet`
- `test.parquet`

The current transformer script already uses this original code snippet input.

Run the 30-epoch experiment with early stopping:

```bash
cd F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code
.\.venv\Scripts\activate
$env:TRANSFORMER_EPOCHS="30"
$env:EARLY_STOPPING_PATIENCE="3"
python ml_pipeline\transformer_classifier.py
```

This trains and compares:
- BERT
- RoBERTa
- CodeBERT

Record in report:
- best epoch
- validation macro F1 curve
- test accuracy
- test macro F1
- per-class F1
- total training time
- whether early stopping happened

## 5. Indentation Error Detection Validation

Run:

```bash
cd F:\Heulwen\IT GREENWICH\Final Project\Tutor_AI_code
.\.venv\Scripts\activate
python ml_pipeline\indentation_self_labeled_eval.py
```

Outputs:
- `ml_pipeline/data/processed/indentation_self_labeled_eval.csv`
- `ml_pipeline/data/processed/indentation_self_labeled_eval_summary.txt`

Report explanation:
- Indentation errors are detected before ML inference.
- The system uses Python `ast.parse`.
- If Python raises `IndentationError`, label is `indentation_error`.
- If Python raises other `SyntaxError`, label is `syntax_error`.
- If parsing succeeds, code goes to ML classifier.

Cases to discuss:
- Missing block after `def`, `if`, `for`, `while`: indentation error.
- Unexpected top-level indent: indentation error.
- Missing colon: syntax error, not indentation error.
- Unclosed bracket or quote: syntax error, not indentation error.
- Correctly indented code: no parse-level error.

## 6. Revised Timeline

### Week 1: Tutor Feedback Experiments

- Run TF-IDF 15-epoch experiment with early stopping.
- Run transformer 30-epoch experiment for BERT, RoBERTa, CodeBERT.
- Record training time and best epoch for each model.
- Run indentation self-labeled evaluation.

### Week 2: Result Analysis

- Update result tables.
- Compare SVM TF-IDF, TF-IDF epoch model, BERT, RoBERTa, CodeBERT.
- Explain why CodeBERT is selected or revise selection if 30-epoch results change.
- Write indentation detection explanation and incorrect case analysis.

### Week 3: Backend And Frontend Integration

- Connect frontend `/analyze` page to FastAPI `POST /analyze`.
- Test model inference with mock mode and real model mode.
- Add clear fallback if model files are missing.

### Week 4: Learning Features

- Build lesson detail pages.
- Save analysis sessions.
- Display progress chart from stored session data.
- Add badge progress rules.

### Week 5: Final Report

- Write methodology, experiment setup, results, discussion, limitations, and ERD.
- Add screenshots from frontend.
- Add model comparison charts.
- Add indentation validation table.

### Week 6: Final Demo Polish

- Prepare scripted demo cases.
- Test backend, frontend, model loading, and mock fallback.
- Clean README and artifact download instructions.
- Final proofread and submission packaging.
