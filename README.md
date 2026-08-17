# CeraVe TikTok Review Pipeline

An end-to-end Python pipeline for collecting, reviewing, transcribing, verifying, and analyzing TikTok videos related to CeraVe skincare products.

The project processes a manually reviewed dataset of 50 TikTok videos and combines textual, audio, and visual signals in a resumable multi-stage pipeline. The final sentiment-analysis stage uses a local instruction-tuned LLM, while a dedicated validation stage evaluates the main pipeline outputs.

## Dataset Summary

The final dataset contains:

- Total videos: 50
- Positive reviews: 17
- Negative reviews: 17
- Neutral reviews: 16
- Duplicate video URLs: 0
- Videos containing detected speech: 50

The manually reviewed dataset is stored in:

```text
data/cerave_reviews_final.csv
```

Its main columns include:

- `video_url`: TikTok video URL
- `video_description`: Original TikTok description, when available
- `like_count`: Number of likes
- `sentiment`: Manually assigned sentiment label

## Pipeline Overview

The project is organized as a sequence of connected processing stages.

### 1. TikTok Data Collection

Collects TikTok video metadata and stores the raw results in CSV format.

### 2. Review Preparation and Manual Validation

Prepares the collected videos for manual review and records:

- Whether each video should be retained
- The manually assigned sentiment label
- Review-related validation information

### 3. Final Dataset Creation

Combines the reviewed batches, removes duplicate URLs, normalizes sentiment labels, and creates the final dataset.

The target sentiment distribution is:

- 17 positive
- 17 negative
- 16 neutral

### 4. Video Downloading and Speech-to-Text

Downloads the selected videos, extracts their audio, and converts spoken content into text.

The pipeline preserves previous successful results and skips videos that have already been processed.

### 5. Transcription Verification

Verifies generated transcriptions using a second transcription output and an agreement-based verification process.

Final verification results:

- Total transcriptions: 50
- Verified transcriptions: 50
- Automatically verified: 47
- Manually verified: 3
- Mean agreement score: 86.90%
- Median agreement score: 88.08%
- Minimum agreement score: 56.00%
- Maximum agreement score: 100.00%

The agreement score represents agreement between transcription verification outputs and is **not reported as transcription accuracy**.

### 6. Facial-Expression Analysis

Analyzes representative face frames from each video using a pretrained facial-expression model.

The stage records:

- Predicted facial expression
- Confidence score
- Number of analyzed face frames
- Window agreement
- Confidence margin
- Reliability status

### 7. DeepFace Analysis

Runs a second facial-expression analysis using DeepFace.

The outputs of both facial-analysis approaches are used to derive model-agreement and reliability information.

Facial-expression information is treated as a supporting signal rather than a standalone sentiment label.

### 8. Speech Detection

Uses the Silero Voice Activity Detection model in ONNX format to verify whether human speech is present in each video.

The stage records:

- Audio availability
- Speech presence
- Audio duration
- Estimated speech duration
- Speech ratio
- Maximum speech probability
- Processing status

All 50 videos in the final dataset contained detected speech.

### 9. LLM Sentiment Analysis

Performs sentiment classification using a local instruction-tuned language model:

```text
Qwen2.5-3B-Instruct Q4_K_M (GGUF)
```

The model is executed locally through `llama-cpp-python`.

Each review is classified as exactly one of:

- `positive`
- `neutral`
- `negative`

The transcription is used as the primary sentiment evidence.

Facial-expression information is used only as supporting evidence, with its influence controlled by the facial reliability level.

The selected production configuration is the zero-shot prompt.

Several prompt configurations were evaluated. An experimental V4 prompt improved neutral recall but reduced overall accuracy, so it was not selected.

| Configuration | Overall Accuracy | Positive Recall | Neutral Recall | Negative Recall |
|---|---:|---:|---:|---:|
| Selected zero-shot | 76.00% | 82.35% | 43.75% | 100.00% |
| Experimental V4 | 68.00% | 52.94% | 56.25% | 94.12% |

The selected zero-shot configuration was retained because it provided better overall classification performance.

### 10. Final Pipeline Validation

A dedicated validation module evaluates the main predictive outputs of the pipeline:

- Transcription verification
- Facial-expression prediction
- LLM sentiment classification

The final report is generated automatically and stored in:

```text
data/final_validation_report.txt
```

## Validation Results

### Transcription Verification

- Total transcriptions: 50
- Verified: 50
- Automatically verified: 47
- Manually verified: 3
- Mean agreement: 86.90%
- Median agreement: 88.08%
- Minimum agreement: 56.00%
- Maximum agreement: 100.00%

The agreement score is a verification measure and should not be interpreted as transcription accuracy.

### Facial-Expression Validation

Manual facial-expression labels were compared with the final facial-expression predictions.

- Evaluated videos: 50
- Correct predictions: 18
- Incorrect predictions: 32
- Exact-match accuracy: 36.00%

Accuracy by model reliability:

- High reliability: 7/11 — 63.64%
- Medium reliability: 6/14 — 42.86%
- Low reliability: 3/23 — 13.04%
- Not applicable: 2/2 — 100.00%

These results show that facial-expression predictions are substantially more reliable in the high-reliability subset.

For this reason, facial information is treated as supporting evidence in downstream sentiment analysis, particularly when reliability is low.

### LLM Sentiment Validation

The selected zero-shot configuration was evaluated against the manually assigned sentiment labels.

- Evaluated reviews: 50
- Correct predictions: 38
- Incorrect predictions: 12
- Overall accuracy: 76.00%

Per-class results:

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Positive | 0.8750 | 0.8235 | 0.8485 | 17 |
| Neutral | 0.7778 | 0.4375 | 0.5600 | 16 |
| Negative | 0.6800 | 1.0000 | 0.8095 | 17 |

The neutral class remains the most challenging class and represents an important area for future improvement.

## Main Output Files

```text
data/cerave_reviews_final.csv
data/cerave_reviews_enriched.csv
data/cerave_reviews_with_facial_expression.csv
data/cerave_reviews_with_two_models.csv
data/speech_check.csv
data/transcription_verification.csv
data/cerave_reviews_with_llm_zero_shot.csv
data/final_validation_report.txt
data/llm_error_analysis.csv
data/llm_prompt_comparison.txt
```

## Project Structure

```text
cerave-tiktok-review-pipeline/
├── data/
│   ├── archive/
│   ├── audios/
│   ├── videos/
│   ├── cerave_reviews_final.csv
│   ├── cerave_reviews_enriched.csv
│   ├── cerave_reviews_with_facial_expression.csv
│   ├── cerave_reviews_with_two_models.csv
│   ├── cerave_reviews_with_llm_zero_shot.csv
│   ├── speech_check.csv
│   ├── transcription_verification.csv
│   ├── final_validation_report.txt
│   ├── llm_error_analysis.csv
│   └── llm_prompt_comparison.txt
│
├── models/
│   ├── facial_expression/
│   ├── llm/
│   │   └── qwen2.5-3b-instruct-q4_k_m.gguf
│   └── silero_vad.onnx
│
├── src/
│   ├── data_collection/
│   ├── facial_expression/
│   ├── speech_to_text/
│   ├── llm_analysis/
│   │   ├── __init__.py
│   │   ├── analyze_facial_effect.py
│   │   ├── evaluate_llm.py
│   │   ├── export_llm_errors.py
│   │   ├── llm_runner.py
│   │   ├── process_reviews.py
│   │   └── prompt_builder.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   └── final_validation.py
│   │
│   └── speech_detector.py
│
├── tests/
├── .env.example
├── .gitignore
├── config.py
├── main.py
├── requirements.txt
├── requirements-facial.txt
└── README.md
```

## Configuration

The project uses configuration files according to component responsibility:

- `config.py`: shared paths and pipeline environment settings
- `src/speech_to_text/config.py`: speech-to-text configuration
- `src/facial_expression/config.py`: facial-expression configuration

Create a `.env` file based on `.env.example`.

Example:

```env
MODEL_NAME=tiny.en
VERIFICATION_MODEL_NAME=medium
DEVICE=cpu
COMPUTE_TYPE=int8
TEST_LIMIT=50

FACIAL_ENV_NAME=facial-expression-clean
LLM_ENV_NAME=llm-analysis
```

If required on the local machine, `CONDA_EXE` can also be configured with the path to the local Conda executable.

The real `.env` file is ignored by Git and should not be committed.

## Installation

The pipeline uses separate environments to isolate specialized dependencies.

### 1. Main Environment

Create and activate the main virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the main requirements:

```powershell
pip install -r requirements.txt
```

### 2. Facial-Analysis Environment

Create the Conda environment:

```powershell
conda create -n facial-expression-clean python=3.10 -y
```

Install the facial-analysis dependencies:

```powershell
conda run -n facial-expression-clean python -m pip install -r requirements-facial.txt
```

The project uses CPU versions of PyTorch and Torchvision:

```powershell
conda run -n facial-expression-clean python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

### 3. LLM Environment

The LLM stage runs in a separate Conda environment named:

```text
llm-analysis
```

The environment requires `llama-cpp-python`, which provides the `llama_cpp.Llama` interface used by the project.

The current implementation expects the local model at:

```text
models/llm/qwen2.5-3b-instruct-q4_k_m.gguf
```

The model file is excluded from Git because of its size and must be provided locally before running the LLM stage.

## Usage

Run the complete pipeline from the project root:

```powershell
python main.py
```

`main.py` is the main entry point and coordinates the main processing stages.

The complete workflow is:

```text
Data collection and manual review
→ Final dataset creation
→ Video downloading and transcription
→ Transcription verification
→ Facial-expression analysis
→ DeepFace analysis
→ Speech detection
→ LLM sentiment analysis
→ Final validation
```

The validation module can also be executed independently:

```powershell
python -m src.validation.final_validation
```

## Resumable Processing

The pipeline is designed to preserve completed work.

When `main.py` is executed again, it checks existing outputs and processing statuses before running each stage.

Completed records are skipped, while incomplete or pending records continue from saved progress.

Example messages include:

```text
Final dataset is ready. Skipping data collection.
Previous results were found. Continuing from the saved progress.
All facial-expression results are already completed.
Videos selected: 0
All zero_shot reviews are already classified.
All pipeline stages are completed.
```

This avoids unnecessary model execution and reduces the risk of losing previously completed work.

## Data and Processing Notes

- Sentiment labels were manually assigned after reviewing the videos.
- Duplicate TikTok URLs were removed.
- Missing descriptions were retained as empty values because descriptions are optional.
- Clearly promotional or sponsored content was excluded when identified.
- Intermediate outputs are saved to preserve processing progress.
- Video, audio, facial-model, and LLM-model files excluded by `.gitignore` are not committed.
- Facial-expression predictions are treated as supporting evidence because validation showed substantial performance differences across reliability levels.
- Prompt variants were compared empirically rather than selecting a prompt only through qualitative judgment.
- The selected LLM configuration was chosen based on validation against the manually assigned sentiment labels.

## Technologies Used

- Python
- Pandas
- Playwright
- yt-dlp
- FFmpeg
- Faster Whisper
- PyTorch
- TensorFlow
- MediaPipe
- DeepFace
- OpenCV
- ONNX Runtime
- Silero VAD
- llama.cpp / llama-cpp-python
- Qwen2.5-3B-Instruct

## Limitations

- The dataset contains only 50 manually reviewed TikTok videos, so evaluation results should be interpreted within this dataset rather than as general performance estimates.
- Facial-expression classification achieved 36% exact-match accuracy overall and should not be treated as a standalone sentiment classifier.
- Neutral sentiment was the most difficult class for the selected LLM configuration.
- Facial expressions may not always correspond directly to the reviewer's opinion about a product.
- Prompt refinements were evaluated on the same labeled dataset, so additional held-out data would be required for stronger generalization claims.
- The local LLM model file is not stored in the repository because of its size.

## Future Work

Possible future improvements include:

- Expanding the manually reviewed dataset
- Creating a separate held-out evaluation set
- Improving neutral-sentiment classification
- Improving facial-expression reliability
- Comparing text-only and multimodal sentiment configurations on larger datasets
- Evaluating speech detection against manually verified speech labels
- Adding broader automated test coverage
- Developing a dedicated multimodal sentiment-fusion model
- Providing an interactive dashboard for exploring pipeline results

## Author

**Nancy Khaled Al-Jabbarin**

Computer Science Student — Artificial Intelligence Track  
An-Najah National University
