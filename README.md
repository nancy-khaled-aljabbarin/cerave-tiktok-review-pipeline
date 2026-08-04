# CeraVe TikTok Review Pipeline

An end-to-end Python pipeline for collecting, reviewing, transcribing, validating, and analyzing TikTok videos related to CeraVe skincare products.

The project processes a final dataset of 50 manually reviewed TikTok videos and combines textual, audio, and visual analysis within one resumable pipeline.

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

- Whether the video should be retained
- The final sentiment label
- Review-related validation information

### 3. Final Dataset Creation

Combines the reviewed batches, removes duplicate URLs, normalizes sentiment labels, and creates the final balanced dataset.

The target sentiment distribution is:

- 17 positive
- 17 negative
- 16 neutral

### 4. Video Downloading and Speech-to-Text

Downloads the selected videos, extracts their audio, and converts spoken content into text.

The pipeline preserves previous transcription results and skips videos that have already been processed successfully.

### 5. Facial Expression Analysis

Analyzes representative face frames from each video using a pretrained facial-expression model.

The analysis records information such as:

- Predicted facial expression
- Confidence score
- Number of analyzed face frames
- Window agreement
- Reliability status

### 6. DeepFace Analysis

Runs a second facial-expression analysis using DeepFace and stores its predictions alongside the first model’s results.

This provides an additional model output for comparison and validation.

### 7. Speech Detection

Uses the Silero Voice Activity Detection model in ONNX format to verify whether human speech is present in each video.

The stage records:

- Audio availability
- Speech presence
- Audio duration
- Estimated speech duration
- Speech ratio
- Maximum speech probability
- Processing status

## Main Output Files

```text
data/cerave_reviews_final.csv
data/cerave_reviews_enriched.csv
data/cerave_reviews_with_facial_expression.csv
data/cerave_reviews_with_two_models.csv
data/speech_check.csv
data/transcription_verification.csv
data/facial_expression_validation_report.txt
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
│   ├── speech_check.csv
│   └── transcription_verification.csv
│
├── models/
│   ├── facial_expression/
│   └── silero_vad.onnx
│
├── src/
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── build_final_dataset.py
│   │   ├── check_video_speech.py
│   │   ├── collect_tiktok.py
│   │   ├── prepare_reviews.py
│   │   └── quick_match.py
│   │
│   ├── facial_expression/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── deepface_analyzer.py
│   │   ├── model_loader.py
│   │   ├── process_deepface.py
│   │   ├── process_facial_expressions.py
│   │   └── video_analyzer.py
│   │
│   ├── speech_to_text/
│   │   ├── __init__.py
│   │   ├── audio_extractor.py
│   │   ├── config.py
│   │   ├── feature_extractor.py
│   │   ├── process_final_dataset.py
│   │   ├── text_preprocessor.py
│   │   ├── transcriber.py
│   │   ├── validate_results.py
│   │   ├── verify_transcriptions.py
│   │   └── video_downloader.py
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

The project uses separate configuration files based on responsibility:

- `config.py`: Shared project paths and pipeline environment settings
- `src/speech_to_text/config.py`: Speech-to-text settings
- `src/facial_expression/config.py`: Facial-expression model settings

Create a `.env` file from `.env.example` and provide the required values.

Example:

```env
MODEL_NAME=tiny.en
VERIFICATION_MODEL_NAME=medium
DEVICE=cpu
COMPUTE_TYPE=int8
TEST_LIMIT=50
CONDA_EXE=C:\path\to\anaconda3\Scripts\conda.exe
FACIAL_ENV_NAME=facial-expression-clean
```

The real `.env` file is ignored by Git and should not be committed.

## Installation

The project uses two environments because the facial-analysis dependencies require a separate compatible environment.

### Main Environment

Create and activate the main virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the main requirements:

```powershell
pip install -r requirements.txt
```

### Facial-Analysis Environment

Create the Conda environment:

```powershell
conda create -n facial-expression-clean python=3.10 -y
```

Install its dependencies:

```powershell
conda run -n facial-expression-clean python -m pip install -r requirements-facial.txt
```

The project uses the CPU versions of PyTorch and Torchvision. They can be installed using:

```powershell
conda run -n facial-expression-clean python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

## Usage

Run the complete pipeline from the project root:

```powershell
python main.py
```

`main.py` is the main entry point for the project.

It coordinates the processing stages in this order:

```text
Data collection and review
→ Final dataset creation
→ Video downloading and transcription
→ Facial-expression analysis
→ DeepFace analysis
→ Speech detection
```

## Resumable Processing

The pipeline is designed to preserve completed work.

When `main.py` is executed again, it checks the existing files and processing statuses before running each stage.

Completed steps are skipped, while incomplete or pending records continue from the saved progress.

Examples include:

```text
Final dataset is ready. Skipping data collection.
Previous results were found. Continuing from the saved progress.
All facial-expression results are already completed.
Videos selected: 0
```

This prevents unnecessary model execution and protects previously generated results.

## Validation Results

The latest complete pipeline run produced:

- Transcription successful: 50
- Transcription failed: 0
- DeepFace successful: 48
- DeepFace no visible face: 2
- Speech detection successful: 50
- Videos with detected speech: 50

Detailed facial-expression validation results are stored in:

```text
data/facial_expression_validation_report.txt
```

## Data and Processing Notes

- Sentiment labels were assigned manually after reviewing the videos.
- Duplicate TikTok URLs were removed.
- Missing descriptions were retained as empty values because descriptions are optional.
- Clearly promotional or sponsored content was excluded when identified.
- Video, audio, and model files excluded by `.gitignore` are not committed to the repository.
- Intermediate results are saved regularly to reduce the risk of losing completed work.
- The project separates data collection, speech processing, and facial analysis according to the Single Responsibility Principle.

## Technologies Used

- Python
- Pandas
- NumPy
- FFmpeg
- Faster Whisper
- PyTorch
- TensorFlow
- MediaPipe
- DeepFace
- ONNX Runtime
- Silero VAD
- OpenCV
- Hugging Face models

## Future Work

Possible future improvements include:

- Comparing textual, acoustic, and visual sentiment signals
- Developing a multimodal sentiment-classification model
- Improving facial-expression reliability
- Evaluating speech detection against manually verified labels
- Adding automated tests for all pipeline stages
- Providing an interactive dashboard for exploring the results

## Author

**Nancy Khaled Al-Jabbarin**

Computer Science student — Artificial Intelligence Track  
An-Najah National University
