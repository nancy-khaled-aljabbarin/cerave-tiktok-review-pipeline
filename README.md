````markdown
## CeraVe TikTok Review Pipeline

This project provides a complete pipeline for collecting, reviewing, validating, and organizing TikTok videos related to CeraVe skincare products.

The final dataset contains 50 manually reviewed TikTok videos. Each video contains spoken review content and is labeled with one of three sentiment classes: positive, negative, or neutral.

## Dataset Summary

- Total videos: 50
- Positive: 17
- Negative: 17
- Neutral: 16
- Duplicate URLs: 0
- Videos with spoken reviews: 50

## Final Dataset

The final dataset is stored in:

```text
data/cerave_reviews_final.csv
````

It contains the following columns:

* `video_url`: TikTok video URL
* `video_description`: Original TikTok description when available
* `like_count`: Number of likes on the video
* `sentiment`: Manually assigned sentiment label

## Project Structure

```text
cerave-tiktok-review-pipeline/
├── data/
│   ├── cerave_reviews_final.csv
│   ├── raw_tiktok_videos.csv
│   ├── raw_tiktok_videos_batch1.csv
│   ├── review_ready_batch1.csv
│   ├── review_tiktok_videos_batch1.csv
│   ├── review_tiktok_videos_batch2.csv
│   └── speech_check.csv
│
├── src/
│   └── dataset_pipeline/
│       ├── __init__.py
│       ├── collect_tiktok.py
│       ├── prepare_reviews.py
│       ├── quick_match.py
│       ├── build_final_dataset.py
│       └── check_video_speech.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Pipeline Overview

### 1. Video Collection

`collect_tiktok.py` collects TikTok video information and stores the raw data in CSV format.

### 2. Review Preparation

`prepare_reviews.py` prepares collected videos for manual review and adds the required review columns.

### 3. Manual Review Matching

`quick_match.py` matches manually reviewed videos with the collected dataset and creates a review-ready file.

### 4. Final Dataset Creation

`build_final_dataset.py` combines the reviewed batches, removes duplicate URLs, normalizes sentiment labels, and creates the balanced final dataset.

The target sentiment distribution is:

* 17 positive reviews
* 17 negative reviews
* 16 neutral reviews

### 5. Spoken Review Validation

`check_video_speech.py` verifies that every selected TikTok video contains spoken review content.

The final validation result is:

```text
speech: 50
no_speech: 0
```

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the final dataset builder from the project root:

```bash
python src/dataset_pipeline/build_final_dataset.py
```

Run the spoken review validation:

```bash
python src/dataset_pipeline/check_video_speech.py
```

## Data Validation

The final dataset was checked for:

* Balanced sentiment distribution
* Duplicate TikTok URLs
* Missing required values
* Spoken review content
* Correct manual sentiment labels

Final results:

```text
Total rows: 50
Positive: 17
Negative: 17
Neutral: 16
Duplicate URLs: 0
Spoken reviews: 50
```

## Notes

* Video descriptions were preserved as originally collected.
* Missing descriptions were left empty because the description field is optional.
* Sentiment labels were assigned manually after reviewing each video.
* Sponsored or clearly promotional content was avoided when identified.
* Videos without spoken review content were replaced before creating the final dataset.
* The final dataset contains reviews of different CeraVe products under the same brand.

## Future Work

The next stage of the project will focus on:

* Converting spoken video content into text
* Extracting text, audio, and video features
* Building a sentiment classification model
* Creating a complete video analysis pipeline

```
```
