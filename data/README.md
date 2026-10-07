# NewsGuard Dataset Documentation

## Directory Structure
- `raw/news_dataset.csv`: Complete raw dataset containing 5,200 news samples (REAL and FAKE balanced).
- `processed/train.csv`: Stratified 80% training split (4,160 samples).
- `processed/val.csv`: Stratified 10% validation split (520 samples).
- `processed/test.csv`: Stratified 10% held-out test split (520 samples).

## Dataset Columns
- `id`: Unique article identifier (`real_XXXX` or `fake_XXXX`).
- `title`: Article headline.
- `text`: Main article body text.
- `subject`: Domain category (`politics`, `science`, `economy`, `conspiracy`, `clickbait`, etc.).
- `label`: Binary classification target (`1` = REAL news, `0` = FAKE news).
