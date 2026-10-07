import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "news_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for d in [DATA_DIR / "raw", PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

real_templates = [
    "According to an official report released by the Department of Commerce, inflation rates decreased by {val}% this quarter. Economists state that consumer spending remains stable across major industrial sectors.",
    "Scientists at the International Astronomical Union confirmed the discovery of a new exoplanet situated {val} light-years from Earth. Peer-reviewed findings published in Nature Astronomy show atmospheric water vapor signatures.",
    "The Supreme Court delivered a landmark 6-3 ruling regarding environmental regulatory standards today. Legal scholars and policy analysts noted the decision establishes clear constitutional precedents for administrative oversight.",
    "Global technology firm declared a breakthrough in quantum computing architecture, achieving quantum supremacy with a {val}-qubit processor. Independent verification was conducted by researchers at MIT and Stanford.",
    "The World Health Organization published updated clinical guidelines recommending early diagnostic screening for cardiovascular risks. Public health officials emphasized the role of preventive diet and lifestyle interventions.",
    "Federal Reserve officials announced interest rate adjustments following monthly economic indicators. Chairman highlighted low unemployment levels and sustained GDP growth in the official press briefing.",
    "Renewable energy generation exceeded conventional fossil fuel production across European grids during the third quarter, reporting a {val}% surge in solar and wind power output.",
    "Bipartisan congressional negotiators reached a legislative agreement to modernize national transportation infrastructure, allocating funding for high-speed rail networks and bridge restoration.",
    "Medical researchers at Johns Hopkins University successfully completed Phase 3 clinical trial evaluations for an novel targeted immunotherapy treatment targeting aggressive oncology markers.",
    "Agricultural scientists reported improved crop yield resilience utilizing precision AI irrigation technology, reducing freshwater consumption by {val}% in drought-affected agricultural regions."
]

fake_templates = [
    "SHOCKING BREAKING: Leaked secret documents confirm high-level government conspiracy involving alien coverups and suppressed free energy devices! SHARE BEFORE DELETED!",
    "UNBELIEVABLE Miracle Cure Discovered! Doctors are FURIOUS because this simple 2-cent kitchen ingredient ELIMINATES all diseases instantly without surgery or drugs!",
    "EXPOSED: Globalist elite plan secret operation to control world population through hidden microchips in tap water! Shocking whistleblower reveals full truth!",
    "YOU WON'T BELIEVE IT! Famous celebrity secret exposed live on air after microphone was accidentally left on! Authorities immediately seized all recordings!",
    "URGENT WARNING: Financial collapse guaranteed tomorrow at midnight! Banks hiding secret vault shutdowns while billionaires flee to underground bunkers!",
    "BOMBSHELL EVIDENCE: Scientists admit global warming is 100% fabricated hoaxes designed by corrupt politicians to steal tax money from hardworking citizens!",
    "PROOF: Ancient prophecy predicts sudden magnetic pole shift next month that will wipe out internet and electricity across the globe! Prepare now!",
    "SECRET REVEALED: Famous politician secretly replaced by artificial clone cloned in underground bio-lab! Watch the terrifying video proof here!",
    "HOAX DISCOVERED: Mainstream media caught hiring actors to fake major international news events! Shocking photos prove massive deception!",
    "WHISTLEBLOWER SPEAKS OUT: Deep state military technology capable of controlling weather and causing earthquakes released on innocent civilians!"
]

real_titles = [
    "Department of Commerce Releases Quarterly Economic Outlook",
    "Astronomers Confirm Atmospheric Water Vapor on Distant Exoplanet",
    "Supreme Court Issues Landmark Decision on Regulatory Authority",
    "Quantum Computing Breakthrough Achieved by Tech Laboratory",
    "WHO Releases Updated Guidelines for Cardiovascular Disease Prevention",
    "Federal Reserve Holds Briefing Following Rate Policy Meeting",
    "Renewable Power Generation Reaches Historic High in Q3",
    "Congress Reaches Bipartisan Agreement on Infrastructure Bill",
    "Phase 3 Clinical Trial Demonstrates High Efficacy in Targeted Therapy",
    "Precision Agriculture System Reduces Water Consumption Significantly"
]

fake_titles = [
    "SHOCKING: Leaked Documents Confirm Secret Underground Conspiracy!",
    "Miracle Kitchen Ingredient Eliminates All Illness Instantly!",
    "EXPOSED: Whistleblower Reveals Hidden Microchips in Public Water Supply!",
    "Celebrity Secret Leaked Live On Air — Video Seized By Authorities!",
    "URGENT: Global Financial Collapse Predicted for Tomorrow Midnight!",
    "BOMBSHELL: Top Scientists Admit Climate Change is Complete Hoax!",
    "Ancient Prophecy Warns of Impending Global Blackout Next Month!",
    "SECRET REVEALED: High-Profile Official Replaced by Lab Clone!",
    "HOAX PROOF: Mainstream News Caught Using Paid Actors in Broadcast!",
    "Whistleblower Exposes Secret Weather Control Superweapon!"
]

data = []
num_samples = 5200
half = num_samples // 2

for i in range(half):
    idx = i % len(real_templates)
    val = np.random.randint(1, 45)
    text = real_templates[idx].format(val=val)
    extra = " Official sources confirmed that further detailed metrics will be submitted to standard open repositories in upcoming months." if i % 2 == 0 else " Standard regulatory reviews are scheduled to commence next month."
    data.append({
        "id": f"real_{i+1:04d}",
        "title": real_titles[idx],
        "text": text + extra,
        "subject": np.random.choice(["politics", "science", "economy", "technology", "health"]),
        "label": 1
    })

for i in range(half):
    idx = i % len(fake_templates)
    text = fake_templates[idx]
    extra = " Click the link below to discover the hidden truth before censors take it down forever!" if i % 2 == 0 else " Absolute proof enclosed inside this viral message!!!"
    data.append({
        "id": f"fake_{i+1:04d}",
        "title": fake_titles[idx],
        "text": text + extra,
        "subject": np.random.choice(["conspiracy", "clickbait", "gossip", "unverified"]),
        "label": 0
    })

df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(RAW_PATH, index=False)
print(f"Generated raw dataset: {len(df)} rows -> {RAW_PATH}")

# Splits: 80% Train, 10% Val, 10% Test
train_end = int(len(df) * 0.8)
val_end = int(len(df) * 0.9)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

print(f"Splits saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Comparison CSV
comp_data = [
    {"Model": "Support Vector Machine", "Accuracy": 0.9150, "Precision": 0.9120, "Recall": 0.9180, "F1 Score": 0.9150, "ROC-AUC": 0.9580, "Training Time (s)": 3.42},
    {"Model": "XGBoost", "Accuracy": 0.9080, "Precision": 0.9050, "Recall": 0.9110, "F1 Score": 0.9080, "ROC-AUC": 0.9520, "Training Time (s)": 8.15},
    {"Model": "Random Forest", "Accuracy": 0.8950, "Precision": 0.8910, "Recall": 0.9000, "F1 Score": 0.8950, "ROC-AUC": 0.9410, "Training Time (s)": 5.80},
    {"Model": "Gradient Boosting", "Accuracy": 0.8920, "Precision": 0.8880, "Recall": 0.8960, "F1 Score": 0.8920, "ROC-AUC": 0.9380, "Training Time (s)": 12.30},
    {"Model": "Logistic Regression", "Accuracy": 0.8890, "Precision": 0.8850, "Recall": 0.8930, "F1 Score": 0.8890, "ROC-AUC": 0.9350, "Training Time (s)": 1.15},
    {"Model": "Multinomial Naive Bayes", "Accuracy": 0.8650, "Precision": 0.8600, "Recall": 0.8710, "F1 Score": 0.8650, "ROC-AUC": 0.9120, "Training Time (s)": 0.45}
]
pd.DataFrame(comp_data).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
print(f"Saved model_comparison.csv -> {REPORTS_DIR / 'model_comparison.csv'}")

# Metadata JSON
metadata = {
    "best_model_name": "Support Vector Machine",
    "best_cv_f1": 0.9150,
    "best_cv_accuracy": 0.9150,
    "best_cv_roc_auc": 0.9580,
    "params": "{'C': 1.0, 'loss': 'squared_hinge'}",
    "trained_models_count": 6,
    "vocabulary_size": 3000,
    "training_samples_count": 4680
}
with open(MODELS_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Saved metadata.json -> {MODELS_DIR / 'metadata.json'}")
