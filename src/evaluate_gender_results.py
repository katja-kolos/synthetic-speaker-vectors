import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
 
import torch
import torch.nn.functional as F
 
# Make the sibling `models` package importable (for GenderClassifier).
_MODULE_PATH = os.path.abspath("../models")
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)
 
from classifiers.gender_classifier import GenderClassifier

# Default values, can be overwritten in main

INPUT_DIM = 128  # size of GST embeddings we use
DEFAULT_EMBEDDINGS_DIR = '../../workspace/synthetic_speaker_embeddings'
DEFAULT_METRICS_DIR = '../../workspace/metrics/metrics'
DEFAULT_CLASSIFIER_PATH = '../../workspace/models/gender_classifier_v4.1a.pt'
 
 
def load_classifier(classifier_path: str, input_dim: int = INPUT_DIM, device: str = 'cpu') -> GenderClassifier:
    model = GenderClassifier(input_dim)
    state_dict = torch.load(classifier_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
 
 
def load_embeddings(embeddings_path: str, device: str = 'cpu') -> torch.Tensor:
    embeddings = torch.load(embeddings_path, map_location=device)
    assert isinstance(embeddings, torch.Tensor), f"Expected a torch.Tensor, got {type(embeddings)}"
    return embeddings
 
 
def classify_gender(embeddings: torch.Tensor, classifier_model: GenderClassifier):
    with torch.no_grad():
        output = classifier_model(embeddings)
    probabilities = F.softmax(output, dim=-1)
    predicted_classes = torch.argmax(probabilities, dim=-1)
    return predicted_classes, probabilities
 
 
def summarize(predicted_classes, probabilities) -> Dict:
    """Turn per-sample predictions into a distribution + mean confidence per class."""
    gender_dist = {'female': 0, 'male': 0}
    confidences = {'female': 0.0, 'male': 0.0}
    for i in range(len(predicted_classes)):
        # label map: {'m': 0, 'f': 1}
        gender = 'female' if predicted_classes[i] == 1 else 'male'
        gender_dist[gender] += 1
        confidences[gender] += probabilities[i, predicted_classes[i]].item()
    for gender in ('female', 'male'):
        if gender_dist[gender] > 0:
            confidences[gender] /= gender_dist[gender]
    return {'distribution': gender_dist, 'confidences': confidences}
 
 
def save_metrics(metrics_dir: str, model_version: str, summary: Dict) -> str:
    metrics_folder = os.path.join(metrics_dir, model_version.lower())
    os.makedirs(metrics_folder, exist_ok=True)
    metrics_file = os.path.join(metrics_folder, 'gender_distribution.txt')
 
    gender_dist = summary['distribution']
    confidences = summary['confidences']
 
    with open(metrics_file, 'w') as f:
        f.write(f"Gender distribution for model {model_version}:\n")
        f.write(f"Number of females: {gender_dist.get('female', 0)}\n")
        f.write(f"Number of males: {gender_dist.get('male', 0)}\n")
        f.write("\nClassifier confidences (mean):\n")
        f.write(f"Female: {confidences['female']:.4f}\n")
        f.write(f"Male: {confidences['male']:.4f}\n")
    return metrics_file
 
 
def process_file(
    embeddings_path: str,
    model_version: str,
    classifier_model: GenderClassifier,
    metrics_dir: str,
    device: str = 'cpu',
) -> Optional[Dict]:
    """Classify one embeddings file and write its metrics. Returns the summary dict."""
    if not os.path.exists(embeddings_path):
        print(f"[skip] {embeddings_path} does not exist")
        return None
    print(f"[process] {embeddings_path} -> {model_version}")
    embeddings = load_embeddings(embeddings_path, device=device)
    predicted_classes, probabilities = classify_gender(embeddings, classifier_model)
    summary = summarize(predicted_classes, probabilities)
    save_metrics(metrics_dir, model_version, summary)
    return summary
 

# script may need to be run on full directory of results or a single file, or a specific setting, e.g. "_v0.6"
@dataclass
class Job:
    kind: str                                  # "single" | "directory" | "conditional"
    path: str                                  # file (single) or dir (directory/conditional)
    model_version: str = ""                    # required for "single" and "conditional"
    suffix: str = ""                           # "conditional" only, e.g. "_0.6"
    file_prefixes: Tuple[str, str] = ("f", "m")  # "conditional" only: female/male prefixes
    skip_entries: Tuple[str, ...] = ("WGAN-public",)  # "directory" only: subfolders to skip
 
 
def run_job(job: Job, classifier_model: GenderClassifier, metrics_dir: str, device: str = 'cpu') -> None:
    if job.kind == "single":
        process_file(job.path, job.model_version, classifier_model, metrics_dir, device)
 
    elif job.kind == "conditional":
        f_prefix, m_prefix = job.file_prefixes
        female_path = os.path.join(job.path, f"{f_prefix}vectors_file{job.suffix}.pt")
        male_path = os.path.join(job.path, f"{m_prefix}vectors_file{job.suffix}.pt")
        process_file(female_path, f"{job.model_version}-f{job.suffix}", classifier_model, metrics_dir, device)
        process_file(male_path, f"{job.model_version}-m{job.suffix}", classifier_model, metrics_dir, device)
 
    elif job.kind == "directory":
        for entry in sorted(os.listdir(job.path)):
            if entry in job.skip_entries:
                continue
            entry_path = os.path.join(job.path, entry)
            embeddings_path = os.path.join(entry_path, 'vectors_file.pt')
            if os.path.isdir(entry_path) and os.path.exists(embeddings_path):
                process_file(embeddings_path, entry, classifier_model, metrics_dir, device)
 
    else:
        raise ValueError(f"Unknown job kind: {job.kind!r}")
 
 
JOBS: List[Job] = [
    Job(
        kind="single",
        path='/home/users1/kolosea/hiwi/thesis/workspace/synthetic_speaker_embeddings/cg-diffusion-v1.5/mvectors_file_0.4.pt',
        model_version='cg-diffusion-v1.5m_0.4',
    ),
 
    # Job(kind="conditional",
    #     path='.../synthetic_speaker_embeddings/cg-diffusion-v1.5',
    #     model_version="cg-diffusion-v1.5", suffix="_0.72"),
]
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Classify gender of synthetic speaker embeddings.")
    parser.add_argument("--classifier-path", default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument("--metrics-dir", default=DEFAULT_METRICS_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
 
    classifier_model = load_classifier(args.classifier_path, device=args.device)
 
    for job in JOBS:
        run_job(job, classifier_model, args.metrics_dir, device=args.device)
 
 
if __name__ == "__main__":
    main()