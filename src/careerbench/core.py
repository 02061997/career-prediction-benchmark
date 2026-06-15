from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


DOMAIN_LABELS = (
    "internet_publishing",
    "retail",
    "education",
    "marketing",
    "business_consulting",
    "finance",
)
POSITION_LABELS = (
    "recruiter",
    "manager",
    "software_engineer",
    "supervisor",
    "director",
    "analyst",
    "project_manager",
    "student",
)
FEATURE_NAMES = (
    "marketing",
    "arts",
    "computer_science",
    "engineering",
    "business",
    "finance",
    "education",
    "law",
    "healthcare",
    "publication_0",
    "publications_1_10",
    "publications_10_plus",
    "experience_0_2",
    "experience_3_5",
    "experience_6_10",
    "experience_10_plus",
    "position_new",
    "position_experienced",
    "position_expert",
    "management_history",
    "technical_history",
    "research_history",
    "customer_facing",
    "graduate_degree",
    "leadership",
    "career_change",
)

SENSITIVE_ATTRIBUTE_NAMES = (
    "name",
    "email",
    "phone",
    "address",
    "age",
    "gender",
    "race",
    "ethnicity",
    "nationality",
)

PRIVACY_FAIRNESS_LIMITATIONS = (
    "The executable fixture is synthetic and contains no direct identifiers.",
    "The private LinkedIn-derived records are not redistributed.",
    "No demographic sensitive attributes are modeled, so group fairness metrics are not claimed.",
    "Career prediction can reinforce labor-market bias and should not be used for automated hiring decisions.",
)


@dataclass(frozen=True)
class CareerDataset:
    features: pd.DataFrame
    targets: pd.DataFrame


def binary_entropy(values: np.ndarray) -> float:
    probability = np.bincount(values.astype(int), minlength=2) / len(values)
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def joint_entropy(left: np.ndarray, right: np.ndarray) -> float:
    pairs = np.column_stack([left, right])
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    probability = counts / counts.sum()
    return float(-(probability * np.log2(probability)).sum())


def generate_reference_dataset(samples: int = 420, seed: int = 7) -> CareerDataset:
    """Generate a privacy-safe fixture with the paper's dimensions and target cardinalities."""
    rng = np.random.default_rng(seed)
    latent_domain = rng.integers(0, len(DOMAIN_LABELS), samples)
    latent_seniority = rng.integers(0, 4, samples)
    latent_technical = rng.binomial(1, 0.48, samples)
    latent_research = rng.binomial(1, 0.30, samples)

    probabilities = rng.uniform(0.08, 0.40, (len(DOMAIN_LABELS), len(FEATURE_NAMES)))
    for domain in range(len(DOMAIN_LABELS)):
        probabilities[domain, domain] = 0.88
        probabilities[domain, 9 + (domain % 3)] = 0.65
        probabilities[domain, 20 + (domain % 4)] = 0.72
    features = rng.binomial(1, probabilities[latent_domain])
    features[:, 12:16] = 0
    features[np.arange(samples), 12 + latent_seniority] = 1
    features[:, 17] = (latent_seniority >= 1).astype(int)
    features[:, 18] = (latent_seniority >= 3).astype(int)
    features[:, 20] = latent_technical
    features[:, 21] = latent_research
    features[:, 23] = ((latent_research == 1) | (latent_seniority >= 2)).astype(int)
    features[:, 24] = (latent_seniority >= 2).astype(int)

    domain_noise = rng.random(samples) < 0.08
    domains = latent_domain.copy()
    domains[domain_noise] = rng.integers(0, len(DOMAIN_LABELS), domain_noise.sum())

    positions = np.select(
        [
            (latent_technical == 1) & (latent_seniority <= 1),
            (latent_technical == 1) & (latent_seniority == 2),
            latent_seniority == 3,
            (latent_domain == 1) & (latent_seniority >= 1),
            (latent_domain == 4) & (latent_seniority >= 1),
            latent_research == 1,
            latent_seniority == 0,
        ],
        [2, 6, 4, 3, 1, 5, 7],
        default=0,
    )
    position_noise = rng.random(samples) < 0.06
    positions[position_noise] = rng.integers(0, len(POSITION_LABELS), position_noise.sum())

    return CareerDataset(
        pd.DataFrame(features, columns=FEATURE_NAMES),
        pd.DataFrame({"domain": domains, "position": positions}),
    )


def privacy_audit(data: CareerDataset) -> dict[str, object]:
    feature_names = {name.lower() for name in data.features.columns}
    target_names = {name.lower() for name in data.targets.columns}
    forbidden = set(SENSITIVE_ATTRIBUTE_NAMES)
    return {
        "direct_identifier_columns_present": bool(feature_names & forbidden),
        "sensitive_attribute_columns_present": bool((feature_names | target_names) & forbidden),
        "binary_feature_matrix": bool(data.features.isin([0, 1]).all().all()),
        "synthetic_fixture": True,
        "private_records_redistributed": False,
        "fairness_metrics_claimed": False,
        "limitations": list(PRIVACY_FAIRNESS_LIMITATIONS),
    }


def select_features(features: pd.DataFrame, targets: pd.DataFrame, count: int = 11) -> list[str]:
    """Paper-aligned entropy screening with target relevance and redundancy control."""
    scores: dict[str, float] = {}
    selected: list[str] = []
    for name in features:
        values = features[name].to_numpy()
        relevance = sum(mutual_info_score(values, targets[target]) for target in targets)
        scores[name] = binary_entropy(values) + relevance
    while len(selected) < count:
        candidates = []
        for name in features.columns.difference(selected, sort=False):
            redundancy = 0.0
            if selected:
                redundancy = max(
                    mutual_info_score(features[name], features[chosen]) for chosen in selected
                )
            candidates.append((scores[name] - 0.35 * redundancy, name))
        selected.append(max(candidates)[1])
    return selected


def classifiers(seed: int = 7) -> dict[str, MultiOutputClassifier]:
    return {
        "random_forest": MultiOutputClassifier(
            RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1)
        ),
        "decision_tree": MultiOutputClassifier(DecisionTreeClassifier(random_state=seed)),
        "extra_tree": MultiOutputClassifier(ExtraTreeClassifier(random_state=seed)),
        "extra_trees": MultiOutputClassifier(
            ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
        ),
        "k_neighbors": MultiOutputClassifier(KNeighborsClassifier(n_neighbors=1)),
        "radius_neighbors": MultiOutputClassifier(
            RadiusNeighborsClassifier(radius=2.0, outlier_label="most_frequent")
        ),
    }


def evaluate(seed: int = 7) -> tuple[list[dict], pd.DataFrame, list[str]]:
    data = generate_reference_dataset(seed=seed)
    selected = select_features(data.features, data.targets)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.features))
    train, test = order[:300], order[300:]
    rows: list[dict] = []
    predictions = data.targets.iloc[test].reset_index(drop=True).rename(
        columns={"domain": "true_domain", "position": "true_position"}
    )
    for name, model in classifiers(seed).items():
        model.fit(data.features.iloc[train][selected], data.targets.iloc[train])
        predicted = model.predict(data.features.iloc[test][selected])
        rows.append(
            {
                "model": name,
                "domain_accuracy": accuracy_score(data.targets.iloc[test, 0], predicted[:, 0]),
                "position_accuracy": accuracy_score(
                    data.targets.iloc[test, 1], predicted[:, 1]
                ),
                "joint_accuracy": np.mean(
                    np.all(predicted == data.targets.iloc[test].to_numpy(), axis=1)
                ),
            }
        )
        predictions[f"{name}_domain"] = predicted[:, 0]
        predictions[f"{name}_position"] = predicted[:, 1]
    return rows, predictions, selected


PUBLISHED_RESULTS = {
    "random_forest": {"domain_accuracy": 0.9121, "position_accuracy": 0.9597},
    "decision_tree": {"domain_accuracy": 0.9060, "position_accuracy": 0.9323},
    "extra_tree": {"domain_accuracy": 0.8965, "position_accuracy": 0.9050},
    "extra_trees": {"domain_accuracy": 0.8587, "position_accuracy": 0.9070},
    "k_neighbors": {"domain_accuracy": 0.9028, "position_accuracy": 0.9262},
    "radius_neighbors": {"domain_accuracy": 0.6065, "position_accuracy": 0.4567},
}
