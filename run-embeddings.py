#!/usr/bin/env python3
"""Generate text, image, or multimodal embeddings for AllSides articles.

The script reads one or more AllSides JSONL datasets, matches each article to a
locally extracted full article, creates embeddings with a SentenceTransformer
model, and saves resumable pickle checkpoints.

Important comparability rule
----------------------------
Every selected article must have at least one usable local image, even when the
chosen input condition is text-only. Images are sent to the model only when the
input condition includes ``image``; the image requirement simply keeps the
article sample comparable across experimental conditions.

Expected directory layout
-------------------------
<data-dir>/
├── allsides_Jan-May_2025.jsonl
├── allsides_Jun2025_May2026.jsonl
└── full_articles/
    ├── apnews.com.json
    ├── reuters.com.json
    └── ...

Examples
--------
Run the default multimodal condition using the two default datasets::

    python run-embeddings-user-friendly.py --data-dir ./data

Run two conditions and store checkpoints in a separate directory::

    python run-embeddings-user-friendly.py \
        --data-dir ./data \
        --output-dir ./results \
        --input-condition headline-text-image headline-text

Validate files and article coverage without loading the embedding model::

    python run-embeddings-user-friendly.py --data-dir ./data --validate-only

The pickle output keeps the original nested structure:
``results[headline][leaning_group][article_url] = embedding``.
Existing checkpoints are resumed unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm


LOGGER = logging.getLogger("run_embeddings")

DEFAULT_MODEL_NAME = "nvidia/llama-nemotron-embed-vl-1b-v2"
DEFAULT_DATASET_FILES = (
    "allsides_Jan-May_2025.jsonl",
    "allsides_Jun2025_May2026.jsonl",
)
DEFAULT_INPUT_CONDITIONS = ("headline-text-image",)

# The values document the expected source file, while membership in this mapping
# determines whether a source is part of the supported, comparable sample.
SOURCE_FILE_BY_NAME = {
    "Associated Press": "apnews.com.json",
    "Fox Business": "foxbusiness.com.json",
    "Fox News (Opinion)": "foxnews.com.json",
    "Fox News Digital": "foxnews.com.json",
    "NBC Los Angeles": "nbcnews.com.json",
    "NBC News Digital": "nbcnews.com.json",
    "Newsweek": "newsweek.com.json",
    "Newsweek Fact Check": "newsweek.com.json",
    "New York Post (News)": "nypost.com.json",
    "New York Post (Opinion)": "nypost.com.json",
    "New York Times (News)": "nytimes.com.json",
    "New York Times (Opinion)": "nytimes.com.json",
    "Politico": "politico.com.json",
    "Reuters": "reuters.com.json",
    "The Guardian": "theguardian.com.json",
    "The Hill": "thehill.com.json",
    "Washington Examiner": "washingtonexaminer.com.json",
    "Washington Post": "washingtonpost.com.json",
    "Washington Post Fact Check": "washingtonpost.com.json",
}

# These names are intentionally hyphenated because membership checks such as
# ``"headline" in input_condition`` are used when constructing model inputs.
INPUT_CONDITION_DESCRIPTIONS = {
    "headline": "headline only",
    "headline-text": "headline and full article text",
    "image": "article images only",
    "summary": "dataset summary only",
    "headline-summary": "headline and dataset summary",
    "headline-image": "headline and article images",
    "headline-text-image": "headline, full article text, and article images",
    "headline-summary-image": "headline, dataset summary, and article images",
}

# Each event can have a primary article and zero or more alternative articles
# for the three broad leaning groups.
LEANING_FIELDS = (
    ("left", "more_left"),
    ("center", "more_center"),
    ("right", "more_right"),
)

Results = dict[str, dict[str, dict[str, np.ndarray]]]


@dataclass
class ArticleResolutionStats:
    """Counters explaining why candidate articles were accepted or rejected."""

    accepted: int = 0
    unsupported_source: int = 0
    missing_url: int = 0
    missing_full_article: int = 0
    missing_image_metadata: int = 0
    missing_image_file: int = 0
    unreadable_image: int = 0

    @property
    def rejected(self) -> int:
        return (
            self.unsupported_source
            + self.missing_url
            + self.missing_full_article
            + self.missing_image_metadata
            + self.missing_image_file
            + self.unreadable_image
        )


@dataclass
class RunStats:
    """High-level progress counters for one input condition."""

    headlines_seen: int = 0
    headlines_changed: int = 0
    articles_embedded: int = 0
    articles_already_present: int = 0
    checkpoints_written: int = 0


def configure_logging(level: str) -> None:
    """Configure concise console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(levelname)s: %(message)s",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments without importing the large model package."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate resumable text/image embeddings for locally stored "
            "AllSides articles. Every eligible article must have a usable image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "--dir_path",
        dest="data_dir",
        type=Path,
        default=Path("."),
        help="Directory containing the dataset file(s) and full_articles/.",
    )
    parser.add_argument(
        "--filename",
        type=Path,
        default=None,
        help=(
            "Use one JSONL dataset instead of the two default files. Relative "
            "paths are resolved inside --data-dir."
        ),
    )
    parser.add_argument(
        "--full-articles-dir",
        type=Path,
        default=None,
        help="Directory containing extracted full-article JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for embedding pickle files; defaults to --data-dir.",
    )
    parser.add_argument(
        "--input-condition",
        nargs="+",
        choices=sorted(INPUT_CONDITION_DESCRIPTIONS),
        default=list(DEFAULT_INPUT_CONDITIONS),
        help="One or more input conditions to embed.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name or local model path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Model device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Batch size passed to the embedding model.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=positive_int,
        default=1,
        help="Write a checkpoint after this many changed headlines.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Process only the first N dataset rows, useful for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing checkpoint and start this condition again.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files and article/image coverage without loading the model.",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List supported input conditions and sources, then exit.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging verbosity.",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    """Argparse validator for strictly positive integer options."""

    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}.") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return number


def print_supported_options() -> None:
    """Print available experimental conditions and supported article sources."""

    print("Input conditions:")
    for name, description in INPUT_CONDITION_DESCRIPTIONS.items():
        print(f"  {name:<24} {description}")

    print("\nSupported sources:")
    for source in sorted(SOURCE_FILE_BY_NAME):
        print(f"  {source}")


def resolve_user_path(path: Path, base_dir: Path) -> Path:
    """Resolve a relative user path against ``base_dir``."""

    return path.expanduser().resolve() if path.is_absolute() else (base_dir / path).resolve()


def resolve_dataset_paths(data_dir: Path, filename: Path | None) -> list[Path]:
    """Resolve and validate the JSONL dataset path(s)."""

    if filename is not None:
        paths = [resolve_user_path(filename, data_dir)]
    else:
        paths = [data_dir / name for name in DEFAULT_DATASET_FILES]

    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing dataset file(s):\n{formatted}")
    return paths


def load_datasets(paths: Sequence[Path], limit: int | None = None) -> pd.DataFrame:
    """Load and concatenate JSONL datasets, then validate required columns."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        LOGGER.info("Loading dataset: %s", path)
        try:
            frames.append(pd.read_json(path, lines=True))
        except ValueError as exc:
            raise ValueError(f"Could not parse JSONL dataset {path}: {exc}") from exc

    dataset = pd.concat(frames, ignore_index=True)
    required_columns = {"headline"}
    for primary, alternatives in LEANING_FIELDS:
        required_columns.update((primary, alternatives))

    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    if limit is not None:
        dataset = dataset.head(limit).copy()
        LOGGER.info("Test limit active: processing %d rows.", len(dataset))

    if dataset.empty:
        raise ValueError("The loaded dataset contains no rows.")

    return dataset


def load_full_articles(full_articles_dir: Path) -> pd.DataFrame:
    """Load all extracted full-article JSON files into one lookup table."""

    if not full_articles_dir.is_dir():
        raise FileNotFoundError(
            f"Full-article directory does not exist: {full_articles_dir}"
        )

    json_files = sorted(full_articles_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No .json files were found in {full_articles_dir}."
        )

    rows: list[dict[str, Any]] = []
    for path in tqdm(json_files, desc="Loading full articles", unit="file"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read full-article file {path}: {exc}") from exc

        if not isinstance(data, dict):
            LOGGER.warning("Skipping %s because its top-level JSON is not an object.", path)
            continue

        for title, leanings in data.items():
            if not isinstance(leanings, dict):
                continue
            for leaning, article in leanings.items():
                if not isinstance(article, dict):
                    continue
                rows.append(
                    {
                        "title": title,
                        "leaning": leaning,
                        "source_file": path.name,
                        **article,
                    }
                )

    if not rows:
        raise ValueError("No usable article records were found in full_articles/.")

    articles = pd.DataFrame(rows)
    required_columns = {
        "url",
        "extracted_headline",
        "extracted_body_text",
        "extracted_images",
        "domain",
        "leaning",
    }
    missing_columns = sorted(required_columns - set(articles.columns))
    if missing_columns:
        raise ValueError(
            "Full-article data is missing required fields: "
            + ", ".join(missing_columns)
        )

    # URL is the stable join key used by the AllSides records. Keeping the first
    # duplicate mirrors the original script's ``values[0]`` behavior.
    articles = articles.drop_duplicates(subset="url", keep="first").reset_index(drop=True)
    LOGGER.info("Loaded %d unique full articles.", len(articles))
    return articles


def build_article_lookup(full_articles: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    """Create an O(1) URL lookup instead of repeatedly filtering a DataFrame."""

    return {
        str(row["url"]): row
        for row in full_articles.to_dict(orient="records")
        if pd.notna(row.get("url"))
    }


def as_candidate_list(value: Any) -> list[Mapping[str, Any]]:
    """Normalize an alternatives cell into a list of candidate dictionaries."""

    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [item for item in value if isinstance(item, dict)]
    return []


def extract_image_paths(image_metadata: Any) -> list[str]:
    """Extract non-empty, non-GIF local paths from article image metadata."""

    if not isinstance(image_metadata, list):
        return []

    paths: list[str] = []
    for item in image_metadata:
        if not isinstance(item, dict):
            continue
        local_path = item.get("local_path")
        if not isinstance(local_path, str) or not local_path.strip():
            continue
        if local_path.lower().endswith(".gif"):
            continue
        paths.append(local_path)
    return paths


def resolve_image_path(local_path: str, data_dir: Path) -> Path:
    """Resolve an image path stored in article metadata."""

    path = Path(local_path).expanduser()
    return path.resolve() if path.is_absolute() else (data_dir / path).resolve()


def image_is_readable(path: Path) -> bool:
    """Quickly verify that Pillow can identify and decode an image file."""

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def resolve_article(
    candidate: Mapping[str, Any],
    *,
    article_lookup: Mapping[str, Mapping[str, Any]],
    data_dir: Path,
    stats: ArticleResolutionStats,
) -> dict[str, Any] | None:
    """Resolve one dataset candidate to a complete, image-eligible article.

    Eligibility deliberately does not depend on the input condition. Every
    accepted article must belong to a supported source and have at least one
    readable, non-GIF local image. This keeps text-only and image-based samples
    directly comparable.
    """

    source = candidate.get("source")
    if source not in SOURCE_FILE_BY_NAME:
        stats.unsupported_source += 1
        return None

    url = candidate.get("link")
    if not isinstance(url, str) or not url:
        stats.missing_url += 1
        return None

    full_article = article_lookup.get(url)
    if full_article is None:
        stats.missing_full_article += 1
        return None

    raw_image_paths = extract_image_paths(full_article.get("extracted_images"))
    if not raw_image_paths:
        stats.missing_image_metadata += 1
        return None

    usable_paths: list[Path] = []
    saw_existing_file = False
    for raw_path in raw_image_paths:
        path = resolve_image_path(raw_path, data_dir)
        if not path.is_file():
            continue
        saw_existing_file = True
        if image_is_readable(path):
            usable_paths.append(path)

    if not usable_paths:
        if saw_existing_file:
            stats.unreadable_image += 1
        else:
            stats.missing_image_file += 1
        return None

    stats.accepted += 1
    return {
        "extracted_headline": str(full_article.get("extracted_headline", "")),
        "extracted_body_text": str(full_article.get("extracted_body_text", "")),
        "leaning": str(full_article.get("leaning", "")),
        "link": url,
        "local_path": usable_paths,
        "source": str(full_article.get("domain", "")),
        "summary": str(candidate.get("summary", "") or ""),
    }


def compose_text(article: Mapping[str, Any], input_condition: str) -> str:
    """Build the textual component requested by an input condition."""

    parts: list[str] = []
    if "headline" in input_condition:
        parts.append(str(article.get("extracted_headline", "")))
    if "summary" in input_condition:
        parts.append(str(article.get("summary", "")))
    if "text" in input_condition:
        parts.append(str(article.get("extracted_body_text", "")))
    return "\n".join(part for part in parts if part)


def load_rgb_image(path: Path) -> Image.Image:
    """Load an image into memory and close the underlying file immediately."""

    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not load image {path}: {exc}") from exc


def encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    """Use the model's document encoder when available, otherwise ``encode``."""

    encode_document = getattr(model, "encode_document", None)
    if callable(encode_document):
        return np.asarray(encode_document(texts, batch_size=batch_size))
    LOGGER.warning(
        "Model has no encode_document() method; falling back to encode() for text."
    )
    return np.asarray(model.encode(texts, batch_size=batch_size))


def mean_embeddings_by_article(
    embeddings: Sequence[np.ndarray],
    article_indexes: Sequence[int],
    article_count: int,
) -> list[np.ndarray]:
    """Average per-image embeddings into exactly one embedding per article."""

    grouped: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    for article_index, embedding in zip(article_indexes, embeddings, strict=True):
        grouped[article_index].append(np.asarray(embedding))

    missing = [index for index in range(article_count) if not grouped[index]]
    if missing:
        raise RuntimeError(
            "No image embedding was produced for article index(es): "
            + ", ".join(map(str, missing))
        )

    return [np.mean(grouped[index], axis=0) for index in range(article_count)]


def compute_embeddings_multiple(
    model: Any,
    articles: Sequence[Mapping[str, Any]],
    input_condition: str,
    batch_size: int,
) -> list[np.ndarray]:
    """Compute one embedding for each article while batching model calls.

    For multimodal conditions, each image is paired with the article's text and
    encoded separately. The resulting image-level embeddings are averaged to
    obtain one article-level embedding, preserving the original script's logic.
    """

    texts = [compose_text(article, input_condition) for article in articles]
    includes_image = "image" in input_condition
    includes_text = any(token in input_condition for token in ("headline", "summary", "text"))

    if not includes_image:
        return [np.asarray(item) for item in encode_texts(model, texts, batch_size)]

    image_inputs: list[Image.Image] = []
    image_article_indexes: list[int] = []
    multimodal_inputs: list[dict[str, Any]] = []

    try:
        for article_index, (article, text) in enumerate(zip(articles, texts, strict=True)):
            for path in article["local_path"]:
                image = load_rgb_image(Path(path))
                image_inputs.append(image)
                image_article_indexes.append(article_index)
                if includes_text:
                    multimodal_inputs.append({"image": image, "text": text})

        if includes_text:
            raw_embeddings = model.encode(multimodal_inputs, batch_size=batch_size)
        else:
            raw_embeddings = model.encode(image_inputs, batch_size=batch_size)

        return mean_embeddings_by_article(
            raw_embeddings,
            image_article_indexes,
            len(articles),
        )
    finally:
        # Pillow images retain memory even after the source file is closed. Close
        # the in-memory objects as soon as the model call has completed.
        for image in image_inputs:
            image.close()


def output_filename(input_condition: str, filename: Path | None) -> str:
    """Return the legacy-compatible checkpoint filename used by this project."""

    # Retaining the original suffix prevents existing pipelines and checkpoints
    # from breaking. The custom filename is reduced to its basename for safety.
    suffix = "__" if filename is None else f"__{filename.name}"
    return f"embeddings_{input_condition}{suffix}.pickle"


def load_checkpoint(path: Path, overwrite: bool) -> Results:
    """Load a previous checkpoint, with a backup fallback for partial writes."""

    if overwrite or not path.exists():
        return {}

    backup = path.with_suffix(path.suffix + ".backup")
    try:
        with path.open("rb") as handle:
            results = pickle.load(handle)
        if not isinstance(results, dict):
            raise TypeError("checkpoint root is not a dictionary")
        LOGGER.info("Resuming checkpoint with %d headlines: %s", len(results), path)
        return results
    except (OSError, EOFError, pickle.UnpicklingError, TypeError) as exc:
        LOGGER.warning("Could not load checkpoint %s: %s", path, exc)
        if backup.exists():
            LOGGER.warning("Trying backup checkpoint: %s", backup)
            with backup.open("rb") as handle:
                results = pickle.load(handle)
            if not isinstance(results, dict):
                raise TypeError(f"Backup checkpoint {backup} is not a dictionary.")
            return results
        raise RuntimeError(
            f"Checkpoint {path} is unreadable and no usable backup exists. "
            "Use --overwrite to start again."
        ) from exc


def save_checkpoint_atomic(results: Results, path: Path) -> None:
    """Write a pickle checkpoint atomically and retain one backup copy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".backup")

    if path.exists():
        shutil.copy2(path, backup)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def result_contains_group(results: Results, headline: str, group: str) -> bool:
    """Return whether a leaning group has already been saved for a headline."""

    return group in results.get(headline, {})


def collect_pending_articles(
    row: pd.Series,
    *,
    results: Results,
    article_lookup: Mapping[str, Mapping[str, Any]],
    data_dir: Path,
    resolution_stats: ArticleResolutionStats,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]], int]:
    """Collect unresolved primary and alternative articles for one headline."""

    headline = str(row["headline"])
    articles: list[dict[str, Any]] = []
    article_destinations: list[tuple[str, str]] = []
    already_present = 0

    for primary_group, alternatives_group in LEANING_FIELDS:
        if result_contains_group(results, headline, primary_group):
            already_present += 1
        else:
            primary_candidate = row.get(primary_group)
            if isinstance(primary_candidate, dict):
                article = resolve_article(
                    primary_candidate,
                    article_lookup=article_lookup,
                    data_dir=data_dir,
                    stats=resolution_stats,
                )
                if article is not None:
                    articles.append(article)
                    article_destinations.append((article["link"], primary_group))

        if result_contains_group(results, headline, alternatives_group):
            already_present += 1
        else:
            for candidate in as_candidate_list(row.get(alternatives_group)):
                article = resolve_article(
                    candidate,
                    article_lookup=article_lookup,
                    data_dir=data_dir,
                    stats=resolution_stats,
                )
                if article is not None:
                    articles.append(article)
                    article_destinations.append((article["link"], alternatives_group))

    return articles, article_destinations, already_present


def store_embeddings(
    results: Results,
    headline: str,
    destinations: Sequence[tuple[str, str]],
    embeddings: Sequence[np.ndarray],
) -> None:
    """Store article embeddings in the project's existing nested result format."""

    if len(destinations) != len(embeddings):
        raise RuntimeError(
            f"Expected {len(destinations)} embeddings, received {len(embeddings)}."
        )

    headline_results = results.setdefault(headline, {})
    for (url, group), embedding in zip(destinations, embeddings, strict=True):
        group_results = headline_results.setdefault(group, {})
        group_results[url] = np.asarray(embedding)


def run_embedding_generation(
    dataset: pd.DataFrame,
    *,
    model: Any,
    input_condition: str,
    results: Results,
    output_path: Path,
    article_lookup: Mapping[str, Mapping[str, Any]],
    data_dir: Path,
    batch_size: int,
    checkpoint_every: int,
    resolution_stats: ArticleResolutionStats,
) -> RunStats:
    """Generate missing embeddings for one condition and save resumable progress."""

    run_stats = RunStats()
    changed_since_checkpoint = 0

    progress = tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc=input_condition,
        unit="headline",
    )
    for _, row in progress:
        run_stats.headlines_seen += 1
        headline = str(row["headline"])

        articles, destinations, already_present = collect_pending_articles(
            row,
            results=results,
            article_lookup=article_lookup,
            data_dir=data_dir,
            resolution_stats=resolution_stats,
        )
        run_stats.articles_already_present += already_present

        if not articles:
            continue

        embeddings = compute_embeddings_multiple(
            model,
            articles,
            input_condition,
            batch_size,
        )
        store_embeddings(results, headline, destinations, embeddings)

        run_stats.headlines_changed += 1
        run_stats.articles_embedded += len(articles)
        changed_since_checkpoint += 1
        progress.set_postfix(embedded=run_stats.articles_embedded)

        if changed_since_checkpoint >= checkpoint_every:
            save_checkpoint_atomic(results, output_path)
            run_stats.checkpoints_written += 1
            changed_since_checkpoint = 0

    # Always save once at the end when unsaved changes remain. If nothing changed
    # and no checkpoint exists, save an empty result so the completed run is clear.
    if changed_since_checkpoint > 0 or not output_path.exists():
        save_checkpoint_atomic(results, output_path)
        run_stats.checkpoints_written += 1

    return run_stats


def validate_article_coverage(
    dataset: pd.DataFrame,
    *,
    article_lookup: Mapping[str, Mapping[str, Any]],
    data_dir: Path,
) -> ArticleResolutionStats:
    """Resolve every candidate once and summarize image-eligibility coverage."""

    stats = ArticleResolutionStats()
    for _, row in tqdm(
        dataset.iterrows(),
        total=len(dataset),
        desc="Validating candidates",
        unit="headline",
    ):
        for primary_group, alternatives_group in LEANING_FIELDS:
            candidates: list[Mapping[str, Any]] = []
            primary = row.get(primary_group)
            if isinstance(primary, dict):
                candidates.append(primary)
            candidates.extend(as_candidate_list(row.get(alternatives_group)))

            for candidate in candidates:
                resolve_article(
                    candidate,
                    article_lookup=article_lookup,
                    data_dir=data_dir,
                    stats=stats,
                )
    return stats


def load_embedding_model(model_name: str, device: str | None) -> Any:
    """Load SentenceTransformer only when embedding work actually begins."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to generate embeddings. Install "
            "it in the active environment, or use --validate-only first."
        ) from exc

    LOGGER.info("Loading embedding model: %s", model_name)
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device

    try:
        model = SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"Could not load embedding model {model_name!r}: {exc}") from exc

    model.eval()
    return model


def log_resolution_summary(stats: ArticleResolutionStats) -> None:
    """Log a compact explanation of article eligibility outcomes."""

    LOGGER.info("Eligible article candidates: %d", stats.accepted)
    LOGGER.info("Rejected article candidates: %d", stats.rejected)
    if stats.rejected:
        LOGGER.info("  Unsupported source: %d", stats.unsupported_source)
        LOGGER.info("  Missing candidate URL: %d", stats.missing_url)
        LOGGER.info("  No matching full article: %d", stats.missing_full_article)
        LOGGER.info("  No usable image metadata: %d", stats.missing_image_metadata)
        LOGGER.info("  Image file missing: %d", stats.missing_image_file)
        LOGGER.info("  Image unreadable: %d", stats.unreadable_image)


def main(argv: Sequence[str] | None = None) -> int:
    """Program entry point."""

    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.list_options:
        print_supported_options()
        return 0

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    full_articles_dir = (
        resolve_user_path(args.full_articles_dir, data_dir)
        if args.full_articles_dir is not None
        else data_dir / "full_articles"
    )
    output_dir = (
        resolve_user_path(args.output_dir, data_dir)
        if args.output_dir is not None
        else data_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths = resolve_dataset_paths(data_dir, args.filename)
    dataset = load_datasets(dataset_paths, args.limit)
    full_articles = load_full_articles(full_articles_dir)
    article_lookup = build_article_lookup(full_articles)

    LOGGER.info("Dataset rows: %d", len(dataset))
    LOGGER.info("Comparability rule: every selected article must have a usable image.")

    if args.validate_only:
        coverage = validate_article_coverage(
            dataset,
            article_lookup=article_lookup,
            data_dir=data_dir,
        )
        log_resolution_summary(coverage)
        return 0

    model = load_embedding_model(args.model, args.device)

    for input_condition in dict.fromkeys(args.input_condition):
        output_path = output_dir / output_filename(input_condition, args.filename)
        LOGGER.info("Starting condition: %s", input_condition)
        LOGGER.info("Output checkpoint: %s", output_path)

        results = load_checkpoint(output_path, args.overwrite)
        resolution_stats = ArticleResolutionStats()
        run_stats = run_embedding_generation(
            dataset,
            model=model,
            input_condition=input_condition,
            results=results,
            output_path=output_path,
            article_lookup=article_lookup,
            data_dir=data_dir,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            resolution_stats=resolution_stats,
        )

        LOGGER.info(
            "Finished %s: %d new article embeddings across %d changed headlines.",
            input_condition,
            run_stats.articles_embedded,
            run_stats.headlines_changed,
        )
        LOGGER.info("Checkpoint writes: %d", run_stats.checkpoints_written)
        log_resolution_summary(resolution_stats)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user. The most recent checkpoint remains available.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            LOGGER.exception("Detailed traceback")
        raise SystemExit(1)
