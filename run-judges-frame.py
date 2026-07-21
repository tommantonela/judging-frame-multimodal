"""Run LLM-as-a-judge framing experiments on matched news articles.

The script supports five experiment modes:

* ``individual``: score each article separately on a framing dimension.
* ``individual-metadata``: run individual scoring with source/label metadata.
* ``triadic``: rank three articles about the same event.
* ``pairs``: compare all three article pairs for an event.
* ``ideology_triadic``: infer ideological orientation for three articles.

Expected directory layout
-------------------------
The directory passed with ``--dir-path`` should contain either:

1. ``allsides_Jan-May_2025.jsonl`` and
   ``allsides_Jun2025_May2026.jsonl``; or
2. ``allsides_Jan2025_May2026_combined.jsonl``.

It must also contain a ``full_articles/`` directory with the source JSON files
and any local article images referenced by those files. To keep the evaluated
sample comparable across input conditions, every selected article must have at
least one usable non-GIF image, even when the model receives only text. Images
are attached to the prompt only when the selected input condition includes them.

Examples
--------
Run one headline-and-text individual experiment::

    python run-judges-frame-commented.py \
        --dir-path ./data \
        --experiment individual \
        --dimension emotional-intensity \
        --input-condition headline-text \
        --llm gemma4:e2b

Run every configured dimension and compatible input condition::

    python run-judges-frame-commented.py --dir-path ./data --experiment individual

Notes
-----
* Results are checkpointed after each event, so interrupted runs can resume.
* Existing result files are copied to ``__backup.pickle`` before resuming.
* The Ollama model must already be available locally.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


# LLM-specific dependencies are imported lazily. This keeps commands such as
# --help and --list-options usable even before the optional packages are installed.
HumanMessage = None
SystemMessage = None
ChatOllama = None
ResponseError = None


def load_llm_dependencies() -> None:
    """Import optional LLM packages and provide a clear installation error."""
    global HumanMessage, SystemMessage, ChatOllama, ResponseError

    if all(value is not None for value in (HumanMessage, SystemMessage, ChatOllama, ResponseError)):
        return

    try:
        from langchain_core.messages import HumanMessage as _HumanMessage
        from langchain_core.messages import SystemMessage as _SystemMessage
        from langchain_ollama import ChatOllama as _ChatOllama
        from ollama import ResponseError as _ResponseError
    except ModuleNotFoundError as error:
        missing_package = error.name or "an LLM dependency"
        raise RuntimeError(
            f"Missing Python package '{missing_package}'. Install the project "
            "dependencies (including langchain-core, langchain-ollama, and ollama) "
            "before running an experiment."
        ) from error

    HumanMessage = _HumanMessage
    SystemMessage = _SystemMessage
    ChatOllama = _ChatOllama
    ResponseError = _ResponseError


# Reproducible shuffling reduces systematic A/B/C position effects.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Models used when the user does not pass --llm.
DEFAULT_MODELS = ("gemma4:e2b",)

# Each primary leaning may have one or more fallback candidates in the dataset.
LEANING_COLUMNS = (
    ("left", "more_left"),
    ("center", "more_center"),
    ("right", "more_right"),
)

# The three unique pairs among three articles.
PAIR_INDEXES = ((0, 1), (1, 2), (0, 2))

# ------------------------------------------- PROMPT TEMPLATES INDIVIDUAL -------------------------
# -------------------------------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a neutral and consistent news-content evaluation assistant, designed to apply predefined analytical criteria without relying on personal opinions, political assumptions, or external context. You behave as a controlled measurement instrument for media analysis, prioritizing consistency, evidence from the provided content, and strict adherence to the requested output format."

PROMPT_INDIVIDUAL_EVALUATION_START = """
You are evaluating a news item for one specific bias-related framing construct.

Your task is not to determine whether the article is biased overall. Your task is only to assess the degree to which the specified construct is present in the provided content.

Construct:
{CONSTRUCT_NAME}

Definition:
{CONSTRUCT_DEFINITION}

Important instructions:

1. Evaluate only the provided content.
2. Do not use external knowledge about the event, outlet, author, country, or political context.
3. Do not infer or assign the ideological orientation of the article.
4. Do not evaluate whether the article is factually true or false.
5. Do not evaluate whether the article is good or bad journalism.
5. Do not evaluate whether the article is fair, balanced, objective, or biased overall.
6. Focus only on observable evidence related to the specified construct.
7. If the evidence is ambiguous, limited, or mixed, choose the best-supported score and lower the confidence.
8. If there are contradictions or tensions within the provided input, base the score on the full provided input and mention the tension in the evidence field.
9. Do not mention information that is not present in the provided content.

Input condition:
You will receive: {INPUT_CONDITION}

News item:

{ARTICLE_HEADLINE}

{ARTICLE_SUMMARY}

{ARTICLE_SOURCE}

{ARTICLE_LEANING}

"""

PROMPT_INDIVIDUAL_EVALUATION_IMAGE = """

{IMAGE_INPUT}

"""

PROMPT_INDIVIDUAL_EVALUATION_END = """
Scale:
Use a 1–5 Likert scale.

{SCALE_ANCHORS}

Return only valid JSON with the following fields:
{{
"score": 1,
"confidence": 1,
"evidence": "Brief explanation grounded only in the provided content.",
"uncertainty_reason": "Brief reason for uncertainty, or null if not applicable."
}}
"""

# ------------------------------------------- PROMPT TEMPLATES TRIADIC ----------------------------
# -------------------------------------------------------------------------------------------------

PROMPT_TRIADIC_EVALUATION_START = """
You are comparing three news items about the same event for one specific bias-related framing construct.

Your task is not to determine which article is most biased overall. Your task is only to rank the articles according to the degree to which the specified construct is present.

Construct to assess:
{CONSTRUCT_NAME}

Definition:
{CONSTRUCT_DEFINITION}

Evaluation rules:

1. The article labels A, B, and C are arbitrary.
2. Do not assume that Article A, Article B, or Article C corresponds to any political orientation.
3. Evaluate only the provided content.
4. Do not use external knowledge about the event, sources, outlets, authors, countries, or political context.
5. Do not evaluate whether any article is factually true or false.
6. Do not evaluate whether any article is good or bad journalism.
7. Do not evaluate whether any article is fair, balanced, objective, or biased overall.
8. Focus only on observable evidence related to the specified construct.
9. If there are contradictions or tensions within the provided input, base your ranking on the full provided input and mention the tension in the rationale.
10. Ties are allowed if two or more articles show approximately the same level of the construct.
11. If the evidence is ambiguous, limited, or mixed, choose the best-supported ranking and lower the confidence.
12. Do not mention information that is not present in the provided content.

Input condition:
{INPUT_CONDITION}
"""

PROMPT_TRIADIC_EVALUATION_ARTICLE = """
Article {ARTICLE_LABEL}:
{ARTICLE_HEADLINE}

{ARTICLE_SUMMARY}

{ARTICLE_SOURCE}

{ARTICLE_LEANING}
"""

PROMPT_TRIADIC_EVALUATION_END = """
Return only valid JSON in exactly this format:
{{
"ranking_low_to_high": ["Article A", "Article B", "Article C"],
"ties": [],
"highest_article": "Article C",
"lowest_article": "Article A",
"confidence": 1,
"rationale": "Brief explanation grounded only in the provided content."
}}
"""

# ------------------------------------------- PROMPT TEMPLATES PAIRS ------------------------------
# -------------------------------------------------------------------------------------------------


PROMPT_PAIRS_EVALUATION_START = """
You are comparing two news items about the same event for one specific bias-related framing construct.

Your task is not to determine which article is most biased overall. Your task is only to determine which article shows a stronger presence of the specified construct.

Construct to assess:
{CONSTRUCT_NAME}

Definition:
{CONSTRUCT_DEFINITION}

Evaluation rules:

1. The article labels A and B are arbitrary.
2. Do not assume that Article A or Article B corresponds to any political orientation.
3. Evaluate only the provided content.
4. Do not use external knowledge about the event, sources, outlets, authors, countries, or political context.
5. Do not evaluate whether either article is factually true or false.
6. Do not evaluate whether either article is good or bad journalism.
7. Do not evaluate whether either article is fair, balanced, objective, or biased overall.
8. Focus only on observable evidence related to the specified construct.
9. If there are contradictions or tensions within the provided input, base your comparison on the full provided input and mention the tension in the rationale.
10. If both articles show approximately the same level of the construct, choose "tie".
11. If the evidence is ambiguous, limited, or mixed, choose the best-supported comparison and lower the confidence.
12. Do not mention information that is not present in the provided content.

Input condition:
{INPUT_CONDITION}
"""

PROMPT_PAIRS_EVALUATION_END = """
Return only valid JSON in exactly this format:
{{
"stronger_article": "Article A | Article B | tie",
"relation": "A > B | B > A | A = B",
"confidence": 1,
"rationale": "Brief explanation grounded only in the provided content."
}}
"""

# ------------------------------------------- PROMPT TEMPLATES IDEOLOGY TRIADIC -------------------
# -------------------------------------------------------------------------------------------------

PROMPT_IDEOLOGY_TRIADIC_EVALUATION_START = """
You are comparing three news items about the same event.

Your task is to infer whether each article appears to be left-leaning, center/neutral, right-leaning, or unclear based only on the provided content.

This is not a task about truthfulness, journalistic quality, or overall bias. It is only an inference about the apparent ideological orientation expressed through the article’s language, emphasis, actor portrayal, responsibility attribution and issue framing.

Evaluation rules:

1. The article labels A, B, and C are arbitrary.
2. Do not assume that Article A, Article B, or Article C corresponds to any political orientation.
3. Do not assume that all three ideological categories must be present.
4. Use "unclear" when the provided content does not contain enough evidence to infer orientation.
5. Evaluate only the provided content.
6. Do not use external knowledge about the event, sources, outlets, authors, countries, public figures, or political context.
7. Do not evaluate whether any article is factually true or false.
8. Do not evaluate whether any article is good or bad journalism.
9. Do not evaluate whether any article is fair, balanced, objective, or biased overall.
10. If there are contradictions or tensions within the provided input, base your inference on the full provided input and mention the tension in the evidence field.
11. If the evidence is ambiguous, limited, or mixed, choose the best-supported category and lower the confidence.
12. Do not mention information that is not present in the provided content.

Input condition:
{INPUT_CONDITION}
"""

PROMPT_IDEOLOGY_TRIADIC_EVALUATION_END = """
Return only valid JSON in exactly this format:
{{
"article_A": {{
"inferred_orientation": "left-leaning | center/neutral | right-leaning | unclear",
"confidence": 1,
"evidence": "Brief explanation grounded only in the provided content."
}},
"article_B": {{
"inferred_orientation": "left-leaning | center/neutral | right-leaning | unclear",
"confidence": 1,
"evidence": "Brief explanation grounded only in the provided content."
}},
"article_C": {{
"inferred_orientation": "left-leaning | center/neutral | right-leaning | unclear",
"confidence": 1,
"evidence": "Brief explanation grounded only in the provided content."
}}
}}
"""

# ------------------------------------------- MAPPINGS --------------------------------------------
# -------------------------------------------------------------------------------------------------

input_conditions = {}

input_conditions['headline'] = "the headline of the news article."
input_conditions['headline-text'] = "the headline and text of the news article."
input_conditions['image'] = "the images of the news article."
# input_conditions['headline-image'] = "the headline and images of the news article."
input_conditions['headline-text-image'] = "the headline, text and images of the news article."

input_conditions['headline-text-source'] = "the headline, text and source of the news article."
input_conditions['image-source'] = "the images and source of the news article."

input_conditions['headline-text-source-label'] = "the headline, text and source of the news article."
input_conditions['image-source-label'] = "the images, source and label of the news article."

input_conditions['headline-text-image-source'] = "the headline, text, images and source of the news article."
input_conditions['headline-text-image-source-label'] = "the headline, text, images source and label of the news article."

dimensions = {}

dimensions['emotional-intensity'] = {"CONSTRUCT_NAME": "Emotional intensity",
                "CONSTRUCT_DEFINITION": """
Emotional intensity refers to the degree to which the news item expresses or evokes strong affective cues, such as anger, fear, sadness, sympathy, outrage, hope, shock, or urgency. It may appear through emotionally charged wording, dramatic descriptions, affective images, or emphasis on emotionally salient consequences.

Do not score emotional intensity based only on whether the event itself is serious or negative. Score the degree to which the provided content presents the event in an emotionally charged way.
""",
                "SCALE_ANCHORS": """
1 = Very low: The presentation is neutral, factual, restrained, and contains little or no affective language or imagery.
2 = Low: The presentation is mostly neutral, with minor emotional cues or mildly affective wording or imagery.
3 = Moderate: Emotional cues are noticeable, but they do not dominate the presentation.
4 = High: The presentation is clearly emotionally charged, with strong affective wording, imagery, or emphasis on emotionally salient consequences.
5 = Very high: The presentation is dominated by intense emotional cues, such as outrage, fear, grief, shock, urgency, sympathy, or dramatic affective emphasis.
"""
                 }

dimensions['conflict-framing'] = {"CONSTRUCT_NAME": 'Conflict framing',
                "CONSTRUCT_DEFINITION": """
Conflict framing refers to the degree to which the news item emphasizes disagreement, confrontation, clashes, disputes, blame, struggle, or opposition between actors, groups, institutions, countries, or political positions. It captures whether the event is presented mainly through tension or confrontation.

Do not confuse conflict framing with polarization framing. Conflict framing can involve any disagreement or confrontation, while polarization framing specifically involves broader social, political, or ideological division into opposing camps.""",
                "SCALE_ANCHORS": """
1 = Very low: The item does not emphasize disagreement, confrontation, dispute, blame, or opposition.
2 = Low: The item includes minor references to disagreement or tension, but conflict is not central.
3 = Moderate: Conflict, disagreement, or confrontation is noticeable and contributes to how the event is presented.
4 = High: Conflict is a central organizing frame, with clear emphasis on clashes, disputes, blame, struggle, or opposition between actors.
5 = Very high: The item is dominated by conflict framing, presenting the event primarily through confrontation, adversarial relations, blame, or struggle.
"""
                 }

dimensions['polarization-framing'] = {"CONSTRUCT_NAME": 'Polarization framing',
                "CONSTRUCT_DEFINITION": """
Polarization framing refers to the degree to which the news item presents the event through opposing camps, ideological division, us-versus-them dynamics, or deep social or political antagonism. It captures whether the content frames the issue as part of a broader divide between groups, parties, communities, identities, or worldviews.

Do not score polarization highly merely because conflict is present. A news item may describe conflict without framing society or politics as deeply divided into opposing sides.
""",
                "SCALE_ANCHORS": """
1 = Very low: The item does not present the event through opposing camps, ideological division, or us-versus-them dynamics.
2 = Low: The item contains minor cues of social, political, or ideological division, but these are not central.
3 = Moderate: The item noticeably frames the event through division between groups, parties, communities, identities, or viewpoints.
4 = High: The item strongly emphasizes opposing camps, ideological antagonism, or us-versus-them dynamics.
5 = Very high: The item is dominated by polarization framing, presenting the event primarily as part of a deep, antagonistic divide between opposing sides.
"""
                 }

# dimensions['human-interest'] = {"CONSTRUCT_NAME": "Human interest",
#                 "CONSTRUCT_DEFINITION": """
# Human-interest refers to the degree to which the news item foregrounds individual experiences, personal stories, suffering, vulnerability, victimhood, fear, grief, or lived consequences for ordinary people or affected groups. It captures whether the event is personalized through human consequences rather than presented mainly through institutions, policies, statistics, or abstract processes.

# Do not score this highly only because people are mentioned. Score it highly when personal suffering, vulnerability, emotional experience, or victimhood is central to the presentation.
# """,
#                 "SCALE_ANCHORS": """
# 1 = Very low: The item does not foreground individual experiences, suffering, vulnerability, victimhood, or personal consequences.
# 2 = Low: The item briefly mentions affected individuals or groups, but personal experience or vulnerability is not central.
# 3 = Moderate: Human consequences, suffering, vulnerability, or individual experiences are noticeable in the presentation.
# 4 = High: Human-interest or victimization framing is central, with strong emphasis on personal stories, affected people, vulnerability, suffering, fear, grief, or lived consequences.
# 5 = Very high: The item is dominated by human-interest or victimization framing, presenting the event primarily through personal suffering, victimhood, emotional experience, or vulnerable affected groups.
# """
#                  }       

# dimensions['elite-focus'] = {"CONSTRUCT_NAME": "Elite focus",
#                 "CONSTRUCT_DEFINITION": """
# Elite focus refers to the degree to which the news item foregrounds powerful or high-status actors, such as politicians, government officials, institutional leaders, courts, corporations, military authorities, police, celebrities, international organizations, or other influential figures. It captures whether the event is framed primarily around elite actors, their actions, statements, conflicts, or decisions.

# Do not score elite focus highly merely because an institution or leader is briefly mentioned. Score it highly when elite actors are central to how the event is presented.
# """,
#                 "SCALE_ANCHORS": """
# 1 = Very low: The item does not foreground elite actors, institutions, leaders, or powerful figures.
# 2 = Low: Elite actors or institutions are mentioned, but they are peripheral to the presentation.
# 3 = Moderate: Elite actors, institutions, leaders, or powerful figures are noticeable and relevant to how the event is presented.
# 4 = High: Elite actors or institutions are central to the presentation, with strong emphasis on their actions, statements, decisions, or conflicts.
# 5 = Very high: The item is dominated by elite focus, presenting the event primarily through powerful actors, leaders, institutions, authorities, corporations, courts, military, police, celebrities, or international organizations.
# """
#                  }   

dimensions['sensationalism'] = {"CONSTRUCT_NAME": "Sensationalism",
                "CONSTRUCT_DEFINITION": """
Sensationalism refers to the degree to which the news item presents the event in an exaggerated, dramatic, shocking, alarming, scandal-oriented, emotionally provocative, or attention-grabbing way. It may appear through loaded wording, dramatic emphasis, disproportionate urgency, shocking imagery, overstatement, or presentation choices designed to attract attention rather than neutrally describe the event.

Do not score sensationalism highly only because the event itself is serious, violent, tragic, or politically important. Score sensationalism based on the style of presentation, not on the severity of the event.
""",
                "SCALE_ANCHORS": """
1 = Very low: The presentation is neutral, restrained, and descriptive, with no exaggerated, shocking, alarmist, or attention-grabbing style.
2 = Low: The presentation is mostly restrained, with minor dramatic, attention-grabbing, or loaded elements.
3 = Moderate: The presentation contains noticeable dramatic, shocking, alarming, scandal-oriented, or attention-grabbing elements, but they do not dominate.
4 = High: The presentation is clearly sensationalized, with strong use of exaggerated, dramatic, shocking, alarmist, scandal-oriented, or emotionally provocative style.
5 = Very high: The item is dominated by sensationalism, presenting the event primarily through exaggerated, shocking, alarmist, scandal-oriented, or attention-grabbing language, imagery, or emphasis.
"""
                 }             

mapping_sources = {}
mapping_sources['Associated Press'] = 'apnews.com.json'

mapping_sources['Fox Business'] = "foxbusiness.com.json"

mapping_sources['Fox News (Opinion)'] = "foxnews.com.json"
mapping_sources['Fox News Digital'] = "foxnews.com.json"

mapping_sources['NBC Los Angeles'] = "nbcnews.com.json"
mapping_sources['NBC News Digital'] = "nbcnews.com.json"

mapping_sources['Newsweek'] = "newsweek.com.json"
mapping_sources['Newsweek Fact Check'] = "newsweek.com.json"

mapping_sources['New York Post (News)'] = "nypost.com.json"
mapping_sources['New York Post (Opinion)'] = "nypost.com.json"

mapping_sources['New York Times (News)'] = "nytimes.com.json"
mapping_sources['New York Times (Opinion)'] = "nytimes.com.json"

mapping_sources['Politico'] = "politico.com.json"

mapping_sources['Reuters'] = "reuters.com.json"

mapping_sources['The Guardian'] = "theguardian.com.json"

mapping_sources['The Hill'] = "thehill.com.json"

mapping_sources['Washington Examiner'] = "washingtonexaminer.com.json"

mapping_sources['Washington Post'] = "washingtonpost.com.json"
mapping_sources['Washington Post Fact Check'] = "washingtonpost.com.json"

# ================================================================================================
# Prompt construction and model invocation
# ================================================================================================


def convert_to_base64(file_path: Path) -> str:
    """Load an image, convert it to JPEG, and return its Base64 representation.

    Converting all images to JPEG gives the multimodal model a consistent input
    format. Transparent/paletted images are converted to RGB first.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Article image not found: {file_path}")

    with Image.open(file_path) as image:
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resolve_image_path(base_dir: Path, image_path: str) -> Path:
    """Resolve an image path stored in the dataset against the data directory."""
    candidate = Path(image_path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def add_article_images(
    messages: list[dict[str, str]],
    article: Mapping[str, Any],
    base_dir: Path,
) -> None:
    """Append every usable article image to a LangChain multimodal message."""
    for image_path in article.get("local_path", []):
        full_path = resolve_image_path(base_dir, image_path)
        encoded_image = convert_to_base64(full_path)

        messages.extend(
            [
                {"type": "text", "text": "Image of the news article:\n"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{encoded_image}",
                },
                {"type": "text", "text": "\n\n"},
            ]
        )


def optional_article_field(
    input_condition: str,
    required_token: str,
    label: str,
    value: Any,
) -> str:
    """Return a labeled article field only when the input condition requests it."""
    if required_token not in input_condition:
        return ""
    return f"{label}{'' if value is None else value}"


def prompt_func_individual(data: Mapping[str, Any]) -> list[Any]:
    """Build the system and human messages for an individual article judgment."""
    article = data["article_info"]
    dimension_config = dimensions[data["dimension"]]
    input_condition = data["input_condition"]

    prompt_values = {
        "CONSTRUCT_NAME": dimension_config["CONSTRUCT_NAME"],
        "CONSTRUCT_DEFINITION": dimension_config["CONSTRUCT_DEFINITION"],
        "INPUT_CONDITION": input_conditions[input_condition],
        "ARTICLE_HEADLINE": optional_article_field(
            input_condition, "headline", "Headline: ", article.get("extracted_headline", "")
        ),
        "ARTICLE_SUMMARY": optional_article_field(
            input_condition, "text", "Text:\n", article.get("extracted_body_text", "")
        ),
        "ARTICLE_SOURCE": optional_article_field(
            input_condition, "source", "Source: ", article.get("source", "")
        ),
        "ARTICLE_LEANING": optional_article_field(
            input_condition, "label", "Political leaning: ", article.get("leaning", "")
        ),
        "SCALE_ANCHORS": dimension_config["SCALE_ANCHORS"],
    }

    content: list[dict[str, str]] = [
        {"type": "text", "text": data["prompt_start"].format(**prompt_values)}
    ]

    if "image" in input_condition:
        add_article_images(content, article, data["base_dir"])

    content.append(
        {"type": "text", "text": data["prompt_end"].format(**prompt_values)}
    )

    return [
        SystemMessage(content=data["system_prompt"]),
        HumanMessage(content=content),
    ]


def prompt_func_multiple(data: Mapping[str, Any]) -> list[Any]:
    """Build messages for pairwise, triadic, or ideology-triadic judgments."""
    dimension_key = data["dimension"]
    input_condition = data["input_condition"]

    prompt_values = {
        "CONSTRUCT_NAME": ""
        if dimension_key is None
        else dimensions[dimension_key]["CONSTRUCT_NAME"],
        "CONSTRUCT_DEFINITION": ""
        if dimension_key is None
        else dimensions[dimension_key]["CONSTRUCT_DEFINITION"],
        "INPUT_CONDITION": input_conditions[input_condition],
    }

    content: list[dict[str, str]] = [
        {"type": "text", "text": data["prompt_start"].format(**prompt_values)}
    ]

    for article in data["article_info"][: data["arts_to_compare"]]:
        article_values = {
            "ARTICLE_LABEL": article["art_label"],
            "ARTICLE_HEADLINE": optional_article_field(
                input_condition,
                "headline",
                "Headline: ",
                article.get("extracted_headline", ""),
            ),
            "ARTICLE_SUMMARY": optional_article_field(
                input_condition,
                "text",
                "Text:\n",
                article.get("extracted_body_text", ""),
            ),
            "ARTICLE_SOURCE": optional_article_field(
                input_condition, "source", "Source: ", article.get("source", "")
            ),
            "ARTICLE_LEANING": optional_article_field(
                input_condition,
                "label",
                "Political leaning: ",
                article.get("leaning", ""),
            ),
        }

        content.append(
            {
                "type": "text",
                "text": data["prompt_articles"].format(**article_values),
            }
        )

        if "image" in input_condition:
            add_article_images(content, article, data["base_dir"])

    content.append({"type": "text", "text": data["prompt_end"].format()})

    return [
        SystemMessage(content=data["system_prompt"]),
        HumanMessage(content=content),
    ]


# This registry keeps prompt templates and prompt-building functions together.
experiments = {
    "individual": {
        "prompt_building": prompt_func_individual,
        "prompt_start": PROMPT_INDIVIDUAL_EVALUATION_START,
        "prompt_end": PROMPT_INDIVIDUAL_EVALUATION_END,
    },
    "individual-metadata": {
        "prompt_building": prompt_func_individual,
        "prompt_start": PROMPT_INDIVIDUAL_EVALUATION_START,
        "prompt_end": PROMPT_INDIVIDUAL_EVALUATION_END,
    },
    "triadic": {
        "prompt_building": prompt_func_multiple,
        "prompt_start": PROMPT_TRIADIC_EVALUATION_START,
        "prompt_end": PROMPT_TRIADIC_EVALUATION_END,
    },
    "pairs": {
        "prompt_building": prompt_func_multiple,
        "prompt_start": PROMPT_PAIRS_EVALUATION_START,
        "prompt_end": PROMPT_PAIRS_EVALUATION_END,
    },
    "ideology_triadic": {
        "prompt_building": prompt_func_multiple,
        "prompt_start": PROMPT_IDEOLOGY_TRIADIC_EVALUATION_START,
        "prompt_end": PROMPT_IDEOLOGY_TRIADIC_EVALUATION_END,
    },
}


def generate(
    model: ChatOllama,
    article: Any,
    dimension: str | None,
    input_condition: str,
    experiment: str,
    base_dir: Path,
) -> Any:
    """Build a prompt for one experiment and invoke the selected Ollama model."""
    experiment_config = experiments[experiment]
    chain = experiment_config["prompt_building"] | model

    article_count = 1 if "individual" in experiment else 2 if experiment == "pairs" else 3

    return chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT,
            "prompt_start": experiment_config["prompt_start"],
            "prompt_articles": PROMPT_TRIADIC_EVALUATION_ARTICLE,
            "prompt_end": experiment_config["prompt_end"],
            "article_info": article,
            "dimension": dimension,
            "input_condition": input_condition,
            "arts_to_compare": article_count,
            "base_dir": base_dir,
        }
    )


# ================================================================================================
# Dataset and article selection helpers
# ================================================================================================


def normalize_image_paths(raw_images: Any) -> list[str]:
    """Extract non-empty, non-GIF local image paths from an article record."""
    if not isinstance(raw_images, list):
        return []

    paths = [image.get("local_path", "") for image in raw_images if isinstance(image, Mapping)]
    return [path for path in paths if path and not path.lower().endswith(".gif")]


def find_full_article(
    article_reference: Mapping[str, Any],
    full_articles: pd.DataFrame,
) -> dict[str, Any] | None:
    """Look up one AllSides article reference in the extracted full-article table."""
    link = article_reference.get("link")
    if not link:
        return None

    matches = full_articles[full_articles["url"] == link]
    if matches.empty:
        return None

    row = matches.iloc[0]
    return {
        "extracted_headline": row.get("extracted_headline", ""),
        "extracted_body_text": row.get("extracted_body_text", ""),
        "link": row.get("url", link),
        "title": row.get("title", ""),
        "leaning": row.get("leaning", ""),
        "local_path": normalize_image_paths(row.get("extracted_images")),
        "source": row.get("domain", ""),
    }


def extract_article_leaning(
    event: pd.Series,
    leaning: str,
    fallback_column: str,
    include_more: bool,
    full_articles: pd.DataFrame,
) -> dict[str, Any] | None:
    """Select a usable article for one political leaning.

    The primary article is tried first. With ``--include-more``, fallback
    candidates from columns such as ``more_left`` are tried in dataset order.
    Every selected article must include at least one usable image. This
    eligibility rule is intentionally independent of the prompt input condition
    so text-only and image-based experiments use comparable article samples.
    """
    candidates: list[Mapping[str, Any]] = []

    primary = event.get(leaning)
    if isinstance(primary, Mapping):
        candidates.append(primary)

    if include_more:
        fallbacks = event.get(fallback_column, [])
        if isinstance(fallbacks, list):
            candidates.extend(item for item in fallbacks if isinstance(item, Mapping))

    for candidate in candidates:
        # The mapping acts as a source allow-list. Its filename values are kept
        # for compatibility with the original dataset configuration.
        if candidate.get("source") not in mapping_sources:
            continue

        article = find_full_article(candidate, full_articles)
        if article is None:
            continue

        # Keep article eligibility constant across all experiments. The image
        # does not have to be shown to the model, but it must exist in the data.
        if not article["local_path"]:
            continue

        return article

    return None


def load_event_dataset(base_dir: Path) -> pd.DataFrame:
    """Load the split or combined AllSides event dataset."""
    first_split = base_dir / "allsides_Jan-May_2025.jsonl"
    second_split = base_dir / "allsides_Jun2025_May2026.jsonl"
    combined = base_dir / "allsides_Jan2025_May2026_combined.jsonl"

    if first_split.exists() or second_split.exists():
        missing = [str(path.name) for path in (first_split, second_split) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "The split dataset is incomplete. Missing: " + ", ".join(missing)
            )

        frames = [
            pd.read_json(first_split, lines=True),
            pd.read_json(second_split, lines=True),
        ]
        return pd.concat(frames, ignore_index=True)

    if combined.exists():
        return pd.read_json(combined, lines=True)

    raise FileNotFoundError(
        "No AllSides event dataset was found. Expected the two split JSONL "
        f"files or {combined.name} inside {base_dir}."
    )


def load_full_articles(base_dir: Path) -> pd.DataFrame:
    """Flatten every JSON file in ``full_articles/`` into one DataFrame."""
    articles_dir = base_dir / "full_articles"
    if not articles_dir.is_dir():
        raise FileNotFoundError(f"Required directory not found: {articles_dir}")

    json_files = sorted(articles_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {articles_dir}")

    rows: list[dict[str, Any]] = []
    for json_file in tqdm(json_files, desc="Loading full-article files"):
        with json_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for title, leanings in data.items():
            if not isinstance(leanings, Mapping):
                continue
            for leaning, article in leanings.items():
                if not isinstance(article, Mapping):
                    continue
                rows.append({"title": title, "leaning": leaning, **article})

    if not rows:
        raise ValueError(f"The JSON files in {articles_dir} contained no article records.")

    return pd.DataFrame(rows)


# ================================================================================================
# Results, checkpoints, and experiment runners
# ================================================================================================


def backup_path_for(output_file: Path) -> Path:
    """Return the conventional backup filename used by this script."""
    return output_file.with_name(f"{output_file.stem}__backup{output_file.suffix}")


def load_existing_results(output_file: Path) -> dict[str, Any]:
    """Load prior results and create a backup before the run resumes."""
    if not output_file.exists():
        return {}

    backup_file = backup_path_for(output_file)
    shutil.copy2(output_file, backup_file)

    try:
        with output_file.open("rb") as handle:
            return pickle.load(handle)
    except (EOFError, pickle.UnpicklingError):
        if backup_file.exists():
            print(f"Warning: {output_file.name} is unreadable; using its backup.")
            with backup_file.open("rb") as handle:
                return pickle.load(handle)
        raise


def save_results(results: Mapping[str, Any], output_file: Path) -> None:
    """Atomically checkpoint results to reduce corruption after interruptions."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")

    with temporary_file.open("wb") as handle:
        pickle.dump(dict(results), handle)

    temporary_file.replace(output_file)


def run_analysis_articles_individual(
    events: pd.DataFrame,
    model: ChatOllama,
    model_name: str,
    dimension: str,
    input_condition: str,
    results: MutableMapping[str, Any],
    experiment: str,
    include_more: bool,
    full_articles: pd.DataFrame,
    base_dir: Path,
    output_file: Path,
) -> None:
    """Run one independent judgment for each available leaning and event."""

    for row_index, event in tqdm(
        events.iterrows(),
        total=len(events),
        desc=f"{experiment}: {dimension} / {input_condition}",
    ):
        event_key = str(event.get("headline", row_index))
        event_results = results.setdefault(event_key, {})

        for leaning, fallback_column in LEANING_COLUMNS:
            # Resume safely: do not query a model for a result already present.
            if leaning in event_results:
                continue

            article = extract_article_leaning(
                event,
                leaning,
                fallback_column,
                include_more,
                full_articles,
            )
            if article is None:
                print(
                    f"Warning: event row {row_index} has no usable {leaning} article "
                    f"with at least one image for input condition '{input_condition}'."
                )
                continue

            try:
                response = generate(
                    model,
                    article,
                    dimension,
                    input_condition,
                    experiment,
                    base_dir,
                )
            except ResponseError as error:
                print(
                    f"Ollama error for row {row_index}, {leaning}, model {model_name} "
                    f"({error.status_code}): {error}"
                )
                continue

            # Store the selected article link. This matters when a fallback article
            # from a ``more_*`` column was used instead of the primary article.
            event_results[leaning] = {
                "link": article["link"],
                "response": response,
            }

        save_results(results, output_file)


def collect_event_articles(
    event: pd.Series,
    include_more: bool,
    full_articles: pd.DataFrame,
) -> list[dict[str, Any]] | None:
    """Collect and shuffle one left, center, and right article for an event."""
    shuffled_leanings = list(LEANING_COLUMNS)
    random.shuffle(shuffled_leanings)

    articles: list[dict[str, Any]] = []
    for leaning, fallback_column in shuffled_leanings:
        article = extract_article_leaning(
            event,
            leaning,
            fallback_column,
            include_more,
            full_articles,
        )
        if article is None:
            return None
        articles.append(article)

    return articles


def label_articles(articles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return copies of the articles labeled A, B, C in their current order."""
    labels = "ABC"
    return [
        {**dict(article), "art_label": labels[index]}
        for index, article in enumerate(articles)
    ]


def run_analysis_articles_multiple(
    events: pd.DataFrame,
    model: ChatOllama,
    model_name: str,
    dimension: str | None,
    input_condition: str,
    results: MutableMapping[str, Any],
    experiment: str,
    include_more: bool,
    full_articles: pd.DataFrame,
    base_dir: Path,
    output_file: Path,
) -> None:
    """Run pairwise, triadic, or ideology-triadic comparisons."""

    for row_index, event in tqdm(
        events.iterrows(),
        total=len(events),
        desc=f"{experiment}: {dimension or 'ideology'} / {input_condition}",
    ):
        event_key = str(event.get("headline", row_index))
        if event_key in results:
            continue

        articles = collect_event_articles(event, include_more, full_articles)
        if articles is None:
            print(
                f"Warning: event row {row_index} does not have three usable articles "
                f"with images for input condition '{input_condition}'."
            )
            continue

        responses: list[dict[str, Any]] = []

        if experiment in {"triadic", "ideology_triadic"}:
            labeled = label_articles(articles)
            try:
                response = generate(
                    model,
                    labeled,
                    dimension,
                    input_condition,
                    experiment,
                    base_dir,
                )
                responses.append(
                    {
                        "article_A": labeled[0]["link"],
                        "article_B": labeled[1]["link"],
                        "article_C": labeled[2]["link"],
                        "leaning_A": labeled[0]["leaning"],
                        "leaning_B": labeled[1]["leaning"],
                        "leaning_C": labeled[2]["leaning"],
                        "response": response,
                    }
                )
            except ResponseError as error:
                print(
                    f"Ollama error for row {row_index}, model {model_name} "
                    f"({error.status_code}): {error}"
                )

        elif experiment == "pairs":
            for first_index, second_index in PAIR_INDEXES:
                pair = label_articles(
                    [articles[first_index], articles[second_index]]
                )
                try:
                    response = generate(
                        model,
                        pair,
                        dimension,
                        input_condition,
                        experiment,
                        base_dir,
                    )
                    responses.append(
                        {
                            "article_A": pair[0]["link"],
                            "article_B": pair[1]["link"],
                            "leaning_A": pair[0]["leaning"],
                            "leaning_B": pair[1]["leaning"],
                            "response": response,
                        }
                    )
                except ResponseError as error:
                    print(
                        f"Ollama error for row {row_index}, pair "
                        f"({first_index}, {second_index}), model {model_name} "
                        f"({error.status_code}): {error}"
                    )

        if responses:
            results[event_key] = responses

        save_results(results, output_file)


def define_llm_models(requested_model: str | None) -> dict[str, ChatOllama]:
    """Create deterministic ChatOllama clients keyed by filename-safe names."""
    model_names = (requested_model,) if requested_model else DEFAULT_MODELS
    return {
        model_name.replace(":", ""): ChatOllama(
            model=model_name,
            temperature=0,
            think=False,
        )
        for model_name in model_names
    }


# ================================================================================================
# Command-line interface
# ================================================================================================


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser and its user-facing help text."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge news-framing experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One framing dimension using headline and article text
  python run-judges-frame-commented.py --dir-path ./data \\
      --experiment individual --dimension emotional-intensity \\
      --input-condition headline-text --llm gemma4:e2b

  # All configured individual experiments
  python run-judges-frame-commented.py --dir-path ./data --experiment individual

  # Pairwise image comparisons, allowing fallback articles
  python run-judges-frame-commented.py --dir-path ./data \\
      --experiment pairs --input-condition image --include-more
""",
    )

    parser.add_argument(
        "--dir-path",
        "--dir_path",
        dest="dir_path",
        type=Path,
        default=Path("."),
        help="Directory containing datasets, full_articles/, images, and results.",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "Exact local Ollama model name, for example 'gemma4:e2b'. "
            "When omitted, DEFAULT_MODELS is used."
        ),
    )
    parser.add_argument(
        "--experiment",
        choices=[*experiments.keys(), "all"],
        default="individual",
        help="Experiment to run. 'all' runs every configured experiment.",
    )
    parser.add_argument(
        "--dimension",
        choices=list(dimensions.keys()),
        default=None,
        help="Framing dimension. Omit it to run every configured dimension.",
    )
    parser.add_argument(
        "--input-condition",
        "--input_condition",
        dest="input_condition",
        choices=list(input_conditions.keys()),
        default=None,
        help="Information supplied to the judge. Omit it to run compatible conditions.",
    )
    parser.add_argument(
        "--include-more",
        "--include_more",
        dest="include_more",
        action="store_true",
        help="Try fallback candidates from more_left/more_center/more_right.",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="Print configured experiments, dimensions, and input conditions, then exit.",
    )

    return parser


def print_available_options() -> None:
    """Print selectable configuration values in a compact, readable form."""
    print("Experiments:")
    for name in [*experiments.keys(), "all"]:
        print(f"  - {name}")

    print("\nDimensions:")
    for name in dimensions:
        print(f"  - {name}")

    print("\nInput conditions:")
    for name in input_conditions:
        print(f"  - {name}")


def compatible_input_conditions(
    experiment: str,
    requested_condition: str | None,
) -> list[str]:
    """Return the requested condition or all conditions compatible with an experiment."""
    if requested_condition is not None:
        return [requested_condition]

    uses_metadata = "metadata" in experiment
    return [
        condition
        for condition in input_conditions
        if (
            uses_metadata and ("source" in condition or "label" in condition)
        )
        or (
            not uses_metadata and "source" not in condition and "label" not in condition
        )
    ]


def result_file_path(
    base_dir: Path,
    experiment: str,
    input_condition: str,
    model_name: str,
    dimension: str | None,
) -> Path:
    """Build a consistent output filename for one experiment configuration."""
    if dimension is None:
        filename = f"results_{experiment}_{input_condition}_{model_name}.pickle"
    else:
        filename = (
            f"results_{experiment}_{input_condition}_{dimension}_{model_name}.pickle"
        )
    return base_dir / filename


def run_experiment(
    experiment: str,
    args: argparse.Namespace,
    events: pd.DataFrame,
    full_articles: pd.DataFrame,
    models_by_name: Mapping[str, ChatOllama],
    base_dir: Path,
) -> None:
    """Run every selected model/dimension/input-condition combination."""
    selected_conditions = compatible_input_conditions(
        experiment, args.input_condition
    )
    selected_dimensions: list[str | None]
    if experiment == "ideology_triadic":
        selected_dimensions = [None]
    else:
        selected_dimensions = (
            [args.dimension] if args.dimension else list(dimensions.keys())
        )

    print(f"\nExperiment: {experiment}")
    print(f"Dimensions: {[value or 'not applicable' for value in selected_dimensions]}")
    print(f"Input conditions: {selected_conditions}")

    for model_name, model in models_by_name.items():
        print(f"\nModel: {model_name}")

        for input_condition in selected_conditions:
            for dimension in selected_dimensions:
                output_file = result_file_path(
                    base_dir,
                    experiment,
                    input_condition,
                    model_name,
                    dimension,
                )
                results = load_existing_results(output_file)

                print(
                    f"Starting {experiment} | {input_condition} | "
                    f"{dimension or 'ideology'} | resumed events: {len(results)}"
                )

                if experiment.startswith("individual"):
                    assert dimension is not None
                    run_analysis_articles_individual(
                        events,
                        model,
                        model_name,
                        dimension,
                        input_condition,
                        results,
                        experiment,
                        args.include_more,
                        full_articles,
                        base_dir,
                        output_file,
                    )
                else:
                    run_analysis_articles_multiple(
                        events,
                        model,
                        model_name,
                        dimension,
                        input_condition,
                        results,
                        experiment,
                        args.include_more,
                        full_articles,
                        base_dir,
                        output_file,
                    )

                print(f"Saved {len(results)} event result(s) to {output_file}")


def main() -> None:
    """Parse arguments, load data once, and run the selected experiments."""
    parser = build_parser()
    args = parser.parse_args()

    if args.list_options:
        print_available_options()
        return

    base_dir = args.dir_path.expanduser().resolve()
    if not base_dir.is_dir():
        parser.error(f"--dir-path is not a directory: {base_dir}")

    # Load shared datasets only once, even when several experiments are run.
    try:
        events = load_event_dataset(base_dir)
        full_articles = load_full_articles(base_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(f"Loaded {len(events):,} event rows.")
    print(f"Loaded {len(full_articles):,} full-article rows.")

    try:
        load_llm_dependencies()
    except RuntimeError as error:
        parser.error(str(error))

    models_by_name = define_llm_models(args.llm)
    selected_experiments = (
        list(experiments.keys()) if args.experiment == "all" else [args.experiment]
    )

    for experiment in selected_experiments:
        run_experiment(
            experiment,
            args,
            events,
            full_articles,
            models_by_name,
            base_dir,
        )


if __name__ == "__main__":
    main()
