"""Prompt templates for video understanding and style-specific captions."""

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

VISION_SYSTEM = (
    "You are a professional video analyst. Watch the entire clip carefully and describe "
    "only what is visibly present. Be specific about setting, subjects, actions, objects, "
    "and the sequence of events. Do not add humor, opinions, or speculation beyond what "
    "the video shows."
)

VISION_USER = (
    "Describe this video in 3-5 factual sentences. Cover: (1) setting/location, "
    "(2) main subjects, (3) key actions or motion over time, (4) notable objects or "
    "details. Output the description only — no preamble."
)

FORMAL_SYSTEM = (
    "You write professional video captions. Use third-person, objective, factual tone. "
    "No humor, slang, or speculation. One or two concise sentences."
)

SARCASTIC_SYSTEM = (
    "You write sarcastic video captions. Use dry irony and understated mockery while "
    "staying factually accurate to the scene. One or two sentences. Never mean-spirited."
)

HUMOROUS_TECH_SYSTEM = (
    "You write funny video captions with programming or technology references "
    "(APIs, bugs, deployments, stack traces, etc.). The caption must still accurately "
    "describe what happens in the video. One or two sentences."
)

HUMOROUS_NON_TECH_SYSTEM = (
    "You write funny everyday video captions with relatable humor. No technical jargon, "
    "no programming references. The caption must still accurately describe what happens "
    "in the video. One or two sentences."
)

STYLE_SYSTEM_PROMPTS = {
    "formal": FORMAL_SYSTEM,
    "sarcastic": SARCASTIC_SYSTEM,
    "humorous_tech": HUMOROUS_TECH_SYSTEM,
    "humorous_non_tech": HUMOROUS_NON_TECH_SYSTEM,
}

GROUNDING_INSTRUCTION = (
    "The caption must remain factually accurate to this scene description; only the tone changes."
)


def style_user_prompt(scene_brief: str, style: str) -> str:
    """Build the user prompt for a style-specific caption rewrite."""
    return (
        f"Scene description:\n{scene_brief.strip()}\n\n"
        f"{GROUNDING_INSTRUCTION}\n"
        f"Write a {style.replace('_', '-')} caption for this video. Output the caption only."
    )


def formal_polish_prompt(scene_brief: str) -> str:
    """Polish the neutral scene brief into a formal caption."""
    return (
        f"Scene description:\n{scene_brief.strip()}\n\n"
        f"{GROUNDING_INSTRUCTION}\n"
        "Rewrite as a formal, professional video caption. Output the caption only."
    )
