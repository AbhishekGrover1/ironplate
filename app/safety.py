"""Safety checks that run before any intent handling.

Emergencies, self-harm language and disordered-eating language never get calorie
targets or plans. They get a short, supportive reply that points to real help.
"""
from __future__ import annotations

import re
from typing import Optional

CRISIS = re.compile(
    r"\b(kill myself|end my life|ending my life|suicid\w*|want to die|self[- ]?harm|hurt myself|no reason to live|better off dead)\b")
MEDICAL = re.compile(
    r"\b(chest pain|can'?t breathe|cannot breathe|trouble breathing|fainted|passed out|heart attack|severe pain|blood in (?:my )?(?:stool|urine|vomit)|overdos\w*)\b")
EATING = re.compile(
    r"\b(anorexi\w*|bulimi\w*|purg(?:e|ing)\b|starv(?:e|ing) myself|laxatives? (?:to|for)|make myself (?:throw up|vomit)|throw(?:ing)? up (?:after|to)|"
    r"binge(?:d|s|ing)?\b(?!\s*watch)|eat nothing|not eat (?:at all|anything)|stop eating|(?:under|below|less than) 800 calories|"
    r"(?:500|600|700) calories a day|hate my body|fast for (?:days|a week))\b")
PREGNANCY = re.compile(r"\b(pregnan\w*|breastfeeding|breast feeding|nursing mother|lactating|trying to conceive)\b")
MOOD = re.compile(
    r"\b(?:i(?:'m| am| feel)\s+(?:so\s+|really\s+)?(?:sad|depressed|anxious|lonely|hopeless|worthless|stressed|overwhelmed|miserable)|feeling (?:sad|depressed|down|low|hopeless|anxious|lonely))\b")

REPLIES = {
    "crisis": ("I'm really sorry you're going through this. You deserve support right now. "
               "If you might act on these thoughts or you are in danger, please call your local emergency number "
               "or a crisis line now, or reach out to someone you trust and stay with them. "
               "I'm only a nutrition and training assistant, so I can't help with this the way a person can."),
    "medical": ("That sounds like it needs medical attention rather than a fitness plan. "
                "If it is severe or sudden, please contact emergency services or a doctor right away."),
    "eating": ("I can't set calorie targets or plans around that. What you describe sounds hard, and a doctor or a "
               "registered dietitian who works with eating concerns can help. A local eating-disorder support "
               "service is another good place to start. I'm here for food logging and training whenever you want it."),
    "pregnancy": ("Calorie and macro needs change during pregnancy and breastfeeding, so I can't set targets or "
                  "meal plans for that. Please check with your doctor or a registered dietitian. "
                  "I can still help with food logging."),
    "mood": ("I'm sorry you're feeling that way. I'm only a nutrition and training assistant, so I can't offer much "
             "support, but talking to someone you trust or a health professional can really help. "
             "Whenever you're ready, I can help with food or training."),
}


def check(text: str) -> Optional[tuple[str, str]]:
    """Return (kind, reply) for messages that must not get a normal answer."""
    t = text.lower().replace("’", "'")
    for kind, rx in (("crisis", CRISIS), ("medical", MEDICAL), ("eating", EATING), ("pregnancy", PREGNANCY), ("mood", MOOD)):
        if rx.search(t):
            return kind, REPLIES[kind]
    return None
