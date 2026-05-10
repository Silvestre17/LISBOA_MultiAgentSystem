# ==========================================================================
# Master Thesis - Geographic Scope Utilities
#   - André Filipe Gomes Silvestre, 20240502
#
#   Shared helpers for keeping LISBOA responses inside the supported
#   Lisbon Metropolitan Area scope.
# ==========================================================================

import re
import unicodedata
from typing import Iterable, List, Tuple


OUTSIDE_AML_ROUTE_PLACES: Tuple[Tuple[str, str], ...] = (
    ("torres vedras", "Torres Vedras"),
    ("porto", "Porto"),
    ("coimbra", "Coimbra"),
    ("aveiro", "Aveiro"),
    ("braga", "Braga"),
    ("guimaraes", "Guimarães"),
    ("guimarães", "Guimarães"),
    ("faro", "Faro"),
    ("algarve", "Algarve"),
    ("evora", "Évora"),
    ("évora", "Évora"),
    ("leiria", "Leiria"),
    ("santarem", "Santarém"),
    ("santarém", "Santarém"),
    ("viana do castelo", "Viana do Castelo"),
    ("vila real", "Vila Real"),
    ("braganca", "Bragança"),
    ("bragança", "Bragança"),
    ("viseu", "Viseu"),
    ("guarda", "Guarda"),
    ("castelo branco", "Castelo Branco"),
    ("beja", "Beja"),
    ("madrid", "Madrid"),
    ("barcelona", "Barcelona"),
    ("paris", "Paris"),
    ("londres", "Londres"),
    ("london", "London"),
    ("mexico", "México"),
    ("méxico", "México"),
    ("new york", "New York"),
    ("roma", "Roma"),
    ("rome", "Rome"),
    ("berlim", "Berlim"),
    ("berlin", "Berlin"),
    ("narnia", "Narnia"),
)

AML_AMBIGUOUS_PLACE_EXCLUSIONS: Tuple[str, ...] = (
    "porto salvo",
    "porto brandao",
    "porto brandão",
    "avenida do mexico",
    "avenida do méxico",
    "av do mexico",
    "av do méxico",
    "av. do mexico",
    "av. do méxico",
)


def normalize_scope_text(text: str) -> str:
    """Return a lowercase accent-stripped string for geographic matching."""
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"[!?.,;:]+", " ", normalized.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def extract_outside_aml_mentions(text: str) -> List[str]:
    """Return outside-AML place labels explicitly mentioned in text.

    The matcher is conservative and protects known AML expressions that contain
    an otherwise out-of-scope token, such as Porto Salvo or Avenida do México.
    """
    normalized = f" {normalize_scope_text(text)} "
    if not normalized.strip():
        return []

    protected = normalized
    for exclusion in AML_AMBIGUOUS_PLACE_EXCLUSIONS:
        protected = re.sub(
            rf"\b{re.escape(normalize_scope_text(exclusion))}\b",
            " ",
            protected,
        )

    found: List[Tuple[int, str]] = []
    for place, label in OUTSIDE_AML_ROUTE_PLACES:
        needle = normalize_scope_text(place)
        match = re.search(rf"\b{re.escape(needle)}\b", protected)
        if match and label not in [existing_label for _index, existing_label in found]:
            found.append((match.start(), label))
    labels: List[str] = [label for _index, label in sorted(found, key=lambda item: item[0])]
    return labels


def route_mentions_outside_aml(text: str) -> bool:
    """Return whether a route-like text mentions an outside-AML destination."""
    return bool(extract_outside_aml_mentions(text))


def join_scope_labels(labels: Iterable[str], language: str = "pt") -> str:
    """Join place labels for user-facing scope messages."""
    values = [str(label).strip() for label in labels if str(label).strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    conjunction = " e " if language == "pt" else " and "
    return ", ".join(values[:-1]) + conjunction + values[-1]


def build_geographic_out_of_scope_response(
    user_message: str,
    language: str = "pt",
) -> str:
    """Build a friendly route-scope response for outside-AML journeys."""
    labels = extract_outside_aml_mentions(user_message)
    place_text = join_scope_labels(labels, language=language)
    is_pt = language == "pt"

    if is_pt:
        direct = (
            f"não consigo validar uma rota para **{place_text}** com os dados confirmáveis deste sistema."
            if place_text
            else "essa rota sai do âmbito geográfico que consigo validar com qualidade neste sistema."
        )
        return "\n".join(
            [
                "### 🧭 **Fora do Âmbito de Mobilidade do LISBOA**",
                "",
                f"✅ **Resposta direta:** {direct}",
                "",
                "---",
                "",
                "O LISBOA está focado na **Área Metropolitana de Lisboa** e valida diretamente Metro, Carris Urban, Carris Metropolitana e CP suburbano/AML.",
                "",
                "💡 **Posso ajudar com:**",
                "- **Metro:** rotas dentro de Lisboa 🚇",
                "- **Autocarros:** Carris Urban e Carris Metropolitana na AML 🚌",
                "- **Comboios:** CP suburbano/AML suportado 🚆",
                "- **Percursos:** planos dentro de Lisboa/AML 🗺️",
            ]
        ).strip()

    direct = (
        f"I cannot validate a route to **{place_text}** with this system's confirmable data."
        if place_text
        else "that route is outside the geographic scope I can validate with quality in this system."
    )
    return "\n".join(
        [
            "### 🧭 **Outside LISBOA's Mobility Scope**",
            "",
            f"✅ **Direct answer:** {direct}",
            "",
            "---",
            "",
            "LISBOA is focused on the **Lisbon Metropolitan Area** and directly validates Metro, Carris Urban, Carris Metropolitana, and CP suburban/AML services.",
            "",
            "💡 **I can help with:**",
            "- **Metro:** routes within Lisbon 🚇",
            "- **Buses:** Carris Urban and Carris Metropolitana in the AML 🚌",
            "- **Trains:** supported CP suburban/AML rail 🚆",
            "- **Routes:** plans inside Lisbon/AML 🗺️",
        ]
    ).strip()
