# ==========================================================================
# Master Thesis - Geographic Scope Utilities
#   - Andre Filipe Gomes Silvestre, 20240502
#
#   Shared helpers for keeping LISBOA responses inside the supported
#   Lisbon Metropolitan Area scope.
# ==========================================================================

import re
import unicodedata
from typing import Iterable, List, Tuple


AML_MUNICIPALITY_NAMES: Tuple[str, ...] = (
    "Alcochete",
    "Almada",
    "Amadora",
    "Barreiro",
    "Cascais",
    "Lisboa",
    "Loures",
    "Mafra",
    "Moita",
    "Montijo",
    "Odivelas",
    "Oeiras",
    "Palmela",
    "Seixal",
    "Sesimbra",
    "Setúbal",
    "Sintra",
    "Vila Franca de Xira",
)

# Approximate municipality-centre points used only as broad fallback anchors
# when a user names a municipality and live geocoding is unavailable.
AML_MUNICIPALITY_CENTROIDS: dict[str, tuple[float, float]] = {
    "Alcochete": (38.7553, -8.9609),
    "Almada": (38.6765, -9.1651),
    "Amadora": (38.7597, -9.2397),
    "Barreiro": (38.6631, -9.0724),
    "Cascais": (38.6979, -9.4215),
    "Lisboa": (38.7223, -9.1393),
    "Loures": (38.8309, -9.1685),
    "Mafra": (38.9379, -9.3276),
    "Moita": (38.6508, -8.9904),
    "Montijo": (38.7067, -8.9739),
    "Odivelas": (38.7927, -9.1838),
    "Oeiras": (38.6971, -9.3017),
    "Palmela": (38.5690, -8.9013),
    "Seixal": (38.6400, -9.1015),
    "Sesimbra": (38.4445, -9.1015),
    "Setúbal": (38.5244, -8.8882),
    "Sintra": (38.7989, -9.3869),
    "Vila Franca de Xira": (38.9553, -8.9897),
}

AML_MUNICIPALITY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("lisbon", "Lisboa"),
    ("setubal", "Setúbal"),
    ("setúbal", "Setúbal"),
    ("vila franca", "Vila Franca de Xira"),
    ("vila franca de xira", "Vila Franca de Xira"),
    *tuple((name, name) for name in AML_MUNICIPALITY_NAMES),
)

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
    "avenida de madrid",
    "av de madrid",
    "av. de madrid",
    "rua de madrid",
    "praca de londres",
    "praça de londres",
    "avenida de londres",
    "av de londres",
    "av. de londres",
    "avenida de paris",
    "av de paris",
    "av. de paris",
    "avenida de berlim",
    "av de berlim",
    "av. de berlim",
    "avenida de roma",
    "av de roma",
    "av. de roma",
    "estacao roma",
    "estação roma",
    "roma areeiro",
    "roma-areeiro",
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


def extract_aml_municipality_mentions(text: str) -> List[str]:
    """Return official AML municipality labels explicitly mentioned in text."""
    normalized = f" {normalize_scope_text(text)} "
    if not normalized.strip():
        return []

    found: List[Tuple[int, str]] = []
    seen: set[str] = set()
    for alias, label in AML_MUNICIPALITY_ALIASES:
        needle = normalize_scope_text(alias)
        match = re.search(rf"\b{re.escape(needle)}\b", normalized)
        if match and label not in seen:
            seen.add(label)
            found.append((match.start(), label))
    return [label for _index, label in sorted(found, key=lambda item: item[0])]


def mentions_aml_municipality(text: str) -> bool:
    """Return whether text explicitly mentions at least one AML municipality."""
    return bool(extract_aml_municipality_mentions(text))


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
    seen: set[str] = set()
    for place, label in OUTSIDE_AML_ROUTE_PLACES:
        needle = normalize_scope_text(place)
        match = re.search(rf"\b{re.escape(needle)}\b", protected)
        if match and label not in seen:
            seen.add(label)
            found.append((match.start(), label))
    return [label for _index, label in sorted(found, key=lambda item: item[0])]


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
    *,
    mobility: bool = True,
) -> str:
    """Build a friendly geographic-scope response for outside-AML requests."""
    labels = extract_outside_aml_mentions(user_message)
    place_text = join_scope_labels(labels, language=language)
    is_pt = language == "pt"

    if is_pt:
        if mobility:
            title = "### 🧭 **Fora do Âmbito de Mobilidade do LISBOA**"
            direct = (
                f"não consigo validar uma rota para **{place_text}** com os dados confirmáveis deste sistema."
                if place_text
                else "essa rota sai do âmbito geográfico que consigo validar com qualidade neste sistema."
            )
        else:
            title = "### 🧭 **Fora do Âmbito Geográfico do LISBOA**"
            direct = (
                f"não consigo validar pedidos para **{place_text}** com os dados confirmáveis deste sistema."
                if place_text
                else "esse pedido sai do âmbito geográfico que consigo validar com qualidade neste sistema."
            )
        return "\n".join(
            [
                title,
                "",
                f"✅ **Resposta direta:** {direct}",
                "",
                "---",
                "",
                "O LISBOA está focado na **Área Metropolitana de Lisboa** e só deve responder com detalhe quando consegue validar a informação com as fontes disponíveis.",
                "",
                "💡 **Posso ajudar com:**",
                "- **Meteorologia:** previsão e avisos para Lisboa 🌤️",
                "- **Mobilidade:** Metro, Carris Urban, Carris Metropolitana e CP suburbano/AML 🚇",
                "- **Cultura e locais:** eventos, atrações, restaurantes e serviços na AML 📍",
                "- **Planeamento:** roteiros e percursos dentro de Lisboa/AML 🗺️",
            ]
        ).strip()

    if mobility:
        title = "### 🧭 **Outside LISBOA's Mobility Scope**"
        direct = (
            f"I cannot validate a route to **{place_text}** with this system's confirmable data."
            if place_text
            else "that route is outside the geographic scope I can validate with quality in this system."
        )
    else:
        title = "### 🧭 **Outside LISBOA's Geographic Scope**"
        direct = (
            f"I cannot validate requests for **{place_text}** with this system's confirmable data."
            if place_text
            else "that request is outside the geographic scope I can validate with quality in this system."
        )
    return "\n".join(
        [
            title,
            "",
            f"✅ **Direct answer:** {direct}",
            "",
            "---",
            "",
            "LISBOA is focused on the **Lisbon Metropolitan Area** and should only answer in detail when it can validate the information with available sources.",
            "",
            "💡 **I can help with:**",
            "- **Weather:** Lisbon forecasts and warnings 🌤️",
            "- **Mobility:** Metro, Carris Urban, Carris Metropolitana, and CP suburban/AML 🚇",
            "- **Culture and places:** events, attractions, restaurants, and services in the AML 📍",
            "- **Planning:** itineraries and routes inside Lisbon/AML 🗺️",
        ]
    ).strip()
