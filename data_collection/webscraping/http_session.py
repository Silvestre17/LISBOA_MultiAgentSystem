# ==========================================================================
# Master Thesis - Shared HTTP Session for the VisitLisboa Scrapers
#   - André Filipe Gomes Silvestre, 20240502
#
# Description:
#   Builds a requests session that retries transient failures (read and
#   connect timeouts, 429 and 5xx responses) with exponential backoff, so a
#   single slow response from visitlisboa.com does not abort a scraper run.
# ==========================================================================

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MAX_RETRIES = 3
BACKOFF_FACTOR_SECONDS = 10
RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


def create_session() -> requests.Session:
    """Create a requests session with retries for transient errors.

    Returns:
        requests.Session: Session whose GET requests are retried up to
        ``MAX_RETRIES`` times: the first retry is immediate, then urllib3
        waits 20 s and 40 s.
    """
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR_SECONDS,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
