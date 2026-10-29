
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter


# ============================================================
# LOAD RIOT API KEY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

RIOT_API_KEY = os.getenv("RIOT_API_KEY")

if not RIOT_API_KEY:
    raise RuntimeError(
        "RIOT_API_KEY was not found in backend/.env"
    )


HEADERS = {
    "X-Riot-Token": RIOT_API_KEY
}


# ============================================================
# RIOT REGION ROUTES
# ============================================================

REGION_ROUTES = {
    "EUW1": "europe",
    "EUN1": "europe",
    "TR1": "europe",
    "RU": "europe",

    "NA1": "americas",
    "BR1": "americas",
    "LA1": "americas",
    "LA2": "americas",

    "KR": "asia",
    "JP1": "asia",
}


def get_regional_route(platform: str) -> str:
    platform = platform.upper()

    if platform not in REGION_ROUTES:
        raise ValueError(
            f"Unsupported platform: {platform}"
        )

    return REGION_ROUTES[platform]


# ============================================================
# REUSABLE HTTP CONNECTION
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)

# Connection pooling, without hidden automatic retries.
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=0,
)

session.mount("https://", adapter)


# ============================================================
# HANDLE RIOT API ERRORS
# ============================================================

def check_response(response):
    if response.status_code == 404:
        raise ValueError(
            "The requested Riot resource was not found."
        )

    if response.status_code == 401:
        raise ValueError(
            "Riot API authentication failed."
        )

    if response.status_code == 403:
        raise ValueError(
            "Riot API key is invalid, expired, or lacks access."
        )

    if response.status_code == 429:
        raise ValueError(
            "Riot API rate limit reached. Try again shortly."
        )

    response.raise_for_status()


# ============================================================
# RIOT API REQUEST WITH RETRIES
# ============================================================

def riot_get(url, params=None):
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = session.get(
                url,
                params=params,
                timeout=(5, 10),
            )

            # Retry temporary server errors.
            if response.status_code in (500, 502, 503, 504):
                if attempt < max_attempts - 1:
                    print(
                        f"Riot server error {response.status_code}. "
                        f"Retrying ({attempt + 1}/{max_attempts - 1})..."
                    )

                    time.sleep(attempt + 1)
                    continue

            check_response(response)

            return response.json()

        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as error:

            if attempt == max_attempts - 1:
                raise ValueError(
                    "Could not connect to Riot Games API "
                    "after multiple attempts. "
                    "Please try again later."
                ) from error

            print(
                f"Riot API connection failed: "
                f"{type(error).__name__}. "
                f"Retrying ({attempt + 1}/{max_attempts - 1})..."
            )

            time.sleep(attempt + 1)

    raise ValueError(
        "Riot API request failed after multiple attempts."
    )


# ============================================================
# RIOT ID -> ACCOUNT / PUUID
# ============================================================

def get_account_by_riot_id(
    game_name: str,
    tag_line: str,
    platform: str,
):
    regional_route = get_regional_route(platform)

    encoded_game_name = quote(
        game_name,
        safe=""
    )

    encoded_tag_line = quote(
        tag_line,
        safe=""
    )

    url = (
        f"https://{regional_route}.api.riotgames.com"
        f"/riot/account/v1/accounts/by-riot-id/"
        f"{encoded_game_name}/{encoded_tag_line}"
    )

    try:
        return riot_get(url)

    except ValueError as error:
        if str(error) == "The requested Riot resource was not found.":
            raise ValueError("Riot ID was not found.") from error
        raise


# ============================================================
# PUUID -> RECENT RANKED SOLO MATCH IDS
# ============================================================

def get_recent_match_ids(
    puuid: str,
    platform: str,
    count: int = 5,
    queue: int = 420,
):
    regional_route = get_regional_route(platform)

    encoded_puuid = quote(
        puuid,
        safe=""
    )

    url = (
        f"https://{regional_route}.api.riotgames.com"
        f"/lol/match/v5/matches/by-puuid/"
        f"{encoded_puuid}/ids"
    )

    params = {
        "start": 0,
        "count": count,
    }

    # 420 = Ranked Solo/Duo
    if queue is not None:
        params["queue"] = queue

    return riot_get(
        url,
        params=params,
    )


# ============================================================
# MATCH ID -> FULL MATCH DATA
# ============================================================

def get_match(
    match_id: str,
    platform: str,
):
    regional_route = get_regional_route(platform)

    encoded_match_id = quote(
        match_id,
        safe=""
    )

    url = (
        f"https://{regional_route}.api.riotgames.com"
        f"/lol/match/v5/matches/{encoded_match_id}"
    )

    try:
        return riot_get(url)

    except ValueError as error:
        if str(error) == "The requested Riot resource was not found.":
            raise ValueError("Match was not found.") from error
        raise
