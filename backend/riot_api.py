import os
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv


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
# HANDLE RIOT API ERRORS
# ============================================================

def check_response(response):
    if response.status_code == 404:
        raise ValueError(
            "The requested Riot resource was not found."
        )

    if response.status_code == 403:
        raise ValueError(
            "Riot API key is invalid or expired."
        )

    if response.status_code == 429:
        raise ValueError(
            "Riot API rate limit reached. Try again shortly."
        )

    response.raise_for_status()


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

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=15,
    )

    if response.status_code == 404:
        raise ValueError(
            "Riot ID was not found."
        )

    check_response(response)

    return response.json()


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

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=15,
    )

    check_response(response)

    return response.json()


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

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=15,
    )

    if response.status_code == 404:
        raise ValueError(
            "Match was not found."
        )

    check_response(response)

    return response.json()