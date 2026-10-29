from riot_api import (
    get_account_by_riot_id,
    get_recent_match_ids,
    get_match,
)

from statistics import median
from predictor import predict_rank, RANK_ORDER


RANKED_SOLO_QUEUE = 420


# ============================================================
# CONVERT RIOT PARTICIPANT -> RANKSCOPE FORMAT
# ============================================================

def convert_participant(
    participant,
    game_start_timestamp,
    game_duration,
):
    challenges = participant.get(
        "challenges",
        {}
    )

    return {
        "gameStartTimestamp":
            game_start_timestamp,

        "summonerId":
            participant["summonerId"],

        "teamPosition":
            participant["teamPosition"],

        "teamId":
            participant["teamId"],

        "gameDuration":
            game_duration,

        "kills":
            participant["kills"],

        "deaths":
            participant["deaths"],

        "assists":
            participant["assists"],

        "totalMinionsKilled":
            participant["totalMinionsKilled"],

        "neutralMinionsKilled":
            participant.get(
                "neutralMinionsKilled",
                0
            ),

        "goldEarned":
            participant["goldEarned"],

        "champExperience":
            participant["champExperience"],

        "totalDamageDealtToChampions":
            participant[
                "totalDamageDealtToChampions"
            ],

        "totalDamageTaken":
            participant["totalDamageTaken"],

        "visionScore":
            participant["visionScore"],

        "wardsPlaced":
            participant.get(
                "wardsPlaced",
                0
            ),

        "wardsKilled":
            participant.get(
                "wardsKilled",
                0
            ),

        "visionWardsBoughtInGame":
            participant.get(
                "visionWardsBoughtInGame",
                0
            ),

        "controlWardsPlaced":
            challenges.get(
                "controlWardsPlaced",
                0
            ),

        "timeCCingOthers":
            participant.get(
                "timeCCingOthers",
                0
            ),

        "laneMinionsFirst10Minutes":
            challenges.get(
                "laneMinionsFirst10Minutes",
                0
            ),

        "jungleCsBefore10Minutes":
            challenges.get(
                "jungleCsBefore10Minutes",
                0
            ),

        "turretPlatesTaken":
            challenges.get(
                "turretPlatesTaken",
                0
            ),

        "soloKills":
            challenges.get(
                "soloKills",
                0
            ),

        "turretKills":
            participant.get(
                "turretKills",
                0
            ),

        "turretTakedowns":
            challenges.get(
                "turretTakedowns",
                participant.get(
                    "turretKills",
                    0
                )
            ),

        "totalTimeSpentDead":
            participant.get(
                "totalTimeSpentDead",
                0
            ),

        "damageDealtToObjectives":
            participant.get(
                "damageDealtToObjectives",
                0
            ),

        "maxCsAdvantageOnLaneOpponent":
            challenges.get(
                "maxCsAdvantageOnLaneOpponent",
                0
            ),

        "visionScoreAdvantageLaneOpponent":
            challenges.get(
                "visionScoreAdvantageLaneOpponent",
                0
            ),

        "dragonTakedowns":
            challenges.get(
                "dragonTakedowns",
                0
            ),

        "baronTakedowns":
            challenges.get(
                "baronTakedowns",
                0
            ),

        "riftHeraldTakedowns":
            challenges.get(
                "riftHeraldTakedowns",
                0
            ),

        "dragonKills":
            challenges.get(
                "dragonKills",
                0
            ),

        "baronKills":
            challenges.get(
                "baronKills",
                0
            ),

        "objectivesStolen":
            participant.get(
                "objectivesStolen",
                0
            ),
    }


# ============================================================
# FIND PLAYER INSIDE A MATCH
# ============================================================

def find_player(
    participants,
    puuid,
):
    return next(
        (
            participant
            for participant in participants
            if participant.get("puuid") == puuid
        ),
        None,
    )


# ============================================================
# RIOT ID -> LATEST 5 RANKED SOLO MATCHES
# ============================================================

def get_recent_ranked_matches(
    game_name,
    tag_line,
    platform,
    count=5,
):
    # Riot ID -> account / PUUID
    account = get_account_by_riot_id(
        game_name,
        tag_line,
        platform,
    )

    puuid = account["puuid"]

    # riot_api.py already filters queue 420
    match_ids = get_recent_match_ids(
        puuid,
        platform,
        count=count,
    )

    if not match_ids:
        raise ValueError(
            "No recent Ranked Solo matches were found."
        )

    recent_matches = []

    for match_id in match_ids:
        match = get_match(
            match_id,
            platform,
        )

        info = match["info"]

        # Extra safety check
        if info.get("queueId") != RANKED_SOLO_QUEUE:
            continue

        participants = info.get(
            "participants",
            []
        )

        if len(participants) != 10:
            continue

        player = find_player(
            participants,
            puuid,
        )

        if player is None:
            continue

        duration_seconds = info.get(
            "gameDuration",
            0
        )

        duration_minutes = round(
            duration_seconds / 60,
            1
        )

        recent_matches.append(
            {
                "match_id":
                    match["metadata"]["matchId"],

                "win":
                    player.get("win", False),

                "result":
                    (
                        "Victory"
                        if player.get("win", False)
                        else "Defeat"
                    ),

                "role":
                    player.get(
                        "teamPosition",
                        ""
                    ),

                "kills":
                    player.get(
                        "kills",
                        0
                    ),

                "deaths":
                    player.get(
                        "deaths",
                        0
                    ),

                "assists":
                    player.get(
                        "assists",
                        0
                    ),

                "duration_minutes":
                    duration_minutes,

                "game_start_timestamp":
                    info.get(
                        "gameStartTimestamp"
                    ),
            }
        )

    if not recent_matches:
        raise ValueError(
            "No suitable Ranked Solo matches were found."
        )

    return {
        "game_name":
            account["gameName"],

        "tag_line":
            account["tagLine"],

        "platform":
            platform.upper(),

        "matches":
            recent_matches,
    }


# ============================================================
# PREDICT A SPECIFIC MATCH SELECTED BY THE USER
# ============================================================

def predict_selected_match(
    game_name,
    tag_line,
    platform,
    match_id,
):
    # Get account so we know which player
    # inside the match should be predicted.
    account = get_account_by_riot_id(
        game_name,
        tag_line,
        platform,
    )

    puuid = account["puuid"]

    # Fetch the selected match
    selected_match = get_match(
        match_id,
        platform,
    )

    info = selected_match["info"]

    # Only allow Ranked Solo
    if info.get("queueId") != RANKED_SOLO_QUEUE:
        raise ValueError(
            "The selected match is not a Ranked Solo match."
        )

    riot_participants = info.get(
        "participants",
        []
    )

    if len(riot_participants) != 10:
        raise ValueError(
            "The selected match does not contain 10 participants."
        )

    selected_participant = find_player(
        riot_participants,
        puuid,
    )

    if selected_participant is None:
        raise ValueError(
            "The selected Riot account was not found in this match."
        )

    # Convert all 10 players to RankScope format
    rank_scope_participants = [
        convert_participant(
            participant,
            info["gameStartTimestamp"],
            info["gameDuration"],
        )
        for participant
        in riot_participants
    ]

    # Run RankScope model
    result = predict_rank(
        raw_participants=
            rank_scope_participants,

        selected_player_id=
            selected_participant["summonerId"],
    )

    # Add information useful for frontend
    result["match_id"] = (
        selected_match["metadata"]["matchId"]
    )

    result["game_name"] = (
        account["gameName"]
    )

    result["tag_line"] = (
        account["tagLine"]
    )

    result["platform"] = (
        platform.upper()
    )

    return result


# ============================================================
# OLD FLOW - AUTOMATICALLY USE LATEST MATCH
# ============================================================
#
# Keep this temporarily so your current website/API
# continues working while we add match selection.
#

def predict_from_riot_id(
    game_name,
    tag_line,
    platform,
):
    account = get_account_by_riot_id(
        game_name,
        tag_line,
        platform,
    )

    puuid = account["puuid"]

    match_ids = get_recent_match_ids(
        puuid,
        platform,
        count=1,
    )

    if not match_ids:
        raise ValueError(
            "No recent Ranked Solo match was found."
        )

    return predict_selected_match(
        game_name=game_name,
        tag_line=tag_line,
        platform=platform,
        match_id=match_ids[0],
    )


# ============================================================
# OVERALL RANK PREDICTION - UP TO 20 MATCHES
# ============================================================

def predict_overall_rank(
    game_name,
    tag_line,
    platform,
    count=20,
):
    # Find account once
    account = get_account_by_riot_id(
        game_name,
        tag_line,
        platform,
    )

    puuid = account["puuid"]

    # Get recent Ranked Solo match IDs
    match_ids = get_recent_match_ids(
        puuid,
        platform,
        count=count,
    )

    if not match_ids:
        raise ValueError(
            "No recent Ranked Solo matches were found."
        )

    predictions = []
    rank_indices = []
    failed_matches = []

    for match_id in match_ids:
        try:
            # Retrieve each match once
            match = get_match(match_id, platform)
            info = match["info"]

            if info.get("queueId") != RANKED_SOLO_QUEUE:
                continue

            participants = info.get("participants", [])

            if len(participants) != 10:
                continue

            player = find_player(participants, puuid)

            if player is None:
                continue

            # Prepare all 10 participants for CatBoost
            converted = [
                convert_participant(
                    participant,
                    info["gameStartTimestamp"],
                    info["gameDuration"],
                )
                for participant in participants
            ]

            result = predict_rank(
                raw_participants=converted,
                selected_player_id=player["summonerId"],
            )

            predicted_rank = str(
                result["predicted_rank"]
            ).upper()

            if predicted_rank not in RANK_ORDER:
                raise ValueError(
                    f"Unexpected rank: {predicted_rank}"
                )

            rank_indices.append(
                RANK_ORDER.index(predicted_rank)
            )

            # Information for the frontend match cards
            predictions.append({
                "match_id": match_id,
                "predicted_rank": predicted_rank,
                "win": player.get("win", False),
                "role": player.get("teamPosition", ""),
                "kills": player.get("kills", 0),
                "deaths": player.get("deaths", 0),
                "assists": player.get("assists", 0),
                "cs": (
                    player.get("totalMinionsKilled", 0)
                    + player.get("neutralMinionsKilled", 0)
                ),
                "gold": player.get("goldEarned", 0),
                "damage": player.get(
                    "totalDamageDealtToChampions", 0
                ),
                "vision_score": player.get(
                    "visionScore", 0
                ),
                "duration_minutes": round(
                    info["gameDuration"] / 60, 1
                ),
                "game_start_timestamp": info.get(
                    "gameStartTimestamp"
                ),
            })

        except (ValueError, KeyError, TypeError) as error:
            failed_matches.append({
                "match_id": match_id,
                "error": str(error),
            })

    if not rank_indices:
        raise ValueError(
            "No matches could be successfully analysed."
        )

    # Provisional median calculation.
    # Confirm rounding against the validation notebook.
    median_index = round(median(rank_indices))
    overall_rank = RANK_ORDER[median_index]

    return {
        "player": {
            "game_name": account["gameName"],
            "tag_line": account["tagLine"],
            "platform": platform.upper(),
        },
        "overall_prediction": {
            "predicted_rank": overall_rank,
            "matches_analyzed": len(predictions),
            "method": "median",
        },
        "matches": predictions,
        "failed_matches": failed_matches,
    }
