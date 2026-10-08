from riot_api import (
    get_account_by_riot_id,
    get_recent_match_ids,
    get_match,
)

from predictor import predict_rank


GAME_NAME = "Dark"
TAG_LINE = "Jack"
PLATFORM = "EUW1"


def convert_participant(
    participant,
    game_start_timestamp,
    game_duration,
):
    challenges = participant.get("challenges", {})

    return {
        "gameStartTimestamp": game_start_timestamp,
        "summonerId": participant["summonerId"],
        "teamPosition": participant["teamPosition"],
        "teamId": participant["teamId"],

        "gameDuration": game_duration,

        "kills": participant["kills"],
        "deaths": participant["deaths"],
        "assists": participant["assists"],

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
            participant["totalDamageDealtToChampions"],

        "totalDamageTaken":
            participant["totalDamageTaken"],

        "visionScore":
            participant["visionScore"],

        "wardsPlaced":
            participant.get("wardsPlaced", 0),

        "wardsKilled":
            participant.get("wardsKilled", 0),

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


try:
    # ========================================================
    # RIOT ID -> ACCOUNT
    # ========================================================

    account = get_account_by_riot_id(
        GAME_NAME,
        TAG_LINE,
        PLATFORM,
    )

    puuid = account["puuid"]

    print("\nACCOUNT FOUND")
    print(
        account["gameName"]
        + "#"
        + account["tagLine"]
    )


    # ========================================================
    # ACCOUNT -> RECENT MATCH
    # ========================================================

    match_ids = get_recent_match_ids(
        puuid,
        PLATFORM,
        count=5,
    )

    if not match_ids:
        raise ValueError(
            "No recent matches found."
        )

    latest_match_id = match_ids[0]

    print("\nLATEST MATCH")
    print(latest_match_id)


    # ========================================================
    # GET FULL MATCH
    # ========================================================

    match = get_match(
        latest_match_id,
        PLATFORM,
    )

    info = match["info"]

    riot_participants = info["participants"]

    print("\nPARTICIPANTS FOUND")
    print(len(riot_participants))


    # ========================================================
    # FIND THE SELECTED PLAYER
    # ========================================================

    selected_participant = next(
        (
            participant
            for participant in riot_participants
            if participant["puuid"] == puuid
        ),
        None,
    )

    if selected_participant is None:
        raise ValueError(
            "Player was not found in the match."
        )

    selected_player_id = (
        selected_participant["summonerId"]
    )

    print("\nSELECTED PLAYER")
    print(
        selected_participant.get(
            "riotIdGameName",
            GAME_NAME
        )
        + "#"
        + selected_participant.get(
            "riotIdTagline",
            TAG_LINE
        )
    )

    print(
        "Role:",
        selected_participant["teamPosition"]
    )


    # ========================================================
    # CONVERT RIOT DATA -> RANKSCOPE DATA
    # ========================================================

    rank_scope_participants = [
        convert_participant(
            participant,
            info["gameStartTimestamp"],
            info["gameDuration"],
        )
        for participant in riot_participants
    ]


    # ========================================================
    # RUN RANKSCOPE
    # ========================================================

    result = predict_rank(
        raw_participants=
            rank_scope_participants,

        selected_player_id=
            selected_player_id,
    )


    print("\n" + "=" * 50)
    print("RANKSCOPE RESULT")
    print("=" * 50)

    print(
        "Predicted rank:",
        result["predicted_rank"]
    )

    print(
        "Role:",
        result["role"]
    )

    print(
        "Lobby prediction:",
        result["lobby_prediction"]
    )

    print(
        "Message:",
        result["message"]
    )


except Exception as error:
    print("\nERROR")
    print(type(error).__name__)
    print(error)