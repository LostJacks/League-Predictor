from typing import List, Literal

from pydantic import BaseModel, Field


class ParticipantInput(BaseModel):
    # Match / player information
    gameStartTimestamp: int
    summonerId: str

    teamPosition: Literal[
        "TOP",
        "JUNGLE",
        "MIDDLE",
        "BOTTOM",
        "UTILITY"
    ]

    teamId: int

    # Match information
    gameDuration: float = Field(gt=0)

    # Combat
    kills: int = Field(ge=0)
    deaths: int = Field(ge=0)
    assists: int = Field(ge=0)

    # Farming
    totalMinionsKilled: float = Field(ge=0)
    neutralMinionsKilled: float = Field(default=0, ge=0)

    # Resources
    goldEarned: float = Field(ge=0)
    champExperience: float = Field(ge=0)

    # Damage
    totalDamageDealtToChampions: float = Field(ge=0)
    totalDamageTaken: float = Field(ge=0)

    # Vision
    visionScore: float = Field(ge=0)
    wardsPlaced: float = Field(default=0, ge=0)
    wardsKilled: float = Field(default=0, ge=0)
    visionWardsBoughtInGame: float = Field(default=0, ge=0)
    controlWardsPlaced: float = Field(default=0, ge=0)

    # Crowd control
    timeCCingOthers: float = Field(default=0, ge=0)

    # Early game
    laneMinionsFirst10Minutes: float = Field(default=0, ge=0)
    jungleCsBefore10Minutes: float = Field(default=0, ge=0)

    # Objectives
    turretPlatesTaken: float = Field(default=0, ge=0)
    soloKills: float = Field(default=0, ge=0)
    turretKills: float = Field(default=0, ge=0)
    turretTakedowns: float = Field(default=0, ge=0)

    damageDealtToObjectives: float = Field(default=0, ge=0)

    dragonTakedowns: float = Field(default=0, ge=0)
    baronTakedowns: float = Field(default=0, ge=0)
    riftHeraldTakedowns: float = Field(default=0, ge=0)

    dragonKills: float = Field(default=0, ge=0)
    baronKills: float = Field(default=0, ge=0)
    objectivesStolen: float = Field(default=0, ge=0)

    # Other statistics
    totalTimeSpentDead: float = Field(default=0, ge=0)

    maxCsAdvantageOnLaneOpponent: float = 0
    visionScoreAdvantageLaneOpponent: float = 0


# ============================================================
# RAW PREDICTION REQUEST
# ============================================================

class PredictionRequest(BaseModel):
    selected_player_id: str

    participants: List[ParticipantInput] = Field(
        min_length=10,
        max_length=10
    )


# ============================================================
# RIOT ID REQUEST
# Used to find recent Ranked Solo matches
# ============================================================

class RiotIdPredictionRequest(BaseModel):
    game_name: str
    tag_line: str
    platform: str


# ============================================================
# SELECTED MATCH REQUEST
# Used after the user chooses one of the recent matches
# ============================================================

class SelectedMatchPredictionRequest(BaseModel):
    game_name: str
    tag_line: str
    platform: str
    match_id: str


# ============================================================
# PREDICTION RESPONSE
# ============================================================

class PredictionResponse(BaseModel):
    predicted_rank: str
    rank_id: int
    role: str
    lobby_prediction: float
    message: str