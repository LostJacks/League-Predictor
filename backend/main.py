from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from predictor import (
    predict_rank,
    RANK_ORDER,
    ROLES,
    role_models,
    rescue_models,
    bottom_ensemble_models,
    lobby_model,
)

from riot_service import (
    get_recent_ranked_matches,
    predict_selected_match,
)

from schemas import (
    PredictionRequest,
    PredictionResponse,
    RiotIdPredictionRequest,
    SelectedMatchPredictionRequest,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="RankScope API",
    version="1.0.0",
    description=(
        "REST API for predicting which League of Legends rank "
        "a single match performance most closely resembles."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/model")
def model_info():
    return {
        "status": "ready",
        "role_models": len(role_models),
        "rescue_models": len(rescue_models),
        "bottom_ensemble_models": len(bottom_ensemble_models),
        "lobby_model": lobby_model is not None,
    }


# ============================================================
# RANKS
# ============================================================

@app.get("/api/ranks")
def get_ranks():
    return {
        "ranks": RANK_ORDER
    }


# ============================================================
# ROLES
# ============================================================

@app.get("/api/roles")
def get_roles():
    return {
        "roles": ROLES
    }


# ============================================================
# RECENT RANKED MATCHES
#
# Riot ID -> latest 5 Ranked Solo matches
# Used by the React match-selection screen
# ============================================================

@app.post("/api/matches/recent")
def recent_ranked_matches(
    request: RiotIdPredictionRequest
):
    try:
        result = get_recent_ranked_matches(
            game_name=request.game_name,
            tag_line=request.tag_line,
            platform=request.platform,
            count=5,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


# ============================================================
# PREDICT SELECTED MATCH
#
# User selects one of the returned matches,
# then RankScope analyses that exact match.
# ============================================================

@app.post(
    "/api/predictions",
    status_code=status.HTTP_201_CREATED,
)
def create_riot_prediction(
    request: SelectedMatchPredictionRequest
):
    try:
        result = predict_selected_match(
            game_name=request.game_name,
            tag_line=request.tag_line,
            platform=request.platform,
            match_id=request.match_id,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


# ============================================================
# RAW PREDICTION
#
# Mainly used for testing directly with 10 participants.
# ============================================================

@app.post(
    "/api/predictions/raw",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_raw_prediction(
    request: PredictionRequest
):
    try:
        participants = [
            participant.model_dump()
            for participant in request.participants
        ]

        result = predict_rank(
            raw_participants=participants,
            selected_player_id=request.selected_player_id,
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )