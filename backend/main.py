
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
    predict_overall_rank,
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
    version="1.2.0",
    description=(
        "RankScope predicts League of Legends performance "
        "ranks using CatBoost models. It supports individual "
        "match predictions and overall predictions based on "
        "up to 20 recent Ranked Solo/Duo matches."
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
    return {"status": "ok"}


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
    return {"ranks": RANK_ORDER}


# ============================================================
# ROLES
# ============================================================

@app.get("/api/roles")
def get_roles():
    return {"roles": ROLES}


# ============================================================
# RECENT RANKED MATCHES
#
# Existing endpoint:
# Browse the player's latest 5 ranked matches.
# ============================================================

@app.post("/api/matches/recent")
def recent_ranked_matches(
    request: RiotIdPredictionRequest,
):
    try:
        return get_recent_ranked_matches(
            game_name=request.game_name,
            tag_line=request.tag_line,
            platform=request.platform,
            count=5,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


# ============================================================
# PREDICT SELECTED MATCH
#
# Existing functionality:
# Predict one specific match selected by the user.
# ============================================================

@app.post(
    "/api/predictions",
    status_code=status.HTTP_201_CREATED,
)
def create_riot_prediction(
    request: SelectedMatchPredictionRequest,
):
    try:
        return predict_selected_match(
            game_name=request.game_name,
            tag_line=request.tag_line,
            platform=request.platform,
            match_id=request.match_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


# ============================================================
# OVERALL PREDICTION
#
# NEW WORKFLOW:
#
# 1. Search player using Riot ID.
# 2. Retrieve up to 20 recent Ranked Solo matches.
# 3. Predict each match using the CatBoost ensemble.
# 4. Combine predictions using median voting.
# 5. Return overall rank and individual match statistics.
#
# The processing is handled by riot_service.py.
# ============================================================

@app.post("/api/predictions/overall")
def create_overall_prediction(
    request: RiotIdPredictionRequest,
):
    try:
        return predict_overall_rank(
            game_name=request.game_name,
            tag_line=request.tag_line,
            platform=request.platform,
            count=20,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )


# ============================================================
# RAW PREDICTION
#
# Used for directly testing the model with 10 participants.
# ============================================================

@app.post(
    "/api/predictions/raw",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_raw_prediction(
    request: PredictionRequest,
):
    try:
        participants = [
            participant.model_dump()
            for participant in request.participants
        ]

        return predict_rank(
            raw_participants=participants,
            selected_player_id=request.selected_player_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        )
