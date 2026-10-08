from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

GAME = "gameStartTimestamp"
PLAYER = "summonerId"
ROLE = "teamPosition"
TEAM = "teamId"
RANK_VALUES = np.arange(10)

with open(MODEL_DIR / "preprocessing.pkl", "rb") as file:
    preprocessing = pickle.load(file)

RANK_ORDER = preprocessing["rank_order"]
ROLES = preprocessing["roles"]

ROLE_FEATURES = preprocessing["role_features"]
ROLE_MEDIANS = preprocessing["role_medians"]

LOBBY_MODEL_FEATURES = preprocessing["lobby_model_features"]
LOBBY_MEDIANS = preprocessing["lobby_medians"]

MATCH_FEATURES = preprocessing["match_features"]
TOP_ROLE_FEATURES = preprocessing["top_role_features"]
SPECIALIST_FEATURES = preprocessing["specialist_features"]

RESCUE_MEDIANS = preprocessing["rescue_medians"]
BOTTOM_ENSEMBLE_MEDIANS = preprocessing["bottom_ensemble_medians"]

FUSION_SETTINGS = preprocessing["fusion_settings"]
RESCUE_SETTINGS = preprocessing["rescue_settings"]
BOTTOM_BEST_SETTINGS = preprocessing["bottom_best_settings"]


def load_classifier(filename):
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / filename))
    return model


role_models = {
    role: load_classifier(f"role_{role.lower()}.cbm")
    for role in ROLES
}

rescue_models = {
    role: load_classifier(f"rescue_{role.lower()}.cbm")
    for role in ["BOTTOM", "UTILITY"]
}

bottom_ensemble_models = [
    load_classifier(f"bottom_ensemble_{number}.cbm")
    for number in range(1, 5)
]

lobby_model = CatBoostRegressor()
lobby_model.load_model(str(MODEL_DIR / "lobby.cbm"))


def safe_divide(a, b, add=1e-9):
    return a / (b + add)


def aligned_probability(model, X):
    raw_probability = model.predict_proba(X)
    output = np.zeros((len(X), 10))

    for index, class_id in enumerate(model.classes_):
        output[:, int(class_id)] = raw_probability[:, index]

    return output


def soft_hard_prediction(prediction, sigma):
    prediction = np.asarray(prediction, dtype=float)

    distance = (
        RANK_VALUES[None, :]
        - prediction[:, None]
    )

    probability = np.exp(
        -(distance ** 2)
        / (2 * sigma ** 2)
    )

    return probability / probability.sum(
        axis=1,
        keepdims=True
    )


def fuse_player_and_lobby(
    probability,
    lobby_prediction,
    settings
):
    if settings["method"] == "soft":
        powered = np.power(
            np.maximum(probability, 1e-12),
            settings["power"]
        )

        powered = (
            powered
            / powered.sum(axis=1, keepdims=True)
        )

        expected_rank = powered @ RANK_VALUES

        hard_rank = np.argmax(
            probability,
            axis=1
        )

        player_score = (
            settings["soft_weight"] * expected_rank
            +
            (1 - settings["soft_weight"]) * hard_rank
        )

        final_score = (
            settings["player_weight"] * player_score
            +
            (1 - settings["player_weight"])
            * lobby_prediction
        )

        return np.clip(
            np.rint(final_score),
            0,
            9
        ).astype(int)

    player_probability = np.power(
        np.maximum(probability, 1e-12),
        settings["probability_power"]
    )

    distance = (
        RANK_VALUES[None, :]
        - lobby_prediction[:, None]
    )

    lobby_prior = np.exp(
        -(distance ** 2)
        / (2 * settings["sigma"] ** 2)
    )

    combined = (
        player_probability
        * np.power(
            lobby_prior,
            settings["strength"]
        )
    )

    return np.argmax(
        combined,
        axis=1
    ).astype(int)


def build_features(raw_participants):
    participants = pd.DataFrame(raw_participants).copy()

    if len(participants) != 10:
        raise ValueError(
            "A prediction requires all 10 participants from one match."
        )

    required_columns = [
        GAME,
        PLAYER,
        ROLE,
        TEAM,
        "gameDuration",
        "kills",
        "deaths",
        "assists",
        "totalMinionsKilled",
        "goldEarned",
        "champExperience",
        "totalDamageDealtToChampions",
        "totalDamageTaken",
        "visionScore",
    ]

    missing = [
        column
        for column in required_columns
        if column not in participants.columns
    ]

    if missing:
        raise ValueError(
            "Missing required fields: "
            + ", ".join(missing)
        )

    if participants[GAME].nunique() != 1:
        raise ValueError(
            "All participants must belong to the same match."
        )

    if not participants[ROLE].isin(ROLES).all():
        raise ValueError(
            "Every participant must have a supported teamPosition."
        )

    participants = participants.drop_duplicates(
        subset=[GAME, PLAYER],
        keep="first"
    )

    if len(participants) != 10:
        raise ValueError(
            "The match must contain 10 unique participants."
        )

    optional_zero_columns = [
        "neutralMinionsKilled",
        "wardsPlaced",
        "wardsKilled",
        "visionWardsBoughtInGame",
        "controlWardsPlaced",
        "timeCCingOthers",
        "laneMinionsFirst10Minutes",
        "jungleCsBefore10Minutes",
        "turretPlatesTaken",
        "soloKills",
        "turretKills",
        "turretTakedowns",
        "totalTimeSpentDead",
        "damageDealtToObjectives",
        "maxCsAdvantageOnLaneOpponent",
        "visionScoreAdvantageLaneOpponent",
        "dragonTakedowns",
        "baronTakedowns",
        "riftHeraldTakedowns",
        "dragonKills",
        "baronKills",
        "objectivesStolen",
    ]

    for column in optional_zero_columns:
        if column not in participants.columns:
            participants[column] = 0.0

    numeric_columns = [
        column
        for column in participants.columns
        if column not in [GAME, PLAYER, ROLE]
    ]

    for column in numeric_columns:
        participants[column] = pd.to_numeric(
            participants[column],
            errors="coerce"
        )

    participants["duration_minutes"] = (
        participants["gameDuration"] / 60
    )

    participants.loc[
        participants["duration_minutes"] <= 0,
        "duration_minutes"
    ] = np.nan

    participants["kda"] = (
        participants["kills"]
        + participants["assists"]
    ) / participants["deaths"].clip(lower=1)

    participants["cs_total"] = (
        participants["totalMinionsKilled"].fillna(0)
        + participants["neutralMinionsKilled"].fillna(0)
    )

    rate_sources = {
        "kills_per_min": "kills",
        "deaths_per_min": "deaths",
        "assists_per_min": "assists",
        "cs_per_min": "cs_total",
        "gold_per_min": "goldEarned",
        "xp_per_min": "champExperience",
        "damage_per_min": "totalDamageDealtToChampions",
        "damage_taken_per_min": "totalDamageTaken",
        "vision_per_min": "visionScore",
        "wards_placed_per_min": "wardsPlaced",
        "wards_killed_per_min": "wardsKilled",
        "control_wards_per_min": "controlWardsPlaced",
        "cc_per_min": "timeCCingOthers",
        "objective_damage_per_min": "damageDealtToObjectives",
        "dead_time_per_min": "totalTimeSpentDead",
        "plates_per_min": "turretPlatesTaken",
        "solo_kills_per_min": "soloKills",
        "turret_kills_per_min": "turretKills",
        "turret_takedowns_per_min": "turretTakedowns",
    }

    for new_column, source_column in rate_sources.items():
        participants[new_column] = safe_divide(
            participants[source_column],
            participants["duration_minutes"]
        )

    participants["kp_actions_per_min"] = safe_divide(
        participants["kills"] + participants["assists"],
        participants["duration_minutes"]
    )

    participants["early_cs_per_min"] = (
        participants["laneMinionsFirst10Minutes"] / 10
    )

    participants["early_jungle_cs_per_min"] = (
        participants["jungleCsBefore10Minutes"] / 10
    )

    participants["epic_objective_takedowns"] = (
        participants["dragonTakedowns"].fillna(0)
        + participants["baronTakedowns"].fillna(0)
        + participants["riftHeraldTakedowns"].fillna(0)
    )

    fallback_objectives = (
        participants["dragonKills"].fillna(0)
        + participants["baronKills"].fillna(0)
        + participants["objectivesStolen"].fillna(0)
    )

    participants["epic_objective_takedowns"] = np.where(
        participants["epic_objective_takedowns"] > 0,
        participants["epic_objective_takedowns"],
        fallback_objectives
    )

    participants["epic_objectives_per_min"] = safe_divide(
        participants["epic_objective_takedowns"],
        participants["duration_minutes"]
    )

    participants["objective_actions"] = (
        participants["turretTakedowns"].fillna(0)
        + participants["epic_objective_takedowns"].fillna(0)
    )

    participants["objective_actions_per_min"] = safe_divide(
        participants["objective_actions"],
        participants["duration_minutes"]
    )

    participants["xp_per_cs"] = safe_divide(
        participants["champExperience"],
        participants["cs_total"],
        add=1
    )

    participants["gold_per_cs"] = safe_divide(
        participants["goldEarned"],
        participants["cs_total"],
        add=1
    )

    participants["damage_per_gold"] = safe_divide(
        participants["totalDamageDealtToChampions"],
        participants["goldEarned"],
        add=1
    )

    participants["damage_taken_per_death"] = safe_divide(
        participants["totalDamageTaken"],
        participants["deaths"],
        add=1
    )

    participants["damage_to_taken_ratio"] = safe_divide(
        participants["totalDamageDealtToChampions"],
        participants["totalDamageTaken"],
        add=1
    )

    participants["damage_per_kp_action"] = safe_divide(
        participants["totalDamageDealtToChampions"],
        participants["kills"]
        + participants["assists"],
        add=1
    )

    participants["gold_per_kp_action"] = safe_divide(
        participants["goldEarned"],
        participants["kills"]
        + participants["assists"],
        add=1
    )

    participants["vision_activity_per_min"] = safe_divide(
        participants["wardsPlaced"]
        + participants["wardsKilled"]
        + participants["controlWardsPlaced"],
        participants["duration_minutes"]
    )

    participants["combat_activity_per_min"] = safe_divide(
        participants["kills"]
        + participants["deaths"]
        + participants["assists"],
        participants["duration_minutes"]
    )

    participants["vision_per_ward"] = safe_divide(
        participants["visionScore"],
        participants["wardsPlaced"]
        + participants["controlWardsPlaced"],
        add=1
    )

    participants["cc_per_assist"] = safe_divide(
        participants["timeCCingOthers"],
        participants["assists"],
        add=1
    )

    participants["neutral_cs_ratio"] = safe_divide(
        participants["neutralMinionsKilled"],
        participants["cs_total"],
        add=1
    )

    participants["lane_cs_ratio"] = safe_divide(
        participants["totalMinionsKilled"],
        participants["cs_total"],
        add=1
    )

    participants["early_cs_share"] = safe_divide(
        participants["laneMinionsFirst10Minutes"],
        participants["totalMinionsKilled"],
        add=1
    )

    team_group = [GAME, TEAM]

    def add_share(source, destination):
        team_total = (
            participants
            .groupby(team_group)[source]
            .transform("sum")
        )

        participants[destination] = safe_divide(
            participants[source],
            team_total
        )

    add_share("cs_total", "cs_share")
    add_share("goldEarned", "gold_share")
    add_share("champExperience", "xp_share")
    add_share(
        "totalDamageDealtToChampions",
        "damage_share"
    )
    add_share("visionScore", "vision_share")
    add_share("deaths", "death_share")

    participants["teamDamagePercentage"] = (
        participants["damage_share"]
    )

    participants["damageTakenOnTeamPercentage"] = (
        safe_divide(
            participants["totalDamageTaken"],
            participants.groupby(team_group)[
                "totalDamageTaken"
            ].transform("sum")
        )
    )

    team_kills = (
        participants
        .groupby(team_group)["kills"]
        .transform("sum")
    )

    participants["killParticipation"] = safe_divide(
        participants["kills"] + participants["assists"],
        team_kills
    )

    participants["resource_share_average"] = (
        participants["gold_share"]
        + participants["xp_share"]
        + participants["cs_share"]
    ) / 3

    participants["impact_share_average"] = (
        participants["damage_share"]
        + participants["vision_share"]
    ) / 2

    participants["damage_vs_gold_efficiency"] = safe_divide(
        participants["damage_share"],
        participants["gold_share"]
    )

    participants["xp_vs_gold_efficiency"] = safe_divide(
        participants["xp_share"],
        participants["gold_share"]
    )

    participants["cs_vs_gold_efficiency"] = safe_divide(
        participants["cs_share"],
        participants["gold_share"]
    )

    participants["vision_vs_gold_efficiency"] = safe_divide(
        participants["vision_share"],
        participants["gold_share"]
    )

    participants["damage_minus_gold_share"] = (
        participants["damage_share"]
        - participants["gold_share"]
    )

    participants["xp_minus_gold_share"] = (
        participants["xp_share"]
        - participants["gold_share"]
    )

    participants["cs_minus_gold_share"] = (
        participants["cs_share"]
        - participants["gold_share"]
    )

    participants["vision_minus_gold_share"] = (
        participants["vision_share"]
        - participants["gold_share"]
    )

    opponent_metrics = [
        "cs_per_min",
        "gold_per_min",
        "xp_per_min",
        "damage_per_min",
        "vision_per_min",
        "kda",
    ]

    role_team = (
        participants
        .groupby(
            [GAME, ROLE, TEAM],
            as_index=False
        )[opponent_metrics]
        .mean()
    )

    number_of_teams = (
        role_team
        .groupby([GAME, ROLE])[TEAM]
        .transform("count")
    )

    for metric in opponent_metrics:
        group_sum = (
            role_team
            .groupby([GAME, ROLE])[metric]
            .transform("sum")
        )

        role_team[
            f"opponent_{metric}"
        ] = safe_divide(
            group_sum - role_team[metric],
            number_of_teams - 1
        )

    opponent_columns = [
        f"opponent_{metric}"
        for metric in opponent_metrics
    ]

    participants = participants.merge(
        role_team[
            [GAME, ROLE, TEAM]
            + opponent_columns
        ],
        on=[GAME, ROLE, TEAM],
        how="left",
        validate="many_to_one"
    )

    for metric in [
        "cs",
        "gold",
        "xp",
        "damage",
        "vision",
    ]:
        player_column = f"{metric}_per_min"

        participants[
            f"{metric}_diff_vs_opponent"
        ] = (
            participants[player_column]
            - participants[f"opponent_{player_column}"]
        )

    participants["kda_diff_vs_opponent"] = (
        participants["kda"]
        - participants["opponent_kda"]
    )

    ratio_settings = {
        "cs_advantage_ratio": (
            "cs_per_min",
            "opponent_cs_per_min",
            0.5
        ),
        "gold_advantage_ratio": (
            "gold_per_min",
            "opponent_gold_per_min",
            1
        ),
        "xp_advantage_ratio": (
            "xp_per_min",
            "opponent_xp_per_min",
            1
        ),
        "damage_advantage_ratio": (
            "damage_per_min",
            "opponent_damage_per_min",
            1
        ),
        "vision_advantage_ratio": (
            "vision_per_min",
            "opponent_vision_per_min",
            0.1
        ),
    }

    for new_column, (
        player_column,
        opponent_column,
        add
    ) in ratio_settings.items():
        participants[new_column] = safe_divide(
            participants[player_column]
            - participants[opponent_column],
            participants[opponent_column],
            add=add
        ).clip(-5, 5)

    participants["kda_advantage_ratio"] = safe_divide(
        participants["kda"]
        - participants["opponent_kda"],
        np.abs(participants["opponent_kda"]),
        add=1
    ).clip(-5, 5)

    context_sources = {
        "cs": "cs_per_min",
        "gold": "gold_per_min",
        "xp": "xp_per_min",
        "damage": "damage_per_min",
        "vision": "vision_per_min",
    }

    for prefix, source in context_sources.items():
        participants[
            f"{prefix}_rank_match"
        ] = (
            participants
            .groupby(GAME)[source]
            .rank(
                ascending=False,
                method="average"
            )
        )

        participants[
            f"{prefix}_rank_team"
        ] = (
            participants
            .groupby([GAME, TEAM])[source]
            .rank(
                ascending=False,
                method="average"
            )
        )

        participants[
            f"{prefix}_percentile_match"
        ] = (
            participants
            .groupby(GAME)[source]
            .rank(
                pct=True,
                ascending=True
            )
        )

        match_average = (
            participants
            .groupby(GAME)[source]
            .transform("mean")
        )

        participants[
            f"{prefix}_vs_match_average"
        ] = (
            participants[source]
            - match_average
        )

    participants["lf_cs_pm"] = participants["cs_per_min"]
    participants["lf_gold_pm"] = participants["gold_per_min"]
    participants["lf_xp_pm"] = participants["xp_per_min"]
    participants["lf_damage_pm"] = participants["damage_per_min"]
    participants["lf_vision_pm"] = participants["vision_per_min"]
    participants["lf_kills_pm"] = participants["kills_per_min"]
    participants["lf_deaths_pm"] = participants["deaths_per_min"]
    participants["lf_assists_pm"] = participants["assists_per_min"]
    participants["lf_kda"] = participants["kda"]
    participants["lf_objective_pm"] = (
        participants["objective_actions_per_min"]
    )
    participants["lf_cc_pm"] = participants["cc_per_min"]
    participants["lf_wards_pm"] = (
        participants["wards_placed_per_min"]
    )
    participants["lf_wards_killed_pm"] = (
        participants["wards_killed_per_min"]
    )

    lobby_metrics = [
        "lf_cs_pm",
        "lf_gold_pm",
        "lf_xp_pm",
        "lf_damage_pm",
        "lf_vision_pm",
        "lf_kills_pm",
        "lf_deaths_pm",
        "lf_assists_pm",
        "lf_kda",
        "lf_objective_pm",
        "lf_cc_pm",
        "lf_wards_pm",
        "lf_wards_killed_pm",
    ]

    lobby_general = (
        participants
        .groupby(GAME)[lobby_metrics]
        .agg(["mean", "std"])
    )

    lobby_general.columns = [
        f"lobby_{metric}_{stat}"
        for metric, stat in lobby_general.columns
    ]

    lobby_general = lobby_general.reset_index()

    participants = participants.merge(
        lobby_general,
        on=GAME,
        how="left",
        validate="many_to_one"
    )

    team_lobby_metrics = [
        "lf_cs_pm",
        "lf_gold_pm",
        "lf_xp_pm",
        "lf_damage_pm",
        "lf_vision_pm",
    ]

    team_lobby = (
        participants
        .groupby(
            [GAME, TEAM],
            as_index=False
        )[team_lobby_metrics]
        .mean()
    )

    team_count = (
        team_lobby
        .groupby(GAME)[TEAM]
        .transform("count")
    )

    team_difference_features = []

    for metric in team_lobby_metrics:
        total = (
            team_lobby
            .groupby(GAME)[metric]
            .transform("sum")
        )

        enemy_value = safe_divide(
            total - team_lobby[metric],
            team_count - 1
        )

        new_column = (
            f"team_enemy_{metric}_diff"
        )

        team_lobby[new_column] = (
            team_lobby[metric]
            - enemy_value
        )

        team_difference_features.append(
            new_column
        )

    participants = participants.merge(
        team_lobby[
            [GAME, TEAM]
            + team_difference_features
        ],
        on=[GAME, TEAM],
        how="left",
        validate="many_to_one"
    )

    match_team_difference = (
        team_lobby[
            [GAME]
            + team_difference_features
        ]
        .copy()
    )

    for column in team_difference_features:
        match_team_difference[column] = (
            match_team_difference[column].abs()
        )

    match_team_difference = (
        match_team_difference
        .groupby(GAME)[
            team_difference_features
        ]
        .max()
        .reset_index()
    )

    match_team_difference = (
        match_team_difference.rename(
            columns={
                column:
                    f"match_abs_{column}"
                for column
                in team_difference_features
            }
        )
    )

    for metric in team_lobby_metrics:
        mean_column = f"lobby_{metric}_mean"
        std_column = f"lobby_{metric}_std"
        z_column = f"lobby_{metric}_zscore"

        participants[z_column] = (
            participants[metric]
            - participants[mean_column]
        ) / participants[
            std_column
        ].replace(0, np.nan)

    role_mean = (
        participants
        .groupby(
            [GAME, ROLE]
        )[lobby_metrics]
        .mean()
        .unstack(ROLE)
    )

    role_mean.columns = [
        f"role_{role}_{metric}_mean"
        for metric, role
        in role_mean.columns
    ]

    role_std = (
        participants
        .groupby(
            [GAME, ROLE]
        )[lobby_metrics]
        .std()
        .unstack(ROLE)
    )

    role_std.columns = [
        f"role_{role}_{metric}_std"
        for metric, role
        in role_std.columns
    ]

    role_lobby_features = (
        role_mean
        .join(role_std)
        .reset_index()
    )

    match_duration = (
        participants
        .groupby(GAME)["duration_minutes"]
        .mean()
        .rename("duration_minutes")
        .reset_index()
    )

    match_context = (
        lobby_general
        .merge(
            match_duration,
            on=GAME,
            how="left",
            validate="one_to_one"
        )
        .merge(
            match_team_difference,
            on=GAME,
            how="left",
            validate="one_to_one"
        )
    )

    participants.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return (
        participants,
        match_context,
        role_lobby_features
    )


def predict_rank(raw_participants, selected_player_id):
    (
        participants,
        match_context,
        role_lobby_features
    ) = build_features(raw_participants)

    selected_rows = participants[
        participants[PLAYER].astype(str)
        == str(selected_player_id)
    ].copy()

    if len(selected_rows) != 1:
        raise ValueError(
            "selected_player_id must match exactly one participant."
        )

    selected = selected_rows.iloc[[0]].copy()
    role = selected.iloc[0][ROLE]

    full_match_context = (
        match_context
        .merge(
            role_lobby_features,
            on=GAME,
            how="left",
            validate="one_to_one"
        )
    )

    X_lobby = (
        full_match_context
        .reindex(columns=LOBBY_MODEL_FEATURES)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(LOBBY_MEDIANS)
    )

    lobby_prediction = float(
        np.clip(
            lobby_model.predict(X_lobby)[0],
            0,
            9
        )
    )

    selected["lobby_prediction"] = lobby_prediction

    extra_features = list(
        dict.fromkeys(
            MATCH_FEATURES
            + TOP_ROLE_FEATURES
        )
    )

    for column in extra_features:
        if (
            column not in selected.columns
            and column in full_match_context.columns
        ):
            selected[column] = (
                full_match_context.iloc[0][column]
            )

    for current_role in ROLES:
        selected[
            f"player_role_{current_role}"
        ] = int(role == current_role)

    role_features = ROLE_FEATURES[role]

    X_role = (
        selected
        .reindex(columns=role_features)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(ROLE_MEDIANS[role])
    )

    role_probability = aligned_probability(
        role_models[role],
        X_role
    )

    base_prediction = int(
        fuse_player_and_lobby(
            role_probability,
            np.array([lobby_prediction]),
            FUSION_SETTINGS[role]
        )[0]
    )

    final_prediction = base_prediction

    if role in rescue_models:
        X_specialist = (
            selected
            .reindex(columns=SPECIALIST_FEATURES)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(RESCUE_MEDIANS[role])
        )

        specialist_probability = aligned_probability(
            rescue_models[role],
            X_specialist
        )

        rescue_setting = RESCUE_SETTINGS[role]

        base_probability = soft_hard_prediction(
            np.array([base_prediction]),
            rescue_setting["sigma"]
        )

        combined_probability = (
            (1 - rescue_setting["alpha"])
            * base_probability
            +
            rescue_setting["alpha"]
            * specialist_probability
        )

        fused_prediction = int(
            np.argmax(
                combined_probability,
                axis=1
            )[0]
        )

        specialist_confidence = float(
            specialist_probability.max(axis=1)[0]
        )

        if (
            specialist_confidence
            >= rescue_setting["threshold"]
        ):
            final_prediction = fused_prediction

        if role == "BOTTOM":
            X_bottom_ensemble = (
                selected
                .reindex(columns=SPECIALIST_FEATURES)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(BOTTOM_ENSEMBLE_MEDIANS)
            )

            ensemble_probabilities = [
                specialist_probability
            ]

            for ensemble_model in bottom_ensemble_models:
                ensemble_probabilities.append(
                    aligned_probability(
                        ensemble_model,
                        X_bottom_ensemble
                    )
                )

            ensemble_probability = np.mean(
                np.stack(
                    ensemble_probabilities,
                    axis=0
                ),
                axis=0
            )

            sorted_probability = np.sort(
                ensemble_probability,
                axis=1
            )

            ensemble_confidence = float(
                sorted_probability[0, -1]
            )

            ensemble_margin = float(
                sorted_probability[0, -1]
                - sorted_probability[0, -2]
            )

            bottom_setting = BOTTOM_BEST_SETTINGS

            if bottom_setting["method"] == "probability":
                powered = np.power(
                    np.maximum(
                        ensemble_probability,
                        1e-12
                    ),
                    bottom_setting["power"]
                )

                powered = (
                    powered
                    / powered.sum(
                        axis=1,
                        keepdims=True
                    )
                )

                rescue_probability = (
                    soft_hard_prediction(
                        np.array([final_prediction]),
                        bottom_setting["sigma"]
                    )
                )

                combined = (
                    (1 - bottom_setting["alpha"])
                    * rescue_probability
                    +
                    bottom_setting["alpha"]
                    * powered
                )

                candidate = int(
                    np.argmax(
                        combined,
                        axis=1
                    )[0]
                )

                override = (
                    ensemble_confidence
                    >= bottom_setting["confidence"]
                    and
                    ensemble_margin
                    >= bottom_setting["margin"]
                )

                if override:
                    final_prediction = candidate

            else:
                hard_predictions = [
                    np.argmax(
                        probability,
                        axis=1
                    )[0]
                    for probability
                    in ensemble_probabilities
                ]

                vote_counts = np.bincount(
                    hard_predictions,
                    minlength=10
                )

                majority_prediction = int(
                    np.argmax(vote_counts)
                )

                majority_votes = int(
                    vote_counts.max()
                )

                override = (
                    majority_votes
                    >= bottom_setting["minimum_votes"]
                    and
                    ensemble_confidence
                    >= bottom_setting["confidence"]
                    and
                    ensemble_margin
                    >= bottom_setting["margin"]
                )

                if override:
                    final_prediction = majority_prediction

    return {
        "predicted_rank": RANK_ORDER[
            int(final_prediction)
        ],
        "rank_id": int(final_prediction),
        "role": role,
        "lobby_prediction": round(
            lobby_prediction,
            4
        ),
        "message": (
            "In this match, your performance most closely resembles "
            f"{RANK_ORDER[int(final_prediction)].title()}."
        )
    }
