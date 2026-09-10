#!/usr/bin/env python3
"""
Liga MX Odds Builder
Combines historical Football-Data.co.uk CSV (2012-2023) with API-Football (2023-2025)
Outputs unified Parquet with normalized teams, implied probs, overround.
"""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
API_KEY = os.getenv("API_FOOTBALL_KEY")  # export API_FOOTBALL_KEY="tu_key"
LEAGUE_ID = 262  # Liga MX en API-Football
SEASONS_API = [2018, 2019, 2020, 2021, 2022, 2023, 2024]  # 2018/19 through 2024/25
BOOKMAKER_ID = 8  # Bet365 (confiable, amplio historial)

HIST_CSV = Path("/tmp/ligamx_2012_2018.csv")
OUT_PARQUET = Path("ligamx_odds_2012_2025.parquet")
CACHE_DIR = Path("./api_cache")
CACHE_DIR.mkdir(exist_ok=True)

RATE_LIMIT_SEC = 6.1  # 100 req/día = 1 req cada ~864 seg; usamos 6s para burst y dormimos al final del día

# ──────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES DE EQUIPOS
# ──────────────────────────────────────────────────────────────
TEAM_MAP = {
    # Football-Data → API-Football / nombre canónico
    "U.A.N.L.- Tigres": "Tigres UANL",
    "U.N.A.M.- Pumas": "Pumas UNAM",
    "Club Tijuana": "Tijuana",
    "Club Leon": "Leon",
    "Club America": "America",
    "Guadalajara Chivas": "Guadalajara",
    "Atletico San Luis": "Atletico San Luis",
    "Atlético San Luis": "Atletico San Luis",
    "Santos Laguna": "Santos Laguna",
    "Cruz Azul": "Cruz Azul",
    "Monterrey": "Monterrey",
    "Pachuca": "Pachuca",
    "Toluca": "Toluca",
    "Atlas": "Atlas",
    "Queretaro": "Queretaro",
    "Puebla": "Puebla",
    "Monarcas": "Morelia",  # Monarcas Morelia → Mazatlán (2020), pero historial es Morelia
    "Necaxa": "Necaxa",
    "Juarez": "FC Juarez",
    "FC Juarez": "FC Juarez",
    "Mazatlan": "Mazatlan FC",
    "Mazatlan FC": "Mazatlan FC",
    "Lobos BUAP": "Lobos BUAP",  # descendido 2019
    "Veracruz": "Veracruz",  # desafiliado 2019
    "Chiapas": "Chiapas",  # desafiliado 2017
    "Atlante": "Atlante",  # no en primera desde 2014
    "U. de G.": "Leones Negros",
    "Correcaminos": "Correcaminos",
    "Dorados": "Dorados",
    "Celaya": "Celaya",
    "Tampico Madero": "Tampico Madero",
    "Venados": "Venados",
    "Cancun FC": "Cancun FC",
    "Atletico Morelia": "Atletico Morelia",
    "Cimarrones": "Cimarrones",
    "Tepatitlan": "Tepatitlan",
    "Zacatecas": "Mineros Zacatecas",
    "Tlaxcala": "Tlaxcala",
    "Raya2": "Raya2 Expansión",
    "Pumas Tabasco": "Pumas Tabasco",
    "Tapatio": "Tapatio",
}

def normalize_team(name: str) -> str:
    return TEAM_MAP.get(name.strip(), name.strip())


# ──────────────────────────────────────────────────────────────
# 1. CARGA HISTÓRICA (Football-Data via Wayback)
# ──────────────────────────────────────────────────────────────
def load_historical() -> pd.DataFrame:
    if not HIST_CSV.exists():
        raise FileNotFoundError(f"No existe {HIST_CSV}. Descárgalo primero con:\n"
                                f"  curl -sL 'https://web.archive.org/web/2024/https://www.football-data.co.uk/new/MEX.csv' -o {HIST_CSV}")

    df = pd.read_csv(HIST_CSV, encoding="latin-1")
    df = df[df["Country"] == "Mexico"].copy()
    df["League"] = df["League"].str.strip()
    df = df[df["League"] == "Liga MX"].copy()

    # Columnas estándar
    df = df.rename(columns={
        "Season": "season",
        "Date": "date",
        "Home": "home_team_raw",
        "Away": "away_team_raw",
        "HG": "home_score",
        "AG": "away_score",
        "Res": "result",  # H/D/A
        "PH": "odds_home_pinnacle",
        "PD": "odds_draw_pinnacle",
        "PA": "odds_away_pinnacle",
        "MaxH": "odds_home_max",
        "MaxD": "odds_draw_max",
        "MaxA": "odds_away_max",
        "AvgH": "odds_home_avg",
        "AvgD": "odds_draw_avg",
        "AvgA": "odds_away_avg",
    })

    # Normalizar equipos
    df["home_team"] = df["home_team_raw"].apply(normalize_team)
    df["away_team"] = df["away_team_raw"].apply(normalize_team)

    # Parsear fecha DD/MM/YY → datetime
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    df = df.dropna(subset=["date"])

    # Resultado codificado
    df["result_code"] = df["result"].map({"H": 1, "D": 0, "A": -1})

    # Implied probabilities + overround (usando Pinnacle = sharp book)
    for prefix in ["pinnacle", "max", "avg"]:
        h = f"odds_home_{prefix}"
        d = f"odds_draw_{prefix}"
        a = f"odds_away_{prefix}"
        if all(c in df.columns for c in [h, d, a]):
            df[f"imp_home_{prefix}"] = 1 / df[h]
            df[f"imp_draw_{prefix}"] = 1 / df[d]
            df[f"imp_away_{prefix}"] = 1 / df[a]
            df[f"overround_{prefix}"] = df[f"imp_home_{prefix}"] + df[f"imp_draw_{prefix}"] + df[f"imp_away_{prefix}"] - 1

    df["source"] = "football-data"
    return df


# ──────────────────────────────────────────────────────────────
# 2. API-FOOTBALL (2023-2025)
# ──────────────────────────────────────────────────────────────
def api_get(endpoint: str, params: Dict) -> Dict:
    url = f"https://v3.football.api-sports.io/{endpoint}"
    headers = {"x-apisports-key": API_KEY}
    cache_file = CACHE_DIR / f"{endpoint.replace('/', '_')}_{hash(frozenset(params.items()))}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code == 429:
        print("  ⚠ Rate limit hit — esperando 60s...")
        time.sleep(60)
        return api_get(endpoint, params)
    resp.raise_for_status()
    data = resp.json()

    with open(cache_file, "w") as f:
        json.dump(data, f)
    return data


def fetch_season_odds(season: int) -> pd.DataFrame:
    print(f"  📥 Descargando odds {season}/{season+1} (Liga MX)...")
    params = {"league": LEAGUE_ID, "season": season, "bookmaker": BOOKMAKER_ID}
    data = api_get("odds", params)

    rows = []
    for fixture in data.get("response", []):
        fix = fixture["fixture"]
        teams = fixture["teams"]
        goals = fixture["goals"]
        bookmakers = fixture.get("bookmakers", [])

        if not bookmakers:
            continue
        bm = bookmakers[0]  # Bet365
        bets = {b["name"]: b["values"] for b in bm.get("bets", [])}
        match_winner = bets.get("Match Winner", [])
        if len(match_winner) < 3:
            continue

        odds_map = {v["value"]: float(v["odd"]) for v in match_winner}
        rows.append({
            "fixture_id": fix["id"],
            "date": pd.to_datetime(fix["date"]).tz_convert(None),
            "home_team": normalize_team(teams["home"]["name"]),
            "away_team": normalize_team(teams["away"]["name"]),
            "home_score": goals["home"],
            "away_score": goals["away"],
            "odds_home_bet365": odds_map.get("Home"),
            "odds_draw_bet365": odds_map.get("Draw"),
            "odds_away_bet365": odds_map.get("Away"),
            "status": fix["status"]["short"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"    ⚠ Sin datos para {season}")
        return df

    df["season"] = f"{season}/{season+1}"
    df["source"] = "api-football"

    # Implied probs
    for col in ["odds_home_bet365", "odds_draw_bet365", "odds_away_bet365"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["imp_home_bet365"] = 1 / df["odds_home_bet365"]
    df["imp_draw_bet365"] = 1 / df["odds_draw_bet365"]
    df["imp_away_bet365"] = 1 / df["odds_away_bet365"]
    df["overround_bet365"] = df["imp_home_bet365"] + df["imp_draw_bet365"] + df["imp_away_bet365"] - 1

    # Result code
    df["result_code"] = df.apply(
        lambda r: 1 if r["home_score"] > r["away_score"] else (0 if r["home_score"] == r["away_score"] else -1)
        if pd.notna(r["home_score"]) else None, axis=1
    )
    return df


# ──────────────────────────────────────────────────────────────
# 3. UNIFICAR Y EXPORTAR
# ──────────────────────────────────────────────────────────────
def main():
    if not API_KEY:
        raise RuntimeError("Define API_FOOTBALL_KEY en variables de entorno:\n  export API_FOOTBALL_KEY='tu_key'")

    print("📊 Cargando histórico Football-Data (2012-2023)...")
    df_hist = load_historical()
    print(f"   {len(df_hist)} partidos")

    print("📡 Descargando API-Football (2023-2025)...")
    df_api_list = []
    for s in SEASONS_API:
        df_s = fetch_season_odds(s)
        if not df_s.empty:
            df_api_list.append(df_s)
        time.sleep(RATE_LIMIT_SEC)

    df_api = pd.concat(df_api_list, ignore_index=True) if df_api_list else pd.DataFrame()
    print(f"   {len(df_api)} partidos de API")

    # Unificar columnas comunes
    common_cols = [
        "season", "date", "home_team", "away_team",
        "home_score", "away_score", "result_code", "source"
    ]

    # Histórico: renombrar odds pinnacle a nombres genéricos
    df_hist_unif = df_hist[common_cols + [
        "odds_home_pinnacle", "odds_draw_pinnacle", "odds_away_pinnacle",
        "odds_home_max", "odds_draw_max", "odds_away_max",
        "odds_home_avg", "odds_draw_avg", "odds_away_avg",
        "imp_home_pinnacle", "imp_draw_pinnacle", "imp_away_pinnacle",
        "overround_pinnacle"
    ]].copy()

    # API: renombrar odds bet365
    df_api_unif = df_api[common_cols + [
        "odds_home_bet365", "odds_draw_bet365", "odds_away_bet365",
        "imp_home_bet365", "imp_draw_bet365", "imp_away_bet365",
        "overround_bet365"
    ]].copy()

    # Merge outer para mantener todas las columnas
    df_all = pd.concat([df_hist_unif, df_api_unif], ignore_index=True, sort=False)

    # Ordenar
    df_all = df_all.sort_values(["date", "home_team"]).reset_index(drop=True)

    # Guardar
    df_all.to_parquet(OUT_PARQUET, index=False)
    print(f"\n✅ Guardado: {OUT_PARQUET} ({len(df_all)} filas, {df_all['season'].nunique()} temporadas)")
    print(f"   Temporadas: {sorted(df_all['season'].unique())}")
    print(f"   Columnas: {list(df_all.columns)}")


if __name__ == "__main__":
    main()