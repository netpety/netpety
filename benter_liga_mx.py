#!/usr/bin/env python3
"""
Bill Benter-style Soccer Prediction Model for Liga MX
Based on Poisson regression with Dixon-Coles adjustment, time-weighting, and Kelly criterion.
"""

import csv
import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys


# ============================================================
# DATA LOADING
# ============================================================

def load_matches(csv_path: str) -> List[Dict]:
    """Load matches from CSV."""
    matches = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse date
            try:
                row['date_parsed'] = datetime.strptime(row['fecha'], '%a %b %d %Y')
            except ValueError:
                try:
                    row['date_parsed'] = datetime.strptime(row['fecha'], '%Y-%m-%d')
                except ValueError:
                    row['date_parsed'] = None
            
            # Parse goals
            try:
                row['goles_local'] = int(row['goles_local'])
                row['goles_visitante'] = int(row['goles_visitante'])
            except (ValueError, KeyError):
                row['goles_local'] = 0
                row['goles_visitante'] = 0
            
            matches.append(row)
    return matches


# ============================================================
# POISSON REGRESSION WITH DIXON-COLES & TIME WEIGHTING
# ============================================================

class BenterModel:
    """
    Bill Benter-inspired model for soccer:
    - Poisson goals with attack/defense parameters
    - Dixon-Coles adjustment for low-scoring draws
    - Exponential time decay (recent matches matter more)
    - Home advantage parameter
    - Ridge regularization for stability
    """
    
    def __init__(self, 
                 half_life_days: float = 365.0,      # Time decay half-life
                 dc_rho: float = 0.13,                # Dixon-Coles rho parameter
                 ridge_lambda: float = 0.1,           # Ridge regularization
                 home_advantage_init: float = 0.3):   # Initial home advantage (log scale)
        
        self.half_life = half_life_days
        self.rho = dc_rho
        self.ridge = ridge_lambda
        self.home_adv = home_advantage_init
        
        self.teams = set()
        self.team_to_idx = {}
        self.idx_to_team = {}
        self.n_teams = 0
        
        # Model parameters (log scale)
        self.attack = {}      # team -> attack strength
        self.defense = {}     # team -> defense strength
        self.fitted = False
    
    def _time_weight(self, match_date: datetime, ref_date: datetime) -> float:
        """Exponential time decay weight."""
        if match_date is None or ref_date is None:
            return 1.0
        days_diff = (ref_date - match_date).days
        if days_diff < 0:
            return 0.0
        return math.exp(-math.log(2) * days_diff / self.half_life)
    
    def _dixon_coles_tau(self, hg: int, ag: int, lambda_h: float, lambda_a: float) -> float:
        """Dixon-Coles adjustment for low scores (0-0, 1-0, 0-1, 1-1)."""
        if hg == 0 and ag == 0:
            return 1 - self.rho * lambda_h * lambda_a
        elif hg == 1 and ag == 0:
            return 1 + self.rho * lambda_a
        elif hg == 0 and ag == 1:
            return 1 + self.rho * lambda_h
        elif hg == 1 and ag == 1:
            return 1 - self.rho
        return 1.0
    
    def _poisson_log_likelihood(self, hg: int, ag: int, lambda_h: float, lambda_a: float) -> float:
        """Log-likelihood for a single match with Dixon-Coles adjustment."""
        tau = self._dixon_coles_tau(hg, ag, lambda_h, lambda_a)
        if tau <= 0:
            return -1e6
        return (hg * math.log(lambda_h) - lambda_h + 
                ag * math.log(lambda_a) - lambda_a + 
                math.log(tau))
    
    def _expected_goals(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Calculate expected goals for home and away teams."""
        att_h = self.attack.get(home_team, 0.0)
        def_h = self.defense.get(home_team, 0.0)
        att_a = self.attack.get(away_team, 0.0)
        def_a = self.defense.get(away_team, 0.0)
        
        lambda_h = math.exp(att_h - def_a + self.home_adv)
        lambda_a = math.exp(att_a - def_h)
        return lambda_h, lambda_a
    
    def fit(self, matches: List[Dict], ref_date: Optional[datetime] = None) -> 'BenterModel':
        """Fit model using gradient descent on weighted log-likelihood."""
        if ref_date is None:
            ref_date = max((m['date_parsed'] for m in matches if m['date_parsed']), default=datetime.now())
        
        # Collect teams
        for m in matches:
            self.teams.add(m['local'])
            self.teams.add(m['visitante'])
        
        self.team_to_idx = {t: i for i, t in enumerate(sorted(self.teams))}
        self.idx_to_team = {i: t for t, i in self.team_to_idx.items()}
        self.n_teams = len(self.teams)
        
        # Initialize parameters with small random values for better convergence
        import random
        random.seed(42)
        for team in self.teams:
            self.attack[team] = random.uniform(-0.1, 0.1)
            self.defense[team] = random.uniform(-0.1, 0.1)
        
        print(f"Fitting model on {len(matches)} matches, {self.n_teams} teams...")
        print(f"Reference date: {ref_date.date()}")
        
        # Gradient descent with adaptive learning rate
        lr = 0.005  # Smaller initial learning rate
        max_iter = 3000
        tolerance = 1e-6
        
        prev_ll = -float('inf')
        
        for iteration in range(max_iter):
            # Gradients
            grad_attack = defaultdict(float)
            grad_defense = defaultdict(float)
            grad_home = 0.0
            total_ll = 0.0
            total_weight = 0.0
            
            for m in matches:
                home = m['local']
                away = m['visitante']
                hg = m['goles_local']
                ag = m['goles_visitante']
                weight = self._time_weight(m['date_parsed'], ref_date)
                
                if weight == 0:
                    continue
                
                lambda_h, lambda_a = self._expected_goals(home, away)
                
                # Clip lambdas to prevent overflow
                lambda_h = min(max(lambda_h, 0.01), 10.0)
                lambda_a = min(max(lambda_a, 0.01), 10.0)
                
                # Dixon-Coles tau derivative components
                tau = self._dixon_coles_tau(hg, ag, lambda_h, lambda_a)
                if tau <= 0:
                    continue
                
                # Derivatives of log-likelihood w.r.t lambda_h, lambda_a
                d_ll_d_lambda_h = (hg / lambda_h - 1)
                d_ll_d_lambda_a = (ag / lambda_a - 1)
                
                # Add Dixon-Coles correction
                if hg == 0 and ag == 0:
                    d_ll_d_lambda_h += -self.rho * lambda_a / tau
                    d_ll_d_lambda_a += -self.rho * lambda_h / tau
                elif hg == 1 and ag == 0:
                    d_ll_d_lambda_a += self.rho / tau
                elif hg == 0 and ag == 1:
                    d_ll_d_lambda_h += self.rho / tau
                elif hg == 1 and ag == 1:
                    d_ll_d_lambda_h += self.rho / tau
                    d_ll_d_lambda_a += self.rho / tau
                
                # Chain rule: d_lambda/d_param = lambda
                d_ll_d_att_h = d_ll_d_lambda_h * lambda_h
                d_ll_d_def_a = d_ll_d_lambda_h * (-lambda_h)
                d_ll_d_att_a = d_ll_d_lambda_a * lambda_a
                d_ll_d_def_h = d_ll_d_lambda_a * (-lambda_a)
                d_ll_d_home = d_ll_d_lambda_h * lambda_h
                
                # Accumulate weighted gradients
                grad_attack[home] += weight * d_ll_d_att_h
                grad_defense[away] += weight * d_ll_d_def_a
                grad_attack[away] += weight * d_ll_d_att_a
                grad_defense[home] += weight * d_ll_d_def_h
                grad_home += weight * d_ll_d_home
                
                total_ll += weight * self._poisson_log_likelihood(hg, ag, lambda_h, lambda_a)
                total_weight += weight
            
            # Ridge regularization gradients (pull toward zero)
            for team in self.teams:
                grad_attack[team] -= self.ridge * self.attack[team]
                grad_defense[team] -= self.ridge * self.defense[team]
            
            # Gradient clipping
            max_grad = 1.0
            for team in self.teams:
                grad_attack[team] = max(min(grad_attack[team], max_grad), -max_grad)
                grad_defense[team] = max(min(grad_defense[team], max_grad), -max_grad)
            grad_home = max(min(grad_home, max_grad), -max_grad)
            
            # Update parameters
            max_change = 0.0
            for team in self.teams:
                self.attack[team] += lr * grad_attack[team]
                self.defense[team] += lr * grad_defense[team]
                max_change = max(max_change, abs(lr * grad_attack[team]), abs(lr * grad_defense[team]))
            
            self.home_adv += lr * grad_home
            max_change = max(max_change, abs(lr * grad_home))
            
            # Clip parameters to prevent explosion
            for team in self.teams:
                self.attack[team] = max(min(self.attack[team], 3.0), -3.0)
                self.defense[team] = max(min(self.defense[team], 3.0), -3.0)
            self.home_adv = max(min(self.home_adv, 2.0), -2.0)
            
            # Center parameters for identifiability
            avg_att = sum(self.attack.values()) / self.n_teams
            avg_def = sum(self.defense.values()) / self.n_teams
            for team in self.teams:
                self.attack[team] -= avg_att
                self.defense[team] -= avg_def
            
            # Learning rate decay
            if iteration > 500 and iteration % 500 == 0:
                lr *= 0.8
            
            # Check convergence
            if iteration % 200 == 0:
                print(f"  Iter {iteration}: LL={total_ll/total_weight:.4f}, home_adv={self.home_adv:.4f}, max_grad={max_change:.6f}, lr={lr:.5f}")
            
            if max_change < tolerance:
                print(f"  Converged at iteration {iteration}")
                break
            
            prev_ll = total_ll
        
        self.fitted = True
        print(f"\nModel fitted. Home advantage (log): {self.home_adv:.4f} -> multiplicative: {math.exp(self.home_adv):.4f}")
        return self
    
    def predict_match(self, home_team: str, away_team: str, max_goals: int = 6) -> Dict:
        """Predict match outcome probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        lambda_h, lambda_a = self._expected_goals(home_team, away_team)
        
        # Probability matrix
        prob_matrix = [[0.0 for _ in range(max_goals + 1)] for _ in range(max_goals + 1)]
        
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                # Poisson probabilities
                p_h = math.exp(-lambda_h) * (lambda_h ** hg) / math.factorial(hg)
                p_a = math.exp(-lambda_a) * (lambda_a ** ag) / math.factorial(ag)
                
                # Dixon-Coles adjustment
                tau = self._dixon_coles_tau(hg, ag, lambda_h, lambda_a)
                prob_matrix[hg][ag] = p_h * p_a * tau
        
        # Renormalize
        total = sum(sum(row) for row in prob_matrix)
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                prob_matrix[hg][ag] /= total
        
        # Outcome probabilities
        p_home = sum(prob_matrix[hg][ag] for hg in range(max_goals + 1) 
                     for ag in range(max_goals + 1) if hg > ag)
        p_draw = sum(prob_matrix[hg][ag] for hg in range(max_goals + 1) 
                     for ag in range(max_goals + 1) if hg == ag)
        p_away = sum(prob_matrix[hg][ag] for hg in range(max_goals + 1) 
                     for ag in range(max_goals + 1) if hg < ag)
        
        # Most likely scores
        score_probs = []
        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                score_probs.append(((hg, ag), prob_matrix[hg][ag]))
        score_probs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'lambda_home': lambda_h,
            'lambda_away': lambda_a,
            'p_home': p_home,
            'p_draw': p_draw,
            'p_away': p_away,
            'fair_odds_home': 1/p_home if p_home > 0 else 999,
            'fair_odds_draw': 1/p_draw if p_draw > 0 else 999,
            'fair_odds_away': 1/p_away if p_away > 0 else 999,
            'most_likely_scores': score_probs[:5],
            'score_matrix': prob_matrix
        }
    
    def kelly_stake(self, prob: float, odds: float, kelly_fraction: float = 0.25) -> float:
        """Kelly criterion stake (fraction of bankroll)."""
        if prob <= 0 or odds <= 1:
            return 0.0
        edge = prob * odds - 1
        if edge <= 0:
            return 0.0
        return kelly_fraction * edge / (odds - 1)
    
    def value_bets(self, home_team: str, away_team: str, 
                   book_odds: Dict[str, float], 
                   kelly_fraction: float = 0.25) -> List[Dict]:
        """Find value bets vs bookmaker odds."""
        pred = self.predict_match(home_team, away_team)
        value_bets = []
        
        for outcome, fair_odds_key, book_key in [
            ('home', 'fair_odds_home', 'home'),
            ('draw', 'fair_odds_draw', 'draw'),
            ('away', 'fair_odds_away', 'away')
        ]:
            fair = pred[fair_odds_key]
            book = book_odds.get(book_key, 0)
            prob = pred[f'p_{outcome}']
            
            if book > 0 and fair > 0:
                implied_prob = 1 / book
                value = prob - implied_prob
                kelly = self.kelly_stake(prob, book, kelly_fraction)
                
                if value > 0.01:  # At least 1% edge
                    value_bets.append({
                        'outcome': outcome,
                        'probability': prob,
                        'fair_odds': fair,
                        'book_odds': book,
                        'implied_prob': implied_prob,
                        'edge': value,
                        'kelly_stake': kelly
                    })
        
        return value_bets
    
    def team_ratings(self) -> List[Dict]:
        """Get team attack/defense ratings sorted by overall strength."""
        ratings = []
        for team in sorted(self.teams):
            att = self.attack[team]
            deff = self.defense[team]
            overall = att - deff  # Higher = stronger
            ratings.append({
                'team': team,
                'attack': att,
                'defense': deff,
                'overall': overall,
                'exp_goals_for': math.exp(att),
                'exp_goals_against': math.exp(deff)
            })
        ratings.sort(key=lambda x: x['overall'], reverse=True)
        return ratings


# ============================================================
# BACKTESTING / VALIDATION
# ============================================================

def backtest(model: BenterModel, matches: List[Dict], 
             start_date: datetime, end_date: datetime) -> Dict:
    """Test model predictions on historical matches."""
    test_matches = [m for m in matches 
                    if m['date_parsed'] and start_date <= m['date_parsed'] <= end_date]
    
    if not test_matches:
        return {'error': 'No matches in date range'}
    
    correct = 0
    total = 0
    log_loss = 0.0
    brier_score = 0.0
    
    for m in test_matches:
        pred = model.predict_match(m['local'], m['visitante'])
        
        # Actual result
        hg = m['goles_local']
        ag = m['goles_visitante']
        if hg > ag:
            actual = 'home'
        elif hg == ag:
            actual = 'draw'
        else:
            actual = 'away'
        
        # Predicted
        predicted = max(['home', 'draw', 'away'], key=lambda x: pred[f'p_{x}'])
        
        if predicted == actual:
            correct += 1
        total += 1
        
        # Log loss
        p_actual = pred[f'p_{actual}']
        if p_actual > 0:
            log_loss -= math.log(p_actual)
        
        # Brier score
        for outcome in ['home', 'draw', 'away']:
            pred_prob = pred[f'p_{outcome}']
            actual_prob = 1.0 if outcome == actual else 0.0
            brier_score += (pred_prob - actual_prob) ** 2
    
    return {
        'matches': total,
        'accuracy': correct / total if total > 0 else 0,
        'avg_log_loss': log_loss / total if total > 0 else 0,
        'brier_score': brier_score / (3 * total) if total > 0 else 0
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Load data
    print("Loading Liga MX data...")
    matches = load_matches('liga_mx_7_temporadas_formato.csv')
    print(f"Loaded {len(matches)} matches")
    
    # Filter completed matches with valid scores
    completed = [m for m in matches if m['date_parsed'] and m['goles_local'] >= 0]
    print(f"Completed matches: {len(completed)}")
    
    # Sort by date
    completed.sort(key=lambda x: x['date_parsed'])
    
    # Split: train on first 80%, test on last 20%
    split_idx = int(len(completed) * 0.8)
    train_matches = completed[:split_idx]
    test_matches = completed[split_idx:]
    
    print(f"Training: {len(train_matches)} matches (up to {train_matches[-1]['date_parsed'].date()})")
    print(f"Testing:  {len(test_matches)} matches (from {test_matches[0]['date_parsed'].date()})")
    
    # Fit model
    ref_date = train_matches[-1]['date_parsed']
    model = BenterModel(
        half_life_days=365,    # 1-year half-life
        dc_rho=0.13,           # Standard Dixon-Coles value
        ridge_lambda=0.1,      # Regularization
        home_advantage_init=0.3
    )
    model.fit(train_matches, ref_date)
    
    # Backtest
    print("\n=== BACKTEST RESULTS ===")
    test_start = test_matches[0]['date_parsed']
    test_end = test_matches[-1]['date_parsed']
    results = backtest(model, test_matches, test_start, test_end)
    print(f"Matches: {results['matches']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Avg Log Loss: {results['avg_log_loss']:.4f}")
    print(f"Brier Score: {results['brier_score']:.4f}")
    
    # Team ratings
    print("\n=== TEAM RATINGS (Overall Strength) ===")
    ratings = model.team_ratings()
    for i, r in enumerate(ratings[:18]):
        print(f"{i+1:2d}. {r['team']:25s} Att: {r['attack']:+.3f} Def: {r['defense']:+.3f} "
              f"Overall: {r['overall']:+.3f} (xGF: {r['exp_goals_for']:.2f}, xGA: {r['exp_goals_against']:.2f})")
    
    # Example predictions for upcoming matches (last round of data)
    print("\n=== EXAMPLE PREDICTIONS ===")
    # Get unique matchups from last few rounds
    recent = [m for m in completed if m['date_parsed'] > datetime(2024, 10, 1)]
    seen = set()
    for m in recent[-20:]:
        key = (m['local'], m['visitante'])
        if key not in seen:
            seen.add(key)
            pred = model.predict_match(m['local'], m['visitante'])
            print(f"\n{m['local']} vs {m['visitante']}")
            print(f"  xG: {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}")
            print(f"  1X2: {pred['p_home']:.1%} / {pred['p_draw']:.1%} / {pred['p_away']:.1%}")
            print(f"  Fair odds: {pred['fair_odds_home']:.2f} / {pred['fair_odds_draw']:.2f} / {pred['fair_odds_away']:.2f}")
            print(f"  Top scores: {', '.join(f'{s[0]}-{s[1]} ({p:.1%})' for s, p in pred['most_likely_scores'][:3])}")
    
    return model


if __name__ == '__main__':
    model = main()