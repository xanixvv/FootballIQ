
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

const API_URL = "http://127.0.0.1:8000";

function displayName(name = "") {
  return name
    .replace(/ FC$/, "")
    .replace(/ AFC$/, "")
    .replace("Manchester City", "Man City")
    .replace("Manchester United", "Man United")
    .replace("Brighton & Hove Albion", "Brighton");
}

export default function MatchDetails() {
  const { matchId } = useParams();
  const navigate = useNavigate();
  const [match, setMatch] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadMatch() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(`${API_URL}/fixtures/predictions`);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const selectedMatch = data.matches.find(
          (item) => String(item.id) === matchId
        );

        if (!selectedMatch) {
          setError("This match was not found in the upcoming fixtures.");
          return;
        }

        setMatch(selectedMatch);
      } catch (err) {
        console.error(err);
        setError("Could not load this match. Check that the backend is running.");
      } finally {
        setLoading(false);
      }
    }

    loadMatch();
  }, [matchId]);

  if (loading) {
    return <main className="detail-shell"><p>Loading match analysis...</p></main>;
  }

  if (error || !match) {
    return (
      <main className="detail-shell">
        <button className="back-button" onClick={() => navigate("/")}>
          ← All fixtures
        </button>
        <div className="message-panel error">{error || "Match unavailable."}</div>
      </main>
    );
  }

  const { prediction, history } = match;
  const probabilities = [
    { label: "Home win", key: "home_win", color: "#b6f36b" },
    { label: "Draw", key: "draw", color: "#f5c76b" },
    { label: "Away win", key: "away_win", color: "#79b8ff" },
  ];

  const outcomeLabels = { H: "Home win", D: "Draw", A: "Away win" };

  return (
    <main className="detail-shell">
      <button className="back-button" onClick={() => navigate("/")}>
        ← All fixtures
      </button>

      <div className="detail-kicker">FOOTBALLIQ / MATCH ANALYSIS</div>

      <section className="detail-hero">
        <div className="detail-meta">
          PREMIER LEAGUE · GAMEWEEK {match.matchday ?? "—"}
        </div>

        <h1>{displayName(match.home_team)}</h1>
        <div className="detail-versus">VS</div>
        <h1>{displayName(match.away_team)}</h1>

        <p>
          {new Date(match.kickoff).toLocaleString("en-IN", {
            weekday: "long",
            day: "numeric",
            month: "long",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </p>
      </section>

      <section className="detail-panel">
        <div className="section-kicker">MODEL FORECAST</div>
        <h2>Outcome probabilities<span>.</span></h2>
        <p className="detail-muted">
          The model's estimates for the three possible match outcomes.
        </p>

        <div className="detail-probabilities">
          {probabilities.map((item) => {
            const value = prediction.probabilities[item.key] ?? 0;

            return (
              <div className="detail-probability" key={item.key}>
                <div className="detail-probability-heading">
                  <span>{item.label}</span>
                  <strong>{(value * 100).toFixed(1)}%</strong>
                </div>

                <div className="detail-track">
                  <div
                    className="detail-fill"
                    style={{ width: `${value * 100}%`, background: item.color }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        <div className="predicted-result">
          <span>Most likely outcome according to this model</span>
          <strong>{outcomeLabels[prediction.predicted_outcome] ?? "Unknown"}</strong>
        </div>

        <p className="detail-disclaimer">
          Experimental prediction, not a guarantee. These probabilities have
          not been established as reliably calibrated.
        </p>
      </section>

      <section className="detail-panel">
        <div className="section-kicker">AVAILABLE DATA</div>
        <h2>Historical coverage<span>.</span></h2>
        <p className="detail-muted">
          Number of historical matches available to build each team's features.
        </p>

        <div className="history-grid">
          <div className="history-box">
            <span>{displayName(match.home_team)}</span>
            <strong>{history.home_matches_available}</strong>
            <small>HISTORICAL MATCHES</small>
          </div>

          <div className="history-box">
            <span>{displayName(match.away_team)}</span>
            <strong>{history.away_matches_available}</strong>
            <small>HISTORICAL MATCHES</small>
          </div>
        </div>

        {history.limited_history && (
          <div className="detail-warning">
            Limited historical coverage detected. One or both teams may be
            represented by fallback feature values, making this estimate less
            dependable.
          </div>
        )}
      </section>

      <button className="back-button bottom-back" onClick={() => navigate("/")}>
        ← Back to upcoming fixtures
      </button>
    </main>
  );
}
