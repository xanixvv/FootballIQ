
import { useEffect, useState } from "react";
import {
  Routes,
  Route,
  Link,
  useNavigate,
  useParams,
} from "react-router-dom";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const COMPETITIONS = {
  PL: {
    name: "Premier League",
    shortName: "PREMIER LEAGUE",
    description:
      "Explore upcoming English football fixtures, experimental outcome predictions, and team matchups.",
    theme: "pl",
  },
  CL: {
    name: "UEFA Champions League",
    shortName: "CHAMPIONS LEAGUE",
    description:
      "Explore upcoming European fixtures and follow the biggest continental matchups.",
    theme: "cl",
  },
};

function formatTeamName(name = "") {
  return name
    .replace(/ FC$/, "")
    .replace(/ AFC$/, "")
    .replace("Manchester City", "Man City")
    .replace("Manchester United", "Man United")
    .replace("Brighton & Hove Albion", "Brighton");
}

function formatKickoff(dateString) {
  if (!dateString) return "Kickoff TBA";

  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return "Kickoff TBA";

  return date.toLocaleString("en-IN", {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}


function matchesCompetition(liveMatch, competitionCode) {
  const league = (liveMatch.league || "").toLowerCase();
  const country = (liveMatch.country || "").toLowerCase();

  if (competitionCode === "PL") {
    return league === "premier league" && country === "england";
  }

  if (competitionCode === "CL") {
    return league.includes("champions league");
  }

  return false;
}

function formatPercentage(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return `${(number * 100).toFixed(1)}%`;
}

function getProbabilities(prediction) {
  const probabilities = prediction?.probabilities;
  if (!probabilities || typeof probabilities !== "object") return null;

  const values = [probabilities.home_win, probabilities.draw, probabilities.away_win];
  if (values.some((value) => !Number.isFinite(Number(value)))) return null;

  return {
    home_win: Number(probabilities.home_win),
    draw: Number(probabilities.draw),
    away_win: Number(probabilities.away_win),
  };
}

function getMarketProbabilities(odds) {
  if (!odds) return null;

  const values = [
    Number(odds.home),
    Number(odds.draw),
    Number(odds.away),
  ];

  if (values.some((value) => !Number.isFinite(value) || value <= 1)) {
    return null;
  }

  const implied = values.map((value) => 1 / value);
  const total = implied.reduce((sum, value) => sum + value, 0);

  return {
    home_win: implied[0] / total,
    draw: implied[1] / total,
    away_win: implied[2] / total,
  };
}

function formatOdds(value) {
  const odds = Number(value);
  return Number.isFinite(odds) && odds > 1
    ? odds.toFixed(2)
    : "—";
}

function getOutcomeLabel(outcome) {
  return {
    H: "Home win",
    D: "Draw",
    A: "Away win",
  }[outcome] ?? "Unavailable";
}

function Brand({ light = false }) {
  return (
    <Link className="brand" to="/" aria-label="FootballIQ home">
      <span className="brand-mark">F</span>
      <span>
        FOOTBALL<span className="brand-light">IQ</span>
      </span>
    </Link>
  );
}

function Header({ section = "FOOTBALL INTELLIGENCE" }) {
  return (
    <header className="topbar">
      <Brand />
      <nav className="main-nav" aria-label="Main navigation">
        <Link to="/">Overview</Link>
        <Link to="/competition/PL">Premier League</Link>
        <Link to="/competition/CL">Champions League</Link>
        <Link to="/live" className="nav-live"><span className="status-dot" /> Live Centre</Link>
        <Link to="/market-value">Market Value</Link>
      </nav>
      <div className="live-status header-status">
        <span className="status-dot" />
        {section}
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <Brand />
      <span>Built with data. Driven by the game.</span>
      <span>Predictions are estimates, not guarantees.</span>
    </footer>
  );
}

function PageShell({ children }) {
  return (
    <main className="app-shell">
      <Header />
      {children}
      <Footer />
    </main>
  );
}

function DashboardCard({
  eyebrow,
  title,
  description,
  to,
  theme,
  symbol,
  features,
  status,
}) {
  return (
    <Link
      to={to}
      className={`hub-card hub-card-${theme}`}
    >
      <div className="hub-card-top">
        <span className="hub-card-eyebrow">{eyebrow}</span>
        <span className="hub-card-icon" aria-hidden="true">
          {symbol}
        </span>
      </div>

      <div className="hub-card-content">
        <h2>{title}</h2>
        <p>{description}</p>
      </div>

      <div className="hub-card-features">
        {features.map((feature) => (
          <span key={feature}>{feature}</span>
        ))}
      </div>

      <div className="hub-card-bottom">
        <span className="hub-card-status">{status}</span>
        <span className="hub-card-arrow" aria-hidden="true">
          ↗
        </span>
      </div>
    </Link>
  );
}

function Dashboard() {
  return (
    <PageShell>
      <section className="hero dashboard-hero">
        <div className="eyebrow">
          <span className="eyebrow-line" />
          FOOTBALL INTELLIGENCE, REIMAGINED
        </div>

        <h1>
          Read the game.
          <br />
          <span>Discover what's next.</span>
        </h1>

        <p>
          Your football analytics hub for match predictions,
          European competitions, and player market intelligence.
        </p>

        <div className="hero-stats">
          <div>
            <strong>02</strong>
            <span>COMPETITIONS</span>
          </div>
          <div>
            <strong>02</strong>
            <span>EXPERIMENTAL PREDICTION MODELS</span>
          </div>
          <div>
            <strong>01</strong>
            <span>PLANNED VALUATION MODEL</span>
          </div>
        </div>
      </section>

      <section className="hub-section">
        <div className="section-heading">
          <div>
            <span className="section-kicker">YOUR CONTROL CENTRE</span>
            <h2>Explore FootballIQ<span>.</span></h2>
            <p>Choose an area to start exploring.</p>
          </div>
        </div>

        <div className="hub-grid">
          <DashboardCard
            eyebrow="ENGLISH FOOTBALL"
            title="Premier League"
            description="Explore upcoming fixtures, team matchups and experimental home-win, draw and away-win probabilities."
            to="/competition/PL"
            theme="pl"
            symbol="⚽"
            features={["Upcoming fixtures", "ML predictions", "Match analysis"]}
            status="Experimental predictions available"
          />

          <DashboardCard
            eyebrow="EUROPEAN FOOTBALL"
            title="UEFA Champions League"
            description="Explore upcoming European fixtures, experimental match predictions, and recent-form analysis."
            to="/competition/CL"
            theme="cl"
            symbol="✦"
            features={[
              "Upcoming fixtures",
              "Experimental predictions",
              "Match analysis",
            ]}
            status="Experimental predictions available"
          />

          <DashboardCard
            eyebrow="TRANSFER INTELLIGENCE"
            title="Player Market Value"
            description="Build towards player valuation using performance, age, position, playing time and transfer-market data."
            to="/market-value"
            theme="market"
            symbol="↗"
            features={["Player profiles", "Value estimates", "Transfer insights"]}
            status="Valuation model coming soon"
          />

          <DashboardCard
            eyebrow="LIVE FOOTBALL"
            title="Live Match Centre"
            description="Track real football scores, match minutes, team logos and competitions as matches happen."
            to="/live"
            theme="live"
            symbol="◉"
            features={["Live scores", "Match status", "Competition tracker"]}
            status="Connected to live data"
 />
        </div>
      </section>
    </PageShell>
  );
}

function ProbabilityBar({ prediction }) {
  const safeProbabilities = getProbabilities(prediction);
  if (!safeProbabilities) return null;

  const probabilities = [
    {
      label: "HOME",
      value: safeProbabilities.home_win,
      color: "#b6f36b",
    },
    {
      label: "DRAW",
      value: safeProbabilities.draw,
      color: "#f5c76b",
    },
    {
      label: "AWAY",
      value: safeProbabilities.away_win,
      color: "#79b8ff",
    },
  ];

  return (
    <div className="probability-section">
      <div className="probability-bar">
        {probabilities.map((item) => (
          <div
            key={item.label}
            title={`${item.label}: ${formatPercentage(item.value)}`}
            style={{
              width: `${Math.max(0, Number(item.value ?? 0) * 100)}%`,
              backgroundColor: item.color,
            }}
          />
        ))}
      </div>

      <div className="probability-labels">
        {probabilities.map((item) => (
          <div key={item.label}>
            <span style={{ color: item.color }}>{item.label}</span>
            <strong>{formatPercentage(item.value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function BookmakerOdds({ odds, marketProbabilities }) {
  if (!odds) {
    return (
      <div className="bookmaker-section">
        <div className="bookmaker-heading">
          <span>BOOKMAKER ODDS</span>
          <span className="odds-unavailable">Unavailable</span>
        </div>
        <p className="form-empty">
          No matching bookmaker odds for this fixture.
        </p>
      </div>
    );
  }

  const market = marketProbabilities ?? getMarketProbabilities(odds);

  return (
    <div className="bookmaker-section">
      <div className="bookmaker-heading">
        <span>BOOKMAKER ODDS</span>
        <span className="odds-source">
          {odds.source ?? "Bookmaker"}
        </span>
      </div>

      <div className="odds-grid">
        {[
          {
            label: "HOME",
            value: odds.home,
            probability: market?.home_win,
          },
          {
            label: "DRAW",
            value: odds.draw,
            probability: market?.draw,
          },
          {
            label: "AWAY",
            value: odds.away,
            probability: market?.away_win,
          },
        ].map((item) => (
          <div className="odds-item" key={item.label}>
            <span>{item.label}</span>
            <strong>{formatOdds(item.value)}</strong>
            {item.probability != null && (
              <small>{formatPercentage(item.probability)}</small>
            )}
          </div>
        ))}
      </div>

      <p className="odds-disclaimer">
        Market probabilities are derived from decimal odds and
        normalised to remove the bookmaker margin. They are estimates,
        not guarantees.
      </p>
    </div>
  );
}

function HistoryWarning({ history, compact = false }) {
  if (!history) return null;

  const homeCount = Number(history.home_matches_available ?? 0);
  const awayCount = Number(history.away_matches_available ?? 0);
  const coldStart = history.cold_start ?? (homeCount === 0 || awayCount === 0);
  const limitedHistory = history.limited_history ?? (homeCount < 5 || awayCount < 5);
  if (!coldStart && !limitedHistory) return null;

  const level = history.warning_level ?? (coldStart ? "high" : "moderate");
  const message = history.warning_message ?? (
    coldStart
      ? "Cold start: no prior UCL matches for one or both teams were found in FootballIQ's available historical dataset. The prediction is still shown, but has very limited form evidence."
      : "Limited UCL history: one or both teams have fewer than five available prior matches. Treat this prediction with extra caution."
  );

  return (
    <aside className={`ucl-history-warning ucl-history-warning-${level}${compact ? " ucl-history-warning-compact" : ""}`} role="note">
      <strong>{coldStart ? "⚠ Cold-start prediction" : "⚠ Limited historical data"}</strong>
      <p>{message}</p>
      <div className="ucl-history-counts">
        <span>Home: {homeCount}/5 prior matches</span>
        <span>Away: {awayCount}/5 prior matches</span>
      </div>
    </aside>
  );
}

function MatchCard({ match, competitionCode }) {
  const navigate = useNavigate();
  const prediction = match.prediction;
  const competition = COMPETITIONS[competitionCode];

  function openMatch() {
    navigate(`/match/${competitionCode}/${match.id}`);
  }

  return (
    <article
      className="match-card clickable-card"
      onClick={openMatch}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openMatch();
        }
      }}
      role="link"
      tabIndex={0}
      aria-label={`View analysis for ${match.home_team} versus ${match.away_team}`}
    >
      <div className="match-meta">
        <span>{competition?.shortName ?? match.competition}</span>
        <span>GW {match.matchday ?? "—"}</span>
      </div>

      <div className="kickoff">{formatKickoff(match.kickoff)}</div>

      <div className="teams">
        <div className="team">
          <div className="team-avatar">
            {formatTeamName(match.home_team).charAt(0)}
          </div>
          <span>{formatTeamName(match.home_team)}</span>
          <small>HOME</small>
        </div>

        <div className="versus">VS</div>

        <div className="team">
          <div className="team-avatar">
            {formatTeamName(match.away_team).charAt(0)}
          </div>
          <span>{formatTeamName(match.away_team)}</span>
          <small>AWAY</small>
        </div>
      </div>

      {prediction ? (
        <>
          <div className="prediction-heading">
            <span>MODEL PREDICTION</span>
            <strong>
              {getOutcomeLabel(prediction.predicted_outcome)}
            </strong>
          </div>
         <p className="prediction-label">
  {prediction.model_type ??
    (prediction.model_version === "footballiq_form_odds_model.joblib"
      ? "Form + bookmaker odds model"
      : "Form-based model")}
</p>

<BookmakerOdds
  odds={prediction.odds}
  marketProbabilities={prediction.market_probabilities}
/>
{competitionCode === "CL" && (
  <HistoryWarning history={match.history} compact />
)}

          <ProbabilityBar prediction={prediction} />

          {competitionCode === "PL" && match.history?.limited_history && (
            <p className="history-warning">
              Limited historical data — this estimate may be less reliable.
            </p>
          )}
        </>
      ) : (
        <div className="prediction-heading">
          <span>MATCH PREDICTION</span>
          <strong>{match.prediction_error ? "Prediction unavailable" : "Coming soon"}</strong>
          <p>
            {match.prediction_error
              ? `FootballIQ couldn't generate this prediction: ${match.prediction_error}`
              : "A competition-specific model is not available yet."}
          </p>
          {competitionCode === "CL" && <HistoryWarning history={match.history} compact />}
        </div>
      )}

      <div className="match-card-link">
        Explore match <span>→</span>
      </div>
    </article>
  );
}

function isUpcomingFixture(match) {
  const kickoff = new Date(match.kickoff).getTime();

  if (!Number.isFinite(kickoff) || kickoff < Date.now()) {
    return false;
  }

  const status = String(match.status ?? "").toUpperCase();

  return ![
    "FINISHED",
    "COMPLETED",
    "CLOSED",
    "CANCELLED",
    "POSTPONED",
    "ABANDONED",
  ].includes(status);
}

function CompetitionPage() {
  const { competitionCode: rawCode } = useParams();
  const competitionCode = (rawCode ?? "").toUpperCase();
  const competition = COMPETITIONS[competitionCode];

  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [apiNote, setApiNote] = useState("");
  
const [liveMatches, setLiveMatches] = useState([]);
const [liveLoading, setLiveLoading] = useState(false);
const [liveError, setLiveError] = useState("");
const [liveUpdatedAt, setLiveUpdatedAt] = useState(null);


  useEffect(() => {
    let cancelled = false;

    async function loadMatches() {
      setLoading(true);
      setError("");
      setApiNote("");
      setMatches([]);

      try {
        const response = await fetch(
          `${API_URL}/fixtures/predictions?competition=${encodeURIComponent(competitionCode)}`
        );

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : `Backend returned HTTP ${response.status}`
          );
        }

        if (!cancelled) {
          setMatches(data.matches ?? []);
          setApiNote(data.note ?? "");
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err.message ||
              "Couldn't load fixtures. Make sure the FastAPI backend is running."
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    if (competition) {
      loadMatches();
    } else {
      setLoading(false);
    }

    return () => {
      cancelled = true;
    };
  }, [competitionCode, competition]);

  

  async function loadLiveMatches() {
    setLiveLoading(true);
    setLiveError("");

    try {
      const response = await fetch(`${API_URL}/live/predictions`);
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Could not load live matches."
        );
      }

      setLiveMatches(
        (data.matches || []).filter((match) =>
          matchesCompetition(match, competitionCode)
        )
      );

      setLiveUpdatedAt(new Date());
    } catch (err) {
      setLiveError(err.message || "Could not load live matches.");
    } finally {
      setLiveLoading(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function loadSafely() {
      if (cancelled) return;
      await loadLiveMatches();
    }

    loadSafely();

    const refreshInterval = setInterval(() => {
      if (!cancelled) loadSafely();
    }, 60_000);

    return () => {
      cancelled = true;
      clearInterval(refreshInterval);
    };
  }, [competitionCode]);


  if (!competition) {
    return (
      <PageShell>
        <section className="message-panel">
          <h2>Competition not found</h2>
          <p>Choose a competition from the dashboard.</p>
          <Link className="back-button" to="/">
            ← Back to FootballIQ
          </Link>
        </section>
      </PageShell>
    );
  }


const upcomingMatches = matches.filter(isUpcomingFixture);

const query = search.trim().toLowerCase();

const filteredMatches = upcomingMatches.filter((match) =>
  `${match.home_team} ${match.away_team}`
    .toLowerCase()
    .includes(query)
);

  const visibleMatches = showAll
    ? filteredMatches
    : filteredMatches.slice(0, 6);

  return (
    <PageShell>
      <section className={`competition-hero competition-hero-${competition.theme}`}>
        <Link className="back-button" to="/">
          ← Dashboard
        </Link>

        <div className="eyebrow">
          <span className="eyebrow-line" />
          {competition.shortName}
        </div>

        <h1>
          {competition.name}
          <br />
          <span>Match centre.</span>
        </h1>

        <p>{competition.description}</p>

        <div className="hero-stats">
          <div>
            <strong>{loading ? "—" : upcomingMatches.length}</strong>
            <span>UPCOMING FIXTURES</span>
          </div>
          <div>
            <strong>{competitionCode === "PL" ? "ML" : "—"}</strong>
            <span>
              {competitionCode === "PL"
                ? "EXPERIMENTAL MODEL"
                : "MODEL IN DEVELOPMENT"}
            </span>
          </div>
          <div>
            <strong>14</strong>
            <span>DAY LOOKAHEAD</span>
          </div>
        </div>
      </section>

      
      <section className="live-section">
        <div className="live-section-heading">
          <div>
            <span className="live-indicator" />
            <h2>Live Now</h2>
          </div>

          <button
            type="button"
            onClick={loadLiveMatches}
            disabled={liveLoading}
          >
            {liveLoading ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        {liveError && (
          <p className="live-message">{liveError}</p>
        )}

        {!liveError && liveMatches.length === 0 && (
          <p className="live-message">
            {liveLoading
              ? "Checking for live matches..."
              : "No matches from this competition are live right now."}
          </p>
        )}

        <div className="live-match-grid">
          {liveMatches.map((match) => (
            <article
              className="live-match-card"
              key={match.fixture_id}
            >
              <p className="live-league">
                {match.league} · {match.country}
              </p>

              <div className="live-team-row">
                {match.home_logo && (
                  <img src={match.home_logo} alt="" />
                )}
                <span>{formatTeamName(match.home_team)}</span>
                <strong>{match.home_goals ?? 0}</strong>
              </div>

              <div className="live-team-row">
                {match.away_logo && (
                  <img src={match.away_logo} alt="" />
                )}
                <span>{formatTeamName(match.away_team)}</span>
                <strong>{match.away_goals ?? 0}</strong>
              </div>

              <p className="live-match-status">
                {match.minute != null
                  ? `${match.minute}' · ${match.status || "Live"}`
                  : match.status || "Live"}
              </p>

              {match.in_play_prediction && (
                <div className="inplay-prediction">
                  <h4>In-play outcome estimate</h4>

                  <div className="inplay-probabilities">
                    <div>
                      <strong>
                        {Math.round(
                          match.in_play_prediction.home * 100
                        )}%
                      </strong>
                      <span>Home</span>
                    </div>

                    <div>
                      <strong>
                        {Math.round(
                          match.in_play_prediction.draw * 100
                        )}%
                      </strong>
                      <span>Draw</span>
                    </div>

                    <div>
                      <strong>
                        {Math.round(
                          match.in_play_prediction.away * 100
                        )}%
                      </strong>
                      <span>Away</span>
                    </div>
                  </div>

                  <p className="experimental-warning">
                    Experimental baseline — not validated ML
                  </p>

                  <h4>Prediction timeline</h4>

                  {match.prediction_history?.length ? (
                    <div className="prediction-timeline">
                      {match.prediction_history.map(
                        (snapshot, index) => (
                          <div
                            className="prediction-snapshot"
                            key={`${match.fixture_id}-${snapshot.minute}-${index}`}
                          >
                            <span>
                              {snapshot.minute}' ·{" "}
                              {snapshot.home_goals}–
                              {snapshot.away_goals}
                            </span>

                            <span>
                              H {Math.round(snapshot.home_win * 100)}%
                              {" · "}
                              D {Math.round(snapshot.draw * 100)}%
                              {" · "}
                              A {Math.round(snapshot.away_win * 100)}%
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <p className="live-message">
                      The first snapshot will appear when match data
                      is recorded.
                    </p>
                  )}
                </div>
              )}
            </article>
          ))}
        </div>

        {liveUpdatedAt && (
          <p className="live-updated">
            Last checked: {liveUpdatedAt.toLocaleTimeString("en-IN")}
          </p>
        )}
      </section>


      <section className="fixtures-section">
        <div className="section-heading">
          <div>
            <span className="section-kicker">THE MATCH CENTRE</span>
            <h2>Upcoming fixtures<span>.</span></h2>
            <p>Search teams and open a match to see its available analysis.</p>
          </div>

          <label className="search-box">
            <span>⌕</span>
            <input
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setShowAll(false);
              }}
              placeholder="Search teams..."
              aria-label="Search teams"
            />
          </label>
        </div>

        {loading && (
          <div className="message-panel" role="status" aria-live="polite">
            Loading {competition.name} fixtures...
          </div>
        )}

        {error && (
          <div className="message-panel error">
            <h3>Couldn't load fixtures</h3>
            <p>{error}</p>
            <button
              className="show-more"
              onClick={() => window.location.reload()}
            >
              Try again
            </button>
          </div>
        )}

        {!loading && !error && matches.length === 0 && (
          <div className="message-panel">
            No upcoming fixtures were returned for this competition in the selected date range.
          </div>
        )}

        {!loading && !error && matches.length > 0 && (
          <>
            <div className="results-line">
              <span>
                SHOWING {visibleMatches.length} OF {filteredMatches.length} MATCHES
              </span>
             <span className="experimental-tag">
              {matches.some((match) => Boolean(match.prediction))
                ? "EXPERIMENTAL PREDICTIONS"
                : "FIXTURES ONLY"}
            </span>
            </div>

            {apiNote && (
              <p className="history-warning">{apiNote}</p>
            )}

            {filteredMatches.length === 0 ? (
              <div className="message-panel">
                No teams match that search. Try another team name.
              </div>
            ) : (
              <div className="match-grid">
                {visibleMatches.map((match) => (
                  <MatchCard
                    key={`${competitionCode}-${match.id}`}
                    match={match}
                    competitionCode={competitionCode}
                  />
                ))}
              </div>
            )}

            {filteredMatches.length > 6 && (
              <button
                className="show-more"
                onClick={() => setShowAll((previous) => !previous)}
              >
                {showAll
                  ? "Show fewer matches ↑"
                  : "Explore all fixtures →"}
              </button>
            )}
          </>
        )}
      </section>
    </PageShell>
  );
}

function FormResults({ results = [] }) {
  if (results.length === 0) {
    return (
      <p className="form-empty">
        Not enough historical data to show recent form.
      </p>
    );
  }

  return (
    <div className="form-results" aria-label="Recent match results">
      {results.map((result, index) => {
        const normalized = String(result).toUpperCase();
        const label =
          normalized === "W"
            ? "Win"
            : normalized === "D"
              ? "Draw"
              : "Loss";

        return (
          <span
            key={`${normalized}-${index}`}
            className={`form-result form-result-${normalized.toLowerCase()}`}
            title={label}
            aria-label={label}
          >
            {normalized}
          </span>
        );
      })}
    </div>
  );
}

function TeamFormCard({ form, title, venueLabel }) {
  const hasHistory = (form?.matches_played ?? 0) > 0;

  return (
    <article className="team-form-card">
      <div className="team-form-heading">
        <div>
          <span className="section-kicker">{venueLabel}</span>
          <h3>{title}</h3>
        </div>
        <span className="form-count">
          {form?.matches_played ?? 0} / 5 matches
        </span>
      </div>

      <FormResults results={form?.form ?? []} />

      {hasHistory ? (
        <>
          <div className="team-form-stats">
            <div>
              <span>Points</span>
              <strong>{form.points}</strong>
            </div>
            <div>
              <span>Goals scored / game</span>
              <strong>
                {form.average_goals_scored ?? "—"}
              </strong>
            </div>
            <div>
              <span>Goals conceded / game</span>
              <strong>
                {form.average_goals_conceded ?? "—"}
              </strong>
            </div>
          </div>

          {form.matches_played < 5 && (
            <p className="form-empty">
              Based on only {form.matches_played} available matches.
            </p>
          )}
        </>
      ) : (
        <p className="form-empty">
          No usable historical matches found in the current dataset.
          This does not mean the team has zero points or poor form.
        </p>
      )}
    </article>
  );
}

function TeamFormSection({ match }) {
  const teamForm = match?.team_form;

  if (!teamForm) {
    return (
      <section className="detail-panel">
        <span className="section-kicker">TEAM FORM</span>
        <h2>Form data unavailable</h2>
        <p>
          The backend did not return recent-form data for this fixture.
        </p>
      </section>
    );
  }

  return (
    <section className="detail-panel">
      <span className="section-kicker">LAST FIVE MATCHES</span>
      <h2>Form & performance<span>.</span></h2>
      <p>
        Recent results and average goals from the historical matches
        available to FootballIQ.
      </p>

      <div className="team-form-grid">
        <TeamFormCard
          form={teamForm.home}
          title={formatTeamName(match.home_team)}
          venueLabel="HOME TEAM · ALL VENUES"
        />

        <TeamFormCard
          form={teamForm.away}
          title={formatTeamName(match.away_team)}
          venueLabel="AWAY TEAM · ALL VENUES"
        />
      </div>

      <div className="venue-form-section">
        <span className="section-kicker">VENUE-SPECIFIC PERFORMANCE</span>
        <h3>At home vs away<span>.</span></h3>

        <div className="team-form-grid">
          <TeamFormCard
            form={teamForm.home_venue}
            title={formatTeamName(match.home_team)}
            venueLabel="HOME TEAM · HOME MATCHES"
          />

          <TeamFormCard
            form={teamForm.away_venue}
            title={formatTeamName(match.away_team)}
            venueLabel="AWAY TEAM · AWAY MATCHES"
          />
        </div>
      </div>
    </section>
  );
}

function MatchDetailsPage() {
  const { competitionCode: rawCode, matchId } = useParams();
  const competitionCode = (rawCode ?? "").toUpperCase();
  const competition = COMPETITIONS[competitionCode];

  const [match, setMatch] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadMatch() {
      setLoading(true);
      setError("");
      setMatch(null);

      try {
        const response = await fetch(
          `${API_URL}/fixtures/predictions?competition=${encodeURIComponent(competitionCode)}`
        );

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : `Backend returned HTTP ${response.status}`
          );
        }

        const selectedMatch = (data.matches ?? []).find(
          (item) => String(item.id) === String(matchId)
        );

        if (!selectedMatch) {
          throw new Error(
            "This match wasn't found in the current upcoming fixtures. It may have moved outside the API's date range."
          );
        }

        if (!cancelled) setMatch(selectedMatch);
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Couldn't load this match.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    if (competition) loadMatch();
    else {
      setError("Competition not found.");
      setLoading(false);
    }

    return () => {
      cancelled = true;
    };
  }, [competitionCode, competition, matchId]);

  return (
    <PageShell>
      <section className="detail-shell">
        <Link
          className="back-button"
          to={`/competition/${competitionCode}`}
        >
          ← Back to {competition?.name ?? "fixtures"}
        </Link>

        {loading && (
          <div className="message-panel" role="status" aria-live="polite">Loading match details...</div>
        )}

        {error && (
          <div className="message-panel error">
            <h2>Unable to open match</h2>
            <p>{error}</p>
          </div>
        )}

        {!loading && !error && match && (
          <>
            <div className="detail-header">
              <span className="section-kicker">
                {match.competition ?? competition.name}
              </span>
              <h1>Match centre<span>.</span></h1>
              <p>{formatKickoff(match.kickoff)}</p>
            </div>

            <div className="detail-matchup">
              <div className="detail-team">
                <div className="detail-team-avatar">
                  {formatTeamName(match.home_team).charAt(0)}
                </div>
                <h2>{formatTeamName(match.home_team)}</h2>
                <span>HOME</span>
              </div>

              <div className="detail-versus">VS</div>

              <div className="detail-team">
                <div className="detail-team-avatar">
                  {formatTeamName(match.away_team).charAt(0)}
                </div>
                <h2>{formatTeamName(match.away_team)}</h2>
                <span>AWAY</span>
              </div>
            </div>

            <section className="detail-panel">
              <span className="section-kicker">OUTCOME ANALYSIS</span>

              {match.prediction ? (
                <>
                  <h2>
                    {getOutcomeLabel(match.prediction.predicted_outcome)}
                  </h2>
                  <p>
                    Experimental machine-learning estimate for this match.
                    These probabilities are not guarantees.
                  </p>
                  <ProbabilityBar prediction={match.prediction} />
                  <BookmakerOdds
  odds={match.prediction.odds}
  marketProbabilities={match.prediction.market_probabilities}
/>

                  {getProbabilities(match.prediction) ? (
                    <div className="detail-probability-grid">
                      <div>
                        <span>Home win</span>
                        <strong>{formatPercentage(getProbabilities(match.prediction).home_win)}</strong>
                      </div>
                      <div>
                        <span>Draw</span>
                        <strong>{formatPercentage(getProbabilities(match.prediction).draw)}</strong>
                      </div>
                      <div>
                        <span>Away win</span>
                        <strong>{formatPercentage(getProbabilities(match.prediction).away_win)}</strong>
                      </div>
                    </div>
                  ) : (
                    <p className="history-warning">
                      The prediction is present, but its probability values are missing or invalid.
                    </p>
                  )}

                  {competitionCode === "CL" ? (
                    <HistoryWarning history={match.history} />
                  ) : match.history && (
                    <p className="history-warning">
                      Historical matches available — home: {" "}
                      {match.history.home_matches_available}, away: {" "}
                      {match.history.away_matches_available}.
                      {match.history.limited_history ? " Historical coverage is limited." : ""}
                    </p>
                  )}
                </>
              ) : (
                <>
                  <h2>Predictions coming soon</h2>
                  <p>
                    FootballIQ has the real fixture, but a suitable
                    competition-specific prediction model is not available
                    yet. We won't apply the Premier League model to this
                    competition and present it as reliable.
                  </p>
                </>
              )}
            </section>
            
            <TeamFormSection match={match} />


            <section className="detail-panel">
              <span className="section-kicker">FIXTURE INFORMATION</span>
              <div className="detail-info-row">
                <span>Competition</span>
                <strong>{match.competition ?? competition.name}</strong>
              </div>
              <div className="detail-info-row">
                <span>Matchday</span>
                <strong>{match.matchday ?? "Not available"}</strong>
              </div>
              <div className="detail-info-row">
                <span>Status</span>
                <strong>{match.status ?? "Not available"}</strong>
              </div>
            </section>
          </>
        )}
      </section>
    </PageShell>
  );
}


function MarketValuePage() {
  const [query, setQuery] = useState("");
  const [players, setPlayers] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [currentValue, setCurrentValue] = useState(null);
  const [latestReported, setLatestReported] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [direction, setDirection] = useState(null);
  const [searching, setSearching] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function searchPlayers(event) {
    event.preventDefault();

    if (!query.trim()) return;

    setSearching(true);
    setError("");
    setPlayers([]);
    setSelectedPlayer(null);
    setCurrentValue(null);
    setForecast(null);
    setDirection(null);
    setLatestReported(null);

    try {
      const response = await fetch(
        `${API_URL}/players/search?q=${encodeURIComponent(query.trim())}`
      );

      if (!response.ok) {
        throw new Error("Player search failed. Check that the backend is running.");
      }

      const data = await response.json();
      const results = data.players ?? data.results ?? [];

      setPlayers(results);

      if (results.length === 0) {
        setError("No players found. Try another name.");
      }
    } catch (err) {
      setError(err.message || "Unable to search players.");
    } finally {
      setSearching(false);
    }
  }

  async function selectPlayer(player) {
    const playerId = player.player_id ?? player.id;

    if (playerId == null) {
      setError("This search result has no player ID.");
      return;
    }

    setSelectedPlayer(player);
    setLoading(true);
    setError("");
    setCurrentValue(null);
    setForecast(null);
    setDirection(null);
    setPlayers([]);
    setLatestReported(null);

    const endpoints = [
      ["current", `/players/${playerId}/market-value`],
      ["latestReported", `/players/${playerId}/market-value/latest-reported`],
      ["forecast", `/players/${playerId}/market-value/forecast`],
      ["direction", `/players/${playerId}/market-value/direction`],
    ];

    try {
      const results = await Promise.all(
        endpoints.map(async ([key, path]) => {
          try {
            const response = await fetch(`${API_URL}${path}`);

            if (!response.ok) {
              const body = await response.json().catch(() => ({}));
              return {
                key,
                data: null,
                error: body.detail || `Request failed (${response.status})`,
              };
            }

            return { key, data: await response.json(), error: null };
          } catch {
            return { key, data: null, error: "Could not connect to the API." };
          }
        })
      );

      for (const result of results) {
  if (result.key === "current") setCurrentValue(result.data);
  if (result.key === "latestReported") {
    setLatestReported(result.data);
  }
  if (result.key === "forecast") setForecast(result.data);
  if (result.key === "direction") setDirection(result.data);
}

      const failures = results.filter((result) => !result.data);
      if (failures.length === results.length) {
        setError("Could not load player valuation data. Check the backend.");
      }
    } finally {
      setLoading(false);
    }
  }

  function formatMoney(value) {
    if (value == null || !Number.isFinite(Number(value))) return "Unavailable";

    return new Intl.NumberFormat("en-GB", {
      style: "currency",
      currency: "EUR",
      maximumFractionDigits: 0,
    }).format(Number(value));
  }

  function findValue(data, keys) {
    if (!data) return null;

    for (const key of keys) {
      if (data[key] != null) return data[key];
    }

    return null;
  }

  function displayDirection(data) {
    const value = findValue(data, [
      "predicted_direction",
      "direction",
      "prediction",
      "predicted_class",
    ]);

    if (value == null) return "Unavailable";

    return String(value).replaceAll("_", " ");
  }

  function displayDate(data) {
    return findValue(data, [
      "feature_data_date",
      "data_date",
      "latest_valuation_date",
      "valuation_date",
      "training_cutoff",
    ]) ?? "Date not supplied by API";
  }

  function displayError(data) {
    return data?.detail ?? data?.error ?? null;
  }

  function calculateAge(dateOfBirth) {
  if (!dateOfBirth) return null;

  const birthDate = new Date(dateOfBirth);
  if (Number.isNaN(birthDate.getTime())) return null;

  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();

  const birthdayHasPassed =
    today.getMonth() > birthDate.getMonth() ||
    (today.getMonth() === birthDate.getMonth() &&
      today.getDate() >= birthDate.getDate());

  if (!birthdayHasPassed) age--;

  return age >= 0 ? age : null;
}

  return (
    <PageShell>
    <div className="market-value-page">
      <section className="market-value-hero">
        <p className="eyebrow">FOOTBALLIQ · PLAYER ANALYTICS</p>
        <h1>What is a player really worth?</h1>
        <p>
          Explore recorded market values, FootballIQ estimates, future
          forecasts, and predicted valuation direction.
        </p>

        <form onSubmit={searchPlayers} className="player-search-form">
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search a player, e.g. Erling Haaland"
            aria-label="Search players"
          />
          <button type="submit" disabled={searching}>
            {searching ? "Searching..." : "Search"}
          </button>
        </form>

        {error && <p role="alert" className="error-message">{error}</p>}

       {players.length > 0 && (
  <div className="player-search-results">
    {players.map((player) => (
      <button
        type="button"
        key={player.player_id ?? player.id}
        onClick={() => selectPlayer(player)}
        className="player-result"
      >
        <span className="player-result-avatar">
          {player.image_url ? (
            <img
              src={player.image_url}
              alt=""
              loading="lazy"
              onError={(event) => {
                event.currentTarget.style.display = "none";
              }}
            />
          ) : (
            <span className="player-avatar-fallback">
              {(player.name || "?").charAt(0)}
            </span>
          )}
        </span>

        <span className="player-result-info">
          <strong>
            {player.name ?? player.player_name ?? "Unknown player"}
          </strong>

          <span className="player-result-meta">
            {[
              player.current_club,
              player.sub_position || player.position,
              player.nationality,
            ]
              .filter(Boolean)
              .join(" · ")}
          </span>
        </span>

        <span className="player-result-value">
          {player.current_market_value != null
            ? formatMoney(player.current_market_value)
            : "Value unavailable"}
          <span className="player-result-arrow">↗</span>
        </span>
      </button>
    ))}
  </div>
)}

        {!selectedPlayer && players.length === 0 && !searching && !error && (
          <div className="market-discovery">
            <div className="market-discovery-copy">
              <span className="market-section-kicker">THE TRANSFER LAB</span>
              <h2>More than a price tag.</h2>
              <p>Explore a player's profile, recorded market value, FootballIQ estimates and future valuation signals in one report.</p>
              <div className="market-quick-searches">
                {["Erling Haaland", "Kylian Mbappé", "Jude Bellingham", "Lamine Yamal"].map((name) => (
                  <button key={name} type="button" onClick={() => setQuery(name)}>{name} <span>↗</span></button>
                ))}
              </div>
            </div>
            <div className="market-feature-grid">
              <article><span className="market-feature-icon">↗</span><small>01 / VALUATION</small><strong>Market value</strong><p>Compare the latest recorded value with FootballIQ's experimental estimate.</p></article>
              <article><span className="market-feature-icon">⌁</span><small>02 / PROJECTION</small><strong>Future outlook</strong><p>Review forecast direction and the dates behind each model output.</p></article>
              <article><span className="market-feature-icon">◉</span><small>03 / PLAYER DNA</small><strong>Role &amp; zones</strong><p>Understand a player's position and explore a pitch-zone visual.</p></article>
            </div>
          </div>
        )}
      </section>

      {selectedPlayer && (
        <section className="player-valuation-results">
         <div className="player-profile-heading">
  <div className="player-profile-identity">
    {selectedPlayer.image_url ? (
      <img
        className="player-profile-photo"
        src={selectedPlayer.image_url}
        alt={selectedPlayer.name || "Player"}
        onError={(event) => {
          event.currentTarget.style.display = "none";
        }}
      />
    ) : (
      <div className="player-profile-photo player-avatar-fallback">
        {(selectedPlayer.name || "?").charAt(0)}
      </div>
    )}

    <div className="player-profile-info">
      <p className="eyebrow">FOOTBALLIQ PLAYER REPORT</p>

      <h2>
        {selectedPlayer.name ??
          selectedPlayer.player_name ??
          `Player ${selectedPlayer.player_id ?? selectedPlayer.id}`}
      </h2>

      <p className="player-profile-subtitle">
        {[
          selectedPlayer.current_club,
          selectedPlayer.sub_position || selectedPlayer.position,
          selectedPlayer.nationality,
        ]
          .filter(Boolean)
          .join(" · ")}
      </p>

      <div className="player-profile-tags">
        {calculateAge(selectedPlayer.date_of_birth) != null && (
          <span>{calculateAge(selectedPlayer.date_of_birth)} years old</span>
        )}

        {selectedPlayer.foot && (
          <span>{selectedPlayer.foot}-footed</span>
        )}

        {selectedPlayer.height != null && (
          <span>{selectedPlayer.height} cm</span>
        )}
      </div>

      <p className="player-profile-recorded-value">
        Latest recorded value:{" "}
        <strong>
          {formatMoney(selectedPlayer.current_market_value)}
        </strong>
      </p>
    </div>
  </div>

  <button
    type="button"
    className="player-profile-back"
    onClick={() => {
      setSelectedPlayer(null);
      setCurrentValue(null);
      setForecast(null);
      setDirection(null);
      setLatestReported(null);
      setError("");
    }}
  >
    ← Back to search
  </button>
</div>

          {loading && <p>Loading valuation models...</p>}

          <div className="valuation-cards">
            
<article className="valuation-card latest-reported-card">
  <p>Latest reported market value</p>

  <h3>
    {latestReported
      ? formatMoney(
          latestReported.latest_reported_market_value_eur
        )
      : "Unavailable"}
  </h3>

  <p>
    Valuation date:{" "}
    {latestReported?.valuation_date ?? "Date unavailable"}
  </p>

  {latestReported?.data_age_days != null && (
    <p>
      Record age: {latestReported.data_age_days.toLocaleString()} days
    </p>
  )}

  <small>
    Historical recorded value from FootballIQ's stored dataset.
    This is not a live market value.
  </small>
</article>

            <article className="valuation-card">
              <p>FootballIQ estimated value</p>
              <h3>
                {currentValue
                  ? formatMoney(findValue(currentValue, [
                      "estimated_market_value",
                      "predicted_market_value",
                      "predicted_value_eur",
                    ]))
                  : "Unavailable"}
              </h3>
              <p>
                Data date: {displayDate(currentValue)}
              </p>
              <small>Experimental ML estimate, not an official valuation.</small>
              {displayError(currentValue) && (
                <p>{displayError(currentValue)}</p>
              )}
            </article>

            <article className="valuation-card">
              <p>Future valuation forecast</p>
              <h3>
                {forecast
  ? formatMoney(findValue(forecast, [
      "forecast_market_value_eur",
      "forecast_market_value",
      "predicted_market_value",
      "predicted_value_eur",
      "predicted_value",
    ]))
  : "Unavailable"}
              </h3>
              <p>
                Data date: {displayDate(forecast)}
              </p>
              <small>
                Forecast target: the next recorded valuation approximately
                90–120 days after the starting snapshot.
              </small>
              {displayError(forecast) && <p>{displayError(forecast)}</p>}
            </article>

            <article className="valuation-card">
              <p>Predicted value direction</p>
              <h3>{direction ? displayDirection(direction) : "Unavailable"}</h3>
              <p>Data date: {displayDate(direction)}</p>
              <small>
                A model prediction, not a guarantee of future market movement.
              </small>
              {displayError(direction) && <p>{displayError(direction)}</p>}
            </article>
          </div>

          <section className="player-analysis-grid">
            <article className="player-pitch-panel">
              <div className="player-panel-heading">
                <div><span className="market-section-kicker">PLAYER DNA</span><h3>Position &amp; zones</h3></div>
                <span className="pitch-illustration-tag">POSITIONAL ILLUSTRATION</span>
              </div>
              <p className="player-panel-description">A role-based visual based on the listed position, not measured match tracking or event data.</p>
              <div className="player-pitch-wrap">
                <svg className="player-pitch" viewBox="0 0 640 390" role="img" aria-label={`Illustrative football pitch for ${selectedPlayer.sub_position || selectedPlayer.position || "player position"}`}>
                  <defs>
                    <radialGradient id="zoneGlow"><stop offset="0%" stopColor="#54f0c2" stopOpacity=".88"/><stop offset="100%" stopColor="#54f0c2" stopOpacity="0"/></radialGradient>
                  </defs>
                  <rect x="10" y="10" width="620" height="370" rx="12" fill="#0b4839" stroke="#75c9a8" strokeWidth="2"/>
                  <path d="M320 10 V380 M10 195 H630" stroke="#75c9a8" strokeOpacity=".55" strokeWidth="2" fill="none"/>
                  <circle cx="320" cy="195" r="58" fill="none" stroke="#75c9a8" strokeOpacity=".65" strokeWidth="2"/>
                  <circle cx="320" cy="195" r="3" fill="#a8f5d8"/>
                  <rect x="10" y="105" width="95" height="180" fill="none" stroke="#75c9a8" strokeOpacity=".7" strokeWidth="2"/>
                  <rect x="535" y="105" width="95" height="180" fill="none" stroke="#75c9a8" strokeOpacity=".7" strokeWidth="2"/>
                  <rect x="10" y="150" width="35" height="90" fill="none" stroke="#75c9a8" strokeOpacity=".7" strokeWidth="2"/>
                  <rect x="595" y="150" width="35" height="90" fill="none" stroke="#75c9a8" strokeOpacity=".7" strokeWidth="2"/>
                  <ellipse cx="465" cy="165" rx="105" ry="100" fill="url(#zoneGlow)"/>
                  <ellipse cx="505" cy="230" rx="75" ry="65" fill="url(#zoneGlow)" opacity=".72"/>
                  <text x="320" y="34" fill="#c5f8e5" fontSize="12" textAnchor="middle" letterSpacing="2">ATTACKING HALF</text>
                  <text x="320" y="365" fill="#c5f8e5" fontSize="12" textAnchor="middle" letterSpacing="2">DEFENSIVE HALF</text>
                </svg>
              </div>
              <div className="pitch-legend"><span><i /> Illustrative role zone</span><span>Actual heatmap requires match event data</span></div>
            </article>
            <article className="player-insights-panel">
              <span className="market-section-kicker">REPORT GUIDE</span>
              <h3>Read the valuation in context.</h3>
              <div className="insight-row"><span>01</span><div><strong>Recorded value</strong><p>A historical value in the dataset, not necessarily today's price.</p></div></div>
              <div className="insight-row"><span>02</span><div><strong>FootballIQ estimate</strong><p>An experimental model output based on available player data.</p></div></div>
              <div className="insight-row"><span>03</span><div><strong>Forecast</strong><p>A forward-looking estimate; uncertainty can be substantial.</p></div></div>
            </article>
          </section>

          <p className="valuation-disclaimer">
            FootballIQ uses historical data currently ending in June 2026.
            These results are experimental estimates and may not reflect
            a player's market value today or their actual transfer fee.
          </p>

          <details className="valuation-technical-details">
            <summary>View model response details</summary>
            <h3>Current-value model</h3>
            <pre>{JSON.stringify(currentValue, null, 2)}</pre>
            <h3>Forecast model</h3>
            <pre>{JSON.stringify(forecast, null, 2)}</pre>
            <h3>Direction model</h3>
            <pre>{JSON.stringify(direction, null, 2)}</pre>
          </details>
        </section>
      )}
    </div>
    </PageShell>
  );
}


function LiveMatchCentre() {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  async function loadLiveMatches() {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/live/predictions`);
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Unable to retrieve live matches."
        );
      }

      setMatches(data.matches ?? []);
      setLastUpdated(new Date());
      setLoaded(true);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to FootballIQ. Check that the backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadLiveMatches();
  }, []);

  return (
    <PageShell>
      <section className="live-centre-hero">
        <Link className="back-button" to="/">
          ← Dashboard
        </Link>

        <div className="eyebrow">
          <span className="eyebrow-line" />
          FOOTBALLIQ · LIVE DATA
        </div>

        <h1>
          Feel the game.
          <br />
          <span>Follow it live.</span>
        </h1>

        <p>
          Follow live scores, match minutes and football competitions
          using real-time fixture data from API-Football.
        </p>

        <div className="live-centre-toolbar">
          <div className="live-centre-status">
            <span className="status-dot" />
            {loading
              ? "Updating scores..."
              : loaded
                ? `${matches.length} live ${matches.length === 1 ? "match" : "matches"}`
                : "Ready to load"}
          </div>

          <button
            type="button"
            className="show-more"
            onClick={loadLiveMatches}
            disabled={loading}
          >
            {loading ? "Loading..." : "↻ Refresh scores"}
          </button>
        </div>

        {lastUpdated && (
          <p className="live-last-updated">
            Last updated: {lastUpdated.toLocaleTimeString("en-IN")}
          </p>
        )}
      </section>

      <section className="live-matches-section">
        <div className="section-heading">
          <div>
            <span className="section-kicker">MATCH TRACKER</span>
            <h2>Live matches<span>.</span></h2>
            <p>Scores and match status from the football data provider.</p>
          </div>
        </div>

        {error && (
          <div className="message-panel error" role="alert">
            <h3>Couldn't load live scores</h3>
            <p>{error}</p>
            <button
              type="button"
              className="show-more"
              onClick={loadLiveMatches}
              disabled={loading}
            >
              Try again
            </button>
          </div>
        )}

        {loading && !loaded && (
          <div className="message-panel" role="status">
            Fetching live matches...
          </div>
        )}

        {!loading && loaded && matches.length === 0 && !error && (
          <div className="message-panel">
            <h3>No matches currently live</h3>
            <p>
              There are no live fixtures in the provider's current response.
              Try refreshing later.
            </p>
          </div>
        )}

        {matches.length > 0 && (
          <div className="live-match-grid">
            {matches.map((match) => (
              <article
                className="live-match-card"
                key={match.fixture_id}
              >
                <div className="live-match-card-top">
                  <div>
                    <span className="live-match-league">
                      {match.league}
                    </span>
                    <span className="live-match-country">
                      {match.country}
                    </span>
                  </div>

                  <span className="live-match-badge">
                    <span className="status-dot" />
                    LIVE
                  </span>
                </div>

                <div className="live-match-status">
                  {match.minute != null
                    ? `${match.minute}'`
                    : match.status || "In progress"}
                  {" · "}
                  {match.status || "Live"}
                </div>

                <div className="live-match-teams">
                  <div className="live-match-team">
                    {match.home_logo ? (
                      <img
                        src={match.home_logo}
                        alt=""
                        loading="lazy"
                      />
                    ) : (
                      <span className="live-team-placeholder">
                        {formatTeamName(match.home_team).charAt(0)}
                      </span>
                    )}

                    <span>{formatTeamName(match.home_team)}</span>
                  </div>

                  <strong className="live-match-score">
                    {match.home_goals ?? 0}
                    <span> – </span>
                    {match.away_goals ?? 0}
                  </strong>

                  <div className="live-match-team">
                    {match.away_logo ? (
                      <img
                        src={match.away_logo}
                        alt=""
                        loading="lazy"
                      />
                    ) : (
                      <span className="live-team-placeholder">
                        {formatTeamName(match.away_team).charAt(0)}
                      </span>
                    )}

                    <span>{formatTeamName(match.away_team)}</span>
                  </div>
                </div>

                <div className="live-match-card-footer">
                  <span>Fixture #{match.fixture_id}</span>
                  <span>{match.status || "In progress"}</span>
                </div>
              </article>
            ))}
          </div>
        )}

        <p className="live-data-disclaimer">
          Live data is provided by API-Football. Scores and match events may
          be delayed. Refresh to request the latest available data.
          Live match display does not yet include FootballIQ in-play
          prediction probabilities.
        </p>
      </section>
    </PageShell>
  );
}



function NotFoundPage() {
  return (
    <PageShell>
      <section className="detail-shell">
        <h1>Page not found<span>.</span></h1>
        <p>Let's get you back to the FootballIQ dashboard.</p>
        <Link className="back-button" to="/">
          ← Back to FootballIQ
        </Link>
      </section>
    </PageShell>
  );
}


export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route
        path="/competition/:competitionCode"
        element={<CompetitionPage />}
      />
      <Route
        path="/match/:competitionCode/:matchId"
        element={<MatchDetailsPage />}
      />
      <Route
        path="/market-value"
        element={<MarketValuePage />}
      />
      <Route path="/live" element={<LiveMatchCentre />} />
      <Route path="*" element={<NotFoundPage />} />
      
    </Routes>
  );
}

