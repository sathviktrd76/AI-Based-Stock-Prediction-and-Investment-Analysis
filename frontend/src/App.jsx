import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

const PERIODS = [
  { label: "1 Month", value: "1mo" },
  { label: "3 Months", value: "3mo" },
  { label: "6 Months", value: "6mo" },
  { label: "1 Year", value: "1y" },
];

function formatNumber(value, digits = 2) {
  if (
    value === null ||
    value === undefined ||
    value === "" ||
    !Number.isFinite(Number(value))
  ) {
    return "—";
  }

  return Number(value).toLocaleString("en-IN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatPrice(value) {
  return value == null || !Number.isFinite(Number(value))
    ? "—"
    : `₹${formatNumber(value)}`;
}

function formatDate(value) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return String(value);

  return date.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "2-digit",
  });
}

function displayMetric(value) {
  if (value === null || value === undefined) return "—";

  const numericValue = Number(value);

  return Number.isFinite(numericValue)
    ? numericValue.toFixed(2)
    : String(value);
}

function getPredictionTone(value) {
  const prediction = String(value || "").toUpperCase();

  if (prediction.includes("UP")) return "up";
  if (prediction.includes("DOWN")) return "down";

  return "neutral";
}

function getFeatureImportance(data) {
  if (!data) return [];

  const entries = Array.isArray(data)
    ? data.map((item) => [
        item.feature ?? item.name ?? "Feature",
        Number(item.importance ?? item.value ?? 0),
      ])
    : Object.entries(data).map(([name, value]) => [
        name,
        Number(value),
      ]);

  return entries
    .filter(([, value]) => Number.isFinite(value))
    .sort((a, b) => b[1] - a[1])
    .map(([feature, importance]) => ({
      feature: String(feature).replaceAll("_", " "),
      importance,
      percentage: importance * 100,
    }));
}

function MetricCard({
  label,
  value,
  footnote,
  icon,
  tone = "",
}) {
  return (
    <article className="metric-card">
      <div className="metric-topline">
        <p className="metric-label">{label}</p>
        <span className="metric-icon" aria-hidden="true">
          {icon}
        </span>
      </div>

      <h2 className={tone}>{value}</h2>
      <p className="metric-footnote">{footnote}</p>
    </article>
  );
}

function App() {
  const [symbol, setSymbol] = useState("RELIANCE.NS");
  const [search, setSearch] = useState("RELIANCE.NS");
  const [period, setPeriod] = useState("1mo");

  const [stock, setStock] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [indicators, setIndicators] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadStock(
    selectedSymbol = symbol,
    selectedPeriod = period
  ) {
    const normalizedSymbol = selectedSymbol.trim().toUpperCase();

    if (!normalizedSymbol) {
      setError("Please enter a stock symbol.");
      return;
    }

    setLoading(true);
    setError("");
    setStock(null);
    setPrediction(null);
    setIndicators(null);

    try {
      const stockResponse = await fetch(
        `${API_URL}/api/stock/${encodeURIComponent(
          normalizedSymbol
        )}?period=${encodeURIComponent(selectedPeriod)}`
      );

      const stockResult = await stockResponse.json();

      if (!stockResponse.ok) {
        throw new Error(
          stockResult.error || "Unable to retrieve stock data."
        );
      }

      if (!Array.isArray(stockResult.data) || stockResult.data.length === 0) {
        throw new Error("No historical stock data was returned.");
      }

      setStock(stockResult);
      setSymbol(normalizedSymbol);
      setSearch(normalizedSymbol);
      setPeriod(selectedPeriod);

      const [predictionResult, indicatorResult] = await Promise.all([
        fetch(
          `${API_URL}/api/predict/${encodeURIComponent(normalizedSymbol)}`
        ).then(async (response) => {
          const result = await response.json();

          if (!response.ok) {
            throw new Error(
              result.error || "Unable to generate the prediction."
            );
          }

          return result;
        }),

        fetch(
          `${API_URL}/api/indicators/${encodeURIComponent(normalizedSymbol)}`
        ).then(async (response) => {
          const result = await response.json();

          if (!response.ok) {
            throw new Error(
              result.error || "Unable to calculate technical indicators."
            );
          }

          return result;
        }),
      ]);

      setPrediction(predictionResult);
      setIndicators(indicatorResult);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Unable to connect to the backend. Check that your Flask server is running."
          : err.message || "Something went wrong while loading the stock."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadStock("RELIANCE.NS", "1mo");
  }, []);

  const prices = stock?.data || [];
  const latest = prices.length ? prices[prices.length - 1] : null;
  const previous = prices.length > 1 ? prices[prices.length - 2] : null;

  const latestPrice = latest?.close;
  const previousPrice = previous?.close;

  const priceChange =
    latestPrice != null && previousPrice != null
      ? Number(latestPrice) - Number(previousPrice)
      : null;

  const percentageChange =
    priceChange != null && Number(previousPrice) !== 0
      ? (priceChange / Number(previousPrice)) * 100
      : null;

  const predictionValue = prediction?.prediction || "—";
  const predictionTone = getPredictionTone(predictionValue);

  const confidence =
    prediction?.confidence != null &&
    Number.isFinite(Number(prediction.confidence))
      ? Number(prediction.confidence) * 100
      : null;

  const evaluation = prediction?.evaluation || {};
  const featureImportance = getFeatureImportance(
    prediction?.feature_importance
  );

  const maxImportance = Math.max(
    ...featureImportance.map((item) => item.importance),
    0
  );

  function getPredictionExplanation() {
    if (!indicators) {
      return "Technical context will appear after the indicators are calculated.";
    }

    const signals = [];
    const close = Number(indicators.close);

    const rsi = Number(indicators.rsi_14);
    const sma20 = Number(indicators.sma_20);
    const ema20 = Number(indicators.ema_20);
    const sma50 = Number(indicators.sma_50);

    if (Number.isFinite(rsi)) {
      if (rsi < 30) {
        signals.push("RSI indicates potentially oversold conditions");
      } else if (rsi > 70) {
        signals.push("RSI indicates potentially overbought conditions");
      } else {
        signals.push("RSI is within the conventional neutral range");
      }
    }

    if (Number.isFinite(close) && Number.isFinite(sma20)) {
      signals.push(
        close >= sma20
          ? "price is above its 20-day average"
          : "price is below its 20-day average"
      );
    }

    if (Number.isFinite(close) && Number.isFinite(ema20)) {
      signals.push(
        close >= ema20
          ? "price is above its 20-day exponential average"
          : "price is below its 20-day exponential average"
      );
    }

    if (Number.isFinite(sma20) && Number.isFinite(sma50)) {
      signals.push(
        sma20 >= sma50
          ? "the 20-day average is above the 50-day average"
          : "the 20-day average is below the 50-day average"
      );
    }

    if (!signals.length) {
      return "Insufficient technical indicator data to describe the current market context.";
    }

    return `${signals.join("; ")}. These are technical observations, not a direct explanation of the Random Forest model's internal decision.`;
  }

  function handleSearch(event) {
    event.preventDefault();
    loadStock(search, period);
  }

  function handlePeriodChange(event) {
    const nextPeriod = event.target.value;
    setPeriod(nextPeriod);
    loadStock(symbol, nextPeriod);
  }

  function scrollToSection(sectionId) {
    document.getElementById(sectionId)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  const evaluationRows = Object.entries(evaluation);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button
          type="button"
          className="brand"
          onClick={() => scrollToSection("overview")}
          aria-label="StockBeacon home"
        >
          <span className="brand-icon" aria-hidden="true">
            <svg viewBox="0 0 48 48" fill="none">
              <path
                d="M24 5L39 35H9L24 5Z"
                fill="currentColor"
                opacity=".19"
              />
              <path
                d="M24 9L36 34H12L24 9Z"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinejoin="round"
              />
              <circle cx="24" cy="25" r="5" fill="currentColor" />
              <path
                d="M24 30V39M17 39H31"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinecap="round"
              />
            </svg>
          </span>

          <span className="brand-copy">
            <strong>StockBeacon</strong>
            <small>AI MARKET INTELLIGENCE</small>
          </span>
        </button>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation" aria-label="Main navigation">
          <button
            type="button"
            className="nav-item active"
            onClick={() => scrollToSection("overview")}
          >
            <span className="nav-icon">⌂</span>
            Overview
          </button>

          <button
            type="button"
            className="nav-item"
            onClick={() => scrollToSection("market-analysis")}
          >
            <span className="nav-icon">⌁</span>
            Market analysis
          </button>

          <button
            type="button"
            className="nav-item"
            onClick={() => scrollToSection("ai-prediction")}
          >
            <span className="nav-icon">✳</span>
            AI prediction
          </button>

          <button
            type="button"
            className="nav-item"
            onClick={() => scrollToSection("technical-signals")}
          >
            <span className="nav-icon">◫</span>
            Technical signals
          </button>

          <button
            type="button"
            className="nav-item"
            onClick={() => scrollToSection("model-evaluation")}
          >
            <span className="nav-icon">▥</span>
            Model evaluation
          </button>
        </nav>

        <div className="sidebar-note">
          <div className="note-symbol" aria-hidden="true">
            ✦
          </div>
          <p className="note-kicker">THE STOCKBEACON IDEA</p>
          <h3>See beyond the price.</h3>
          <p>
            Explore market signals, understand model outputs, and make
            sense of historical data.
          </p>
        </div>

        <div className="sidebar-footer">
          <span className="status-dot" />
          <span>Market intelligence workspace</span>
          <p>Academic project · For learning purposes</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">YOUR MARKET, IN PERSPECTIVE</p>
            <h1>Market overview</h1>
          </div>

          <div className="topbar-right">
            <span className="topbar-badge">
              <span className="status-dot" />
              AI-powered analysis
            </span>

            <div className="profile-mark" aria-label="StockBeacon">
              SB
            </div>
          </div>
        </header>

        <section className="welcome-row" id="overview">
          <div className="welcome-copy">
            <p className="section-kicker">YOUR DAILY MARKET BRIEF</p>
            <h2>
              Find the signal in <span>every move.</span>
            </h2>
            <p>
              Historical prices, technical indicators, and machine-learning
              insights in one place.
            </p>
          </div>

          <form className="search-form" onSubmit={handleSearch}>
            <label className="search-label" htmlFor="stock-search">
              STOCK SYMBOL
            </label>

            <div className="search-controls">
              <span className="search-icon" aria-hidden="true">
                ⌕
              </span>

              <input
                id="stock-search"
                type="text"
                placeholder="e.g. RELIANCE.NS"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                aria-label="Stock symbol"
              />

              <button type="submit" disabled={loading}>
                {loading ? "Loading…" : "Analyze"}
                {!loading && <span aria-hidden="true"> ↗</span>}
              </button>
            </div>
          </form>
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <span aria-hidden="true">!</span>
            <div>
              <strong>Unable to complete analysis</strong>
              <p>{error}</p>
            </div>
            <button
              type="button"
              onClick={() => loadStock(symbol, period)}
            >
              Retry
            </button>
          </div>
        )}

        <section className="metric-grid" aria-label="Market summary">
          <MetricCard
            label="LATEST PRICE"
            value={formatPrice(latestPrice)}
            footnote={latest ? `As of ${formatDate(latest.date)}` : "Awaiting market data"}
            icon="₹"
          />

          <MetricCard
            label="DAILY PRICE CHANGE"
            value={
              priceChange == null
                ? "—"
                : `${priceChange >= 0 ? "+" : "−"}₹${formatNumber(
                    Math.abs(priceChange)
                  )}`
            }
            footnote={
              percentageChange == null
                ? "Compared with previous available session"
                : `${percentageChange >= 0 ? "+" : ""}${formatNumber(
                    percentageChange
                  )}% vs previous available session`
            }
            icon={priceChange == null ? "↔" : priceChange >= 0 ? "↗" : "↘"}
            tone={
              priceChange == null
                ? ""
                : priceChange >= 0
                  ? "positive"
                  : "negative"
            }
          />

          <MetricCard
            label="AI TREND SIGNAL"
            value={predictionValue}
            footnote="Random Forest · 5-day classification target"
            icon="✳"
            tone={`prediction-value ${predictionTone}`}
          />

          <MetricCard
            label="MODEL CONFIDENCE"
            value={confidence == null ? "—" : `${formatNumber(confidence, 1)}%`}
            footnote="Estimated probability for the predicted class"
            icon="◎"
            tone="confidence-value"
          />
        </section>

        <section className="content-grid" id="market-analysis">
          <article className="panel chart-panel">
            <div className="panel-header">
              <div>
                <p className="section-kicker">PRICE EXPLORER</p>
                <h3>Historical performance</h3>
                <p className="panel-subtitle">
                  {symbol} · Closing price over time
                </p>
              </div>

              <label className="period-control">
                <span>PERIOD</span>
                <select
                  value={period}
                  onChange={handlePeriodChange}
                  disabled={loading}
                  aria-label="Historical chart period"
                >
                  {PERIODS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="chart-summary">
              <div>
                <span className="chart-summary-label">LATEST CLOSE</span>
                <strong>{formatPrice(latestPrice)}</strong>
              </div>

              <div className="chart-period-tag">
                <span className="chart-line-key" />
                Closing price
              </div>
            </div>

            <div className="chart-container">
  {loading ? (
    <div className="chart-empty">
      <span className="loading-spinner" />
      Loading historical prices…
    </div>
  ) : prices.length > 0 ? (
    <div className="stock-chart">
      <ResponsiveContainer width="100%" height="100%" minHeight={280}>
        <AreaChart
          data={prices}
          margin={{ top: 12, right: 12, left: 8, bottom: 4 }}
        >
          <defs>
            <linearGradient
              id="stockBeaconGradient"
              x1="0"
              y1="0"
              x2="0"
              y2="1"
            >
              <stop
                offset="0%"
                stopColor="#159b72"
                stopOpacity={0.24}
              />
              <stop
                offset="100%"
                stopColor="#159b72"
                stopOpacity={0.02}
              />
            </linearGradient>
          </defs>

          <CartesianGrid
            stroke="#e9eee9"
            strokeDasharray="3 5"
            vertical={false}
          />

          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            tick={{ fill: "#89958d", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            minTickGap={32}
          />

          <YAxis
            domain={["auto", "auto"]}
            tickFormatter={(value) =>
              `₹${Number(value).toLocaleString("en-IN")}`
            }
            tick={{ fill: "#89958d", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            width={75}
          />

          <Tooltip
            labelFormatter={formatDate}
            formatter={(value) => [
              formatPrice(value),
              "Closing price",
            ]}
            contentStyle={{
              border: "1px solid #e4ebe5",
              borderRadius: "12px",
              background: "#ffffff",
              fontSize: "12px",
            }}
          />

          <Area
            type="monotone"
            dataKey="close"
            stroke="#159b72"
            strokeWidth={2.5}
            fill="url(#stockBeaconGradient)"
            dot={false}
            activeDot={{
              r: 5,
              stroke: "#ffffff",
              strokeWidth: 2,
              fill: "#159b72",
            }}
            connectNulls
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  ) : (
    <div className="chart-empty">
      No historical price data is available for this selection.
    </div>
  )}
</div>

            <div className="chart-footer">
              <span>
                {prices.length} historical observations
              </span>
              <span>
                Source: Yahoo Finance via yfinance
              </span>
            </div>
          </article>

          <article className="panel prediction-panel" id="ai-prediction">
            <div className="panel-header">
              <div>
                <p className="section-kicker">MACHINE LEARNING</p>
                <h3>AI trend outlook</h3>
                <p className="panel-subtitle">
                  A short-term classification signal
                </p>
              </div>

              <span className="model-badge">RF MODEL</span>
            </div>

            <div className={`prediction-box ${predictionTone}`}>
              <div className="prediction-orbit" aria-hidden="true">
                <span>
                  {predictionTone === "up"
                    ? "↗"
                    : predictionTone === "down"
                      ? "↘"
                      : "→"}
                </span>
              </div>

              <p className="prediction-caption">MODEL OUTPUT</p>

              <h2>{loading ? "Analyzing…" : predictionValue}</h2>

              <p className="prediction-description">
                {predictionTone === "up"
                  ? "The model classifies the expected short-term trend as upward."
                  : predictionTone === "down"
                    ? "The model classifies the expected short-term trend as downward."
                    : "The model classifies the expected short-term trend as neutral or unavailable."}
              </p>

              {confidence != null && !loading && (
                <div className="confidence-track">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${Math.max(0, Math.min(100, confidence))}%`,
                    }}
                  />
                </div>
              )}

              <div className="confidence-caption">
                <span>Predicted-class confidence</span>
                <strong>
                  {confidence == null ? "—" : `${formatNumber(confidence, 1)}%`}
                </strong>
              </div>
            </div>

            <div className="prediction-detail">
              <span>Selected model</span>
              <strong>{prediction?.model || "Random Forest"}</strong>
            </div>

            <div className="prediction-detail">
              <span>Prediction target</span>
              <strong>5 trading days</strong>
            </div>

            <div className="prediction-detail">
              <span>Training observations</span>
              <strong>
                {prediction?.training_samples ?? "—"}
              </strong>
            </div>

            <div className="prediction-detail">
              <span>Testing observations</span>
              <strong>
                {prediction?.testing_samples ?? "—"}
              </strong>
            </div>

            <div className="disclaimer">
              <strong>Research, not financial advice.</strong>
              <p>
                Predictions are experimental classifications based on
                historical data. Model confidence does not guarantee accuracy
                or future returns.
              </p>
            </div>
          </article>
        </section>

        <section className="panel indicators-panel" id="technical-signals">
          <div className="panel-header">
            <div>
              <p className="section-kicker">MARKET MOMENTUM</p>
              <h3>Technical signals</h3>
              <p className="panel-subtitle">
                Common indicators for understanding historical price behaviour
              </p>
            </div>

            <span className="section-count">04 INDICATORS</span>
          </div>

          <div className="indicator-grid">
            <article className="indicator-card">
              <div className="indicator-card-top">
                <span className="indicator-symbol">S</span>
                <span className="indicator-period">20 DAY</span>
              </div>
              <p>Simple moving average</p>
              <h3>{formatPrice(indicators?.sma_20)}</h3>
              <span>Shorter-term price trend</span>
            </article>

            <article className="indicator-card">
              <div className="indicator-card-top">
                <span className="indicator-symbol blue">S</span>
                <span className="indicator-period">50 DAY</span>
              </div>
              <p>Simple moving average</p>
              <h3>{formatPrice(indicators?.sma_50)}</h3>
              <span>Broader price trend</span>
            </article>

            <article className="indicator-card">
              <div className="indicator-card-top">
                <span className="indicator-symbol gold">E</span>
                <span className="indicator-period">20 DAY</span>
              </div>
              <p>Exponential moving average</p>
              <h3>{formatPrice(indicators?.ema_20)}</h3>
              <span>Weights recent prices more heavily</span>
            </article>

            <article className="indicator-card rsi-card">
              <div className="indicator-card-top">
                <span className="indicator-symbol purple">R</span>
                <span className="indicator-period">14 DAY</span>
              </div>
              <p>Relative strength index</p>
              <h3>{formatNumber(indicators?.rsi_14)}</h3>

              <div className="rsi-track" aria-label="RSI scale">
                <span className="rsi-zone oversold" />
                <span className="rsi-zone neutral-zone" />
                <span className="rsi-zone overbought" />

                {Number.isFinite(Number(indicators?.rsi14)) &&
                  indicators?.rsi14 != null && (
                    <span
                      className="rsi-marker"
                      style={{
                        left: `${Math.max(
                          0,
                          Math.min(100, Number(indicators.rsi14))
                        )}%`,
                      }}
                    />
                  )}
              </div>

              <div className="rsi-labels">
                <span>Oversold</span>
                <span>Neutral</span>
                <span>Overbought</span>
              </div>
            </article>
          </div>

          <p className="indicator-note">
            Indicators describe historical market conditions. They are not
            standalone buy or sell recommendations.
          </p>
        </section>

        <section className="panel explanation-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">INTERPRETING THE DATA</p>
              <h3>What the technical signals suggest</h3>
              <p className="panel-subtitle">
                A human-readable summary of selected indicator values
              </p>
            </div>

            <span className="explanation-icon" aria-hidden="true">
              ✳
            </span>
          </div>

          <div className="explanation-summary">
            <strong>Technical context</strong>
            <p>{getPredictionExplanation()}</p>
          </div>

          <div className="signal-list">
            <article className="signal-item">
              <span className="signal-marker">↗</span>
              <div>
                <h4>Moving averages</h4>
                <p>
                  SMA and EMA smooth historical prices to help examine
                  underlying trends. Crossovers and price positions can offer
                  context, but do not guarantee future direction.
                </p>
              </div>
            </article>

            <article className="signal-item">
              <span className="signal-marker purple">◉</span>
              <div>
                <h4>Relative strength index</h4>
                <p>
                  RSI is commonly interpreted using 30 and 70 as reference
                  levels for potentially oversold and overbought conditions.
                  Readings can remain elevated or depressed during strong trends.
                </p>
              </div>
            </article>
          </div>
        </section>

        <section className="panel feature-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">MODEL INTERPRETABILITY</p>
              <h3>What influences the model?</h3>
              <p className="panel-subtitle">
                Random Forest feature importance for the selected training data
              </p>
            </div>

            <span className="section-count">MODEL INSIGHTS</span>
          </div>

          {featureImportance.length ? (
            <div className="importance-list">
              {featureImportance.map((item) => (
                <div className="importance-row" key={item.feature}>
                  <div className="importance-heading">
                    <span>{item.feature}</span>
                    <strong>{formatNumber(item.percentage, 1)}%</strong>
                  </div>

                  <div className="importance-track">
                    <div
                      className="importance-fill"
                      style={{
                        width: `${
                          maxImportance > 0
                            ? (item.importance / maxImportance) * 100
                            : 0
                        }%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="chart-empty importance-empty">
              Feature importance will appear after the model returns its
              analysis results.
            </div>
          )}

          <p className="indicator-note">
            Feature importance indicates how much a feature contributes to
            the fitted Random Forest's decisions. It does not indicate whether
            a feature pushes a prediction up or down, and it does not establish
            causation.
          </p>
        </section>

        <section className="panel evaluation-panel" id="model-evaluation">
          <div className="panel-header">
            <div>
              <p className="section-kicker">MODEL PERFORMANCE</p>
              <h3>Evaluation & comparison</h3>
              <p className="panel-subtitle">
                Metrics calculated on the held-out test observations
              </p>
            </div>

            <span className="evaluation-badge">TEST SET</span>
          </div>

          {evaluationRows.length ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>MODEL</th>
                    <th>ACCURACY</th>
                    <th>PRECISION</th>
                    <th>RECALL</th>
                    <th>F1 SCORE</th>
                  </tr>
                </thead>

                <tbody>
                  {evaluationRows.map(([modelName, metrics]) => (
                    <tr key={modelName}>
                      <td className="model-name">{modelName}</td>
                      <td>{formatNumber(Number(metrics.accuracy) * 100, 1)}%</td>
                      <td>
                        {formatNumber(
                          Number(metrics.precision_macro) * 100,
                          1
                        )}%
                      </td>
                      <td>
                        {formatNumber(Number(metrics.recall_macro) * 100, 1)}%
                      </td>
                      <td>
                        {formatNumber(Number(metrics.f1_macro) * 100, 1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="empty-evaluation">
              Model evaluation metrics will appear when training and testing
              are complete.
            </p>
          )}

          <div className="evaluation-footnote">
            <span className="evaluation-info">i</span>
            <p>
              Historical test results are not a guarantee of future performance.
              Accuracy alone can be misleading when comparing classification
              models, so precision, recall, and F1 score are also reported.
            </p>
          </div>
        </section>

        <footer className="page-footer">
          <div className="footer-brand">
            <span className="footer-symbol">✳</span>
            <strong>StockBeacon</strong>
            <span>Illuminating the Signals Behind Stocks.</span>
          </div>

          <p>
            Built for academic exploration · Not investment advice
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;