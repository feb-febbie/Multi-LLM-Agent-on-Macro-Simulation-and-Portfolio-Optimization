# Multi-LLM Agent for Macro Simulation and Portfolio Optimization

An end-to-end, automated financial pipeline that combines Large Language Models (LLMs) with advanced statistical modeling and convex optimization to simulate macroeconomic shocks, analyze portfolio risk, and generate personalized, dynamically optimized investment strategies.

## 🚀 Overview

This project bridges qualitative macroeconomic theory with quantitative financial engineering. It features a three-stage architecture:

1. **Macro Scenario Generator:** A multi-agent LLM workflow (built with LangGraph) that synthesizes custom macroeconomic events (e.g., a Fed rate hike), extracts numerical factor shifts, and iteratively refines its own logical consistency.
2. **Portfolio Analyzer & Risk Simulator:** A quantitative engine that maps macro shocks to financial factors using Principal Component Analysis (PCA) and Ridge Regression, then runs Monte Carlo simulations to assess portfolio VaR and CVaR.
3. **Strategy Designer & Explainer:** An optimization engine that profiles user risk appetite and leverages convex optimization to allocate a maximum of 30 assets, outputting a clear, automated storytelling report.

## 🛠️ Tech Stack

* **AI & Orchestration:** LangGraph, LangChain, OpenAI API (GPT-4 / OSS variants), Groq API (Llama-3)
* **Quantitative & Statistical Modeling:** `scikit-learn` (PCA, Ridge Regression), `numpy`, `pandas`
* **Convex Optimization:** `cvxpy`, `PyPortfolioOpt` (solvers: ECOS, SCS, CLARABEL)
* **Financial Data:** `yfinance`, S&P 500 daily data, FRED quarterly macroeconomic data

## 🧠 System Architecture

### Stage 1: Macro Scenario Generator

* **Scenario Simulation:** Generates highly detailed "what-if" macroeconomic scenarios based on user prompts.
* **Multi-Agent Extraction (LangGraph):** * **Extractor Agent:** Parses the narrative into a structured JSON of specific macro factor changes (e.g., CPI, GDP, FEDFUNDS).
* **Critic Agent:** Reviews the extraction for logical inconsistencies (e.g., ensuring inflation and interest rate dynamics align with economic theory).
* **Revision & Evaluation Agents:** Iteratively refines the data until it meets a high threshold for completeness and economic logic.



### Stage 2: Factor Modeling & Risk Simulator

* **Factor Exposure:** Uses PCA to identify underlying common drivers in the cross-section of S&P 500 returns. Portfolio exposures are fitted using Ridge Regression to handle multicollinearity.
* **Shock Mapping:** Translates the JSON macro shocks (from Stage 1) into directional shifts in the PCA and market factors.
* **Monte Carlo Simulation:** Runs 20,000 paths to simulate portfolio P&L under the stressed macroeconomic conditions, accurately calculating **95% Value at Risk (VaR)** and **Conditional VaR (CVaR)**.

### Stage 3: Strategy Designer & Explainer

* **Risk Profiling:** An LLM agent converses with the user to infer their risk appetite, assigning a numeric risk aversion penalty ($\lambda$).
* **Convex Optimization:** Formulates a portfolio optimization problem maximizing risk-adjusted returns subject to constraints ($w \ge 0$, $\sum w = 1$, $w \le 0.25$, max 30 assets). It actively penalizes volatility and applies an L2 regularization term to ensure diversification.
* **Automated Reporting:** Generates a "Portfolio Storytelling Report" comparing the user's current holdings against the optimized target weights, detailing expected returns, volatility, and recommended actions.

## 📂 Project Structure & Outputs

The pipeline automatically manages data persistence and logging, saving key artifacts to your designated output directory (`Stage2_Outputs/`):

* `final_state.json`: The fully evaluated macro-shock scenario.
* `exposures_table.csv` & `risk_contrib.csv`: Calculated factor betas and variance contributions per asset.
* `pnl_paths.csv` & `pnl_hist.png`: Monte Carlo simulation raw paths and visual distribution.
* `summary.json`: Top-level risk metrics (Factor/Idiosyncratic Variance, VaR, CVaR).

## ⚡ Quick Start

1. **Install Dependencies:**
```bash
pip install langgraph langchain langchain-openai openai yfinance PyPortfolioOpt cvxpy ecos scs osqp clarabel scikit-learn

```


2. **Environment Variables:**
Set up your API keys for the LLM routing:
```python
import os
os.environ["OPENAI_API_KEY"] = "your_openai_or_openrouter_api_key"
os.environ["GROQ_API_KEY"] = "your_groq_api_key"

```


3. **Run the Pipeline:**
Execute the notebook sequentially. Start by prompting the Macro Simulator with an economic event, let the Risk Simulator process the matrix transformations, and answer the Profiler's prompt to receive your optimized asset allocation report.

---
