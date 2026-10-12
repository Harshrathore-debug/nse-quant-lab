#!/usr/bin/env bash
# Refresh data/raw from the open-source NSE end-of-day mirror (updated daily).
# Usage: bash scripts/fetch_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
FILES=(
  "nifty 50" "nifty 100" "nifty 200" "nifty 1d rate index" "nifty 10 yr benchmark g-sec"
  "nifty bank" "nifty it" "nifty fmcg" "nifty pharma" "nifty auto" "nifty metal" "nifty energy"
  "nifty realty" "nifty infrastructure" "nifty media" "nifty psu bank" "nifty financial services"
  "nifty50 equal weight" "nifty100 low volatility 30" "nifty200 momentum 30" "nifty200 quality 30"
  "nifty alpha 50" "nifty50 value 20" goldbees
  adanient adaniports apollohosp asianpaint axisbank bajaj-auto bajfinance bajajfinsv bel bhartiartl
  cipla coalindia drreddy eichermot grasim hcltech hdfcbank hdfclife hindalco hindunilvr icicibank
  indigo infy itc jswsteel kotakbank lt "m&m" maruti nestleind ntpc ongc powergrid reliance sbilife
  sbin shriramfin sunpharma tataconsum tatasteel tcs techm titan trent ultracemco wipro
  indusindbk bankbaroda pnb federalbnk idfcfirstb aubank bandhanbnk canbk
)
git clone --depth 1 --filter=blob:none --sparse https://github.com/BennyThadikaran/eod2_data.git "$TMP/eod2"
PATTERNS=(); for f in "${FILES[@]}"; do PATTERNS+=("/daily/$f.csv"); done
git -C "$TMP/eod2" sparse-checkout set --no-cone "${PATTERNS[@]}"
mkdir -p data/raw
for f in "${FILES[@]}"; do cp "$TMP/eod2/daily/$f.csv" "data/raw/$f.csv"; done
rm -rf "$TMP"
echo "Refreshed ${#FILES[@]} files into data/raw"
