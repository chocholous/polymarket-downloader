#!/usr/bin/env python3
"""
Polymarket Historical Data Downloader

Downloads complete historical price data from the last 30 days for all markets.
Respects API rate limits and saves data to JSON and CSV formats.

Note: The CLOB API has a ~14 day max interval limit for price history,
so we chunk requests to get the full 30 days.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration
CLOB_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

# Rate limiting: CLOB market data endpoints allow 80-200 requests/10s
# Being conservative with 4 requests/second
RATE_LIMIT_DELAY = 0.25  # seconds between requests
BATCH_DELAY = 2.0  # seconds between batches of requests

# Maximum interval for price history (API limit is ~14 days)
MAX_INTERVAL_DAYS = 14


class RateLimiter:
    """Simple rate limiter to respect API limits."""

    def __init__(self, requests_per_second: float = 4.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.request_count = 0
        self.batch_size = 40

    def wait(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self.last_request_time = time.time()
        self.request_count += 1

        # Extra pause every batch to be safe
        if self.request_count % self.batch_size == 0:
            logger.debug(f"Batch pause after {self.request_count} requests")
            time.sleep(BATCH_DELAY)


class PolymarketDownloader:
    """Download historical data from Polymarket."""

    def __init__(self, output_dir: str = "data"):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PolymarketHistoricalDownloader/1.0"
        })
        self.rate_limiter = RateLimiter()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_request(self, url: str, params: Optional[dict] = None, retries: int = 3) -> Optional[Any]:
        """Make a rate-limited request with retry logic."""
        self.rate_limiter.wait()

        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    if response.text:
                        return response.json()
                    return None
                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = (2 ** attempt) * 5
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                elif response.status_code == 400:
                    # Bad request - don't retry
                    logger.debug(f"Bad request: {response.text[:100]}")
                    return None
                elif response.status_code == 404:
                    logger.debug(f"Not found: {url}")
                    return None
                elif response.status_code == 503:
                    # Service unavailable - retry
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Service unavailable, waiting {wait_time}s before retry")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Request failed with status {response.status_code}: {url}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

        return None

    def get_all_events_gamma(self, include_closed: bool = False) -> List[Dict]:
        """Fetch all events from the GAMMA API with pagination."""
        logger.info("Fetching events from GAMMA API...")
        all_events = []
        offset = 0
        limit = 100  # Max per request

        while True:
            params = {
                "order": "volume",
                "ascending": "false",
                "limit": limit,
                "offset": offset
            }
            if not include_closed:
                params["closed"] = "false"

            data = self._make_request(f"{GAMMA_BASE_URL}/events", params)

            if not data or len(data) == 0:
                break

            all_events.extend(data)
            logger.info(f"Fetched {len(data)} events (total: {len(all_events)})")

            if len(data) < limit:
                break

            offset += limit

        logger.info(f"Total events fetched: {len(all_events)}")
        return all_events

    def get_all_markets_gamma(self, include_closed: bool = False) -> List[Dict]:
        """Fetch all markets from GAMMA API events."""
        events = self.get_all_events_gamma(include_closed=include_closed)

        markets = []
        for event in events:
            event_markets = event.get("markets", [])
            for market in event_markets:
                market["event_title"] = event.get("title", "")
                market["event_slug"] = event.get("slug", "")
                markets.append(market)

        logger.info(f"Total markets extracted: {len(markets)}")
        return markets

    def get_price_history_chunked(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity: int = 60
    ) -> List[Dict]:
        """
        Fetch price history for a token, chunking requests to handle API limits.

        The API has a ~14 day max interval, so we split longer requests into chunks.
        """
        all_history = []
        chunk_seconds = MAX_INTERVAL_DAYS * 24 * 60 * 60  # 14 days in seconds

        current_start = start_ts
        while current_start < end_ts:
            current_end = min(current_start + chunk_seconds, end_ts)

            # Build URL with params directly to avoid encoding issues
            url = (
                f"{CLOB_BASE_URL}/prices-history"
                f"?market={token_id}"
                f"&startTs={current_start}"
                f"&endTs={current_end}"
                f"&fidelity={fidelity}"
            )

            data = self._make_request(url)

            if data and "history" in data:
                chunk_history = data["history"]
                all_history.extend(chunk_history)

            current_start = current_end

        # Sort by timestamp and remove duplicates
        if all_history:
            seen = set()
            unique_history = []
            for point in sorted(all_history, key=lambda x: x.get("t", 0)):
                t = point.get("t")
                if t not in seen:
                    seen.add(t)
                    unique_history.append(point)
            return unique_history

        return all_history

    def download_historical_data(
        self,
        days: int = 30,
        fidelity: int = 60,
        include_closed: bool = False,
        active_only: bool = True,
        min_volume: float = 0,
        exclude_patterns: List[str] = None,
        max_markets: int = 0
    ):
        """
        Download historical price data for all markets.

        Args:
            days: Number of days of historical data to fetch
            fidelity: Resolution in minutes (default: 60 = hourly)
            include_closed: Whether to include closed events
            active_only: Only download for markets that are accepting orders
            min_volume: Minimum volume to include market
            exclude_patterns: List of patterns to exclude from market questions
            max_markets: Maximum number of markets to process (0 = no limit)
        """
        if exclude_patterns is None:
            exclude_patterns = []

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())

        logger.info(f"Downloading data from {start_time} to {end_time}")
        logger.info(f"Fidelity: {fidelity} minutes")
        logger.info(f"Using chunked requests (max {MAX_INTERVAL_DAYS} days per request)")

        # Get all markets from GAMMA API
        markets = self.get_all_markets_gamma(include_closed=include_closed)

        if not markets:
            logger.error("No markets found")
            return

        # Filter markets if needed
        if active_only:
            markets = [m for m in markets if m.get("active") and m.get("acceptingOrders")]
            logger.info(f"Filtered to {len(markets)} active markets accepting orders")

        # Filter by volume
        if min_volume > 0:
            markets = [m for m in markets if float(m.get("volume", 0) or 0) >= min_volume]
            logger.info(f"Filtered to {len(markets)} markets with volume >= {min_volume}")

        # Filter out excluded patterns
        if exclude_patterns:
            original_count = len(markets)
            markets = [
                m for m in markets
                if not any(pattern.lower() in m.get("question", "").lower() for pattern in exclude_patterns)
            ]
            logger.info(f"Excluded {original_count - len(markets)} markets matching patterns: {exclude_patterns}")
            logger.info(f"Remaining markets: {len(markets)}")

        # Apply max markets limit
        if max_markets > 0 and len(markets) > max_markets:
            # Sort by volume (descending) to get most important markets first
            markets = sorted(markets, key=lambda x: float(x.get("volume", 0) or 0), reverse=True)
            markets = markets[:max_markets]
            logger.info(f"Limited to top {max_markets} markets by volume")

        # Save markets metadata
        markets_file = self.output_dir / "markets.json"
        with open(markets_file, "w") as f:
            json.dump(markets, f, indent=2)
        logger.info(f"Saved markets metadata to {markets_file}")

        # Extract all unique tokens with their metadata
        # For binary markets (Yes/No), we only need one token as No = 1 - Yes
        tokens = []
        for market in markets:
            clob_token_ids = market.get("clobTokenIds", [])
            outcomes = market.get("outcomes", [])

            # Only take the first token (usually "Yes") for binary markets
            # This is sufficient since No price = 1 - Yes price
            if clob_token_ids:
                token_id = clob_token_ids[0]
                outcome = outcomes[0] if outcomes else "Yes"
                tokens.append({
                    "token_id": token_id,
                    "outcome": outcome,
                    "condition_id": market.get("conditionId"),
                    "question": market.get("question", ""),
                    "market_slug": market.get("slug", ""),
                    "event_title": market.get("event_title", ""),
                    "event_slug": market.get("event_slug", ""),
                    "active": market.get("active", False),
                    "closed": market.get("closed", False),
                    "volume": market.get("volume", 0),
                    "liquidity": market.get("liquidity", 0)
                })

        logger.info(f"Found {len(tokens)} tokens to download price history for ({len(markets)} markets)")

        # Download price history for each token
        all_history = []
        failed_tokens = []

        for token_info in tqdm(tokens, desc="Downloading price history"):
            token_id = token_info.get("token_id")
            if not token_id:
                continue

            try:
                history = self.get_price_history_chunked(
                    token_id, start_ts, end_ts, fidelity
                )

                if history:
                    record = {
                        **token_info,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "fidelity_minutes": fidelity,
                        "data_points": len(history),
                        "history": history
                    }
                    all_history.append(record)
                else:
                    failed_tokens.append({
                        "token_id": token_id,
                        "question": token_info.get("question", "")[:50]
                    })
            except Exception as e:
                logger.error(f"Error fetching history for {token_id}: {e}")
                failed_tokens.append({
                    "token_id": token_id,
                    "error": str(e)
                })

        logger.info(f"Successfully downloaded history for {len(all_history)} tokens")
        if failed_tokens:
            logger.warning(f"Failed/empty history for {len(failed_tokens)} tokens")

        # Save all historical data
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_file = self.output_dir / f"price_history_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump({
                "download_timestamp": timestamp,
                "start_date": start_time.isoformat(),
                "end_date": end_time.isoformat(),
                "days": days,
                "fidelity_minutes": fidelity,
                "total_tokens": len(all_history),
                "total_data_points": sum(r.get("data_points", 0) for r in all_history),
                "data": all_history
            }, f, indent=2)
        logger.info(f"Saved JSON data to {json_file}")

        # Save flattened data as CSV for easier analysis
        self._save_as_csv(all_history, timestamp)

        # Save failed tokens list
        if failed_tokens:
            failed_file = self.output_dir / f"failed_tokens_{timestamp}.json"
            with open(failed_file, "w") as f:
                json.dump(failed_tokens, f, indent=2)
            logger.info(f"Saved failed tokens list to {failed_file}")

        # Print summary
        total_points = sum(r.get("data_points", 0) for r in all_history)
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Markets processed: {len(markets)}")
        logger.info(f"Tokens with data: {len(all_history)}")
        logger.info(f"Total data points: {total_points:,}")
        logger.info(f"Failed/empty tokens: {len(failed_tokens)}")
        logger.info(f"Output directory: {self.output_dir}")

        return all_history

    def _save_as_csv(self, all_history: list, timestamp: str):
        """Save historical data as flattened CSV."""
        try:
            import pandas as pd

            rows = []
            for record in all_history:
                for point in record.get("history", []):
                    ts = point.get("t", 0)
                    rows.append({
                        "token_id": record.get("token_id"),
                        "condition_id": record.get("condition_id"),
                        "outcome": record.get("outcome"),
                        "question": record.get("question"),
                        "market_slug": record.get("market_slug"),
                        "event_title": record.get("event_title"),
                        "timestamp": ts,
                        "datetime": datetime.utcfromtimestamp(ts).isoformat() if ts else "",
                        "price": point.get("p"),
                        "active": record.get("active"),
                        "closed": record.get("closed"),
                        "volume": record.get("volume"),
                        "liquidity": record.get("liquidity")
                    })

            if rows:
                df = pd.DataFrame(rows)
                csv_file = self.output_dir / f"price_history_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved CSV data to {csv_file} ({len(rows):,} rows)")
        except ImportError:
            logger.warning("pandas not installed, skipping CSV export")


def main():
    parser = argparse.ArgumentParser(
        description="Download historical price data from Polymarket"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data (default: 30)"
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=60,
        help="Data resolution in minutes (default: 60 = hourly)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Include closed events/markets"
    )
    parser.add_argument(
        "--all-markets",
        action="store_true",
        help="Include all markets, not just those accepting orders"
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0,
        help="Minimum market volume to include (default: 0)"
    )
    parser.add_argument(
        "--exclude-patterns",
        type=str,
        nargs="+",
        default=["Up or Down"],
        help="Exclude markets matching these patterns (default: 'Up or Down')"
    )
    parser.add_argument(
        "--include-all-patterns",
        action="store_true",
        help="Don't exclude any patterns (includes short-interval crypto markets)"
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=0,
        help="Maximum number of markets to download (0 = no limit)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Polymarket Historical Data Downloader")
    logger.info("=" * 60)

    exclude_patterns = [] if args.include_all_patterns else args.exclude_patterns

    downloader = PolymarketDownloader(output_dir=args.output)
    downloader.download_historical_data(
        days=args.days,
        fidelity=args.fidelity,
        include_closed=args.include_closed,
        active_only=not args.all_markets,
        min_volume=args.min_volume,
        exclude_patterns=exclude_patterns,
        max_markets=args.max_markets
    )

    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
