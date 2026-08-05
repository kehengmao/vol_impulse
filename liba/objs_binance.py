from dataclasses import dataclass, field
import logging
import time
import os
import websocket
import urllib.request
import json
import threading
from collections import deque
import pandas as pd
import itertools
import math
from datetime import datetime

LOGGER = logging.getLogger(__name__)

API_BASE_URL = "https://fapi.binance.com"
WS_BASE_URL = "wss://fstream.binance.com"

ENDPOINT_EXCHANGE_INFO = f"{API_BASE_URL}/fapi/v1/exchangeInfo"
ENDPOINT_BOOK_TICKER = f"{API_BASE_URL}/fapi/v1/ticker/bookTicker"
ENDPOINT_TICKER_24HR = f"{API_BASE_URL}/fapi/v1/ticker/24hr"
ENDPOINT_AGG_TRADES = f"{API_BASE_URL}/fapi/v1/aggTrades"

TICK_PRICE = "p"
TICK_QTY = "q"
TICK_MAKER = "m"
TICK_ID = "a"
TICK_TIME = "T"

SNAP_BID_PRICE = "b"
SNAP_BID_QTY = "B"
SNAP_ASK_PRICE = "a"
SNAP_ASK_QTY = "A"


LIMIT_AGG_TRADES = 1000
MAX_FETCH_REQUESTS = 500
DEFAULT_TICK_SIZE = 0.1

INIT_SLEEP = 0.1

MS_PER_SCD = 1000.0


def _safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def get_binance_tick_size(target_symbol: str) -> float:
    """Return the Binance Futures price tick size for ``target_symbol``."""
    symbol = target_symbol.strip().upper()
    if not symbol:
        raise ValueError("target_symbol must not be empty")

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:

            http_proxy, https_proxy = get_local_proxies()
            proxies = {}
            if http_proxy: proxies['http'] = http_proxy
            if https_proxy: proxies['https'] = https_proxy

            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)


            req = urllib.request.Request(ENDPOINT_EXCHANGE_INFO)
            with urllib.request.urlopen(req, timeout=10) as response:
                info = json.loads(response.read().decode())


            tick_size = None
            for s in info.get('symbols', []):
                if s['symbol'] == symbol:
                    for f in s['filters']:
                        if f['filterType'] == 'PRICE_FILTER':
                            tick_size = float(f['tickSize'])
                            break
                    break

            if tick_size is not None:
                LOGGER.info("Binance tick size for %s: %s", symbol, tick_size)
                return tick_size

            LOGGER.error("Symbol %s was not found in Binance Futures exchange info", symbol)
            return None

        except Exception as e:
            LOGGER.warning(
                "Tick-size lookup failed for %s (attempt %d/%d): %s",
                symbol,
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                LOGGER.debug("Retrying Binance exchange-info request in %d seconds", retry_delay)
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                LOGGER.error("Unable to resolve a Binance tick size for %s", symbol)
                return None

def get_local_proxies():
    proxies = urllib.request.getproxies()
    http_proxy = proxies.get('http')
    https_proxy = proxies.get('https')
    if http_proxy and not https_proxy: https_proxy = http_proxy
    elif https_proxy and not http_proxy: http_proxy = https_proxy
    return http_proxy, https_proxy



@dataclass
class BinanceHistorySynthesizer:
    """Build fixed-interval market snapshots from Binance aggregate trades."""
    symbol: str
    total_length: int
    tick_size: float

    _current_volume: float = field(init=False, default=0.0)
    _last_agg_id: int = field(init=False, default=0)
    _max_weight_1m: int = field(init=False, default=1200)

    def __post_init__(self):
        self._max_weight_1m = self._fetch_max_minute_weight()
        LOGGER.debug("Binance request-weight limit: %d per minute", self._max_weight_1m)

    def _fetch_max_minute_weight(self) -> int:
        """Read Binance's current per-minute request-weight limit."""
        try:

            url = "https://api.binance.com/api/v3/exchangeInfo"

            http_proxy, https_proxy = get_local_proxies()
            proxies = {}
            if http_proxy: proxies['http'] = http_proxy
            if https_proxy: proxies['https'] = https_proxy


            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            req = urllib.request.Request(url)

            with opener.open(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                for limit in data.get('rateLimits', []):
                    if limit.get('rateLimitType') == 'REQUEST_WEIGHT' and limit.get('interval') == 'MINUTE':
                        return int(limit.get('limit', 6000))
        except Exception as e:
            LOGGER.warning("Could not read Binance rate limits; using 1200: %s", e)
            return 1200

        return 6000

    def get_current_volume(self) -> float:
        return self._current_volume

    def set_initial_state(self, volume: float, last_agg_id: int):
        self._current_volume = volume
        self._last_agg_id = last_agg_id

    def process_realtime_tick(self, payload: dict):
        agg_id = int(payload.get(TICK_ID, 0))
        if agg_id > self._last_agg_id:
            last_price = _safe_float(payload.get(TICK_PRICE, 0))
            qty = _safe_float(payload.get(TICK_QTY, 0))
            if last_price is None or qty is None:
                return None

            self._current_volume += qty
            self._last_agg_id = agg_id
            return last_price
        return None

    def process_realtime_snapshot(self, payload: dict) -> dict:
        bid = _safe_float(payload.get(SNAP_BID_PRICE, 0))
        bid_vol = _safe_float(payload.get(SNAP_BID_QTY, 0))
        ask = _safe_float(payload.get(SNAP_ASK_PRICE, 0))
        ask_vol = _safe_float(payload.get(SNAP_ASK_QTY, 0))
        if bid is None or bid_vol is None or ask is None or ask_vol is None:
            return {}
        return {
            "bid": bid,
            "bid_vol": bid_vol,
            "ask": ask,
            "ask_vol": ask_vol,
        }

    def build_history_backwards(self, anchor_time: int, anchor_cum_vol: float,
                                anchor_bid_vol: float, anchor_ask_vol: float,
                                snapshot_interval_ms: float):
        """Build historical snapshots ending at ``anchor_time``."""
        LOGGER.info("Loading Binance history for %s", self.symbol.upper())
        try:
            http_proxy, https_proxy = get_local_proxies()
            proxies = {}
            if http_proxy: proxies['http'] = http_proxy
            if https_proxy: proxies['https'] = https_proxy
            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)

            total_ms = self.total_length * snapshot_interval_ms
            target_start_ms = anchor_time - total_ms

            all_trades = []
            current_end = anchor_time
            req_count = 0
            used_weight_1m = 0
            total_fetched_len = 0

            while current_end > target_start_ms and req_count < MAX_FETCH_REQUESTS:
                url = f"{ENDPOINT_AGG_TRADES}?symbol={self.symbol.upper()}&endTime={int(current_end)}&limit={LIMIT_AGG_TRADES}"
                req = urllib.request.Request(url)
                data = None

                for retry in range(5):
                    try:
                        with urllib.request.urlopen(req, timeout=10) as response:

                            weight = response.headers.get('X-MBX-USED-WEIGHT-1M')
                            if weight and weight.isdigit():
                                used_weight_1m = int(weight)

                            data = json.loads(response.read().decode())
                        break
                    except Exception as e:

                        if hasattr(e, 'code') and e.code in (429, 418):
                            retry_after = e.headers.get('Retry-After', 5) if hasattr(e, 'headers') else 5
                            LOGGER.warning(
                                "Binance rate limit response %s; retrying in %s seconds",
                                e.code,
                                retry_after,
                            )
                            time.sleep(int(retry_after))
                            continue

                        LOGGER.warning("Historical trade request failed (attempt %d/5): %s", retry + 1, e)
                        time.sleep(2)

                if not data:
                    break

                all_trades.extend(data)
                total_fetched_len += len(data)
                current_end = data[0][TICK_TIME] - 1
                req_count += 1

                LOGGER.debug(
                    "Fetched %d aggregate trades; request weight %d/%d",
                    total_fetched_len,
                    used_weight_1m,
                    self._max_weight_1m,
                )

                if len(data) < LIMIT_AGG_TRADES:
                    data.clear()
                    break


                safe_threshold = int(self._max_weight_1m * 0.90)
                if used_weight_1m > safe_threshold:
                    LOGGER.warning(
                        "Binance request weight %d exceeded the safe threshold %d",
                        used_weight_1m,
                        safe_threshold,
                    )
                    time.sleep(1.5)


            LOGGER.info("Fetched %d aggregate trades for %s", total_fetched_len, self.symbol.upper())

            if not all_trades:
                return []

            all_trades.sort(key=lambda x: x[TICK_TIME])

            actual_start_ms = max(target_start_ms, all_trades[0][TICK_TIME])
            actual_snapshot_length = int((anchor_time - actual_start_ms) / snapshot_interval_ms)
            actual_snapshot_length = min(actual_snapshot_length, self.total_length)

            if actual_snapshot_length <= 0:
                return []

            t_size = self.tick_size if self.tick_size else DEFAULT_TICK_SIZE

            first_price = _safe_float(all_trades[0][TICK_PRICE])
            if first_price is None:
                return []

            last_price = first_price
            if all_trades[0][TICK_MAKER]:
                bid, ask = last_price, last_price + t_size
            else:
                bid, ask = last_price - t_size, last_price

            trade_idx = 0
            num_trades = len(all_trades)

            history_data = []
            total_historical_delta = 0.0

            for i in range(actual_snapshot_length):
                bucket_start = actual_start_ms + i * snapshot_interval_ms
                bucket_end = bucket_start + snapshot_interval_ms
                bucket_vol_delta = 0.0

                while trade_idx < num_trades and all_trades[trade_idx][TICK_TIME] < bucket_end:
                    t = all_trades[trade_idx]
                    p = _safe_float(t[TICK_PRICE])
                    q = _safe_float(t[TICK_QTY])
                    m = t[TICK_MAKER]

                    if p is not None and q is not None:
                        last_price = p
                        bucket_vol_delta += q

                        if m:
                            bid, ask = p, p + t_size
                        else:
                            bid, ask = p - t_size, p


                    all_trades[trade_idx] = None
                    trade_idx += 1

                total_historical_delta += bucket_vol_delta

                dt = datetime.fromtimestamp(bucket_end / 1000.0)
                local_time_str = dt.strftime('%H:%M:%S.%f')[:-3]


                history_data.append({
                    "local_time": local_time_str,
                    "symbol": self.symbol.upper(),
                    "last": round(last_price, 6),
                    "bid": round(bid, 6),
                    "ask": round(ask, 6),
                    "bid_vol": round(anchor_bid_vol, 4),
                    "ask_vol": round(anchor_ask_vol, 4),
                    "volume": bucket_vol_delta
                })


            all_trades.clear()
            del all_trades


            current_cum_vol = anchor_cum_vol - total_historical_delta
            for i in range(actual_snapshot_length):
                current_cum_vol += history_data[i]["volume"]
                history_data[i]["volume"] = round(current_cum_vol, 4)


            if actual_snapshot_length < self.total_length and history_data:
                missing_length = self.total_length - actual_snapshot_length
                LOGGER.warning(
                    "Binance returned %d/%d historical snapshots; padding %d leading samples",
                    actual_snapshot_length,
                    self.total_length,
                    missing_length,
                )

                padding_data = []
                for j in range(missing_length):
                    pad_item = history_data[0].copy()

                    pad_time_ms = actual_start_ms - (missing_length - 1 - j) * snapshot_interval_ms
                    pad_item["local_time"] = datetime.fromtimestamp(pad_time_ms / 1000.0).strftime('%H:%M:%S.%f')[:-3]
                    padding_data.append(pad_item)

                history_data = padding_data + history_data


                padding_data.clear()

            return history_data

        except Exception as e:
            LOGGER.exception("Unable to build Binance history for %s: %s", self.symbol.upper(), e)
            return []


@dataclass
class BinanceFuturesDataWS:
    """Thread-safe Binance Futures REST bootstrap and WebSocket sampler."""
    symbol: str = "BTCUSDT"
    total_length: int = 10000
    interval_ms: int = 100
    actual_snapshots_per_snapshot: int = 1

    _history_pool: deque = field(init=False)
    _current_state: dict = field(init=False)
    _is_updated: bool = field(init=False, default=False)
    _running: bool = field(init=False, default=False)
    _lock: threading.Lock = field(init=False)
    _ws: websocket.WebSocketApp = field(init=False, default=None)
    _last_msg_time: float = field(init=False, default=0.0)

    tick_size: float = field(init=False, default=None)
    synthesizer: BinanceHistorySynthesizer = field(init=False)

    def __post_init__(self):
        normalized_symbol = self.symbol.strip().upper()
        if not normalized_symbol or not normalized_symbol.isalnum():
            raise ValueError("symbol must contain only letters and digits")
        if self.total_length < 2:
            raise ValueError("total_length must be at least 2")
        if self.interval_ms <= 0:
            raise ValueError("interval_ms must be greater than zero")
        if self.actual_snapshots_per_snapshot < 1:
            raise ValueError("actual_snapshots_per_snapshot must be at least 1")

        http_proxy, https_proxy = get_local_proxies()
        if http_proxy: os.environ['http_proxy'] = http_proxy
        if https_proxy: os.environ['https_proxy'] = https_proxy

        self.symbol = normalized_symbol.lower()


        self._actual_snapshot_length = self.total_length * self.actual_snapshots_per_snapshot
        self._history_pool = deque(maxlen=self._actual_snapshot_length)

        self.tick_size = get_binance_tick_size(self.symbol)
        if self.tick_size is None or self.tick_size <= 0:
            raise ConnectionError(
                f"Could not resolve a positive Binance tick size for {normalized_symbol}"
            )
        self._lock = threading.Lock()
        self._last_msg_time = time.time()

        # ====================================================

        # ====================================================
        anchor_time = int(time.time() * 1000)
        anchor_bid_vol, anchor_ask_vol, anchor_cum_vol = 0.0, 0.0, 0.0
        anchor_bid, anchor_ask, last_price = 0.0, 0.0, 0.0

        max_retries = 3
        for attempt in range(max_retries):
            try:
                proxies = {}
                if http_proxy: proxies['http'] = http_proxy
                if https_proxy: proxies['https'] = https_proxy
                proxy_handler = urllib.request.ProxyHandler(proxies)
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)

                book_req = urllib.request.Request(f"{ENDPOINT_BOOK_TICKER}?symbol={self.symbol.upper()}")
                try:
                    with urllib.request.urlopen(book_req, timeout=10) as resp:
                        book_data = json.loads(resp.read().decode())
                        anchor_bid_vol = _safe_float(book_data.get('bidQty', 0)) or 0.0
                        anchor_ask_vol = _safe_float(book_data.get('askQty', 0)) or 0.0
                        anchor_bid = _safe_float(book_data.get('bidPrice', 0)) or 0.0
                        anchor_ask = _safe_float(book_data.get('askPrice', 0)) or 0.0
                except Exception as e:
                    raise RuntimeError(f"Binance book-ticker bootstrap failed: {e!r}") from e

                ticker_req = urllib.request.Request(f"{ENDPOINT_TICKER_24HR}?symbol={self.symbol.upper()}")
                try:
                    with urllib.request.urlopen(ticker_req, timeout=10) as resp:
                        ticker_data = json.loads(resp.read().decode())
                        anchor_cum_vol = _safe_float(ticker_data.get('volume', 0)) or 0.0
                        last_price = _safe_float(ticker_data.get('lastPrice', 0)) or 0.0
                        anchor_time = int(ticker_data.get('closeTime', anchor_time))
                except Exception as e:
                    raise RuntimeError(f"Binance 24-hour ticker bootstrap failed: {e!r}") from e

                LOGGER.info("Binance bootstrap completed for %s", self.symbol.upper())
                break
            except Exception as e:
                LOGGER.warning(
                    "Binance bootstrap failed for %s (attempt %d/%d): %r",
                    self.symbol.upper(),
                    attempt + 1,
                    max_retries,
                    e,
                )


                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    LOGGER.debug("Running network diagnostics after a Binance timeout")


                    try:
                        ip_req = urllib.request.Request("https://api.ipify.org?format=json")
                        with urllib.request.urlopen(ip_req, timeout=10) as ip_resp:
                            ip_info = json.loads(ip_resp.read().decode())
                            LOGGER.debug("Public IP: %s", ip_info.get('ip'))
                    except Exception as ip_e:
                        LOGGER.debug("Public-IP diagnostic failed: %r", ip_e)


                    try:
                        ping_req = urllib.request.Request("https://api.binance.com/api/v3/ping")
                        with urllib.request.urlopen(ping_req, timeout=10) as ping_resp:
                            LOGGER.debug("Binance spot ping returned HTTP %s", ping_resp.status)
                    except Exception as ping_e:
                        if hasattr(ping_e, 'code') and ping_e.code in (403, 451):
                            LOGGER.warning("Binance access diagnostic returned HTTP %s", ping_e.code)
                        else:
                            LOGGER.debug("Binance ping diagnostic failed: %r", ping_e)

                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise ConnectionError(
                        f"Could not bootstrap Binance Futures data for {self.symbol.upper()}"
                    ) from e

        self._current_state = {
            "last": last_price,
            "bid": anchor_bid,
            "bid_vol": anchor_bid_vol,
            "ask": anchor_ask,
            "ask_vol": anchor_ask_vol,
            "volume": anchor_cum_vol
        }

        self.synthesizer = BinanceHistorySynthesizer(
            symbol=self.symbol,
            total_length=self._actual_snapshot_length,
            tick_size=self.tick_size
        )
        self.synthesizer.set_initial_state(anchor_cum_vol, 0)

        # ====================================================


        # ====================================================
        self._start()
        LOGGER.info("Binance WebSocket sampling started for %s", self.symbol.upper())

        # ====================================================

        # ====================================================

        history_synth = BinanceHistorySynthesizer(
            symbol=self.symbol,
            total_length=self._actual_snapshot_length,
            tick_size=self.tick_size,
        )

        history_data = history_synth.build_history_backwards(
            anchor_time, anchor_cum_vol, anchor_bid_vol, anchor_ask_vol, self.interval_ms
        )

        # ====================================================

        # ====================================================
        if history_data:
            with self._lock:
                current_realtime_data = list(self._history_pool)
                merged = history_data + current_realtime_data
                self._history_pool = deque(merged[-self._actual_snapshot_length:], maxlen=self._actual_snapshot_length)
                self._is_updated = True
                LOGGER.info("Loaded %d initial snapshots", len(self._history_pool))

    def _on_message(self, ws, message):
        self._last_msg_time = time.time()
        try:
            data = json.loads(message)
            stream = data.get("stream", "")
            payload = data.get("data", {})

            with self._lock:
                if "@bookTicker" in stream:
                    snap_data = self.synthesizer.process_realtime_snapshot(payload)
                    self._current_state.update(snap_data)
                elif "@aggTrade" in stream:
                    last_price = self.synthesizer.process_realtime_tick(payload)
                    if last_price is not None:
                        self._current_state["last"] = last_price
                        self._current_state["volume"] = self.synthesizer.get_current_volume()
        except Exception:
            pass

    def _is_valid_market_state(self, state: dict) -> bool:
        required = ('last', 'bid', 'bid_vol', 'ask', 'ask_vol', 'volume')
        for k in required:
            if k not in state:
                return False
            v = _safe_float(state.get(k))
            if v is None:
                return False
        return True

    def _is_valid_snapshot(self, snap: dict) -> bool:
        if snap.get('local_time') is None:
            return False
        return self._is_valid_market_state(snap)

    def _on_error(self, ws, error):
        LOGGER.warning("Binance WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        LOGGER.info("Binance WebSocket closed: %s %s", close_status_code, close_msg)

    def _on_open(self, ws):
        LOGGER.info("Binance WebSocket connected for %s", self.symbol.upper())

    def _ws_loop(self):
        stream_url = f"{WS_BASE_URL}/stream?streams={self.symbol}@bookTicker/{self.symbol}@aggTrade"
        self._ws = websocket.WebSocketApp(
            stream_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self._ws.run_forever()

    def _sample_loop(self):
        last_snapshot = None

        # ==========================================

        # ==========================================


        while self._running:
            with self._lock:
                last = self._current_state.get('last', 0.0)
            if last != 0.0:
                break
            time.sleep(INIT_SLEEP)

        # ==========================================

        # ==========================================
        warning_interval = max(5.0, (self.interval_ms * 10) / 1000.0)
        last_ws_warn_time = time.time()
        last_snapshot_warn_time = time.time()

        while self._running:


            loop_start_time = time.time()


            if loop_start_time - self._last_msg_time > warning_interval:
                if loop_start_time - last_ws_warn_time > warning_interval:
                    LOGGER.warning("No Binance WebSocket message received for %.1f seconds", warning_interval)
                    last_ws_warn_time = loop_start_time


            if loop_start_time - last_snapshot_warn_time > warning_interval:
                LOGGER.warning("No valid market snapshot produced for %.1f seconds", warning_interval)
                last_snapshot_warn_time = loop_start_time




            with self._lock:
                current_snapshot = self._current_state.copy()

            # ==========================================

            # ==========================================


            if current_snapshot != last_snapshot:
                if not self._is_valid_market_state(current_snapshot):
                    time.sleep(max(0.0, (self.interval_ms / MS_PER_SCD) - (time.time() - loop_start_time)))
                    continue


                full_snapshot = {

                    "local_time": pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3],

                    "symbol": self.symbol.upper(),

                    **current_snapshot
                }

                if not self._is_valid_snapshot(full_snapshot):
                    time.sleep(max(0.0, (self.interval_ms / MS_PER_SCD) - (time.time() - loop_start_time)))
                    continue


                try:
                    full_snapshot['bid'] = float(full_snapshot['bid'])
                    full_snapshot['ask'] = float(full_snapshot['ask'])
                    full_snapshot['last'] = float(full_snapshot['last'])
                    full_snapshot['bid_vol'] = float(full_snapshot['bid_vol'])
                    full_snapshot['ask_vol'] = float(full_snapshot['ask_vol'])
                    full_snapshot['volume'] = float(full_snapshot['volume'])
                except (ValueError, TypeError):
                    continue

                with self._lock:
                    self._history_pool.append(full_snapshot)
                    self._is_updated = True


                last_snapshot = current_snapshot
                last_snapshot_warn_time = time.time()

            # ==========================================

            # ==========================================

            cost_time = time.time() - loop_start_time





            sleep_time = max(0.0, (self.interval_ms / MS_PER_SCD) - cost_time)


            time.sleep(sleep_time)

    def reconnect(self):
        """Reconnect the stream and backfill the detected data gap."""
        LOGGER.info("Reconnecting Binance stream for %s", self.symbol.upper())


        self._running = False
        if getattr(self, '_ws', None):
            try:
                self._ws.close()
            except Exception:
                pass

        time.sleep(1)


        now_time = time.time()

        missing_ms = (now_time - self._last_msg_time) * 1000
        missing_length = int(missing_ms / self.interval_ms)

        if missing_length <= 0:
            LOGGER.info("No Binance snapshot gap detected")
            self._last_msg_time = time.time()
            self._start()
            return


        if missing_length > self._actual_snapshot_length:
            missing_length = self._actual_snapshot_length

        LOGGER.info(
            "Backfilling up to %d snapshots after a %.1f-second gap",
            missing_length,
            missing_ms / 1000,
        )


        anchor_time = int(time.time() * 1000)
        anchor_bid_vol, anchor_ask_vol, anchor_cum_vol = 0.0, 0.0, 0.0
        anchor_bid, anchor_ask, last_price = 0.0, 0.0, 0.0

        http_proxy, https_proxy = get_local_proxies()

        for attempt in range(3):
            try:
                proxies = {}
                if http_proxy: proxies['http'] = http_proxy
                if https_proxy: proxies['https'] = https_proxy
                proxy_handler = urllib.request.ProxyHandler(proxies)
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)

                book_req = urllib.request.Request(f"{ENDPOINT_BOOK_TICKER}?symbol={self.symbol.upper()}")
                with urllib.request.urlopen(book_req, timeout=5) as resp:
                    book_data = json.loads(resp.read().decode())
                    anchor_bid_vol = _safe_float(book_data.get('bidQty', 0)) or 0.0
                    anchor_ask_vol = _safe_float(book_data.get('askQty', 0)) or 0.0
                    anchor_bid = _safe_float(book_data.get('bidPrice', 0)) or 0.0
                    anchor_ask = _safe_float(book_data.get('askPrice', 0)) or 0.0

                ticker_req = urllib.request.Request(f"{ENDPOINT_TICKER_24HR}?symbol={self.symbol.upper()}")
                with urllib.request.urlopen(ticker_req, timeout=5) as resp:
                    ticker_data = json.loads(resp.read().decode())
                    anchor_cum_vol = _safe_float(ticker_data.get('volume', 0)) or 0.0
                    last_price = _safe_float(ticker_data.get('lastPrice', 0)) or 0.0
                    anchor_time = int(ticker_data.get('closeTime', anchor_time))
                break
            except Exception as e:
                LOGGER.warning("Binance reconnect bootstrap failed: %s", e)
                time.sleep(1)

        with self._lock:
            self._current_state.update({
                "last": last_price,
                "bid": anchor_bid,
                "bid_vol": anchor_bid_vol,
                "ask": anchor_ask,
                "ask_vol": anchor_ask_vol,
                "volume": anchor_cum_vol
            })
            self.synthesizer.set_initial_state(anchor_cum_vol, self.synthesizer._last_agg_id)
            old_pool_data = list(self._history_pool)


        self._last_msg_time = time.time()
        self._start()
        LOGGER.info("Binance WebSocket sampling restarted")

        history_synth = BinanceHistorySynthesizer(
            symbol=self.symbol,
            total_length=missing_length,
            tick_size=self.tick_size,
        )

        history_data = history_synth.build_history_backwards(
            anchor_time, anchor_cum_vol, anchor_bid_vol, anchor_ask_vol, self.interval_ms
        )


        if history_data:
            with self._lock:
                new_realtime_data = list(self._history_pool)



                merged = old_pool_data + history_data + new_realtime_data


                seen_times = set()
                deduped = []
                for item in reversed(merged):
                    if item["local_time"] not in seen_times:
                        seen_times.add(item["local_time"])
                        deduped.append(item)

                deduped.reverse()


                self._history_pool = deque(deduped[-self._actual_snapshot_length:], maxlen=self._actual_snapshot_length)
                self._is_updated = True
                LOGGER.info("Reconnected with %d buffered snapshots", len(self._history_pool))


                history_data.clear()
                old_pool_data.clear()
                new_realtime_data.clear()
                merged.clear()
                deduped.clear()

    def _start(self):
        if not self._running:
            self._running = True
            threading.Thread(target=self._ws_loop, daemon=True).start()
            threading.Thread(target=self._sample_loop, daemon=True).start()

    def __del__(self):
        self._running = False
        if getattr(self, '_ws', None):
            self._ws.close()

    @property
    def is_updated(self) -> bool:
        with self._lock:
            return bool(self._is_updated)

    def fetch(self, length=None, force=False):
        """Return the latest sampled snapshots as a DataFrame."""
        with self._lock:
            if not self._is_updated and not force:
                return pd.DataFrame()

            pool_len = len(self._history_pool)

            actual_snapshot_length = length
            if length is not None and length > 0 and self.actual_snapshots_per_snapshot > 1:
                actual_snapshot_length = length * self.actual_snapshots_per_snapshot

            if actual_snapshot_length is not None and actual_snapshot_length > 0 and actual_snapshot_length < pool_len:
                start_idx = pool_len - actual_snapshot_length
                pool_data = list(itertools.islice(self._history_pool, start_idx, pool_len))
            else:
                pool_data = tuple(self._history_pool)

            self._is_updated = False

        if not pool_data:
            return pd.DataFrame()


        df = pd.DataFrame(pool_data)

        if self.actual_snapshots_per_snapshot > 1:
            df = df.iloc[::-self.actual_snapshots_per_snapshot][::-1].reset_index(drop=True)
            if length is not None and length > 0:
                df = df.tail(length).reset_index(drop=True)

        return df
