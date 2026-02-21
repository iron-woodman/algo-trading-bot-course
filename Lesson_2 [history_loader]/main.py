"""Основной модуль для выгрузки истории OHLCV с Bybit с использованием Polars и multiprocessing."""

import os
import time
import ccxt
import polars as pl
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv
from ohlc_data_checker import check_ohlcv_gaps

# Загрузка переменных окружения
load_dotenv()

# Глобальная конфигурация из .env
YEAR: int = int(os.getenv('YEAR', '2025'))
TIMEFRAME: str = os.getenv('TIMEFRAME', '30m')
LIMIT: int = int(os.getenv('LIMIT', '200'))
DATA_DIR: str = os.getenv('DATA_DIR', 'data')
MIN_VOLUME_USDT: float = float(os.getenv('MIN_VOLUME_USDT', '1000000.0'))
TARGET_COINS_FILE: str = os.getenv('TARGET_COINS_FILE', 'TARGET_COINS.txt')
ALL_COINS_FILE: str = os.getenv('ALL_COINS_FILE', 'ALL_COINS.txt')
SORTED_COINS_FILE: str = os.getenv('SORTED_COINS_FILE', 'SORTED_COINS.txt')
MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', str(cpu_count())))

BYBIT_API_KEY: str = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET: str = os.getenv('BYBIT_API_SECRET', '')


def get_exchange() -> ccxt.bybit:
    """Инициализация экземпляра биржи Bybit."""
    config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    }
    if BYBIT_API_KEY and BYBIT_API_SECRET:
        config['apiKey'] = BYBIT_API_KEY
        config['secret'] = BYBIT_API_SECRET
    
    return ccxt.bybit(config)


def fetch_ohlcv_batch(exchange: ccxt.bybit, symbol: str, timeframe: str, since: int, limit: int) -> list[list[Any]]:
    """Выполняет запрос к API для получения батча OHLCV данных."""
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        print(f"\nОшибка API при загрузке {symbol}: {e}. Повтор через 5 сек...")
        time.sleep(5)
        return []


def save_all_usdt_futures() -> list[str]:
    """Извлекает список ликвидных бессрочных USDT фьючерсов и сохраняет в файл."""
    exchange = get_exchange()
    print("Получение списка инструментов...")
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"Ошибка при загрузке списка инструментов: {e}")
        return []

    candidates: list[str] = []
    for symbol, market in markets.items():
        is_usdt: bool = market.get('quote') == 'USDT'
        is_swap: bool = market.get('type') == 'swap'
        is_linear: bool = market.get('linear', False)
        if is_usdt and is_swap and is_linear:
            candidates.append(symbol)

    if not candidates:
        print("Инструменты не найдены.")
        return []

    print(f"Запрос объемов торгов для {len(candidates)} инструментов...")
    # Словарь для хранения символ: объем
    symbol_volumes: dict[str, float] = {}
    
    try:
        tickers = exchange.fetch_tickers(candidates)
        for symbol, ticker in tickers.items():
            volume_usdt: float = float(ticker.get('quoteVolume') or 0.0)
            if volume_usdt >= MIN_VOLUME_USDT:
                symbol_volumes[symbol] = volume_usdt
            
    except Exception as e:
        print(f"Ошибка при получении объемов (возможно, Rate Limit): {e}")
        print("Будет использован полный список без фильтрации по объему.")
        # Если не удалось получить объемы, считаем все кандидаты с нулевым объемом
        symbol_volumes = {s: 0.0 for s in candidates}

    # 1. Алфавитный список (как было раньше)
    final_list_alpha: list[str] = sorted(symbol_volumes.keys())
    with open(ALL_COINS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_list_alpha))
    print(f"Список инструментов ({len(final_list_alpha)}) сохранен в {ALL_COINS_FILE}")

    # 2. Список, отсортированный по ликвидности (убывание объема)
    # Сортируем ключи словаря по значениям (объемам) в обратном порядке
    final_list_sorted: list[str] = sorted(symbol_volumes.keys(), key=lambda x: symbol_volumes[x], reverse=True)
    with open(SORTED_COINS_FILE, 'w', encoding='utf-8') as f:
        # Сохраняем в формате "SYMBOL: VOLUME" для наглядности (опционально)
        # Но для совместимости с TARGET_COINS.txt лучше просто SYMBOL
        f.write('\n'.join(final_list_sorted))
    print(f"Список, отсортированный по ликвидности, сохранен в {SORTED_COINS_FILE}")

    return final_list_alpha


def fill_gaps_automatically(exchange: ccxt.bybit, symbol: str, timeframe: str, df: pl.DataFrame, 
                            start: Optional[datetime] = None, end: Optional[datetime] = None) -> pl.DataFrame:
    """Ищет и автоматически подгружает пропущенные данные OHLCV."""
    print(f"[{symbol}] Проверка на наличие пропусков...")
    report, gaps_df = check_ohlcv_gaps(df, timeframe=timeframe, start=start, end=end)
    
    if gaps_df.is_empty():
        print(f"[{symbol}] Пропусков не обнаружено.")
        return df

    print(f"[{symbol}] Обнаружено пропусков: {report['missing_points']}. Начинаю дозагрузку...")
    
    # Для Polars мы будем собирать новые данные в список и потом объединять
    new_batches: list[list[Any]] = []
    
    for gap in gaps_df.to_dicts():
        since: int = int(gap['gap_start'].timestamp() * 1000)
        ohlcv = fetch_ohlcv_batch(exchange, symbol, timeframe, since, LIMIT)
        if ohlcv:
            new_batches.extend(ohlcv)
            time.sleep(exchange.rateLimit / 1000 if exchange.rateLimit else 0.2)

    if new_batches:
        new_df = pl.DataFrame(new_batches, schema=['timestamp', 'open', 'high', 'low', 'close', 'volume'], orient="row")
        # Преобразование timestamp в datetime для конкатенации
        new_df = new_df.with_columns(
            pl.from_epoch(pl.col('timestamp'), time_unit="ms").dt.replace_time_zone("UTC").alias('datetime')
        )
        # Объединение, удаление дубликатов и сортировка
        df = pl.concat([df, new_df]).unique(subset=['timestamp']).sort('timestamp')
        print(f"[{symbol}] Дозагрузка завершена. Добавлено свечей: {len(new_batches)}")
    
    return df


def download_history_for_symbol(symbol: str) -> None:
    """Загружает и сохраняет историю OHLCV для одного символа в формате Parquet."""
    # Формирование пути к файлу для проверки существования
    clean_symbol: str = symbol.split(':')[0].replace('/', '_')
    outfile: str = os.path.join(DATA_DIR, f'{clean_symbol}_{TIMEFRAME}_{YEAR}.parquet')

    if os.path.exists(outfile):
        print(f"[{symbol}] Файл истории {outfile} уже существует. Пропускаю загрузку.")
        return

    exchange = get_exchange()
    print(f"Начало загрузки истории для {symbol}, таймфрейм {TIMEFRAME}")

    # Настройка временных границ
    start_dt = datetime(YEAR, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(YEAR + 1, 1, 1, tzinfo=timezone.utc)
    start_ts: int = int(start_dt.timestamp() * 1000)
    end_ts: int = int(end_dt.timestamp() * 1000)
    
    # Расчет шага (таймфрейма) в миллисекундах
    tf_ms: int = exchange.parse_timeframe(TIMEFRAME) * 1000
    
    all_ohlcv: list[list[Any]] = []
    current_since: int = start_ts

    while current_since < end_ts:
        ohlcv = fetch_ohlcv_batch(exchange, symbol, TIMEFRAME, current_since, LIMIT)
        if not ohlcv:
            break
        
        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        # Предотвращение зацикливания и корректный переход к следующему батчу
        current_since = max(last_ts + tf_ms, current_since + tf_ms)
        
        # Печать прогресса для каждого процесса может быть шумной, но полезной
        if len(all_ohlcv) % (LIMIT * 5) == 0:
            print(f"[{symbol}] Загружено: {len(all_ohlcv)} свечей")
            
        time.sleep(exchange.rateLimit / 1000 if exchange.rateLimit else 0.1)

    if not all_ohlcv:
        print(f"[{symbol}] Данные не загружены. Проверьте символ.")
        return

    # Создание Polars DataFrame
    df = pl.DataFrame(all_ohlcv, schema=['timestamp', 'open', 'high', 'low', 'close', 'volume'], orient="row")
    df = df.with_columns(
        pl.from_epoch(pl.col('timestamp'), time_unit="ms").dt.replace_time_zone("UTC").alias('datetime')
    )
    
    # Автозаполнение пропусков
    df = fill_gaps_automatically(exchange, symbol, TIMEFRAME, df, start=start_dt, end=end_dt)

    # Фильтрация по строгому интервалу
    df = df.filter((pl.col('datetime') >= start_dt) & (pl.col('datetime') < end_dt))

    # Сохранение истории в формате Parquet
    clean_symbol: str = symbol.split(':')[0].replace('/', '_')
    outfile: str = os.path.join(DATA_DIR, f'{clean_symbol}_{TIMEFRAME}_{YEAR}.parquet')
    df.write_parquet(outfile)
    
    print(f"[{symbol}] Выгрузка завершена. Итого свечей: {len(df)}. Файл: {outfile}")


def main() -> None:
    """Точка входа в программу."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Обновляем список ликвидных монет (опционально)
    _ = save_all_usdt_futures()

    # 2. Читаем целевые монеты
    if not os.path.exists(TARGET_COINS_FILE):
        print(f"Файл {TARGET_COINS_FILE} не найден.")
        return

    with open(TARGET_COINS_FILE, 'r', encoding='utf-8') as f:
        target_symbols: list[str] = [line.strip() for line in f if line.strip()]

    if not target_symbols:
        print(f"Файл {TARGET_COINS_FILE} пуст.")
        return

    print(f"Загрузка истории для {len(target_symbols)} инструментов с использованием {MAX_WORKERS} воркеров")

    # Начало замера времени загрузки истории
    start_time: float = time.time()
    start_dt_now: datetime = datetime.now()
    print(f"\nНачало процесса загрузки: {start_dt_now.strftime('%Y-%m-%d %H:%M:%S')}")

    # 3. Параллельная загрузка истории с использованием multiprocessing.Pool
    with Pool(processes=MAX_WORKERS) as pool:
        pool.map(download_history_for_symbol, target_symbols)

    # Окончание замера времени
    end_time: float = time.time()
    end_dt_now: datetime = datetime.now()
    duration: float = end_time - start_time
    
    print(f"\nЗагрузка истории завершена: {end_dt_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Затрачено времени: {timedelta(seconds=int(duration))}")

    print("\nВсе задачи по выгрузке завершены.")


if __name__ == "__main__":
    main()
