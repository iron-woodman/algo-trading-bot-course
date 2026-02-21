"""Основной модуль для выгрузки списка инструментов и истории OHLCV с Bybit."""
import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timezone
from typing import Any, Optional
from ohlc_data_checker import check_ohlcv_gaps

# Конфигурация
YEAR: int = 2025
TIMEFRAME: str = '30m'
LIMIT: int = 200
DATA_DIR: str = 'data'
MIN_VOLUME_USDT: float = 1_000_000.0  # Минимальный суточный объем торгов для включения в список


def fetch_ohlcv_batch(exchange: ccxt.bybit, symbol: str, timeframe: str, since: int, limit: int) -> list[list[Any]]:
    """Выполняет запрос к API с обработкой ошибок."""
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        print(f"\nОшибка API при загрузке {symbol}: {e}. Повтор через 5 сек...")
        time.sleep(5)
        return []


def save_all_usdt_futures(exchange: ccxt.bybit) -> list[str]:
    """Извлекает список ликвидных бессрочных USDT фьючерсов и сохраняет в файл."""
    print("Получение списка инструментов...")
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"Ошибка при загрузке списка инструментов: {e}")
        return []

    # Первичный список кандидатов (бессрочные USDT фьючерсы)
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
    final_list: list[str] = []
    try:
        # Запрашиваем тикеры для всех кандидатов сразу
        tickers = exchange.fetch_tickers(candidates)
        for symbol, ticker in tickers.items():
            # Извлекаем суточный объем в котируемой валюте (USDT)
            volume_usdt: float = float(ticker.get('quoteVolume') or 0.0)
            if volume_usdt >= MIN_VOLUME_USDT:
                final_list.append(symbol)
            
    except Exception as e:
        print(f"Ошибка при получении объемов (возможно, Rate Limit): {e}")
        print("Будет использован полный список без фильтрации по объему.")
        final_list = candidates

    coins_file: str = 'ALL_COINS.txt'
    with open(coins_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(final_list)))
    
    print(f"Список ликвидных инструментов ({len(final_list)}) сохранен в {coins_file}")
    return final_list


def fill_gaps_automatically(exchange: ccxt.bybit, symbol: str, timeframe: str, df: pd.DataFrame, 
                            start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Ищет и автоматически подгружает пропущенные данные."""
    print("Проверка на наличие пропусков...")
    report, gaps_df = check_ohlcv_gaps(df, timeframe=timeframe, start=start, end=end)
    
    if gaps_df.empty:
        print("Пропусков не обнаружено.")
        return df

    print(f"Обнаружено пропусков: {report['missing_points']}. Начинаю дозагрузку...")
    
    added_count: int = 0
    for _, gap in gaps_df.iterrows():
        since: int = int(gap['gap_start'].timestamp() * 1000)
        ohlcv = fetch_ohlcv_batch(exchange, symbol, timeframe, since, LIMIT)
        if ohlcv:
            new_data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_data['datetime'] = pd.to_datetime(new_data['timestamp'], unit='ms', utc=True)
            df = pd.concat([df, new_data]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            added_count += len(ohlcv)
            time.sleep(exchange.rateLimit / 1000 if exchange.rateLimit else 0.2)

    print(f"Дозагрузка завершена. Добавлено свечей: {added_count}")
    return df


def download_history_for_symbol(exchange: ccxt.bybit, symbol: str, timeframe: str, year: int) -> None:
    """Загружает и сохраняет историю OHLCV для одного символа."""
    print(f"\nНачало загрузки истории для {symbol}, таймфрейм {timeframe}")

    # Настройка временных границ
    start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    start_ts: int = int(start_dt.timestamp() * 1000)
    end_ts: int = int(end_dt.timestamp() * 1000)
    
    # Расчет шага в миллисекундах
    tf_min = 5
    if timeframe.endswith('m'): tf_min = int(timeframe[:-1])
    elif timeframe.endswith('h'): tf_min = int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'): tf_min = int(timeframe[:-1]) * 1440
    tf_ms: int = tf_min * 60 * 1000
    
    all_ohlcv: list[list[Any]] = []
    current_since: int = start_ts

    while current_since < end_ts:
        ohlcv = fetch_ohlcv_batch(exchange, symbol, timeframe, current_since, LIMIT)
        if not ohlcv:
            break
        
        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        current_since = max(last_ts + tf_ms, current_since + tf_ms)
        
        print(f"Загружено: {len(all_ohlcv)} свечей", end='\r')
        time.sleep(exchange.rateLimit / 1000 if exchange.rateLimit else 0.1)

    if not all_ohlcv:
        print(f"\nДанные для {symbol} не загружены. Проверьте доступность биржи или корректность символа.")
        return

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    
    # Автозаполнение пропусков и финальная фильтрация
    start_pdt = pd.Timestamp(start_dt)
    end_pdt = pd.Timestamp(end_dt)
    
    # Сначала заполняем пропуски (внутри функции check_ohlcv_gaps произойдет реиндексация)
    df = fill_gaps_automatically(exchange, symbol, timeframe, df, start=start_pdt, end=end_pdt)

    # Строгая проверка на соответствие интервалу (удаляем лишние свечи, если они были подгружены в батчах)
    initial_count: int = len(df)
    df = df[(df['datetime'] >= start_pdt) & (df['datetime'] < end_pdt)]
    removed_count: int = initial_count - len(df)
    
    if removed_count > 0:
        print(f"Удалено {removed_count} свечей, вышедших за границы интервала {year} года.")

    # Сохранение истории в data/
    clean_symbol: str = symbol.split(':')[0].replace('/', '_')
    outfile: str = os.path.join(DATA_DIR, f'{clean_symbol}_{timeframe}_{year}.csv')
    df.to_csv(outfile, index=False)
    
    print(f"\nВыгрузка для {symbol} завершена.")
    print(f"Файл истории: {outfile}")
    print(f"Итого свечей: {len(df)}")


def main() -> None:
    """Точка входа."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Инициализация биржи
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })

    # 1. Сохраняем список всех ликвидных монет
    _ = save_all_usdt_futures(exchange)

    # 2. Читаем список целевых монет из TARGET_COINS.txt
    target_coins_file: str = 'TARGET_COINS.txt'
    if not os.path.exists(target_coins_file):
        print(f"Файл {target_coins_file} не найден. Обработка завершена.")
        return

    with open(target_coins_file, 'r', encoding='utf-8') as f:
        target_symbols: list[str] = [line.strip() for line in f if line.strip()]

    if not target_symbols:
        print(f"Файл {target_coins_file} пуст. Обработка завершена.")
        return

    print(f"Загрузка истории для {len(target_symbols)} инструментов из {target_coins_file}")

    # 3. Последовательная загрузка истории для каждой монеты
    for symbol in target_symbols:
        download_history_for_symbol(exchange, symbol, TIMEFRAME, YEAR)

    print("\nВсе задачи выполнены.")


if __name__ == "__main__":
    main()
