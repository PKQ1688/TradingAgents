import pandas as pd
import yfinance as yf
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        data_dir: Annotated[
            str,
            "directory where the stock data is stored (deprecated, now uses yfinance API).",
        ],
        online: Annotated[
            bool,
            "whether to use online tools to fetch data or offline tools. Now always uses yfinance API.",
        ] = True,
    ):
        """
        Get stock statistics using yfinance API with caching for performance.
        """
        try:
            # Get today's date as YYYY-mm-dd to add to cache
            today_date = pd.Timestamp.today()
            curr_date_dt = pd.to_datetime(curr_date)

            end_date = today_date
            start_date = today_date - pd.DateOffset(
                years=2
            )  # Reduced to 2 years for better performance
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")

            # Get config and ensure cache directory exists
            config = get_config()
            os.makedirs(config["data_cache_dir"], exist_ok=True)

            data_file = os.path.join(
                config["data_cache_dir"],
                f"{symbol}-YFin-data-{start_date}-{end_date}.csv",
            )

            # Check if cached data exists and is recent (less than 1 day old)
            if os.path.exists(data_file):
                file_age = pd.Timestamp.now() - pd.Timestamp.fromtimestamp(
                    os.path.getmtime(data_file)
                )
                if file_age.days < 1:
                    data = pd.read_csv(data_file)
                    data["Date"] = pd.to_datetime(data["Date"])
                else:
                    # Cache is old, fetch new data
                    data = yf.download(
                        symbol,
                        start=start_date,
                        end=end_date,
                        multi_level_index=False,
                        progress=False,
                        auto_adjust=True,
                    )
                    data = data.reset_index()
                    data.to_csv(data_file, index=False)
            else:
                # No cache, fetch new data
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    multi_level_index=False,
                    progress=False,
                    auto_adjust=True,
                )
                data = data.reset_index()
                data.to_csv(data_file, index=False)

            df = wrap(data)
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            curr_date = curr_date_dt.strftime("%Y-%m-%d")

            df[indicator]  # trigger stockstats to calculate the indicator
            matching_rows = df[df["Date"].str.startswith(curr_date)]

            if not matching_rows.empty:
                indicator_value = matching_rows[indicator].values[0]
                return indicator_value
            else:
                return "N/A: Not a trading day (weekend or holiday)"

        except Exception as e:
            return f"Error calculating {indicator} for {symbol}: {str(e)}"
