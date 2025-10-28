import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from configuration.Logger_config import setup_logger, logger


class MarketDataDownloader:
    """
    Advanced market data downloader with context data integration
    """

    def __init__(self, config):
        self.config = config
        self.stocks_list = config['data_download']['stocks_list']
        self.start_date = config['data_download']['start_date']
        self.end_date = config['data_download'].get('end_date') or datetime.datetime.now().strftime('%Y-%m-%d')

    def download_stock_data(self):
        """
        Download historical stock data
        """
        logger.info(f"Downloading stock data for {len(self.stocks_list)} symbols...")
        logger.info(f"Period: {self.start_date} to {self.end_date}")

        try:
            # Download with progress
            data = yf.download(
                self.stocks_list,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=True
            )

            # Handle single vs multiple stocks
            if len(self.stocks_list) == 1:
                data['Stock'] = self.stocks_list[0]
                data = data.reset_index()
            else:
                # Convert multi-index to columns
                data = data.stack(level='Ticker', future_stack=True).reset_index()
                data.rename(columns={'Ticker': 'Stock'}, inplace=True)

            # Standardize column names
            data.rename(columns={
                'Date': 'Date',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }, inplace=True)

            # Ensure correct column order
            cols = ['Date', 'Stock', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data[[col for col in cols if col in data.columns]]

            # Sort by stock and date
            data = data.sort_values(by=['Stock', 'Date']).reset_index(drop=True)

            # Basic data quality checks
            logger.info(f"Downloaded {len(data)} rows")
            logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            logger.info(f"Stocks: {data['Stock'].unique().tolist()}")

            # Check for missing values
            missing = data.isnull().sum()
            if missing.sum() > 0:
                logger.warning(f"Missing values detected:\n{missing[missing > 0]}")

            return data

        except Exception as e:
            logger.error(f"Error downloading stock data: {e}")
            raise

    def download_market_indices(self):
        """
        Download market indices for context (SPY, VIX, etc.)
        """
        logger.info("Downloading market indices...")

        indices = {
            'SPY': 'S&P 500',
            '^VIX': 'VIX',
            'DX-Y.NYB': 'Dollar Index',
            '^TNX': '10Y Treasury Yield'
        }

        index_data = {}

        for symbol, name in indices.items():
            try:
                data = yf.download(
                    symbol,
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    progress=False
                )

                if not data.empty:
                    index_data[symbol] = data[['Close']].rename(columns={'Close': name})
                    logger.info(f"Downloaded {name}: {len(data)} rows")

            except Exception as e:
                logger.warning(f"Could not download {name}: {e}")

        # Combine all indices
        if index_data:
            combined = pd.concat(index_data.values(), axis=1)

            # Flatten columns if multi-index
            if isinstance(combined.columns, pd.MultiIndex):
                combined.columns = [col[0] if isinstance(col, tuple) else col for col in combined.columns]

            # Reset index to get Date as column
            combined = combined.reset_index()

            # Rename index column to Date if needed
            if 'index' in combined.columns:
                combined = combined.rename(columns={'index': 'Date'})

            return combined

        return None

    def download_economic_data(self):
        """
        Download economic indicators from FRED
        Requires fredapi library and API key
        """
        logger.info("Attempting to download economic data...")

        try:
            from fredapi import Fred

            # Try to import API key from api_keys.py first, then fall back to env var
            try:
                from configuration.api_keys import FRED_API_KEY
                api_key = FRED_API_KEY
            except ImportError:
                import os
                api_key = os.getenv('FRED_API_KEY')

            if not api_key or api_key == "your_fred_api_key_here":
                logger.warning("FRED_API_KEY not configured in api_keys.py")
                logger.info("Get a free key from: https://fred.stlouisfed.org/docs/api/api_key.html")
                return None

            fred = Fred(api_key=api_key)

            indicators = self.config.get('features', {}).get('macro_economic', {}).get('indicators', [])

            if not indicators:
                logger.info("No economic indicators configured")
                return None

            economic_data = {}

            for indicator in indicators:
                try:
                    data = fred.get_series(indicator, observation_start=self.start_date)
                    economic_data[indicator] = data
                    logger.info(f"Downloaded {indicator}: {len(data)} observations")
                except Exception as e:
                    logger.warning(f"Could not download {indicator}: {e}")

            if economic_data:
                combined = pd.DataFrame(economic_data)
                combined = combined.reset_index()
                combined.rename(columns={'index': 'Date'}, inplace=True)
                return combined

        except ImportError:
            logger.warning("fredapi not installed. Skipping economic data.")
        except Exception as e:
            logger.warning(f"Error downloading economic data: {e}")

        return None

    def download_news_sentiment(self):
        """
        Download and analyze news sentiment
        Requires newsapi library and API key
        """
        logger.info("Attempting to download news sentiment...")

        try:
            from newsapi import NewsApiClient

            # Try to import API key from api_keys.py first, then fall back to env var
            try:
                from configuration.api_keys import NEWS_API_KEY
                api_key = NEWS_API_KEY
            except ImportError:
                import os
                api_key = os.getenv('NEWS_API_KEY')

            if not api_key or api_key == "your_news_api_key_here":
                logger.warning("NEWS_API_KEY not configured in api_keys.py")
                logger.info("Get a free key from: https://newsapi.org/")
                return None

            newsapi = NewsApiClient(api_key=api_key)

            sentiment_data = []

            for stock in self.stocks_list:
                try:
                    # Get company name from ticker
                    ticker = yf.Ticker(stock)
                    company_name = ticker.info.get('longName', stock)

                    # Get recent news
                    articles = newsapi.get_everything(
                        q=company_name,
                        language='en',
                        sort_by='publishedAt',
                        page_size=100
                    )

                    if articles['status'] == 'ok':
                        logger.info(f"Retrieved {len(articles['articles'])} articles for {stock}")

                        for article in articles['articles']:
                            sentiment_data.append({
                                'Stock': stock,
                                'Date': pd.to_datetime(article['publishedAt']).date(),
                                'Title': article['title'],
                                'Description': article['description']
                            })

                except Exception as e:
                    logger.warning(f"Could not fetch news for {stock}: {e}")

            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                return df

        except ImportError:
            logger.warning("newsapi not installed. Skipping news sentiment.")
        except Exception as e:
            logger.warning(f"Error downloading news: {e}")

        return None

    def download_all(self):
        """
        Download all data sources
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE DATA DOWNLOAD")
        logger.info("=" * 80)

        results = {}

        # 1. Stock data (required)
        results['stock_data'] = self.download_stock_data()

        # 2. Market indices (optional but recommended)
        try:
            results['market_indices'] = self.download_market_indices()
        except Exception as e:
            logger.warning(f"Could not download market indices: {e}")
            results['market_indices'] = None

        # 3. Economic data (optional)
        if self.config.get('features', {}).get('macro_economic', {}).get('enabled', False):
            results['economic_data'] = self.download_economic_data()
        else:
            results['economic_data'] = None

        # 4. News sentiment (optional)
        if self.config.get('features', {}).get('sentiment', {}).get('enabled', False):
            results['news_data'] = self.download_news_sentiment()
        else:
            results['news_data'] = None

        logger.info("=" * 80)
        logger.info("DATA DOWNLOAD COMPLETE")
        logger.info("=" * 80)

        return results


def download_market_data(stocks_list, start_date, end_date=None):
    """
    Legacy function for backward compatibility
    """
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    data = yf.download(stocks_list, start=start_date, end=end_date, auto_adjust=True)
    data = data.stack(level='Ticker', future_stack=True).reset_index()

    data.rename(
        columns={
            'Date': 'Date',
            'Ticker': 'Stock',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
        },
        inplace=True,
    )

    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Stock']]
    data = data.sort_values(by=['Stock', 'Date']).reset_index(drop=True)

    stock_list = data['Stock'].unique()
    logger.info(f"Downloaded data for stocks: {stock_list}")

    return data, stock_list





