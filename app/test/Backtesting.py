import numpy as np
import pandas as pd
from configuration.Logger_config import setup_logger, logger


class Backtester:
    """
    Comprehensive backtesting system for trading strategies
    Evaluates predictions as actual trading decisions
    """
    
    def __init__(self, config):
        self.config = config
        self.backtest_config = config.get('backtesting', {})
        self.initial_capital = self.backtest_config.get('initial_capital', 100000)
        self.commission = self.backtest_config.get('commission', 0.001)
        self.slippage = self.backtest_config.get('slippage', 0.0005)
        
        self.portfolio_history = []
        self.trade_history = []
        
    def run_backtest(self, data, predictions, strategy='threshold_based', **strategy_params):
        """
        Run backtest with given predictions and strategy
        
        Args:
            data: DataFrame with Date, Stock, Close, and actual returns
            predictions: Array or Series of predicted returns/direction
            strategy: Trading strategy to use
            **strategy_params: Additional strategy parameters
            
        Returns:
            results: Dictionary with performance metrics
        """
        logger.info(f"Running backtest with strategy: {strategy}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Prepare data
        df = data.copy()
        df['Prediction'] = predictions
        
        # Remove NaN predictions
        df = df.dropna(subset=['Prediction'])
        
        if strategy == 'threshold_based':
            results = self._threshold_strategy(df, **strategy_params)
        elif strategy == 'top_k':
            results = self._top_k_strategy(df, **strategy_params)
        elif strategy == 'portfolio_optimization':
            results = self._portfolio_optimization_strategy(df, **strategy_params)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(results)
        
        return metrics, results
    
    def _threshold_strategy(self, df, threshold=0.01, hold_days=1):
        """
        Buy when prediction > threshold, sell after hold_days
        
        Args:
            threshold: Minimum predicted return to trigger buy
            hold_days: Days to hold position
        """
        logger.info(f"Threshold strategy - threshold: {threshold:.4f}, hold: {hold_days} days")
        
        cash = self.initial_capital
        positions = {}  # {stock: {'shares': X, 'entry_price': Y, 'entry_date': Z}}
        portfolio_values = []
        
        # Sort by date
        df = df.sort_values(['Date', 'Stock']).reset_index(drop=True)
        dates = df['Date'].unique()
        
        for date in dates:
            day_data = df[df['Date'] == date]
            
            # Check existing positions for exit
            for stock in list(positions.keys()):
                position = positions[stock]
                days_held = (date - position['entry_date']).days
                
                if days_held >= hold_days:
                    # Exit position
                    stock_data = day_data[day_data['Stock'] == stock]
                    if not stock_data.empty:
                        exit_price = stock_data.iloc[0]['Close']
                        
                        # Apply slippage and commission
                        exit_price = exit_price * (1 - self.slippage)
                        proceeds = position['shares'] * exit_price
                        commission_cost = proceeds * self.commission
                        
                        cash += proceeds - commission_cost
                        
                        # Record trade
                        self.trade_history.append({
                            'stock': stock,
                            'entry_date': position['entry_date'],
                            'exit_date': date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'return': (exit_price - position['entry_price']) / position['entry_price'],
                            'pnl': proceeds - commission_cost - (position['shares'] * position['entry_price'])
                        })
                        
                        del positions[stock]
            
            # Check for new entries
            # Allocate equal capital to each signal
            signals = day_data[day_data['Prediction'] > threshold]
            
            if len(signals) > 0 and cash > 0:
                capital_per_stock = cash / (len(signals) + len(positions))
                
                for _, signal in signals.iterrows():
                    stock = signal['Stock']
                    
                    # Skip if already holding
                    if stock in positions:
                        continue
                    
                    entry_price = signal['Close'] * (1 + self.slippage)
                    shares = int(capital_per_stock / entry_price)
                    
                    if shares > 0:
                        cost = shares * entry_price
                        commission_cost = cost * self.commission
                        total_cost = cost + commission_cost
                        
                        if total_cost <= cash:
                            cash -= total_cost
                            
                            positions[stock] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': date
                            }
            
            # Calculate portfolio value
            positions_value = 0
            for stock, position in positions.items():
                stock_data = day_data[day_data['Stock'] == stock]
                if not stock_data.empty:
                    current_price = stock_data.iloc[0]['Close']
                    positions_value += position['shares'] * current_price
            
            total_value = cash + positions_value
            
            portfolio_values.append({
                'Date': date,
                'Cash': cash,
                'Positions_Value': positions_value,
                'Total_Value': total_value,
                'Return': (total_value - self.initial_capital) / self.initial_capital
            })
        
        results_df = pd.DataFrame(portfolio_values)
        self.portfolio_history = results_df
        
        return results_df
    
    def _top_k_strategy(self, df, k=5, rebalance_days=5):
        """
        Hold top K stocks by predicted return, rebalance every N days
        """
        logger.info(f"Top-K strategy - K: {k}, rebalance: {rebalance_days} days")
        
        cash = self.initial_capital
        positions = {}
        portfolio_values = []
        
        df = df.sort_values(['Date', 'Stock']).reset_index(drop=True)
        dates = df['Date'].unique()
        
        last_rebalance = None
        
        for date in dates:
            day_data = df[df['Date'] == date]
            
            # Check if rebalance needed
            should_rebalance = False
            if last_rebalance is None:
                should_rebalance = True
            elif (date - last_rebalance).days >= rebalance_days:
                should_rebalance = True
            
            if should_rebalance:
                # Close all positions
                for stock, position in positions.items():
                    stock_data = day_data[day_data['Stock'] == stock]
                    if not stock_data.empty:
                        exit_price = stock_data.iloc[0]['Close'] * (1 - self.slippage)
                        proceeds = position['shares'] * exit_price
                        commission_cost = proceeds * self.commission
                        cash += proceeds - commission_cost
                
                positions = {}
                
                # Select top K stocks
                top_stocks = day_data.nlargest(k, 'Prediction')
                
                if len(top_stocks) > 0:
                    capital_per_stock = (cash * 0.95) / len(top_stocks)  # Keep 5% cash
                    
                    for _, stock_row in top_stocks.iterrows():
                        stock = stock_row['Stock']
                        entry_price = stock_row['Close'] * (1 + self.slippage)
                        shares = int(capital_per_stock / entry_price)
                        
                        if shares > 0:
                            cost = shares * entry_price
                            commission_cost = cost * self.commission
                            total_cost = cost + commission_cost
                            
                            cash -= total_cost
                            positions[stock] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': date
                            }
                
                last_rebalance = date
            
            # Calculate portfolio value
            positions_value = 0
            for stock, position in positions.items():
                stock_data = day_data[day_data['Stock'] == stock]
                if not stock_data.empty:
                    current_price = stock_data.iloc[0]['Close']
                    positions_value += position['shares'] * current_price
            
            total_value = cash + positions_value
            
            portfolio_values.append({
                'Date': date,
                'Cash': cash,
                'Positions_Value': positions_value,
                'Total_Value': total_value,
                'Return': (total_value - self.initial_capital) / self.initial_capital
            })
        
        results_df = pd.DataFrame(portfolio_values)
        self.portfolio_history = results_df
        
        return results_df
    
    def _portfolio_optimization_strategy(self, df, **params):
        """
        Placeholder for portfolio optimization (Kelly criterion, mean-variance, etc.)
        """
        logger.warning("Portfolio optimization strategy not fully implemented yet")
        return self._top_k_strategy(df, k=5)
    
    def _calculate_metrics(self, results_df):
        """
        Calculate comprehensive performance metrics
        """
        if results_df.empty:
            logger.warning("Empty results, cannot calculate metrics")
            return {}
        
        # Returns
        returns = results_df['Return'].values
        daily_returns = results_df['Total_Value'].pct_change().dropna()
        
        # Total return
        total_return = returns[-1] if len(returns) > 0 else 0
        
        # Annual return (assuming 252 trading days)
        days = len(returns)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate from trades
        if len(self.trade_history) > 0:
            trades_df = pd.DataFrame(self.trade_history)
            winning_trades = (trades_df['return'] > 0).sum()
            total_trades = len(trades_df)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            total_trades = 0
            avg_win = 0
            avg_loss = 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return / downside_std) if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = (annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        metrics = {
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Total_Trades': total_trades,
            'Avg_Win': avg_win,
            'Avg_Loss': avg_loss,
            'Final_Value': results_df['Total_Value'].iloc[-1] if len(results_df) > 0 else 0
        }
        
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Initial Capital:    ${self.initial_capital:>15,.2f}")
        logger.info(f"Final Value:        ${metrics['Final_Value']:>15,.2f}")
        logger.info(f"Total Return:       {metrics['Total_Return']:>15.2%}")
        logger.info(f"Annual Return:      {metrics['Annual_Return']:>15.2%}")
        logger.info(f"Volatility:         {metrics['Volatility']:>15.2%}")
        logger.info(f"Sharpe Ratio:       {metrics['Sharpe_Ratio']:>15.2f}")
        logger.info(f"Sortino Ratio:      {metrics['Sortino_Ratio']:>15.2f}")
        logger.info(f"Max Drawdown:       {metrics['Max_Drawdown']:>15.2%}")
        logger.info(f"Win Rate:           {metrics['Win_Rate']:>15.2%}")
        logger.info(f"Profit Factor:      {metrics['Profit_Factor']:>15.2f}")
        logger.info(f"Total Trades:       {metrics['Total_Trades']:>15.0f}")
        logger.info("="*60)
        
        return metrics
    
    def plot_results(self):
        """
        Plot backtest results
        """
        if self.portfolio_history is None or len(self.portfolio_history) == 0:
            logger.warning("No backtest results to plot")
            return None
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            df = self.portfolio_history
            
            # Portfolio value
            axes[0].plot(df['Date'], df['Total_Value'], label='Portfolio Value', linewidth=2)
            axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].set_title('Portfolio Value Over Time')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Returns
            axes[1].plot(df['Date'], df['Return'] * 100, label='Return %', linewidth=2, color='green')
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_ylabel('Return (%)')
            axes[1].set_title('Cumulative Returns')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Drawdown
            cumulative = df['Total_Value'] / df['Total_Value'].iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            axes[2].fill_between(df['Date'], drawdown * 100, 0, color='red', alpha=0.3)
            axes[2].set_ylabel('Drawdown (%)')
            axes[2].set_xlabel('Date')
            axes[2].set_title('Drawdown')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None
