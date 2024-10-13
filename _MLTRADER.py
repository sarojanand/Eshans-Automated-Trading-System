from dotenv import load_dotenv
import colorama
import os
import sys
import numpy as np
import logging
from datetime import timedelta
from alpaca_trade_api import REST
from finbert_utils import estimate_sentiment
from lumibot.strategies import Strategy
import math

colorama.init(autoreset=True)

load_dotenv()
   
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
BASE_URL = os.getenv('BASE_URL')

ALPACA_CREDS = {
    'API_KEY': API_KEY,
    'API_SECRET': API_SECRET,
    'PAPER': True
}

trades = []
cash_history = []
date_history = []

logging.basicConfig(level=logging.INFO, filename='trading_bot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class _MLTRADER(Strategy):
    """
    EATS MLTRADER is a machine learning model that can be used for paper trading.
    To use MLTRADER, please read requirements.txt.
    ! DO NOT CHANGE METHODS `on_trading_iteration` 
    """

    
    def initialize(self, symbol: str = "SPY" , cash_at_risk: float = .5):
        
        """
        Initialize the trading strategy with essential parameters and settings.

            - Sets the symbol for trading (e.g., 'SSNC') and the fraction of cash to risk per trade.
            - Defines the sleep time between trading iterations (e.g., '24H' for 24 hours).
            - Initializes the `last_trade` attribute to track the type of the last trade.
            - Creates an instance of the Alpaca API client using the provided API credentials.

        Args:
            symbol (str): The trading symbol.
            cash_at_risk (float): The proportion of cash to risk on each trade (default is 0.5 or 50%).
            
        >>> MLTRADER().backtest(
        >>> YahooBacktesting,
        >>> start_date,
        >>> end_date,
        >>>     parameters={'symbol': str(ticker),
        >>>     "cash_at_risk": .5}
        >>> )
        """
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.debug_mode = False
        self.api = REST(base_url=BASE_URL, key_id=API_KEY,
                        secret_key=API_SECRET)
        
        

    def position_sizing(self):
        """
        Calculate the position size for the current trade based on available cash and the last price.

            - Retrieves the current cash balance and the last price of the trading symbol.
            - Calculates the number of shares/contracts to trade based on the proportion of cash at risk.
            - Rounds the quantity to the nearest whole number to comply with trading requirements.

        Returns:
            tuple: A tuple containing:
                - cash (float): The available cash balance.
                - last_price (float): The last recorded price of the trading symbol.
                - quantity (float): The calculated number of shares/contracts to trade.
        """
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        # Ensure that the cash and last_price are positive and avoid division by zero
        if cash <= 0 or last_price <= 0:
            self.log("Cash or last price is non-positive. Cannot calculate position size.", level='ALERT')
            return cash, last_price, 0
        # Enhanced position sizing with additional logic
        position_size = cash * self.cash_at_risk / last_price
        # Ensure position size is a positive integer
        quantity = max(1, round(position_size))
        
        # Returns the cash, last price and quatity when the position sizing function is called
        return cash, last_price, quantity

    
    def get_dates(self):
        """
        Retrieve the current date and the date three days prior for data fetching.

            - Computes the current date and the date exactly three days before it.
            - Formats these dates as strings to be used for querying historical data or news.

        Returns:
            tuple: A tuple containing:
                - today (str): The current date in 'YYYY-MM-DD' format.
                - three_days_prior (str): The date three days prior to today in 'YYYY-MM-DD' format.
        """
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        """
        Analyze recent news sentiment and return the sentiment probability and type.

            - Fetches news articles for the trading symbol from the past three days.
            - Extracts headlines from the news articles.
            - Uses a sentiment analysis model to estimate the sentiment and its probability.
            - Logs the sentiment and probability for review.

        Returns:
            tuple: A tuple containing:
                - probability (float): The probability score indicating the strength of the sentiment.
                - sentiment (str): The sentiment type (e.g., 'positive', 'negative').
        """
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol,
                                 start=three_days_prior,
                                 end=today)

        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        self.log(f"Sentiment: {sentiment}, Probability: {probability}")
        return probability, sentiment

        
    def dynamic_risk_management(self, last_price, risk_tolerance, profit_margin, cap_limit):
        """
        Calculate and return adaptive take profit and stop loss prices based on the last trade price.

        - This function incorporates a percentage-based approach with caps to limit extreme values.
        - Risk tolerance and profit margin are adjustable parameters.
        - The take profit and stop loss levels are calculated based on these parameters and capped to prevent extreme values.

        Args:
            last_price (float): The last trade price of the symbol.
            risk_tolerance (float): The percentage below the last price to set the stop loss. Default is 2%.
            profit_margin (float): The percentage above the last price to set the take profit. Default is 10%.
            cap_limit (float): The maximum percentage cap for the take profit and stop loss. Default is 30%.

        Returns:
            tuple: A tuple containing the take profit price and stop loss price.
        """
        # Calculate initial take profit and stop loss prices based on percentage
        take_profit = last_price * (1 + profit_margin)
        stop_loss = last_price * (1 - risk_tolerance)
        
        # Cap the take profit and stop loss to prevent extreme values
        max_price = last_price * (1 + cap_limit)
        min_price = last_price * (1 - cap_limit)
        
        # Ensure that the take profit and stop loss do not exceed the capped limits
        take_profit = min(take_profit, max_price)
        stop_loss = max(stop_loss, min_price)
        
        return take_profit, stop_loss

    def print_strategy_parameters(self):
        """
        Print the current strategy parameters of the trading bot.

        This method outputs key attributes related to the bot's strategy, such as the trading symbol,
        the amount of cash at risk, the sleep time between trades, details of the last trade, 
        and whether the bot is in debug mode.
        
        Raises:
            Exception: If an error occurs while accessing the parameters.
        """
        try:
            print(f"Symbol: {self.symbol}")
            print(f"Cash at Risk: {self.cash_at_risk}")
            print(f"Sleeptime: {self.sleeptime}")
            print(f"Last Trade: {self.last_trade}")
            print(f"Debug Mode: {self.debug_mode}")
        except Exception as e:
            print(f"Error printing strategy parameters: {str(e)}")
            sys.exit(1)

    def log_cash_and_position_details(self):
        """
        Log the current cash balance, the last price of the trading symbol, and the position size.

        This method calculates and logs the financial status, including the available cash,
        the most recent price of the trading symbol, and the position size based on the current strategy.

        Raises:
            Exception: If an error occurs while calculating or logging the details.
        """
        try:
            cash, last_price, quantity = self.position_sizing()
            self.log(f"Current Cash: {cash}")
            self.log(f"Last Price of {self.symbol}: {last_price}")
            self.log(f"Position Size: {quantity}")
        except Exception as e:
            print(f"Error logging cash and position details: {str(e)}")
            sys.exit(1)

    def display_sentiment_analysis(self):
        """
        Perform sentiment analysis on the trading symbol and display the results.

        This method fetches the sentiment and probability values from the sentiment analysis tool,
        then prints these values to help understand the market sentiment for the current trading symbol.

        Raises:
            Exception: If an error occurs while fetching or displaying the sentiment analysis.
        """
        try:
            probability, sentiment = self.get_sentiment()
            print(f"Sentiment: {sentiment}")
            print(f"Probability: {probability}")
        except Exception as e:
            print(f"Error displaying sentiment analysis: {str(e)}")
            sys.exit(1)

    def debug_mode(self, mode: bool):
        """
        Enable or disable the debug mode for the trading bot.

        This method toggles the debug mode based on the input boolean value. 
        When enabled, the bot may provide more detailed logging or output for debugging purposes.

        Args:
            mode (bool): A boolean value to enable or disable debug mode.

        Raises:
            Exception: If the input is not a boolean value or an error occurs while setting the mode.
        """
        try:
            self.debug_mode = mode
            print(f"Debug Mode {'Enabled' if self.debug_mode else 'Disabled'}")
        except Exception as e:
            print(f"Invalid input. Please provide a boolean value. {str(e)}")
            sys.exit(1)

    
    def print_trade_history(self):
        """
        Print the history of trades made by the trading bot.

        This method iterates through the list of trades and prints each trade's details,
        including the trade action and the price at which it was executed.

        Raises:
            Exception: If an error occurs while accessing or printing the trade history.
        """
        try:
            if trades:
                for trade in trades:
                    print(f"Trade: {trade[0]}, Price: {trade[1]}")
            else:
                print("No trades have been made yet.")
        except Exception as e:
            print(f"Error displaying trade history: {str(e)}")
            sys.exit(1)

    def check_api_connection(self):
        """
        Check the connection status with the trading API.

        This method attempts to connect to the trading API (e.g., Alpaca) and logs the status
        of the connection along with the account status.

        Raises:
            Exception: If the connection to the API fails.
        """
        try:
            account = self.api.get_account()
            self.log("Connected to Alpaca API")
            self.log(f"Account status: {account.status}")
        except Exception as e:
            self.log(f"Failed to connect to Alpaca API: {str(e)}", level='ERROR')

    def check_trading_status(self):
        """
        Check whether the trading market is currently open.

        This method retrieves the current trading market status from the API
        and logs whether the market is open or closed.

        Raises:
            Exception: If an error occurs while fetching the market status.
        """
        try:
            self.api.get_clock()
            self.log("Trading is currently open")
        except Exception as e:
            self.log(f"Trading is currently closed: {str(e)}", level='ERROR')
    
    def on_trading_iteration(self):
        """
        ! This method is the center of the trading logic. Do not delete this method.
        Execute the main trading logic for each iteration.

            - Retrieve the current cash balance, last price of the symbol, and calculate the position size.
            - Obtain the sentiment and probability from sentiment analysis.
            - Based on sentiment and probability, decide whether to buy or sell:
                - If sentiment is positive and probability > 0.999:
                    - If the last trade was a 'sell', close the position by selling all.
                    - Calculate take profit and stop loss prices using dynamic risk management.
                    - Create and submit a 'buy' order with bracket conditions (take profit and stop loss).
                    - Record the trade as 'buy' and append it to the trades list.
                    - Send an alert about the 'buy' action.
                - If sentiment is negative and probability > 0.999:
                    - If the last trade was a 'buy', close the position by selling all.
                    - Calculate take profit and stop loss prices using dynamic risk management.
                    - Create and submit a 'sell' order with bracket conditions (take profit and stop loss).
                    - Record the trade as 'sell' and append it to the trades list.
                    - Send an alert about the 'sell' action.
            - Append the current cash balance and date to their respective histories for performance tracking.
        """
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        # For dynamic_risk_management functions
        risk_tolerance=0.02
        profit_margin=0.10
        cap_limit=0.30
        
        # If the cash balance is higher than 0, we can excute trades
        if cash > 0:
            if sentiment == 'positive' and probability > .999:
                if self.last_trade == "sell":
                    self.sell_all()
                    self.trader_alert("Position closed due to positive sentiment. Selling all holdings.", 'ALERT')

                take_profit, stop_loss = self.dynamic_risk_management(last_price, risk_tolerance, profit_margin, cap_limit)
                order = self.create_order(
                    self.symbol,
                    quantity,
                    'buy',
                    type='bracket',
                    take_profit_price=take_profit,
                    stop_loss_price=stop_loss,
                    position_filled=False
                )

                self.submit_order(order)
                self.last_trade = 'buy'
                trades.append(('buy', last_price))  
 

            elif sentiment == 'negative' and probability > .999:
                if self.last_trade == "buy":
                    self.sell_all()
                    self.trader_alert("Position closed due to negative sentiment. Selling all holdings.")

                take_profit, stop_loss = self.dynamic_risk_management(last_price, risk_tolerance, profit_margin, cap_limit)
                order = self.create_order(
                    self.symbol,
                    quantity,
                    'sell',
                    type='bracket',
                    take_profit_price=take_profit,
                    stop_loss_price=stop_loss,
                    position_filled=False
                )
                self.submit_order(order)
                self.last_trade = 'sell'
                trades.append(('sell', last_price))
            
                
        # If there is no cash or we are in debt, we will sell of the the trades and stop the system
        elif cash <= 0:
            self.trader_alert("Insufficient cash to execute any trades. Closing all trades...", "ALERT")
            self.log(self.cash)
            self.sell_all()
            if self.cash <= 0:
                self.trader_alert("In Debt. Closing all Trades Immediately!", "ALERT")
            sys.exit()
            
        # Adding to the history list to get the max drawdown and cash history
        cash_history.append(self.get_cash())
        date_history.append(self.get_datetime())



    
    def log(self, message, level='INFO'):
        if level == 'INFO':
            logging.info(message)
        elif level == 'ERROR':
            logging.error(message)
        print(message)

    
    def _calculate_performance_metrics(self):
        """

        Calculate and log performance metrics for the trading strategy.

            - Calculates the returns from executed trades.
            - Computes the Sharpe Ratio, which is a measure of risk-adjusted return.
            - Calculates the Maximum Drawdown, indicating the largest peak-to-trough decline.
            - Determines the Win Rate, the proportion of profitable trades.
            - Logs the calculated metrics or indicates if no trades were executed.

        """

        if len(trades) > 0:
            returns = np.array([trade[1] for trade in trades])
            if len(returns) > 1:  # Ensure there is enough data for calculations
                returns = np.diff(returns) / returns[:-1]
                # Assuming daily returns
                sharpe_ratio = np.mean(returns) / \
                    np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = np.nan

            max_drawdown = self._calculate_max_drawdown()
            win_rate = len(
                [trade for trade in trades if trade[0] == 'buy']) / len(trades)

            self.log(f"Sharpe Ratio: {sharpe_ratio}")
            self.log(f"Max Drawdown: {max_drawdown}")
            self.log(f"Win Rate: {win_rate}")
        else:
            self.log("No trades executed, unable to calculate performance metrics.")
            

     
    def _calculate_max_drawdown(self):
        """

        Calculate and return the maximum drawdown.

            - Maximum drawdown is the largest peak-to-trough decline in the cash balance.
            - Uses the cash history to compute the cumulative maximum cash balance at each point.
            - Calculates drawdown as the percentage decline from the cumulative maximum.
            - Returns the maximum drawdown observed during the period.

        """
        if len(cash_history) > 1:
            cum_max = np.maximum.accumulate(cash_history)
            drawdown = (cash_history - cum_max) / cum_max
            max_drawdown = np.min(drawdown)
            return max_drawdown
        return 0