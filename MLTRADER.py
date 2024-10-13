"""
EATS MLTRADER

Author: Eshan Jha

License: MIT, see LICENSE for details

"""


from startup import startup
import os
from matplotlib import pyplot as plt
import matplotlib
from dotenv import load_dotenv
import colorama
from colorama import Fore
import logging
from _MLTRADER import _MLTRADER
from alpaca_trade_api import REST
from datetime import datetime
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting    

startup()

matplotlib.use('Qt5Agg')

colorama.init(autoreset=True)

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
BASE_URL = os.getenv('BASE_URL')


trades = []
cash_history = []
date_history = []


logging.basicConfig(level=logging.INFO, filename='trading_bot.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')


ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}   


class MLTRADER(_MLTRADER):
    
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
        

        
    @staticmethod
    def handle_error(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                print('Error! Something went wrong:', e)
        return wrapper
    
    @handle_error
    def export_trade_history_to_csv(self, filename="trade_history.csv"):
        """
        Export the trade history to a CSV file.

            - This function writes the trade history (including trade type, price, and timestamp) to a CSV file.
            - The CSV file can be used for further analysis, record-keeping, or sharing results with others.

        Args:
            filename (str): The name of the CSV file to save the trade history (default is 'trade_history.csv').

        """
        import csv

        try:
            with open(filename, mode='w', newline='EATS MLTRADER TRADE HISTORY') as file:
                writer = csv.writer(file)
                writer.writerow(["Trade Type", "Price", "Timestamp"])
                for trade, timestamp in zip(trades, date_history):
                    writer.writerow([trade[0], trade[1], timestamp])
            self.log(f"Trade history successfully exported to {filename}")
        except Exception as e:
            self.log(f"Error exporting trade history to CSV: {str(e)}", level='ERROR')

    @handle_error
    def trader_alert(self, message, level = 'INFO'):
        """
        Generate an alert based on the provided message.

        - This method sends an alert to the trader. For this example, it simply prints the alert message.
        - You can extend this method to send notifications via email, SMS, or other channels.

        Args:
            message (str): The message to be sent as an alert.
        """
        if level == 'INFO':
            print(Fore.GREEN + f"{message}") 
        elif level == 'WARNING':
            print(Fore.YELLOW + f"WARNING: {message}")
        elif level == 'ALERT':
            print(Fore.BLACK + f"ALERT: {message}")
        elif level == 'ERROR':
            print(Fore.RED + f"ERROR: {message}")
        elif not level == 'ALERT' or not level == 'WARNING' or not level == 'INFO' or not level == 'ERROR':
            raise ValueError('Level must be ALERT, WARNING, ERROR or INFO')
        
        # You can integrate other alert mechanisms here (e.g., email, SMS).
    
    @handle_error
    def plot_performance(self):
        """
        Generate and display a graphical plot of the trading performance over time.

            - Creates a line plot showing the change in cash balance over time.
            - Uses `matplotlib` to create a figure with a specified size (10x6 inches).
            - Plots `cash_history` against `date_history`, where `cash_history` represents the cash balance at various points in time,
            and `date_history` represents the corresponding dates.
            - Labels the x-axis as 'Date' and the y-axis as 'Cash Balance'.
            - Adds a title 'Trading Performance' to the plot and a legend to identify the data series.
            - Enables grid lines for better readability.
            - Saves the plot as an image file named 'trading_performance.png' for offline viewing.
            - Displays the plot in a window for interactive review.

        This function is useful for visualizing the overall performance of the trading strategy and analyzing the changes in cash balance over the trading period.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(date_history, cash_history, label='Cash Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cash Balance')
        plt.title('Trading Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig('trading_performance.png')
        plt.show()
        

    @handle_error
    def get_results(self):
        """
        Calculate and display the performance metrics of the trading strategy.

            - This function aggregates the results of the trading strategy by calculating performance metrics such as Sharpe Ratio,
            Maximum Drawdown, and Win Rate.
            - It calls the `calculate_performance_metrics` method to perform these calculations and log the results.
            - This method is intended to be called after the backtesting or live trading session to evaluate the overall performance.
        """
        self._calculate_performance_metrics()

    @handle_error
    def load_gui(self):
        """
        Generate and display the graphical user interface (GUI) for performance visualization.

            - This function is responsible for creating a visual representation of the trading performance over time.
            - It calls the `plot_performance` method to generate a plot of cash balance over time and save it as an image file.
            - The GUI functionality is typically used for reporting and reviewing the results of the trading strategy visually.
        """
        self.plot_performance()
        
if __name__ == '__main__':
    start_date = datetime(2020, 7, 1)
    end_date = datetime(2024, 8, 19)

    broker = Alpaca(ALPACA_CREDS)

    
    strategy = MLTRADER(name='mlstrat', broker=broker,
                        parameters={'symbol': 'SPY',
                                    "cash_at_risk": .5})

    strategy.initialize(symbol='SPY')

    strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            benchmark_asset='SPY',
            parameters={'symbol': 'SPY',
                        "cash_at_risk": .5}
    )
       
    strategy.get_results()
    strategy.load_gui()
