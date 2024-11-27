from flask import Flask, request, render_template, redirect, url_for
import sys
import os
from datetime import datetime
from lumibot.brokers import Alpaca
from dotenv import load_dotenv
from lumibot.backtesting import YahooDataBacktesting    
from MLTRADER import MLTRADER
from yahooquery import Ticker
from multiprocessing import Process

dotenv_envirorment = load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
BASE_URL = os.getenv('BASE_URL')

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}   

start_date = datetime(2007, 3, 1)
end_date = datetime(2024, 9, 15)

broker = Alpaca(ALPACA_CREDS)

def run_backtest(ticker):
    strategy = MLTRADER(name='mlstrat', broker=broker, budget= 1,
                        parameters={'symbol': ticker,
                                    "cash_at_risk": .5})

    strategy.initialize(symbol=ticker)

    strategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset=ticker,
        parameters={'symbol': ticker,
                    "cash_at_risk": .5}
    )


    print('Trading completed successfully! Your results are loading...')
    strategy.get_results()
    strategy.load_gui()
    strategy.log_cash_and_position_details()
    
def validate_ticker(summary):
    if not isinstance(summary, dict):
        try:
            if (summary.startswith('No fundamentals data found ')) == True:
                print('This ticker is not available in the yahoo database')
                sys.exit(1)
            elif (summary.startswith('Quote not found for ticker symbol')) == True:
                print('This quote is not found in the yahoo database')
                sys.exit(1)
        except:
            raise KeyError('String can not be used like this')
    else:
        print(summary)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('index'))
    metrics = {
        'total_trades':  0,
        'average_return': 0,
        'current_balance': 100000,
        'symbol': 'Not set'
    }
    return render_template('dashboard.html', metrics=metrics)

@app.route('/ticker', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('ticker')
        ticker = Ticker(f"{symbol}")
        if ticker:
            # Start the backtest in a new process
            summary = ticker.summary_detail[symbol]
            validate_ticker(summary)
            p = Process(target=run_backtest, args=(symbol,))
            p.start()
            return render_template('result.html', ticker=symbol)
    return render_template('ticker.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
