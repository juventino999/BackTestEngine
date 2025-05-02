"""
Inputs:
    - Models to predict returns of n financial securities
        - n datasets to use as inputs to the models
        - dataframe (n columns) of real prices during the backtest period used to 
            backtest the strategy (prices at the end of each period should be named '[TICKER]_close')
            - all indices should be of type DateTimeIndex
    - buy fee/sell fee are in basis points (as are returns in the provided example)
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from math import e
from statsmodels.regression.linear_model import RegressionResultsWrapper
from tqdm import trange

from Security import Security
from BackTestHelper import BackTestHelper 

import matplotlib.pyplot as plt

        
class BackTestEngine():
    def __init__(self, models: dict, securities: list, prices: pd.DataFrame, initial_investment = 1):
        """"
        models: a dictionary with models (key) and a tuple with two DataFrames of inputs/outputs (value) and a string (name of the security)
            model: the model model will generate predictions for returns in the next time step (will call .predict() on the model). 
            tuple: 
                df: DataFrame of features for the model to use as inputs 
                labels: a Series of labels (actual returns) (actually, should be close prices for the minute) -> these are unused for now
                coin: name of the security
            Each model in the dictionary will be used for predictions, and then results will be aggregated
        securities: list of Coin objects that will comprise the total portfolio
        prices: Dataframe of actual price data. Should have the security ticker and '_close' for column names (ex. 'LTC_close') 
                and have the same index as predictions (all periods included, don't skip rows)
        trading_rule (WIP): trading rule to change portfolio weights at each time stemp. If None, use the default trading rule. 
        For the moment, this is long only. 
        """
        self.models = models # dictionary of models and data to go with the models
        self.securities = securities # list of Security objects

        self.portfolio_value = initial_investment
        self.tradeable_value = initial_investment # amount that can be traded in the given time step
        self.cash = initial_investment # amount of uninvested cash

        self.prices = prices



    def make_prediction(self, start = '2024-07-01', end = '2024-10-01'):
        """
        Use self.models to predict returns for the whole backtest period. 
        Output: a DataFrame of predictions.
        The timestamp indicates the predicted return for the next minute (timestamp 2024-07-01 00:02:00 is the predicted return for 2024-07-01 00:03:00)
        """
        # for each model in the dictionary, use the df to generate a tensor of predictions (vector/series for sm models, matrix/dataframe for CNN). 
        # If needed, aggregate results into a single df: want a table with predictions for each security for each time period
        if len(self.models) == 1: # if there is 1 model in the dictionary
            for model, data in self.models.items():
                if type(model) == RegressionResultsWrapper: # if the model is an sm model
                    x = sm.add_constant(data[0]) # add constant (don't do this as a preprocessing step before handing the data to this class)
                else:
                    x = data[0]
                self.predictions = model.predict(x) # predictions will be a Series (sm) or DataFrame (nn)
                self.labels = data[1] # get Series (or DataFrame) of actual returns

        elif len(self.models) > 1: # if there is more than one model in the dictionary
            # create two empty DataFrames (for predictions and labels) with dt indices from 2024 to end of 2025
            start_date = start # start of validation period
            end_date = end # end of validation period
            complete_index = pd.date_range(start=start_date, end=end_date, freq='min', tz='utc', inclusive='left') # date range: don't include october
            self.predictions = pd.DataFrame(index=complete_index)
            self.labels = pd.DataFrame(index=complete_index)

            for model, data in self.models.items():
                if type(model) == RegressionResultsWrapper: # if the model is an sm model
                    x = sm.add_constant(data[0]) # add constant 
                else:
                    x = data[0]
                predictions = model.predict(x) # series of predictions
                predictions.name = data[2]
                labels = data[1] # actual returns are the second item in the tuple
                labels.name = data[2]
                # merge these with the preds, make sure the series both have correct names (should be the name of the coin)
                self.predictions = self.predictions.merge(predictions, left_index= True, right_index=True, how = 'left')
                self.labels = self.labels.merge(labels, left_index= True, right_index=True, how = 'left')
                # need the closing prices too, not just the returns


    def trading_rule(self, x = None, strategy = 'softmax', buy_fee = 0, sell_fee = 0):
        """
        Define how to trade each security given the amount of each security in the portfolio, the amount of cash unallocated,
        and the predicted return of each security in the next time step.
        """
        if strategy == 'softmax':
            return np.exp(1*x)/sum(np.exp(1*x)) # for each row of returns, return the softmax as weights
        if strategy == 'fee aware':
            """
            Each minute, group currencies into price groups based on predicted price changes:
	            - Sell:
		            ○ If a currency is predicted to decrease by more than the cost of selling (sell fee), sell the coin
	            - Hold: 
		            ○ If a currency is predicted to decrease by less than the sell fee, hold the coin (usually)
			            § Here a model that predicted more steps into the future would be nice. If it were to decrease in the next few steps (to a total decrease of more than the sell fee), would like to sell
		            ○ If a currency is predicted to increase by less than the buy fee, hold (usually)
	            - Buy:
		            ○ If a currency is predicted to increase by more than the buy fee, buy 
		            ○ If a currency is predicted to increase by more than the sell fee and the buy fee, sell holds (first the ones set to decrease, then the ones set to increase) and buy. 
            The input is a row of a dataframe. 
            """
            helper = BackTestHelper(self.securities, buy_fee, sell_fee) # create a helper class
            self.weights_unfilled = helper.all_weights(self)  # call the helper function
            # this should be refactored to match the style of the softmax version below
    def get_weights(self, x = None, buy_fee = 0, sell_fee = 0, trading_rule = 'softmax'):
        """
        Make predictions, then Iterate through df: apply the trading rule and record new portfolio and returns at each step.
        If using a model for each security, call make_prediction() for each model
        Predictions are a vector (Series) with timestamps. More efficient to calculate them all in one go, so grab the appropriate prediction. 
        """
        # don't actually need to iterate
        #self.weights = self.predictions.dropna(how = 'all', axis = 0).apply(lambda x: self.trading_rule(x.fillna(-99999)), axis = 1) # drop rows that are all na, then apply the trading rule to each row. 
        # if NaN prediction, don't want any weight in the coin (for the sake of calculating returns), so fill NaNs with v negative number

        # what to do if there are NaNs for some but not all of the coins?
        # what to do if all of the predictions are NaN? -> use last minute predictions 
            # doesn't actually matter: can just drop it. Don't have labels (which means I don't have prices), can do whatever.
                # gonna just drop all NaN rows for now
        if trading_rule == 'softmax':
            self.weights_unfilled = self.predictions.apply(lambda x: self.trading_rule(x.fillna(-99999)), axis = 1) # if there are no predictions (is NaN), set weight = 0 by filling with a large negative value
        if trading_rule == 'fee aware':
            self.trading_rule(strategy='fee aware', buy_fee = buy_fee, sell_fee = sell_fee) # assigns to weights_unfilled
        self.weights = self.weights_unfilled.ffill() # if all weights are NaN, don't trade. need to add this line because if all weights in a row are NaN, softmax won't work and will return NaN for the whole row

        # don't drop NaNs: iterating over all the minutes no matter if there are preds or not. 

    def get_new_portfolio_value(self, new_prices, sell_fee: float = 0, verbose = False):
        """
        Update the portfolio value to reflect price changes in new time step
        new_prices: row of the prices dataframe for the current minute
            - columns must match a value in coin.name, should be the ticker of the coin
        """
        self.portfolio_value = 0 # zero out portfolio value
        self.tradeable_value = 0

        for coin in self.securities: # each coin is a Coin object
            if verbose == True:
                print(f'Amount of {coin.name} in the portfolio at the beginning of the minute: {coin.shares}')
                print(f'{coin.name} value at the beginning of the minute: {coin.value}')

            new_price = new_prices[f'{coin.name}_close'] # get the new price for the coin
            if verbose == True:
                print(f'{coin.name} new price: {new_price}')

            if (not np.isnan(new_price)): # if the price for this minute isn't NaN
                coin.tradeable = True # you can trade this coin
                #print(f'Coin shares: {coin.shares}')
                #print(f'Coin price: {new_price}')
                coin.value = coin.shares*new_price # the new value of the holding of the coin is: the amount owned * new price
                #print(f'Coin value: {coin.value}')
                self.tradeable_value += coin.value*(1-(sell_fee/10000)) # can only allocate this much of the coin to trade for 
                #print(self.tradeable_value)
                # other coins (because lose some value when you sell)
            else:
                coin.tradeable = False
            self.portfolio_value += coin.value # add the holding for each coin to the portfolio value 
            # coin value (unchanged from the last minute if there is no new price available)
            #print(f'{coin.name} value: {coin.value}')
            #print(f'{coin.name} tradeable: {coin.tradeable}')
            if verbose == True:
                print(f'New {coin.name} value: {coin.value}')
            
        self.portfolio_value += self.cash # add cash to the portfolio value
        self.tradeable_value += self.cash # add cash to the amount that can be used for trading

        if self.tradeable_value < 0:
            self.tradeable_value = 0
        if verbose == True:
            print(f'Portfolio value after price changes: {self.portfolio_value}')
            print(f'Tradeable value after price changes: {self.tradeable_value}')
    
    def buy_or_sell(self, coin: Coin, current_price: float, target_weight: float, sell_fee: float = 0, buy_fee: float = 0):
        """
        Given a coin, a current price, and a current weight, rebalance the portfolio to achieve the current weight
        coin: Coin object to modify
        current price: price for the current minute
            - current setup: iterate thru rows of price data. pass each row to get_new_portfolio_value[], then index into each one and pass it thru this function
                - could also set this up differently but I think it's the same number of loops
        sell_fee: fee to subtract when selling coin

        probably should refactor this to have the same format as get_new_portfolio_value() to avoid confusion during implementation (need to iterate outside for this function, while it's built in for the other function)
        """
        if coin.tradeable: # if you can trade the coin:
            #print(f'{coin.name} desired weight: {target_weight}')
            #print(f'Tradeable value: {self.tradeable_value}')
            desired_shares = target_weight*self.tradeable_value / current_price # the desired amount of shares is the target weight * portfolio value 
            #print(f'Desired shares: {desired_shares}')
            # (switching to tradeable value to account for NaN prices, won't always be able to sell one [if price is NaN] to buy another)
            diff_shares = desired_shares - coin.shares # the amount of shares to buy is the desired shares - current shares
            #print(f'Diff shares: {diff_shares}')

            if diff_shares < 0: # if the amount of shares you want to buy is negative (you want to sell)
                coin.shares += diff_shares
                self.cash += -diff_shares*current_price*(1-(sell_fee/10000))


            elif diff_shares > 0: # amount of shares you want to buy is positive (you want to buy)
                diff_shares = diff_shares / (1+(buy_fee/10000)) # if you want to buy shares, you can only buy as many as this: you have to account 
                # for the purchasing power you lose with the buyer fee 

                coin.shares += diff_shares
                self.cash -= diff_shares*current_price*(1+(buy_fee/10000)) # pay the current price + the buy fee for each share
                # multiply that times the number of shares you buy, then subtract that amount from cash

    def backtest(self, verbose = False, buy_fee: float = 0, sell_fee: float = 0): # implement methods to run the backtest
        values = {}
        for i in range(len(self.prices)):
            if verbose == True:
                print(f'Minute: {i}')
                print(f'Portfolio value before prices changes: {self.portfolio_value}')
                print(f'Cash component of portfolio: {self.cash}')
            self.get_new_portfolio_value(self.prices.iloc[i], sell_fee=sell_fee)
            if verbose == True:
                print(f'Portfolio value after prices changes: {self.portfolio_value}')
            for coin in self.securities:
                self.buy_or_sell(
                                coin = coin, 
                                current_price= self.prices[f'{coin.name}_close'].iloc[i], 
                                target_weight= self.weights[coin.name].iloc[i],
                                sell_fee=sell_fee,
                                buy_fee=buy_fee
                                )
            #if verbose == True:
            #    print(f'Portfolio value (end of minute): {self.portfolio_value}') # should be the same as the previous print
            self.get_new_portfolio_value(self.prices.iloc[i], sell_fee=sell_fee) # get new portfolio value AGAIN after buying/selling - should
            # be the same value, but need to reset individual coin values while there are still valid prices (check out markdown comment way below)
            values[self.prices.index[i]] = self.portfolio_value # make a dictionary of portfolio values over time (at the end of each minute)
        return values
    
if __name__ == '__main__':
    # load models and group them in dictionary
    model = sm.load('./data/LTC.pickle')
    df_ltc = pd.read_csv('./data/LTC_backtest_input.csv', index_col=0)
    df_ltc.index = pd.to_datetime(df_ltc.index)

    model_eth = sm.load('./data/ETH.pickle')
    df_eth = pd.read_csv('./data/ETH_backtest_input.csv', index_col=0)
    df_eth.index = pd.to_datetime(df_eth.index)

    model_btc = sm.load('./data/XBT.pickle')
    df_btc = pd.read_csv('./data/XBT_backtest_input.csv', index_col=0)
    df_btc.index = pd.to_datetime(df_btc.index)

    model_link = sm.load('./data/LINK.pickle')
    df_link = pd.read_csv('./data/LINK_backtest_input.csv', index_col=0)
    df_link.index = pd.to_datetime(df_link.index)

    models = {
            model: (df_ltc.drop('LTC_label', axis = 1), df_ltc['LTC_label'], 'LTC'),
            model_eth: (df_eth.drop('ETH_label', axis = 1), df_eth['ETH_label'], 'ETH'),
            model_btc: (df_btc.drop('XBT_label', axis = 1), df_btc['XBT_label'], 'XBT'),
            model_link: (df_link.drop('LINK_label', axis = 1), df_link['LINK_label'], 'LINK'),
           } 
    
    ## load price data 
    prices = pd.read_csv('./data/backtest_prices.csv')
    prices.index = pd.DatetimeIndex(prices['timestamp'])
    prices.drop('timestamp', axis = 1, inplace=True)

    ## data for testing only: synthetic data with no price changes
    # prices = pd.read_csv('backtest_no_price_changes.csv')
    # prices.index = pd.DatetimeIndex(prices['Unnamed: 0'])
    # prices.drop('Unnamed: 0', axis = 1, inplace=True)
    # prices.mask(np.random.random(prices.shape) < .3) # randomly NaN 30% of cells


    start_date = '2024-07-01' # start of validation period
    end_date = '2024-10-01' # end of validation period
    complete_index = pd.date_range(start=start_date, end=end_date, freq='min', tz='utc', inclusive='left') # date range: don't include october
    prices_reindexed = prices['2024-07':'2024-09'].reindex(complete_index)

    # define securities to backtest
    ltc = Coin('LTC')
    eth = Coin('ETH')
    xbt = Coin('XBT')
    link = Coin('LINK')
    coins = [ltc, eth, xbt, link]

    # initialize and run backtest

    bak = BackTestEngine(models, coins, prices_reindexed)

    bak.make_prediction()

    bak.predictions = bak.predictions

    bak.get_weights(trading_rule = 'fee aware', buy_fee= 1, sell_fee=1)

    backtest_values = bak.backtest(verbose = True, buy_fee= 1, sell_fee=1) # might want to make it so put the fees in the init so dont have to put them twice, or can leave it as is 
    # leaving it as is adjusts the threshold for buy/sell/hold recommendations (via get_weights())

    baktest = pd.Series(backtest_values, name = 'portfolio_value')

    #print(baktest)

    
    # Plot the Series as a line graph
    baktest.plot(kind='line', marker='o', linestyle='-')

    # Customize the plot
    plt.title('Portfolio Value Over Time (1bp buy/sell fee)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio value (USD)')
    plt.ylim(0, 2)
    plt.grid(True)

    # Show the plot
    plt.show()
