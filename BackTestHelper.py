import pandas as pd
from tqdm import trange

class BackTestHelper():
    def __init__(self, coins, sell_fee = 0, buy_fee = 0):
        """
        coins: list of Securtiy objects used as securities for a BackTestEngine object
        sell_fee: n basis points of the value of the sold securities
        buy_fee: n basis points of the value of the purchased securities
        """
        self.coins = coins
        self.sell_fee = sell_fee
        self.buy_fee = buy_fee
        
    def __assign_recommendations(self, p):
        """
        to be used with pd.Series().apply()
        take a pd.Series object (a row of predicted price changes) and assign each predicted return a buy/sell recommendation
        """
        rec = {}
        for coin in self.coins: 
            pred = p[f'{coin.name}'] 
            # take the input (each row) and then index into each prediction (one for each coin)
            if pred > self.buy_fee + self.sell_fee: # if the coin is going to see returns greater than the cost to buy it and the cost to sell other stuff, sell other stuff to buy it (definitely buy)
                rec[coin.name] = 'large buy'
            elif pred > self.buy_fee: # if the coin is going to increase more in value than the buy fee, buy it
                rec[coin.name] = 'buy'
            elif pred < -self.sell_fee: # if the coin is going to decrease in price more than the sell fee, sell it 
                rec[coin.name] = 'sell'
            else: # if prediction lies between sell fee and buy fee, hold it (this is a greedy algorithm: only looking one step in the future/can't look more than one step ahead)
                rec[coin.name] = 'hold'
        return pd.Series(rec)
    
    def __redist_sells_revised(self, merged, inp, sell_holds = False):
        """
        To be used with pd.Dataframe().apply()
        Given buy recommendations, sell the securities with a "sell" recommendation and redistribute their weight to securities with "buy" and "large buy" recommendations. 
        merged: a pd.DataFrame object of two merged Series: a Series containing the securities (indices) and the current buy/sell/hold recommendations for the current minute and \
        a Series containing the securities (indices) and the weights from the previous minute
        inp: a (recommendation, old weight) pair (in the form of a Series or dictionary). 
        """
        merged = merged.copy() # this is slow, would rather not do this 
        merged.loc['Cash', merged.columns] = 'sell', 1-merged['old_weight'].sum() # add cash (also slow), the residual weight (allows for allocating cash without adding it as a security)
        if inp['recommendation'] == 'sell': 
            return 0 # get rid of all sells
        elif inp['recommendation'] == 'hold': # if the recommendation is to hold
            if sell_holds == False: # if keeping holds
                return inp['old_weight'] # don't do anything
            else: # if selling holds
                return 0 # weight goes to 0 
        else: # if the recommendation is to buy or to large buy
            freed_weight = merged.groupby('recommendation').sum()['old_weight'] # group by the recommendation and then sum (find the total weight attributed to each recommendation),
            freed_weight_sell = freed_weight.get('sell', 0)  # then find the value attributed to 'sell' (which is the total weight attributed to securities that are recommended to be sold)
            rec_value_counts =  merged.value_counts('recommendation')
            number_of_buys = rec_value_counts.get('large buy', 0) + rec_value_counts.get('buy', 0)  # find the count of each recommendation, then select the amount of 'large buy' or 'buy' recommendations. If there are no buys, return 0 (and weights will add up to less than 1)
            new_weight = freed_weight_sell / number_of_buys # for all buy and large buy recs, want to equally distribute the weight from the sold recs on top of the existing weight
        if inp['recommendation'] == 'large buy':
            freed_weight_hold = freed_weight.get('hold', 0) # find the total weight in the holds
            number_of_large_buys = rec_value_counts.get('large buy', 0)  # select the amount of 'large buy' recommendations. 
            new_weight += freed_weight_hold / number_of_large_buys # for alllarge buy recs, want to equally distribute the weight from the sold recs on top of the existing weight + the new weight
        return inp['old_weight'] + new_weight


    def __set_weights_revised(self, recommendation, old_weights): 
        """
        recommendation: a Series containing the securities (indices) and the current buy/sell/hold recommendations for the current minute
        old_weights: a Series containing the securities (indices) and the weights from the previous minute
        """
        large_buys = 'large buy' in recommendation.values # bool: do we need to sell our holds 
        recommendation.name = 'recommendation'
        old_weights.name = 'old_weight'
        merged = pd.merge(recommendation, old_weights, left_index=True, right_index=True, how='outer')
        new_weights = merged.apply(lambda x: self.__redist_sells_revised(merged, x, large_buys), axis = 1)
        return new_weights
    
    def all_weights(self, bak):
        """
        Using helper functions, create a full set of weights using recommendations generated from a predictions dataframe
        """
        recommendations = bak.predictions.apply(lambda x: self.__assign_recommendations(x), axis = 1, result_type='expand') 
        weights = pd.DataFrame(columns=recommendations.columns, index = recommendations.index) # empty weights df
        weight = pd.Series({column: 0 for column in recommendations.columns}) # initialize all weights to 0
        weights.iloc[0] = self.__set_weights_revised(recommendations.iloc[0], weight) # set the weights for the first minute using the recs for first minute and the initialized 0 weights
        for i in trange(1, len(weights)): # for the rest of the time steps
            weights.iloc[i] = self.__set_weights_revised(recommendations.iloc[i], weights.iloc[i-1]) # set the weights using the recs and the weights from the previous minute
        return weights