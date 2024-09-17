import pandas as pd
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
idx = ["A", "B", "C", "D", "E", "F"]
ret_horizon = [1, 3, 5, 10, 30, 60, 120, 240]
market_width_df = pd.DataFrame(columns = idx, index = "width")
market_width_df.loc["width"] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
df = pd.read_csv("data.csv")
class tradebot:
    def __init__(self, lamb, fr, Cov):
        self.lamb = lamb
        self.pos = np.zeros(len(idx))
        self.fr = fr
        self.price_record = pd.DataFrame(columns = idx, index = range(ret_horizon[-1]))
        self.cash = 1000000
        self.Cov = Cov
        
    def trade(self, alpha, price):
        x = cp.Variable(len(alpha))
        objective = 0.5 * self.lamb * cp.quad_form(x, self.Cov) + (self.lamb * np.dot(self.Cov, self.pos)) @ x - alpha * self.fr @ x
        abs_terms = cp.sum(cp.abs(np.multiply(self.pos, x)))
        objective += abs_terms
        problem = cp.Problem(cp.Minimize(objective))
        problem.solve()
        #check if cash is enough
        if np.dot(price, x.value) > self.cash:
            x.value = x.value * self.cash / np.dot(price, x.value)
        #check if 
        self.pos += x.value
        self.cash -= np.dot(price, x.value) + np.sum(np.abs(np.multiply(market_width_df.loc["width"], x.value)))

    def update(self, price):
        self.price_record = pd.concat([self.price_record, price], axis = 1)
        if len(self.price_record) > ret_horizon[-1]:
            self.price_record = self.price_record.iloc[-ret_horizon[-1]:]
    def update_feature(self):
        #generate return of each stock of each ret_horizon by price_record
        returns = pd.DataFrame(columns = idx, index = ret_horizon)
        for i in range(len(ret_horizon)):
            returns.loc[ret_horizon[i]] = self.price_record.iloc[-ret_horizon[i]:].pct_change().sum()
        #flatten returns to a vector
        return returns.values.flatten()
    def generate_alpha(self, models):
        #generate alpha by returns
        alpha = np.zeros(len(idx))
        returns = self.update_feature()
        for i in range(len(idx)):
            alpha[i] = models[idx[i]].predict(returns)
        return alpha
    def simulate(self, models, starttime = 0, endtime = -1, df):
        if endtime == -1:
            endtime = len(df)
        for i in range(starttime, endtime):
            price = df.loc[i][idx]
            alpha = self.generate_alpha(models)
            self.trade(alpha, price)
            self.update(price)
            
            

class fastertradebot(tradebot):
    def __init__(self, lamb, fr, Cov):
        super().__init__(lamb, fr, Cov)
    def trade(self, alpha, price):
        x = cp.Variable(len(alpha))
        objective = 0.5 * self.lamb * cp.quad_form(x, self.Cov) + (self.lamb * np.dot(self.Cov, self.pos)) @ x - alpha * self.fr @ x
        abs_terms = cp.sum(cp.abs(np.multiply(self.pos, x)))
        objective += abs_terms
        problem = cp.Problem(cp.Minimize(objective))
        problem.solve()
        #check if cash is enough
        if np.dot(price, x.value) > self.cash:
            x.value = x.value * self.cash / np.dot(price, x.value)
        #check if 
        self.pos += x.value
        self.cash -= np.dot(price, x.value) + np.sum(np.abs(np.multiply(market_width_df.loc["width"] / price, x.value)))
        self.update(price)
        returns = pd.DataFrame(columns = idx, index = ret_horizon)
        for i in range(len(ret_horizon)):
            returns.loc[ret_horizon[i]] = self.price_record.iloc[-ret_horizon[i]:].pct_change().sum()
        #flatten returns to a vector
        return returns.values.flatten()
    def generate_alpha(self, models):
        #generate alpha by returns
        alpha = np.zeros(len(idx))
        returns = self.update_feature()
        for i in range(len(idx)):
            alpha[i] = models[idx[i]].predict(returns)
        return alpha
    #generate pnl graph within the process of simulation
    def simulate(self, models, starttime = 0, endtime = -1, df):
        plt.figure()
        if endtime == -1:
            endtime = len(df)
        for i in range(starttime, endtime):
            price = df.loc[i][idx]
            self.update(price)
            alpha = self.generate_alpha(models)
            self.trade(alpha, price)
            #generate pnl graph
            pnl = self.cash + np.sum(self.pos)
            plt.scatter(i, pnl)
            if i % 100 == 0:
                plt.show()

risk_limit_df = pd.DataFrame(columns = idx, index = ["limit"])
risk_limit_df.loc["limit"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


import scipy.optimize as opt


class fullbot(tradebot):
    def __init__(self, lamb, fr, Cov, grosslimit = 5e7, netlimit = 1e7):
        super().__init__(lamb, fr, Cov)
        self.grosslimit = grosslimit
        self.netlimit = netlimit
    def trade(self, alpha, price):
        #x proportional to fr * alpha
        #but the proportion is limited by the risk limit: 
        # risk_limit_df / price * (x + pos) < netlimit
        # abs(risk_limit_df / price * (x + pos)) < grosslimit
        def obj(x):
            return 0.5 * self.lamb * x @ self.Cov @ x + self.lamb * np.dot(self.Cov, self.pos) @ x - alpha @ self.fr @ x + np.sum(np.abs(np.multiply(x, market_width_df.loc["width"] / price)))
        def cons(x):
            return [self.netlimit - np.abs(np.sum(np.multiply(risk_limit_df.loc["limit"] / price, x + self.pos))), self.grosslimit - np.sum(np.abs(np.multiply(risk_limit_df.loc["limit"] / price, x + self.pos)))]
        x0 = np.zeros(len(alpha))

        res = opt.minimize(obj, x0, constraints = {"type": "ineq", "fun": cons})
        x = res.x
        self.pos += x.value
        self.cash -= np.dot(price, x.value) + np.sum(np.abs(np.multiply(market_width_df.loc["width"] / price, x.value)))
        self.update(price)
        returns = pd.DataFrame(columns = idx, index = ret_horizon)
        for i in range(len(ret_horizon)):
            returns.loc[ret_horizon[i]] = self.price_record.iloc[-ret_horizon[i]:].pct_change().sum()
        #flatten returns to a vector
        return returns.values.flatten()
    def generate_alpha(self, models):
        #generate alpha by returns
        alpha = np.zeros(len(idx))
        returns = self.update_feature()
        for i in range(len(idx)):
            alpha[i] = models[idx[i]].predict(returns)
        return alpha
    #generate pnl graph within the process of simulation
    def simulate(self, models, starttime = 0, endtime = -1, df):
        plt.figure()
        if endtime == -1:
            endtime = len(df)
        pnls = []
        for i in range(starttime, endtime):
            price = df.loc[i][idx]
            self.update(price)
            alpha = self.generate_alpha(models)
            self.trade(alpha, price)
            #generate pnl graph
            pnl = self.cash + np.sum(self.pos)
            pnls.append(pnl)
            if i % 100 == 0:
                plt.show()
        plt.plot(range(len(pnls)), pnls, label = "pnl")