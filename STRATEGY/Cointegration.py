from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import sys
import random
from MAIN.Basics import Strategy


def get_src_cls(source_name):
    return getattr(sys.modules[__name__], source_name)


class EGCointegration(Strategy):

    def __init__(self, x, y, on, col_name, is_cleaned=True):
        if is_cleaned is not True:
            x, y = EGCointegration.clean_data(x, y, on, col_name)
        self.timestamp = x[on].values
        self.x = x[col_name].values.reshape(-1, )
        self.y = y[col_name].values.reshape(-1, )
        self.beta = None
        self.resid_mean = None
        self.resid_std = None
        self.p = 0
        self._reward = 0
        self._record = None

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, value):
        self._record = value

    @classmethod
    def clean_data(cls, x, y, on, col_name):
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_df = pd.merge(left=x, right=y, on=on, how='outer')
        clean_df  = merged_df.loc[merged_df.notnull().all(axis=1), :]
        df_x = pd.DataFrame()
        df_y = pd.DataFrame()
        df_x[on] = clean_df[on].values
        df_y[on] = clean_df[on].values
        df_x[col_name] = clean_df[col_name + '_x'].values
        df_y[col_name] = clean_df[col_name + '_y'].values
        return df_x, df_y

    def cal_spread(self, x, y, is_norm):
        resid = y - x * self.beta
        resid = (resid - self.resid_mean) / self.resid_std if is_norm is True else resid
        return resid

    def get_sample(self, start, end):
        assert start < end <= len(self.x), 'Error:Invalid Indexing.'
        x_sample    = self.x[start:end]
        y_sample    = self.y[start:end]
        time_sample = self.timestamp[start:end]
        return x_sample, y_sample, time_sample

    @staticmethod
    def get_p_value(x, y):
        _, p_val, _ = coint(x, y)
        return p_val

    def run_ols(self, x, y):
        reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
        self.beta = reg.coef_[0]

    def calibrate(self, start, end, cl):
        x, y, _ = self.get_sample(start, end)
        self.p  = self.get_p_value(x, y)
        if self.p < cl:
            self.run_ols(x, y)
            resid = self.cal_spread(x, y, is_norm=False)
            self.resid_mean = resid.mean()
            self.resid_std  = resid.std()

    def gen_signal(self, start, end, trade_th, stop_loss, transaction_cost):
        stop_loss  = trade_th + stop_loss
        x, y, time = self.get_sample(start, end)
        spread = self.cal_spread(x, y, is_norm=True)
        price  = self.cal_spread(x, y, is_norm=False)

        spread_t0 = spread[:-1]
        spread_t1 = spread[1:]
        price     = price[1:]
        t_t1      = time[1:]

        ind_buy  = np.logical_and(spread_t0 > -trade_th, spread_t1 <= -trade_th).reshape(-1, )
        ind_sell = np.logical_and(spread_t0 <  trade_th, spread_t1 >=  trade_th).reshape(-1, )
        ind_stop = np.logical_or(np.logical_or(np.logical_and(spread_t0 > -stop_loss, spread_t1 <= -stop_loss).reshape(-1, ),
                                               np.logical_and(spread_t0 <  stop_loss, spread_t1 >=  stop_loss).reshape(-1, )),
                                 np.logical_or(np.logical_and(spread_t0 > 0, spread_t1 <= 0).reshape(-1, ),
                                               np.logical_and(spread_t0 < 0, spread_t1 >= 0).reshape(-1, )))

        order = np.array([None] * len(t_t1))
        order[ind_buy]  = 'Buy'
        order[ind_sell] = 'Sell'
        order[ind_stop] = 'Stop'
        order[-1]       = 'Stop'

        ind_order = order != None
        time      = t_t1[ind_order]
        price     = price[ind_order]
        order     = order[ind_order]
        x         = x[1:][ind_order]
        y         = y[1:][ind_order]
        gross_exp = y + abs(x) * self.beta
        cost      = abs(gross_exp * transaction_cost)
        return time, price, order, gross_exp, cost

    @staticmethod
    def gen_trade_record(time, price, order, cost):
        if len(order) == 0:
            return None

        n_buy_sell  = sum(order != 'Stop')
        trade_time  = np.array([None] * n_buy_sell, object)
        trade_price = np.array([None] * n_buy_sell, float)
        trade_cost  = np.array([None] * n_buy_sell, float)
        close_time  = np.array([None] * n_buy_sell, object)
        close_price = np.array([None] * n_buy_sell, float)
        close_cost  = np.array([None] * n_buy_sell, float)
        long_short  = np.array([0]    * n_buy_sell, int)

        current_holding = 0
        j = 0

        for i in range(len(order)):
            sign_holding = int(np.sign(current_holding))
            if order[i] == 'Buy':
                close_pos                    = (sign_holding < 0) * -current_holding
                close_time[j  - close_pos:j] = time[i]
                close_price[j - close_pos:j] = price[i]
                close_cost[j  - close_pos:j] = cost[i]
                trade_time[j]                = time[i]
                trade_price[j]               = price[i]
                long_short[j]                = 1
                trade_cost[j]                = cost[i]
                buy_sell        = close_pos + 1
                current_holding = current_holding + buy_sell
                j += 1
            elif order[i] == 'Sell':
                close_pos                    = (sign_holding > 0) * -current_holding
                close_time[j  + close_pos:j] = time[i]
                close_price[j + close_pos:j] = price[i]
                close_cost[j  + close_pos:j] = cost[i]
                trade_time[j]                = time[i]
                trade_price[j]               = price[i]
                long_short[j]                = -1
                trade_cost[j]                = cost[i]
                buy_sell        = close_pos - 1
                current_holding = current_holding + buy_sell
                j += 1
            else:
                close_pos                    = abs(current_holding)
                close_time[j  - close_pos:j] = time[i]
                close_price[j - close_pos:j] = price[i]
                close_cost[j  - close_pos:j] = cost[i]
                current_holding = 0

        profit       = (close_price - trade_price) * long_short
        trade_record = {'trade_time' : trade_time,
                        'trade_price': trade_price,
                        'close_time' : close_time,
                        'close_price': close_price,
                        'long_short' : long_short,
                        'trade_cost' : trade_cost,
                        'close_cost' : close_cost,
                        'profit'     : profit}

        return trade_record

    @staticmethod
    def get_indices(index, n_hist, n_forward):
        assert n_hist <= index, 'Error:Invalid number of historical observations.'
        start_hist    = index - n_hist
        end_hist      = index
        start_forward = index
        end_forward   = index + n_forward
        return start_hist, end_hist, start_forward, end_forward

    def process(self, n_hist, n_forward, trade_th, stop_loss, cl, transaction_cost, index=None, **kwargs):
        index = random.randint(n_hist, len(self.x) - n_forward) if index is None else index
        start_hist, end_hist, start_forward, end_forward = self.get_indices(index, n_hist, n_forward)
        self.calibrate(start_hist, end_hist, cl)
        self.reward = 0
        self.record = None
        if self.p < cl:
            time, price, order, gross_exp, cost = self.gen_signal(start_forward, end_forward, trade_th, stop_loss, transaction_cost)
            trade_record = self.gen_trade_record(time, price, order, cost)
            returns      = (trade_record['profit'] -
                            trade_record['trade_cost'] -
                            trade_record['close_cost']) / abs(trade_record['trade_price'])
            if (len(returns) > 0) and (np.any(np.isnan(returns)) is not True):
                self.reward = min(returns.mean(), 10)
            self.record = trade_record
