"""
CAPM Beta Algorithmic Trading Case
Rotman BMO Finance Research and Trading Lab, Uniersity of Toronto (C) All rights reserved.

Preamble:
-> Code will have a small start up period; however, trades should only be executed once forward market price is available,
hence there should not be any issue caused.

-> Code only runs effectively if the News articles are formatted as they are now. The only way to get the required new data is by parsing the text.
"""
import warnings
warnings.filterwarnings("ignore")
import signal
import requests
from time import sleep
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

CAPM_vals = {}
expected_return = {}
historical_prices = {'ALPHA': [], 'GAMMA': [], 'THETA': []}
current_position = {'ALPHA': 0, 'GAMMA': 0, 'THETA': 0}
trade_history = []

# class that passes error message, ends the program
class ApiException(Exception):
    pass


# code that lets us shut down if CTRL C is pressed
def signal_handler(signum, frame):
    global shutdown
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    shutdown = True


API_KEY = {'X-API-Key': 'SAD5CP5E'}
shutdown = False
session = requests.Session()
session.headers.update(API_KEY)


# code that gets the current tick
def get_tick(session):
    resp = session.get('http://localhost:9999/v1/case')
    if resp.ok:
        case = resp.json()
        return case['tick']
    raise ApiException('fail - cant get tick')


# code that parses the first and latest news instances for forward market predictions and the risk free rate
# Important: this code only works if the only '%' character is in front of the RISK FREE RATE and the onle '$' character is in front of the forward price suggestions
def get_news(session):
    news = session.get('http://localhost:9999/v1/news')
    if news.ok:
        newsbook = news.json()
        for i in range(len(newsbook[-1]['body'])):
            if newsbook[-1]['body'][i] == '%':
                CAPM_vals['%Rf'] = round(float(newsbook[-1]['body'][i - 4:i]) / 100, 4)
        latest_news = newsbook[0]
        if len(newsbook) > 1:
            for j in range(len(latest_news['body']) - 1, 1, -1):
                while latest_news['body'][j] != '$':
                    j -= 1
            CAPM_vals['forward'] = float(latest_news['body'][j + 1:-1])
        return CAPM_vals
    raise ApiException('timeout')


# gets all the price data for all securities
def pop_prices(session):
    price_act = session.get('http://localhost:9999/v1/securities')
    if price_act.ok:
        prices = price_act.json()
        return prices
    raise ApiException('fail - cant get securities')

#########################################################################

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_and_predict(historical_prices, look_back):
    if len(historical_prices) <= look_back:
        print("Not enough historical data to form a dataset. Need at least", look_back + 1, "data points.")
        return None

    data_array = np.array(historical_prices).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_array)

    X, y = create_dataset(scaled_data, look_back)

    if X.shape[0] == 0:  # Check if X is empty after dataset creation
        print("No samples were created. Check the create_dataset function and input data.")
        return None

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y.ravel())

    predicted_price = model.predict(X)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

    return predicted_price


#########################################################################

def pattern1_buy_or_sell(session, expected_return, current_allocations, historical_prices, future_prices):
    global current_position
    # 动态计算买入和卖出阈值
    if '%RM' in CAPM_vals and CAPM_vals['%RM']:
        try:
            risk_premium = float(CAPM_vals['%RM']) - CAPM_vals['%Rf']
            buy_threshold = risk_premium * 1.1
            sell_threshold = risk_premium * 0.9
        except ValueError:
            # 如果 '%RM' 不是有效数字，则跳过此次交易
            return
    else:
        # 如果 '%RM' 未定义或为空，则跳过此次交易
        return
    order_type = 'MARKET'

    for ticker, exp_return in expected_return.items():
        # 获取目标持仓和当前持仓
        target_allocation = current_allocations[ticker]
        predicted_price = future_prices.get(ticker)
        # print(current_position)
        if predicted_price is None:
            print(f"No predicted price available for {ticker}, skipping buy/sell decision.")
            continue

        order_quantity = 100

        if exp_return != 'Wait for market forward price':
            # 获取最近的股价
            last_price = historical_prices[ticker][-1]

            # 买入逻辑：预期回报高于阈值，并且最近股价高于移动平均线
            if exp_return > buy_threshold and predicted_price > last_price:
                if exp_return > 1.25 * buy_threshold:
                    # 高预期回报，增加交易量
                    order_quantity = min(10000, max(1000, int((exp_return - buy_threshold) / buy_threshold * 1000)))
                else:
                    # 默认交易量
                    order_quantity = 100
                if current_position[ticker] + order_quantity <= target_allocation:
                    response1 = session.post('http://localhost:9999/v1/orders',
                                params={'ticker': ticker, 'type': order_type, 'quantity': order_quantity, 'action': 'BUY'})
                    print(response1.text)
                    if response1.ok:
                        trade_history.append({
                            'type': 'BUY',
                            'ticker': ticker,
                            'quantity': order_quantity,
                            'price': historical_prices[ticker][-1],
                            'stop_loss_price': historical_prices[ticker][-1] * 0.85,
                            'take_profit_price': historical_prices[ticker][-1] * 1.25
                        })
                        # print(f"Latest Trade: {trade_history[-1]}")
                        current_position[ticker] += order_quantity  # 更新持仓量

            # 卖出逻辑：预期回报低于阈值，或股价低于移动平均线
            elif (exp_return < sell_threshold and predicted_price < last_price) or current_position[ticker] > target_allocation:
                if exp_return < 0.75 * sell_threshold:
                    # 高风险，减少交易量
                    order_quantity = min(10000, max(1000, int((sell_threshold - exp_return) / sell_threshold * 1000)))
                else:
                    # 默认交易量
                    order_quantity = 100
                if current_position[ticker] - order_quantity >= target_allocation:
                    response2 = session.post('http://localhost:9999/v1/orders',
                                params={'ticker': ticker, 'type': order_type, 'quantity': order_quantity, 'action': 'SELL'})
                    print(response2.text)
                    if response2.ok:
                        trade_history.append({
                            'type': 'SELL',
                            'ticker': ticker,
                            'quantity': order_quantity,
                            'price': historical_prices[ticker][-1],
                            'stop_loss_price': historical_prices[ticker][-1] * 1.2,
                            'take_profit_price': historical_prices[ticker][-1] * 0.75
                        })
                        current_position[ticker] -= order_quantity  # 更新持仓量

#----------------------------------------------

def pattern2_buy_or_sell(session, expected_return, current_allocations, historical_prices, future_prices):
    global current_position
    # 动态计算买入和卖出阈值
    if '%RM' in CAPM_vals and CAPM_vals['%RM']:
        try:
            risk_premium = float(CAPM_vals['%RM']) - CAPM_vals['%Rf']
            buy_threshold = risk_premium * 1.1
            sell_threshold = risk_premium * 0.9
        except ValueError:
            # 如果 '%RM' 不是有效数字，则跳过此次交易
            return
    else:
        # 如果 '%RM' 未定义或为空，则跳过此次交易
        return
    order_type = 'MARKET'

    for ticker, exp_return in expected_return.items():
        # 获取目标持仓和当前持仓
        target_allocation = current_allocations[ticker]
        predicted_price = future_prices.get(ticker)
        # print(current_position)
        if predicted_price is None:
            print(f"No predicted price available for {ticker}, skipping buy/sell decision.")
            continue

        order_quantity = 100

        if exp_return != 'Wait for market forward price':
            # 获取最近的股价
            last_price = historical_prices[ticker][-1]

            # 买入逻辑：预期回报高于阈值，并且最近股价高于移动平均线
            if (exp_return > buy_threshold and predicted_price > last_price) or current_position[ticker] > target_allocation:
                if exp_return > 1.25 * buy_threshold:
                    # 高预期回报，增加交易量
                    order_quantity = min(10000, max(1000, int((exp_return - buy_threshold) / buy_threshold * 1000)))
                else:
                    # 默认交易量
                    order_quantity = 100
                if current_position[ticker] - order_quantity >= target_allocation:
                    response1 = session.post('http://localhost:9999/v1/orders',
                                params={'ticker': ticker, 'type': order_type, 'quantity': order_quantity, 'action': 'SELL'})
                    # print(response1.text)
                    if response1.ok:
                        trade_history.append({
                            'type': 'SELL',
                            'ticker': ticker,
                            'quantity': order_quantity,
                            'price': historical_prices[ticker][-1],
                            'stop_loss_price': historical_prices[ticker][-1] * 1.2,
                            'take_profit_price': historical_prices[ticker][-1] * 0.75
                        })
                        # print(f"Latest Trade: {trade_history[-1]}")
                        current_position[ticker] -= order_quantity  # 更新持仓量
            # 卖出逻辑：预期回报低于阈值，或股价低于移动平均线
            elif exp_return < sell_threshold and predicted_price < last_price and current_position[ticker] < target_allocation - order_quantity:
                if exp_return < 0.75 * sell_threshold:
                    # 高风险，减少交易量
                    order_quantity = min(10000, max(1000, int((sell_threshold - exp_return) / sell_threshold * 1000)))
                else:
                    # 默认交易量
                    order_quantity = 100
                if current_position[ticker] + order_quantity <= target_allocation:
                    response2 = session.post('http://localhost:9999/v1/orders',
                                params={'ticker': ticker, 'type': order_type, 'quantity': order_quantity, 'action': 'BUY'})
                    # print(response2.text)
                    if response2.ok:
                        trade_history.append({
                            'type': 'BUY',
                            'ticker': ticker,
                            'quantity': order_quantity,
                            'price': historical_prices[ticker][-1],
                            'stop_loss_price': historical_prices[ticker][-1] * 0.85,
                            'take_profit_price': historical_prices[ticker][-1] * 1.25
                        })
                        current_position[ticker] += order_quantity  # 更新持仓量

def check_stop_loss_take_profit(session, historical_prices):
    global trade_history, current_position
    trades_to_remove = []

    for trade in trade_history:
        current_price = historical_prices[trade['ticker']][-1]
        action = None
        if trade['type'] == 'BUY' and (
                current_price <= trade['stop_loss_price'] or current_price >= trade['take_profit_price']):
            # 执行卖出操作以止盈或止损
            action = 'SELL'
            current_position[trade['ticker']] -= trade['quantity']
        elif trade['type'] == 'SELL' and (
                current_price >= trade['stop_loss_price'] or current_price <= trade['take_profit_price']):
            # 执行买入操作以止损或止盈
            action = 'BUY'
            current_position[trade['ticker']] += trade['quantity']

        if action:
            response = session.post('http://localhost:9999/v1/orders',
                                    params={'ticker': trade['ticker'], 'type': 'MARKET', 'quantity': trade['quantity'],
                                            'action': action})
            if response.ok:
                print(f"Executed trade: {action} {trade['quantity']} of {trade['ticker']} at price {current_price}")
                trades_to_remove.append(trade)

    for trade in trades_to_remove:
        trade_history.remove(trade)


def update_historical_price(session):
    global historical_prices
    current_prices = pop_prices(session)
    # print("Current Prices:", current_prices)
    for price_data in current_prices:
        ticker = price_data['ticker']
        if ticker in historical_prices:
            latest_price = price_data['last']
            historical_prices[ticker].append(latest_price)
            # print(f"Updated {ticker}: Latest Price = {latest_price}, Recent History = {historical_prices[ticker][-5:]}")

def calculate_stock_score(historical_prices):
    scores = {}
    days_30_weight = 0.7  # 30天的权重
    days_60_weight = 0.2  # 60天的权重
    days_90_weight = 0.1  # 90天的权重

    for ticker, prices in historical_prices.items():
        if len(prices) < 2:  # 需要至少两个价格点来计算回报率
            print(f"Not enough data for {ticker}")
            continue

        # 计算回报率
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

        # 计算不同时间段的平均回报率
        avg_return_30_days = sum(returns[-30:]) / min(30, len(returns))
        avg_return_60_days = sum(returns[-60:-30]) / min(30, len(returns) - 30) if len(returns) >= 60 else avg_return_30_days
        avg_return_90_days = sum(returns[-90:-60]) / min(30, len(returns) - 60) if len(returns) >= 90 else avg_return_60_days

        score = (avg_return_30_days * days_30_weight) + (avg_return_60_days * days_60_weight) + (avg_return_90_days * days_90_weight)
        scores[ticker] = score

    return scores


def adjust_allocations(stock_scores, current_allocations, total_investment_limit, max_order_size=10000, max_gross_limit=250000, max_net_limit=100000):
    # 计算股票得分的总和
    total_score = sum(stock_scores.values())

    if total_score == 0:
        print("Total score is zero, skipping allocation adjustments.")
        return current_allocations

    # 确定每只股票的目标持仓比例
    target_allocations = {ticker: score / total_score for ticker, score in stock_scores.items()}

    # 根据目标持仓比例和总投资限额计算每只股票的目标持仓量
    target_values = {ticker: int(target_allocations[ticker] * total_investment_limit) for ticker in target_allocations}

    # 调整当前持仓，使之接近目标持仓
    new_allocations = {}
    total_trades = 0  # 总交易量（包括买入和卖出）
    net_trades = 0    # 净交易量（买入减去卖出）

    for ticker in current_allocations:
        if ticker in target_values:
            # 计算调整量
            adjustment = target_values[ticker] - current_allocations[ticker]

            # 限制每次调整的最大值为最大单一订单规模
            adjustment = max(min(adjustment, max_order_size), -max_order_size)

            # 检查交易限制
            if total_trades + abs(adjustment) > max_gross_limit or net_trades + adjustment > max_net_limit:
                continue  # 如果超出限制，则跳过此次调整

            # 应用调整
            new_allocations[ticker] = current_allocations[ticker] + adjustment
            total_trades += abs(adjustment)
            net_trades += adjustment
        else:
            # 对于不在目标持仓中的股票，保持不变
            new_allocations[ticker] = current_allocations[ticker]

    return new_allocations



def main():
    with requests.Session() as session:
        session.headers.update(API_KEY)

        # 初始化每只股票的持仓比例
        current_allocations = {'ALPHA': 1 / 3, 'GAMMA': 1 / 3, 'THETA': 1 / 3}
        total_investment_limit = 100000  # 假设总投资限额为100,000


        ritm = pd.DataFrame(columns=['RITM', 'BID', 'ASK', 'LAST', '%Rm'])
        alpha = pd.DataFrame(columns=['ALPHA', 'BID', 'ASK', 'LAST', '%Ri', '%Rm'])
        gamma = pd.DataFrame(columns=['GAMMA', 'BID', 'ASK', 'LAST', '%Ri', '%Rm'])
        theta = pd.DataFrame(columns=['THETA', 'BID', 'ASK', 'LAST', '%Ri', '%Rm'])
        while get_tick(session) < 600 and not shutdown:
            # update the forward market price and rf rate
            get_news(session)

            ##update RITM bid-ask dataframe
            pdt_RITM = pd.DataFrame(pop_prices(session)[0])
            ritmp = pd.DataFrame(
                {'RITM': '', 'BID': pdt_RITM['bid'], 'ASK': pdt_RITM['ask'], 'LAST': pdt_RITM['last'], '%Rm': ''})
            if ritm['BID'].empty or ritmp['LAST'].iloc[0] != ritm['LAST'].iloc[0]:
                ritm = pd.concat([ritmp, ritm.loc[:]]).reset_index(drop=True)
                ritm['%Rm'] = (ritm['LAST'] / ritm['LAST'].shift(-1)) - 1
                if ritm.shape[0] >= 31:
                    ritm = ritm.iloc[:30]

            # generate expected market return paramter
            if 'forward' in CAPM_vals.keys():
                CAPM_vals['%RM'] = (CAPM_vals['forward'] - ritm['LAST'].iloc[0]) / ritm['LAST'].iloc[0]
            else:
                CAPM_vals['%RM'] = ''

            ##update ALPHA bid-ask dataframe
            pdt_ALPHA = pd.DataFrame(pop_prices(session)[1])
            alphap = pd.DataFrame(
                {'ALPHA': '', 'BID': pdt_ALPHA['bid'], 'ASK': pdt_ALPHA['ask'], 'LAST': pdt_ALPHA['last'], '%Ri': '',
                 '%Rm': ''})
            if alpha['BID'].empty or alphap['LAST'].iloc[0] != alpha['LAST'].iloc[0]:
                alpha = pd.concat([alphap, alpha.loc[:]]).reset_index(drop=True)
                alpha['%Ri'] = (alpha['LAST'] / alpha['LAST'].shift(-1)) - 1
                alpha['%Rm'] = (ritm['LAST'] / ritm['LAST'].shift(-1)) - 1
                if alpha.shape[0] >= 31:
                    alpha = alpha.iloc[:30]

            ##update GAMMA bid-ask dataframe
            pdt_GAMMA = pd.DataFrame(pop_prices(session)[2])
            gammap = pd.DataFrame(
                {'GAMMA': '', 'BID': pdt_GAMMA['bid'], 'ASK': pdt_GAMMA['ask'], 'LAST': pdt_GAMMA['last'], '%Ri': '',
                 '%Rm': ''})
            if gamma['BID'].empty or gammap['LAST'].iloc[0] != gamma['LAST'].iloc[0]:
                gamma = pd.concat([gammap, gamma.loc[:]]).reset_index(drop=True)
                gamma['%Ri'] = (gamma['LAST'] / gamma['LAST'].shift(-1)) - 1
                gamma['%Rm'] = (ritm['LAST'] / ritm['LAST'].shift(-1)) - 1
                if gamma.shape[0] >= 31:
                    gamma = gamma.iloc[:30]

            ##update THETA bid-ask dataframe
            pdt_THETA = pd.DataFrame(pop_prices(session)[3])
            thetap = pd.DataFrame(
                {'THETA': '', 'BID': pdt_THETA['bid'], 'ASK': pdt_THETA['ask'], 'LAST': pdt_THETA['last'], '%Ri': '',
                 '%Rm': ''})
            if theta['BID'].empty or thetap['LAST'].iloc[0] != theta['LAST'].iloc[0]:
                theta = pd.concat([thetap, theta.loc[:]]).reset_index(drop=True)
                theta['%Ri'] = (theta['LAST'] / theta['LAST'].shift(-1)) - 1
                theta['%Rm'] = (ritm['LAST'] / ritm['LAST'].shift(-1)) - 1
                if theta.shape[0] >= 31:
                    theta = theta.iloc[:30]

            beta_alpha = (alpha['%Ri'].cov(ritm['%Rm'])) / (ritm['%Rm'].var())
            beta_gamma = (gamma['%Ri'].cov(ritm['%Rm'])) / (ritm['%Rm'].var())
            beta_theta = (theta['%Ri'].cov(ritm['%Rm'])) / (ritm['%Rm'].var())

            CAPM_vals['Beta - ALPHA'] = beta_alpha
            CAPM_vals['Beta - GAMMA'] = beta_gamma
            CAPM_vals['Beta - THETA'] = beta_theta

            if CAPM_vals['%RM'] != '':
                er_alpha = CAPM_vals['%Rf'] + CAPM_vals['Beta - ALPHA'] * (CAPM_vals['%RM'] - CAPM_vals['%Rf'])
                er_gamma = CAPM_vals['%Rf'] + CAPM_vals['Beta - GAMMA'] * (CAPM_vals['%RM'] - CAPM_vals['%Rf'])
                er_theta = CAPM_vals['%Rf'] + CAPM_vals['Beta - THETA'] * (CAPM_vals['%RM'] - CAPM_vals['%Rf'])
            else:
                er_alpha = 'Wait for market forward price'
                er_gamma = 'Wait for market forward price'
                er_theta = 'Wait for market forward price'

            expected_return['ALPHA'] = er_alpha
            expected_return['GAMMA'] = er_gamma
            expected_return['THETA'] = er_theta

            update_historical_price(session)
            future_prices = {}
            for ticker in ['ALPHA', 'GAMMA', 'THETA']:
                predicted_price = train_and_predict(np.array(historical_prices[ticker]), look_back=40)
                if predicted_price is not None:
                    future_prices[ticker] = predicted_price[-1][0]  # 使用最新的预测价格

            update_historical_price(session)

            stock_scores = calculate_stock_score(historical_prices)
            # print("Stock Scores:", stock_scores)

            current_allocations = adjust_allocations(stock_scores, current_allocations, total_investment_limit)
            #print("Adjusted Allocations:", current_allocations)

            # Uncomment this string to enable Buy/Sell
            sleep(0.5)
            pattern2_buy_or_sell(session, expected_return, current_allocations, historical_prices, future_prices)
            # check_stop_loss_take_profit(session, historical_prices)
            print("Current positions after pattern1 buy or sell:", current_position)
            # print(expected_return)


if __name__ == '__main__':
    main()


