
import itertools
import numpy as np
import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)  # 最多显示数据的行数

# 导入指数
def import_index_data(path, back_trader_start=None, back_trader_end=None):
    # 导入指数数据
    df_index = pd.read_csv(path, parse_dates=['candle_end_time'], encoding='gbk')
    df_index['指数涨跌幅'] = df_index['close'].pct_change()
    df_index = df_index[['candle_end_time', '指数涨跌幅']]
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)
    df_index.rename(columns={'candle_end_time': '交易日期'}, inplace=True)

    if back_trader_start:
        df_index = df_index[df_index['交易日期'] >= pd.to_datetime(back_trader_start)]
    if back_trader_end:
        df_index = df_index[df_index['交易日期'] <= pd.to_datetime(back_trader_end)]

    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index


# 计算策略评价指标
def strategy_evaluate(equity, select_stock):

    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(equity['equity_curve'].iloc[-1], 2)

    # ===计算年化收益
    annual_return = (equity['equity_curve'].iloc[-1]) ** (
            '1 days 00:00:00' / (equity['交易日期'].iloc[-1] - equity['交易日期'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity['max2here'] = equity['equity_curve'].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity['dd2here'] = equity['equity_curve'] / equity['max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(equity.sort_values(by=['dd2here']).iloc[0][['交易日期', 'dd2here']])
    # 计算最大回撤开始时间
    start_date = equity[equity['交易日期'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['交易日期']
    # 将无关的变量删除
    equity.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)

    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)

    # ===统计每个周期
    results.loc[0, '盈利周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0])  # 盈利笔数
    results.loc[0, '亏损周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0])  # 亏损笔数
    results.loc[0, '胜率'] = format(results.loc[0, '盈利周期数'] / len(select_stock), '.2%')  # 胜率
    results.loc[0, '每周期平均收益'] = format(select_stock['选股下周期涨跌幅'].mean(), '.2%')  # 每笔交易平均盈亏
    results.loc[0, '盈亏收益比'] = round(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0]['选股下周期涨跌幅'].mean() / \
                                    select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0]['选股下周期涨跌幅'].mean() * (-1), 2)  # 盈亏比
    results.loc[0, '单周期最大盈利'] = format(select_stock['选股下周期涨跌幅'].max(), '.2%')  # 单笔最大盈利
    results.loc[0, '单周期大亏损'] = format(select_stock['选股下周期涨跌幅'].min(), '.2%')  # 单笔最大亏损

    # ===连续盈利亏损
    results.loc[0, '最大连续盈利周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] > 0, 1, np.nan))])  # 最大连续盈利次数
    results.loc[0, '最大连续亏损周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] <= 0, 1, np.nan))])  # 最大连续亏损次数

    # ===每年、每月收益率
    equity.set_index('交易日期', inplace=True)
    year_return = equity[['涨跌幅']].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    monthly_return = equity[['涨跌幅']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)

    return results.T, year_return, monthly_return


def create_empty_data(index_data, period):
    empty_df = index_data[['交易日期']].copy()
    empty_df['涨跌幅'] = 0.0
    empty_df['周期最后交易日'] = empty_df['交易日期']
    empty_df.set_index('交易日期', inplace=True)
    agg_dict = {'周期最后交易日': 'last'}
    empty_period_df = empty_df.resample(rule=period).agg(agg_dict)

    empty_period_df['每天涨跌幅'] = empty_df['涨跌幅'].resample(period).apply(lambda x: list(x))
    # 删除没交易的日期
    empty_period_df.dropna(subset=['周期最后交易日'], inplace=True)

    empty_period_df['选股下周期每天涨跌幅'] = empty_period_df['每天涨跌幅'].shift(-1)
    empty_period_df.dropna(subset=['选股下周期每天涨跌幅'], inplace=True)

    # 填仓其他列
    empty_period_df['股票数量'] = 0
    empty_period_df['买入股票代码'] = 'empty'
    empty_period_df['买入股票名称'] = 'empty'
    empty_period_df['选股下周期涨跌幅'] = 0.0
    empty_period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)
    empty_period_df.set_index('交易日期', inplace=True)

    empty_period_df = empty_period_df[['股票数量', '买入股票代码', '买入股票名称', '选股下周期涨跌幅', '选股下周期每天涨跌幅']]
    return empty_period_df
