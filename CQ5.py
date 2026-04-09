from typing import Counter

import fastf1
from fastf1 import plotting
import pandas as pd
import random

from requests import session

# 设置缓存
fastf1.Cache.enable_cache('f1_cache') 

class F1StrategyEngine:
    def __init__(self):
        pass

    def get_driver_strategy(self, laps):
        """
        [工具方法] 获取单名车手的轮胎序列
        """
        # 注意：这里需要 self 参数
        strategy_data = laps[['Stint', 'Compound']].drop_duplicates().sort_values(by='Stint')
        return strategy_data['Compound'].tolist()

    def get_historical_tire_strategies(self, gp_name, years=[2023, 2024, 2025]):
        """
        [数据抓取方法] 统计历史前 10 名的策略
        """
        all_strategies = []
    
        for year in years:
            try:
                print(f"loading {year} {gp_name} data...")
                session = fastf1.get_session(year, gp_name, 'R')
                session.load(laps=True, telemetry=False, weather=False)
            
                # 获取前 10 名缩写
                top_10_drivers = session.results.head(10)['Abbreviation'].tolist()
            
                for drv in top_10_drivers:
                    driver_laps = session.laps.pick_driver(drv)
                    # 关键修改：使用 self. 调用类内部方法
                    strategy_sequence = self.get_driver_strategy(driver_laps) 
                    
                    # 关键修改：必须将结果存入列表
                    strategy_str = " -> ".join(strategy_sequence)
                    all_strategies.append(strategy_str)
                    print(f"  - {drv}: {strategy_str}")
                    
            except Exception as e:
                print(f"skip {year} {gp_name}: {e}")
                continue

        # 统计出现次数最高的 5 种组合
        return Counter(all_strategies).most_common(5)

    def get_inference(self, year, gp_name, team_name, weather_condition='Dry'):
        """
        [推理接口] 供 Member C 调用
        """
        # ... 这里保留你之前的 get_inference 逻辑 ...
        # 调用方法时记得加 self.
        pass

        """
        核心推理接口：供 Member C 调用
        """
        try:
            # 1. 快速获取历史参考数据 (以 2023 年该站为例)
            session = fastf1.get_session(year, gp_name, 'R')
            session.load(laps=True, telemetry=False, weather=True)
            
            # 2. 模拟分类模型：预测 Safety Car 风险 (CQ3)
            # 实际上你会用 A 提供的特征跑你的 RandomForest
            sc_risk_score = self._predict_sc_risk(session, weather_condition)
            
            # 3. 模拟推荐逻辑：Strategy Recommendation
            # 基于 Case-based reasoning 的逻辑原型
            recommendation = self._generate_recommendation(team_name, weather_condition, session)
            
            return {
                "status": "success",
                "data": {
                    "sc_risk": sc_risk_score, # 'High', 'Medium', 'Low'
                    "recommended_strategy": recommendation['strategy'],
                    "reasoning": recommendation['reasoning'],
                    "historical_reference": f"{year} {gp_name} Data used"
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}



# --- 模拟运行 ---
# 1. 实例化类
engine = F1StrategyEngine()

# 2. 调用实例的方法
print("\n=== Top 5 Historical Strategies for Silverstone ===")
strategies = engine.get_historical_tire_strategies('Silverstone', years=[2023, 2024, 2025])

print("\n--- 最终统计结果 ---")
for i, (strat, count) in enumerate(strategies):
    print(f"{i+1}. {strat} (被 {count} 位前十名完赛车手使用)")





import json
# 跑完统计后
results = engine.get_historical_tire_strategies('Silverstone')
with open('silverstone_strategies.json', 'w') as f:
    json.dump(results, f)