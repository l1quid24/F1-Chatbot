import json

def lambda_handler(event, context):
    # 1. 加载你上传的那个 JSON 文件
    with open('strategies_kb.json', 'r') as f:
        kb = json.load(f)

    # 2. 从 Dialogflow 的请求中获取赛道名
    body = json.loads(event['body'])
    gp_name = body['queryResult']['parameters'].get('f1_track')

    # 3. 查表回复
    if gp_name in kb:
        best_strat = kb[gp_name][0][0] # 拿到频率最高的那个
        reply = f"根据历史数据，{gp_name} 最推荐的策略是 {best_strat}。"
    else:
        reply = f"抱歉，我暂时没有 {gp_name} 的历史策略数据。"

    return {
        'statusCode': 200,
        'body': json.dumps({"fulfillmentText": reply})
    }