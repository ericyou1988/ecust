import requests
import json

while True:
    city = input('请输入城市，回车退出：\n')
    if not city:
        break
    try:
        req = requests.get('http://wthrcdn.etouch.cn/weather_mini?city=%s' %city)
    except:
        print("查询失败")
        break

    req_city = json.loads(req.text)
    weath = req_city.get('data')
    if weath:
        print(weath['forecast'][0].get('date'))
        print(weath['forecast'][0].get('high'))
        print(weath['forecast'][0].get('low'))
        print(weath['forecast'][0].get('fengxiang'))
        print(weath['forecast'][0].get('type'))
    else:
        print("未获得")