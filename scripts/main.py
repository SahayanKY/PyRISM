import sys
import json

from rism.rism import RISM

# pyrism input.json

# インプット読み込み
inputFile = sys.argv[1]
jsonDict = json.load(open(inputFile, 'r'))

temperature = jsonDict.get('temperature', 300)
closure = jsonDict.get('closure', None)
# closureに関しては後で実装方法を練り直す
rismDict = jsonDict['1DRISM']

# RISMオブジェクト生成
rism = RISM(rismDict, temperature, closure)
# 1D-RISM計算
rism.solve()




