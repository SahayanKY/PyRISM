import sys
import json

from rism.rism import RISM
from rism.closure import parseClosureType

# pyrism input.json

# インプット読み込み
inputFile = sys.argv[1]
jsonDict = json.load(open(inputFile, 'r'))

temperature = jsonDict.get('temperature', 300)
closureType = parseClosureType(jsonDict)
rismDict = jsonDict['1DRISM']

# RISMオブジェクト生成
rism = RISM(rismDict, temperature, closureType)
# 1D-RISM計算
rism.solve()




