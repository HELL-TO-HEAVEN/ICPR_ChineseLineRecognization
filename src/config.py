import logging, sys
from pandas import Timestamp
logging.basicConfig(format='%(asctime)s %(funcName)-20s: %(levelname)+8s: %(message)s', level=logging.INFO,)

CHARFILE= '~/PycharmProjects/ICPR_ChineseLineRecognization/data/Train/ChineseChar'

loglevel= logging.DEBUG
log = logging.getLogger(name='global')
log.setLevel(loglevel)
console = logging.StreamHandler(stream= sys.stdout)
formatter = logging.Formatter('%(asctime)s %(funcName)-16s[line:%(lineno)+4s]: %(levelname)+8s: %(message)s')
console.setLevel(loglevel)
console.setFormatter(formatter)
log.addHandler(console)
log.propagate = False

