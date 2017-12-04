import pickle

class Loader(object):
    def __init__(self, base_dir = "../data/"):
        self.base_dir = base_dir
        self.data_info = {
            "000300": {
                "file": "000300.pkl",
                "start": "2007-01-04",
                "end": "2016-12-30",
                "level": "minute",
                "info": "沪深300指数",
                "source": "米筐量化平台"
            }
        }

    def load(self, code):
        if code not in self.data_info:
            print("Invalid code: " + str(code))

        with open(self.base_dir + self.data_info[code]["file"], "rb") as f:
            return pickle.load(f), self.data_info[code]

    def get_data_list(self):
        [print(self.data_info[key]) for key in self.data_info]
