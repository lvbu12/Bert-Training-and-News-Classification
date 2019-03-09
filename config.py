# -*- coding: utf-8 -*-
import json
import six
import copy


class Config(object):

    def __init__(self):
        pass

    @classmethod
    def from_dict(cls, json_object):
        config = Config()
        for (key, val) in six.iteritems(json_object):
            config.__dict__[key] = val

        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as f:
            text = f.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file):
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(self.to_json_string())

if __name__ == "__main__":

    config = Config.from_json_file('Configs/para_cls.json')
    config_dict = config.to_dict()
    print(type(config_dict))
    print(config_dict)
    print(config.encoder_learned_pos)
    config.para_cls_size = 14
    config.para_cls_load_model_path = "Para_cls_models"
    config.para_cls_save_model_path = "Para_cls_models"
    config.to_json_file('zz.json')
