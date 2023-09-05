# coding: utf-8
import yaml


class YamlParser:
    def __init__(self, yaml_file):
        """
        Initialize the class with the yaml file to be parsed
        :param yaml_file: str: path to the yaml file
        """
        self.yaml_file = yaml_file

    def parse(self):
        """
        Parse the yaml file and return the data as a dictionary
        :return: dict: data from the yaml file
        """
        with open(self.yaml_file, 'r') as file:
            # use safe_load to prevent malicious yaml code injection
            data = yaml.safe_load(file)
        return data

    def parse_recursive(self):
        """
        Parse the yaml file recursively and return the data as a dictionary
        """
        with open(self.yaml_file, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        
        def parse_dict(data):
            for key, value in data.items():
                if isinstance(value, dict):
                    parse_dict(value)
                else:
                    data[key] = value

        parse_dict(data)
        return data
