"""
    The module contain Porperties class to manage paramters & stats in various parts of the system
    The module might be used in Maya, so its Python 2.7 compatible
"""

import json


class Properties():
    """Keeps, loads, and saves cofiguration & statistic information
        Supports gets&sets as a dictionary
        Provides shortcuts for batch-init configurations

        One of the usages -- store system-dependent basic cofiguration
    """
    def __init__(self, filename="", clean_stats=False):
        self.properties = {}

        if filename:
            self.properties = self._from_file(filename)
            if clean_stats:  # only makes sense when initialized from file =) 
                self.clean_stats()

    def has(self, key):
        """Used to quety if a top-level property/section is already defined"""
        return key in self.properties

    def set_section_config(self, section, **kwconfig):
        """adds or modifies a (top level) section and updates its configuration info
        """
        # create new section
        if section not in self.properties:
            self.properties[section] = {
                'config': kwconfig,
                'stats': {}
            }
            return
        # section exists
        for key, value in kwconfig.items():
            self.properties[section]['config'][key] = value
        
    def set_section_stats(self, section, **kwstats):
        """adds or modifies a (top level) section and updates its statistical info
        """
        # create new section
        if section not in self.properties:
            self.properties[section] = {
                'config': {},
                'stats': kwstats
            }
            return
        # section exists
        for key, value in kwstats.items():
            self.properties[section]['stats'][key] = value

    def set_basic(self, **kwconfig):
        """Adds/updates info on the top level of properties
            Only to be used for basic information!
        """
        # section exists
        for key, value in kwconfig.items():
            self.properties[key] = value

    def clean_stats(self):
        """ Remove info from all Stats sub sections """
        for key, value in self.properties.items():
            # detect section
            if isinstance(value, dict) and 'stats' in value:
                value['stats'] = {}

    def serialize(self, filename):
        with open(filename, 'w') as f_json:
            json.dump(self.properties, f_json, indent=2, sort_keys=True)

    def _from_file(self, filename):
        """ Load properties from previously created file """
        with open(filename, 'r') as f_json:
            return json.load(f_json)

    def __getitem__(self, key):
        return self.properties[key]

    def __setitem__(self, key, value):
        self.properties[key] = value

    def __contains__(self, key):
        return key in self.properties

    def __str__(self):
        return str(self.properties)
