import json
class profile_manager:
    fields = ["id", "th"]

    def __init__(self):
        self.profiles = json.load(open('profile.json', 'r'))

    def _get_index(self, _id):
        return [i for i, p in enumerate(self.profiles) if p["id"] == _id][0]

    def update_id(self, _id):
        self.profile_id = _id

    def update(self, profile):
        self.profiles[self._get_index(self.profile_id)] = profile
        json.dump(self.profiles, open('profile.json', 'w'))

    def update_field(self, field, val):
        profile = self.profiles[self._get_index(self.profile_id)]
        profile[field] = val
        self.update(profile)

    def get_field(self, field):
        if field in self.profiles[self._get_index(self.profile_id)]:
            return self.profiles[self._get_index(self.profile_id)][field]
        else:
            return None

    def get_id(self):
        return self.profile_id

    def get_len(self):
        return 5

