import json
import os

PROFILES_PATH=os.path.join(os.path.dirname(__file__),"..","profiles.json")


def _load_all():
    if not os.path.isfile(PROFILES_PATH):
        return {}
    with open(PROFILES_PATH,"r",encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_all(profiles):
    with open(PROFILES_PATH,"w",encoding="utf-8") as f:
        json.dump(profiles,f,ensure_ascii=False,indent=2)


def get_profile(name):
    profiles=_load_all()
    key=name.strip().lower()
    return profiles.get(key)


def save_profile(name,profile_dict):
    profiles=_load_all()
    key=name.strip().lower()
    profile_dict["display_name"]=name.strip()
    profiles[key]=profile_dict
    _save_all(profiles)


def list_names():
    profiles=_load_all()
    return [p.get("display_name",k) for k,p in profiles.items()]