"""
FelipedelosH
"""

_data = ""
_places_codes = {}

with open("INCAUTACIONES_DE_MARIHUANA_20241108.csv", "r", encoding="UTF-8") as f:
    _data = f.read()

for i in _data.split("\n"):
    if str(i).strip() != "":
        _d = str(i).split(",")

        _cod = _d[3]
        _name = _d[4]

        if _name not in _places_codes.keys():
            _places_codes[_name] = _cod


print(_places_codes)
