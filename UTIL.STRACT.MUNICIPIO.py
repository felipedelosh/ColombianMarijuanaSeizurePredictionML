"""
FelipedelosH
"""

_data = ""
_places_codes = {}

with open("INCAUTACIONES_DE_MARIHUANA_20241108.csv", "r", encoding="UTF-8") as f:
    _data = f.read()

# Only unique code values save
for i in _data.split("\n"):
    if str(i).strip() != "":
        _d = str(i).split(",")

        _cod = _d[3]
        _name = _d[4]

        if _name not in _places_codes.keys():
            _places_codes[_name] = _cod


# Prepare to save in municipios.txt
_output = ""
_counter = 0
for i in _places_codes:
    # Only valid places codes: int
    try:
        _int = int(_places_codes[i])
        _counter = _counter + 1
        _output = _output + f"{i}:{_places_codes[i]}\n"
    except:
        pass


_output = f"TOTAL: {_counter}\n" + _output

with open("municipios.txt", "w", encoding="UTF-8") as f:
    f.write(_output)
