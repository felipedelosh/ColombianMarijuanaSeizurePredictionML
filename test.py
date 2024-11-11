from datetime import datetime
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d-%H.%M")
_output_model_filename = f"model-{formatted_date}.h5"

print(_output_model_filename)

with open(_output_model_filename, "w") as f:
    f.write("Crazu")