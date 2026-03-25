def model_function(text):
    if "place" in text.lower():
        return "pozitiv"
    elif "nu" in text.lower():
        return "negativ"
    else:
        return "neutru"
