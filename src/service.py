def model_function(text):
    if "nu" in text.lower():
        return "negativ"
    elif "place" in text.lower():
        return "pozitiv"
    else:
        return "neutru"
