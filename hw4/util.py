import re

# Remove non alphebats and stopwords
def processStr(inputStr, stopwords):
    regex = re.compile("[^a-zA-Z\s]")
    inputStr = regex.sub("", inputStr).lower()
    words = inputStr.split(" ")
    return [w for w in words if w
            not in stopwords and w != ""]
