import requests
import random
import json
from hashlib import md5
import time

#text = 'Hello World! This is 1st paragraph.'
#from_lang = 'en'
#dest =  'zh'

def translate(text, dest,src):
    src='auto'
#    print(text, dest, src)
    # Set your own appid/appkey.
    appid = '20200510000447341'
    appkey = '6sW6EaNF1uZzzvDnISBT'
    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path
    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + text + str(salt) + appkey)
    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': text, 'from': src, 'to': dest, 'salt': salt, 'sign': sign}
    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()
    # Show response
#    print(json.dumps(result, indent=4, ensure_ascii=False))
    time.sleep(1.1)
    return(result['trans_result'][0]['dst'])

#translate(text, dest, src='auto')

#translate(query,from_lang,to_lang)

#def translate(text, dest, src='auto'):
#    result = baidutranslate(text, src, dest)
#    time.sleep(10)
#    return result

#translate_client = translate.Client()
#translate('拟合', 'en', src='auto')

#def detect_lang(text):
#    """Detects the text's language."""
#
#    # Text can also be a sequence of strings, in which case this method
#    # will return a sequence of results for each text.
#    result = translate_client.detect_language(text)
#    return result["language"]


#def translate(text, dest, src=''):
#    """Translates text into the target language.
#
#    Target must be an ISO 639-1 language code.
#    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
#    """
#
#    if isinstance(text, six.binary_type):
#        text = text.decode("utf-8")
#
#    # Text can also be a sequence of strings, in which case this method
#    # will return a sequence of results for each text.
#    result = translate_client.translate(text, target_language=dest, source_language=src, model='base')
#    if result["translatedText"] == text:
#        result = translate_client.translate(text, target_language=dest)
#    return result["translatedText"]


#baidutranslate('hello','en','zh')
