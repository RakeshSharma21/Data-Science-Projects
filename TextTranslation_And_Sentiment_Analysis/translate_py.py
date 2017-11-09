
from yandex_translate import YandexTranslate

def translate_language(sourcetext,translationfrom,translationTo):
    translate = YandexTranslate('trnsl.1.1.20171108T161528Z.11fd5cf559085f7e.68f45fa90a6197a56fc8353410bec46f7386e419')
    print('Languages:', translate.langs)
    print('Translate directions:', translate.directions)
    print('Detect language:', translate.detect(sourcetext))
    translated_text= translate.translate(sourcetext, translationfrom+'-'+translationTo)
    arr=translated_text['text'];
    return arr
    print (arr)


from nltk.sentiment.vader import SentimentIntensityAnalyzer
#sentences=[]
from nltk import tokenize
#lines_list = tokenize.sent_tokenize(translated_text)
#sentences.extend(lines_list)
def SentimentAnalyzer(analysinglevel,data):
    sid = SentimentIntensityAnalyzer()
    if analysinglevel=='File':
        for sentence in data:
             print('\n',sentence,'\n')
             ss = sid.polarity_scores(sentence)
             for k in ss:
                 print('{0}: {1},\n '.format(k, ss[k]), end='')
    elif analysinglevel=='Line':
        for sentence in data:
            for sen in sentence.split(","):               
                 print('\n',sen,'\n')
                 ss = sid.polarity_scores(sen)
                 for k in ss:
                     print('{0}: {1},\n '.format(k, ss[k]), end='')
    elif analysinglevel=='Token':
        lines_list = tokenize.sent_tokenize(",".join(str(x) for x in data))
        for sentence in lines_list:
             print('\n',sentence,'\n')
             ss = sid.polarity_scores(sentence)
             for k in ss:
                 print('{0}: {1}, \n'.format(k, ss[k]), end='')
                 
chineese_text = open('chineese.txt', 'r', encoding="utf8").read().replace('\n', ',')
arr=translate_language(chineese_text,'zh','en')

SentimentAnalyzer('Line',arr)

