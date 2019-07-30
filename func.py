import os
import nltk
import re
import numpy as np
import pandas as pd
from collections import Counter

def gettext(filename):
    handle = open(filename, 'r', encoding='UTF-8')
    text = handle.read()
    handle.close()
    return text


def wordlemmatize(tags):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    words_l = []
    for tag in tags:
        if tag[1] and tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
            words_l.append(lemmatizer.lemmatize(tag[0], pos='n'))
        elif tag[1] and tag[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            words_l.append(lemmatizer.lemmatize(tag[0], pos='v'))
        elif tag[1] and tag[1] in ['JJ', 'JJR', 'JJS']:
            words_l.append(lemmatizer.lemmatize(tag[0], pos='a'))
        elif tag[1] and tag[1] in ['RB', 'RBR', 'RBS']:
            words_l.append(lemmatizer.lemmatize(tag[0], pos='r'))
        else:
            words_l.append(tag[0])
    return words_l


def F_measure(tags_s):
    Flist = {
        'noum': ['NN', 'NNS', 'NNP', 'NNPS'],
        'adj': ['JJ', 'JJR', 'JJS'],
        'prep': ['IN'],
        'art': ['CD'],
        'pron': ['PRP', 'WP'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adv': ['RB', 'RBR', 'RBS'],
        'int': ['UH']
    }
    tagshapelist = [0] * len(tags_s)

    for i, tag in enumerate(tags_s):
        tagshapelist[i] = tag[1]
    ff = dict(Counter(tagshapelist))

    a = 0
    b = 0
    for i, poslist in enumerate(Flist.values()):
        if i < 4:
            for pos in poslist:
                if pos in ff.keys():
                    a += ff[pos]
        if i >= 4:
            for pos in poslist:
                if pos in ff.keys():
                    b += ff[pos]
    F_feature = 0.5 * (a - b + 100)
    return F_feature


def Gender_Preferential_Features(words_l):
    GPFlist = ['able', 'al', 'ful', 'ible', 'ic', 'ive', 'less', 'ly', 'ous']
    GPF_feature = [0] * (len(GPFlist) + 1)

    for i, trigger in enumerate(GPFlist):
        flag = [0] * len(words_l)
        for j, word in enumerate(words_l):
            flag[j] = word.endswith(trigger)
        GPF_feature[i] = sum(flag)
    GPF_feature[-1] = words.count('sorry') + words.count('sry')
    if sum(GPF_feature) != 0:
        GPF_feature = np.array(GPF_feature) / sum(GPF_feature)
    return GPF_feature


def Word_Classes_Feature(words_l):
    classlist = {
        'Conversation': [
            'know', 'people', 'think', 'person', 'tell', 'feel', 'friends',
            'talk', 'new', 'talking', 'mean', 'ask', 'understand', 'feelings',
            'care', 'thinking', 'friend', 'relationship', 'realize',
            'question', 'answer', 'saying'
        ],
        'AtHome': [
            'woke', 'home', 'sleep', 'today', 'eat', 'tired', 'wake', 'watch',
            'watched', 'dinner', 'ate', 'bed', 'day', 'house', 'tv', 'early',
            'boring', 'yesterday', 'watching', 'sit'
        ],
        'Family': [
            'years', 'family', 'mother', 'children', 'father', 'kids',
            'parents', 'old', 'year', 'child', 'son', 'married', 'sister',
            'dad', 'brother', 'moved', 'age', 'young', 'months', 'three',
            'wife', 'living', 'college', 'four', 'high', 'five', 'died', 'six',
            'baby', 'boy', 'spend', 'christmas'
        ],
        'Time': [
            'friday', 'saturday', 'weekend', 'week', 'sunday', 'night',
            'monday', 'tuesday', 'thursday', 'wednesday', 'morning',
            'tomorrow', 'tonight', 'evening', 'days', 'afternoon', 'weeks',
            'hours', 'july', 'busy', 'meeting', 'hour', 'month', 'june'
        ],
        'Work': [
            'work', 'working', 'job', 'trying', 'right', 'met', 'figure',
            'meet', 'start', 'better', 'starting', 'try', 'worked', 'idea'
        ],
        'PastActions': [
            'said', 'asked', 'told', 'looked', 'walked', 'called', 'talked',
            'wanted', 'kept', 'took', 'sat', 'gave', 'knew', 'felt', 'turned',
            'stopped', 'saw', 'ran', 'tried', 'picked', 'left', 'ended'
        ],
        'Games': [
            'game', 'games', 'team', 'win', 'play', 'played', 'playing', 'won',
            'season', 'beat', 'final', 'two', 'hit', 'first', 'video',
            'second', 'run', 'star', 'third', 'shot', 'table', 'round', 'ten',
            'chance', 'club', 'big', 'straight'
        ],
        'Internet': [
            'site', 'email', 'page', 'please', 'website', 'web', 'post',
            'link', 'check', 'blog', 'mail', 'information', 'free', 'send',
            'comments', 'comment', 'using', 'internet', 'online', 'name',
            'service', 'list', 'computer', 'add', 'thanks', 'update', 'message'
        ],
        'Location': [
            'street', 'place', 'town', 'road', 'city', 'walking', 'trip',
            'headed', 'front', 'car', 'beer', 'apartment', 'bus', 'area',
            'park', 'building', 'walk', 'small', 'places', 'ride', 'driving',
            'looking', 'local', 'sitting', 'drive', 'bar', 'bad', 'standing',
            'floor', 'weather', 'beach', 'view'
        ],
        'Fun': [
            'fun', 'im', 'cool', 'mom', 'summer', 'awesome', 'lol', 'stuff',
            'pretty', 'ill', 'mad', 'funny', 'weird'
        ],
        'Food/Clothes': [
            'food', 'eating', 'weight', 'lunch', 'water', 'hair', 'life',
            'white', 'wearing', 'color', 'ice', 'red', 'fat', 'body', 'black',
            'clothes', 'hot', 'drink', 'wear', 'blue', 'minutes', 'shirt',
            'green', 'coffee', 'total', 'store', 'shopping'
        ],
        'Poetic': [
            'eyes', 'heart', 'soul', 'pain', 'light', 'deep', 'smile',
            'dreams', 'dark', 'hold', 'hands', 'head', 'hand', 'alone', 'sun',
            'dream', 'mind', 'cold', 'fall', 'air', 'voice', 'touch', 'blood',
            'feet', 'words', 'hear', 'rain', 'mouth'
        ],
        'Books/Movies': [
            'book', 'read', 'reading', 'books', 'story', 'writing', 'written',
            'movie', 'stories', 'movies', 'film', 'write', 'character', 'fact',
            'thoughts', 'title', 'short', 'take', 'wrote'
        ],
        'Religion': [
            'god', 'jesus', 'lord', 'church', 'earth', 'world', 'word',
            'lives', 'power', 'human', 'believe', 'given', 'truth', 'thank',
            'death', 'evil', 'own', 'peace', 'speak', 'bring', 'truly'
        ],
        'Romance': [
            'forget', 'forever', 'remember', 'gone', 'true', 'face', 'spent',
            'times', 'love', 'cry', 'hurt', 'wish', 'loved'
        ],
        'Swearing': [
            'shit', 'fuck', 'fucking', 'ass', 'bitch', 'damn', 'hell', 'sucks',
            'stupid', 'hate', 'drunk', 'crap', 'kill', 'guy', 'gay', 'kid',
            'sex', 'crazy'
        ],
        'Politics': [
            'bush', 'president', 'Iraq', 'kerry', 'war', 'american',
            'political', 'states', 'america', 'country', 'government', 'john',
            'national', 'news', 'state', 'support', 'issues', 'article',
            'michael', 'bill', 'report', 'public', 'issue', 'history', 'party',
            'york', 'law', 'major', 'act', 'fight', 'poor'
        ],
        'Music': [
            'music', 'songs', 'song', 'band', 'cd', 'rock', 'listening',
            'listen', 'show', 'favorite', 'radio', 'sound', 'heard', 'shows',
            'sounds', 'amazing', 'dance'
        ],
        'School': [
            'school', 'teacher', 'class', 'study', 'test', 'finish', 'english',
            'students', 'period', 'paper', 'pass'
        ],
        'Business': [
            'system', 'based', 'process', 'business', 'control', 'example',
            'personal', 'experience', 'general'
        ],
        'Positive': [
            'absolutely', 'abundance', 'ace', 'active', 'admirable', 'adore',
            'agree', 'amazing', 'appealing', 'attraction', 'bargain',
            'beaming', 'beautiful', 'best', 'better', 'boost', 'breakthrough',
            'breeze', 'brilliant', 'brimming', 'charming', 'clean', 'clear',
            'colorful', 'compliment', 'confidence', 'cool', 'courteous',
            'cuddly', 'dazzling', 'delicious', 'delightful', 'dynamic', 'easy',
            'ecstatic', 'efficient', 'enhance', 'enjoy', 'enormous',
            'excellent', 'exotic', 'expert', 'exquisite', 'flair', 'free',
            'generous', 'genius', 'great', 'graceful', 'heavenly', 'ideal',
            'immaculate', 'impressive', 'incredible', 'inspire', 'luxurious',
            'outstanding', 'royal', 'speed', 'splendid', 'spectacular',
            'superb', 'sweet', 'sure', 'supreme', 'terrific', 'treat',
            'treasure', 'ultra', 'unbeatable', 'ultimate', 'unique', 'wow',
            'zest'
        ],
        'Negative': [
            'wrong', 'stupid', 'bad', 'evil', 'dumb', 'foolish', 'grotesque',
            'harm', 'fear', 'horrible', 'idiot', 'lame', 'mean', 'poor',
            'heinous', 'hideous', 'deficient', 'petty', 'awful', 'hopeless',
            'fool', 'risk', 'immoral', 'risky', 'spoil', 'spoiled', 'malign',
            'vicious', 'wicked', 'fright', 'ugly', 'atrocious', 'moron',
            'hate', 'spiteful', 'meager', 'malicious', 'lacking'
        ],
        'Emotion': [
            'aggressive', 'alienated', 'angry', 'annoyed', 'anxious',
            'careful', 'cautious', 'confused', 'curious', 'depressed',
            'determined', 'disappointed', 'discouraged', 'disgusted',
            'ecstatic', 'embarrassed', 'enthusiastic', 'envious', 'excited',
            'exhausted', 'frightened', 'frustrated', 'guilty', 'happy',
            'helpless', 'hopeful', 'hostile', 'humiliated', 'hurt',
            'hysterical', 'innocent', 'interested', 'jealous', 'lonely',
            'mischievous', 'miserable', 'optimistic', 'paranoid', 'peaceful',
            'proud', 'puzzled', 'regretful', 'relieved', 'sad', 'satisfied',
            'shocked', 'shy', 'sorry', 'surprised', 'suspicious', 'thoughtful',
            'undecided', 'withdrawn'
        ]
    }
    WC_feature = [0] * len(classlist)

    for i, ws in enumerate(classlist.values()):
        for w in ws:
            WC_feature[i] += words_l.count(w)
    if sum(WC_feature) != 0:
        WC_feature = np.array(WC_feature) / sum(WC_feature)
    return WC_feature


def CorpusPOS(sentences):
    posTagList = [
        'NN', 'CC', 'LS', 'PDT', 'POS', 'SYM', 'NNS', 'NNP', 'NNPS', 'FW',
        'CD', 'JJ', 'JJR', 'JJS', 'IN', 'TO', 'DT', 'EX', 'PRP', 'PRP$', 'WDT',
        'WP', 'WP$', 'MD', 'VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'RB',
        'RBR', 'RBS', 'RP', 'WRB', 'UH', '.'
    ]
    outfile = open('CorpusPOS.txt', 'a')
    for sentence in sentences:
        tagSentence = ""
        tokensWord = nltk.word_tokenize(sentence)
        textToken = nltk.Text(tokensWord)
        tags = nltk.pos_tag(tokensWord)

        for a, b in tags:
            if b in posTagList:
                tagSentence = tagSentence + b + " "
        tagSentence = tagSentence + "\n"
        outfile.write(tagSentence)

    outfile.close()


def calc_probabilities(cPOS):
    from nltk import ngrams

    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    fourgram_p = {}
    fivegram_p = {}
    sixgram_p = {}
    sevengram_p = {}

    unigram = {}
    bigram = {}
    trigram = {}
    fourgram = {}
    fivegram = {}
    sixgram = {}
    sevengram = {}
    uni_count = biCount = triCount = fourCount = fiveCount = sixCount = sevenCount = 0

    for sentence in cPOS:
        tokens = sentence.split()

        for word in tokens:
            uni_count += 1

            if word in unigram:
                unigram[word] += 1
            else:
                unigram[word] = 1

        bigram_tuples = tuple(nltk.bigrams(tokens))
        for item in bigram_tuples:
            biCount += 1
            if item in bigram:
                bigram[item] += 1
            else:
                bigram[item] = 1

        trigram_tuples = tuple(nltk.trigrams(tokens))
        for item in trigram_tuples:
            triCount += 1
            if item in trigram:
                trigram[item] += 1
            else:
                trigram[item] = 1

        fourgram_tuples = ngrams(tokens, 4)
        for item in fourgram_tuples:
            fourCount += 1
            if item in fourgram:
                fourgram[item] += 1
            else:
                fourgram[item] = 1

        fivegram_tuples = ngrams(tokens, 5)
        for item in fivegram_tuples:
            fiveCount += 1
            if item in fivegram:
                fivegram[item] += 1
            else:
                fivegram[item] = 1

        sixgram_tuples = ngrams(tokens, 6)
        for item in sixgram_tuples:
            sixCount += 1
            if item in sixgram:
                sixgram[item] += 1
            else:
                sixgram[item] = 1

        sevengram_tuples = ngrams(tokens, 7)
        for item in sevengram_tuples:
            sevenCount += 1
            if item in sevengram:
                sevengram[item] += 1
            else:
                sevengram[item] = 1

    # calculate unigram probability
    for word in unigram:
        temp = [word]
        unigram_p[tuple(temp)] = (float(unigram[word]) / uni_count)

    # calculate bigram probability
    for word in bigram:
        bigram_p[tuple(word)] = (float(bigram[word]) / biCount)

    # calculate trigram probability
    for word in trigram:
        trigram_p[tuple(word)] = (float(trigram[word]) / triCount)

    # calculate fourgram probability
    for word in fourgram:
        fourgram_p[tuple(word)] = (float(fourgram[word]) / fourCount)

    # calculate fivegram probability
    for word in fivegram:
        fivegram_p[tuple(word)] = (float(fivegram[word]) / fiveCount)

    for word in sixgram:
        sixgram_p[tuple(word)] = (float(sixgram[word]) / sixCount)

    for word in sevengram:
        sevengram_p[tuple(word)] = (float(sevengram[word]) / sevenCount)

    return unigram_p, bigram_p, trigram_p, fourgram_p, fivegram_p, sixgram_p, sevengram_p


def q1_output(unigrams, bigrams, trigrams, fourgrams, fivegrams, sixgrams,
              sevengrams):
    #output probabilities
    outfile = open('probabilities.txt', 'a')
    for unigram in unigrams:
        outfile.write(unigram[0] + ':' + str(unigrams[unigram]) + '\n')
    for bigram in bigrams:
        outfile.write(bigram[0] + ' ' + bigram[1] + ':' +
                      str(bigrams[bigram]) + '\n')
    for trigram in trigrams:
        outfile.write(trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ':' +
                      str(trigrams[trigram]) + '\n')

    for fourgram in fourgrams:
        outfile.write(fourgram[0] + ' ' + fourgram[1] + ' ' + fourgram[2] +
                      ' ' + fourgram[3] + ':' + str(fourgrams[fourgram]) +
                      '\n')

    for fivegram in fivegrams:
        outfile.write(fivegram[0] + ' ' + fivegram[1] + ' ' + fivegram[2] +
                      ' ' + fivegram[3] + ' ' + fivegram[4] + ':' +
                      str(fivegrams[fivegram]) + '\n')

    for sixgram in sixgrams:
        outfile.write(sixgram[0] + ' ' + sixgram[1] + ' ' + sixgram[2] + ' ' +
                      sixgram[3] + ' ' + sixgram[4] + ' ' + sixgram[5] + ':' +
                      str(sixgrams[sixgram]) + '\n')

    for sevengram in sevengrams:
        outfile.write(sevengram[0] + ' ' + sevengram[1] + ' ' + sevengram[2] +
                      ' ' + sevengram[3] + ' ' + sevengram[4] + ' ' +
                      sevengram[5] + ' ' + sevengram[6] + ':' +
                      str(sevengrams[sevengram]) + '\n')

    outfile.close()


def prob(sequence):
    if sequence in Prob.keys():
        return Prob[sequence]
    else:
        return 0


def fairSCP(sequence):
    numerator = prob(sequence) * prob(sequence)
    sequence = sequence.split()

    denominator = 0

    for j in range(1, len(sequence)):
        seq1 = ""
        seq2 = ""
        cnt = 1

        for tag in sequence:
            if cnt <= j:
                seq1 = seq1 + tag + " "
                cnt += 1
            else:
                seq2 = seq2 + tag + " "

        seq2 = seq2[:-1]
        seq1 = seq1[:-1]

        denominator += prob(seq1) * prob(seq2)

    denominator = denominator * 1.0 / (len(sequence) - 1)

    if denominator == 0:
        return 0.0

    SCP = numerator * 1.0 / denominator

    return SCP


def candidateGen(Fk):
    Ck = []

    for item in Fk:
        for tag in tagList:
            itemTemp = item + " " + tag
            Ck.append(itemTemp)

    return Ck


def minePOSPats(cPOS):
    minSup = 0.3
    minAdherence = 0.2
    C = [{} for i in range(7)]
    F = [[] for i in range(7)]
    SP = [[] for i in range(7)]
    Cand = [[] for i in range(7)]

    Doc = cPOS
    n = len(Doc)

    for post in Doc:
        for tag in tagList:
            if tag in post:
                if tag in C[0].keys():
                    C[0][tag] += 1
                else:
                    C[0][tag] = 1

    for a in C[0]:
        if C[0][a] * 1.0 / n >= minSup:
            F[0].append(a)

    SP[0] = F[0]
    temp = {}
    for k in range(1, 7):
        Cand[k] = candidateGen(F[k - 1])
        for post in Doc:
            for candidate in Cand[k]:
                if candidate in post:
                    if candidate in C[k].keys():
                        C[k][candidate] += 1
                    else:
                        C[k][candidate] = 1

        for a in C[k]:
            if C[k][a] * 1.0 / n >= minSup:
                F[k].append(a)

        for a in F[k]:
            if fairSCP(a) >= minAdherence:
                SP[k].append(a)

    SPFinal = []
    SPFinal = SP[0] + SP[1] + SP[2] + SP[3] + SP[4] + SP[5] + SP[6]

    return SPFinal


def getsingle(features, n):
    single = []
    for item in features:
        single.append(item[n])
    return single
