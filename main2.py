import nltk
from nltk.stem.porter import PorterStemmer
from stanfordcorenlp import StanfordCoreNLP

#nltk.download()
corpus = "dict/emotion-lexicon.txt"
rel_attr = ["sadness", "eff", "fear", "disgust"]
victim_cat = {'not_victim':0, 'maybe':1, 'likely':2, 'confident':3, 'very_confident':4}
self_words = ['i', 'me', 'our', 'we', 'myself', 'mine']
other_words = ['you', 'u', 'ur', 'your', 'yourself', 'he', 'she', 'they', 'themselves']
th_off1 = 0.5
th_off2 = 4.0
th_vic = 30
porter_stemmer = PorterStemmer()

class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

class POSTagger(object):
    def __init__(self):
        pass
        
    def pos_tag(self, sentences):
        """
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        """

        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        #adapt format
        pos = [[(word, stem(word), [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos

class DictionaryTagger(object):
    def __init__(self, dictionary_path):
        rows = open(dictionary_path, 'r').readlines()
        rows = [r.strip().split('\t') for r in rows]
#         for x in rows:
#             if len(x) < 3:
#                 print x
        
        rows = [[stem(x[0]), x[1], x[2]] for x in rows ]
        self.dictionary = {}
        self.max_key_size = 0
        for line in rows:
            if int(line[2]) == 1 and line[1] in rel_attr or line[1] == 'sup':
                if line[0] in self.dictionary:
                    self.dictionary[line[0]].extend([line[1]])
                else:
                    self.dictionary[line[0]] = [line[1]]
                    self.max_key_size = max(self.max_key_size, len(line[0]))
    
    def tag(self, postagged_sentence):
        return self.tag_sentence(postagged_sentence, True)
                    
    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        """
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence

def preprocess(text):
    splitted_sentences = splitter.split(text)
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)    #also does lammentization
    return pos_tagged_sentences 

def stem(word):
    return porter_stemmer.stem(word).encode('ascii','ignore')

def effWords(dict_tagged_sentence):
    isFlagged = 0
    rel_ind = []
    i = 0
    for entry in dict_tagged_sentence:
        i=i+1
        if 'sup' in entry[2]:
            isFlagged = 1
            rel_ind.extend([[i,entry[2][-1],'sup']])
        if set(rel_attr).intersection(entry[2]):
            isFlagged = 1
            rel_ind.extend([[i, entry[2][-1], 'eff']])
    return isFlagged, rel_ind

def checkDependence(index, rel_type, parse_stan, pos):
    for i in range(len(parse_stan)):
        if parse_stan[i][0].encode('ascii','ignore') in rel_type:
            if parse_stan[i][pos] == index:
                return i
    return -1
            
    
if __name__ == "__main__":
    splitter = Splitter()
    postagger = POSTagger()
#     nlp = StanfordCoreNLP('http://corenlp.run', port=80)
    print "ok"
#     nlp = StanfordCoreNLP(r'/Users/bhavishyamittal/Desktop/hack/stanford-corenlp/')
    nlp = StanfordCoreNLP(r'../stanford-corenlp/')
    print "ok2"
    dicttagger = DictionaryTagger(corpus)
    while(1):
        victim_score = 0
        text = raw_input('sample text -> ')
        prev_off_score = float(raw_input('prev off score -> ' ))
        curr_off_score = float(raw_input('curr off score -> ' ))
        splitted_sentences = splitter.split(text)
        
        for sent in splitted_sentences:
            sent_raw = ' '.join(sent)
            print sent_raw
            pos = nlp.pos_tag(sent_raw)
            print pos
            pos = [(word.encode('ascii','ignore'), stem(word), [postag.encode('ascii','ignore')]) for (word, postag) in pos]
            dict_tagged_sentence = dicttagger.tag(pos)
            print dict_tagged_sentence
            parse_stan = nlp.dependency_parse(sent_raw)
            print parse_stan
            
            isFlagged, rel_ind = effWords(dict_tagged_sentence)
            print "rel_ind = ", rel_ind, isFlagged
            if isFlagged:
                #check rel_ind and see
                #rule 1: eff_JJ/VB --> (subj/obj) --> self : Kill me, I am useless, ...
                #rule 2: sup_MD --> (aux, dep) --> JJ? --> (subj) --> self : I should ...,
                #rule 4: sup_MD --> (aux, dep) --> VB?/JJ --> (neg) --> (subj) --> other
                for entry in rel_ind:
                    print "Processing entry: ", entry
                    if entry[2] == 'eff':   #rule 1
                        print "has eff"
                        rel_stan_ind = checkDependence(entry[0], ['nsubj', 'agent', 'dobj', 'iobj'], parse_stan, 1)
                        if rel_stan_ind != -1:
                            print 'rel_eff->', parse_stan[rel_stan_ind]
                            if dict_tagged_sentence[parse_stan[rel_stan_ind][2] - 1][0].lower() in self_words:
                                victim_score = victim_score + 50;
                    
                    if entry[2] == 'sup' and prev_off_score > th_off1:
                        rel_stan_ind = checkDependence(entry[0], ['aux', 'dep'], parse_stan, 2)
                        if rel_stan_ind != -1:
                            print 'rel_sup->', parse_stan[rel_stan_ind]
                            verb_index = parse_stan[rel_stan_ind][1]
                            neg_flag = checkDependence(verb_index, ['neg'], parse_stan, 1)
                            if neg_flag != -1:
                                print 'rel_sup_neg->', parse_stan[neg_flag]
                                sub_stan_ind = checkDependence(verb_index, ['nsubj', 'agent'], parse_stan, 1)
                                if sub_stan_ind != -1:
                                    print 'rel_sup_sub->', parse_stan[sub_stan_ind]
                                    if dict_tagged_sentence[parse_stan[sub_stan_ind][2] - 1][0].lower() in other_words:
                                        victim_score = victim_score + 20
                                    else:
                                        #subj is entirely different
                                        victim_score = victim_score + 10
                                else:
                                    victim_score = victim_score + 10
                            else:
                                #I should go home, #I should....
                                victim_score = victim_score + 10
                        else:
                            #this scenaario should not happen
                            pass
                
                print victim_score
                if victim_score > th_vic:
                    if prev_off_score > th_off1:
                        print 'confident'
                    else:
                        print 'likely'
            else:
                if prev_off_score/curr_off_score < th_off2:
                    print 'not_victim'
#                     return victim_cat['not_victim']
                else:
                    print 'likely'
#                     return victim_cat['likely']
                        
            
