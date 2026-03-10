#POS Tagging
import nltk  
nltk.download('punkt')  
nltk.download('averaged_perceptron_tagger_eng')  
nltk.download('maxent_ne_chunker_tab')  
nltk.download('words')

from nltk import pos_tag  
from nltk.tokenize import word_tokenize  
text = "The hungry grey cat runs after the little mouse."  
tokens = word_tokenize(text)  
tagged = pos_tag(tokens)  
print(tagged)

from nltk import ne_chunk  
from nltk.tokenize import word_tokenize  
text = "vincent van gogh was born in Netherlands."  
tokens = word_tokenize(text)  
tagged = pos_tag(tokens)  
entities = ne_chunk(tagged)  
print(entities)

def extract_features(word, index, sentence):
    features = {
        'word': word,
        'is_capitalized': word[0].upper() == word[0],
        'prev_word': sentence[index-1] if index > 0 else None,
        'next_word': sentence[index+1] if index < len(sentence)-1 else None,
    }
    return features

train_sentence = word_tokenize("Vincent Van Gogh was born in the Netherlands")

training_data = [
    (extract_features('Vincent', 0, train_sentence), 'PERSON'),
    (extract_features('Van', 1, train_sentence), 'PERSON'),
    (extract_features('Gogh', 2, train_sentence), 'PERSON'),
    (extract_features('Netherlands', 7, train_sentence), 'GPE')
]

classifier = nltk.MaxentClassifier.train(training_data)
test_sentence = word_tokenize("Microsoft is a big company in Seattle.")
test_features = [extract_features(w, i, test_sentence) for i, w in enumerate(test_sentence)]
predicted_tags = classifier.classify_many(test_features)

print(predicted_tags)

