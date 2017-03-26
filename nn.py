import word2vec
import string

textfile = 'data/machine_learning'
input_file = textfile+'-phrases'

f = open(textfile, 'r+')
data = f.read().lower().translate(None, string.punctuation+'0123456789')
f.seek(0)
f.write(data)
f.close()

# create phrases that could be better input (e.g. 'computer science' -> 'computer_science')
word2vec.word2phrase(textfile, textfile+'-phrases', verbose=True)

# training: create word vectors in binary form
word2vec.word2vec(textfile+'-phrases', textfile+'.bin', size=100, verbose=True)

# clustering: cluster vectors based on trained model
word2vec.word2clusters(input_file, textfile+'-clusters', 100, verbose=True)


# predicting...

# print matrix size, |V| = vocab size
model = word2vec.load(textfile+'.bin')
print "\n\nEmbedding matrix (|V| x n) :", model.vectors.shape

# find similar words (cosine similarity)
indexes, metrics = model.cosine('program')
similar_words = model.generate_response(indexes, metrics).tolist()
print "\n\nFind similar words to 'program': "
for word in similar_words:
	print word

# find analogy, e.g. king - man + woman = queen

