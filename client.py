from bert_serving.client import BertClient
import numpy as np
bc = BertClient()
while True:
    sentence = input("input value: ")
    sentence.replace('\n', ' ')

    doc_vec = bc.encode([sentence])
    query = bc.encode([sentence])[0]
    score = np.sum(query*doc_vec, axis=1) / np.linalg.norm(doc_vec, axis=1)
    print(score)
#print(bc.encode(['안녕하세요']))
