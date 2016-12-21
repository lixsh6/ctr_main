from __future__ import print_function
import time
import csv
import numpy as np
import numpy.linalg
import scipy.optimize
from six.moves import xrange

from collabotm import *
''' #yelp2013
num_topics		= 20
n_voca			= 89694
n_users			= 1631
n_items			= 1633
'''

#imdb
num_topics		= 20
n_voca			= 89652
n_users			= 1310
n_items			= 1635

dataName 		= 'imdb'
rootDir			= '../data/'+dataName+'/'
docFile			= rootDir + dataName + '_train_mul.dat'
userInfoFile	= rootDir + dataName + '_train_user-info.dat'
TestInfoFile	= rootDir + dataName +'_test_user-info.dat'

def readDocuments(filename):
	doc_ids = [];doc_cnts = [];
	fin = open(filename,'r')
	for line in fin:
		doc_id = [];doc_cnt = [];
		for element in line.split(' ')[1:]:
			word,count = map(eval,element.split(':'))
			doc_id.append(word)
			doc_cnt.append(count)
		doc_ids.append(doc_id)
		doc_cnts.append(doc_cnt)
	return doc_ids,doc_cnts

'''
Rating:
[0]: doc_id
[1]: user_marked
'''

def readRating(filename):
	fin = open(filename,'r')
	ratings = []

	for row in fin:
		elements = row.split('\t\t')
		elements = map(eval,elements)
		ratings.append((elements[0],elements[1],elements[2]))
	return ratings


def main():	
	print ('Start readRating...')
	ratings = readRating(userInfoFile)
	print ('Start readDocuments...')
	doc_ids,doc_cnts = readDocuments(docFile)
	#print (doc_ids,doc_cnts)
	print ('Start creating model...')
	CTR = CollaborativeTopicModel(num_topics,n_voca,n_users, n_items, doc_ids, doc_cnts, ratings)
	print ('Import testfile...')
	CTR.importTestFile(TestInfoFile)

	print ('Start fitting...')
	CTR.fit(100)
	print ('Testing')
	#CTR.test(TestInfoFile)

if __name__ == '__main__':
	main()




