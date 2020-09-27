import jsonlines
import csv
import difflib
import os
import pickle
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
import time

def list_add(a,b):
	c = []
	for i in range(len(a)):
		c.append(a[i]+b[i])
	return c


def read_newsroom_human_score(human_score_file):

	system_types = ['fragments', 'textrank', 'abstractive', 
					'pointer_c', 'pointer_n', 'pointer_s','lede3']
	num_systems = len(system_types)

	human_scores = {}  
	summaries = {}
	id_to_title = {}
	with open(human_score_file)as f:
		f_csv = csv.reader(f)
		header = next(f_csv)
		for row in f_csv:
			article_id, system, article, summary, title, coherence, fluency, informativeness, relevent = row
			current_scores = [coherence,fluency,informativeness, relevent]
			current_scores = [float(sc) for sc in current_scores]
			system_index = system_types.index(system)

			if article_id not in summaries:
				summaries[article_id] = [None]*num_systems
			summary = summary.replace('&#x27;','\'')
			summary = summary.replace('&#x27','\'')
			summaries[article_id][system_index] = summary

			temp_score = [0]*4
			if article_id not in human_scores:
				human_scores[article_id] = [temp_score]* num_systems
			human_scores[article_id][system_index] = list_add(human_scores[article_id][system_index], current_scores)
			title = title.replace('&#x27','\'')
			if article_id not in id_to_title:
				for substr in ['&amp', ';']:
					title = title.replace(substr,'')
				id_to_title[article_id] = title

	for article_id in human_scores:
		temp_score = human_scores[article_id]
		human_scores[article_id] = [[sc/3 for sc in scores] for scores in temp_score]

	return human_scores, summaries, id_to_title



def read_newsroom_articles_and_reference(dataset_dir, articles_file, id_to_title):
	def get_match_rate(str1, str2): 
		return difflib.SequenceMatcher(None, str1, str2).quick_ratio()

	def find_matched_titleID(title, title_to_id):
		title_list = list(title_to_id.keys())

		for t in title_list:

			if(get_match_rate(title, t)>0.90):
				print('-'*100)
				print(title)
				print(t, get_match_rate(title, t))
				print('-'*100)
# 
				return title_to_id[t]
		return None


	title_to_id = {v:k for k,v in id_to_title.items()}
	
	index = 0
	articles = {}
	references = {}

	articles_cache_file = dataset_dir + 'articles_with_id'   
	references_cache_file = dataset_dir + 'references_with_id'  
	if os.path.exists(articles_cache_file) and os.path.exists(references_cache_file):
		with open(articles_cache_file, 'rb') as handle:
			articles = pickle.load(handle)
		with open(references_cache_file, 'rb') as handle:
			references = pickle.load(handle)

	else:

		with open(articles_file, "r+", encoding="utf8") as f:
			for item in jsonlines.Reader(f):	
				title = item['title']
				article = item["text"]
				reference = item["summary"]


				matched_id_in_human_score = find_matched_titleID(title, title_to_id)
				if matched_id_in_human_score != None:
					article_id = matched_id_in_human_score
				else:
					while(str(index) in id_to_title): 
						index += 1
					article_id = str(index)
					index += 1

				articles[article_id] = article
				references[article_id] = reference


		with open(articles_cache_file, 'wb') as handle:
			pickle.dump(articles, handle, protocol=pickle.HIGHEST_PROTOCOL)	
		with open(references_cache_file, 'wb') as handle:
			pickle.dump(references, handle, protocol=pickle.HIGHEST_PROTOCOL)	

	return articles, references



def read_newsroom(dataset_dir):

	articles_file = dataset_dir + 'test-stats.jsonl'
	human_score_file = dataset_dir + 'human-eval.csv' 
	human_scores, summaries_of_human_scores, id_to_title = read_newsroom_human_score(human_score_file)
	articles, references = read_newsroom_articles_and_reference(dataset_dir, articles_file, id_to_title)

	references_of_human_scores = {}
	for article_id in summaries_of_human_scores:
		references_of_human_scores[article_id] = references[article_id]

	return articles, references, human_scores


