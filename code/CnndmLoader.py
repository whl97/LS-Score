import jsonlines


def read_cnndm_human_score(human_score_file):
	human_scores = {}
	system_types = ['reference', 'ml', 'ml+rl', 'seq2seq', 'pointer']
	with open(human_score_file, "r+", encoding="utf8") as f:
		for item in jsonlines.Reader(f):
			if item['id'] not in human_scores:
				human_scores[item['id']] = [None]*5

			system_idx = system_types.index(item['system'])
			human_scores[item['id']][system_idx] = [item['prompts']['hter']['gold'], 
											item['prompts']['overall']['gold'], 
											item['prompts']['grammar']['gold'],
											item['prompts']['redundancy']['gold']]	

	return human_scores




def read_cnndm_summaries_and_reference(summaries_file):
	system_types = ['reference', 'ml', 'ml+rl', 'seq2seq', 'pointer']	

	references = {}
	summaries = {}
	with open(summaries_file, "r+", encoding="utf8") as f:
		for item in jsonlines.Reader(f):	
			if item['id'] not in summaries:
				summaries[item['id']] = [None]*5
			system_idx = system_types.index(item['system'])
			summaries[item['id']][system_idx] = item['text']

			if system_idx == 0:	
				references[item['id']] = item['text']
	summaries = dict((key, value) for key, value in summaries.items() if all(value) == True)
	return summaries, references


def read_cnndm_articles(articles_file):
	articles = {}
	with open(articles_file, "r+", encoding="utf8") as f:
		for item in jsonlines.Reader(f):
			articles[item['id']] = item['text']
	return articles


def read_cnndm(dataset_dir):

	human_score_file = dataset_dir + 'human_score.jsonl'
	summaries_file = dataset_dir + 'summary.jsonl'
	articles_file = dataset_dir + 'articles.jsonl'

	articles = read_cnndm_articles(articles_file)
	summaries, references = read_cnndm_summaries_and_reference(summaries_file)
	human_scores = read_cnndm_human_score(human_score_file)

	return articles, references, summaries, human_scores




