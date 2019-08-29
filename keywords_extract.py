# extract the keywords of privacy policy

file_dir = "./dataset/keywords_filter.txt"
keywords_file = open(file_dir)
keywords = keywords_file.read()
result = set()
for key in keywords.split(','):
    result.add(key.strip().lower())
f = open("./dataset/attention_score.txt", "w")
f.write('\n'.join(result))
