from mrjob.job import MRJob
import re
import chardet

pattern1 = r"[0-9]*,*[0-9]+,[0-9]{3}" # number of bytes
pattern2 = r"\.[a-z]+" # file type
pattern3 = r" [^ ]+\.[a-z]+$" # file name

def str2int(input):
	a = "".join(input.split(','))
	return int(a)

class CountWorker(MRJob):
	""" Count the number of each type of files. """

	def mapper(self, file_name, line):
		""" Input Context: Date, Time, #Bytes, Name """
		file_type = re.search(pattern2, line).group()[1:]
		yield file_type, 1

	def reducer(self, key, values):
		yield key, sum(values)


class RankWorker(MRJob):
	""" Ranking in the descending order of #Bytes. """

	def mapper(self, file_name, line):
		""" Input Context: Date, Time, #Bytes, Name """
		num_bytes = str2int(re.search(pattern1, line).group())
		Name = re.search(pattern3, line).group()[1:]
		yield 1, (num_bytes, Name)

	def reducer(self, key, values):
		values = sorted(values)
		for num_bytes, Name in values[::-1]:
			yield num_bytes, Name

def deal_with_output(file_path):
	lines = []
	with open(file_path, "r") as f:
		while 1:
			line = f.readline()
			if not line: break
			tmp = line.strip().split("\t")
			
			num_bytes = int(tmp[0])
			name = "".join([tmp[1][1:-1]])
			
			print(num_bytes, name.encode("iso-8859-1").decode("gb18030"))

if __name__ == '__main__':
	#CountWorker.run()
	#RankWorker.run()
	deal_with_output("./out2.txt")

