def convertf(fn_in, fn_out, lbl):
	with open(fn_in, 'r') as in_:
		with open(fn_out, 'w') as out_:
			for line in in_:
				out_.write('%s,%d\n' % (line.rstrip(), lbl))
convertf('positive_train.csv', 'positive_train.csv2', 1)
convertf('negative_train.csv', 'negative_train.csv2', 0)