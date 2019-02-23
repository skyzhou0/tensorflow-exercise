# Creator: Hao Sky Zhou.
'''
Part I

Data
transactions.tsv
Dataset of in-store customer transactions. Each transaction would consist of a 
number of items. The transaction value is the total cost of the items in the 
basket.
Columns:
-	transaction.value – Value of the basket in £’s
-	gender – The gender of the customer
-	store.type – The type of store the transaction occurred in
Q1) What is the average transaction value?
Q2) Is there significant difference between spend in different store types?
'''

import pandas as pd 

if __name__ == '__main__':
	df = pd.read_csv('transactions.tsv', sep ='\t')

	print( 'the average transaction value is £{}'.format(round(df['transaction.value'].mean(), 2)) )
	# the average transaction value is £286.18.

	df.groupby(['store.type']).mean()
	'''
	            transaction.value
	store.type                   
	Express             86.349468
	Extra              533.165920
	Metro              421.018131
	Superstore         155.166490
	'''
	# Yes, there have been significant difference between average spend in different store. 
	# For example, the average transaction at Extra is over 500% higher than that of at Express store.




