import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os



script_dir = os.path.dirname(__file__)

file_path = os.path.join(script_dir, "Order Data.xlsx")


order_data = pd.read_excel(file_path, sheet_name='Orders')

print("Initial Data:")
print(order_data.head())

order_data['Product Name'] = order_data['Product Name'].str.strip()  # Remove spaces from product names
order_data.dropna(axis=0, subset=['Order ID'], inplace=True)  # Drop rows with missing Order IDs
order_data['Order ID'] = order_data['Order ID'].astype('str')  # Ensure Order ID is string



print(order_data['Product Name'].unique())
print("Zipper Ring Binder Pockets" in order_data['Product Name'].unique())

# Print the data after cleaning
print("\nData after cleaning:")
print(order_data.head())

basket = (order_data.groupby(['Order ID', 'Product Name'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Order ID'))

print("\nBasket matrix:")
print(basket.head())

basket_encoded = basket.map(lambda x: x > 0)  # Apply boolean encoding

print("\nEncoded basket matrix:")
print(basket_encoded.head())

frequent_itemsets = apriori(basket_encoded, min_support=0.00025, use_colnames=True)

print("\nFrequent itemsets:")
print(frequent_itemsets)

if frequent_itemsets.empty:
    print("No frequent itemsets found. Try lowering the support threshold.")
else:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    filtered_rules = rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.1)].copy()
    filtered_rules.loc[:, 'antecedents'] = filtered_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    filtered_rules.loc[:, 'consequents'] = filtered_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    print("\nFiltered rules:")
    print(filtered_rules.head())
    output_file_path = os.path.join(script_dir, "market_basket_analysis_data.xlsx")  # Specify where to save the file
    filtered_rules.to_excel(output_file_path, index=False)