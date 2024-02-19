# Missing Values


# Value Mapping

## Normal expression to extract keywords

In Titanic project, we need to extract the people's title such as Mr. and Miss. from the name column. 
We observe that the title is always followed by a dot. Therefore, we can write a normal expression to extract.

```Python
data = [train_df, test_df]
titles = {"Mr":1, "Miss":2, "Mrs":3, "Master":4,"Rare":5}

for dataset in data:
    # get titles, the words that are followed by dots
    dataset["Title"] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
    # replace rare titles
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
```

In extracting the deck from the Cabin we also used it.
```Python
    dataset['Cabin'] = dataset['Cabin'].fillna('U0')
    dataset['Deck'] = dataset['Cabin'].map(lambda x:re.compile("([a-zA-Z]+)")
```

## Mapping from `str` to integers

We use the information from the dictionaries to map the data from string to integer.

```Python
dataset['Deck'] = dataset['Deck'].map(deck)

dataset["Title"] = dataset["Title"].map(titles)
```