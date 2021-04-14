# Machine Learning using Grammatical Evolution Project
--------------
## Grammar
Three grammar files were created for use throughout this project which include:
- `FypV2.pybnf`
- `mammography.pybnf`
- `original.pybnf`

The primary grammar file used was `FypV2.pybnf`.
#### **`FypV2.pybnf`**
As mentioned above this grammar file was the primary grammar file used throughout this project and during experimentation and testing. The difference between this file and the other grammar files is that this file includes extra comparsion operators and more in depth "if" statements.

#### **`mammography.pybnf`**
This grammar file was used in the earlier stages of development before the introduction of the above grammar file and contains some of the same elements as the above grammar file. This grammar file can be seen as an earlier version of the above grammar file as the only real differences are that it contains less terminal options and less if-statement non-terminal options.

#### **`original.pybnf`**
This grammar file was the initial grammar file developed for use in this project. The other two grammar files were built upon this grammar file. The reason for not sticking with this grammar file was due to how generic it was and did not contain some key functions which the others did.