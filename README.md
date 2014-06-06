#Description of RaDD: Race Against Drift-Diffusion model of response inhibition#


##Code Block with Ruby Syntax Highlighting
```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```


##Code Block with Python Syntax Highlighting
```python
x="Hello World"
for i, x in enumerate(y):
	print x[i]
	if x[i]==x[-1]:
		print x
```


##Code Block Without Syntax Highlighting	
> Code blocks without syntax highlighting are created by simply indenting (at least 8 spaces) 

> For example, the following git commands are used for adding or deleting files in your local repo:
	
	> cp ../markdown/readme/README.md ./
	> git add README.md
	> git rm README.markdown


##Embedding links
[Visit GitHub!](www.github.com).


##Tables
First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell

| Name | Description          |
| ------------- | ----------- |
| Help      | Display the help window.|
| Close     | Closes a window     |

| Name | Description          |
| ------------- | ----------- |
| Help      | ~~Display the~~ help window.|
| Close     | _Closes_ a window     |

| Left-Aligned  | Center Aligned  | Right Aligned |
| :------------ |:---------------:| -----:|
| col 3 is      | some wordy text | $1600 |
| col 2 is      | centered        |   $12 |
| zebra stripes | are neat        |    $1 |
