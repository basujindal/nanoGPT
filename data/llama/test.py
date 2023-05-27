#  open summary.txt and replace \n with space

file_name = 'summary.txt'

with open(file_name, 'r') as file:
    content = file.read()

# with open(file_name, 'w', newline='\n') as file:
#     file.write(content)


with open(file_name, 'r') as file:
    content = file.read()

content = content.replace('\n', ' ')


with open(file_name, 'w') as file:
    file.write(content)
    
print(content)




