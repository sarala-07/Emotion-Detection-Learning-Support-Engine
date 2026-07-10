from preprocessing import clean_text

sample = "I am feeling very stressed because I cannot understand recursion!"

print("Original:")
print(sample)

print()

print("Processed:")
print(clean_text(sample))