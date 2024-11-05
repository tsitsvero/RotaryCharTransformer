from datasets import load_dataset

# Load the enwik8 dataset
dataset = load_dataset("LTCB/enwik8", split="train", trust_remote_code=True)

# Extract the raw text from the dataset
raw_text = ''.join(dataset['text'])

# Calculate the total number of characters in the dataset
total_characters = len(raw_text)
print(f"Total number of characters in the dataset: {total_characters}")

# Define the number of characters for training
train_character_limit = 90_000_000

# Ensure sufficient data is available for training
if total_characters < train_character_limit:
    raise ValueError(f"Insufficient data for training. Only {total_characters} characters available.")

# Determine the remaining characters for validation and testing
remaining_characters = total_characters - train_character_limit

# Define validation character size (maximum of 5 million)
validation_size = min(5_000_000, remaining_characters)

# Use the rest for testing
test_size = remaining_characters - validation_size

# Split the text into training, validation, and test sets
train_data = raw_text[:train_character_limit]
validation_data = raw_text[train_character_limit:train_character_limit + validation_size]
test_data = raw_text[train_character_limit + validation_size:]

# Save each split to a separate file
with open('train.txt', 'w') as train_file:
    train_file.write(train_data)
with open('valid.txt', 'w') as valid_file:
    valid_file.write(validation_data)
with open('test.txt', 'w') as test_file:
    test_file.write(test_data)

print("Data successfully split and saved as train.txt, valid.txt, and test.txt")

# Verify the sizes of the splits
print(f"Training characters: {len(train_data)}")
print(f"Validation characters: {len(validation_data)}")
print(f"Test characters: {len(test_data)}")