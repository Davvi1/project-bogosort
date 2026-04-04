#packages
import re
import pandas as pd

#define function
def count_elongated_tokens(text):
    """
    Count tokens with letter elongation (e.g., 'coooool', 'waaaaack').
    A token is elongated if any character repeats 3+ times consecutively.
    """
    tokens = text.split()
    elongated_count = 0
    for token in tokens:
        if re.search(r'(.)\1{2,}', token, re.IGNORECASE):
            elongated_count += 1
    return elongated_count


def count_consecutive_punctuation(text):
    """
    Count occurrences of 3+ consecutive punctuation marks (e.g., '!!!', '...', '???').
    """
    return len(re.findall(r'[^\w\s]{3,}', text))


# TEST -- WILL BE REFINED WITH PROPER TEST SYNTAX LATER ON
if __name__ == "__main__":
    print(count_elongated_tokens("I looooove this"))                     # 1
    print(count_elongated_tokens("hello world"))                         # 0

    print(count_consecutive_punctuation("really!!! what??? No !!"))      # 2
    print(count_consecutive_punctuation("wait... what??"))               # 1
    print(count_consecutive_punctuation("hello world"))                  # 0
