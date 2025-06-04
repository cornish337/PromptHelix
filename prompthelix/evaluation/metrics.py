# prompthelix/evaluation/metrics.py

import re

def calculate_exact_match(generated_output: str, expected_output: str) -> float:
    """
    Calculates if the generated output exactly matches the expected output.
    Returns 1.0 if they match, 0.0 otherwise.
    Case-sensitive.
    """
    if generated_output is None or expected_output is None:
        return 0.0
    return 1.0 if generated_output == expected_output else 0.0

def calculate_keyword_overlap(generated_output: str, expected_output: str, keywords: list = None) -> float:
    """
    Calculates the Jaccard similarity based on keyword overlap.
    If no keywords are provided, it will split the strings into words.
    """
    if generated_output is None or expected_output is None:
        return 0.0

    if keywords:
        generated_words = set(k for k in keywords if k in generated_output)
        expected_words = set(k for k in keywords if k in expected_output)
    else:
        # Simple whitespace and punctuation splitting
        generated_words = set(re.findall(r'\w+', generated_output.lower()))
        expected_words = set(re.findall(r'\w+', expected_output.lower()))

    if not generated_words and not expected_words:
        return 1.0  # Both empty, perfect match in terms of word content
    if not expected_words: # Avoid division by zero if expected is empty but generated is not
        return 0.0

    intersection = generated_words.intersection(expected_words)
    union = generated_words.union(expected_words)

    if not union: # Should not happen if expected_words is not empty, but as a safeguard
        return 0.0

    return len(intersection) / len(union)

def calculate_output_length(generated_output: str, expected_output: str = None) -> int:
    """
    Calculates the length of the generated output.
    The expected_output is not used but included for consistent signature.
    """
    if generated_output is None:
        return 0
    return len(generated_output)

def calculate_bleu_score(generated_output: str, expected_output: str) -> float:
    """
    Placeholder for BLEU score calculation.
    Actual BLEU score calculation requires a library like nltk.
    For now, this is a simplified version or could raise NotImplementedError.
    """
    # This is a very naive placeholder.
    # For a real BLEU score, you'd use:
    # from nltk.translate.bleu_score import sentence_bleu
    # reference = [expected_output.split()]
    # candidate = generated_output.split()
    # return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    # print("Warning: calculate_bleu_score is a placeholder and not a true BLEU score.")

    # Simple word overlap as a proxy for now
    if generated_output is None or expected_output is None:
        return 0.0

    gen_words = set(generated_output.lower().split())
    exp_words = set(expected_output.lower().split())

    if not exp_words:
        return 0.0 if gen_words else 1.0 # if both empty, 1.0, else 0.0

    common_words = gen_words.intersection(exp_words)

    # Return a ratio of common words to words in expected output
    # This is NOT BLEU, just a simple stand-in
    return len(common_words) / len(exp_words) if exp_words else 0.0


# Example of how these might be used by the Evaluator (for illustration)
if __name__ == '__main__':
    gen_out = "The quick brown fox"
    exp_out = "The quick brown dog"
    exp_out_exact = "The quick brown fox"

    print(f"Exact Match (different): {calculate_exact_match(gen_out, exp_out)}")
    print(f"Exact Match (same): {calculate_exact_match(gen_out, exp_out_exact)}")

    print(f"Keyword Overlap (default): {calculate_keyword_overlap(gen_out, exp_out)}")
    my_keywords = ["quick", "fox", "lazy", "dog"]
    print(f"Keyword Overlap (custom): {calculate_keyword_overlap(gen_out, exp_out, keywords=my_keywords)}")

    print(f"Output Length: {calculate_output_length(gen_out)}")

    print(f"BLEU Score (placeholder): {calculate_bleu_score(gen_out, exp_out)}")
    print(f"BLEU Score (placeholder, exact): {calculate_bleu_score(gen_out, exp_out_exact)}")

    # Test with None inputs
    print(f"Exact Match (None): {calculate_exact_match(None, exp_out)}")
    print(f"Keyword Overlap (None): {calculate_keyword_overlap(gen_out, None)}")
    print(f"BLEU Score (None): {calculate_bleu_score(None, None)}")
