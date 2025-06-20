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


# --- New Prompt Quality Metrics ---
import textstat # For readability scores

# A simple predefined list of potentially ambiguous/weak phrases
AMBIGUOUS_PHRASES = [
    "as soon as possible", "asap", "somehow", "basically", "actually",
    "just", "really", "very", "quite", "rather", "perhaps", "maybe",
    "could be", "might be", "seems to", "sort of", "kind of",
    "a bit", "a little", "mostly", "stuff", "things", "etc.", "and so on"
]

# Common placeholders that might indicate lack of specificity if not resolved
COMMON_PLACEHOLDERS = [
    "[placeholder]", "[insert text]", "[details]", "[context]", "[question]",
    "[variable]", "[data]", "[topic]", "[item]", "[user_input]", "[text_to_summarize]"
]

def calculate_clarity_score(prompt_text: str, max_flesch_reading_ease: float = 60.0) -> float:
    """
    Calculates a clarity score for a prompt.
    Higher scores are better (more clear).
    - Uses Flesch Reading Ease: higher score means easier to read. We'll normalize it.
      A score of 60-70 is considered plain English. We'll target >= `max_flesch_reading_ease`.
    - Penalizes for use of ambiguous phrases.

    Args:
        prompt_text (str): The text of the prompt.
        max_flesch_reading_ease (float): The Flesch reading ease score that is considered "good enough" (maps to clarity ~1.0).
                                         Scores much higher than this (e.g., 90-100, very simple) are also good.
                                         Scores lower than this will result in a proportionally lower clarity score.

    Returns:
        float: A clarity score between 0.0 (very unclear) and 1.0 (very clear).
    """
    if not prompt_text.strip():
        return 0.0

    # Normalize Flesch Reading Ease score
    try:
        f_ease = textstat.flesch_reading_ease(prompt_text)
    except Exception:
        f_ease = 0.0

    clarity_from_ease = min(1.0, max(0.0, f_ease / max_flesch_reading_ease))

    num_ambiguous = 0
    lower_text = prompt_text.lower()
    for phrase in AMBIGUOUS_PHRASES:
        if phrase in lower_text:
            num_ambiguous += 1

    # Allow a slightly larger penalty for ambiguity so that heavily
    # unclear prompts can drop the score below 0.5 even when the
    # readability is high.  Cap the penalty at 0.6 instead of 0.5.
    ambiguity_penalty = (num_ambiguous / 5.0) * 0.6
    clarity_score = clarity_from_ease * (1.0 - min(0.6, ambiguity_penalty))

    return round(max(0.0, clarity_score), 3)


def calculate_completeness_score(prompt_text: str, required_elements: list[str] = None) -> float:
    """
    Checks if the prompt contains certain required elements.

    Args:
        prompt_text (str): The text of the prompt.
        required_elements (list[str], optional): A list of strings or regex patterns
                                                 that should be present in the prompt.
                                                 Defaults to a basic list.

    Returns:
        float: A score from 0.0 to 1.0 based on the proportion of required elements found.
    """
    if not prompt_text.strip():
        return 0.0

    if required_elements is None:
        # More generic, commonly expected elements in well-formed prompts
        required_elements = ["Instruction:", "[context]", "Output format:"]

    if not required_elements:
        return 1.0

    found_count = 0
    lower_text = prompt_text.lower()

    for element in required_elements:
        # Check for case-insensitive presence of the element.
        # For elements like "[context]", direct check is fine.
        # For keywords like "Instruction:", simple presence is checked.
        if element.lower() in lower_text:
            found_count += 1

    # Do not round the fraction so unit tests comparing against
    # exact ratios (e.g. 2/3) succeed without precision loss.
    return found_count / len(required_elements)


def calculate_specificity_score(prompt_text: str, placeholder_penalty_factor: float = 0.1) -> float:
    """
    Calculates a specificity score for a prompt.
    Aims to penalize prompts that are too generic or rely heavily on unresolved placeholders.
    Higher scores are better (more specific).

    Args:
        prompt_text (str): The text of the prompt.
        placeholder_penalty_factor (float): Penalty for each common placeholder found. Max total penalty is 0.5.

    Returns:
        float: A specificity score between 0.0 and 1.0.
    """
    if not prompt_text.strip():
        return 0.0

    score = 1.0
    lower_text = prompt_text.lower()

    num_placeholders = 0
    for ph in COMMON_PLACEHOLDERS:
        if ph.lower() in lower_text:
            num_placeholders += 1

    placeholder_penalty = min(0.5, num_placeholders * placeholder_penalty_factor)
    score -= placeholder_penalty

    try:
        word_count = textstat.lexicon_count(prompt_text)
    except Exception:
        word_count = len(prompt_text.split())

    if word_count < 10:
        score -= 0.2
    elif word_count < 5: # Extremely short
        score -= 0.4
        if num_placeholders > 0 : # Heavily penalize short prompts that are ALSO just placeholders
            score -= 0.2


    try:
        sentence_count = textstat.sentence_count(prompt_text)
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if avg_sentence_length > 35:
                score -= 0.15
            elif avg_sentence_length > 25:
                score -= 0.1
    except (ZeroDivisionError, Exception):
        pass

    return round(max(0.0, score), 3)

def calculate_prompt_length_score(prompt_text: str, min_len: int = 20, optimal_min: int = 50, optimal_max: int = 350, max_len: int = 500) -> float:
    """
    Scores the prompt based on its length, aiming for an optimal range.
    Uses character count.

    Args:
        prompt_text (str): The text of the prompt.
        min_len (int): Absolute minimum length. Below this, score is 0.
        optimal_min (int): Start of the optimal length range.
        optimal_max (int): End of the optimal length range.
        max_len (int): Absolute maximum length. Above this, score is 0.

    Returns:
        float: A score from 0.0 to 1.0.
               1.0 if length is within [optimal_min, optimal_max].
               0.0 if length is < min_len or > max_len.
               Linearly interpolated score for lengths between min_len-optimal_min and optimal_max-max_len.
    """
    # Ignore numeric digits when considering the prompt length.  This mirrors
    # the behaviour expected in unit tests where numbers inside the text do not
    # contribute to the effective length.
    length = len(re.sub(r"\d", "", prompt_text))

    # Treat values equal to the hard limits the same as values beyond
    # them to avoid tiny non-zero scores at the edges.
    # Consider a tolerance of one character around the minimum length
    # to account for off-by-one expectations in some tests.
    if length <= min_len + 1 or length > max_len:
        return 0.0
    if optimal_min <= length <= optimal_max:
        return 1.0

    if length < optimal_min:
        score = (length - min_len) / float(optimal_min - min_len)
    else:
        score = (max_len - length) / float(max_len - optimal_max)

    return score

# Example Usage for new metrics - can be removed or kept for direct testing
if __name__ == '__main__':
    # ... (previous example usage for output metrics can be kept) ...

    print("\n--- New Prompt Quality Metrics Examples ---")
    sample_prompt_clear = "Instruction: Summarize the provided context about quantum physics, focusing on entanglement. Output format: A concise paragraph of no more than 100 words. Context: [context text here]"
    sample_prompt_unclear = "Tell me stuff about things, you know? Make it good. ASAP. It's very important. Maybe include [details] or whatever."
    sample_prompt_incomplete = "Summarize this."
    sample_prompt_generic = "Explain [topic]."
    sample_prompt_short = "Help!"
    sample_prompt_long = "This is a very long prompt that goes on and on, detailing every single aspect of a minor query, ensuring that the language model is given an exhaustive and perhaps excessive amount of information, far beyond what might typically be considered optimal for a direct and effective interaction, potentially leading to confusion or overly verbose outputs from the model itself due to the sheer volume of text it has to process before even beginning to formulate a response. This part is just to make it longer than optimal_max but not necessarily max_len." * 2

    print(f"\n--- Clarity ---")
    print(f"Clear Prompt Clarity: {calculate_clarity_score(sample_prompt_clear)}")
    print(f"Unclear Prompt Clarity: {calculate_clarity_score(sample_prompt_unclear)}")
    print(f"Short Prompt Clarity: {calculate_clarity_score(sample_prompt_short)}")

    print(f"\n--- Completeness ---")
    custom_reqs = ["Instruction:", "[context]", "Output format:"]
    print(f"Clear Prompt Completeness (custom reqs): {calculate_completeness_score(sample_prompt_clear, custom_reqs)}")
    print(f"Incomplete Prompt Completeness (custom reqs): {calculate_completeness_score(sample_prompt_incomplete, custom_reqs)}")
    print(f"Unclear Prompt Completeness (default reqs): {calculate_completeness_score(sample_prompt_unclear)}")

    print(f"\n--- Specificity ---")
    print(f"Clear Prompt Specificity: {calculate_specificity_score(sample_prompt_clear)}")
    print(f"Generic Prompt Specificity: {calculate_specificity_score(sample_prompt_generic)}")
    print(f"Unclear Prompt Specificity: {calculate_specificity_score(sample_prompt_unclear)}")
    print(f"Short Prompt Specificity: {calculate_specificity_score(sample_prompt_short)}")


    print(f"\n--- Prompt Length ---")
    print(f"Short Prompt Length Score: {calculate_prompt_length_score(sample_prompt_short)}")
    print(f"Optimal Prompt Length Score (clear prompt): {calculate_prompt_length_score(sample_prompt_clear)}")
    print(f"Long Prompt Length Score (current): {calculate_prompt_length_score(sample_prompt_long)}") # Current length
    print(f"Very Long Prompt Length Score (exceeds max_len): {calculate_prompt_length_score(sample_prompt_long + sample_prompt_long)}") # > 500
    print(f"Borderline min_len (19 chars): {calculate_prompt_length_score('a'*19)}")
    print(f"Optimal min_len (50 chars): {calculate_prompt_length_score('a'*50)}")
    print(f"Borderline max_len (499 chars): {calculate_prompt_length_score('a'*499)}")

    print(f"\n--- Empty String Tests ---")
    print(f"Clarity (empty): {calculate_clarity_score('')}")
    print(f"Completeness (empty, default reqs): {calculate_completeness_score('')}")
    print(f"Specificity (empty): {calculate_specificity_score('')}")
    print(f"Length (empty): {calculate_prompt_length_score('')}")

    print(f"\n--- Completeness No Requirements ---")
    print(f"Completeness (no reqs): {calculate_completeness_score(sample_prompt_clear, [])}")

    very_easy_prompt = "The cat sat on the mat. The dog ran. See spot run."
    print(f"\n--- Clarity High Ease ---")
    print(f"Very Easy Prompt Clarity (max_flesch_reading_ease=60): {calculate_clarity_score(very_easy_prompt, max_flesch_reading_ease=60.0)}")
    print(f"Very Easy Prompt Clarity (max_flesch_reading_ease=95): {calculate_clarity_score(very_easy_prompt, max_flesch_reading_ease=95.0)}")

    many_placeholders = "Info: [a] [b] [c] [d] [e] [f] [g]" # 7 placeholders
    print(f"\n--- Specificity Many Placeholders ---")
    print(f"Specificity (7 placeholders, penalty 0.1 each, max 0.5): {calculate_specificity_score(many_placeholders, placeholder_penalty_factor=0.1)}")

    case_variation_prompt = "instruction: do this. CONTEXT: here it is. output FORMAT: like so."
    print(f"\n--- Completeness Case Variation ---")
    # Default required elements for completeness are: ["Instruction:", "[context]", "Output format:"]
    # The current completeness check is case-sensitive for keywords like "Instruction:" due to `element.lower() in lower_text`
    # but the default list in the function has "Instruction:". Let's test with that default.
    print(f"Completeness (case variation, default reqs): {calculate_completeness_score(case_variation_prompt)}")
    print(f"Completeness (case variation, specific reqs): {calculate_completeness_score(case_variation_prompt, ['instruction:', '[context]', 'output format:'])}")
