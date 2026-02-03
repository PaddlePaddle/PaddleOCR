#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify French accented character handling in OCR text recognition.

This script tests that French words with accented characters (é, è, à, ç, etc.)
and contractions (n'êtes, l'été) are properly grouped as single words and not
split at each accented character.
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppocr.postprocess.rec_postprocess import BaseRecLabelDecode


def test_french_word_grouping():
    """Test that French words with accents are properly grouped."""

    # Initialize the decoder
    decoder = BaseRecLabelDecode(character_dict_path=None, use_space_char=True)

    # Test cases with French accented words
    test_cases = [
        {
            "name": "Simple accented word: été (summer)",
            "text": "été",
            "expected_words": [["é", "t", "é"]],
            "expected_states": ["en&num"],
        },
        {
            "name": "Word with ç: français (French)",
            "text": "français",
            "expected_words": [["f", "r", "a", "n", "ç", "a", "i", "s"]],
            "expected_states": ["en&num"],
        },
        {
            "name": "Contraction: n'êtes (you are)",
            "text": "n'êtes",
            "expected_words": [["n", "'", "ê", "t", "e", "s"]],
            "expected_states": ["en&num"],
        },
        {
            "name": "Multiple accents: élève (student)",
            "text": "élève",
            "expected_words": [["é", "l", "è", "v", "e"]],
            "expected_states": ["en&num"],
        },
        {
            "name": "Word with à: à demain (see you tomorrow)",
            "text": "à demain",
            "expected_words": [["à"], ["d", "e", "m", "a", "i", "n"]],
            "expected_states": ["en&num", "en&num"],
        },
        {
            "name": "Complex: C'était très français (It was very French)",
            "text": "C'était très français",
            "expected_words": [
                ["C", "'", "é", "t", "a", "i", "t"],
                ["t", "r", "è", "s"],
                ["f", "r", "a", "n", "ç", "a", "i", "s"],
            ],
            "expected_states": ["en&num", "en&num", "en&num"],
        },
    ]

    print("=" * 70)
    print("Testing French Accented Character Word Grouping")
    print("=" * 70)

    all_passed = True

    for test in test_cases:
        text = test["name"]
        test_text = test["text"]

        # Create a mock selection array (all characters are valid)
        selection = np.ones(len(test_text), dtype=bool)

        # Call get_word_info
        word_list, word_col_list, state_list = decoder.get_word_info(
            test_text, selection
        )

        # Check results
        passed = True

        if len(word_list) != len(test["expected_words"]):
            passed = False
            print(f"\nFAILED: {text}")
            print(
                f"   Expected {len(test['expected_words'])} words, got {len(word_list)}"
            )
        elif state_list != test["expected_states"]:
            passed = False
            print(f"\nFAILED: {text}")
            print(f"   Expected states: {test['expected_states']}")
            print(f"   Got states: {state_list}")
        else:
            # Check if words match
            for i, (expected, actual) in enumerate(
                zip(test["expected_words"], word_list)
            ):
                if expected != actual:
                    passed = False
                    print(f"\nFAILED: {text}")
                    print(f"   Word {i}: Expected {expected}, got {actual}")
                    break

        if passed:
            print(f"\nPASSED: {text}")
            print(f"   Text: '{test_text}'")
            print(f"   Words: {[''.join(w) for w in word_list]}")
            print(f"   States: {state_list}")
        else:
            all_passed = False
            print(f"   Text: '{test_text}'")
            print(f"   Expected words: {[''.join(w) for w in test['expected_words']]}")
            print(f"   Got words: {[''.join(w) for w in word_list]}")

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED! French accented words are properly grouped.")
    else:
        print("Some tests FAILED. Please review the output above.")
    print("=" * 70)

    assert all_passed, "Some French accent tests failed"


if __name__ == "__main__":
    test_french_word_grouping()
