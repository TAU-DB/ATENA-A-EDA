import math
import random
from collections import Counter
from typing import List

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp

import networkx as nx

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate import bleu_score as nltk_bleu
from zss import Node, simple_distance

from atena.evaluation.distance import display_distance
from atena.simulation.actions import AbstractAction, ActionType, FilterAction, GroupAction
from atena.simulation.actions_simulator import ActionsSimulator
from atena.simulation.dataset import SchemaName, DatasetName


class PositiveNegativeStats(object):
    """
    A helper class that holds:
    TP - true positives
    FP - false positives
    FN - false negatives
    """

    def __init__(self, TP, FP, FN):
        self.TP = TP
        self.FP = FP
        self.FN = FN

    @property
    def precision(self):
        return self.TP / (self.TP + self.FP)

    @property
    def recall(self):
        return self.TP / (self.TP + self.FN)

    @property
    def f1(self):
        if (self.precision + self.recall) == 0:
            return 0

        return 2 * self.precision * self.recall / (self.precision + self.recall)


def precision_score_without_back(references, candidate, back_token):
    """
    Calculate macro precision for non-back actions of candidate compared to reference actions
    Args:
        references:
        candidate:
        back_token:

    Returns:

    """
    positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token=back_token)
    return positive_negative_stats.precision


def positive_negative_stats_without_back(references, candidate, back_token):
    """
    Returns a PositiveNegativeStats object containing the number of true positives, false positives and
    false negatives.
    Args:
        references:
        candidate:
        back_token:

    Returns:

    """
    references = remove_back_tokens_from_nested_lists(references, back_token=back_token)
    candidate = remove_back_tokens_from_nested_lists(candidate, back_token=back_token)

    candidate = list(set(candidate))
    candidate_success = [False] * len(candidate)
    for idx, cand_token in enumerate(candidate):
        cand_token_found = False
        for reference in references:
            for ref_token in reference:
                if ref_token == cand_token:
                    candidate_success[idx] = True
                    cand_token_found = True
                    break
            if cand_token_found:
                break

    references_tokens = list(set([ref_token for reference in references for ref_token in reference]))
    references_success = [False] * len(references_tokens)
    for idx, ref_token in enumerate(references_tokens):
        for cand_token in candidate:
            if ref_token == cand_token:
                references_success[idx] = True
                break

    TP, FP = candidate_success.count(True), candidate_success.count(False)
    FN = references_success.count(False)

    return PositiveNegativeStats(TP, FP, FN)


def micro_precision_without_back(references, candidates, back_token):
    """
    Calculate micro presicion for non-back actions of candidate compared to reference actions

    Args:
        references: A list of references for a single dataset only.
        candidates: A list of candidates for a single dataset only.
        back_token:

    Returns:

    """
    true_positives = false_positives = 0
    for candidate in candidates:
        positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token)
        true_positives += positive_negative_stats.TP
        false_positives += positive_negative_stats.FP
    return true_positives / (true_positives + false_positives)


def micro_recall_without_back(references, candidates, back_token):
    """
    Calculate micro recall for non-back actions of candidate compared to reference actions

    Args:
        references:
        candidates:
        back_token:

    Returns:

    """
    true_positives = false_negatives = 0
    for candidate in candidates:
        positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token)
        true_positives += positive_negative_stats.TP
        false_negatives += positive_negative_stats.FN
    return true_positives / (true_positives + false_negatives)


def micro_f1_without_back(references, candidates, back_token):
    """
    Calculate micro F1 score for non-back actions of candidate compared to reference actions

    Args:
        references:
        candidates:
        back_token:

    Returns:

    """
    precision = micro_precision_without_back(references, candidates, back_token)
    recall = micro_recall_without_back(references, candidates, back_token)
    if (precision + recall) == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def recall_score_without_back(references, candidate, back_token):
    """
    Calculate macro recall for non-back actions of candidate compared to reference actions

    Args:
        references:
        candidate:
        back_token:

    Returns:

    """
    positive_negative_stats = positive_negative_stats_without_back(references, candidate, back_token=back_token)
    return positive_negative_stats.recall


def f1_score_without_back(references, candidate, back_token):
    """
    Calculate macro F1 score for non-back actions of candidate compared to reference actions

    Args:
        references:
        candidate:
        back_token:

    Returns:

    """
    precision = precision_score_without_back(references, candidate, back_token=back_token)
    recall = recall_score_without_back(references, candidate, back_token=back_token)

    if (precision + recall) == 0:
        return 0

    return 2 * precision * recall / (precision + recall)


def paired_pvalue(lst1, lst2):
    """
    Calculating the p-value in a paired t test.
    The pairs are ordered by their index in lst1, lst2 (i.e. (lst1[i], lst2[i]) is the ith pair)
    Args:
        lst1:
        lst2:

    Returns:

    """
    assert len(lst1) == len(lst2)
    a = np.hstack(
        np.array(
            [np.array(lst1[i]) - np.array(lst2[i]) for i in range(len(lst1))]
        )
    )
    b = np.zeros_like(a)
    t_statistic, p_val = sp.stats.ttest_ind(a, b)
    # One-sided p-value
    p_val /= 2
    return t_statistic, p_val


def remove_back_tokens_from_nested_lists(lst, back_token):
    """
    Remove all back tokens in the list and its nested lists recursively.
    Args:
        lst: A list that can be nested with multiple levels
        back_token:

    Returns:

    """
    if isinstance(lst, list):
        new_lst = []
        for elem in lst:
            new_elem = remove_back_tokens_from_nested_lists(elem, back_token)
            if new_elem is not None:
                new_lst.append(new_elem)
        return new_lst
    else:
        if lst == back_token:
            return None
        return lst


def corpus_bleu_without_back(
        list_of_references,
        hypotheses,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):
    """
    Calculates regular corpus BLEU for the sessions without back actions.
    Args:
        list_of_references:
        hypotheses:
        back_token:
        weights:
        smoothing_function:
        auto_reweigh:

    Returns:

    """
    list_of_references = remove_back_tokens_from_nested_lists(list_of_references, back_token)
    hypotheses = remove_back_tokens_from_nested_lists(hypotheses, back_token)
    return corpus_bleu(list_of_references,
                       hypotheses,
                       weights=weights,
                       smoothing_function=smoothing_function,
                       auto_reweigh=auto_reweigh
                       )


def sentence_bleu_without_back(
        references,
        hypothesis,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):
    """
    Calculates regular sentence BLEU for the sessions without back actions.
    Args:
        references:
        hypothesis:
        back_token:
        weights:
        smoothing_function:
        auto_reweigh:

    Returns:

    """
    return corpus_bleu_without_back(
        [references],
        [hypothesis],
        back_token=back_token,
        weights=weights,
        smoothing_function=smoothing_function,
        auto_reweigh=auto_reweigh)


def every_tree_paths(tree, min_len, max_len, back_token='[back]'):
    """
    Return all paths of length >= min_len and <=max_len in a tree that is represented by a stack
    containing `back_token` to go back upwards in the tree.
    Paths are generated from a sequence of items, as an iterator.
    Args:
        tree:
        min_len:
        max_len:
        back_token:

    Returns:

    """

    if max_len == -1:
        max_len = len(tree)
    for n in range(min_len, max_len + 1):
        for ng in tree_paths(tree, n, back_token=back_token):
            yield ng


def tree_paths(
        tree,
        n,
        back_token='[back]',
):
    """
    Return all paths of length n in a tree that is represented by a stack
    containing `back_token` to go back upwards in the tree.
    Paths are generated from a sequence of items, as an iterator.

    This function is somewhat similar to `ngrams`:
        `from nltk.util import ngrams`

    :param tree: the source tree to be converted into paths
    :type tree: sequence or iter
    :param n: the length of the paths
    :type n: int
    :param back_token
    :rtype: sequence or iter
    """
    sequence = iter(tree)
    history = []
    while sequence:
        # PEP 479, prevent RuntimeError from being raised when StopIteration bubbles out of generator
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        if next_item == back_token:
            if history:
                history.pop()
        else:
            history.append(next_item)
            if len(history) >= n:
                yield (tuple(history[-n:]))


def tree_corpus_bleu(
        list_of_references,
        hypotheses,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):
    """
    Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all
    the hypotheses and their respective references.

    Instead of relating the "sentences" as a sequences of tokens, it is interpreted as a tree
    based on all the `back_token`s in the sentence. Instead of ngrams it uses paths of length
    n in the tree.

    Instead of averaging the sentence level BLEU scores (i.e. marco-average
    precision), the original BLEU metric (Papineni et al. 2002) accounts for
    the micro-average precision (i.e. summing the numerators and denominators
    for each hypothesis-reference(s) pairs before the division).

    Note: this function is based on `corpus_bleu` funcion of nltk

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> corpus_bleu(list_of_references, hypotheses)

    :param list_of_references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :param smoothing_function:
    :type smoothing_function: SmoothingFunction
    :param auto_reweigh: Option to re-normalize the weights uniformly.
    :type auto_reweigh: bool
    :return: The corpus-level BLEU score.
    :rtype: float
    """
    # Before proceeding to compute BLEU, perform sanity checks.

    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    assert len(list_of_references) == len(hypotheses), (
        "The number of hypotheses and their reference(s) should be the " "same "
    )

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i, back_token=back_token)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += nltk_bleu.closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = nltk_bleu.brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    # Collects the various precision values for the different ngram orders.
    p_n = [
        nltk_bleu.Fraction(p_numerators[i], p_denominators[i], _normalize=False)
        for i, _ in enumerate(weights, start=1)
    ]

    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0

    # If there's no smoothing, set use method0 from SmoothinFunction class.
    if not smoothing_function:
        smoothing_function = nltk_bleu.SmoothingFunction().method0
    # Smoothen the modified precision.
    # Note: smoothing_function() may convert values into floats;
    #       it tries to retain the Fraction object as much as the
    #       smoothing method allows.
    p_n = smoothing_function(
        p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths
    )
    s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, p_n))
    s = bp * math.exp(math.fsum(s))
    return s


def tree_corpus_gleu(list_of_references, hypotheses, back_token, min_len=1, max_len=4):
    """
    Calculate a single corpus-level tree (!) GLEU score (aka. system-level GLEU) for all
    the hypotheses and their respective references.

    Instead of averaging the sentence level GLEU scores (i.e. macro-average
    precision), Wu et al. (2016) sum up the matching tokens and the max of
    hypothesis and reference tokens for each sentence, then compute using the
    aggregate values.

    From Mike Schuster (via email):
        "For the corpus, we just add up the two statistics n_match and
         n_all = max(n_all_output, n_all_target) for all sentences, then
         calculate gleu_score = n_match / n_all, so it is not just a mean of
         the sentence gleu scores (in our case, longer sentences count more,
         which I think makes sense as they are more difficult to translate)."

    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...         'ensures', 'that', 'the', 'military', 'always',
    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...          'heed', 'Party', 'commands']
    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...          'guarantees', 'the', 'military', 'forces', 'always',
    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...          'army', 'always', 'to', 'heed', 'the', 'directions',
    ...          'of', 'the', 'party']

    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
    ...         'interested', 'in', 'world', 'history']
    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
    ...          'because', 'he', 'read', 'the', 'book']

    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
    >>> hypotheses = [hyp1, hyp2]
    >>> corpus_gleu(list_of_references, hypotheses) # doctest: +ELLIPSIS
    0.5673...

    The example below show that corpus_gleu() is different from averaging
    sentence_gleu() for hypotheses

    >>> score1 = sentence_gleu([ref1a], hyp1)
    >>> score2 = sentence_gleu([ref2a], hyp2)
    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS
    0.6144...

    :param list_of_references: a list of reference sentences, w.r.t. hypotheses
    :type list_of_references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :return: The corpus-level GLEU score.
    :rtype: float
    """
    # sanity check
    assert len(list_of_references) == len(
        hypotheses
    ), "The number of hypotheses and their reference(s) should be the same"

    # sum matches and max-token-lengths over all sentences
    corpus_n_match = 0
    corpus_n_all = 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_ngrams = Counter(every_tree_paths(hypothesis, min_len, max_len, back_token=back_token))
        tpfp = sum(hyp_ngrams.values())  # True positives + False positives.

        hyp_counts = []
        for reference in references:
            ref_ngrams = Counter(every_tree_paths(reference, min_len, max_len, back_token=back_token))
            tpfn = sum(ref_ngrams.values())  # True positives + False negatives.

            overlap_ngrams = ref_ngrams & hyp_ngrams
            tp = sum(overlap_ngrams.values())  # True positives.

            # While GLEU is defined as the minimum of precision and
            # recall, we can reduce the number of division operations by one by
            # instead finding the maximum of the denominators for the precision
            # and recall formulae, since the numerators are the same:
            #     precision = tp / tpfp
            #     recall = tp / tpfn
            #     gleu_score = min(precision, recall) == tp / max(tpfp, tpfn)
            n_all = max(tpfp, tpfn)

            if n_all > 0:
                hyp_counts.append((tp, n_all))

        # use the reference yielding the highest score
        if hyp_counts:
            n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
            corpus_n_match += n_match
            corpus_n_all += n_all

    # corner case: empty corpus or empty references---don't divide by zero!
    if corpus_n_all == 0:
        gleu_score = 0.0
    else:
        gleu_score = corpus_n_match / corpus_n_all

    return gleu_score


def tree_sentence_gleu(references, hypothesis, min_len=1, max_len=4, back_token='[back]'):
    """
    Calculates the sentence level tree (!) GLEU (Google-BLEU) score described in

        Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi,
        Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey,
        Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser,
        Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens,
        George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith,
        Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes,
        Jeffrey Dean. (2016) Google’s Neural Machine Translation System:
        Bridging the Gap between Human and Machine Translation.
        eprint arXiv:1609.08144. https://arxiv.org/pdf/1609.08144v2.pdf
        Retrieved on 27 Oct 2016.

    From Wu et al. (2016):
        "The BLEU score has some undesirable properties when used for single
         sentences, as it was designed to be a corpus measure. We therefore
         use a slightly different score for our RL experiments which we call
         the 'GLEU score'. For the GLEU score, we record all sub-sequences of
         1, 2, 3 or 4 tokens in output and target sequence (n-grams). We then
         compute a recall, which is the ratio of the number of matching n-grams
         to the number of total n-grams in the target (ground truth) sequence,
         and a precision, which is the ratio of the number of matching n-grams
         to the number of total n-grams in the generated output sequence. Then
         GLEU score is simply the minimum of recall and precision. This GLEU
         score's range is always between 0 (no matches) and 1 (all match) and
         it is symmetrical when switching output and target. According to
         our experiments, GLEU score correlates quite well with the BLEU
         metric on a corpus level but does not have its drawbacks for our per
         sentence reward objective."

    Note: The initial implementation only allowed a single reference, but now
          a list of references is required (which is consistent with
          bleu_score.sentence_bleu()).

    The infamous "the the the ... " example

        >>> ref = 'the cat is on the mat'.split()
        >>> hyp = 'the the the the the the the'.split()
        >>> sentence_gleu([ref], hyp)  # doctest: +ELLIPSIS
        0.0909...

    An example to evaluate normal machine translation outputs

        >>> ref1 = str('It is a guide to action that ensures that the military '
        ...            'will forever heed Party commands').split()
        >>> hyp1 = str('It is a guide to action which ensures that the military '
        ...            'always obeys the commands of the party').split()
        >>> hyp2 = str('It is to insure the troops forever hearing the activity '
        ...            'guidebook that party direct').split()
        >>> sentence_gleu([ref1], hyp1) # doctest: +ELLIPSIS
        0.4393...
        >>> sentence_gleu([ref1], hyp2) # doctest: +ELLIPSIS
        0.1206...

    :param references: a list of reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param min_len: The minimum order of n-gram this function should extract.
    :type min_len: int
    :param max_len: The maximum order of n-gram this function should extract.
    :type max_len: int
    :return: the sentence level GLEU score.
    :rtype: float
    """
    return tree_corpus_gleu([references], [hypothesis], min_len=min_len, max_len=max_len, back_token=back_token)


def tree_sentence_gleu_n(list_of_references, hypotheses, back_token, n):
    """
    A helper function to calculate the sentence tree (!) GLEU with paths of length 1 through n (inclusive)
    Args:
        list_of_references:
        hypotheses:
        back_token:
        n:

    Returns:

    """
    return tree_sentence_gleu(list_of_references, hypotheses, back_token=back_token, min_len=1, max_len=n)


def tree_corpus_gleu_n(list_of_references, hypotheses, back_token, n):
    """
    A helper function to calculate the corpus tree (!) GLEU with paths of length 1 through n (inclusive)
    Args:
        list_of_references:
        hypotheses:
        back_token:
        n:

    Returns:

    """
    return tree_corpus_gleu(list_of_references, hypotheses, back_token=back_token, min_len=1, max_len=n)


def tree_corpus_bleu_n(
        list_of_references,
        hypotheses,
        back_token,
        n,
        smoothing_function=None,
        auto_reweigh=False,
):
    """
    A helper function to calculate the corpus tree (!) BLEU with paths of length 1 through n (inclusive)
    such that weights are devided equally between different lengths.
    Args:
        list_of_references:
        hypotheses:
        back_token:
        n:
        smoothing_function:
        auto_reweigh:

    Returns:

    """
    if n == 1:
        weights = (1,)
    elif n == 2:
        weights = (1 / 2, 1 / 2)
    elif n == 3:
        weights = (1 / 3, 1 / 3, 1 / 3)
    elif n == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        weights = (1 / n for i in range(n))

    return tree_corpus_bleu(list_of_references, hypotheses, back_token=back_token, weights=weights,
                            smoothing_function=smoothing_function, auto_reweigh=auto_reweigh)


def modified_precision(references, hypothesis, n, back_token):
    """
    Calculate modified npath (similar to ngram) precision.

    The normal precision method may lead to some wrong translations with
    high-precision, e.g., the translation, in which a word of reference
    repeats several times, has very high precision.

    This function only returns the Fraction object that contains the numerator
    and denominator necessary to calculate the corpus-level precision.
    To calculate the modified precision for a single pair of hypothesis and
    references, cast the Fraction object into a float.


    Note: this function is based on `modified_precision` in nltk, but in this
    case for trees


    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Extracts all ngrams in hypothesis
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(tree_paths(hypothesis, n, back_token=back_token)) if len(hypothesis) >= n else Counter()
    # Extract a union of references' counts.
    # max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = (
            Counter(tree_paths(reference, n, back_token=back_token)) if len(reference) >= n else Counter()
        )
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()
    }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return nltk_bleu.Fraction(numerator, denominator, _normalize=False)


def tree_sentence_bleu(
        references,
        hypothesis,
        back_token,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
):
    """
    Calculate Tree BLEU score (Bilingual Evaluation Understudy) from
    Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.
    "BLEU: a method for automatic evaluation of machine translation."
    In Proceedings of ACL. http://www.aclweb.org/anthology/P02-1040.pdf

    :param references: reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :param smoothing_function:
    :type smoothing_function: SmoothingFunction
    :param auto_reweigh: Option to re-normalize the weights uniformly.
    :type auto_reweigh: bool
    :return: The sentence-level BLEU score.
    :rtype: float
    """
    return tree_corpus_bleu(
        [references], [hypothesis], back_token, weights, smoothing_function, auto_reweigh
    )


def tree_sentence_bleu_n(
        references,
        hypothesis,
        back_token,
        n,
        smoothing_function=None,
        auto_reweigh=False,
):
    """
    A helper function to calculate the sentence tree (!) BLEU with paths of length 1 through n (inclusive)
    such that weights are devided equally between different lengths.
    Args:
        references:
        hypothesis:
        back_token:
        n:
        smoothing_function:
        auto_reweigh:

    Returns:

    """
    return tree_corpus_bleu_n(
        [references], [hypothesis], back_token, n, smoothing_function, auto_reweigh
    )


def construct_displays_tree(dhist, ahist: List[AbstractAction]):
    tree_size = 1
    root = Node(dhist[0])
    stack = [root]

    nodes_left = dhist[1:]

    for disp, act in zip(nodes_left, ahist):
        act_type = act.action_type
        if act_type is ActionType.BACK:
            if len(stack) > 1:
                stack.pop()
        else:
            tree_size += 1
            new_node = Node(disp)
            stack[-1].addkid(new_node)
            stack.append(new_node)

    return root, tree_size


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def construct_nx_displays_tree(dhist, actions_lst: List[AbstractAction]):
    G = nx.DiGraph()
    edge_labels = {}
    stack = [0]

    nodes_left = dhist[1:]

    for node_id, (disp, act) in enumerate(zip(nodes_left, actions_lst), start=1):
        action_type = act.action_type
        if action_type is ActionType.BACK:
            if len(stack) > 1:
                stack.pop()
        else:
            assert isinstance(act, (FilterAction, GroupAction))
            act_column = act.grouped_column if action_type is ActionType.GROUP else act.filtered_column
            G.add_edge(stack[-1], node_id, act=str(act))
            filter_operator = "" if action_type is ActionType.GROUP else ", " + act.filter_operator.name
            filter_term = "" if action_type is ActionType.GROUP else "\n" + str(act.filter_term)
            edge_labels[(stack[-1], node_id)] = f'{action_type[0].upper()}, {act_column}{filter_operator}{filter_term}'
            stack.append(node_id)

    return G, edge_labels


def construct_nx_display_tree_from_actions_lst(
        schema_name: SchemaName,
        dataset_name: DatasetName,
        actions_lst: List[AbstractAction]
):
    end_of_simulation_state = ActionsSimulator.get_end_of_simulation_state(schema_name, dataset_name, actions_lst)
    displays_history = end_of_simulation_state.displays_history

    # Create tree
    G, edge_labels = construct_nx_displays_tree(displays_history, actions_lst)

    return G, edge_labels


def get_number_of_back_actions_in_the_end(actions_lst):
    count = 0
    reverse_actions_lst = actions_lst[::-1]

    while reverse_actions_lst[count][0] == 0:
        count += 1

    return count


def draw_nx_display_tree(actions_lst, dataset_number, filter_terms_lst=None):
    # Create tree
    G, edge_labels = construct_nx_display_tree_from_actions_lst(actions_lst, dataset_number, filter_terms_lst=
    filter_terms_lst)

    # Draw the tree
    pos = hierarchy_pos(G, 0)

    figw, figh = 20.0, 8.0
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(figw, figh))

    # Find the node of the last action to make it green
    num_of_back_in_the_end = get_number_of_back_actions_in_the_end(actions_lst)
    last_node = list(G.nodes())[-1]
    while num_of_back_in_the_end and list(G.in_edges(last_node))[-1]:
        num_of_back_in_the_end -= 1
        last_node = list(G.in_edges(last_node))[-1][0]
    current_node_in_tree = last_node
    node_colors = ['green' if node == current_node_in_tree else 'red' for node in G.nodes]

    nx.draw_networkx(G, pos=pos, node_color=node_colors)
    text = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    for _, t in text.items():
        t.set_rotation('horizontal')
    plt.show()

    return G


def draw_ref_and_candidate_trees_side_by_side(
        ref: List[AbstractAction],
        cand: List[AbstractAction],
        schema_name: SchemaName,
        dataset_name: DatasetName,
):
    region = 120  # for pylab 1x2 subplot layout
    figw, figh = 18.0, 16.0
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(figw, figh))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.01, hspace=0.01)
    for actions_lst, title in [(ref, "reference"), (cand, "candidate")]:
        G, edge_labels = construct_nx_display_tree_from_actions_lst(schema_name, dataset_name, actions_lst)
        pos = hierarchy_pos(G, 0)
        region += 1
        plt.subplot(region)
        plt.title(title)
        nx.draw_networkx(G, pos=pos)
        text = nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
        for _, t in text.items():
            t.set_rotation('horizontal')

    plt.show()


def display_label_sitance(disp1, disp2, normalize, unit=5000):
    if not normalize:
        unit = 1

    if disp1 == disp2 == "":
        return 0
    elif disp1 == "" or disp2 == "":
        return unit

    # the * 2.0 is the that delete and then insert will cost like relabel
    # should be considered
    return display_distance(disp1, disp2).display_distance * unit * 2.0


def compute_display_TED(root1, root2, normalize):
    cost, ops = simple_distance(
        root1, root2, label_dist=lambda x, y: display_label_sitance(x, y, normalize), return_operations=True)
    return cost, ops


def compute_minimum_display_TED_from_actions(references: List[List[AbstractAction]],
                                             candidate: List[AbstractAction],
                                             schema_name: SchemaName,
                                             dataset_name: DatasetName,
                                             normalize=True,
                                             return_min_ops=False
                                             ):
    references_dhist_ahist_pairs = [
        (ActionsSimulator.get_end_of_simulation_state(schema_name, dataset_name, act_lst).displays_history, act_lst)
        for act_lst
        in references
    ]

    cand_dhist, cand_ahist = (
        ActionsSimulator.get_end_of_simulation_state(schema_name, dataset_name, candidate).displays_history, candidate)

    cand_tree, cand_tree_size = construct_displays_tree(cand_dhist, cand_ahist)

    # List of tree edit distances
    teds_lst = []
    ops_lst = []

    for dhist, ahist in references_dhist_ahist_pairs:
        ref_tree, ref_tree_size = construct_displays_tree(dhist, ahist)
        ted, ops = compute_display_TED(cand_tree, ref_tree, normalize)
        if normalize:
            ted = 2 * ted / ((cand_tree_size + ref_tree_size) * 5000 + ted)
        teds_lst.append(ted)
        ops_lst.append(ops)

    # (distance, tree) index pair
    if return_min_ops:
        min_ops = ops_lst[np.argmin(teds_lst)]
        return min(teds_lst), np.argmin(teds_lst), teds_lst, min_ops
    return min(teds_lst), np.argmin(teds_lst), teds_lst
