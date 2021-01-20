from math import exp, sqrt, pi, log
from statistics import mean, stdev


def normalize_probabilities(probabilities, sum_probabilities):
    for i in range(len(probabilities)):
        probabilities[i +
                      1] = normalize_p(probabilities[i+1], sum_probabilities)
    return probabilities


def pdf(x, mean, stdev):
    left = (1 / (sqrt(2 * pi)*stdev))
    right = exp(-((x-mean)**2)/(2 * stdev**2))
    return left * right


def summarize_col(arr):
    col_mean = mean(arr)
    col_stdev = stdev(arr)
    return col_mean, col_stdev


def sumarize_p(probabilities):
    sum_probabilities = list()
    for i in range(len(probabilities)):
        probabilities[i+1] = calc_p(probabilities[i+1])
        sum_probabilities.append(probabilities[i+1])

    return sum_probabilities


def calc_p(pdf_arr):
    product = 1
    for num in pdf_arr:
        product += log(num, 10)
    return exp(product)


def normalize_p(x, pdf_arr):
    return x / sum([(val) for val in pdf_arr])
