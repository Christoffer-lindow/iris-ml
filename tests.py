import unittest
from statistics import stdev, mean
from naivebayes import NaiveBayes
from datamanipulations import seperate_classes, get_values_for_col
from statutils import summarize_col, calc_p, pdf, normalize_p


class DataManipulationTest(unittest.TestCase):
    def test_seperate_case_1(self):
        input_arr = [[1, 2, 1], [7, 3, 1], [10, 3, 1],
                     [5, 4, 2], [5, 3, 2], [1, 2, 3], [3, 4, 3]]
        expected = {1: [[1, 2], [7, 3], [10, 3]],
                    2: [[5, 4], [5, 3]], 3: [[1, 2], [3, 4]]}
        actual = seperate_classes(input_arr)
        self.assertDictEqual(actual, expected)

    def test_seperate_case_2(self):
        input_arr = [[1, 2, 1], [1, 2, 2], [3, 2, 1], [3, 4, 3], [3, 2, 2]]
        expected = {1: [[1, 2], [3, 2]], 2: [[1, 2], [3, 2]], 3: [[3, 4]]}
        actual = seperate_classes(input_arr)
        self.assertDictEqual(actual, expected)

    def test_get_col_arr_case_1(self):
        input_arr = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        expected = [1, 1, 1]
        actual = get_values_for_col(0, input_arr)
        self.assertListEqual(actual, expected)

    def test_get_col_arr_case_2(self):
        input_arr = [[3.321, 45.321, 34.321, 1], [
            3.231, 423.3211, 34.3213, 1], [1, 2, 3, 2], [1, 20, 3.2, 3]]
        expected = [1, 1, 2, 3]
        actual = get_values_for_col(3, input_arr)
        self.assertListEqual(actual, expected)


class BayesTest(unittest.TestCase):

    def test_summarize_col_case_1(self):
        input_list = [1.4, 1.3, 1.5, 1.4, 1.7, 1.4]
        mean_expected = 1.45
        std_expected = 0.14
        mean_actual, std_actual = summarize_col(input_list)
        self.assertEqual(mean_actual, mean_expected)
        self.assertAlmostEqual(std_actual, std_expected,2)

    def test_pdf_case_1(self):
        mean_val = 1.45
        stdev = 0.14
        x = 1.6
        expected = 1.601
        actual = pdf(x, mean_val, stdev)
        self.assertAlmostEqual(actual, expected, 2)

    def test_pdf_case_2(self):
        mean_val = 0.25
        stdev = 0.08
        x = 8
        expected = 1.970e-09
        actual = pdf(x, mean_val, stdev)
        self.assertAlmostEqual(actual, expected, 2)

    def test_pdf_case_3(self):
        mean_val = 4.40
        stdev = 0.52
        x = 1.6
        expected = 3.424e-07
        actual = pdf(x, mean_val, stdev)
        self.assertAlmostEqual(actual, expected, 2)

    def test_pdf_case_4(self):
        mean_val = 1.40
        stdev = 0.24
        x = 0.8
        expected = 0.081
        actual = pdf(x, mean_val, stdev)
        self.assertAlmostEqual(actual, expected, 0)

    def test_get_p_case_1(self):
        input_arr = [1.601, 1.970e-09]
        expected = 3.154e-09
        actual = calc_p(input_arr)
        self.assertAlmostEqual(actual, expected,2)

    def test_get_p_case2(self):
        input_arr = [3.424e-07, 0.081]
        expected = 2.776e-08
        actual = calc_p(input_arr)
        self.assertAlmostEqual(actual, expected,2)

    def test_normalize_p_case1(self):
        input_arr = [3.154e-09, 2.776e-08]
        x_1 = 3.154e-09
        x_2 = 2.776e-08
        expected_p1 = 0.102
        expected_p2 = 0.898
        p1 = normalize_p(x_1, input_arr)
        p2 = normalize_p(x_2, input_arr)
        self.assertAlmostEqual(p1, expected_p1, 2)
        self.assertAlmostEqual(p2, expected_p2, 2)
        self.assertAlmostEqual(p1+p2, 1, 10)


class MeanTest(unittest.TestCase):
    def test_mean_case_1(self):
        input_list = [1, 2, 3, 4, 5]
        expected = 3
        actual = mean(input_list)
        self.assertEqual(actual, expected)

    def test_mean_case_2(self):
        input_list = [10, 15, 20, 15, 20, 10]
        expected = 15
        actual = mean(input_list)
        self.assertEqual(actual, expected)


class StdTest(unittest.TestCase):

    def test_std_case_1(self):
        input_list = [1.4, 1.3, 1.5, 1.4, 1.7, 1.4]
        mean_expected = 1.45
        std_expected = 0.14
        mean_actual = mean(input_list)
        std_actual = stdev(input_list)
        self.assertAlmostEqual(mean_actual, mean_expected, 2)
        self.assertAlmostEqual(std_actual, std_expected, 2)

    def test_std_case_2(self):
        input_list = [0.2, 0.2, 0.2, 0.2, 0.4, 0.3]
        mean_expected = 0.25
        std_expected = 0.08
        mean_actual = mean(input_list)
        std_actual = stdev(input_list)
        self.assertAlmostEqual(mean_actual, mean_expected, 2)
        self.assertAlmostEqual(std_actual, std_expected, 2)


if __name__ == '__main__':
    unittest.main()
