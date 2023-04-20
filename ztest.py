import numpy as np
import scipy.stats as stats
import logging


class TwoSampleZTest:
    """
    Class to perform a two-sample z-test for comparing means of two samples.
    """
    def __init__(self, sample1, sample2, alpha=0.05):
        """
        Initialize the TwoSampleZTest object.

        Args:
            sample1 (array-like): Sample 1 data.
            sample2 (array-like): Sample 2 data.
            alpha (float, optional): Significance level for hypothesis test. Defaults to 0.05.
        """
        self.sample1 = sample1
        self.sample2 = sample2
        self.alpha = alpha
        self.z_stat = None
        self.p_value = None
        self.z_critical = None

    def calculate_z_statistic(self):
        """
        Calculate the z-statistic for the two-sample z-test.

        Returns:
            float: The calculated z-statistic.
        """
        self.z_stat, _ = stats.ttest_ind(self.sample1, self.sample2)
        return self.z_stat

    def calculate_p_value(self):
        """
        Calculate the p-value for the two-sample z-test.

        Returns:
            float: The calculated p-value.
        """
        _, self.p_value = stats.ttest_ind(self.sample1, self.sample2)
        return self.p_value

    def calculate_critical_value(self):
        """
        Calculate the critical value for the two-sample z-test.

        Returns:
            float: The calculated critical value.
        """
        self.z_critical = stats.norm.ppf(1 - self.alpha/2)
        return self.z_critical

    def compare_p_value_with_alpha(self):
        """
        Compare the calculated p-value with the significance level (alpha).

        Returns:
            str: A string indicating the result of the hypothesis test.
        """
        if self.p_value < self.alpha:
            return "Reject null hypothesis. There is a significant difference between the means."
        else:
            return "Fail to reject null hypothesis. There is no significant difference between the means."

    def run_z_test(self):
        """
        Run the two-sample z-test.

        Returns:
            str: A string indicating the result of the hypothesis test.
        """
        try:
            self.calculate_z_statistic()
            self.calculate_p_value()
            self.calculate_critical_value()
            result = self.compare_p_value_with_alpha()
            logging.info("Two-sample z-test completed successfully.")
        except Exception as e:
            logging.error("Error occurred during two-sample z-test: {}".format(e))
            result = "Error occurred during hypothesis test. Please check the input data."
        return result

# Generate sample data
np.random.seed(0)
sample1 = np.random.normal(loc=10, scale=2, size=100)  # Sample 1 with mean 10 and standard deviation 2
sample2 = np.random.normal(loc=12, scale=2, size=100)  # Sample 2 with mean 12 and standard deviation 2

# Set up logging
logging.basicConfig(filename='z_test.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Perform two-sample z-test using OOP approach
z_test = TwoSampleZTest(sample1, sample2, alpha=0.05)
result = z_test.run_z_test()

# Print results
print("Sample 1 mean:", np.mean(sample1))
print("Sample 2 mean:", np.mean(sample2))
print("z-statistic:", z_test.z_stat)
print("p-value:", z_test.p_value)
print("Critical value:", z_test.z_critical)
print("Result:", result)
