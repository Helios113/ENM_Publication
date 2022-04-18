class ProgressBar:
    def __init__(self, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
        """
        :param total:
        :param prefix:
        :param suffix:
        :param decimals:
        :param length:
        :param fill:
        :param print_end:
        """
        if total < 1:
            raise ValueError("Total number needs to larger than 0")
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end

    def draw(self, iteration):
        """
        :param iteration:
        :return:
        """
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end=self.print_end)
        # Print New Line on Complete
        if iteration == self.total:
            print()
