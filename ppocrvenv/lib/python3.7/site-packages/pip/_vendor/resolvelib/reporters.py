class BaseReporter(object):
    """Delegate class to provider progress reporting for the resolver.
    """

    def starting(self):
        """Called before the resolution actually starts.
        """

    def starting_round(self, index):
        """Called before each round of resolution starts.

        The index is zero-based.
        """

    def ending_round(self, index, state):
        """Called before each round of resolution ends.

        This is NOT called if the resolution ends at this round. Use `ending`
        if you want to report finalization. The index is zero-based.
        """

    def ending(self, state):
        """Called before the resolution ends successfully.
        """

    def adding_requirement(self, requirement):
        """Called when the resolver adds a new requirement into the resolve criteria.
        """

    def backtracking(self, candidate):
        """Called when the resolver rejects a candidate during backtracking.
        """

    def pinning(self, candidate):
        """Called when adding a candidate to the potential solution.
        """
